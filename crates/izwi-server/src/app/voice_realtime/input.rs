//! Streamed input: VAD scoring, endpointing and the utterances it cuts.

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct StreamingInputConfig {
    pub(crate) vad_threshold: f32,
    pub(crate) min_speech_ms: u32,
    pub(crate) silence_duration_ms: u32,
    pub(crate) max_utterance_ms: u32,
    pub(crate) pre_roll_ms: u32,
    pub(crate) barge_in_min_speech_ms: u32,
    pub(crate) snr_margin_db: f32,
    pub(crate) input_sample_rate_hint: Option<u32>,
    /// When set, the silence that ends an utterance follows the speaker's
    /// own pauses, between this floor and `silence_duration_ms`.
    pub(crate) adaptive_silence_floor_ms: Option<u32>,
    /// Hand the utterance out for transcription at a short pause, before
    /// the silence that ends it has passed.
    pub(crate) early_transcription: bool,
}

/// An utterance as it stood at a pause that may turn out to end it.
#[derive(Debug)]
pub(crate) struct EarlyUtterance {
    pub(crate) utterance_seq: u64,
    pub(crate) wav_bytes: Vec<u8>,
}

/// The shortest pause worth learning from: shorter gaps are stops inside
/// words.
const MIN_LEARNED_PAUSE_MS: f32 = 120.0;
/// How many of the speaker's latest pauses the window follows.
const LEARNED_PAUSES: usize = 12;
/// A turn that starts this soon after the last one ended was the same
/// thought: the window that split them was too short.
pub(crate) const RESUMED_SPEECH_WITHIN: Duration = Duration::from_millis(1_500);

/// Learns how long this speaker pauses in the middle of an utterance and
/// sizes the silence that ends one from it. A speaker whose pauses are
/// short is answered sooner; one cut off mid-thought (they speak again
/// right after an utterance ended) gets a longer window back.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveSilence {
    floor_ms: u32,
    ceiling_ms: u32,
    pauses_ms: VecDeque<f32>,
}

impl AdaptiveSilence {
    pub(crate) fn new(floor_ms: u32, ceiling_ms: u32) -> Self {
        Self {
            floor_ms: floor_ms.min(ceiling_ms),
            ceiling_ms,
            pauses_ms: VecDeque::new(),
        }
    }

    /// The speaker paused for `pause_ms` and went on.
    pub(crate) fn observe(&mut self, pause_ms: f32) {
        if pause_ms < MIN_LEARNED_PAUSE_MS {
            return;
        }
        if self.pauses_ms.len() == LEARNED_PAUSES {
            self.pauses_ms.pop_front();
        }
        self.pauses_ms.push_back(pause_ms);
    }

    /// The silence that ends an utterance now: the configured one until
    /// four pauses are known, then half as long again as the speaker's
    /// longer pauses (the second longest of those remembered).
    pub(crate) fn window_ms(&self) -> u32 {
        if self.pauses_ms.len() < 4 {
            return self.ceiling_ms;
        }
        let mut pauses: Vec<f32> = self.pauses_ms.iter().copied().collect();
        pauses.sort_by(f32::total_cmp);
        let long_pause = pauses[pauses.len() - 2];
        ((long_pause * 1.5).round() as u32).clamp(self.floor_ms, self.ceiling_ms)
    }
}

impl StreamingInputConfig {
    pub(crate) fn evidence_config(&self) -> SpeechEvidenceConfig {
        SpeechEvidenceConfig {
            threshold: self.vad_threshold,
            snr_margin_db: self.snr_margin_db,
            ..SpeechEvidenceConfig::default()
        }
    }

    /// The endpoint rules, ending an utterance after `silence_ms` of
    /// silence when given and the configured silence otherwise.
    pub(crate) fn endpoint_config(&self, silence_ms: Option<u32>) -> EndpointConfig {
        EndpointConfig {
            start_threshold: self.vad_threshold,
            end_threshold: (self.vad_threshold * 0.7).min(self.vad_threshold),
            min_speech_ms: self.min_speech_ms,
            silence_ms: silence_ms.unwrap_or(self.silence_duration_ms),
            max_utterance_ms: self.max_utterance_ms,
            snr_margin_db: self.snr_margin_db,
            continue_snr_margin_db: self.snr_margin_db * 0.5,
        }
    }
}

#[derive(Debug)]
pub(crate) struct StreamingActiveUtterance {
    pub(crate) utterance_id: String,
    pub(crate) utterance_seq: u64,
    pub(crate) samples_i16: Vec<i16>,
    pub(crate) voiced_ms: f32,
    pub(crate) total_ms: f32,
    pub(crate) silence_ms: f32,
    pub(crate) barge_in_confirmed: bool,
    /// The utterance was handed out at the pause it is in now.
    pub(crate) early_sent: bool,
    /// The offline no-speech gate's rule over this utterance's frames. An
    /// utterance without it never becomes a turn, for modular and unified
    /// (speech-to-speech) turns alike.
    pub(crate) evidence: SpeechEvidence,
}

pub(crate) struct StreamingInputState {
    pub(crate) config: StreamingInputConfig,
    pub(crate) vad_scorer: VadScorer,
    pub(crate) endpoint: EndpointDetector,
    pub(crate) next_utterance_seq: u64,
    pub(crate) frame_seq_last: Option<u32>,
    pub(crate) current_sample_rate: Option<u32>,
    pub(crate) pre_roll: VecDeque<i16>,
    pub(crate) active: Option<StreamingActiveUtterance>,
    pub(crate) last_frame_at: Option<Instant>,
    pub(crate) adaptive_silence: Option<AdaptiveSilence>,
    /// When silence last ended an utterance, and after how much of it.
    pub(crate) last_silence_end: Option<(Instant, u32)>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum UtteranceEndReason {
    Silence,
    MaxDuration,
    ClientPause,
    StreamStopped,
}

impl UtteranceEndReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::MaxDuration => "max_duration",
            Self::ClientPause => "client_pause",
            Self::StreamStopped => "stream_stopped",
        }
    }
}

impl From<EndpointEndReason> for UtteranceEndReason {
    fn from(reason: EndpointEndReason) -> Self {
        match reason {
            EndpointEndReason::Silence => Self::Silence,
            EndpointEndReason::MaxDuration => Self::MaxDuration,
            EndpointEndReason::StreamStopped => Self::StreamStopped,
        }
    }
}

#[derive(Debug)]
pub(crate) struct SpeechStartEvent {
    pub(crate) utterance_id: String,
    pub(crate) utterance_seq: u64,
    /// Samples of pre-roll the utterance opened with, ahead of the frame
    /// that started it.
    pub(crate) pre_roll_samples: usize,
}

/// An utterance the endpoint ended with enough speech to answer.
#[derive(Debug)]
pub(crate) struct EndedUtterance {
    pub(crate) utterance_id: String,
    pub(crate) utterance_seq: u64,
    /// When the endpoint decision ended the utterance; latencies run from
    /// here.
    pub(crate) endpointed_at: Instant,
    /// Trailing silence waited for before the decision, when silence (or a
    /// client pause) ended the utterance.
    pub(crate) endpointing_delay: Option<Duration>,
    /// The utterance's audio, pre-roll included, and as a WAV file.
    pub(crate) samples: Vec<i16>,
    pub(crate) sample_rate: u32,
    pub(crate) wav_bytes: Vec<u8>,
}

/// How an active utterance ended.
#[derive(Debug)]
pub(crate) enum UtteranceOutcome {
    /// Enough voiced speech: it becomes a turn.
    Finalized(EndedUtterance, UtteranceEndReason),
    /// Too little voiced speech, or no speech evidence: the client resumes
    /// whatever it ducked on `user_speech_start`.
    Rejected(SpeechStartEvent),
}

#[derive(Debug)]
pub(crate) struct StreamingFrameResult {
    pub(crate) sequence_gap: Option<(u32, u32)>,
    pub(crate) speech_start: Option<SpeechStartEvent>,
    /// The active utterance has just confirmed a barge-in.
    pub(crate) barge_in: bool,
    pub(crate) utterance_end: Option<UtteranceOutcome>,
    /// The utterance at a pause: transcribing it now saves the time if the
    /// pause turns out to end it.
    pub(crate) early: Option<EarlyUtterance>,
    /// The speaker went on after such a pause: what was handed out is
    /// stale.
    pub(crate) speech_resumed: bool,
}

impl StreamingInputState {
    pub(crate) fn new(config: StreamingInputConfig) -> Self {
        let endpoint = EndpointDetector::new(config.endpoint_config(None));
        let adaptive_silence = config
            .adaptive_silence_floor_ms
            .map(|floor_ms| AdaptiveSilence::new(floor_ms, config.silence_duration_ms));
        Self {
            adaptive_silence,
            last_silence_end: None,
            config,
            vad_scorer: VadScorer::new(),
            endpoint,
            next_utterance_seq: 0,
            frame_seq_last: None,
            current_sample_rate: None,
            pre_roll: VecDeque::new(),
            active: None,
            last_frame_at: None,
        }
    }

    pub(crate) fn handle_pcm16_frame(
        &mut self,
        frame_seq: u32,
        sample_rate: u32,
        payload: &[u8],
    ) -> Result<StreamingFrameResult, String> {
        if !(8_000..=192_000).contains(&sample_rate) {
            return Err(format!("Invalid input sample_rate {sample_rate}"));
        }
        let sequence_gap = if let Some(last) = self.frame_seq_last {
            if frame_seq <= last {
                return Err(format!(
                    "Stale or duplicate input frame sequence {frame_seq}; last accepted was {last}"
                ));
            }
            (frame_seq > last.saturating_add(1)).then_some((last.saturating_add(1), frame_seq))
        } else {
            None
        };
        self.frame_seq_last = Some(frame_seq);
        self.last_frame_at = Some(Instant::now());

        if payload.is_empty() {
            return Ok(StreamingFrameResult {
                sequence_gap,
                speech_start: None,
                barge_in: false,
                utterance_end: None,
                early: None,
                speech_resumed: false,
            });
        }
        if !payload.len().is_multiple_of(2) {
            return Err("PCM16 payload length must be even".to_string());
        }

        if let Some(current_sr) = self.current_sample_rate {
            if current_sr != sample_rate {
                return Err(format!(
                    "Input stream sample rate changed mid-stream ({current_sr} -> {sample_rate})"
                ));
            }
        } else {
            self.current_sample_rate = Some(sample_rate);
        }

        let samples = pcm16_bytes_to_i16(payload);
        if samples.is_empty() {
            return Ok(StreamingFrameResult {
                sequence_gap,
                speech_start: None,
                barge_in: false,
                utterance_end: None,
                early: None,
                speech_resumed: false,
            });
        }

        let frame_ms = (samples.len() as f32 * 1000.0) / (sample_rate as f32);
        let vad_frames = self
            .vad_scorer
            .push_i16(&samples, sample_rate)
            .map_err(|err| format!("VAD scoring error: {err}"))?;
        let mut chunk_has_speech = false;
        let mut chunk_voiced_ms = 0.0f32;
        let mut speech_started = false;
        let mut noise_rejected = false;
        let mut end_reason: Option<UtteranceEndReason> = None;

        for vad_frame in &vad_frames {
            // Voiced time, which confirms barge-in, counts only frames that
            // clear the SNR margin as well as the score threshold.
            let decision = self.endpoint.process_frame(vad_frame);
            if decision.is_speech {
                chunk_has_speech = true;
                chunk_voiced_ms += vad_frame.duration_ms();
            }
            for event in decision.events {
                match event {
                    EndpointEvent::SpeechStart => speech_started = true,
                    EndpointEvent::SpeechEnd(reason) => end_reason = Some(reason.into()),
                    EndpointEvent::NoiseRejected => noise_rejected = true,
                }
            }
        }

        let mut result = StreamingFrameResult {
            sequence_gap,
            speech_start: None,
            barge_in: false,
            utterance_end: None,
            early: None,
            speech_resumed: false,
        };
        // A third of the silence that ends an utterance, when that leaves
        // time worth saving.
        let window_ms = self.silence_window_ms();
        let early_after_ms = (window_ms / 3).clamp(200, 400);
        let early_after_ms = (self.config.early_transcription && window_ms >= early_after_ms + 150)
            .then_some(early_after_ms as f32);
        let min_speech_ms = self.config.min_speech_ms as f32;
        let sample_rate_now = sample_rate;

        if speech_started && self.active.is_none() {
            // Speech right after silence ended an utterance: that silence
            // was a pause, longer than the window allowed.
            if let (Some(adaptive), Some((ended_at, window_ms))) =
                (self.adaptive_silence.as_mut(), self.last_silence_end.take())
            {
                let gap = ended_at.elapsed();
                if gap <= RESUMED_SPEECH_WITHIN {
                    adaptive.observe(window_ms as f32 + gap.as_secs_f32() * 1000.0);
                    self.endpoint.set_silence_ms(adaptive.window_ms());
                }
            }
            let utterance_seq = self.next_utterance_seq.saturating_add(1);
            self.next_utterance_seq = utterance_seq;
            let utterance_id = format!("utt-{utterance_seq}");

            let mut capture = StreamingActiveUtterance {
                utterance_id: utterance_id.clone(),
                utterance_seq,
                samples_i16: Vec::new(),
                voiced_ms: 0.0,
                total_ms: 0.0,
                silence_ms: 0.0,
                barge_in_confirmed: false,
                early_sent: false,
                evidence: SpeechEvidence::new(self.config.evidence_config()),
            };
            let pre_roll_samples = self.pre_roll.len();
            if !self.pre_roll.is_empty() {
                capture.samples_i16.extend(self.pre_roll.iter().copied());
            }
            if capture.samples_i16.len().saturating_add(samples.len()) > MAX_UTTERANCE_PCM16_SAMPLES
            {
                return Err("Streamed utterance exceeded the PCM16 buffer limit".to_string());
            }
            capture.samples_i16.extend_from_slice(&samples);
            self.active = Some(capture);

            result.speech_start = Some(SpeechStartEvent {
                utterance_id,
                utterance_seq,
                pre_roll_samples,
            });
        } else if let Some(active) = self.active.as_mut() {
            if active.samples_i16.len().saturating_add(samples.len()) > MAX_UTTERANCE_PCM16_SAMPLES
            {
                return Err("Streamed utterance exceeded the PCM16 buffer limit".to_string());
            }
            active.samples_i16.extend_from_slice(&samples);
        } else {
            self.push_pre_roll(&samples, sample_rate);
            return Ok(result);
        }

        if let Some(active) = self.active.as_mut() {
            for vad_frame in &vad_frames {
                active.evidence.push(vad_frame);
            }
            active.voiced_ms += chunk_voiced_ms;
            active.total_ms += frame_ms;
            if chunk_has_speech {
                if let Some(adaptive) = self.adaptive_silence.as_mut() {
                    adaptive.observe(active.silence_ms);
                    self.endpoint.set_silence_ms(adaptive.window_ms());
                }
                active.silence_ms = 0.0;
                result.speech_resumed = std::mem::take(&mut active.early_sent);
            } else {
                active.silence_ms += frame_ms;
                if early_after_ms.is_some_and(|after| active.silence_ms >= after)
                    && !active.early_sent
                    && end_reason.is_none()
                    && !noise_rejected
                    && active.voiced_ms >= min_speech_ms
                    && active.evidence.is_sufficient()
                {
                    active.early_sent = true;
                    result.early = Some(EarlyUtterance {
                        utterance_seq: active.utterance_seq,
                        wav_bytes: wav_bytes_from_pcm16_mono(&active.samples_i16, sample_rate_now)?,
                    });
                }
            }
            // An utterance that ends or is rejected in this frame never
            // barges in; finalizing one preempts the turn on its own.
            if !active.barge_in_confirmed
                && end_reason.is_none()
                && !noise_rejected
                && active.voiced_ms >= self.config.barge_in_min_speech_ms as f32
            {
                active.barge_in_confirmed = true;
                result.barge_in = true;
            }
        }

        if let Some(reason) = end_reason {
            result.utterance_end = self.finalize_active_utterance(reason)?;
        } else if noise_rejected {
            result.utterance_end = self.reject_active_utterance();
        }

        if !chunk_has_speech {
            self.push_pre_roll(&samples, sample_rate);
        }

        Ok(result)
    }

    /// The silence that ends an utterance now.
    pub(crate) fn silence_window_ms(&self) -> u32 {
        self.adaptive_silence
            .as_ref()
            .map_or(self.config.silence_duration_ms, AdaptiveSilence::window_ms)
    }

    fn endpoint_rules(&self) -> EndpointConfig {
        self.config.endpoint_config(Some(self.silence_window_ms()))
    }

    pub(crate) fn reject_active_utterance(&mut self) -> Option<UtteranceOutcome> {
        let active = self.active.take()?;
        Some(UtteranceOutcome::Rejected(SpeechStartEvent {
            utterance_id: active.utterance_id,
            utterance_seq: active.utterance_seq,
            pre_roll_samples: 0,
        }))
    }

    pub(crate) fn finish_stream(&mut self) -> Result<Option<UtteranceOutcome>, String> {
        match self.endpoint.finish() {
            Some(EndpointEvent::NoiseRejected) => Ok(self.reject_active_utterance()),
            _ => self.finalize_active_utterance(UtteranceEndReason::StreamStopped),
        }
    }

    /// Ends the active utterance when the client stops sending audio without
    /// `input_stream_stop`. Silence and the utterance cap are measured in
    /// audio time by the endpoint detector, so delayed but contiguous frames
    /// (a network stall) cannot end an utterance early.
    pub(crate) fn on_tick(&mut self) -> Result<Option<UtteranceOutcome>, String> {
        if self.active.is_none() {
            return Ok(None);
        }
        let pause_limit = Duration::from_millis(
            u64::from(self.config.silence_duration_ms) + CLIENT_PAUSE_STALL_ALLOWANCE_MS,
        );
        if self
            .last_frame_at
            .is_some_and(|last_frame| last_frame.elapsed() >= pause_limit)
        {
            return self.finalize_active_utterance(UtteranceEndReason::ClientPause);
        }
        Ok(None)
    }

    /// End the active utterance: a turn when it holds enough voiced speech,
    /// otherwise a rejection the client must hear about, since it already saw
    /// `user_speech_start`.
    pub(crate) fn finalize_active_utterance(
        &mut self,
        reason: UtteranceEndReason,
    ) -> Result<Option<UtteranceOutcome>, String> {
        let Some(active) = self.active.take() else {
            return Ok(None);
        };
        if matches!(reason, UtteranceEndReason::Silence) {
            self.last_silence_end = Some((Instant::now(), self.silence_window_ms()));
        }
        self.endpoint = EndpointDetector::new(self.endpoint_rules());
        if active.voiced_ms < self.config.min_speech_ms as f32 || !active.evidence.is_sufficient() {
            return Ok(Some(UtteranceOutcome::Rejected(SpeechStartEvent {
                utterance_id: active.utterance_id,
                utterance_seq: active.utterance_seq,
                pre_roll_samples: 0,
            })));
        }
        let sample_rate = self
            .current_sample_rate
            .or(self.config.input_sample_rate_hint)
            .ok_or_else(|| "Missing input sample rate for streamed audio".to_string())?;

        let wav_bytes = wav_bytes_from_pcm16_mono(&active.samples_i16, sample_rate)?;
        if wav_bytes.len() > MAX_UTTERANCE_BYTES {
            return Err(format!(
                "Streamed utterance exceeded max encoded size ({} > {})",
                wav_bytes.len(),
                MAX_UTTERANCE_BYTES
            ));
        }

        let trailing_silence =
            Duration::from_micros((active.silence_ms.max(0.0) * 1000.0).round() as u64);
        let endpointing_delay = match reason {
            UtteranceEndReason::Silence => Some(trailing_silence),
            UtteranceEndReason::ClientPause => Some(
                trailing_silence
                    + self
                        .last_frame_at
                        .map_or(Duration::ZERO, |last_frame| last_frame.elapsed()),
            ),
            UtteranceEndReason::MaxDuration | UtteranceEndReason::StreamStopped => None,
        };
        let ended = EndedUtterance {
            utterance_id: active.utterance_id,
            utterance_seq: active.utterance_seq,
            endpointed_at: Instant::now(),
            endpointing_delay,
            samples: active.samples_i16,
            sample_rate,
            wav_bytes,
        };

        Ok(Some(UtteranceOutcome::Finalized(ended, reason)))
    }

    /// End the active utterance now, as if the stream had stopped, and keep
    /// listening: an OpenAI `input_audio_buffer.commit` under server VAD.
    pub(crate) fn commit_active(&mut self) -> Result<Option<UtteranceOutcome>, String> {
        let outcome = self.finish_stream();
        self.endpoint = EndpointDetector::new(self.endpoint_rules());
        outcome
    }

    /// Drop the active utterance and the pre-roll without a decision: an
    /// OpenAI `input_audio_buffer.clear`.
    pub(crate) fn clear(&mut self) {
        self.active = None;
        self.pre_roll.clear();
        self.endpoint = EndpointDetector::new(self.endpoint_rules());
    }

    /// Whether an utterance is being captured.
    pub(crate) fn has_active_utterance(&self) -> bool {
        self.active.is_some()
    }

    pub(crate) fn push_pre_roll(&mut self, samples: &[i16], sample_rate: u32) {
        let max_samples = ((sample_rate as u64 * self.config.pre_roll_ms as u64) / 1000) as usize;
        if max_samples == 0 {
            self.pre_roll.clear();
            return;
        }

        self.pre_roll.extend(samples.iter().copied());
        if self.pre_roll.len() > max_samples {
            let drain = self.pre_roll.len() - max_samples;
            self.pre_roll.drain(0..drain);
        }
    }
}

pub(crate) fn normalize_stream_vad_threshold(value: Option<f32>) -> f32 {
    value
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(sanitize_score_threshold)
        .unwrap_or(DEFAULT_STREAM_VAD_THRESHOLD)
}
