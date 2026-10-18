//! Noise gate for streaming models that hallucinate on non-speech.
//!
//! A realtime ASR model that continues a language-model prompt (Voxtral)
//! transcribes noise as a confident phrase, and its streamed text cannot be
//! taken back once emitted. The gate therefore works on the model's input:
//! it holds audio for a short lookahead, decides with the SNR-aware endpoint
//! detector which spans are confirmed speech, and replaces everything else
//! with digital silence. The sample count and timing never change, so model
//! state, segment rollover and audio-time bookkeeping are unaffected.

use std::collections::VecDeque;

use crate::{EndpointConfig, EndpointDetector, EndpointEvent, VadScorer, DEFAULT_PRE_ROLL_MS};

/// Audio held back before it is released, so an utterance can be confirmed
/// (reach `min_speech_ms` of voiced speech) before its first samples leave.
pub const DEFAULT_NOISE_GATE_LOOKAHEAD_MS: u32 = 500;

#[derive(Debug, Clone)]
pub struct NoiseGateConfig {
    /// Speech start, continuation, minimum speech and end rules.
    pub endpoint: EndpointConfig,
    /// Delay between receiving audio and releasing it.
    pub lookahead_ms: u32,
    /// Audio kept before a confirmed utterance's first speech frame.
    pub pre_roll_ms: u32,
}

impl Default for NoiseGateConfig {
    fn default() -> Self {
        Self {
            endpoint: EndpointConfig::default(),
            lookahead_ms: DEFAULT_NOISE_GATE_LOOKAHEAD_MS,
            pre_roll_ms: DEFAULT_PRE_ROLL_MS,
        }
    }
}

/// See the module docs. Feed it with [`NoiseGate::push`] and drain it with
/// [`NoiseGate::flush`] when the stream ends.
pub struct NoiseGate {
    config: NoiseGateConfig,
    scorer: VadScorer,
    detector: EndpointDetector,
    sample_rate: Option<u32>,
    /// Input samples not yet released, starting at absolute index `released`.
    pending: VecDeque<f32>,
    released: u64,
    received: u64,
    /// Start (VAD ms) of the utterance the detector is in, if any.
    utterance_start_ms: Option<f32>,
    utterance_confirmed: bool,
    /// Confirmed speech spans in input ms; an open span has no end yet.
    spans: VecDeque<(f64, Option<f64>)>,
}

impl NoiseGate {
    pub fn new(config: NoiseGateConfig) -> Self {
        let detector = EndpointDetector::new(config.endpoint.clone());
        Self {
            config,
            scorer: VadScorer::new(),
            detector,
            sample_rate: None,
            pending: VecDeque::new(),
            released: 0,
            received: 0,
            utterance_start_ms: None,
            utterance_confirmed: false,
            spans: VecDeque::new(),
        }
    }

    /// Take `samples` in and return the audio now old enough to release,
    /// with everything outside confirmed speech zeroed. A sample-rate change
    /// releases what is held under the old rate first and starts afresh.
    pub fn push(&mut self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut out = Vec::new();
        if self.sample_rate.is_some_and(|rate| rate != sample_rate) {
            out = self.flush();
            *self = Self::new(self.config.clone());
        }
        self.sample_rate = Some(sample_rate);
        if samples.is_empty() {
            return out;
        }

        match self.scorer.push_f32(samples, sample_rate) {
            Ok(frames) => {
                for frame in frames {
                    let decision = self.detector.process_frame(&frame);
                    for event in &decision.events {
                        match event {
                            EndpointEvent::SpeechStart => {
                                self.utterance_start_ms = Some(frame.start_ms);
                                self.utterance_confirmed = false;
                            }
                            EndpointEvent::SpeechEnd(_) => {
                                self.close_utterance(frame.end_ms as f64);
                            }
                            EndpointEvent::NoiseRejected => self.drop_utterance(),
                        }
                    }
                    if self.detector.is_active()
                        && decision.speech_ms >= self.detector.config().min_speech_ms as f32
                    {
                        self.confirm_utterance();
                    }
                }
            }
            // Unscorable audio (a zero rate) cannot be judged: pass it on.
            Err(_) => {
                out.extend_from_slice(samples);
                return out;
            }
        }

        self.pending.extend(samples.iter().copied());
        self.received += samples.len() as u64;
        let lookahead = self.ms_to_samples(self.config.lookahead_ms as f64);
        let horizon = self.received.saturating_sub(lookahead);
        out.extend(self.release_until(horizon));
        out
    }

    /// Release everything held, judging an utterance still in progress by
    /// the voiced speech it reached.
    pub fn flush(&mut self) -> Vec<f32> {
        if let Some(EndpointEvent::SpeechEnd(_)) = self.detector.finish() {
            self.confirm_utterance();
        }
        let end_ms = self.samples_to_ms(self.received);
        self.close_utterance(end_ms);
        self.release_until(self.received)
    }

    /// Delay between receiving a sample and releasing it.
    pub fn lookahead_ms(&self) -> u32 {
        self.config.lookahead_ms
    }

    /// Rate of the audio held, once any has been pushed.
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    fn confirm_utterance(&mut self) {
        if self.utterance_confirmed {
            return;
        }
        let Some(start_ms) = self.utterance_start_ms else {
            return;
        };
        self.utterance_confirmed = true;
        let start = (start_ms as f64 - self.config.pre_roll_ms as f64).max(0.0);
        self.spans.push_back((start, None));
    }

    fn close_utterance(&mut self, end_ms: f64) {
        if self.utterance_confirmed {
            if let Some((_, end)) = self.spans.back_mut() {
                end.get_or_insert(end_ms);
            }
        }
        self.drop_utterance();
    }

    fn drop_utterance(&mut self) {
        self.utterance_start_ms = None;
        self.utterance_confirmed = false;
    }

    fn release_until(&mut self, horizon: u64) -> Vec<f32> {
        let count = horizon.saturating_sub(self.released) as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let Some(sample) = self.pending.pop_front() else {
                break;
            };
            let t_ms = self.samples_to_ms(self.released);
            while self
                .spans
                .front()
                .is_some_and(|(_, end)| end.is_some_and(|end| end <= t_ms))
            {
                self.spans.pop_front();
            }
            let in_speech = self
                .spans
                .front()
                .is_some_and(|(start, end)| *start <= t_ms && end.is_none_or(|end| t_ms < end));
            out.push(if in_speech { sample } else { 0.0 });
            self.released += 1;
        }
        out
    }

    fn ms_to_samples(&self, ms: f64) -> u64 {
        let rate = self.sample_rate.unwrap_or(crate::VAD_SAMPLE_RATE) as f64;
        (ms * rate / 1000.0).round() as u64
    }

    fn samples_to_ms(&self, samples: u64) -> f64 {
        let rate = self.sample_rate.unwrap_or(crate::VAD_SAMPLE_RATE) as f64;
        samples as f64 * 1000.0 / rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_signals::{fan_noise, mains_hum, pink_noise, white_noise};
    use crate::{resample_linear_f32, VAD_SAMPLE_RATE};
    use std::path::Path;

    const SR: usize = VAD_SAMPLE_RATE as usize;

    fn fox() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
        let mut reader = hound::WavReader::open(path).expect("speech fixture");
        let rate = reader.spec().sample_rate;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.expect("sample") as f32 / 32768.0)
            .collect();
        resample_linear_f32(&samples, rate, VAD_SAMPLE_RATE).expect("resample")
    }

    /// The whole stream through the gate in 20 ms packets, as clients send it.
    fn gate(samples: &[f32]) -> Vec<f32> {
        let mut gate = NoiseGate::new(NoiseGateConfig::default());
        let mut out = Vec::new();
        for packet in samples.chunks(320) {
            out.extend(gate.push(packet, VAD_SAMPLE_RATE));
        }
        out.extend(gate.flush());
        out
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32)
            .sqrt()
    }

    #[test]
    fn steady_noise_is_silenced_without_changing_length() {
        let len = SR * 10;
        for (name, noise) in [
            ("pink 0.05", pink_noise(len, 0.05, 4)),
            ("pink 0.3", pink_noise(len, 0.3, 4)),
            ("white 0.2", white_noise(len, 0.2, 4)),
            ("fan 0.3", fan_noise(len, 0.3, 4)),
            ("hum 0.3", mains_hum(len, 0.3)),
        ] {
            let out = gate(&noise);
            assert_eq!(out.len(), noise.len(), "{name}");
            let passed = out.iter().filter(|sample| **sample != 0.0).count();
            assert_eq!(passed, 0, "{name}: {passed} noise samples reached the model");
        }
    }

    #[test]
    fn speech_in_noise_passes_intact_with_its_onset() {
        let fox = fox();
        let len = SR * 10;
        for (name, noise) in [
            ("pink", pink_noise(len, 0.5, 7)),
            ("fan", fan_noise(len, 0.5, 7)),
            ("hum", mains_hum(len, 0.5)),
        ] {
            let gain = rms(&fox) / rms(&noise) * 10f32.powf(-10.0 / 20.0);
            let mut input: Vec<f32> = noise.iter().map(|sample| sample * gain).collect();
            let at = SR * 3;
            for (idx, sample) in fox.iter().enumerate() {
                input[at + idx] += sample;
            }

            let out = gate(&input);
            assert_eq!(out.len(), input.len(), "{name}");
            // Fox speech spans about 240-3330 ms of the fixture: all of it
            // passes unchanged.
            let speech = at + SR * 240 / 1000..at + SR * 3330 / 1000;
            assert_eq!(&out[speech.clone()], &input[speech], "{name} speech altered");
            // The noise before it does not.
            assert!(out[..SR * 2].iter().all(|sample| *sample == 0.0), "{name} lead-in");
            assert!(out[SR * 8..].iter().all(|sample| *sample == 0.0), "{name} tail");
        }
    }

    #[test]
    fn speech_from_the_first_sample_and_to_the_last_passes() {
        let fox = fox();
        let out = gate(&fox);
        let speech = SR * 240 / 1000..SR * 3330 / 1000;
        assert_eq!(&out[speech.clone()], &fox[speech]);
    }

    #[test]
    fn release_lags_input_by_the_lookahead() {
        let mut gate = NoiseGate::new(NoiseGateConfig::default());
        let lookahead = SR * DEFAULT_NOISE_GATE_LOOKAHEAD_MS as usize / 1000;
        let mut released = 0;
        for _ in 0..100 {
            released += gate.push(&[0.0; 320], VAD_SAMPLE_RATE).len();
        }
        assert_eq!(released, 100 * 320 - lookahead);
        assert_eq!(gate.flush().len(), lookahead);
    }

    #[test]
    fn a_rate_change_releases_held_audio_first() {
        let mut gate = NoiseGate::new(NoiseGateConfig::default());
        let first = gate.push(&vec![0.0; 16_000], 16_000);
        let second = gate.push(&vec![0.0; 24_000], 24_000);
        // Everything held at 16 kHz comes out before 24 kHz audio is held.
        assert_eq!(first.len() + second.len(), 16_000 + 24_000 - 12_000);
    }
}
