//! Windowed synthesis of a streamed utterance.
//!
//! Upstream has hooks for streaming (`finalize=False` drops the encoder's
//! three-token lookahead; HiFT takes a cached source) but no streamer, and
//! its S3Gen attends over every frame, so a window cannot reproduce the
//! whole utterance exactly. Each window instead re-renders a little of what
//! came before for context:
//!
//! * once T3 has sampled enough new tokens (plus the lookahead), S3Gen runs
//!   on the reference prompt, up to `left_context_tokens` already-rendered
//!   tokens and the new ones, with `finalize=false` until T3 has stopped;
//! * only the new tokens' mel frames are kept, vocoded after the previous
//!   window's last frames with a continuous harmonic source, and the
//!   overlap crossfaded ([`HiftVocoder::vocode_stream`]);
//! * the flow's noise is drawn per absolute frame, so a frame gets the same
//!   noise in every window that renders it;
//! * PerTh marks the samples over overlapping segments ([`PerthStream`]).
//!
//! Each window costs a full flow over the prompt, so windows grow from a
//! small first one (quick first audio) to `max_tokens`.

use candle_core::Tensor;

use super::hift::{HiftStream, HiftVocoder};
use super::noise::frame_flow_noise;
use super::perth::{PerthStream, PerthWatermarker};
use super::s3gen::{S3GenFlow, S3GenFlowState, S3GenPreparedVoice};
use crate::error::{Error, Result};

/// When a streamed utterance's windows run and what each one sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatterboxStreamPlan {
    /// New tokens in the first window; each later window doubles the
    /// previous one up to `max_tokens`.
    pub first_tokens: usize,
    pub max_tokens: usize,
    /// Rendered tokens each window encodes again ahead of its new ones.
    pub left_context_tokens: usize,
    /// The reference's latest tokens each window keeps; 0 keeps them all.
    pub prompt_tokens: usize,
}

impl ChatterboxStreamPlan {
    /// New tokens in window `index` (from 0).
    pub fn window_tokens(&self, index: usize) -> usize {
        let doubled = self
            .first_tokens
            .checked_shl(index.min(16) as u32)
            .unwrap_or(usize::MAX);
        doubled.min(self.max_tokens).max(1)
    }
}

/// The default plan, chosen by `stream_tests` on Metal.
pub const CHATTERBOX_STREAM_PLAN: ChatterboxStreamPlan = ChatterboxStreamPlan {
    first_tokens: 25,
    max_tokens: 100,
    left_context_tokens: 25,
    prompt_tokens: 0,
};

/// A streamed utterance's acoustic progress.
pub struct ChatterboxStream {
    plan: ChatterboxStreamPlan,
    seed: u64,
    /// Tokens whose audio has been handed over.
    emitted_tokens: usize,
    windows: usize,
    window: Option<StreamWindow>,
    vocoder: HiftStream,
    watermark: Option<PerthStream>,
    /// Samples handed over so far.
    samples: usize,
    done: bool,
    /// Samples cut from the end of the utterance (the multilingual
    /// edition's last token).
    dropped_tail: usize,
}

struct StreamWindow {
    /// The first token the window encodes (left context included).
    first: usize,
    /// The tokens whose frames it hands over.
    start: usize,
    end: usize,
    last: bool,
    voice: S3GenPreparedVoice,
    flow: Option<S3GenFlowState>,
}

/// What one streaming unit did.
pub enum ChatterboxStreamUnit {
    /// The encoder or an Euler step ran; the window is not done.
    Rendering,
    /// The window finished: these samples (possibly none) are settled.
    Rendered(Vec<f32>),
}

impl ChatterboxStream {
    pub fn new(
        plan: ChatterboxStreamPlan,
        seed: u64,
        vocoder: &HiftVocoder,
        watermarker: Option<&PerthWatermarker>,
        sample_rate: u32,
        dropped_tail: usize,
    ) -> Self {
        Self {
            plan,
            seed,
            emitted_tokens: 0,
            windows: 0,
            window: None,
            vocoder: vocoder.stream(seed),
            watermark: watermarker.map(|watermarker| watermarker.stream(sample_rate)),
            samples: 0,
            done: false,
            dropped_tail,
        }
    }

    /// Tokens whose audio has been handed over.
    pub fn emitted_tokens(&self) -> usize {
        self.emitted_tokens
    }

    /// Samples handed over so far.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Windows rendered so far.
    pub fn windows(&self) -> usize {
        self.windows
    }

    /// Whether the last window has been handed over.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Whether a window is in progress or should start, with `available`
    /// renderable tokens and T3 `stopped` or not.
    pub fn window_due(&self, available: usize, stopped: bool, lookahead: usize) -> bool {
        if self.done {
            return false;
        }
        if self.window.is_some() {
            return true;
        }
        if stopped {
            return available > self.emitted_tokens;
        }
        available >= self.emitted_tokens + self.plan.window_tokens(self.windows) + lookahead
    }

    /// Run the next unit of the current window (starting one if needed)
    /// over `tokens`, the renderable tokens so far: the S3Gen encoder, one
    /// Euler step, or the vocoder and watermark.
    pub fn step(
        &mut self,
        flow: &S3GenFlow,
        vocoder: &HiftVocoder,
        watermarker: Option<&PerthWatermarker>,
        voice: &S3GenPreparedVoice,
        tokens: &[u32],
        stopped: bool,
    ) -> Result<ChatterboxStreamUnit> {
        let lookahead = flow.config().pre_lookahead_len;
        if self.window.is_none() {
            if stopped && tokens.is_empty() {
                return Err(Error::InferenceError(
                    "Chatterbox generated no renderable speech tokens".into(),
                ));
            }
            if !self.window_due(tokens.len(), stopped, lookahead) {
                return Err(Error::InferenceError(
                    "Chatterbox stream window run before it was due".into(),
                ));
            }
            let start = self.emitted_tokens;
            let end = if stopped {
                tokens.len()
            } else {
                start + self.plan.window_tokens(self.windows)
            };
            self.window = Some(StreamWindow {
                first: start.saturating_sub(self.plan.left_context_tokens),
                start,
                end,
                last: stopped,
                voice: voice.tail(self.plan.prompt_tokens)?,
                flow: None,
            });
        }
        let window = self.window.as_mut().expect("window started above");
        let cfg = flow.config();
        let Some(state) = window.flow.as_mut() else {
            let encoded_end = if window.last {
                window.end
            } else {
                window.end + lookahead
            };
            let encoded = tokens.get(window.first..encoded_end).ok_or_else(|| {
                Error::InferenceError("Chatterbox stream window outran its tokens".into())
            })?;
            let prompt = window.voice.prompt_frames();
            let frames = prompt + 2 * (window.end - window.first);
            // Absolute frames: the whole reference, then two per token.
            let full_prompt = voice.prompt_frames();
            let mut noise = vec![0f32; cfg.mel_bins * frames];
            let reference = frame_flow_noise(self.seed, full_prompt - prompt, prompt, cfg.mel_bins);
            let generated = frame_flow_noise(
                self.seed,
                full_prompt + 2 * window.first,
                frames - prompt,
                cfg.mel_bins,
            );
            for bin in 0..cfg.mel_bins {
                noise[bin * frames..bin * frames + prompt]
                    .copy_from_slice(&reference[bin * prompt..(bin + 1) * prompt]);
                noise[bin * frames + prompt..(bin + 1) * frames].copy_from_slice(
                    &generated[bin * (frames - prompt)..(bin + 1) * (frames - prompt)],
                );
            }
            window.flow = Some(flow.begin(&window.voice, encoded, &noise, window.last)?);
            return Ok(ChatterboxStreamUnit::Rendering);
        };
        if state.steps_done() < cfg.cfm_steps {
            flow.step(state, 1)?;
            return Ok(ChatterboxStreamUnit::Rendering);
        }
        let mel = flow.finish(state)?;
        let skip = 2 * (window.start - window.first);
        let keep = 2 * (window.end - window.start);
        let mel: Tensor = mel.narrow(1, skip, keep)?.contiguous()?;
        let last = window.last;
        let mut samples = vocoder.vocode_stream(&mut self.vocoder, &mel, last)?;
        if last && self.dropped_tail > 0 && window.end > 1 {
            samples.truncate(samples.len().saturating_sub(self.dropped_tail));
        }
        if let (Some(watermarker), Some(stream)) = (watermarker, self.watermark.as_mut()) {
            samples = watermarker.push(stream, &samples, last)?;
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "Chatterbox vocoder produced non-finite samples".into(),
            ));
        }
        self.emitted_tokens = window.end;
        self.windows += 1;
        self.samples += samples.len();
        self.done = last;
        self.window = None;
        Ok(ChatterboxStreamUnit::Rendered(samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_double_up_to_the_cap() {
        let plan = ChatterboxStreamPlan {
            first_tokens: 20,
            max_tokens: 75,
            left_context_tokens: 10,
            prompt_tokens: 0,
        };
        let sizes = (0..5).map(|i| plan.window_tokens(i)).collect::<Vec<_>>();
        assert_eq!(sizes, vec![20, 40, 75, 75, 75]);
        assert_eq!(plan.window_tokens(200), 75);
    }
}
