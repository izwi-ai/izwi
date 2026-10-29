//! The Parallel Expert Encoder's online path and the projection to the
//! decoder width, run lazily one 40 s window at a time.
//!
//! Over `T = Tv + 1` mel frames, window `k` has a core of 4,000 frames (500
//! encoder frames) and up to 400 frames (50 encoder frames) of context on
//! each side:
//!
//! ```text
//! start = 4000k, end = min(start + 4000, T),
//! cs = max(start - 400, 0), ce = min(end + 400, T)
//! ```
//!
//! The ASR encoder runs on the normalized mel `cs..ce` (positions from 0,
//! `clamp(Tv - cs, 0, ce - cs)` valid frames) and keeps its core; the
//! diarizer takes one streaming step on the raw mel of the same span (the
//! last window does not update its state). Each core frame fuses
//! `asr_norm(asr) + scale * (diar_norm(1[p > 0.5]) @ K)`, `K` the
//! interleaved sin/cos speaker kernel, and projects to the decoder width.
//! The concatenation is trimmed to `ceil(Tv / 8)` audio tokens.
//!
//! A [`PerceptionStream`] produces windows in order, so a prefill span
//! encodes only the windows its audio positions need and a one-hour file
//! never holds the device for its whole encoder pass.

use std::collections::VecDeque;

use candle_core::{DType, Device, Tensor};
use half::bf16;

use super::checkpoint::NemotronCheckpoint;
use super::config::{NemotronTranscribeConfig, ONLINE_CONTEXT_TOKENS, ONLINE_WINDOW_TOKENS};
use super::frontend::NemotronMel;
use super::linear::{NtLinear, Pass};
use super::precision::Fp8Mode;
use super::sortformer::{align_frames, SortformerHead, SortformerState};
use super::speech_transformer::{host_values, NtLayerNorm, SpeechTransformer};
use crate::error::{Error, Result};

/// One encoder window, in mel frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerceptionWindow {
    pub start: usize,
    pub end: usize,
    pub context_start: usize,
    pub context_end: usize,
}

impl PerceptionWindow {
    /// Mel frames of the window that are valid audio.
    pub fn valid_mel(&self, valid_frames: usize) -> usize {
        valid_frames
            .saturating_sub(self.context_start)
            .min(self.context_end - self.context_start)
    }

    /// Encoder frames dropped on the left (context).
    pub fn left_drop(&self, stacking: usize) -> usize {
        (self.start - self.context_start) / stacking
    }

    /// Encoder frames of the core: `ceil(end/8) - ceil(start/8)`, clipped to
    /// what the window encodes after its left context.
    pub fn core_frames(&self, stacking: usize) -> usize {
        let core = self.end.div_ceil(stacking) - self.start.div_ceil(stacking);
        let encoded = (self.context_end - self.context_start).div_ceil(stacking);
        core.min(encoded.saturating_sub(self.left_drop(stacking)))
    }
}

/// The online windows over `frames` (`Tv + 1`) mel frames.
pub fn plan_windows(frames: usize, stacking: usize) -> Vec<PerceptionWindow> {
    let core = ONLINE_WINDOW_TOKENS * stacking;
    let context = ONLINE_CONTEXT_TOKENS * stacking;
    let count = frames.div_ceil(core).max(1);
    (0..count)
        .map(|k| {
            let start = k * core;
            let end = (start + core).min(frames);
            PerceptionWindow {
                start,
                end,
                context_start: start.saturating_sub(context),
                context_end: (end + context).min(frames),
            }
        })
        .collect()
}

/// The fixed speaker kernel `[speakers, width]`: `K[s, 2i] = sin(s w_i)`,
/// `K[s, 2i + 1] = cos(s w_i)`, `w_i = exp(-2i ln(1e4) / width)`, built in
/// F32 and rounded to BF16 as upstream casts it.
pub fn speaker_kernel(speakers: usize, width: usize) -> Vec<f32> {
    let mut kernel = vec![0f32; speakers * width];
    // torch: an F32 arange times the F64 scalar -(ln(1e4) / width) taken as
    // F32, then exp, all F32.
    let scale = (-(10_000f64.ln() / width as f64)) as f32;
    for s in 0..speakers {
        for i in 0..width / 2 {
            let frequency = ((2 * i) as f32 * scale).exp();
            let angle = s as f32 * frequency;
            kernel[s * width + 2 * i] = bf16::from_f32(angle.sin()).to_f32();
            kernel[s * width + 2 * i + 1] = bf16::from_f32(angle.cos()).to_f32();
        }
    }
    kernel
}

/// The perception module's weights.
pub struct NemotronPerception {
    asr_encoder: SpeechTransformer,
    diar_encoder: SpeechTransformer,
    sortformer: SortformerHead,
    asr_norm: NtLayerNorm,
    diar_norm: NtLayerNorm,
    speaker_kernel: Tensor,
    proj: NtLinear,
    threshold: f32,
    kernel_scale: f64,
    speakers: usize,
    device: Device,
}

/// What one encoded window yields.
#[derive(Debug, Clone)]
pub struct EncodedWindow {
    /// The first audio token of the window's core.
    pub first_token: usize,
    /// `[core, hidden]` projected embeddings.
    pub embeddings: Tensor,
    /// The pre-`asr_norm` encoder output `[core, d_model]`, kept when the
    /// CTC stage was requested.
    pub ctc_input: Option<Tensor>,
    /// 80 ms diarizer probabilities, aligned to the core.
    pub speaker_preds: Vec<Vec<f32>>,
    /// Native 10 ms probabilities of the core.
    pub speaker_preds_10ms: Vec<Vec<f32>>,
}

impl NemotronPerception {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let asr_encoder = SpeechTransformer::load(
            checkpoint,
            config,
            config.asr_encoder,
            "perception.encoder.asr_encoder",
            true,
            false,
            device,
            mode,
        )?;
        let diar_encoder = SpeechTransformer::load(
            checkpoint,
            config,
            config.diar_encoder,
            "perception.encoder.diarization_model.encoder",
            true,
            false,
            device,
            mode,
        )?;
        let d = config.asr_encoder.d_model;
        let speakers = config.sortformer.speakers;
        Ok(Self {
            sortformer: SortformerHead::load(checkpoint, config, device)?,
            asr_norm: NtLayerNorm::load(
                checkpoint,
                "perception.encoder.asr_norm",
                d,
                1e-5,
                device,
            )?,
            diar_norm: NtLayerNorm::load(
                checkpoint,
                "perception.encoder.diar_norm",
                speakers,
                1e-5,
                device,
            )?,
            speaker_kernel: Tensor::from_vec(speaker_kernel(speakers, d), (speakers, d), device)?,
            proj: NtLinear::load_dense(
                checkpoint,
                "perception.proj",
                config.decoder.hidden,
                d,
                true,
                device,
            )?,
            threshold: config.speaker_activity_threshold,
            kernel_scale: f64::from(config.speaker_kernel_scale),
            speakers,
            asr_encoder,
            diar_encoder,
            device: device.clone(),
        })
    }

    pub fn stacking(&self) -> usize {
        self.asr_encoder.config().stacking
    }

    pub fn resident_bytes(&self) -> usize {
        self.asr_encoder.resident_bytes()
            + self.diar_encoder.resident_bytes()
            + self.proj.resident_bytes()
    }

    /// Encodes `window` of `mel`: the ASR core, one diarizer step and the
    /// fused, projected core.
    fn encode_window(
        &self,
        mel: &NemotronMel,
        window: &PerceptionWindow,
        state: &mut SortformerState,
        update_state: bool,
        keep_ctc_input: bool,
        first_token: usize,
    ) -> Result<EncodedWindow> {
        let stacking = self.stacking();
        let valid = window.valid_mel(mel.valid_frames());
        let normalized = mel.normalized_window(window.context_start, window.context_end);
        let encoded = self.asr_encoder.forward_mel(&normalized, valid)?;
        let left = window.left_drop(stacking);
        let core = window.core_frames(stacking);
        let asr = encoded.narrow(0, left, core)?;
        let raw = mel.raw_window(window.context_start, window.context_end);
        let diar = self.sortformer.step(
            &self.diar_encoder,
            state,
            &raw,
            valid,
            window.start - window.context_start,
            window.context_end - window.end,
            update_state,
        )?;
        let speaker_preds = align_frames(&diar.preds_80ms, core, self.speakers);
        let activity = speaker_preds
            .iter()
            .flat_map(|row| row.iter().map(|p| f32::from(*p > self.threshold)))
            .collect::<Vec<_>>();
        let activity = Tensor::from_vec(activity, (core, self.speakers), &self.device)?;
        let infusion = self
            .diar_norm
            .forward(&activity)?
            .matmul(&self.speaker_kernel)?;
        let fused = (self.asr_norm.forward(&asr)? + (infusion * self.kernel_scale)?)?;
        let embeddings = self.proj.forward(&fused, Pass::Span)?;
        Ok(EncodedWindow {
            first_token,
            embeddings,
            ctc_input: keep_ctc_input.then_some(asr),
            speaker_preds,
            speaker_preds_10ms: diar.preds_10ms,
        })
    }
}

/// One recording's perception, produced window by window.
pub struct PerceptionStream {
    mel: NemotronMel,
    windows: Vec<PerceptionWindow>,
    next_window: usize,
    /// Tokens produced so far (before the final trim).
    produced: usize,
    /// `ceil(Tv / 8)`: the audio tokens the prompt holds.
    audio_tokens: usize,
    state: SortformerState,
    ready: VecDeque<EncodedWindow>,
    keep_ctc_input: bool,
    /// Everything the CTC stage and diarization segments need, kept per
    /// window when requested.
    retained: Vec<EncodedWindow>,
}

impl std::fmt::Debug for PerceptionStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerceptionStream")
            .field("windows", &self.windows.len())
            .field("next_window", &self.next_window)
            .field("audio_tokens", &self.audio_tokens)
            .finish()
    }
}

impl PerceptionStream {
    pub fn new(mel: NemotronMel, stacking: usize, keep_ctc_input: bool) -> Self {
        let windows = plan_windows(mel.frames(), stacking);
        let audio_tokens = mel.valid_frames().div_ceil(stacking).max(1);
        Self {
            mel,
            windows,
            next_window: 0,
            produced: 0,
            audio_tokens,
            state: SortformerState::default(),
            ready: VecDeque::new(),
            keep_ctc_input,
            retained: Vec::new(),
        }
    }

    pub fn audio_tokens(&self) -> usize {
        self.audio_tokens
    }

    pub fn windows(&self) -> &[PerceptionWindow] {
        &self.windows
    }

    pub fn windows_encoded(&self) -> usize {
        self.next_window
    }

    pub fn valid_frames(&self) -> usize {
        self.mel.valid_frames()
    }

    pub fn host_bytes(&self) -> usize {
        self.mel.host_bytes()
    }

    pub fn sortformer_compressions(&self) -> u64 {
        self.state.compressions
    }

    /// Encodes windows until audio tokens `..end` are available.
    fn ensure(&mut self, perception: &NemotronPerception, end: usize) -> Result<()> {
        while self.produced < end.min(self.audio_tokens) {
            let index = self.next_window;
            let window = *self.windows.get(index).ok_or_else(|| {
                Error::InferenceError(
                    "perception ran out of windows before its audio tokens".into(),
                )
            })?;
            let update = index + 1 < self.windows.len();
            let encoded = perception.encode_window(
                &self.mel,
                &window,
                &mut self.state,
                update,
                self.keep_ctc_input,
                self.produced,
            )?;
            self.next_window += 1;
            self.produced += encoded.embeddings.dim(0)?;
            if self.keep_ctc_input {
                self.retained.push(encoded.clone());
            }
            self.ready.push_back(encoded);
        }
        Ok(())
    }

    /// Audio embeddings for tokens `start..start + len` `[len, hidden]`.
    /// Tokens must be requested in order; windows wholly before `start` are
    /// released.
    pub fn embeddings(
        &mut self,
        perception: &NemotronPerception,
        start: usize,
        len: usize,
    ) -> Result<Tensor> {
        if len == 0 || start + len > self.audio_tokens {
            return Err(Error::InferenceError(format!(
                "audio tokens {start}..{} outside the prompt's {}",
                start + len,
                self.audio_tokens
            )));
        }
        self.ensure(perception, start + len)?;
        while self
            .ready
            .front()
            .is_some_and(|w| w.first_token + w.embeddings.dims()[0] <= start)
        {
            self.ready.pop_front();
        }
        let mut parts = Vec::new();
        for window in &self.ready {
            let rows = window.embeddings.dim(0)?;
            let (lo, hi) = (window.first_token, window.first_token + rows);
            let (a, b) = (start.max(lo), (start + len).min(hi));
            if a < b {
                parts.push(window.embeddings.narrow(0, a - lo, b - a)?);
            }
        }
        let out = Tensor::cat(&parts, 0)?;
        if out.dim(0)? != len {
            return Err(Error::InferenceError(
                "perception windows do not cover the requested audio tokens".into(),
            ));
        }
        Ok(out)
    }

    /// Every window, for a stream that kept them (CTC timestamps).
    pub fn retained_windows(&self) -> &[EncodedWindow] {
        &self.retained
    }

    /// The whole recording's projected embeddings `[audio_tokens, hidden]`
    /// (tests and the reference path).
    pub fn all_embeddings(&mut self, perception: &NemotronPerception) -> Result<Tensor> {
        let tokens = self.audio_tokens;
        self.embeddings(perception, 0, tokens)
    }
}

/// `[tokens, hidden]` F32 host values of a perception output (tests).
pub fn host_embeddings(tensor: &Tensor) -> Result<Vec<f32>> {
    host_values(&tensor.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_follow_the_online_plan() {
        // 95 s: Tv = 9500, T = 9501 → three windows.
        let windows = plan_windows(9501, 8);
        assert_eq!(windows.len(), 3);
        assert_eq!(
            windows[0],
            PerceptionWindow {
                start: 0,
                end: 4000,
                context_start: 0,
                context_end: 4400
            }
        );
        assert_eq!(
            windows[1],
            PerceptionWindow {
                start: 4000,
                end: 8000,
                context_start: 3600,
                context_end: 8400
            }
        );
        assert_eq!(
            windows[2],
            PerceptionWindow {
                start: 8000,
                end: 9501,
                context_start: 7600,
                context_end: 9501
            }
        );
        assert_eq!(windows[1].left_drop(8), 50);
        assert_eq!(windows[0].left_drop(8), 0);
        let cores = windows.iter().map(|w| w.core_frames(8)).collect::<Vec<_>>();
        assert_eq!(cores, vec![500, 500, 188]);
        // ceil(T/8) frames in all; the prompt keeps ceil(Tv/8) = 1188.
        assert_eq!(cores.iter().sum::<usize>(), 9501usize.div_ceil(8));
        assert_eq!(windows[2].valid_mel(9500), 1900);
    }

    #[test]
    fn a_last_window_of_one_frame_is_planned() {
        // Tv = 4000 → T = 4001: the second window holds the zero frame only.
        let windows = plan_windows(4001, 8);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[1].start, 4000);
        assert_eq!(windows[1].end, 4001);
        assert_eq!(windows[1].core_frames(8), 1);
        assert_eq!(windows[1].valid_mel(4000), 400);
        // A 30 s file is one window without context.
        let windows = plan_windows(3001, 8);
        assert_eq!(
            windows,
            vec![PerceptionWindow {
                start: 0,
                end: 3001,
                context_start: 0,
                context_end: 3001
            }]
        );
        // RED: 30 s training chunks would give a different plan for 95 s.
        assert_ne!(plan_windows(9501, 8).len(), 9501usize.div_ceil(3000));
    }

    #[test]
    fn the_speaker_kernel_interleaves_sine_and_cosine_by_speaker() {
        let k = speaker_kernel(8, 1280);
        // Speaker 0: sin(0) = 0, cos(0) = 1.
        assert!(k[..1280].chunks(2).all(|p| p[0] == 0.0 && p[1] == 1.0));
        let frequency = (6.0f32 * (-(10_000f64.ln() / 1280.0)) as f32).exp();
        let angle = 5.0 * frequency;
        assert_eq!(k[5 * 1280 + 6], bf16::from_f32(angle.sin()).to_f32());
        assert_eq!(k[5 * 1280 + 7], bf16::from_f32(angle.cos()).to_f32());
    }
}
