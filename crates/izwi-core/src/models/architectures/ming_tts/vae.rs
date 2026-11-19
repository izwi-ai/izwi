//! Ming's audio VAE (`audio_tokenizer/`): 44.1 kHz audio ↔ 64-wide latents
//! at 12.5 Hz, with a Qwen2 stack on each side and no convolutions.
//!
//! Decode: latent → ×4 linear upsample in time (50 Hz) → linear to the
//! stack's width → the causal, windowed stack → linear to log-magnitude and
//! phase → inverse STFT (3,528-point window, 882 hop, "same" padding).
//! [`MingVaeDecodeState`] decodes latents as they arrive: the upsampler
//! waits one latent, each stack layer keeps the keys a later frame can see,
//! and the overlap-add holds the samples later frames still add to. What it
//! emits for a latent sequence fed in pieces is what it emits for the
//! sequence fed at once.
//!
//! Encode (reference clips only): 882-sample frames → two linears → the
//! stack → a learned token after every four frames → a 4-layer stack → the
//! token positions → mean and scale of the latent.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::config::{
    MingTtsConfig, FRAMES_PER_LATENT, FRAME_SAMPLES, ISTFT_BINS, ISTFT_SIZE, LATENT_DIM,
    LATENT_SAMPLES,
};
use super::stack::{MingWindowStack, MingWindowState};
use crate::error::{Error, Result};

/// Samples trimmed from each end of the overlap-added signal.
const ISTFT_PAD: usize = (ISTFT_SIZE - FRAME_SAMPLES) / 2;
/// The head clips magnitudes here (`torch.clip(mag, max=1e2)`).
const MAX_MAGNITUDE: f32 = 100.0;
/// The floor added to the posterior's scale (`softplus(scale) + 1e-4`).
const POSTERIOR_SCALE_FLOOR: f32 = 1e-4;

/// `x` [rows, in] through `weight` [out, in] as stored (transposed as a
/// view, so no transposed copy is written on the device).
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    // Metal's matmul takes a left operand only with unit row stride: the
    // encoder's group tokens are a strided view.
    let y = x.contiguous()?.matmul(&weight.t()?)?;
    match bias {
        Some(bias) => Ok(y.broadcast_add(bias)?),
        None => Ok(y),
    }
}

fn weight_t(vb: &VarBuilder, output: usize, input: usize) -> Result<Tensor> {
    Ok(vb.get((output, input), "weight")?)
}

/// The inverse STFT of the head: one frame's spectrum to its windowed
/// signal.
struct IstftFrame {
    fft: Arc<dyn Fft<f32>>,
    /// Periodic Hann window, computed in F64.
    window: Vec<f32>,
    window_sq: Vec<f32>,
}

impl IstftFrame {
    fn new() -> Self {
        let window = (0..ISTFT_SIZE)
            .map(|n| {
                let phase = 2.0 * std::f64::consts::PI * n as f64 / ISTFT_SIZE as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect::<Vec<_>>();
        Self {
            fft: FftPlanner::<f32>::new().plan_fft_inverse(ISTFT_SIZE),
            window_sq: window.iter().map(|w| w * w).collect(),
            window,
        }
    }

    /// `head` is one frame of the head's output: 1,765 log-magnitudes, then
    /// 1,765 phases. Returns the frame's 3,528 windowed samples.
    fn frame(&self, head: &[f32], buffer: &mut [Complex<f32>]) -> Vec<f32> {
        let (log_mag, phase) = head.split_at(ISTFT_BINS);
        for bin in 0..ISTFT_BINS {
            let magnitude = log_mag[bin].exp().min(MAX_MAGNITUDE);
            let (sin, cos) = phase[bin].sin_cos();
            buffer[bin] = Complex::new(magnitude * cos, magnitude * sin);
        }
        // A real signal's spectrum: the upper half mirrors the lower, and
        // the imaginary parts of the DC and Nyquist bins do not count.
        buffer[0].im = 0.0;
        buffer[ISTFT_BINS - 1].im = 0.0;
        for bin in 1..ISTFT_BINS - 1 {
            buffer[ISTFT_SIZE - bin] = buffer[bin].conj();
        }
        self.fft.process(buffer);
        let scale = 1.0 / ISTFT_SIZE as f32;
        buffer
            .iter()
            .zip(&self.window)
            .map(|(value, window)| value.re * scale * window)
            .collect()
    }
}

/// One utterance's decode in progress.
pub struct MingVaeDecodeState {
    /// Every latent pushed so far, 64 values each.
    latents: Vec<f32>,
    /// Frames already run through the stack.
    frames: usize,
    stack: MingWindowState,
    /// Overlap-add accumulators from sample `base` of the untrimmed signal.
    signal: Vec<f32>,
    envelope: Vec<f32>,
    base: usize,
    /// The next untrimmed sample to emit.
    cursor: usize,
    finished: bool,
}

impl Default for MingVaeDecodeState {
    fn default() -> Self {
        Self {
            latents: Vec::new(),
            frames: 0,
            stack: MingWindowState::default(),
            signal: Vec::new(),
            envelope: Vec::new(),
            base: 0,
            cursor: ISTFT_PAD,
            finished: false,
        }
    }
}

impl MingVaeDecodeState {
    /// Latents pushed so far.
    pub fn latents(&self) -> usize {
        self.latents.len() / LATENT_DIM
    }

    /// Samples emitted so far.
    pub fn samples(&self) -> usize {
        self.cursor - ISTFT_PAD
    }

    pub fn finished(&self) -> bool {
        self.finished
    }
}

pub struct MingVaeDecoder {
    fc1_t: Tensor,
    fc1_bias: Tensor,
    stack: MingWindowStack,
    head_t: Tensor,
    head_bias: Tensor,
    istft: IstftFrame,
}

impl MingVaeDecoder {
    /// Load the decoder from `vb` (rooted at `audio.decoder`).
    pub fn load(cfg: &MingTtsConfig, vb: VarBuilder, window_keys: usize) -> Result<Self> {
        let hidden = cfg.vae_stack.hidden_size;
        let fc1 = vb.pp("fc1");
        let head = vb.pp("head").pp("out");
        Ok(Self {
            fc1_t: weight_t(&fc1, hidden, LATENT_DIM)?,
            fc1_bias: fc1.get(hidden, "bias")?,
            head_t: weight_t(&head, 2 * ISTFT_BINS, hidden)?,
            head_bias: head.get(2 * ISTFT_BINS, "bias")?,
            stack: MingWindowStack::load(cfg.vae_stack.clone(), vb.pp("decoder"), window_keys)?,
            istft: IstftFrame::new(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.stack.dtype()
    }

    pub fn window_keys(&self) -> usize {
        self.stack.window_keys()
    }

    /// Feed `latents` (64 values each) and return the samples they settle.
    /// `last` ends the utterance: everything left is emitted.
    pub fn push(
        &self,
        state: &mut MingVaeDecodeState,
        latents: &[f32],
        last: bool,
    ) -> Result<Vec<f32>> {
        if state.finished {
            return Err(Error::InvalidInput(
                "Ming VAE decode was already finished".into(),
            ));
        }
        if !latents.len().is_multiple_of(LATENT_DIM)
            || latents.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InferenceError(
                "Ming VAE decode received an invalid latent".into(),
            ));
        }
        state.latents.extend_from_slice(latents);
        let count = state.latents();
        if count == 0 {
            if last {
                state.finished = true;
            }
            return Ok(Vec::new());
        }
        // The last two frames of a latent interpolate towards the next one.
        let ready = if last {
            count * FRAMES_PER_LATENT
        } else {
            count * FRAMES_PER_LATENT - FRAMES_PER_LATENT / 2
        };
        if ready > state.frames {
            let frames = upsample_frames(&state.latents, state.frames, ready);
            let new = ready - state.frames;
            let device = self.stack.device();
            let x = Tensor::from_vec(frames, (new, LATENT_DIM), device)?.to_dtype(self.dtype())?;
            let x = linear(&x, &self.fc1_t, Some(&self.fc1_bias))?;
            let hidden = self.stack.forward(&x, &mut state.stack)?;
            let head = linear(&hidden, &self.head_t, Some(&self.head_bias))?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if head.iter().any(|value| !value.is_finite()) {
                return Err(Error::InferenceError(
                    "Ming VAE decoder produced a non-finite spectrum".into(),
                ));
            }
            let mut buffer = vec![Complex::new(0f32, 0f32); ISTFT_SIZE];
            for (index, frame) in head.as_chunks::<{ 2 * ISTFT_BINS }>().0.iter().enumerate() {
                let start = (state.frames + index) * FRAME_SAMPLES - state.base;
                let end = start + ISTFT_SIZE;
                if state.signal.len() < end {
                    state.signal.resize(end, 0.0);
                    state.envelope.resize(end, 0.0);
                }
                let samples = self.istft.frame(frame, &mut buffer);
                for (offset, sample) in samples.into_iter().enumerate() {
                    state.signal[start + offset] += sample;
                    state.envelope[start + offset] += self.istft.window_sq[offset];
                }
            }
            state.frames = ready;
        }
        // Samples before the next frame's start get nothing more; at the
        // end, everything up to the trimmed tail is final.
        let settled = if last {
            (state.frames * FRAME_SAMPLES + ISTFT_SIZE - FRAME_SAMPLES).saturating_sub(ISTFT_PAD)
        } else {
            state.frames * FRAME_SAMPLES
        };
        let mut out = Vec::with_capacity(settled.saturating_sub(state.cursor));
        while state.cursor < settled {
            let at = state.cursor - state.base;
            let envelope = state.envelope[at];
            if envelope <= 1e-11 {
                return Err(Error::InferenceError(
                    "Ming inverse STFT window envelope vanished".into(),
                ));
            }
            out.push(state.signal[at] / envelope);
            state.cursor += 1;
        }
        // Drop what no later frame or emission reads: everything before
        // both the cursor and the next frame's start.
        let drop = state
            .cursor
            .min(state.frames * FRAME_SAMPLES)
            .saturating_sub(state.base)
            .min(state.signal.len());
        if drop > 0 {
            state.signal.drain(..drop);
            state.envelope.drain(..drop);
            state.base += drop;
        }
        if last {
            state.finished = true;
        }
        Ok(out)
    }

    /// Decode a whole utterance in one call.
    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        self.push(&mut MingVaeDecodeState::default(), latents, true)
    }
}

/// Frames `from..to` of the ×4 linear upsampling of `latents` (PyTorch's
/// `Upsample(mode="linear", align_corners=False)`): half-pixel centres,
/// clamped at both ends. Row-major [to - from, 64].
fn upsample_frames(latents: &[f32], from: usize, to: usize) -> Vec<f32> {
    let count = latents.len() / LATENT_DIM;
    let mut frames = Vec::with_capacity((to - from) * LATENT_DIM);
    for frame in from..to {
        let source = ((frame as f64 + 0.5) / FRAMES_PER_LATENT as f64 - 0.5).max(0.0);
        let low = (source.floor() as usize).min(count - 1);
        let high = (low + 1).min(count - 1);
        let weight = (source - low as f64) as f32;
        let a = &latents[low * LATENT_DIM..(low + 1) * LATENT_DIM];
        let b = &latents[high * LATENT_DIM..(high + 1) * LATENT_DIM];
        frames.extend(
            a.iter()
                .zip(b)
                .map(|(a, b)| a * (1.0 - weight) + b * weight),
        );
    }
    frames
}

pub struct MingVaeEncoder {
    fc1_t: Tensor,
    fc2_t: Tensor,
    fc2_bias: Tensor,
    fc3_t: Tensor,
    fc3_bias: Tensor,
    stack: MingWindowStack,
    group_stack: MingWindowStack,
    /// The token placed after each group of frames, [1, hidden].
    group_token: Tensor,
}

impl MingVaeEncoder {
    /// Load the encoder from `vb` (rooted at `audio.encoder`).
    pub fn load(cfg: &MingTtsConfig, vb: VarBuilder, window_keys: usize) -> Result<Self> {
        let hidden = cfg.vae_stack.hidden_size;
        let fc2 = vb.pp("fc2");
        let fc3 = vb.pp("fc3");
        Ok(Self {
            fc1_t: weight_t(&vb.pp("fc1"), hidden, FRAME_SAMPLES)?,
            fc2_t: weight_t(&fc2, hidden, hidden)?,
            fc2_bias: fc2.get(hidden, "bias")?,
            fc3_t: weight_t(&fc3, 2 * LATENT_DIM, hidden)?,
            fc3_bias: fc3.get(2 * LATENT_DIM, "bias")?,
            group_token: vb.get((1, 1, hidden), "cls_embed")?.reshape((1, hidden))?,
            stack: MingWindowStack::load(cfg.vae_stack.clone(), vb.pp("encoder"), window_keys)?,
            group_stack: MingWindowStack::load(
                cfg.vae_group_stack.clone(),
                vb.pp("aggregator"),
                window_keys,
            )?,
        })
    }

    /// The posterior of `samples` (44.1 kHz mono, a whole number of
    /// latents): mean and scale, each [latents * 64].
    pub fn posterior(&self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if samples.is_empty() || !samples.len().is_multiple_of(LATENT_SAMPLES) {
            return Err(Error::InvalidInput(format!(
                "Ming VAE encodes a whole number of {LATENT_SAMPLES}-sample latents"
            )));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidInput(
                "Ming reference audio is not finite".into(),
            ));
        }
        let frames = samples.len() / FRAME_SAMPLES;
        let latents = frames / FRAMES_PER_LATENT;
        let device = self.stack.device();
        let dtype = self.stack.dtype();
        let x =
            Tensor::from_vec(samples.to_vec(), (frames, FRAME_SAMPLES), device)?.to_dtype(dtype)?;
        let x = linear(
            &linear(&x, &self.fc1_t, None)?,
            &self.fc2_t,
            Some(&self.fc2_bias),
        )?;
        let hidden = self.stack.forward(&x, &mut MingWindowState::default())?;
        // The group token after every four frames.
        let width = hidden.dim(1)?;
        let grouped = Tensor::cat(
            &[
                hidden.reshape((latents, FRAMES_PER_LATENT, width))?,
                self.group_token
                    .broadcast_as((latents, 1, width))?
                    .contiguous()?,
            ],
            1,
        )?
        .reshape((latents * (FRAMES_PER_LATENT + 1), width))?;
        let grouped = self
            .group_stack
            .forward(&grouped, &mut MingWindowState::default())?;
        let tokens = grouped
            .reshape((latents, FRAMES_PER_LATENT + 1, width))?
            .narrow(1, FRAMES_PER_LATENT, 1)?
            .squeeze(1)?
            .contiguous()?;
        let moments = linear(&tokens, &self.fc3_t, Some(&self.fc3_bias))?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let mean = moments
            .narrow(1, 0, LATENT_DIM)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let scale = moments
            .narrow(1, LATENT_DIM, LATENT_DIM)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if mean.iter().chain(&scale).any(|value| !value.is_finite()) {
            return Err(Error::InferenceError(
                "Ming VAE encoder produced a non-finite posterior".into(),
            ));
        }
        Ok((mean, scale))
    }

    /// Latents of `samples`: `mean + (softplus(scale) + 1e-4) * noise`, as
    /// the reference samples its posterior. `noise` is one standard normal
    /// per latent value.
    pub fn encode(&self, samples: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let (mean, scale) = self.posterior(samples)?;
        if noise.len() != mean.len() {
            return Err(Error::InvalidInput(
                "Ming VAE encode needs one noise value per latent value".into(),
            ));
        }
        Ok(mean
            .iter()
            .zip(&scale)
            .zip(noise)
            .map(|((mean, scale), noise)| mean + (softplus(*scale) + POSTERIOR_SCALE_FLOOR) * noise)
            .collect())
    }
}

/// `log(1 + exp(x))`, linear past PyTorch's threshold of 20.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::models::architectures::ming_tts::stack::tests::{
        max_diff, random, tiny_stack_config, tiny_stack_weights,
    };
    use crate::models::shared::noise::SplitMix64;

    pub(crate) fn tiny_vae_config() -> MingTtsConfig {
        let mut cfg = MingTtsConfig::pinned();
        cfg.vae_stack = tiny_stack_config(2);
        cfg.vae_group_stack = tiny_stack_config(1);
        cfg
    }

    pub(crate) fn tiny_decoder_weights(
        cfg: &MingTtsConfig,
        prefix: &str,
        rng: &mut SplitMix64,
        tensors: &mut HashMap<String, Tensor>,
    ) {
        let hidden = cfg.vae_stack.hidden_size;
        tiny_stack_weights(&cfg.vae_stack, &format!("{prefix}decoder."), rng, tensors);
        tensors.insert(
            format!("{prefix}fc1.weight"),
            random(rng, &[hidden, LATENT_DIM], 0.2),
        );
        tensors.insert(format!("{prefix}fc1.bias"), random(rng, &[hidden], 0.1));
        tensors.insert(
            format!("{prefix}head.out.weight"),
            random(rng, &[2 * ISTFT_BINS, hidden], 0.05),
        );
        tensors.insert(
            format!("{prefix}head.out.bias"),
            random(rng, &[2 * ISTFT_BINS], 0.05),
        );
    }

    pub(crate) fn tiny_encoder_weights(
        cfg: &MingTtsConfig,
        prefix: &str,
        rng: &mut SplitMix64,
        tensors: &mut HashMap<String, Tensor>,
    ) {
        let hidden = cfg.vae_stack.hidden_size;
        tiny_stack_weights(&cfg.vae_stack, &format!("{prefix}encoder."), rng, tensors);
        tiny_stack_weights(
            &cfg.vae_group_stack,
            &format!("{prefix}aggregator."),
            rng,
            tensors,
        );
        tensors.insert(
            format!("{prefix}fc1.weight"),
            random(rng, &[hidden, FRAME_SAMPLES], 0.05),
        );
        tensors.insert(
            format!("{prefix}fc2.weight"),
            random(rng, &[hidden, hidden], 0.2),
        );
        tensors.insert(format!("{prefix}fc2.bias"), random(rng, &[hidden], 0.1));
        tensors.insert(
            format!("{prefix}fc3.weight"),
            random(rng, &[2 * LATENT_DIM, hidden], 0.2),
        );
        tensors.insert(
            format!("{prefix}fc3.bias"),
            random(rng, &[2 * LATENT_DIM], 0.1),
        );
        tensors.insert(
            format!("{prefix}cls_embed"),
            random(rng, &[1, 1, hidden], 0.2),
        );
    }

    fn decoder(window_keys: usize) -> MingVaeDecoder {
        let cfg = tiny_vae_config();
        let mut rng = SplitMix64::new(21);
        let mut tensors = HashMap::new();
        tiny_decoder_weights(&cfg, "", &mut rng, &mut tensors);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        MingVaeDecoder::load(&cfg, vb, window_keys).unwrap()
    }

    #[test]
    fn upsampling_matches_half_pixel_linear_interpolation() {
        // Three latents whose first value is 0, 4, 8.
        let mut latents = vec![0f32; 3 * LATENT_DIM];
        latents[LATENT_DIM] = 4.0;
        latents[2 * LATENT_DIM] = 8.0;
        let frames = upsample_frames(&latents, 0, 12);
        let firsts = frames.chunks(LATENT_DIM).map(|f| f[0]).collect::<Vec<_>>();
        // torch.nn.Upsample(scale_factor=4, mode="linear") of [0, 4, 8].
        let expected = [0.0, 0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.0, 8.0];
        assert!(max_diff(&firsts, &expected) < 1e-6, "{firsts:?}");
    }

    #[test]
    fn inverse_stft_of_a_flat_frame_sequence_has_the_trimmed_length() {
        let decoder = decoder(8);
        let latents = SplitMix64::new(2).standard_normals(5 * LATENT_DIM);
        let samples = decoder.decode(&latents).unwrap();
        assert_eq!(samples.len(), 5 * LATENT_SAMPLES);
        assert!(samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn latents_decoded_in_pieces_equal_the_utterance_decoded_at_once() {
        let decoder = decoder(6);
        let latents = SplitMix64::new(3).standard_normals(11 * LATENT_DIM);
        let whole = decoder.decode(&latents).unwrap();
        let mut state = MingVaeDecodeState::default();
        let mut pieces = Vec::new();
        let mut offset = 0;
        let plan = [1usize, 3, 2, 4, 1];
        for (index, count) in plan.iter().enumerate() {
            let last = index + 1 == plan.len();
            let slice = &latents[offset * LATENT_DIM..(offset + count) * LATENT_DIM];
            pieces.extend(decoder.push(&mut state, slice, last).unwrap());
            offset += count;
        }
        assert!(state.finished());
        assert_eq!(state.samples(), whole.len());
        assert_eq!(pieces.len(), whole.len());
        assert!(max_diff(&pieces, &whole) < 2e-4);
        // A non-final push holds back the samples later frames still add to.
        let mut state = MingVaeDecodeState::default();
        let first = decoder
            .push(&mut state, &latents[..2 * LATENT_DIM], false)
            .unwrap();
        assert_eq!(first.len(), 6 * FRAME_SAMPLES - ISTFT_PAD);
    }

    #[test]
    fn encoder_returns_one_latent_per_four_frames() {
        let cfg = tiny_vae_config();
        let mut rng = SplitMix64::new(31);
        let mut tensors = HashMap::new();
        tiny_encoder_weights(&cfg, "", &mut rng, &mut tensors);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let encoder = MingVaeEncoder::load(&cfg, vb, 8).unwrap();
        let samples = rng
            .standard_normals(3 * LATENT_SAMPLES)
            .into_iter()
            .map(|sample| sample * 0.1)
            .collect::<Vec<_>>();
        let (mean, scale) = encoder.posterior(&samples).unwrap();
        assert_eq!(mean.len(), 3 * LATENT_DIM);
        assert_eq!(scale.len(), 3 * LATENT_DIM);
        // Zero noise gives the mean; unit noise moves by the floored scale.
        let zero = encoder.encode(&samples, &vec![0.0; mean.len()]).unwrap();
        assert!(max_diff(&zero, &mean) < 1e-6);
        let unit = encoder.encode(&samples, &vec![1.0; mean.len()]).unwrap();
        for ((unit, mean), scale) in unit.iter().zip(&mean).zip(&scale) {
            let expected = mean + softplus(*scale) + POSTERIOR_SCALE_FLOOR;
            assert!((unit - expected).abs() < 1e-5);
        }
        // The encoder is causal: a longer clip keeps the first latent.
        let (longer, _) = encoder
            .posterior(&[samples.clone(), samples[..LATENT_SAMPLES].to_vec()].concat())
            .unwrap();
        assert!(max_diff(&longer[..LATENT_DIM], &mean[..LATENT_DIM]) < 1e-4);
        assert!(encoder.posterior(&samples[..100]).is_err());
    }

    #[test]
    fn softplus_matches_its_definition() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        assert_eq!(softplus(30.0), 30.0);
        assert!(softplus(-30.0) > 0.0);
    }
}
