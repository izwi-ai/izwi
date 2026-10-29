//! The CTC timestamp head (`model-ctc-bf16.safetensors`), NeMo's
//! `TransformerCTCDecoder` as this checkpoint configures it:
//!
//! 1. `ConvTranspose1d(d, d, k = 2, s = 2)`: `out[2t + k] = b + x[t] · W[:, :, k]`,
//!    so the 80 ms encoder frames become 40 ms CTC frames.
//! 2. The two-layer trunk (`speech_transformer` without pre-encoding: q/k
//!    LayerNorm, RoPE on 40 of 80 dims, positions from 0 in each window).
//! 3. `up + residual_scale · trunk` (BF16 0.1, as upstream runs the head).
//! 4. `Conv1d(d, classes, k = 1)` and `log_softmax`; the blank is the last
//!    class.
//!
//! It reads the stored ASR encoder output (before `asr_norm` and speaker
//! fusion) in the encoder's online windows: 500-frame cores with 50 frames of
//! context on each side, trimmed back to the core. Only the columns the
//! transcript's CTC targets use (the blank first) leave the device, so an
//! hour (90,000 CTC frames over 23,863 classes) never materialises whole.

use candle_core::{Device, IndexOp, Tensor};

use super::checkpoint::{CtcGeometry, NemotronCheckpoint, CTC_PREFIX};
use super::config::{NemotronTranscribeConfig, ONLINE_CONTEXT_TOKENS, ONLINE_WINDOW_TOKENS};
use super::linear::{NtLinear, Pass, Residency};
use super::precision::Fp8Mode;
use super::speech_transformer::{host_values, SpeechTransformer};
use crate::error::{Error, Result};

pub struct NemotronCtcHead {
    /// One `[d, d]` projection per transposed-convolution tap.
    upsample: Vec<NtLinear>,
    trunk: SpeechTransformer,
    residual_scale: f64,
    classifier: NtLinear,
    classes: usize,
    model_proto: Vec<u8>,
}

impl NemotronCtcHead {
    /// Loads the head `geometry` describes from the CTC artifact.
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        geometry: &CtcGeometry,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let d = geometry.feat_in;
        if d != config.asr_encoder.d_model {
            return Err(Error::ModelLoadError(format!(
                "the CTC head reads {d} features but the ASR encoder writes {}",
                config.asr_encoder.d_model
            )));
        }
        let decoder = format!("{CTC_PREFIX}decoder");
        let factor = geometry.upsample;
        let factor_tensor = format!("{decoder}.upsample_factor");
        let stored = checkpoint.bytes(&factor_tensor)?;
        if stored.len() != 8
            || i64::from_le_bytes(stored.try_into().unwrap_or([0; 8])) != factor as i64
        {
            return Err(Error::ModelLoadError(format!(
                "{factor_tensor} disagrees with the CTC config's upsampling factor {factor}"
            )));
        }
        let dtype = Residency::of(device).dense_dtype();
        // ConvTranspose1d weight [in, out, k]: tap k maps x[t] to out[f·t + k]
        // through W[:, :, k], a Linear whose weight is its transpose [out, in].
        let weight =
            checkpoint.f32_values(&format!("{decoder}.upsample.weight"), &[d, d, factor])?;
        let mut upsample = Vec::with_capacity(factor);
        for k in 0..factor {
            let mut tap = vec![0f32; d * d];
            for i in 0..d {
                for o in 0..d {
                    tap[o * d + i] = weight[(i * d + o) * factor + k];
                }
            }
            upsample.push(
                NtLinear::dense_from_values(tap, d, d, device, dtype)?.with_bias(
                    checkpoint,
                    &format!("{decoder}.upsample"),
                    true,
                    device,
                )?,
            );
        }
        let trunk = SpeechTransformer::load(
            checkpoint,
            config,
            geometry.trunk(),
            &format!("{decoder}.transformer"),
            false,
            true,
            device,
            mode,
        )?;
        // The stored residual scale must be the config's (the head applies
        // it in BF16 upstream, which the geometry already rounded to).
        let stored_scale = checkpoint.f32_values(&format!("{decoder}.residual_scale"), &[])?;
        let config_scale = half::bf16::from_f32(stored_scale.first().copied().unwrap_or(f32::NAN));
        if config_scale.to_f32() != geometry.residual_scale {
            return Err(Error::ModelLoadError(format!(
                "the CTC head's stored residual scale {stored_scale:?} disagrees with its config {}",
                geometry.residual_scale
            )));
        }
        let classes = geometry.num_classes + 1;
        let classifier_weight = checkpoint.f32_values(
            &format!("{decoder}.decoder_layers.0.weight"),
            &[classes, d, 1],
        )?;
        let classifier = NtLinear::dense_from_values(classifier_weight, classes, d, device, dtype)?
            .with_bias(
                checkpoint,
                &format!("{decoder}.decoder_layers.0"),
                true,
                device,
            )?;
        let model_proto = checkpoint
            .bytes(&format!("{CTC_PREFIX}tokenizer.model_proto"))?
            .to_vec();
        Ok(Self {
            upsample,
            trunk,
            residual_scale: f64::from(geometry.residual_scale),
            classifier,
            classes,
            model_proto,
        })
    }

    /// Output classes, the blank included.
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The blank class (the last).
    pub fn blank(&self) -> u32 {
        (self.classes - 1) as u32
    }

    /// CTC frames per encoder frame.
    pub fn factor(&self) -> usize {
        self.upsample.len()
    }

    /// The CTC tokenizer's SentencePiece model.
    pub fn model_proto(&self) -> &[u8] {
        &self.model_proto
    }

    pub fn resident_bytes(&self) -> usize {
        self.upsample
            .iter()
            .map(NtLinear::resident_bytes)
            .sum::<usize>()
            + self.trunk.resident_bytes()
            + self.classifier.resident_bytes()
    }

    /// `trunk(x)` for one window `[frames, d]` with `valid` valid frames:
    /// the upsampled, bridged states `[factor · frames, d]`.
    fn trunk_window(&self, x: &Tensor, valid: usize) -> Result<Tensor> {
        let (frames, d) = x.dims2()?;
        let factor = self.factor();
        let taps = self
            .upsample
            .iter()
            .map(|tap| tap.forward(x, Pass::Span))
            .collect::<Result<Vec<_>>>()?;
        // [frames, factor, d] → [factor · frames, d]: frame t's taps in order.
        let up = Tensor::stack(&taps, 1)?.reshape((frames * factor, d))?;
        let bridged = self.trunk.forward_pre_encoded(&up, valid * factor)?;
        Ok((up + (bridged * self.residual_scale)?)?)
    }

    /// Log probabilities of `columns` for every CTC frame of the stored
    /// encoder output `states` `[frames, d]`: host F32 `[factor · frames,
    /// columns]`, row-major. Each frame's softmax runs over every class.
    pub fn log_prob_columns(&self, states: &Tensor, columns: &[u32]) -> Result<Vec<f32>> {
        let (frames, _) = states.dims2()?;
        if let Some(bad) = columns.iter().find(|&&c| c as usize >= self.classes) {
            return Err(Error::InferenceError(format!(
                "CTC column {bad} is outside the head's {} classes",
                self.classes
            )));
        }
        let device = states.device();
        let index = Tensor::from_vec(columns.to_vec(), columns.len(), device)?;
        let factor = self.factor();
        let mut out = Vec::with_capacity(frames * factor * columns.len());
        for start in (0..frames).step_by(ONLINE_WINDOW_TOKENS) {
            let end = (start + ONLINE_WINDOW_TOKENS).min(frames);
            let context_start = start.saturating_sub(ONLINE_CONTEXT_TOKENS);
            let context_end = (end + ONLINE_CONTEXT_TOKENS).min(frames);
            let window = states.i(context_start..context_end)?.contiguous()?;
            let bridged = self.trunk_window(&window, context_end - context_start)?;
            let core =
                bridged.narrow(0, (start - context_start) * factor, (end - start) * factor)?;
            let logits = self.classifier.forward(&core, Pass::Span)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            out.extend(host_values(&log_probs.index_select(&index, 1)?)?);
        }
        Ok(out)
    }
}
