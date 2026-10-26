//! Resemble AI's Chatterbox TTS (English, 0.5B), ported natively.
//!
//! Upstream: <https://github.com/resemble-ai/chatterbox> at 5de7a54a (MIT;
//! the S3Gen flow and HiFT vocoder are Apache-2.0, derived from CosyVoice),
//! weights <https://huggingface.co/ResembleAI/chatterbox> at 5bb1f6ee (MIT).
//!
//! Text → T3 (Llama speech-token LM with classifier-free guidance) → S3Gen
//! flow (Conformer encoder + causal U-Net flow matching) → HiFT vocoder →
//! 24 kHz mono. A voice is the bundled default (`conds.pt`) or one prepared
//! from reference audio alone; no transcript is involved.

pub mod conds;
pub mod config;
pub mod hift;
mod model;
pub mod noise;
mod physical;
pub(crate) mod retained;
pub mod s3gen;
pub mod sampling;
pub mod t3;
pub mod text;
mod weights;

#[cfg(test)]
mod reference_tests;

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::Serialize;

pub use conds::{ChatterboxVoice, S3GenVoiceReference, T3VoiceConditioning};
pub use config::{
    chatterbox_auto_max_speech_tokens, ChatterboxSamplingDefaults, HiftConfig, S3GenConfig,
    T3Config, CHATTERBOX_DEFAULTS, S3GEN_SAMPLE_RATE,
};
pub use model::{ChatterboxTtsDiagnostics, ChatterboxTtsModel, CHATTERBOX_DEFAULT_VOICE};
pub(crate) use physical::{CHATTERBOX_COND_DOMAIN, CHATTERBOX_UNCOND_DOMAIN};
pub use sampling::T3SamplingParams;

use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use hift::HiftVocoder;
use noise::{NoiseStream, SplitMix64};
use s3gen::{S3GenFlow, S3GenFlowState, S3GenPreparedVoice};
use t3::{T3ForwardRow, T3Model, T3PreparedVoice};
use text::ChatterboxTextTokenizer;

/// Where one utterance's acoustic stage stands. Each
/// [`ChatterboxRuntime::acoustic_step`] runs one unit (the S3Gen encoder,
/// one Euler step, or the vocoder), so a scheduler can interleave other
/// work between them.
pub enum ChatterboxAcoustic {
    /// T3 has stopped; S3Gen has not started.
    Pending,
    /// The flow ODE, some of its Euler steps taken.
    Flow(S3GenFlowState),
    /// The finished 24 kHz waveform, trim fade applied.
    Rendered(Vec<f32>),
}

/// The loaded networks of one Chatterbox model.
pub struct ChatterboxRuntime {
    model_dir: PathBuf,
    tokenizer: ChatterboxTextTokenizer,
    t3: T3Model,
    flow: S3GenFlow,
    vocoder: HiftVocoder,
    default_voice: ChatterboxPreparedVoice,
    device: Device,
    dtype: DType,
}

/// A voice embedded for both networks, shared by every request using it.
#[derive(Debug, Clone)]
pub struct ChatterboxPreparedVoice {
    pub t3: T3PreparedVoice,
    pub s3gen: S3GenPreparedVoice,
}

impl ChatterboxPreparedVoice {
    pub fn device_bytes(&self) -> usize {
        self.t3.device_bytes() + self.s3gen.device_bytes()
    }
}

/// Everything one utterance's generation is parameterised by.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatterboxGenerationParams {
    pub sampling: T3SamplingParams,
    pub exaggeration: f32,
    pub max_speech_tokens: usize,
    /// Positive; drives T3 sampling, the flow noise and the vocoder source.
    pub seed: u64,
}

impl ChatterboxGenerationParams {
    pub fn defaults(seed: u64) -> Self {
        let d = CHATTERBOX_DEFAULTS;
        Self {
            sampling: T3SamplingParams {
                temperature: d.temperature,
                repetition_penalty: d.repetition_penalty,
                min_p: d.min_p,
                top_p: d.top_p,
                cfg_weight: d.cfg_weight,
            },
            exaggeration: d.exaggeration,
            max_speech_tokens: d.max_speech_tokens,
            seed,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if !self.exaggeration.is_finite() || !(0.0..=2.0).contains(&self.exaggeration) {
            return Err(Error::InvalidInput(
                "Chatterbox exaggeration must be within [0, 2]".into(),
            ));
        }
        if self.max_speech_tokens == 0 {
            return Err(Error::InvalidInput(
                "Chatterbox needs a positive speech-token budget".into(),
            ));
        }
        if self.seed == 0 {
            return Err(Error::InvalidInput(
                "Chatterbox generation needs a resolved positive seed".into(),
            ));
        }
        Ok(())
    }
}

/// Why T3 stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatterboxStopReason {
    EndOfSpeech,
    TokenBudget,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatterboxGenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Speech tokens S3Gen rendered.
    pub speech_tokens: Vec<u32>,
    pub stop_reason: ChatterboxStopReason,
    pub text_tokens: usize,
    pub t3_ms: f32,
    pub flow_ms: f32,
    pub vocoder_ms: f32,
}

impl ChatterboxRuntime {
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let t3_cfg = T3Config::english();
        let tokenizer = ChatterboxTextTokenizer::load(
            &model_dir.join(config::TOKENIZER_FILE),
            t3_cfg.start_text_token,
            t3_cfg.stop_text_token,
        )?;
        let t3_vb =
            weights::mmap_safetensors(&model_dir.join(config::T3_WEIGHTS_FILE), dtype, device)?;
        let t3 = T3Model::load(t3_cfg, t3_vb)?;
        let s3gen_vb =
            weights::mmap_safetensors(&model_dir.join(config::S3GEN_WEIGHTS_FILE), dtype, device)?;
        let flow = S3GenFlow::load(S3GenConfig::current(), s3gen_vb.clone())?;
        let vocoder = HiftVocoder::load(HiftConfig::current(), s3gen_vb.pp("mel2wav"))?;
        let default = ChatterboxVoice::load_default(&model_dir.join(config::DEFAULT_VOICE_FILE))?;
        let default_voice = ChatterboxPreparedVoice {
            t3: t3.prepare_voice(&default.t3)?,
            s3gen: flow.prepare_voice(&default.s3gen)?,
        };
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            tokenizer,
            t3,
            flow,
            vocoder,
            default_voice,
            device: device.clone(),
            dtype,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn t3(&self) -> &T3Model {
        &self.t3
    }

    pub fn flow(&self) -> &S3GenFlow {
        &self.flow
    }

    pub fn vocoder(&self) -> &HiftVocoder {
        &self.vocoder
    }

    pub fn tokenizer(&self) -> &ChatterboxTextTokenizer {
        &self.tokenizer
    }

    pub fn default_voice(&self) -> &ChatterboxPreparedVoice {
        &self.default_voice
    }

    pub fn prepare_voice(&self, voice: &ChatterboxVoice) -> Result<ChatterboxPreparedVoice> {
        voice.validate()?;
        Ok(ChatterboxPreparedVoice {
            t3: self.t3.prepare_voice(&voice.t3)?,
            s3gen: self.flow.prepare_voice(&voice.s3gen)?,
        })
    }

    /// T3 positions one request needs: conditioning, framed text, the two
    /// start tokens and the speech budget.
    pub fn sequence_tokens(&self, text_tokens: usize, max_speech_tokens: usize) -> usize {
        self.t3.config().conditioning_len() + text_tokens + 2 + max_speech_tokens
    }

    /// One utterance end to end on lifecycle-owned caches: the conditional
    /// sequence's and, with guidance, the unconditional one's (a different
    /// arena).
    pub fn generate_physical(
        &self,
        text: &str,
        voice: &ChatterboxPreparedVoice,
        params: &ChatterboxGenerationParams,
        conditional: &mut PhysicalPagedKvCache,
        mut unconditional: Option<&mut PhysicalPagedKvCache>,
    ) -> Result<ChatterboxGenerationOutput> {
        params.validate()?;
        let guided = params.sampling.guided();
        if guided != unconditional.is_some() {
            return Err(Error::InvalidInput(
                "Chatterbox guidance needs exactly one unconditional cache".into(),
            ));
        }
        let cfg = self.t3.config().clone();
        let text_ids = self.tokenizer.text_sequence(text)?;
        let started = Instant::now();
        let conditioning = self.t3.conditioning(&voice.t3, params.exaggeration)?;
        let (cond_embeds, uncond_embeds) =
            self.t3.prefill_embeds(&conditioning, &text_ids, guided)?;
        let mut logits = self.forward_pair(
            cond_embeds,
            uncond_embeds,
            conditional,
            unconditional.as_deref_mut(),
        )?;
        let mut rng = SplitMix64::for_stream(params.seed, NoiseStream::Sampling);
        let mut history = vec![cfg.start_speech_token];
        let mut stop_reason = ChatterboxStopReason::TokenBudget;
        for step in 0..params.max_speech_tokens {
            let rows = logits.to_vec2::<f32>()?;
            let token = sampling::sample_next(
                &rows[0],
                rows.get(1).map(Vec::as_slice),
                &history,
                &params.sampling,
                &mut rng,
            )?;
            history.push(token);
            if token == cfg.stop_speech_token {
                stop_reason = ChatterboxStopReason::EndOfSpeech;
                break;
            }
            if step + 1 == params.max_speech_tokens {
                break;
            }
            let embed = self.t3.speech_step_embed(token, step + 1)?;
            logits = self.forward_pair(
                embed.clone(),
                guided.then_some(embed),
                conditional,
                unconditional.as_deref_mut(),
            )?;
        }
        let t3_ms = started.elapsed().as_secs_f32() * 1000.0;
        let speech_tokens = renderable_speech_tokens(&history[1..], &cfg);
        let samples_started = Instant::now();
        let (samples, flow_ms) = self.render(voice, &speech_tokens, params.seed)?;
        let vocoder_ms = samples_started.elapsed().as_secs_f32() * 1000.0 - flow_ms;
        Ok(ChatterboxGenerationOutput {
            samples,
            sample_rate: S3GEN_SAMPLE_RATE,
            speech_tokens,
            stop_reason,
            text_tokens: text_ids.len(),
            t3_ms,
            flow_ms,
            vocoder_ms,
        })
    }

    fn forward_pair(
        &self,
        conditional: Tensor,
        unconditional: Option<Tensor>,
        cond_cache: &mut PhysicalPagedKvCache,
        uncond_cache: Option<&mut PhysicalPagedKvCache>,
    ) -> Result<Tensor> {
        let mut rows = vec![T3ForwardRow {
            embeds: conditional,
            cache: cond_cache,
        }];
        if let (Some(embeds), Some(cache)) = (unconditional, uncond_cache) {
            rows.push(T3ForwardRow { embeds, cache });
        }
        self.t3.forward(&mut rows)
    }

    /// S3Gen and HiFT for renderable speech tokens; returns the samples and
    /// the flow's share of the time.
    pub fn render(
        &self,
        voice: &ChatterboxPreparedVoice,
        speech_tokens: &[u32],
        seed: u64,
    ) -> Result<(Vec<f32>, f32)> {
        let started = Instant::now();
        let mut flow_ms = 0.0;
        let mut progress = ChatterboxAcoustic::Pending;
        while !self.acoustic_step(voice, speech_tokens, seed, &mut progress)? {
            if matches!(&progress, ChatterboxAcoustic::Flow(state)
                if state.steps_done() == self.flow.config().cfm_steps)
            {
                flow_ms = started.elapsed().as_secs_f32() * 1000.0;
            }
        }
        let ChatterboxAcoustic::Rendered(samples) = progress else {
            unreachable!("acoustic_step reports done only once rendered");
        };
        Ok((samples, flow_ms))
    }

    /// Acoustic units one utterance takes: the encoder, each Euler step and
    /// the vocoder.
    pub fn acoustic_units(&self) -> usize {
        self.flow.config().cfm_steps + 2
    }

    /// Run the next acoustic unit of `speech_tokens`, waiting for the device
    /// so the unit's cost lands in its own quantum. Returns true once the
    /// waveform is rendered.
    pub fn acoustic_step(
        &self,
        voice: &ChatterboxPreparedVoice,
        speech_tokens: &[u32],
        seed: u64,
        progress: &mut ChatterboxAcoustic,
    ) -> Result<bool> {
        let next = match progress {
            ChatterboxAcoustic::Pending => {
                if speech_tokens.is_empty() {
                    return Err(Error::InferenceError(
                        "Chatterbox generated no renderable speech tokens".into(),
                    ));
                }
                let frames =
                    S3GenFlow::total_frames(voice.s3gen.prompt_tokens(), speech_tokens.len());
                let noise = SplitMix64::for_stream(seed, NoiseStream::FlowNoise)
                    .standard_normals(self.flow.config().mel_bins * frames);
                Some(ChatterboxAcoustic::Flow(self.flow.begin(
                    &voice.s3gen,
                    speech_tokens,
                    &noise,
                    true,
                )?))
            }
            ChatterboxAcoustic::Flow(state)
                if state.steps_done() < self.flow.config().cfm_steps =>
            {
                self.flow.step(state, 1)?;
                None
            }
            ChatterboxAcoustic::Flow(state) => {
                let mel = self.flow.finish(state)?;
                let mut samples = self.vocoder.vocode(&mel, seed)?;
                hift::apply_trim_fade(&mut samples);
                if samples.iter().any(|sample| !sample.is_finite()) {
                    return Err(Error::InferenceError(
                        "Chatterbox vocoder produced non-finite samples".into(),
                    ));
                }
                Some(ChatterboxAcoustic::Rendered(samples))
            }
            ChatterboxAcoustic::Rendered(_) => None,
        };
        if let Some(next) = next {
            *progress = next;
        }
        self.device.synchronize()?;
        Ok(matches!(progress, ChatterboxAcoustic::Rendered(_)))
    }
}

/// Upstream's post-T3 filtering (`drop_invalid_tokens`, then `< 6561`):
/// drop everything up to the first start token, cut at the first stop
/// token, and keep only codes S3Gen can render.
pub fn renderable_speech_tokens(sampled: &[u32], cfg: &T3Config) -> Vec<u32> {
    let start = sampled
        .iter()
        .position(|&token| token == cfg.start_speech_token)
        .map_or(0, |index| index + 1);
    let end = sampled
        .iter()
        .position(|&token| token == cfg.stop_speech_token)
        .unwrap_or(sampled.len());
    if start >= end {
        return Vec::new();
    }
    sampled[start..end]
        .iter()
        .copied()
        .filter(|&token| token < cfg.acoustic_code_count())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderable_tokens_follow_upstream_filtering() {
        let cfg = T3Config::english();
        assert_eq!(renderable_speech_tokens(&[5, 6, 6562, 7], &cfg), vec![5, 6]);
        assert_eq!(renderable_speech_tokens(&[5, 6561, 8, 9], &cfg), vec![8, 9]);
        assert_eq!(renderable_speech_tokens(&[5, 7000, 9], &cfg), vec![5, 9]);
        assert!(renderable_speech_tokens(&[6562, 5], &cfg).is_empty());
    }

    #[test]
    fn generation_params_validate_their_ranges() {
        let mut params = ChatterboxGenerationParams::defaults(3);
        assert!(params.validate().is_ok());
        params.exaggeration = 2.5;
        assert!(params.validate().is_err());
        params = ChatterboxGenerationParams::defaults(0);
        assert!(params.validate().is_err());
    }
}
