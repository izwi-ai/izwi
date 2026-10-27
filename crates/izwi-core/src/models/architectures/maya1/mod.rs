//! Maya Research's Maya1 (3B voice-design TTS), ported natively.
//!
//! Upstream: weights <https://huggingface.co/maya-research/maya1> at
//! b1b9b38c (Apache-2.0); codec <https://huggingface.co/hubertsiuzdak/snac_24khz>
//! at d73ad176 (MIT).
//!
//! A free-text voice description and the text to speak, framed as one
//! prompt → a Llama 3.2-3B decoder sampling SNAC codes, seven per 85 ms
//! frame → SNAC's decoder → 24 kHz mono. The voice comes only from the
//! description (built-in voices are named descriptions); inline tags such
//! as `<laugh>` or `<whisper>` in the text are single tokens the model
//! performs.

pub mod config;
pub mod frames;
pub mod gguf;
mod model;
pub mod noise;
mod physical;
pub mod prompt;
pub(crate) mod retained;
pub mod sampling;
pub mod snac;
pub mod stream;
pub mod trunk;
mod weights;

#[cfg(test)]
mod fixture_support;
#[cfg(test)]
mod reference_tests;

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::Serialize;

pub use config::{
    maya1_auto_max_new_tokens, maya1_voice_preset, Maya1LmConfig, Maya1VoicePreset, MAYA1_DEFAULTS,
    MAYA1_DEFAULT_VOICE, MAYA1_INLINE_TAGS, MAYA1_VOICE_PRESETS, SAMPLE_RATE,
};
pub use gguf::MAYA1_GGUF_FILE;
pub use model::{maya1_lm_precision, Maya1TtsDiagnostics, Maya1TtsModel};
pub(crate) use model::{maya1_reloads_in_f32, Maya1AudioProgress};
pub(crate) use physical::MAYA1_DOMAIN;
pub use sampling::Maya1SamplingParams;
pub use weights::published_tensors;

use crate::error::{Error, Result};
use crate::kernels::numeric_check::{
    run_numeric_self_check, NumericSelfCheckReport, F16_SAFE_MAX_ABS,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::noise::SplitMix64;
use config::{END_OF_SPEECH, HEAD_FIRST_TOKEN, STATE_CONTEXT_TOKENS};
use frames::{is_audio_code, out_of_band_tokens, unpack_frames};
use noise::Maya1Stream;
use prompt::Maya1Tokenizer;
use sampling::HeadHistory;
use snac::SnacDecoder;
use stream::{Maya1AudioStream, MAYA1_WINDOW_PLAN};
use trunk::{Maya1ForwardRow, Maya1Trunk};

/// The loaded networks of one Maya1 model.
pub struct Maya1Runtime {
    model_dir: PathBuf,
    tokenizer: Maya1Tokenizer,
    trunk: Maya1Trunk,
    snac: SnacDecoder,
    device: Device,
}

/// A request's prompt: its token ids and how many tokens its text alone
/// takes (budgets scale with it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Maya1Prompt {
    pub ids: Vec<u32>,
    pub text_tokens: usize,
}

/// Everything one utterance's generation is parameterised by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maya1GenerationParams {
    pub sampling: Maya1SamplingParams,
    pub max_new_tokens: usize,
    /// Positive; drives sampling and the codec noise.
    pub seed: u64,
}

impl Maya1GenerationParams {
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if self.max_new_tokens == 0 || self.max_new_tokens > config::MAX_NEW_TOKENS {
            return Err(Error::InvalidInput(format!(
                "Maya1 generates 1..={} tokens per request",
                config::MAX_NEW_TOKENS
            )));
        }
        if self.seed == 0 {
            return Err(Error::InvalidInput(
                "Maya1 generation needs a resolved positive seed".into(),
            ));
        }
        Ok(())
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Maya1StopReason {
    EndOfSpeech,
    TokenBudget,
}

#[derive(Debug, Clone, Serialize)]
pub struct Maya1GenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Tokens sampled, the stop token included.
    pub tokens: Vec<u32>,
    pub stop_reason: Maya1StopReason,
    pub prompt_tokens: usize,
    /// Audio tokens sampled in another slot's band (decoded modulo 4,096).
    pub out_of_band_tokens: usize,
    pub lm_ms: f32,
    pub codec_ms: f32,
}

impl Maya1Runtime {
    /// Load the decoder in `dtype` and SNAC in F32 from `model_dir`.
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let cfg = Maya1LmConfig::load(model_dir)?;
        Self::load_with_config(model_dir, cfg, device, dtype)
    }

    /// [`Self::load`] with only the first `layers` decoder layers: the
    /// layer-truncated parity fixtures.
    #[cfg(test)]
    pub(crate) fn load_truncated(
        model_dir: &Path,
        device: &Device,
        dtype: DType,
        layers: usize,
    ) -> Result<Self> {
        let cfg = Maya1LmConfig::load(model_dir)?.truncated(layers);
        Self::load_with_config(model_dir, cfg, device, dtype)
    }

    fn load_with_config(
        model_dir: &Path,
        cfg: Maya1LmConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let tokenizer = Maya1Tokenizer::load(&model_dir.join(config::TOKENIZER_FILE))?;
        let vb = weights::mmap_lm(model_dir, dtype, device)?;
        let trunk = Maya1Trunk::load(cfg, vb, STATE_CONTEXT_TOKENS)?;
        Self::assemble(model_dir, tokenizer, trunk, device)
    }

    /// Load the GGUF release from `model_dir`: the decoder's weights in
    /// Q8_0 as stored, run in F32, and SNAC in F32.
    pub fn load_gguf(model_dir: &Path, device: &Device) -> Result<Self> {
        let cfg = Maya1LmConfig::load(model_dir)?;
        let tokenizer = Maya1Tokenizer::load(&model_dir.join(config::TOKENIZER_FILE))?;
        let trunk = gguf::load_gguf_trunk(
            &model_dir.join(MAYA1_GGUF_FILE),
            cfg,
            device,
            STATE_CONTEXT_TOKENS,
        )?;
        Self::assemble(model_dir, tokenizer, trunk, device)
    }

    fn assemble(
        model_dir: &Path,
        tokenizer: Maya1Tokenizer,
        trunk: Maya1Trunk,
        device: &Device,
    ) -> Result<Self> {
        let snac = SnacDecoder::load(&model_dir.join(config::SNAC_WEIGHTS_FILE), device)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            tokenizer,
            trunk,
            snac,
            device: device.clone(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn lm_dtype(&self) -> DType {
        self.trunk.dtype()
    }

    pub fn trunk(&self) -> &Maya1Trunk {
        &self.trunk
    }

    pub fn snac(&self) -> &SnacDecoder {
        &self.snac
    }

    pub fn tokenizer(&self) -> &Maya1Tokenizer {
        &self.tokenizer
    }

    /// The prompt for a voice `description` and `text`, both normalised and
    /// checked here.
    pub fn prompt(&self, description: &str, text: &str) -> Result<Maya1Prompt> {
        let description = prompt::normalize_description(description)?;
        let text = prompt::normalize_text(text)?;
        let ids = self.tokenizer.prompt_ids(&description, &text)?;
        if ids.len() > config::MAX_PROMPT_TOKENS {
            return Err(Error::InvalidInput(format!(
                "Maya1 prompt has {} tokens; one request takes at most {} (shorten the \
                 description or split the text)",
                ids.len(),
                config::MAX_PROMPT_TOKENS
            )));
        }
        Ok(Maya1Prompt {
            ids,
            text_tokens: self.tokenizer.text_tokens(&text)?,
        })
    }

    /// A scratch cache on this model's device for `tokens` positions.
    pub(crate) fn scratch_cache(&self, tokens: usize) -> Result<PhysicalPagedKvCache> {
        let cfg = self.trunk.config();
        PhysicalPagedKvCache::scratch_for_probe(
            &self.device,
            self.trunk.dtype(),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            tokens,
        )
    }

    /// The load-time check of a decoder below F32: a preset's prompt and a
    /// few greedy steps on a probe cache. Every probed tensor must be finite
    /// and inside F16's safe range, or the decoder reloads in F32.
    pub fn numeric_self_check(&self) -> NumericSelfCheckReport {
        const STEPS: usize = 4;
        run_numeric_self_check(
            format!("Maya1 decoder {:?}", self.trunk.dtype()),
            F16_SAFE_MAX_ABS,
            || -> Result<Vec<(String, Tensor)>> {
                let preset = &config::MAYA1_VOICE_PRESETS[0];
                let prompt = self.prompt(
                    preset.description,
                    "The quick brown fox jumps over the lazy dog, twice! <laugh>",
                )?;
                let mut cache = self.scratch_cache(prompt.ids.len() + STEPS)?;
                let mut logits = self.forward_one(&prompt.ids, &mut cache)?;
                let mut probed = vec![("prefill logits".to_string(), logits.clone())];
                for _ in 0..STEPS {
                    let row = logits.get(0)?.argmax(0)?.to_scalar::<u32>()?;
                    logits = self.forward_one(&[HEAD_FIRST_TOKEN + row], &mut cache)?;
                }
                probed.push(("decode logits".to_string(), logits));
                Ok(probed)
            },
        )
    }

    fn forward_one(&self, tokens: &[u32], cache: &mut PhysicalPagedKvCache) -> Result<Tensor> {
        self.trunk.forward(&mut [Maya1ForwardRow { tokens, cache }])
    }

    /// One utterance end to end on `cache`: prefill, sample to the stop
    /// token or the budget, then decode the audio in windows.
    pub fn generate_physical(
        &self,
        description: &str,
        text: &str,
        params: &Maya1GenerationParams,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Maya1GenerationOutput> {
        params.validate()?;
        let prompt = self.prompt(description, text)?;
        let started = Instant::now();
        let mut logits = self.forward_one(&prompt.ids, cache)?;
        let mut rng = SplitMix64::for_stream(params.seed, Maya1Stream::Sampling);
        let mut history = HeadHistory::default();
        let mut tokens = Vec::new();
        let mut stop_reason = Maya1StopReason::TokenBudget;
        while tokens.len() < params.max_new_tokens {
            let row = sampling::sample_next(
                &logits.to_vec2::<f32>()?[0],
                &history,
                tokens.len(),
                &params.sampling,
                &mut rng,
            )?;
            let token = HEAD_FIRST_TOKEN + row as u32;
            tokens.push(token);
            if token == END_OF_SPEECH {
                stop_reason = Maya1StopReason::EndOfSpeech;
                break;
            }
            history.record(row);
            if tokens.len() < params.max_new_tokens {
                logits = self.forward_one(&[token], cache)?;
            }
        }
        let lm_ms = started.elapsed().as_secs_f32() * 1000.0;
        let audio = audio_tokens(&tokens);
        let codes = unpack_frames(&audio);
        let codec_started = Instant::now();
        let mut stream = Maya1AudioStream::new(MAYA1_WINDOW_PLAN, params.seed);
        let mut samples = Vec::new();
        while let Some(window) = stream.next_window(codes.frames(), true) {
            samples.extend(stream.decode(&self.snac, &codes, window)?);
        }
        if samples.is_empty() {
            return Err(Error::InferenceError(
                "Maya1 generated no audio frames".into(),
            ));
        }
        Ok(Maya1GenerationOutput {
            samples,
            sample_rate: SAMPLE_RATE,
            out_of_band_tokens: out_of_band_tokens(&audio),
            tokens,
            stop_reason,
            prompt_tokens: prompt.ids.len(),
            lm_ms,
            codec_ms: codec_started.elapsed().as_secs_f32() * 1000.0,
        })
    }
}

/// The audio codes of a sampled sequence: everything before the stop token
/// that is a code (the sampler never yields anything else).
pub fn audio_tokens(tokens: &[u32]) -> Vec<u32> {
    tokens
        .iter()
        .take_while(|&&token| token != END_OF_SPEECH)
        .copied()
        .filter(|&token| is_audio_code(token))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_tokens_stop_at_the_end_of_speech() {
        let code = config::AUDIO_CODE_START + 5;
        assert_eq!(
            audio_tokens(&[code, code + 1, END_OF_SPEECH, code]),
            vec![code, code + 1]
        );
    }

    #[test]
    fn generation_params_validate_their_ranges() {
        let mut params = Maya1GenerationParams {
            sampling: Maya1SamplingParams::default(),
            max_new_tokens: 100,
            seed: 3,
        };
        assert!(params.validate().is_ok());
        params.seed = 0;
        assert!(params.validate().is_err());
        params.seed = 3;
        params.max_new_tokens = config::MAX_NEW_TOKENS + 1;
        assert!(params.validate().is_err());
    }
}
