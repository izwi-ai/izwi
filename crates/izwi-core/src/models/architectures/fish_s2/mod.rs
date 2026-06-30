//! Fish Audio S2 Pro TTS architecture boundary.

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

pub mod artifacts;
pub mod codec;
pub mod config;
pub mod contracts;
pub mod dac;
pub mod fast;
pub mod slow;
pub mod tokenizer;
pub mod weights;

pub use artifacts::FishS2ArtifactManifest;
pub use codec::{FishS2CodecArtifact, FishS2CodecSupport};
pub use config::{FishS2AudioDecoderConfig, FishS2Config, FishS2TextConfig};
pub use contracts::{
    build_semantic_allowed_mask, remap_fish_qwen3_omni_key, semantic_code_from_token_id,
    semantic_token_id, FishS2DacContract, FishS2PromptTensorShape,
};
pub use dac::{FishS2DacConfig, FishS2DacDecoder};
pub use fast::{
    FishS2FastCache, FishS2FastConfig, FishS2FastDecoder, FishS2GeneratedFrame, FishS2Sampler,
};
pub use slow::{FishS2SlowCache, FishS2SlowConfig, FishS2SlowOutput, FishS2SlowTransformer};
pub use tokenizer::{
    FishS2ConditioningPrompt, FishS2PromptTokenizer, FishS2SpecialTokens, FishS2VqCodes,
};
pub use weights::{FishS2TensorSpec, FishS2WeightIndex, FishS2Weights};

pub struct FishS2TtsModel {
    variant: ModelVariant,
    config: FishS2Config,
    artifacts: FishS2ArtifactManifest,
    codec: FishS2CodecArtifact,
    runtime: Option<FishS2NativeRuntime>,
}

struct FishS2NativeRuntime {
    tokenizer: FishS2PromptTokenizer,
    slow: FishS2SlowTransformer,
    fast: FishS2FastDecoder,
    dac: FishS2DacDecoder,
    semantic_allowed_mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct FishS2Reference {
    pub audio_samples: Vec<f32>,
    pub sample_rate: u32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct FishS2GenerationParams {
    pub max_frames: usize,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone)]
pub struct FishS2GenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames_generated: usize,
    pub diagnostics: FishS2TtsGenerationDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct FishS2TtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub native_runtime_loaded: bool,
    pub sample_rate: u32,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub max_seq_len: usize,
    pub codec_support: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct FishS2TtsGenerationDiagnostics {
    pub model_family: &'static str,
    pub sample_rate: u32,
    pub prompt_tokens: usize,
    pub max_frames: usize,
    pub frames_generated: usize,
    pub stop_reason: String,
    pub reference_encode_ms: f32,
    pub prompt_build_ms: f32,
    pub slow_prefill_ms: f32,
    pub ar_decode_ms: f32,
    pub dac_decode_ms: f32,
    pub total_model_ms: f32,
}

const FISH_S2_SEMANTIC_SAMPLER_SEED: u64 = 0;
const FISH_S2_FAST_SAMPLER_SEED: u64 = 1;
const RAS_WIN_SIZE: usize = 10;
const RAS_HIGH_TEMP: f32 = 1.0;
const RAS_HIGH_TOP_P: f32 = 0.9;

impl FishS2TtsModel {
    pub fn load_metadata(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        if variant != ModelVariant::FishAudioS2Pro {
            return Err(Error::InvalidInput(format!(
                "Unsupported Fish S2 TTS variant: {variant}"
            )));
        }
        let config = FishS2Config::load(model_dir)?;
        let artifacts = FishS2ArtifactManifest::load(model_dir)?;
        let codec = FishS2CodecArtifact::load(model_dir)?;
        Ok(Self {
            variant,
            config,
            artifacts,
            codec,
            runtime: None,
        })
    }

    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let mut model = Self::load_metadata(model_dir, variant)?;
        model.runtime = Some(FishS2NativeRuntime::load(
            model_dir,
            &model.config,
            &model.codec,
            device,
        )?);
        Ok(model)
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn config(&self) -> &FishS2Config {
        &self.config
    }

    pub fn artifacts(&self) -> &FishS2ArtifactManifest {
        &self.artifacts
    }

    pub fn codec(&self) -> &FishS2CodecArtifact {
        &self.codec
    }

    pub fn diagnostics(&self) -> FishS2TtsDiagnostics {
        let sample_rate = self
            .runtime
            .as_ref()
            .map(|runtime| runtime.dac.config().sample_rate)
            .unwrap_or_else(|| FishS2DacConfig::current().sample_rate);
        FishS2TtsDiagnostics {
            model_family: "fish_s2_tts",
            variant: self.variant.to_string(),
            native_runtime_loaded: self.runtime.is_some(),
            sample_rate,
            num_codebooks: self.config.num_codebooks,
            codebook_size: self.config.codebook_size,
            semantic_start_token_id: self.config.semantic_start_token_id,
            semantic_end_token_id: self.config.semantic_end_token_id,
            max_seq_len: self.config.max_seq_len,
            codec_support: fish_s2_codec_support_name(self.codec.support),
        }
    }

    pub fn generate_with_reference(
        &self,
        text: &str,
        reference: FishS2Reference,
        params: FishS2GenerationParams,
    ) -> Result<FishS2GenerationOutput> {
        self.codec.ensure_native_supported()?;
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::ModelLoadError(
                "Fish Audio S2 Pro metadata was loaded without native inference modules"
                    .to_string(),
            )
        })?;
        runtime.generate_with_reference(&self.config, text, reference, params)
    }
}

impl FishS2NativeRuntime {
    fn load(
        model_dir: &Path,
        config: &FishS2Config,
        codec: &FishS2CodecArtifact,
        device: DeviceProfile,
    ) -> Result<Self> {
        codec.ensure_native_supported()?;
        let tokenizer = FishS2PromptTokenizer::load(model_dir, config)?;
        let weights = FishS2Weights::load(model_dir, device, None)?;
        let slow = FishS2SlowTransformer::load(
            FishS2SlowConfig::from_config(config)?,
            weights.var_builder(),
        )?;
        let fast = FishS2FastDecoder::load(
            FishS2FastConfig::from_config(config)?,
            weights.var_builder(),
        )?;
        let codec_weights = codec.load_weights(weights.device(), DType::F32)?;
        let dac = FishS2DacDecoder::load(FishS2DacConfig::current(), codec_weights.var_builder())?;
        let semantic_allowed_mask = slow.semantic_allowed_mask(tokenizer.specials().eos)?;

        Ok(Self {
            tokenizer,
            slow,
            fast,
            dac,
            semantic_allowed_mask,
        })
    }

    fn generate_with_reference(
        &self,
        config: &FishS2Config,
        text: &str,
        reference: FishS2Reference,
        params: FishS2GenerationParams,
    ) -> Result<FishS2GenerationOutput> {
        let text = text.trim();
        let reference_text = reference.text.trim();
        if text.is_empty() {
            return Err(Error::InvalidInput(
                "Fish S2 TTS text input cannot be empty".to_string(),
            ));
        }
        if reference_text.is_empty() {
            return Err(Error::InvalidInput(
                "Fish S2 TTS reference_text cannot be empty".to_string(),
            ));
        }
        if reference.audio_samples.is_empty() {
            return Err(Error::InvalidInput(
                "Fish S2 TTS reference_audio cannot be empty".to_string(),
            ));
        }
        if params.max_frames == 0 {
            return Err(Error::InvalidInput(
                "Fish S2 TTS max_frames must be greater than zero".to_string(),
            ));
        }

        let total_started = Instant::now();
        let started = Instant::now();
        let reference_codes = self
            .dac
            .encode_reference_audio(&reference.audio_samples, reference.sample_rate)?;
        let reference_encode_ms = elapsed_ms(started);

        let started = Instant::now();
        let prompt = self.tokenizer.build_reference_voice_prompt(
            config,
            reference_text,
            reference_codes,
            text,
        )?;
        let prompt_build_ms = elapsed_ms(started);
        if prompt.prompt_length >= config.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Fish S2 prompt length {} exceeds max_seq_len {}",
                prompt.prompt_length, config.max_seq_len
            )));
        }

        let max_frames = params
            .max_frames
            .min(ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
            .min(config.max_seq_len - prompt.prompt_length);
        if max_frames == 0 {
            return Err(Error::InvalidInput(
                "Fish S2 prompt leaves no room for generated audio frames".to_string(),
            ));
        }

        let temperature = sanitize_temperature(params.temperature);
        let top_p = sanitize_top_p(params.top_p);
        let mut semantic_sampler =
            FishS2SemanticSampler::new(temperature, top_p, FISH_S2_SEMANTIC_SAMPLER_SEED);
        let mut fast_sampler = FishS2Sampler::new(temperature, top_p, FISH_S2_FAST_SAMPLER_SEED);
        let im_end_token_id = self.tokenizer.specials().eos;

        let mut slow_cache = self.slow.new_cache();
        let started = Instant::now();
        let mut slow_output = self
            .slow
            .forward_prompt(&prompt, Some(&mut slow_cache), false)?;
        let slow_prefill_ms = elapsed_ms(started);
        let mut generated_codebooks = vec![Vec::new(); config.num_codebooks];
        let mut recent_semantic_tokens = Vec::with_capacity(RAS_WIN_SIZE);
        let mut stop_reason = "max_frames".to_string();

        let started = Instant::now();
        for _ in 0..max_frames {
            let semantic_token_id = sample_semantic_token(
                &slow_output.logits,
                &self.semantic_allowed_mask,
                im_end_token_id,
                &recent_semantic_tokens,
                &mut semantic_sampler,
            )?;
            if semantic_token_id == im_end_token_id {
                stop_reason = "im_end".to_string();
                break;
            }

            let frame = self.fast.generate_frame(
                semantic_token_id,
                &slow_output.hidden_states,
                &mut fast_sampler,
            )?;
            append_generated_frame(&mut generated_codebooks, &frame)?;

            recent_semantic_tokens.push(semantic_token_id);
            if recent_semantic_tokens.len() > RAS_WIN_SIZE {
                recent_semantic_tokens.remove(0);
            }

            let frame_prompt = generated_frame_prompt(config.num_codebooks, &frame)?;
            let frame_embeds = self.slow.embed_prompt(&frame_prompt)?;
            let start_pos = slow_cache.current_len();
            slow_output =
                self.slow
                    .forward_embeds(&frame_embeds, start_pos, Some(&mut slow_cache), false)?;
        }
        let ar_decode_ms = elapsed_ms(started);

        let frames_generated = generated_codebooks.first().map(Vec::len).unwrap_or(0);
        if frames_generated == 0 {
            return Err(Error::InferenceError(
                "Fish S2 generation produced no audio frames".to_string(),
            ));
        }

        let started = Instant::now();
        let samples = self.dac.decode_vq_codes(&FishS2VqCodes {
            codebooks: generated_codebooks,
        })?;
        let dac_decode_ms = elapsed_ms(started);
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "Fish S2 DAC produced empty or non-finite audio".to_string(),
            ));
        }

        let sample_rate = self.dac.config().sample_rate;
        Ok(FishS2GenerationOutput {
            samples,
            sample_rate,
            frames_generated,
            diagnostics: FishS2TtsGenerationDiagnostics {
                model_family: "fish_s2_tts",
                sample_rate,
                prompt_tokens: prompt.prompt_length,
                max_frames,
                frames_generated,
                stop_reason,
                reference_encode_ms,
                prompt_build_ms,
                slow_prefill_ms,
                ar_decode_ms,
                dac_decode_ms,
                total_model_ms: elapsed_ms(total_started),
            },
        })
    }
}

impl Default for FishS2GenerationParams {
    fn default() -> Self {
        Self {
            max_frames: ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES,
            temperature: 0.8,
            top_p: 0.8,
        }
    }
}

struct FishS2SemanticSampler {
    temperature: f32,
    top_p: f32,
    rng: StdRng,
}

impl FishS2SemanticSampler {
    fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

fn sample_semantic_token(
    logits: &Tensor,
    allowed_mask: &[bool],
    im_end_token_id: u32,
    previous_semantic_tokens: &[u32],
    sampler: &mut FishS2SemanticSampler,
) -> Result<u32> {
    let row = last_logits_row(logits)?;
    let values = row.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if values.len() != allowed_mask.len() {
        return Err(Error::InferenceError(format!(
            "Fish S2 semantic mask length {} does not match logits length {}",
            allowed_mask.len(),
            values.len()
        )));
    }

    let mut sampled = sample_masked_values(
        &values,
        allowed_mask,
        sampler.temperature,
        sampler.top_p,
        &mut sampler.rng,
    )?;
    if sampled != im_end_token_id
        && previous_semantic_tokens.contains(&sampled)
        && allowed_mask.get(sampled as usize).copied().unwrap_or(false)
    {
        sampled = sample_masked_values(
            &values,
            allowed_mask,
            RAS_HIGH_TEMP,
            RAS_HIGH_TOP_P,
            &mut sampler.rng,
        )?;
    }
    Ok(sampled)
}

fn last_logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        3 => {
            let seq_len = logits.dim(1)?;
            logits.i((0, seq_len - 1)).map_err(Error::from)
        }
        2 => logits.i(0).map_err(Error::from),
        1 => Ok(logits.clone()),
        _ => Err(Error::InferenceError(format!(
            "Fish S2 expected semantic logits rank 1, 2, or 3, got {:?}",
            logits.dims()
        ))),
    }
}

fn sample_masked_values(
    values: &[f32],
    allowed_mask: &[bool],
    temperature: f32,
    top_p: f32,
    rng: &mut StdRng,
) -> Result<u32> {
    if values.is_empty() {
        return Err(Error::InferenceError(
            "Fish S2 semantic sampler received empty logits".to_string(),
        ));
    }
    if temperature <= 1e-5 || top_p <= 0.0 {
        return argmax_masked_values(values, allowed_mask);
    }

    let temp = temperature.max(1e-5);
    let max = values
        .iter()
        .zip(allowed_mask)
        .filter_map(|(value, allowed)| {
            if *allowed && value.is_finite() {
                Some(*value / temp)
            } else {
                None
            }
        })
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax_masked_values(values, allowed_mask);
    }

    let mut probs = values
        .iter()
        .zip(allowed_mask)
        .enumerate()
        .filter_map(|(idx, (value, allowed))| {
            if *allowed && value.is_finite() {
                Some((idx, ((*value / temp) - max).exp()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_masked_values(values, allowed_mask);
    }
    for (_, prob) in &mut probs {
        *prob /= sum;
    }

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0f32;
    let mut kept = Vec::new();
    for item in probs {
        cumulative += item.1;
        kept.push(item);
        if cumulative >= top_p.max(1e-6) {
            break;
        }
    }
    let kept_sum: f32 = kept.iter().map(|(_, prob)| *prob).sum();
    if !kept_sum.is_finite() || kept_sum <= 0.0 {
        return argmax_masked_values(values, allowed_mask);
    }

    let mut draw = rng.r#gen::<f32>() * kept_sum;
    for (idx, prob) in kept {
        if draw <= prob {
            return u32::try_from(idx).map_err(|_| {
                Error::InferenceError("Fish S2 sampled semantic index overflowed".to_string())
            });
        }
        draw -= prob;
    }
    argmax_masked_values(values, allowed_mask)
}

fn argmax_masked_values(values: &[f32], allowed_mask: &[bool]) -> Result<u32> {
    let mut best_idx = None;
    let mut best = f32::NEG_INFINITY;
    for (idx, (value, allowed)) in values.iter().zip(allowed_mask).enumerate() {
        if *allowed && value.is_finite() && *value > best {
            best = *value;
            best_idx = Some(idx);
        }
    }
    let best_idx = best_idx.ok_or_else(|| {
        Error::InferenceError("Fish S2 semantic sampler had no allowed finite logits".to_string())
    })?;
    u32::try_from(best_idx)
        .map_err(|_| Error::InferenceError("Fish S2 argmax semantic index overflowed".to_string()))
}

fn append_generated_frame(
    generated_codebooks: &mut [Vec<u32>],
    frame: &FishS2GeneratedFrame,
) -> Result<()> {
    if frame.codebooks.len() != generated_codebooks.len() {
        return Err(Error::InferenceError(format!(
            "Fish S2 fast decoder returned {} codebooks, expected {}",
            frame.codebooks.len(),
            generated_codebooks.len()
        )));
    }
    for (row, code) in generated_codebooks.iter_mut().zip(&frame.codebooks) {
        row.push(*code);
    }
    Ok(())
}

fn generated_frame_prompt(
    num_codebooks: usize,
    frame: &FishS2GeneratedFrame,
) -> Result<FishS2ConditioningPrompt> {
    if frame.codebooks.len() != num_codebooks {
        return Err(Error::InferenceError(format!(
            "Fish S2 frame has {} codebooks, expected {num_codebooks}",
            frame.codebooks.len()
        )));
    }
    let mut values = vec![vec![0u32; 1]; num_codebooks + 1];
    values[0][0] = frame.semantic_token_id;
    for (idx, code) in frame.codebooks.iter().enumerate() {
        values[idx + 1][0] = *code;
    }
    Ok(FishS2ConditioningPrompt {
        values,
        vq_mask: vec![true],
        prompt_length: 1,
    })
}

fn sanitize_temperature(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        FishS2GenerationParams::default().temperature
    }
}

fn sanitize_top_p(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        FishS2GenerationParams::default().top_p
    }
}

fn elapsed_ms(started: Instant) -> f32 {
    started.elapsed().as_secs_f32() * 1000.0
}

fn fish_s2_codec_support_name(support: FishS2CodecSupport) -> &'static str {
    match support {
        FishS2CodecSupport::NativePthStateDict => "native_pth_state_dict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn generation_params_default_to_s2_frame_budget() {
        let params = FishS2GenerationParams::default();
        assert_eq!(
            params.max_frames,
            ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES
        );
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.8);
    }

    #[test]
    fn semantic_sampler_obeys_allowed_mask_for_argmax() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![100.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, false, true];
        let mut sampler = FishS2SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, &[], &mut sampler).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn generated_frame_prompt_has_upstream_codebook_shape() {
        let frame = FishS2GeneratedFrame {
            semantic_token_id: 42,
            codebooks: vec![4, 5, 6],
        };
        let prompt = generated_frame_prompt(3, &frame).unwrap();
        assert_eq!(prompt.prompt_length, 1);
        assert_eq!(prompt.vq_mask, vec![true]);
        assert_eq!(prompt.values, vec![vec![42], vec![4], vec![5], vec![6]]);
    }
}
