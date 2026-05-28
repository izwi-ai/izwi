//! Native VibeVoice-1.5B TTS model path.

use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};
use tracing::{debug, info};

use crate::backends::DeviceProfile;
use crate::catalog::{ModelFamily, ModelVariant};
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::{Qwen3Cache, Qwen3Model, Qwen3WeightLayout};
use crate::models::architectures::vibevoice::config::VibeVoiceConfig;
use crate::models::architectures::vibevoice::connector::SpeechConnector;
use crate::models::architectures::vibevoice::diffusion::{
    VibeVoiceDiffusionHead, VibeVoiceDiffusionScheduler,
};
use crate::models::architectures::vibevoice::prompt::VibeVoicePromptTokenizer;
use crate::models::architectures::vibevoice::tokenizer::VibeVoiceAcousticTokenizer;
use crate::models::shared::weights::gguf::load_model_weights;

const TARGET_SAMPLE_RATE: u32 = 24_000;
const SPEECH_TOKEN_COMPRESS_RATIO: usize = 3_200;
const MIN_FRAMES_BEFORE_STOP: usize = 4;
const AUTO_MIN_OUTPUT_FRAMES: usize = 8;
const AUTO_MAX_OUTPUT_FRAMES: usize = 384;
const WORDS_PER_SECOND: f32 = 2.6;
const AUTO_PADDING_SECONDS: f32 = 0.8;
const DEFAULT_CFG_SCALE: f32 = 1.5;

#[derive(Debug, Clone)]
pub struct VibeVoiceSpeakerReference {
    pub audio_samples: Vec<f32>,
    pub sample_rate: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceTtsGenerationParams {
    pub cfg_scale: f32,
    pub diffusion_steps: usize,
    pub max_frames: usize,
    pub auto_frame_budget: bool,
}

impl VibeVoiceTtsGenerationParams {
    pub fn from_generation_config_for_text(
        config: &crate::runtime::GenerationConfig,
        text: &str,
        default_diffusion_steps: usize,
    ) -> Self {
        let opts = &config.options;
        let auto_frame_budget = opts.max_tokens == 0;
        let max_frames = if auto_frame_budget {
            vibevoice_tts_auto_max_frames_for_text(text)
        } else {
            opts.max_tokens
                .clamp(1, ModelVariant::VIBEVOICE_TTS_MAX_OUTPUT_FRAMES)
        };
        Self {
            cfg_scale: DEFAULT_CFG_SCALE,
            diffusion_steps: default_diffusion_steps.max(1),
            max_frames,
            auto_frame_budget,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VibeVoiceTtsOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames_generated: usize,
}

struct EncodedReference {
    scaled_latents: Tensor,
    scaling_factor: f32,
    bias_factor: f32,
}

pub struct VibeVoiceTtsModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: VibeVoiceConfig,
    tokenizer: VibeVoicePromptTokenizer,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    acoustic_connector: SpeechConnector,
    language_model: Qwen3Model,
    prediction_head: VibeVoiceDiffusionHead,
}

impl VibeVoiceTtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::VibeVoice15BTts {
            return Err(Error::InvalidInput(format!(
                "VibeVoiceTtsModel cannot load non-TTS variant {variant}"
            )));
        }
        let config = VibeVoiceConfig::load(model_dir)?;
        if !config.is_tts() {
            return Err(Error::ModelLoadError(
                "VibeVoice TTS loader received a non-TTS config".to_string(),
            ));
        }
        let diffusion_config = config.diffusion_head_config.clone().ok_or_else(|| {
            Error::ModelLoadError("VibeVoice TTS config missing diffusion head".to_string())
        })?;
        let dtype = std::env::var("IZWI_VIBEVOICE_TTS_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .map(|raw| {
                device.select_model_dtype_checked(
                    ModelFamily::VibeVoiceTts,
                    Some(raw),
                    "VibeVoice TTS",
                )
            })
            .transpose()?
            .unwrap_or_else(|| device.select_model_dtype(ModelFamily::VibeVoiceTts, None));
        let vb = load_model_weights(model_dir, dtype, &device.device)?;
        let tokenizer =
            VibeVoicePromptTokenizer::load(model_dir, config.decoder_config.vocab_size)?;
        let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load(
            &config.acoustic_tokenizer_config,
            vb.pp("model.acoustic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.acoustic_connector"),
        )?;
        let prediction_head =
            VibeVoiceDiffusionHead::load(diffusion_config, vb.pp("model.prediction_head"))?;
        let language_model = Qwen3Model::load_with_layout(
            config.decoder_config.clone(),
            vb,
            Qwen3WeightLayout::VIBEVOICE,
        )?;
        let projection_diagnostics = language_model.projection_diagnostics();
        info!(
            "Loaded VibeVoice-1.5B TTS from {:?} on {:?} with dtype {:?} (dense_projections={}, dense_bias_projections={}, quantized_projections={})",
            model_dir,
            device.kind,
            dtype,
            projection_diagnostics.dense_projection_count,
            projection_diagnostics.dense_bias_projection_count,
            projection_diagnostics.quantized_projection_count
        );
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            tokenizer,
            acoustic_tokenizer,
            acoustic_connector,
            language_model,
            prediction_head,
        })
    }

    pub fn default_diffusion_steps(&self) -> usize {
        self.prediction_head
            .config()
            .ddpm_num_inference_steps
            .max(1)
    }

    pub fn available_speakers(&self) -> Vec<String> {
        Vec::new()
    }

    pub fn generate_with_reference(
        &self,
        text: &str,
        reference: &VibeVoiceSpeakerReference,
        speaker: Option<&str>,
        params: VibeVoiceTtsGenerationParams,
    ) -> Result<VibeVoiceTtsOutput> {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice TTS text input cannot be empty".to_string(),
            ));
        }
        if reference.text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice TTS reference_text cannot be empty".to_string(),
            ));
        }
        if reference.audio_samples.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice TTS reference_audio cannot be empty".to_string(),
            ));
        }

        let reference_text = reference.text.trim().to_string();
        let reference = self.encode_reference(reference)?;
        let prompt = self.tokenizer.build_tts_prompt(
            text.trim(),
            speaker.unwrap_or("Speaker 0"),
            Some(&reference_text),
            reference.scaled_latents.dim(1)?,
        )?;
        let input_ids = Tensor::from_vec(
            prompt.input_ids.clone(),
            (1, prompt.input_ids.len()),
            &self.device.device,
        )?;
        let input_embeds = self.language_model.embeddings(&input_ids)?;
        let input_embeds = if let Some(range) = prompt.reference_voice_range.clone() {
            let reference_embeds = self
                .acoustic_connector
                .forward(&reference.scaled_latents.to_dtype(input_embeds.dtype())?)?;
            replace_range_with_features(&input_embeds, range, &reference_embeds)?
        } else {
            input_embeds
        };

        let mut cache = Qwen3Cache::new(self.language_model.num_layers());
        let prefill_hidden = self.language_model.forward_hidden_with_embeds(
            &input_embeds,
            0,
            Some(&mut cache),
            None,
        )?;
        let mut pos = prompt.input_ids.len();
        let mut last_hidden = last_sequence_hidden(&prefill_hidden, "VibeVoice TTS prefill")?;

        let mut negative_cache = Qwen3Cache::new(self.language_model.num_layers());
        let negative_id = self.tokenizer.specials().image_pad;
        let negative_ids = Tensor::from_vec(vec![negative_id], (1, 1), &self.device.device)?;
        let negative_hidden = self.language_model.forward_hidden_with_embeds(
            &self.language_model.embeddings(&negative_ids)?,
            0,
            Some(&mut negative_cache),
            None,
        )?;
        let mut negative_pos = 1usize;
        let mut negative_last_hidden =
            last_sequence_hidden(&negative_hidden, "VibeVoice TTS negative prefill")?;

        let max_frames = params.max_frames.max(1);
        let scheduler = VibeVoiceDiffusionScheduler::new(
            self.prediction_head.config().ddpm_num_steps,
            params.diffusion_steps,
        );
        let mut scaled_latents = Vec::with_capacity(max_frames);
        for frame_idx in 0..max_frames {
            if frame_idx >= MIN_FRAMES_BEFORE_STOP {
                let predicted_id = argmax_from_hidden(&self.language_model, &last_hidden)?;
                if predicted_id == self.tokenizer.specials().speech_end
                    || predicted_id == self.tokenizer.specials().endoftext
                {
                    debug!(
                        "VibeVoice TTS stopped after {frame_idx} frames on token {predicted_id}"
                    );
                    break;
                }
            }

            let latent = self.sample_speech_latent(
                &last_hidden,
                Some(&negative_last_hidden),
                params.cfg_scale,
                &scheduler,
            )?;
            let latent_frame = latent.unsqueeze(1)?;
            let acoustic_embed = self.acoustic_connector.forward(&latent_frame)?;
            scaled_latents.push(latent_frame);

            let hidden = self.language_model.forward_hidden_with_embeds(
                &acoustic_embed,
                pos,
                Some(&mut cache),
                None,
            )?;
            pos += 1;
            last_hidden = last_sequence_hidden(&hidden, "VibeVoice TTS generated frame")?;

            let negative_hidden = self.language_model.forward_hidden_with_embeds(
                &acoustic_embed,
                negative_pos,
                Some(&mut negative_cache),
                None,
            )?;
            negative_pos += 1;
            negative_last_hidden =
                last_sequence_hidden(&negative_hidden, "VibeVoice TTS negative frame")?;
        }

        if scaled_latents.is_empty() {
            return Err(Error::InferenceError(
                "VibeVoice TTS generated no acoustic frames".to_string(),
            ));
        }
        if params.auto_frame_budget && scaled_latents.len() >= max_frames {
            tracing::warn!(
                "VibeVoice TTS reached auto frame budget of {max_frames} frames before EOS"
            );
        }

        let scaled_latents = Tensor::cat(&scaled_latents, 1)?;
        let unscaled = unscale_latents(
            &scaled_latents,
            reference.bias_factor,
            reference.scaling_factor,
        )?;
        let audio = self.acoustic_tokenizer.decode(&unscaled)?;
        let samples = audio
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(VibeVoiceTtsOutput {
            samples,
            sample_rate: TARGET_SAMPLE_RATE,
            frames_generated: scaled_latents.dim(1)?,
        })
    }

    fn encode_reference(&self, reference: &VibeVoiceSpeakerReference) -> Result<EncodedReference> {
        let cleaned = crate::runtime::audio_io::preprocess_reference_audio(
            reference.audio_samples.clone(),
            reference.sample_rate,
        );
        if cleaned.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice TTS reference_audio contains no usable speech".to_string(),
            ));
        }
        let resampled = resample_linear(&cleaned, reference.sample_rate, TARGET_SAMPLE_RATE)?;
        let speech = Tensor::from_vec(
            resampled.clone(),
            (1, 1, resampled.len()),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        let acoustic = self.acoustic_tokenizer.encode(&speech)?;
        let latents = self.acoustic_tokenizer.sample(&acoustic)?;
        let (bias_factor, scaling_factor) = latent_normalization(&latents)?;
        let scaled_latents = scale_latents(&latents, bias_factor, scaling_factor)?;
        Ok(EncodedReference {
            scaled_latents,
            scaling_factor,
            bias_factor,
        })
    }

    fn sample_speech_latent(
        &self,
        condition: &Tensor,
        negative_condition: Option<&Tensor>,
        cfg_scale: f32,
        scheduler: &VibeVoiceDiffusionScheduler,
    ) -> Result<Tensor> {
        let mut speech = Tensor::randn(
            0f32,
            1f32,
            (1, self.config.acoustic_vae_dim()),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        let timesteps = scheduler.timesteps();
        for (idx, &timestep) in timesteps.iter().enumerate() {
            let timestep_tensor =
                Tensor::from_vec(vec![timestep as f32], (1,), &self.device.device)?
                    .to_dtype(self.dtype)?;
            let mut model_output =
                self.prediction_head
                    .forward(&speech, &timestep_tensor, condition)?;
            if let Some(negative_condition) = negative_condition.filter(|_| cfg_scale > 1.0) {
                let negative_output =
                    self.prediction_head
                        .forward(&speech, &timestep_tensor, negative_condition)?;
                let guidance = model_output.broadcast_sub(&negative_output)?;
                let cfg = scalar_like(cfg_scale, &speech)?;
                model_output = negative_output.broadcast_add(&guidance.broadcast_mul(&cfg)?)?;
            }
            let prev_timestep = timesteps.get(idx + 1).copied();
            speech =
                scheduler.step_v_prediction(&model_output, timestep, prev_timestep, &speech)?;
        }
        Ok(speech)
    }
}

pub fn vibevoice_tts_auto_max_frames_for_text(text: &str) -> usize {
    let word_count = text
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .count();
    let char_count = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let effective_words = if word_count > 0 {
        word_count as f32
    } else if char_count > 0 {
        (char_count as f32 / 4.0).ceil()
    } else {
        1.0
    };
    let estimated_secs = AUTO_PADDING_SECONDS + effective_words / WORDS_PER_SECOND;
    let estimated_frames =
        (estimated_secs * ModelVariant::VIBEVOICE_TTS_FRAME_RATE_HZ).ceil() as usize;
    estimated_frames.clamp(AUTO_MIN_OUTPUT_FRAMES, AUTO_MAX_OUTPUT_FRAMES)
}

fn replace_range_with_features(
    embeds: &Tensor,
    range: std::ops::Range<usize>,
    features: &Tensor,
) -> Result<Tensor> {
    let seq_len = embeds.dim(1)?;
    let feature_len = features.dim(1)?;
    if feature_len != range.end.saturating_sub(range.start) {
        return Err(Error::InferenceError(format!(
            "VibeVoice TTS prompt reserved {} reference tokens but encoder produced {feature_len}",
            range.end.saturating_sub(range.start)
        )));
    }
    let mut parts = Vec::new();
    if range.start > 0 {
        parts.push(embeds.narrow(1, 0, range.start)?);
    }
    parts.push(features.clone());
    if range.end < seq_len {
        parts.push(embeds.narrow(1, range.end, seq_len - range.end)?);
    }
    Tensor::cat(&parts, 1).map_err(Error::from)
}

fn last_sequence_hidden(hidden: &Tensor, context: &str) -> Result<Tensor> {
    let seq_len = hidden.dim(1)?;
    if seq_len == 0 {
        return Err(Error::InferenceError(format!(
            "{context} returned empty hidden state"
        )));
    }
    hidden
        .i((0, seq_len - 1, ..))?
        .unsqueeze(0)
        .map_err(Error::from)
}

fn argmax_from_hidden(language_model: &Qwen3Model, hidden: &Tensor) -> Result<u32> {
    let logits = language_model.logits_from_hidden(&hidden.unsqueeze(1)?)?;
    let row = logits.i((0, 0))?.to_dtype(DType::F32)?;
    let values = row.to_vec1::<f32>()?;
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| Error::InferenceError("VibeVoice TTS logits row was empty".to_string()))
}

fn latent_normalization(latents: &Tensor) -> Result<(f32, f32)> {
    let values = latents
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.is_empty() {
        return Err(Error::InferenceError(
            "VibeVoice TTS reference encoder produced no latents".to_string(),
        ));
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let std = variance.sqrt().max(1e-5);
    Ok((-mean, 1.0 / std))
}

fn scale_latents(latents: &Tensor, bias_factor: f32, scaling_factor: f32) -> Result<Tensor> {
    latents
        .broadcast_add(&scalar_like(bias_factor, latents)?)?
        .broadcast_mul(&scalar_like(scaling_factor, latents)?)
        .map_err(Error::from)
}

fn unscale_latents(latents: &Tensor, bias_factor: f32, scaling_factor: f32) -> Result<Tensor> {
    latents
        .broadcast_div(&scalar_like(scaling_factor, latents)?)?
        .broadcast_sub(&scalar_like(bias_factor, latents)?)
        .map_err(Error::from)
}

fn scalar_like(value: f32, like: &Tensor) -> Result<Tensor> {
    Tensor::new(value, like.device())?
        .to_dtype(like.dtype())
        .map_err(Error::from)
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Sample rates must be positive for VibeVoice TTS resampling".to_string(),
        ));
    }
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_budget_scales_with_text_length() {
        let short = vibevoice_tts_auto_max_frames_for_text("hello world");
        let long = vibevoice_tts_auto_max_frames_for_text(&"hello ".repeat(80));

        assert!(short >= AUTO_MIN_OUTPUT_FRAMES);
        assert!(long > short);
        assert!(long <= AUTO_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn generation_config_zero_tokens_enables_auto_budget() {
        let config = crate::runtime::GenerationConfig::default();
        let params =
            VibeVoiceTtsGenerationParams::from_generation_config_for_text(&config, "hello", 20);

        assert!(params.auto_frame_budget);
        assert_eq!(params.diffusion_steps, 20);
        assert_eq!(params.cfg_scale, DEFAULT_CFG_SCALE);
    }

    #[test]
    fn latent_scaling_round_trips() {
        let device = candle_core::Device::Cpu;
        let latents = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let (bias, scale) = latent_normalization(&latents).unwrap();
        let scaled = scale_latents(&latents, bias, scale).unwrap();
        let unscaled = unscale_latents(&scaled, bias, scale).unwrap();
        let values = unscaled.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (actual, expected) in values.iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn resample_linear_preserves_identity_rate() {
        let audio = vec![0.0, 0.5, -0.25];
        assert_eq!(resample_linear(&audio, 24_000, 24_000).unwrap(), audio);
    }

    #[test]
    fn prompt_compress_ratio_matches_model_card_contract() {
        assert_eq!(SPEECH_TOKEN_COMPRESS_RATIO, 3_200);
    }
}
