//! Native VibeVoice-1.5B TTS model path.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Serialize;
use tracing::{debug, info, warn};

use crate::backends::{DeviceKind, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant};
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::{
    Qwen3Cache, Qwen3Model, Qwen3WeightLayout, qwen3_dense_decode_max_tokens,
};
use crate::models::architectures::vibevoice::config::{
    VibeVoiceConfig, VibeVoicePreprocessorConfig,
};
use crate::models::architectures::vibevoice::connector::SpeechConnector;
use crate::models::architectures::vibevoice::diffusion::{
    VibeVoiceDiffusionHead, VibeVoiceDiffusionScheduler, VibeVoiceDiffusionStepTensors,
};
use crate::models::architectures::vibevoice::prompt::{
    VibeVoicePromptTokenizer, VibeVoiceSpecialTokens,
};
use crate::models::architectures::vibevoice::tokenizer::{
    VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer, VibeVoiceTokenizerStreamingCache,
};
use crate::models::shared::attention::paged::{KvCacheQuantization, default_kv_page_size};
use crate::models::shared::weights::gguf::load_model_weights;

const TARGET_SAMPLE_RATE: u32 = 24_000;
const SPEECH_TOKEN_COMPRESS_RATIO: usize = 3_200;
const MIN_FRAMES_BEFORE_STOP: usize = 4;
const AUTO_MIN_OUTPUT_FRAMES: usize = 8;
const AUTO_MAX_OUTPUT_FRAMES: usize = 384;
const WORDS_PER_SECOND: f32 = 2.6;
const AUTO_PADDING_SECONDS: f32 = 0.8;
const DEFAULT_CFG_SCALE: f32 = 1.5;
const VIBEVOICE_CFG_BATCHING_ENV: &str = "IZWI_VIBEVOICE_CFG_BATCHING";

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
    pub profile: VibeVoiceTtsProfile,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VibeVoiceTtsProfile {
    pub reference_encode_ms: f32,
    pub prompt_embed_ms: f32,
    pub positive_prefill_ms: f32,
    pub negative_prefill_ms: f32,
    pub control_score_ms: f32,
    pub diffusion_sample_ms: f32,
    pub feedback_acoustic_decode_ms: f32,
    pub feedback_semantic_encode_ms: f32,
    pub feedback_connector_ms: f32,
    pub positive_decode_ms: f32,
    pub negative_decode_ms: f32,
    pub final_decode_ms: f32,
    pub host_audio_ms: f32,
    pub frames_generated: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct VibeVoiceTtsDiagnostics {
    pub model_family: &'static str,
    pub device_kind: String,
    pub dtype: String,
    pub sample_rate: u32,
    pub latent_normalization: &'static str,
    pub dense_decode_supported: bool,
    pub dense_projection_count: usize,
    pub dense_bias_projection_count: usize,
    pub quantized_projection_count: usize,
    pub cfg_batching_enabled: bool,
}

struct EncodedReference {
    scaled_latents: Tensor,
    normalization: LatentNormalization,
}

struct GeneratedSpeechFeedback {
    embed: Tensor,
    acoustic_decode_ms: f32,
    semantic_encode_ms: f32,
    connector_ms: f32,
}

struct VibeVoiceDiffusionPlan {
    scheduler: VibeVoiceDiffusionScheduler,
    steps: Vec<VibeVoiceDiffusionStepTensors>,
    cfg_tensor: Option<Tensor>,
    batch_cfg_prediction: bool,
}

struct LatentNormalization {
    bias: Tensor,
    scale: Tensor,
    source: LatentNormalizationSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatentNormalizationSource {
    Checkpoint,
    ReferenceStatistics,
}

struct CheckpointLatentNormalization {
    bias: Tensor,
    scale: Tensor,
}

pub struct VibeVoiceTtsModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: VibeVoiceConfig,
    preprocessor: VibeVoicePreprocessorConfig,
    checkpoint_latent_normalization: Option<CheckpointLatentNormalization>,
    tokenizer: VibeVoicePromptTokenizer,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    semantic_tokenizer: VibeVoiceSemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
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
        let preprocessor = VibeVoicePreprocessorConfig::load(model_dir)?;
        if preprocessor.speech_tok_compress_ratio != config.acoustic_tokenizer_config.hop_length() {
            warn!(
                "VibeVoice preprocessor speech_tok_compress_ratio={} differs from acoustic tokenizer hop_length={}",
                preprocessor.speech_tok_compress_ratio,
                config.acoustic_tokenizer_config.hop_length()
            );
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
        let semantic_tokenizer = VibeVoiceSemanticTokenizer::load(
            &config.semantic_tokenizer_config,
            vb.pp("model.semantic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("model.semantic_connector"),
        )?;
        let prediction_head =
            VibeVoiceDiffusionHead::load(diffusion_config, vb.pp("model.prediction_head"))?;
        let checkpoint_latent_normalization =
            load_checkpoint_latent_normalization(vb.pp("model"), config.acoustic_vae_dim())?;
        let language_model = Qwen3Model::load_with_layout(
            config.decoder_config.clone(),
            vb,
            Qwen3WeightLayout::VIBEVOICE,
        )?;
        let projection_diagnostics = language_model.projection_diagnostics();
        let latent_normalization_source = if checkpoint_latent_normalization.is_some() {
            "checkpoint"
        } else {
            "reference_statistics"
        };
        info!(
            "Loaded VibeVoice-1.5B TTS from {:?} on {:?} with dtype {:?} (sample_rate={}, latent_normalization={}, dense_projections={}, dense_bias_projections={}, quantized_projections={})",
            model_dir,
            device.kind,
            dtype,
            preprocessor.target_sample_rate(),
            latent_normalization_source,
            projection_diagnostics.dense_projection_count,
            projection_diagnostics.dense_bias_projection_count,
            projection_diagnostics.quantized_projection_count
        );
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            preprocessor,
            checkpoint_latent_normalization,
            tokenizer,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
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

    pub fn diagnostics(&self) -> VibeVoiceTtsDiagnostics {
        let projection_diagnostics = self.language_model.projection_diagnostics();
        VibeVoiceTtsDiagnostics {
            model_family: "vibevoice_tts",
            device_kind: format!("{:?}", self.device.kind),
            dtype: format!("{:?}", self.dtype),
            sample_rate: self.preprocessor.target_sample_rate(),
            latent_normalization: if self.checkpoint_latent_normalization.is_some() {
                "checkpoint"
            } else {
                "reference_statistics"
            },
            dense_decode_supported: vibevoice_dense_decode_max_tokens_for_device(
                &self.device.device,
                default_kv_page_size(),
                1,
            ) > 0,
            dense_projection_count: projection_diagnostics.dense_projection_count,
            dense_bias_projection_count: projection_diagnostics.dense_bias_projection_count,
            quantized_projection_count: projection_diagnostics.quantized_projection_count,
            cfg_batching_enabled: vibevoice_cfg_batching_enabled(self.device.kind),
        }
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

        let mut profile = VibeVoiceTtsProfile::default();
        let started = Instant::now();
        let reference = self.encode_reference(reference)?;
        profile.reference_encode_ms = elapsed_ms(started);

        let started = Instant::now();
        let prompt = self.tokenizer.build_tts_prompt(
            text.trim(),
            speaker.unwrap_or("Speaker 0"),
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
        profile.prompt_embed_ms = elapsed_ms(started);

        let max_frames = params.max_frames.max(1);
        let mut cache = self.build_decode_cache(prompt.input_ids.len().saturating_add(max_frames));
        let started = Instant::now();
        let prefill_hidden = self.language_model.forward_hidden_with_embeds(
            &input_embeds,
            0,
            Some(&mut cache),
            None,
        )?;
        profile.positive_prefill_ms = elapsed_ms(started);
        let mut pos = prompt.input_ids.len();
        let mut last_hidden = last_sequence_hidden(&prefill_hidden, "VibeVoice TTS prefill")?;

        let mut negative_cache = self.build_decode_cache(1usize.saturating_add(max_frames));
        let negative_id = vibevoice_tts_negative_prefill_token(self.tokenizer.specials());
        let negative_ids = Tensor::from_vec(vec![negative_id], (1, 1), &self.device.device)?;
        let started = Instant::now();
        let negative_hidden = self.language_model.forward_hidden_with_embeds(
            &self.language_model.embeddings(&negative_ids)?,
            0,
            Some(&mut negative_cache),
            None,
        )?;
        profile.negative_prefill_ms = elapsed_ms(started);
        let mut negative_pos = 1usize;
        let mut negative_last_hidden =
            last_sequence_hidden(&negative_hidden, "VibeVoice TTS negative prefill")?;

        let diffusion_plan = vibevoice_diffusion_plan(
            VibeVoiceDiffusionScheduler::new(
                self.prediction_head.config().ddpm_num_steps,
                params.diffusion_steps,
            ),
            &self.device.device,
            self.dtype,
            params.cfg_scale,
            self.device.kind,
        )?;
        let mut feedback_acoustic_cache = VibeVoiceTokenizerStreamingCache::new();
        let mut feedback_semantic_cache = VibeVoiceTokenizerStreamingCache::new();
        let mut scaled_latents = Vec::with_capacity(max_frames);
        for frame_idx in 0..max_frames {
            if frame_idx >= MIN_FRAMES_BEFORE_STOP {
                let started = Instant::now();
                let predicted_id = next_tts_control_token_from_hidden(
                    &self.language_model,
                    &last_hidden,
                    self.tokenizer.specials(),
                )?;
                profile.control_score_ms += elapsed_ms(started);
                if predicted_id == self.tokenizer.specials().speech_end
                    || predicted_id == self.tokenizer.specials().endoftext
                {
                    debug!(
                        "VibeVoice TTS stopped after {frame_idx} frames on token {predicted_id}"
                    );
                    break;
                }
            }

            let started = Instant::now();
            let latent = self.sample_speech_latent(
                &last_hidden,
                Some(&negative_last_hidden),
                &diffusion_plan,
            )?;
            profile.diffusion_sample_ms += elapsed_ms(started);
            let latent_frame = latent.unsqueeze(1)?;
            let feedback = self.generated_speech_embed(
                &latent_frame,
                &reference.normalization,
                &mut feedback_acoustic_cache,
                &mut feedback_semantic_cache,
            )?;
            profile.feedback_acoustic_decode_ms += feedback.acoustic_decode_ms;
            profile.feedback_semantic_encode_ms += feedback.semantic_encode_ms;
            profile.feedback_connector_ms += feedback.connector_ms;
            scaled_latents.push(latent_frame);

            let started = Instant::now();
            let hidden = self.language_model.forward_hidden_with_embeds(
                &feedback.embed,
                pos,
                Some(&mut cache),
                None,
            )?;
            profile.positive_decode_ms += elapsed_ms(started);
            pos += 1;
            last_hidden = last_sequence_hidden(&hidden, "VibeVoice TTS generated frame")?;

            let started = Instant::now();
            let negative_hidden = self.language_model.forward_hidden_with_embeds(
                &feedback.embed,
                negative_pos,
                Some(&mut negative_cache),
                None,
            )?;
            profile.negative_decode_ms += elapsed_ms(started);
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

        profile.frames_generated = scaled_latents.len();
        let scaled_latents = Tensor::cat(&scaled_latents, 1)?;
        let unscaled = unscale_latents(
            &scaled_latents,
            &reference.normalization.bias,
            &reference.normalization.scale,
        )?;
        let started = Instant::now();
        let audio = self.acoustic_tokenizer.decode(&unscaled)?;
        profile.final_decode_ms = elapsed_ms(started);
        let started = Instant::now();
        let samples = audio
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        profile.host_audio_ms = elapsed_ms(started);
        info!(?profile, "VibeVoice TTS generation profile");
        Ok(VibeVoiceTtsOutput {
            samples,
            sample_rate: self.preprocessor.target_sample_rate(),
            frames_generated: scaled_latents.dim(1)?,
            profile,
        })
    }

    fn encode_reference(&self, reference: &VibeVoiceSpeakerReference) -> Result<EncodedReference> {
        let cleaned = preprocess_reference_audio(
            reference.audio_samples.clone(),
            reference.sample_rate,
            &self.preprocessor,
        );
        if cleaned.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice TTS reference_audio contains no usable speech".to_string(),
            ));
        }
        let target_sample_rate = self.preprocessor.target_sample_rate();
        let resampled = resample_linear(&cleaned, reference.sample_rate, target_sample_rate)?;
        let speech = Tensor::from_vec(
            resampled.clone(),
            (1, 1, resampled.len()),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        let acoustic = self.acoustic_tokenizer.encode(&speech)?;
        let latents = acoustic.mode();
        let normalization = self.latent_normalization(&latents)?;
        let scaled_latents = scale_latents(&latents, &normalization.bias, &normalization.scale)?;
        debug!(
            "VibeVoice reference encoded with {:?} latent normalization",
            normalization.source
        );
        Ok(EncodedReference {
            scaled_latents,
            normalization,
        })
    }

    fn latent_normalization(&self, latents: &Tensor) -> Result<LatentNormalization> {
        if let Some(checkpoint) = &self.checkpoint_latent_normalization {
            return Ok(LatentNormalization {
                bias: factor_like(&checkpoint.bias, latents)?,
                scale: factor_like(&checkpoint.scale, latents)?,
                source: LatentNormalizationSource::Checkpoint,
            });
        }
        reference_latent_normalization(latents)
    }

    fn generated_speech_embed(
        &self,
        scaled_latent_frame: &Tensor,
        normalization: &LatentNormalization,
        acoustic_cache: &mut VibeVoiceTokenizerStreamingCache,
        semantic_cache: &mut VibeVoiceTokenizerStreamingCache,
    ) -> Result<GeneratedSpeechFeedback> {
        let started = Instant::now();
        let acoustic_embed = self.acoustic_connector.forward(scaled_latent_frame)?;
        let connector_acoustic_ms = elapsed_ms(started);
        let unscaled_frame = unscale_latents(
            scaled_latent_frame,
            &normalization.bias,
            &normalization.scale,
        )?;
        let started = Instant::now();
        let audio_chunk = self
            .acoustic_tokenizer
            .decode_streaming(&unscaled_frame, acoustic_cache)?;
        let acoustic_decode_ms = elapsed_ms(started);
        let started = Instant::now();
        let semantic = self
            .semantic_tokenizer
            .encode_streaming(&audio_chunk, semantic_cache)?
            .mode();
        let semantic_encode_ms = elapsed_ms(started);
        let started = Instant::now();
        let semantic_embed = self.semantic_connector.forward(&semantic)?;
        let embed = combine_speech_embeddings(
            &acoustic_embed,
            &semantic_embed,
            "VibeVoice TTS generated frame",
        )?;
        Ok(GeneratedSpeechFeedback {
            embed,
            acoustic_decode_ms,
            semantic_encode_ms,
            connector_ms: connector_acoustic_ms + elapsed_ms(started),
        })
    }

    fn build_decode_cache(&self, max_tokens: usize) -> Qwen3Cache {
        let page_size = default_kv_page_size();
        Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            self.language_model.num_layers(),
            page_size,
            KvCacheQuantization::None,
            vibevoice_dense_decode_max_tokens_for_device(
                &self.device.device,
                page_size,
                max_tokens,
            ),
        )
    }

    fn sample_speech_latent(
        &self,
        condition: &Tensor,
        negative_condition: Option<&Tensor>,
        plan: &VibeVoiceDiffusionPlan,
    ) -> Result<Tensor> {
        let mut speech = Tensor::randn(
            0f32,
            1f32,
            (1, self.config.acoustic_vae_dim()),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        for step in &plan.steps {
            let model_output = if let (Some(negative_condition), Some(cfg)) =
                (negative_condition, plan.cfg_tensor.as_ref())
            {
                if plan.batch_cfg_prediction {
                    self.prediction_head.forward_cfg_batched(
                        &speech,
                        &step.timestep_tensor,
                        condition,
                        negative_condition,
                        cfg,
                    )?
                } else {
                    let positive_output =
                        self.prediction_head
                            .forward(&speech, &step.timestep_tensor, condition)?;
                    let negative_output = self.prediction_head.forward(
                        &speech,
                        &step.timestep_tensor,
                        negative_condition,
                    )?;
                    let guidance = positive_output.broadcast_sub(&negative_output)?;
                    negative_output.broadcast_add(&guidance.broadcast_mul(cfg)?)?
                }
            } else {
                self.prediction_head
                    .forward(&speech, &step.timestep_tensor, condition)?
            };
            speech = plan
                .scheduler
                .step_v_prediction_with_tensors(&model_output, &speech, step)?;
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

fn vibevoice_dense_decode_max_tokens_for_device(
    device: &Device,
    page_size: usize,
    max_tokens: usize,
) -> usize {
    let qwen_budget = qwen3_dense_decode_max_tokens(device, page_size, KvCacheQuantization::None);
    cap_vibevoice_dense_decode_tokens(max_tokens, qwen_budget)
}

fn cap_vibevoice_dense_decode_tokens(max_tokens: usize, qwen_budget: usize) -> usize {
    if qwen_budget == 0 {
        0
    } else {
        max_tokens.max(1).min(qwen_budget)
    }
}

fn vibevoice_diffusion_plan(
    scheduler: VibeVoiceDiffusionScheduler,
    device: &Device,
    dtype: DType,
    cfg_scale: f32,
    device_kind: DeviceKind,
) -> Result<VibeVoiceDiffusionPlan> {
    let steps = scheduler.step_tensors(device, dtype)?;
    Ok(VibeVoiceDiffusionPlan {
        scheduler,
        steps,
        cfg_tensor: if cfg_scale > 1.0 {
            Some(Tensor::new(cfg_scale, device)?.to_dtype(dtype)?)
        } else {
            None
        },
        batch_cfg_prediction: vibevoice_cfg_batching_enabled(device_kind),
    })
}

fn vibevoice_cfg_batching_enabled(device_kind: DeviceKind) -> bool {
    let override_value = std::env::var(VIBEVOICE_CFG_BATCHING_ENV).ok();
    vibevoice_cfg_batching_enabled_for(device_kind, override_value.as_deref())
}

fn vibevoice_cfg_batching_enabled_for(
    device_kind: DeviceKind,
    override_value: Option<&str>,
) -> bool {
    let default = vibevoice_cfg_batching_default(device_kind);
    let Some(raw) = override_value
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return default;
    };
    match raw.to_ascii_lowercase().as_str() {
        "auto" => default,
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => {
            warn!(
                "{VIBEVOICE_CFG_BATCHING_ENV}={other:?} is invalid; using auto CFG batching policy"
            );
            default
        }
    }
}

fn vibevoice_cfg_batching_default(device_kind: DeviceKind) -> bool {
    device_kind.is_metal() || device_kind.is_cuda()
}

fn vibevoice_tts_negative_prefill_token(specials: &VibeVoiceSpecialTokens) -> u32 {
    specials.speech_start
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

fn combine_speech_embeddings(
    acoustic: &Tensor,
    semantic: &Tensor,
    context: &str,
) -> Result<Tensor> {
    if acoustic.dims() != semantic.dims() {
        return Err(Error::InferenceError(format!(
            "{context} acoustic/semantic feature shape mismatch: {:?} vs {:?}",
            acoustic.dims(),
            semantic.dims()
        )));
    }
    acoustic.broadcast_add(semantic).map_err(Error::from)
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

fn next_tts_control_token_from_hidden(
    language_model: &Qwen3Model,
    hidden: &Tensor,
    specials: &VibeVoiceSpecialTokens,
) -> Result<u32> {
    let scores =
        language_model.logits_from_hidden_for_tokens(hidden, &tts_control_tokens(specials))?;
    select_next_tts_control_token(&scores)
}

fn tts_control_tokens(specials: &VibeVoiceSpecialTokens) -> [u32; 3] {
    [specials.speech_pad, specials.speech_end, specials.endoftext]
}

fn select_next_tts_control_token(token_scores: &[(u32, f32)]) -> Result<u32> {
    token_scores
        .iter()
        .copied()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token, _)| token)
        .ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS control-token scores were empty".to_string())
        })
}

fn load_checkpoint_latent_normalization(
    vb: VarBuilder,
    latent_dim: usize,
) -> Result<Option<CheckpointLatentNormalization>> {
    let has_bias = vb.contains_tensor("speech_bias_factor");
    let has_scale = vb.contains_tensor("speech_scaling_factor");
    if !has_bias && !has_scale {
        return Ok(None);
    }
    if has_bias != has_scale {
        return Err(Error::ModelLoadError(
            "VibeVoice checkpoint must contain both speech_bias_factor and speech_scaling_factor"
                .to_string(),
        ));
    }

    let bias = vb.get_unchecked_dtype("speech_bias_factor", vb.dtype())?;
    let scale = vb.get_unchecked_dtype("speech_scaling_factor", vb.dtype())?;
    validate_latent_bias_factor("speech_bias_factor", &bias, latent_dim)?;
    validate_latent_scale_factor("speech_scaling_factor", &scale, latent_dim)?;
    Ok(Some(CheckpointLatentNormalization { bias, scale }))
}

fn validate_latent_factor_shape(name: &str, factor: &Tensor, latent_dim: usize) -> Result<()> {
    let count = factor.elem_count();
    if count == 1 || count == latent_dim {
        return Ok(());
    }
    Err(Error::ModelLoadError(format!(
        "VibeVoice {name} has {} values, expected scalar or acoustic latent dim {latent_dim}",
        count
    )))
}

fn validate_latent_bias_factor(name: &str, factor: &Tensor, latent_dim: usize) -> Result<()> {
    validate_latent_factor_shape(name, factor, latent_dim)?;
    for value in latent_factor_values(name, factor)? {
        if !value.is_finite() {
            return Err(Error::ModelLoadError(format!(
                "VibeVoice {name} contains non-finite values"
            )));
        }
    }
    Ok(())
}

fn validate_latent_scale_factor(name: &str, factor: &Tensor, latent_dim: usize) -> Result<()> {
    validate_latent_factor_shape(name, factor, latent_dim)?;
    for value in latent_factor_values(name, factor)? {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::ModelLoadError(format!(
                "VibeVoice {name} must contain only finite positive values"
            )));
        }
    }
    Ok(())
}

fn latent_factor_values(name: &str, factor: &Tensor) -> Result<Vec<f32>> {
    factor
        .to_dtype(DType::F32)
        .and_then(|factor| factor.flatten_all())
        .and_then(|factor| factor.to_vec1::<f32>())
        .map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to validate VibeVoice latent factor {name}: {err}"
            ))
        })
}

fn reference_latent_normalization(latents: &Tensor) -> Result<LatentNormalization> {
    let (bias, scale) = latent_normalization_values(latents)?;
    Ok(LatentNormalization {
        bias: scalar_like(bias, latents)?,
        scale: scalar_like(scale, latents)?,
        source: LatentNormalizationSource::ReferenceStatistics,
    })
}

fn latent_normalization_values(latents: &Tensor) -> Result<(f32, f32)> {
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

fn scale_latents(latents: &Tensor, bias: &Tensor, scale: &Tensor) -> Result<Tensor> {
    // VibeVoice normalizes speech latents as `(audio_tokens + bias) * scale`.
    latents
        .broadcast_add(&factor_like(bias, latents)?)?
        .broadcast_mul(&factor_like(scale, latents)?)
        .map_err(Error::from)
}

fn unscale_latents(latents: &Tensor, bias: &Tensor, scale: &Tensor) -> Result<Tensor> {
    latents
        .broadcast_div(&factor_like(scale, latents)?)?
        .broadcast_sub(&factor_like(bias, latents)?)
        .map_err(Error::from)
}

fn factor_like(factor: &Tensor, like: &Tensor) -> Result<Tensor> {
    factor
        .to_device(like.device())?
        .to_dtype(like.dtype())
        .map_err(Error::from)
}

fn scalar_like(value: f32, like: &Tensor) -> Result<Tensor> {
    Tensor::new(value, like.device())?
        .to_dtype(like.dtype())
        .map_err(Error::from)
}

fn elapsed_ms(started: Instant) -> f32 {
    started.elapsed().as_secs_f32() * 1000.0
}

fn preprocess_reference_audio(
    mut samples: Vec<f32>,
    sample_rate: u32,
    config: &VibeVoicePreprocessorConfig,
) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }

    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    for sample in &mut samples {
        *sample -= mean;
    }

    let initial_peak = samples.iter().fold(0.0f32, |peak, &s| peak.max(s.abs()));
    if initial_peak <= config.eps.max(1e-8) {
        return Vec::new();
    }

    let silence_threshold = (initial_peak * 0.04).max(0.0025);
    let first_idx = samples.iter().position(|s| s.abs() >= silence_threshold);
    let last_idx = samples.iter().rposition(|s| s.abs() >= silence_threshold);
    if let (Some(first), Some(last)) = (first_idx, last_idx) {
        let margin = ((sample_rate as f32) * 0.12) as usize;
        let start = first.saturating_sub(margin);
        let end = (last + margin + 1).min(samples.len());
        samples = samples[start..end].to_vec();
    }

    let max_len = sample_rate as usize * 12;
    if max_len > 0 && samples.len() > max_len {
        let start = highest_energy_window_start(&samples, max_len);
        samples = samples[start..start + max_len].to_vec();
    }

    if config.normalize_audio {
        normalize_reference_loudness(&mut samples, config.target_db_fs, config.eps);
    }

    samples
}

fn normalize_reference_loudness(samples: &mut [f32], target_db_fs: f32, eps: f32) {
    if samples.is_empty() {
        return;
    }
    let rms = (samples
        .iter()
        .map(|&sample| (sample as f64) * (sample as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32;
    if rms <= eps.max(1e-8) {
        return;
    }
    let target_rms = 10f32.powf(target_db_fs / 20.0);
    let gain = target_rms / rms;
    for sample in samples.iter_mut() {
        *sample *= gain;
    }

    let peak = samples.iter().fold(0.0f32, |peak, &s| peak.max(s.abs()));
    if peak > 0.99 {
        let limit = 0.99 / peak;
        for sample in samples.iter_mut() {
            *sample *= limit;
        }
    }
}

fn highest_energy_window_start(samples: &[f32], window: usize) -> usize {
    if samples.is_empty() || window == 0 || samples.len() <= window {
        return 0;
    }

    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0f64);
    for &sample in samples {
        let energy = (sample as f64) * (sample as f64);
        let next = prefix.last().copied().unwrap_or(0.0) + energy;
        prefix.push(next);
    }

    let mut best_start = 0usize;
    let mut best_energy = f64::NEG_INFINITY;
    for start in 0..=(samples.len() - window) {
        let energy = prefix[start + window] - prefix[start];
        if energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
    }
    best_start
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
    fn decode_cache_policy_uses_qwen_gate_and_caps_requested_tokens() {
        let device = Device::Cpu;

        assert_eq!(
            vibevoice_dense_decode_max_tokens_for_device(&device, 64, 128),
            0
        );
        assert_eq!(cap_vibevoice_dense_decode_tokens(32, 128), 32);
        assert_eq!(cap_vibevoice_dense_decode_tokens(4096, 128), 128);
        assert_eq!(cap_vibevoice_dense_decode_tokens(128, 0), 0);
    }

    #[test]
    fn cfg_batching_policy_defaults_to_accelerators() {
        assert!(!vibevoice_cfg_batching_enabled_for(DeviceKind::Cpu, None));
        assert!(vibevoice_cfg_batching_enabled_for(DeviceKind::Metal, None));
        assert!(vibevoice_cfg_batching_enabled_for(DeviceKind::Cuda, None));
    }

    #[test]
    fn cfg_batching_policy_honors_overrides() {
        assert!(vibevoice_cfg_batching_enabled_for(
            DeviceKind::Cpu,
            Some("on")
        ));
        assert!(!vibevoice_cfg_batching_enabled_for(
            DeviceKind::Metal,
            Some("off")
        ));
        assert!(vibevoice_cfg_batching_enabled_for(
            DeviceKind::Cuda,
            Some("auto")
        ));
        assert!(!vibevoice_cfg_batching_enabled_for(
            DeviceKind::Cpu,
            Some("not-a-mode")
        ));
    }

    #[test]
    fn diffusion_plan_reuses_step_tensors_and_cfg_policy() {
        let device = Device::Cpu;
        let plan = vibevoice_diffusion_plan(
            VibeVoiceDiffusionScheduler::new(1000, 3),
            &device,
            DType::F32,
            1.5,
            DeviceKind::Metal,
        )
        .unwrap();

        assert_eq!(plan.steps.len(), 3);
        assert!(plan.batch_cfg_prediction);
        assert_eq!(
            plan.cfg_tensor.as_ref().unwrap().to_vec0::<f32>().unwrap(),
            1.5
        );

        let cpu_no_cfg = vibevoice_diffusion_plan(
            VibeVoiceDiffusionScheduler::new(1000, 3),
            &device,
            DType::F32,
            1.0,
            DeviceKind::Cpu,
        )
        .unwrap();
        assert!(!cpu_no_cfg.batch_cfg_prediction);
        assert!(cpu_no_cfg.cfg_tensor.is_none());
    }

    #[test]
    fn negative_cfg_prefill_uses_speech_start_like_reference_generation() {
        let specials = crate::models::architectures::vibevoice::prompt::VibeVoiceSpecialTokens {
            speech_start: 11,
            image_pad: 22,
            ..Default::default()
        };

        assert_eq!(vibevoice_tts_negative_prefill_token(&specials), 11);
    }

    #[test]
    fn tts_control_token_selection_ignores_non_speech_vocab_logits() {
        assert_eq!(
            select_next_tts_control_token(&[(3, 2.0), (2, 1.0), (1, 0.5)]).unwrap(),
            3
        );
    }

    #[test]
    fn tts_control_token_selection_can_choose_speech_end() {
        assert_eq!(
            select_next_tts_control_token(&[(3, 2.0), (2, 3.0), (1, 0.5)]).unwrap(),
            2
        );
    }

    #[test]
    fn latent_scaling_round_trips() {
        let device = candle_core::Device::Cpu;
        let latents = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let normalization = reference_latent_normalization(&latents).unwrap();
        let scaled = scale_latents(&latents, &normalization.bias, &normalization.scale).unwrap();
        let unscaled = unscale_latents(&scaled, &normalization.bias, &normalization.scale).unwrap();
        let values = unscaled.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (actual, expected) in values.iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn speech_embedding_feedback_adds_acoustic_and_semantic_features() {
        let device = candle_core::Device::Cpu;
        let acoustic = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let semantic = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, -0.5], (1, 2, 2), &device).unwrap();
        let combined = combine_speech_embeddings(&acoustic, &semantic, "test feedback").unwrap();

        assert_eq!(
            combined.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.5, 1.0, 5.0, 3.5]
        );
    }

    #[test]
    fn speech_embedding_feedback_rejects_shape_mismatch() {
        let device = candle_core::Device::Cpu;
        let acoustic = Tensor::zeros((1, 1, 2), DType::F32, &device).unwrap();
        let semantic = Tensor::zeros((1, 2, 2), DType::F32, &device).unwrap();

        let err = combine_speech_embeddings(&acoustic, &semantic, "test feedback")
            .expect_err("shape mismatch");

        assert!(format!("{err}").contains("shape mismatch"));
    }

    #[test]
    fn checkpoint_latent_scaling_matches_vibevoice_reference_formula() {
        let device = candle_core::Device::Cpu;
        let latents = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let bias = Tensor::from_vec(vec![0.5f32, -1.0], (2,), &device).unwrap();
        let scale = Tensor::from_vec(vec![2.0f32, 4.0], (2,), &device).unwrap();
        let scaled = scale_latents(&latents, &bias, &scale).unwrap();
        let scaled_values = scaled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(scaled_values, vec![3.0, 4.0, 7.0, 12.0]);

        let unscaled = unscale_latents(&scaled, &bias, &scale).unwrap();
        let values = unscaled.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (actual, expected) in values.iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn latent_factor_validation_accepts_scalar_or_latent_dim_only() {
        let device = candle_core::Device::Cpu;
        let scalar = Tensor::new(0.5f32, &device).unwrap();
        let vector = Tensor::from_vec(vec![1.0f32; 64], (64,), &device).unwrap();
        let wrong = Tensor::zeros((63,), DType::F32, &device).unwrap();

        validate_latent_bias_factor("factor", &scalar, 64).unwrap();
        validate_latent_scale_factor("factor", &vector, 64).unwrap();
        assert!(validate_latent_bias_factor("factor", &wrong, 64).is_err());
    }

    #[test]
    fn latent_factor_validation_rejects_nonfinite_bias_and_nonpositive_scale() {
        let device = candle_core::Device::Cpu;
        let nan = Tensor::new(f32::NAN, &device).unwrap();
        let zero = Tensor::new(0.0f32, &device).unwrap();
        let negative = Tensor::new(-1.0f32, &device).unwrap();

        assert!(validate_latent_bias_factor("speech_bias_factor", &nan, 64).is_err());
        assert!(validate_latent_scale_factor("speech_scaling_factor", &zero, 64).is_err());
        assert!(validate_latent_scale_factor("speech_scaling_factor", &negative, 64).is_err());
    }

    #[test]
    fn reference_preprocessing_normalizes_to_configured_loudness() {
        let config = VibeVoicePreprocessorConfig {
            target_db_fs: -20.0,
            ..VibeVoicePreprocessorConfig::default()
        };
        let processed = preprocess_reference_audio(
            (0..24_000)
                .map(|idx| if idx % 2 == 0 { 0.2 } else { -0.2 })
                .collect(),
            24_000,
            &config,
        );
        let rms = (processed
            .iter()
            .map(|&sample| (sample as f64) * (sample as f64))
            .sum::<f64>()
            / processed.len() as f64)
            .sqrt() as f32;

        assert!((rms - 0.1).abs() < 1e-4);
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
