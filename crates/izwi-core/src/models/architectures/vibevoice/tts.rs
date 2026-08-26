//! Native VibeVoice-1.5B TTS model path.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::backends::state::{PhysicalStateTransactionId, TensorStateArena};
use crate::backends::{DeviceKind, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant};
use crate::engine::{InvocationTensorLease, StageDescriptor};
use crate::error::{Error, Result};
use crate::kv::v2::{
    InvocationStateBackingKindV2, InvocationWorkspaceLeaseSetV2, StateClock, StateGroupId,
};
use crate::kv::CacheDomainId;
use crate::models::architectures::qwen3::core::{Qwen3Model, Qwen3WeightLayout};
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
    VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer,
};
use crate::models::architectures::vibevoice::{
    vibevoice_invocation_contract, vibevoice_physical_state_spec, VibeVoicePhysicalStateSpec,
    VibeVoiceTokenizerStateDomain, VIBEVOICE_TTS_ACOUSTIC_DOMAIN, VIBEVOICE_TTS_SEMANTIC_DOMAIN,
};
use crate::models::shared::attention::flash::{
    cuda_flash_attention_head_dim_supported, flash_attention_compiled, flash_attention_requested,
    should_enable_flash_attention_v2,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::telemetry::{
    snapshot as kernel_telemetry_snapshot, KernelPathTelemetrySnapshot,
};
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
const VIBEVOICE_CUDA_DDPM_STEPS_ENV: &str = "IZWI_VIBEVOICE_CUDA_DDPM_STEPS";
static NEXT_VIBEVOICE_TTS_STATE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_VIBEVOICE_TTS_MODEL_LOAD_NONCE: AtomicU64 = AtomicU64::new(1);

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
    pub diffusion_steps: usize,
    pub decode_attention_dense_calls: u64,
    pub decode_attention_paged_calls: u64,
    pub rope_kernel_calls: u64,
    pub rope_manual_calls: u64,
    pub fused_attention_attempts: u64,
    pub fused_attention_successes: u64,
    pub fused_attention_fallbacks: u64,
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
    pub dense_decode_max_tokens: usize,
    pub cuda_flash_attention_requested: bool,
    pub cuda_flash_attention_compiled: bool,
    pub cuda_flash_attention_head_dim_supported: bool,
    pub cuda_flash_attention_active: bool,
}

#[derive(Clone)]
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
    cfg_tensor: Option<Tensor>,
    batch_cfg_prediction: bool,
    cuda_prebatched_cfg: bool,
    steps: Vec<VibeVoiceDiffusionPlanStep>,
}

struct VibeVoiceDiffusionPlanStep {
    tensors: VibeVoiceDiffusionStepTensors,
    timestep_embedding: Tensor,
    cuda_batched_timestep_embedding: Option<Tensor>,
}

#[derive(Clone)]
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

#[derive(Clone)]
pub(crate) struct VibeVoiceTtsPreparedArtifact {
    model_identity: [u8; 32],
    input_ids: Arc<[u32]>,
    input_embeds: Tensor,
    negative_embed: Tensor,
    normalization: LatentNormalization,
    preparation_profile: VibeVoiceTtsProfile,
}

impl std::fmt::Debug for VibeVoiceTtsPreparedArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VibeVoiceTtsPreparedArtifact")
            .field("model_identity", &self.model_identity)
            .field("prompt_tokens", &self.input_ids.len())
            .field("retained_tensor_bytes", &self.retained_tensor_bytes().ok())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VibeVoiceTtsRetainedPrefillStep {
    pub(crate) consumed_positive_tokens: usize,
    pub(crate) positive_position: usize,
    pub(crate) negative_position: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VibeVoiceTtsRetainedDecodeStep {
    pub(crate) frames_generated: usize,
    pub(crate) finished: bool,
}

#[derive(Clone)]
pub(crate) struct VibeVoiceTtsTokenizerQuantum {
    pub(crate) arena: Arc<TensorStateArena>,
    pub(crate) transaction: PhysicalStateTransactionId,
}

pub(crate) struct VibeVoiceTtsRetainedState {
    state_id: u64,
    model_identity: [u8; 32],
    artifact: Arc<VibeVoiceTtsPreparedArtifact>,
    params: VibeVoiceTtsGenerationParams,
    positive_cache: PhysicalPagedKvCache,
    negative_cache: PhysicalPagedKvCache,
    positive_position: usize,
    negative_position: usize,
    last_hidden: Option<Tensor>,
    negative_last_hidden: Option<Tensor>,
    acoustic_clock: u64,
    semantic_clock: u64,
    scaled_latents: Vec<Tensor>,
    frame_noises: Vec<Tensor>,
    finished: bool,
    active_quantum: Option<u64>,
    next_quantum: u64,
    staged_step: Option<VibeVoiceTtsRetainedDecodeStep>,
    managed_completions_drained: bool,
}

pub(crate) struct VibeVoiceTtsRetainedCheckpoint {
    state_id: u64,
    quantum: u64,
    payload: Option<VibeVoiceTtsRetainedCheckpointPayload>,
}

struct VibeVoiceTtsRetainedCheckpointPayload {
    positive_cache: Option<PhysicalPagedKvCache>,
    negative_cache: Option<PhysicalPagedKvCache>,
    positive_position: usize,
    negative_position: usize,
    last_hidden: Option<Tensor>,
    negative_last_hidden: Option<Tensor>,
    acoustic_clock: u64,
    semantic_clock: u64,
    scaled_latents: Vec<Tensor>,
    finished: bool,
    staged_step: Option<VibeVoiceTtsRetainedDecodeStep>,
    managed_completions_drained: bool,
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
    model_identity: [u8; 32],
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
        let load_nonce = next_vibevoice_tts_model_load_nonce()?;
        let model_identity = vibevoice_tts_model_identity(model_dir, dtype, &config, load_nonce);
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
            model_identity,
        })
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<VibeVoicePhysicalStateSpec> {
        let tokenizer_domains = [
            VibeVoiceTokenizerStateDomain::new(
                VIBEVOICE_TTS_ACOUSTIC_DOMAIN,
                StateGroupId::new(3),
                StateClock::CodecFrames,
                self.acoustic_tokenizer.decoder_state_geometry(),
            )?,
            VibeVoiceTokenizerStateDomain::new(
                VIBEVOICE_TTS_SEMANTIC_DOMAIN,
                StateGroupId::new(3),
                StateClock::CodecFrames,
                self.semantic_tokenizer.encoder_state_geometry(),
            )?,
        ];
        let contract = vibevoice_invocation_contract(
            &self.language_model,
            self.dtype,
            default_kv_page_size(),
            &[CacheDomainId::new(1), CacheDomainId::new(2)],
            &tokenizer_domains,
        )?;
        let max_context_tokens = self
            .config
            .decoder_config
            .max_position_embeddings
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "VibeVoice TTS decoder config has no maximum context length".into(),
                )
            })?;
        vibevoice_physical_state_spec(stage_graphs, contract, max_context_tokens)
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
        let dense_decode_max_tokens = 0;
        let head_dim = self.language_model.attention_head_dim();
        let cuda_flash_attention_head_dim_supported =
            cuda_flash_attention_head_dim_supported(head_dim);
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
            dense_decode_supported: dense_decode_max_tokens > 0,
            dense_projection_count: projection_diagnostics.dense_projection_count,
            dense_bias_projection_count: projection_diagnostics.dense_bias_projection_count,
            quantized_projection_count: projection_diagnostics.quantized_projection_count,
            cfg_batching_enabled: vibevoice_cfg_batching_enabled(self.device.kind),
            dense_decode_max_tokens,
            cuda_flash_attention_requested: flash_attention_requested(),
            cuda_flash_attention_compiled: flash_attention_compiled(),
            cuda_flash_attention_head_dim_supported,
            cuda_flash_attention_active: should_enable_flash_attention_v2(&self.device.device)
                && cuda_flash_attention_head_dim_supported,
        }
    }

    pub fn generate_with_reference(
        &self,
        text: &str,
        reference: &VibeVoiceSpeakerReference,
        speaker: Option<&str>,
        params: VibeVoiceTtsGenerationParams,
    ) -> Result<VibeVoiceTtsOutput> {
        let _ = (text, reference, speaker, params);
        Err(Error::InferenceError(
            "VibeVoice TTS requires lifecycle-owned physical invocation caches".into(),
        ))
    }

    pub(crate) fn generate_with_reference_physical(
        &self,
        text: &str,
        reference: &VibeVoiceSpeakerReference,
        speaker: Option<&str>,
        params: VibeVoiceTtsGenerationParams,
        leases: &mut InvocationWorkspaceLeaseSetV2,
    ) -> Result<VibeVoiceTtsOutput> {
        self.generate_with_reference_internal(text, reference, speaker, params, leases)
    }

    pub(crate) fn prepare_retained_artifact(
        &self,
        text: &str,
        reference: &VibeVoiceSpeakerReference,
        speaker: Option<&str>,
    ) -> Result<Arc<VibeVoiceTtsPreparedArtifact>> {
        validate_vibevoice_tts_inputs(text, reference)?;
        let mut preparation_profile = VibeVoiceTtsProfile::default();
        let started = Instant::now();
        let reference = self.encode_reference(reference)?;
        preparation_profile.reference_encode_ms = elapsed_ms(started);
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
        let input_embeds = if let Some(range) = prompt.reference_voice_range {
            let reference_embeds = self
                .acoustic_connector
                .forward(&reference.scaled_latents.to_dtype(input_embeds.dtype())?)?;
            replace_range_with_features(&input_embeds, range, &reference_embeds)?
        } else {
            input_embeds
        };
        let negative_id = vibevoice_tts_negative_prefill_token(self.tokenizer.specials());
        let negative_ids = Tensor::from_vec(vec![negative_id], (1, 1), &self.device.device)?;
        let negative_embed = self.language_model.embeddings(&negative_ids)?;
        preparation_profile.prompt_embed_ms = elapsed_ms(started);
        Ok(Arc::new(VibeVoiceTtsPreparedArtifact {
            model_identity: self.model_identity,
            input_ids: prompt.input_ids.into(),
            input_embeds,
            negative_embed,
            normalization: reference.normalization,
            preparation_profile,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<VibeVoiceTtsPreparedArtifact>,
        params: VibeVoiceTtsGenerationParams,
        positive_cache: PhysicalPagedKvCache,
        negative_cache: PhysicalPagedKvCache,
    ) -> Result<VibeVoiceTtsRetainedState> {
        if artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "VibeVoice TTS artifact belongs to another model load".into(),
            ));
        }
        if positive_cache.context_len() != 0
            || negative_cache.context_len() != 0
            || positive_cache.arena().id() == negative_cache.arena().id()
        {
            return Err(Error::InvalidInput(
                "VibeVoice retained TTS requires empty domain-isolated LM caches".into(),
            ));
        }
        let max_frames = params.max_frames.max(1);
        let frame_noises = (0..max_frames)
            .map(|_| {
                Tensor::randn(
                    0f32,
                    1f32,
                    (1, self.config.acoustic_vae_dim()),
                    &self.device.device,
                )?
                .to_dtype(self.dtype)
                .map_err(Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(VibeVoiceTtsRetainedState {
            state_id: next_vibevoice_tts_state_id()?,
            model_identity: self.model_identity,
            artifact,
            params,
            positive_cache,
            negative_cache,
            positive_position: 0,
            negative_position: 0,
            last_hidden: None,
            negative_last_hidden: None,
            acoustic_clock: 0,
            semantic_clock: 0,
            scaled_latents: Vec::with_capacity(max_frames),
            frame_noises,
            finished: false,
            active_quantum: None,
            next_quantum: 1,
            staged_step: None,
            managed_completions_drained: true,
        })
    }

    pub(crate) fn new_retained_state_in_quantum(
        &self,
        artifact: Arc<VibeVoiceTtsPreparedArtifact>,
        params: VibeVoiceTtsGenerationParams,
        positive_cache: PhysicalPagedKvCache,
        negative_cache: PhysicalPagedKvCache,
    ) -> Result<(VibeVoiceTtsRetainedState, VibeVoiceTtsRetainedCheckpoint)> {
        let mut state =
            self.new_retained_state(artifact, params, positive_cache, negative_cache)?;
        state.active_quantum = Some(1);
        state.next_quantum = 2;
        let checkpoint = VibeVoiceTtsRetainedCheckpoint {
            state_id: state.state_id,
            quantum: 1,
            payload: Some(VibeVoiceTtsRetainedCheckpointPayload {
                positive_cache: None,
                negative_cache: None,
                positive_position: 0,
                negative_position: 0,
                last_hidden: None,
                negative_last_hidden: None,
                acoustic_clock: 0,
                semantic_clock: 0,
                scaled_latents: Vec::new(),
                finished: false,
                staged_step: None,
                managed_completions_drained: true,
            }),
        };
        Ok((state, checkpoint))
    }

    pub(crate) fn retained_prefill_step(
        &self,
        state: &mut VibeVoiceTtsRetainedState,
        max_tokens: usize,
    ) -> Result<VibeVoiceTtsRetainedPrefillStep> {
        self.validate_retained_state(state)?;
        if max_tokens == 0 {
            return Err(Error::InvalidInput(
                "VibeVoice retained prefill quantum must be nonzero".into(),
            ));
        }
        if state.active_quantum.is_none() {
            return Err(Error::InferenceError(
                "VibeVoice retained prefill requires an active managed quantum".into(),
            ));
        }
        let prompt_tokens = state.artifact.input_ids.len();
        let remaining = prompt_tokens.saturating_sub(state.positive_position);
        let take = remaining.min(max_tokens);
        if take > 0 {
            let embeds = state
                .artifact
                .input_embeds
                .narrow(1, state.positive_position, take)?;
            let hidden = self.language_model.forward_managed_hidden_with_embeds(
                &embeds,
                state.positive_position,
                &mut state.positive_cache,
                None,
            )?;
            state.positive_position += take;
            state.last_hidden = Some(last_sequence_hidden(
                &hidden,
                "VibeVoice retained TTS positive prefill",
            )?);
            state.managed_completions_drained = false;
        }
        if state.negative_position == 0 {
            let hidden = self.language_model.forward_managed_hidden_with_embeds(
                &state.artifact.negative_embed,
                0,
                &mut state.negative_cache,
                None,
            )?;
            state.negative_position = 1;
            state.negative_last_hidden = Some(last_sequence_hidden(
                &hidden,
                "VibeVoice retained TTS negative prefill",
            )?);
            state.managed_completions_drained = false;
        }
        Ok(VibeVoiceTtsRetainedPrefillStep {
            consumed_positive_tokens: take,
            positive_position: state.positive_position,
            negative_position: state.negative_position,
            complete: state.positive_position == prompt_tokens && state.negative_position == 1,
        })
    }

    pub(crate) fn retained_decode_step(
        &self,
        state: &mut VibeVoiceTtsRetainedState,
        tokenizer_quantum: &VibeVoiceTtsTokenizerQuantum,
    ) -> Result<VibeVoiceTtsRetainedDecodeStep> {
        self.validate_retained_state(state)?;
        if state.active_quantum.is_none() || state.staged_step.is_some() {
            return Err(Error::InferenceError(
                "VibeVoice retained decode requires one clean active quantum".into(),
            ));
        }
        let generated_frames = state.scaled_latents.len();
        if state.positive_position != state.artifact.input_ids.len() + generated_frames
            || state.negative_position != 1 + generated_frames
        {
            return Err(Error::InferenceError(
                "VibeVoice retained decode cannot run before prefill completes".into(),
            ));
        }
        if state.finished {
            return Ok(VibeVoiceTtsRetainedDecodeStep {
                frames_generated: state.scaled_latents.len(),
                finished: true,
            });
        }
        if state.scaled_latents.len() >= state.params.max_frames.max(1) {
            state.finished = true;
            let step = VibeVoiceTtsRetainedDecodeStep {
                frames_generated: state.scaled_latents.len(),
                finished: true,
            };
            state.staged_step = Some(step.clone());
            return Ok(step);
        }
        if state.scaled_latents.len() >= MIN_FRAMES_BEFORE_STOP {
            let predicted = next_tts_control_token_from_hidden(
                &self.language_model,
                state.last_hidden.as_ref().ok_or_else(|| {
                    Error::InferenceError("VibeVoice retained TTS has no positive hidden".into())
                })?,
                self.tokenizer.specials(),
            )?;
            if predicted == self.tokenizer.specials().speech_end
                || predicted == self.tokenizer.specials().endoftext
            {
                state.finished = true;
                let step = VibeVoiceTtsRetainedDecodeStep {
                    frames_generated: state.scaled_latents.len(),
                    finished: true,
                };
                state.staged_step = Some(step.clone());
                return Ok(step);
            }
        }
        let plan = vibevoice_diffusion_plan(
            &self.prediction_head,
            VibeVoiceDiffusionScheduler::new(
                self.prediction_head.config().ddpm_num_steps,
                vibevoice_effective_diffusion_steps(self.device.kind, state.params.diffusion_steps),
            ),
            &self.device.device,
            self.dtype,
            state.params.cfg_scale,
            self.device.kind,
        )?;
        let frame = state.scaled_latents.len();
        let latent = self.sample_speech_latent_from_noise(
            state.last_hidden.as_ref().unwrap(),
            state.negative_last_hidden.as_ref(),
            &plan,
            &state.frame_noises[frame],
        )?;
        let latent_frame = latent.unsqueeze(1)?;
        let feedback = self.generated_speech_embed_retained(
            &latent_frame,
            &state.artifact.normalization,
            state.acoustic_clock,
            state.semantic_clock,
            tokenizer_quantum,
        )?;
        let hidden = self.language_model.forward_managed_hidden_with_embeds(
            &feedback.embed,
            state.positive_position,
            &mut state.positive_cache,
            None,
        )?;
        let negative_hidden = self.language_model.forward_managed_hidden_with_embeds(
            &feedback.embed,
            state.negative_position,
            &mut state.negative_cache,
            None,
        )?;
        state.positive_position += 1;
        state.negative_position += 1;
        state.acoustic_clock += 1;
        state.semantic_clock += 1;
        state.last_hidden = Some(last_sequence_hidden(
            &hidden,
            "VibeVoice retained TTS positive decode",
        )?);
        state.negative_last_hidden = Some(last_sequence_hidden(
            &negative_hidden,
            "VibeVoice retained TTS negative decode",
        )?);
        state.scaled_latents.push(latent_frame);
        state.managed_completions_drained = false;
        state.finished = state.scaled_latents.len() >= state.params.max_frames.max(1);
        let step = VibeVoiceTtsRetainedDecodeStep {
            frames_generated: state.scaled_latents.len(),
            finished: state.finished,
        };
        state.staged_step = Some(step.clone());
        Ok(step)
    }

    /// Native cross-request diffusion cohort. Tokenizer feedback and Qwen
    /// custom-embedding decode remain scalar because their existing physical
    /// APIs do not expose a truthful B>1 entry point.
    pub(crate) fn retained_decode_step_batch(
        &self,
        states: &mut [&mut VibeVoiceTtsRetainedState],
        tokenizer_quanta: &[VibeVoiceTtsTokenizerQuantum],
    ) -> Result<Vec<VibeVoiceTtsRetainedDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        if states.len() != tokenizer_quanta.len() {
            return Err(Error::InvalidInput(
                "VibeVoice retained TTS batch state/tokenizer widths differ".into(),
            ));
        }
        let params = states[0].params.clone();
        for state in states.iter() {
            self.validate_retained_state(state)?;
            if state.params != params
                || state.active_quantum.is_none()
                || state.staged_step.is_some()
                || state.finished
                || state.positive_position
                    != state.artifact.input_ids.len() + state.scaled_latents.len()
                || state.negative_position != 1 + state.scaled_latents.len()
                || state.scaled_latents.len() >= state.params.max_frames.max(1)
            {
                return Err(Error::InvalidInput(
                    "VibeVoice retained diffusion batch requires compatible live rows".into(),
                ));
            }
            if state.scaled_latents.len() >= MIN_FRAMES_BEFORE_STOP {
                let predicted = next_tts_control_token_from_hidden(
                    &self.language_model,
                    state.last_hidden.as_ref().unwrap(),
                    self.tokenizer.specials(),
                )?;
                if predicted != self.tokenizer.specials().speech_pad {
                    return Err(Error::InvalidInput(
                        "VibeVoice terminal control rows require scalar fallback".into(),
                    ));
                }
            }
        }
        let plan = vibevoice_diffusion_plan(
            &self.prediction_head,
            VibeVoiceDiffusionScheduler::new(
                self.prediction_head.config().ddpm_num_steps,
                vibevoice_effective_diffusion_steps(self.device.kind, params.diffusion_steps),
            ),
            &self.device.device,
            self.dtype,
            params.cfg_scale,
            self.device.kind,
        )?;
        let latents = self.sample_speech_latent_cross_request_batch(states, &plan)?;
        let mut steps = Vec::with_capacity(states.len());
        for ((state, quantum), latent) in states
            .iter_mut()
            .zip(tokenizer_quanta)
            .zip(latents.into_iter())
        {
            let latent_frame = latent.unsqueeze(1)?;
            let feedback = self.generated_speech_embed_retained(
                &latent_frame,
                &state.artifact.normalization,
                state.acoustic_clock,
                state.semantic_clock,
                quantum,
            )?;
            let hidden = self.language_model.forward_managed_hidden_with_embeds(
                &feedback.embed,
                state.positive_position,
                &mut state.positive_cache,
                None,
            )?;
            let negative_hidden = self.language_model.forward_managed_hidden_with_embeds(
                &feedback.embed,
                state.negative_position,
                &mut state.negative_cache,
                None,
            )?;
            state.positive_position += 1;
            state.negative_position += 1;
            state.acoustic_clock += 1;
            state.semantic_clock += 1;
            state.last_hidden = Some(last_sequence_hidden(
                &hidden,
                "VibeVoice retained TTS batched positive decode",
            )?);
            state.negative_last_hidden = Some(last_sequence_hidden(
                &negative_hidden,
                "VibeVoice retained TTS batched negative decode",
            )?);
            state.scaled_latents.push(latent_frame);
            state.managed_completions_drained = false;
            state.finished = state.scaled_latents.len() >= state.params.max_frames.max(1);
            let step = VibeVoiceTtsRetainedDecodeStep {
                frames_generated: state.scaled_latents.len(),
                finished: state.finished,
            };
            state.staged_step = Some(step.clone());
            steps.push(step);
        }
        Ok(steps)
    }

    pub(crate) fn finalize_retained_state(
        &self,
        state: &VibeVoiceTtsRetainedState,
    ) -> Result<VibeVoiceTtsOutput> {
        self.validate_retained_state(state)?;
        if !state.finished || state.scaled_latents.is_empty() || state.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "VibeVoice retained TTS can finalize only a committed non-empty terminal state"
                    .into(),
            ));
        }
        let mut profile = state.artifact.preparation_profile.clone();
        profile.frames_generated = state.scaled_latents.len();
        profile.diffusion_steps =
            vibevoice_effective_diffusion_steps(self.device.kind, state.params.diffusion_steps);
        let scaled_latents = Tensor::cat(&state.scaled_latents, 1)?;
        let unscaled = unscale_latents(
            &scaled_latents,
            &state.artifact.normalization.bias,
            &state.artifact.normalization.scale,
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
        Ok(VibeVoiceTtsOutput {
            samples,
            sample_rate: self.preprocessor.target_sample_rate(),
            frames_generated: state.scaled_latents.len(),
            profile,
        })
    }

    fn validate_retained_state(&self, state: &VibeVoiceTtsRetainedState) -> Result<()> {
        if state.model_identity != self.model_identity
            || state.artifact.model_identity != self.model_identity
        {
            return Err(Error::InvalidInput(
                "VibeVoice retained TTS state belongs to another model load".into(),
            ));
        }
        Ok(())
    }

    fn generate_with_reference_internal(
        &self,
        text: &str,
        reference: &VibeVoiceSpeakerReference,
        speaker: Option<&str>,
        params: VibeVoiceTtsGenerationParams,
        leases: &mut InvocationWorkspaceLeaseSetV2,
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
        let domains = leases.domains().collect::<Vec<_>>();
        if domains
            != vec![
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_POSITIVE_DOMAIN,
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_NEGATIVE_DOMAIN,
                VIBEVOICE_TTS_ACOUSTIC_DOMAIN,
                VIBEVOICE_TTS_SEMANTIC_DOMAIN,
            ]
            || leases
                .lease(crate::models::architectures::vibevoice::VIBEVOICE_TTS_POSITIVE_DOMAIN)?
                .kind()
                != InvocationStateBackingKindV2::PagedAttention
            || leases
                .lease(crate::models::architectures::vibevoice::VIBEVOICE_TTS_NEGATIVE_DOMAIN)?
                .kind()
                != InvocationStateBackingKindV2::PagedAttention
            || leases.lease(VIBEVOICE_TTS_ACOUSTIC_DOMAIN)?.kind()
                != InvocationStateBackingKindV2::Tensor
            || leases.lease(VIBEVOICE_TTS_SEMANTIC_DOMAIN)?.kind()
                != InvocationStateBackingKindV2::Tensor
        {
            return Err(Error::InvalidInput(
                "VibeVoice TTS requires exact positive/negative pages and acoustic/semantic tensor state"
                    .into(),
            ));
        }

        let mut profile = VibeVoiceTtsProfile::default();
        let kernel_telemetry_start = kernel_telemetry_snapshot();
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
        let (prefill_hidden, negative_hidden) = {
            let (positive, negative) = leases.lease_pair_mut(
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_POSITIVE_DOMAIN,
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_NEGATIVE_DOMAIN,
            )?;
            let positive_cache = positive.paged_cache_mut()?;
            let negative_cache = negative.paged_cache_mut()?;
            if positive_cache.context_len() != 0
                || negative_cache.context_len() != 0
                || positive_cache.arena().id() == negative_cache.arena().id()
            {
                return Err(Error::InvalidInput(
                    "VibeVoice TTS requires empty, domain-isolated decoder pages".into(),
                ));
            }
            let started = Instant::now();
            let prefill_hidden = self.language_model.forward_managed_hidden_with_embeds(
                &input_embeds,
                0,
                positive_cache,
                None,
            )?;
            profile.positive_prefill_ms = elapsed_ms(started);
            let negative_id = vibevoice_tts_negative_prefill_token(self.tokenizer.specials());
            let negative_ids = Tensor::from_vec(vec![negative_id], (1, 1), &self.device.device)?;
            let started = Instant::now();
            let negative_hidden = self.language_model.forward_managed_hidden_with_embeds(
                &self.language_model.embeddings(&negative_ids)?,
                0,
                negative_cache,
                None,
            )?;
            profile.negative_prefill_ms = elapsed_ms(started);
            (prefill_hidden, negative_hidden)
        };
        let mut pos = prompt.input_ids.len();
        let mut last_hidden = last_sequence_hidden(&prefill_hidden, "VibeVoice TTS prefill")?;

        let mut negative_pos = 1usize;
        let mut negative_last_hidden =
            last_sequence_hidden(&negative_hidden, "VibeVoice TTS negative prefill")?;

        let diffusion_steps =
            vibevoice_effective_diffusion_steps(self.device.kind, params.diffusion_steps);
        let diffusion_plan = vibevoice_diffusion_plan(
            &self.prediction_head,
            VibeVoiceDiffusionScheduler::new(
                self.prediction_head.config().ddpm_num_steps,
                diffusion_steps,
            ),
            &self.device.device,
            self.dtype,
            params.cfg_scale,
            self.device.kind,
        )?;
        profile.diffusion_steps = diffusion_plan.steps.len();
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
            let feedback = {
                let (acoustic, semantic) = leases
                    .lease_pair_mut(VIBEVOICE_TTS_ACOUSTIC_DOMAIN, VIBEVOICE_TTS_SEMANTIC_DOMAIN)?;
                if acoustic.kind() != InvocationStateBackingKindV2::Tensor
                    || semantic.kind() != InvocationStateBackingKindV2::Tensor
                {
                    return Err(Error::InferenceError(
                        "VibeVoice TTS tokenizer domains require tensor backing".into(),
                    ));
                }
                self.generated_speech_embed(
                    &latent_frame,
                    &reference.normalization,
                    acoustic.typed_mut::<InvocationTensorLease>()?,
                    semantic.typed_mut::<InvocationTensorLease>()?,
                )?
            };
            profile.feedback_acoustic_decode_ms += feedback.acoustic_decode_ms;
            profile.feedback_semantic_encode_ms += feedback.semantic_encode_ms;
            profile.feedback_connector_ms += feedback.connector_ms;
            scaled_latents.push(latent_frame);

            let (hidden, negative_hidden) = {
                let (positive, negative) = leases.lease_pair_mut(
                    crate::models::architectures::vibevoice::VIBEVOICE_TTS_POSITIVE_DOMAIN,
                    crate::models::architectures::vibevoice::VIBEVOICE_TTS_NEGATIVE_DOMAIN,
                )?;
                let started = Instant::now();
                let hidden = self.language_model.forward_managed_hidden_with_embeds(
                    &feedback.embed,
                    pos,
                    positive.paged_cache_mut()?,
                    None,
                )?;
                profile.positive_decode_ms += elapsed_ms(started);
                let started = Instant::now();
                let negative_hidden = self.language_model.forward_managed_hidden_with_embeds(
                    &feedback.embed,
                    negative_pos,
                    negative.paged_cache_mut()?,
                    None,
                )?;
                profile.negative_decode_ms += elapsed_ms(started);
                (hidden, negative_hidden)
            };
            pos += 1;
            last_hidden = last_sequence_hidden(&hidden, "VibeVoice TTS generated frame")?;
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
        let kernel_telemetry_end = kernel_telemetry_snapshot();
        apply_kernel_telemetry_delta(&mut profile, &kernel_telemetry_start, &kernel_telemetry_end);
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
        acoustic_state: &mut InvocationTensorLease,
        semantic_state: &mut InvocationTensorLease,
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
        let audio_chunk = self.acoustic_tokenizer.decode_streaming_physical(
            &unscaled_frame,
            VIBEVOICE_TTS_ACOUSTIC_DOMAIN,
            1,
            acoustic_state,
        )?;
        let acoustic_decode_ms = elapsed_ms(started);
        let started = Instant::now();
        let semantic = self
            .semantic_tokenizer
            .encode_streaming_physical(
                &audio_chunk,
                VIBEVOICE_TTS_SEMANTIC_DOMAIN,
                1,
                semantic_state,
            )?
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

    fn generated_speech_embed_retained(
        &self,
        scaled_latent_frame: &Tensor,
        normalization: &LatentNormalization,
        acoustic_clock: u64,
        semantic_clock: u64,
        quantum: &VibeVoiceTtsTokenizerQuantum,
    ) -> Result<GeneratedSpeechFeedback> {
        if acoustic_clock != semantic_clock {
            return Err(Error::InferenceError(
                "VibeVoice retained tokenizer clocks diverged".into(),
            ));
        }
        let started = Instant::now();
        let acoustic_embed = self.acoustic_connector.forward(scaled_latent_frame)?;
        let connector_acoustic_ms = elapsed_ms(started);
        let unscaled_frame = unscale_latents(
            scaled_latent_frame,
            &normalization.bias,
            &normalization.scale,
        )?;
        let started = Instant::now();
        let audio_chunk = self.acoustic_tokenizer.decode_streaming_retained(
            &unscaled_frame,
            VIBEVOICE_TTS_ACOUSTIC_DOMAIN,
            acoustic_clock,
            acoustic_clock + 1,
            quantum.transaction,
            &quantum.arena,
        )?;
        let acoustic_decode_ms = elapsed_ms(started);
        let started = Instant::now();
        let semantic = self
            .semantic_tokenizer
            .encode_streaming_retained_frame(
                &audio_chunk,
                VIBEVOICE_TTS_SEMANTIC_DOMAIN,
                semantic_clock,
                semantic_clock + 1,
                quantum.transaction,
                &quantum.arena,
            )?
            .mode();
        let semantic_encode_ms = elapsed_ms(started);
        let started = Instant::now();
        let semantic_embed = self.semantic_connector.forward(&semantic)?;
        let embed = combine_speech_embeddings(
            &acoustic_embed,
            &semantic_embed,
            "VibeVoice retained TTS generated frame",
        )?;
        Ok(GeneratedSpeechFeedback {
            embed,
            acoustic_decode_ms,
            semantic_encode_ms,
            connector_ms: connector_acoustic_ms + elapsed_ms(started),
        })
    }

    fn sample_speech_latent(
        &self,
        condition: &Tensor,
        negative_condition: Option<&Tensor>,
        plan: &VibeVoiceDiffusionPlan,
    ) -> Result<Tensor> {
        let condition = self.prediction_head.condition_projection(condition)?;
        let negative_condition = if plan.cfg_tensor.is_some() {
            negative_condition
                .map(|condition| self.prediction_head.condition_projection(condition))
                .transpose()?
        } else {
            None
        };
        let cuda_batched_conditions = if plan.cuda_prebatched_cfg {
            negative_condition
                .as_ref()
                .map(|negative| Tensor::cat(&[condition.clone(), negative.clone()], 0))
                .transpose()?
        } else {
            None
        };
        let mut speech = Tensor::randn(
            0f32,
            1f32,
            (1, self.config.acoustic_vae_dim()),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        for step in &plan.steps {
            let model_output = if let (Some(negative_condition), Some(cfg)) =
                (negative_condition.as_ref(), plan.cfg_tensor.as_ref())
            {
                if let (Some(batched_conditions), Some(batched_timestep_embedding)) = (
                    cuda_batched_conditions.as_ref(),
                    step.cuda_batched_timestep_embedding.as_ref(),
                ) {
                    let latents = Tensor::cat(&[speech.clone(), speech.clone()], 0)?;
                    self.prediction_head.forward_cfg_batched_with_prebatched(
                        &latents,
                        batched_timestep_embedding,
                        batched_conditions,
                        cfg,
                    )?
                } else if plan.batch_cfg_prediction {
                    self.prediction_head.forward_cfg_batched_with_precomputed(
                        &speech,
                        &step.timestep_embedding,
                        &condition,
                        negative_condition,
                        cfg,
                    )?
                } else {
                    let positive_output = self.prediction_head.forward_with_precomputed(
                        &speech,
                        &step.timestep_embedding,
                        &condition,
                    )?;
                    let negative_output = self.prediction_head.forward_with_precomputed(
                        &speech,
                        &step.timestep_embedding,
                        negative_condition,
                    )?;
                    let guidance = positive_output.broadcast_sub(&negative_output)?;
                    negative_output.broadcast_add(&guidance.broadcast_mul(cfg)?)?
                }
            } else {
                self.prediction_head.forward_with_precomputed(
                    &speech,
                    &step.timestep_embedding,
                    &condition,
                )?
            };
            speech = plan.scheduler.step_v_prediction_with_tensors(
                &model_output,
                &speech,
                &step.tensors,
            )?;
        }
        Ok(speech)
    }

    fn sample_speech_latent_from_noise(
        &self,
        condition: &Tensor,
        negative_condition: Option<&Tensor>,
        plan: &VibeVoiceDiffusionPlan,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let condition = self.prediction_head.condition_projection(condition)?;
        let negative_condition = if plan.cfg_tensor.is_some() {
            negative_condition
                .map(|condition| self.prediction_head.condition_projection(condition))
                .transpose()?
        } else {
            None
        };
        let mut speech = noise.clone();
        for step in &plan.steps {
            let model_output = if let (Some(negative), Some(cfg)) =
                (negative_condition.as_ref(), plan.cfg_tensor.as_ref())
            {
                let conditions = Tensor::cat(&[condition.clone(), negative.clone()], 0)?;
                let latents = Tensor::cat(&[speech.clone(), speech.clone()], 0)?;
                let timestep = Tensor::cat(
                    &[
                        step.timestep_embedding.clone(),
                        step.timestep_embedding.clone(),
                    ],
                    0,
                )?;
                let outputs = self.prediction_head.forward_with_precomputed(
                    &latents,
                    &timestep,
                    &conditions,
                )?;
                let positive = outputs.narrow(0, 0, 1)?;
                let negative = outputs.narrow(0, 1, 1)?;
                negative.broadcast_add(&positive.broadcast_sub(&negative)?.broadcast_mul(cfg)?)?
            } else {
                self.prediction_head.forward_with_precomputed(
                    &speech,
                    &step.timestep_embedding,
                    &condition,
                )?
            };
            speech = plan.scheduler.step_v_prediction_with_tensors(
                &model_output,
                &speech,
                &step.tensors,
            )?;
        }
        Ok(speech)
    }

    fn sample_speech_latent_cross_request_batch(
        &self,
        states: &[&mut VibeVoiceTtsRetainedState],
        plan: &VibeVoiceDiffusionPlan,
    ) -> Result<Vec<Tensor>> {
        let batch = states.len();
        let conditions = states
            .iter()
            .map(|state| state.last_hidden.as_ref().unwrap())
            .collect::<Vec<_>>();
        let condition = self
            .prediction_head
            .condition_projection(&Tensor::cat(&conditions, 0)?)?;
        let negative_condition = if plan.cfg_tensor.is_some() {
            let negative = states
                .iter()
                .map(|state| state.negative_last_hidden.as_ref().unwrap())
                .collect::<Vec<_>>();
            Some(
                self.prediction_head
                    .condition_projection(&Tensor::cat(&negative, 0)?)?,
            )
        } else {
            None
        };
        let noises = states
            .iter()
            .map(|state| &state.frame_noises[state.scaled_latents.len()])
            .collect::<Vec<_>>();
        let mut speech = Tensor::cat(&noises, 0)?;
        for step in &plan.steps {
            let model_output = if let (Some(negative), Some(cfg)) =
                (negative_condition.as_ref(), plan.cfg_tensor.as_ref())
            {
                let all_conditions = Tensor::cat(&[condition.clone(), negative.clone()], 0)?;
                let all_latents = Tensor::cat(&[speech.clone(), speech.clone()], 0)?;
                let timestep = step
                    .timestep_embedding
                    .broadcast_as((batch * 2, step.timestep_embedding.dim(1)?))?;
                let outputs = self.prediction_head.forward_with_precomputed(
                    &all_latents,
                    &timestep,
                    &all_conditions,
                )?;
                let positive = outputs.narrow(0, 0, batch)?;
                let negative = outputs.narrow(0, batch, batch)?;
                negative.broadcast_add(&positive.broadcast_sub(&negative)?.broadcast_mul(cfg)?)?
            } else {
                let timestep = step
                    .timestep_embedding
                    .broadcast_as((batch, step.timestep_embedding.dim(1)?))?;
                self.prediction_head
                    .forward_with_precomputed(&speech, &timestep, &condition)?
            };
            speech = plan.scheduler.step_v_prediction_with_tensors(
                &model_output,
                &speech,
                &step.tensors,
            )?;
        }
        (0..batch)
            .map(|row| speech.narrow(0, row, 1).map_err(Error::from))
            .collect()
    }
}

impl VibeVoiceTtsPreparedArtifact {
    pub(crate) fn prompt_tokens(&self) -> usize {
        self.input_ids.len()
    }

    pub(crate) fn retained_tensor_bytes(&self) -> Result<u64> {
        [&self.input_embeds, &self.negative_embed]
            .into_iter()
            .try_fold(0u64, |bytes, tensor| {
                let tensor_bytes = u64::try_from(tensor.elem_count())
                    .ok()
                    .and_then(|elements| {
                        elements.checked_mul(u64::try_from(tensor.dtype().size_in_bytes()).ok()?)
                    })
                    .ok_or_else(|| {
                        Error::Overloaded("VibeVoice TTS artifact bytes overflow".into())
                    })?;
                bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("VibeVoice TTS artifact bytes overflow".into())
                })
            })
    }
}

impl VibeVoiceTtsRetainedState {
    pub(crate) fn begin_managed_quantum(
        &mut self,
        positive_cache: PhysicalPagedKvCache,
        negative_cache: PhysicalPagedKvCache,
    ) -> Result<VibeVoiceTtsRetainedCheckpoint> {
        if self.active_quantum.is_some()
            || self.staged_step.is_some()
            || !self.managed_completions_drained
        {
            return Err(Error::InferenceError(
                "VibeVoice TTS retained quantum is already active or has staged output".into(),
            ));
        }
        if positive_cache.arena().id() != self.positive_cache.arena().id()
            || negative_cache.arena().id() != self.negative_cache.arena().id()
            || positive_cache.context_len() != self.positive_position
            || negative_cache.context_len() != self.negative_position
        {
            return Err(Error::InvalidInput(
                "VibeVoice TTS managed cache authority or position changed".into(),
            ));
        }
        let quantum = self.next_quantum;
        self.next_quantum = self
            .next_quantum
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("VibeVoice TTS quantum overflow".into()))?;
        self.active_quantum = Some(quantum);
        Ok(VibeVoiceTtsRetainedCheckpoint {
            state_id: self.state_id,
            quantum,
            payload: Some(VibeVoiceTtsRetainedCheckpointPayload {
                positive_cache: Some(std::mem::replace(&mut self.positive_cache, positive_cache)),
                negative_cache: Some(std::mem::replace(&mut self.negative_cache, negative_cache)),
                positive_position: self.positive_position,
                negative_position: self.negative_position,
                last_hidden: self.last_hidden.clone(),
                negative_last_hidden: self.negative_last_hidden.clone(),
                acoustic_clock: self.acoustic_clock,
                semantic_clock: self.semantic_clock,
                scaled_latents: self.scaled_latents.clone(),
                finished: self.finished,
                staged_step: self.staged_step.clone(),
                managed_completions_drained: self.managed_completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut VibeVoiceTtsRetainedCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "VibeVoice TTS managed completions must be drained before commit".into(),
            ));
        }
        checkpoint.payload.take();
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut VibeVoiceTtsRetainedCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS checkpoint was already consumed".into())
        })?;
        self.positive_cache = payload.positive_cache.ok_or_else(|| {
            Error::InferenceError(
                "initial VibeVoice TTS quantum must be discarded on rollback".into(),
            )
        })?;
        self.negative_cache = payload.negative_cache.ok_or_else(|| {
            Error::InferenceError(
                "initial VibeVoice TTS quantum must be discarded on rollback".into(),
            )
        })?;
        self.positive_position = payload.positive_position;
        self.negative_position = payload.negative_position;
        self.last_hidden = payload.last_hidden;
        self.negative_last_hidden = payload.negative_last_hidden;
        self.acoustic_clock = payload.acoustic_clock;
        self.semantic_clock = payload.semantic_clock;
        self.scaled_latents = payload.scaled_latents;
        self.finished = payload.finished;
        self.staged_step = payload.staged_step;
        self.managed_completions_drained = payload.managed_completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    fn validate_checkpoint(&self, checkpoint: &VibeVoiceTtsRetainedCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "VibeVoice TTS checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn take_staged_step(&mut self) -> Option<VibeVoiceTtsRetainedDecodeStep> {
        self.staged_step.take()
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let mut completions = self.positive_cache.take_completed_writes();
        completions.extend(self.negative_cache.take_completed_writes());
        self.managed_completions_drained = true;
        completions
    }

    pub(crate) const fn positive_position(&self) -> usize {
        self.positive_position
    }

    pub(crate) const fn negative_position(&self) -> usize {
        self.negative_position
    }

    pub(crate) const fn acoustic_clock(&self) -> u64 {
        self.acoustic_clock
    }

    pub(crate) const fn semantic_clock(&self) -> u64 {
        self.semantic_clock
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

fn vibevoice_diffusion_plan(
    prediction_head: &VibeVoiceDiffusionHead,
    scheduler: VibeVoiceDiffusionScheduler,
    device: &Device,
    dtype: DType,
    cfg_scale: f32,
    device_kind: DeviceKind,
) -> Result<VibeVoiceDiffusionPlan> {
    let batch_cfg_prediction = vibevoice_cfg_batching_enabled(device_kind);
    let cfg_enabled = cfg_scale > 1.0;
    let cuda_prebatched_cfg = cfg_enabled && batch_cfg_prediction && device_kind.is_cuda();
    let steps = scheduler
        .step_tensors(device, dtype)?
        .into_iter()
        .map(|tensors| {
            let timestep_embedding =
                prediction_head.timestep_embedding(&tensors.timestep_tensor)?;
            let cuda_batched_timestep_embedding = if cuda_prebatched_cfg {
                Some(Tensor::cat(
                    &[timestep_embedding.clone(), timestep_embedding.clone()],
                    0,
                )?)
            } else {
                None
            };
            Ok(VibeVoiceDiffusionPlanStep {
                tensors,
                timestep_embedding,
                cuda_batched_timestep_embedding,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(VibeVoiceDiffusionPlan {
        scheduler,
        cfg_tensor: if cfg_scale > 1.0 {
            Some(Tensor::new(cfg_scale, device)?.to_dtype(dtype)?)
        } else {
            None
        },
        batch_cfg_prediction,
        cuda_prebatched_cfg,
        steps,
    })
}

fn vibevoice_effective_diffusion_steps(device_kind: DeviceKind, requested_steps: usize) -> usize {
    let override_value = std::env::var(VIBEVOICE_CUDA_DDPM_STEPS_ENV).ok();
    vibevoice_effective_diffusion_steps_for(device_kind, requested_steps, override_value.as_deref())
}

fn vibevoice_effective_diffusion_steps_for(
    device_kind: DeviceKind,
    requested_steps: usize,
    override_value: Option<&str>,
) -> usize {
    let requested_steps = requested_steps.max(1);
    if !device_kind.is_cuda() {
        return requested_steps;
    }
    let Some(raw) = override_value
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return requested_steps;
    };
    match raw.parse::<usize>().ok().filter(|value| *value > 0) {
        Some(steps) => steps,
        None => {
            warn!(
                "{VIBEVOICE_CUDA_DDPM_STEPS_ENV}={raw:?} is invalid; using requested VibeVoice DDPM steps"
            );
            requested_steps
        }
    }
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

fn validate_vibevoice_tts_inputs(text: &str, reference: &VibeVoiceSpeakerReference) -> Result<()> {
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
    Ok(())
}

fn next_vibevoice_tts_state_id() -> Result<u64> {
    NEXT_VIBEVOICE_TTS_STATE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::InferenceError("VibeVoice TTS state identity overflow".into()))
}

fn next_vibevoice_tts_model_load_nonce() -> Result<u64> {
    NEXT_VIBEVOICE_TTS_MODEL_LOAD_NONCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::ModelLoadError("VibeVoice TTS load nonce overflow".into()))
}

fn vibevoice_tts_model_identity(
    model_dir: &Path,
    dtype: DType,
    config: &VibeVoiceConfig,
    load_nonce: u64,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"izwi-vibevoice-tts-model-v1");
    hasher.update(load_nonce.to_le_bytes());
    hasher.update(model_dir.as_os_str().as_encoded_bytes());
    hasher.update(format!("{dtype:?}:{config:?}").as_bytes());
    let mut identity: [u8; 32] = hasher.finalize().into();
    if identity.iter().all(|byte| *byte == 0) {
        identity[0] = 1;
    }
    identity
}

fn apply_kernel_telemetry_delta(
    profile: &mut VibeVoiceTtsProfile,
    start: &KernelPathTelemetrySnapshot,
    end: &KernelPathTelemetrySnapshot,
) {
    profile.decode_attention_dense_calls = end
        .decode_attention_dense_total
        .saturating_sub(start.decode_attention_dense_total);
    profile.decode_attention_paged_calls = end
        .decode_attention_paged_total
        .saturating_sub(start.decode_attention_paged_total);
    profile.rope_kernel_calls = end
        .rope_kernel_total
        .saturating_sub(start.rope_kernel_total);
    profile.rope_manual_calls = end
        .rope_manual_total
        .saturating_sub(start.rope_manual_total);
    profile.fused_attention_attempts = end
        .fused_attention_attempts_total
        .saturating_sub(start.fused_attention_attempts_total);
    profile.fused_attention_successes = end
        .fused_attention_success_total
        .saturating_sub(start.fused_attention_success_total);
    profile.fused_attention_fallbacks = end
        .fused_attention_fallback_total
        .saturating_sub(start.fused_attention_fallback_total);
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
    if latents.device().is_cuda() {
        return reference_latent_normalization_cuda(latents);
    }
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

fn reference_latent_normalization_cuda(latents: &Tensor) -> Result<LatentNormalization> {
    let flattened = latents.to_dtype(DType::F32)?.flatten_all()?;
    let count = flattened.elem_count();
    if count == 0 {
        return Err(Error::InferenceError(
            "VibeVoice TTS reference encoder produced no latents".to_string(),
        ));
    }
    let count = count as f64;
    let mean = (flattened.sum_all()? / count)?;
    let centered = flattened.broadcast_sub(&mean)?;
    let variance = (centered.sqr()?.sum_all()? / count)?;
    let std = variance.sqrt()?.clamp(1e-5f64, f64::MAX)?;
    Ok(LatentNormalization {
        bias: mean.neg()?.to_dtype(latents.dtype())?,
        scale: std.recip()?.to_dtype(latents.dtype())?,
        source: LatentNormalizationSource::ReferenceStatistics,
    })
}

fn scale_latents(latents: &Tensor, bias: &Tensor, scale: &Tensor) -> Result<Tensor> {
    // VibeVoice normalizes speech latents as `(audio_tokens + bias) * scale`.
    if latents.device().is_cuda() {
        return latents
            .broadcast_add(bias)?
            .broadcast_mul(scale)
            .map_err(Error::from);
    }
    latents
        .broadcast_add(&factor_like(bias, latents)?)?
        .broadcast_mul(&factor_like(scale, latents)?)
        .map_err(Error::from)
}

fn unscale_latents(latents: &Tensor, bias: &Tensor, scale: &Tensor) -> Result<Tensor> {
    if latents.device().is_cuda() {
        return latents
            .broadcast_div(scale)?
            .broadcast_sub(bias)
            .map_err(Error::from);
    }
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
    use std::collections::HashMap;

    use candle_nn::VarBuilder;

    #[test]
    fn retained_artifact_accounts_for_model_bound_prompt_tensors() {
        let artifact = VibeVoiceTtsPreparedArtifact {
            model_identity: [7; 32],
            input_ids: vec![1, 2, 3].into(),
            input_embeds: Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap(),
            negative_embed: Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap(),
            normalization: LatentNormalization {
                bias: Tensor::new(0f32, &Device::Cpu).unwrap(),
                scale: Tensor::new(1f32, &Device::Cpu).unwrap(),
                source: LatentNormalizationSource::Checkpoint,
            },
            preparation_profile: VibeVoiceTtsProfile::default(),
        };
        assert_eq!(artifact.prompt_tokens(), 3);
        assert_eq!(artifact.retained_tensor_bytes().unwrap(), 64);
    }

    #[test]
    fn retained_model_and_state_identities_are_unique() {
        let first_model = next_vibevoice_tts_model_load_nonce().unwrap();
        let second_model = next_vibevoice_tts_model_load_nonce().unwrap();
        assert_eq!(first_model.checked_add(1), Some(second_model));
        let first_state = next_vibevoice_tts_state_id().unwrap();
        let second_state = next_vibevoice_tts_state_id().unwrap();
        assert_eq!(first_state.checked_add(1), Some(second_state));
    }

    #[test]
    fn retained_preparation_rejects_incomplete_reference_inputs() {
        let reference = VibeVoiceSpeakerReference {
            audio_samples: Vec::new(),
            sample_rate: 24_000,
            text: "reference".into(),
        };
        assert!(validate_vibevoice_tts_inputs("hello", &reference).is_err());
        let reference = VibeVoiceSpeakerReference {
            audio_samples: vec![0.1],
            sample_rate: 24_000,
            text: String::new(),
        };
        assert!(validate_vibevoice_tts_inputs("hello", &reference).is_err());
    }

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
        let head = tiny_diffusion_head(&device);
        let plan = vibevoice_diffusion_plan(
            &head,
            VibeVoiceDiffusionScheduler::new(1000, 3),
            &device,
            DType::F32,
            1.5,
            DeviceKind::Metal,
        )
        .unwrap();

        assert_eq!(plan.steps.len(), 3);
        assert!(plan.batch_cfg_prediction);
        assert!(!plan.cuda_prebatched_cfg);
        assert!(plan
            .steps
            .iter()
            .all(|step| step.cuda_batched_timestep_embedding.is_none()));
        assert_eq!(
            plan.cfg_tensor.as_ref().unwrap().to_vec0::<f32>().unwrap(),
            1.5
        );

        let cuda_plan = vibevoice_diffusion_plan(
            &head,
            VibeVoiceDiffusionScheduler::new(1000, 3),
            &device,
            DType::F32,
            1.5,
            DeviceKind::Cuda,
        )
        .unwrap();
        assert!(cuda_plan.batch_cfg_prediction);
        assert!(cuda_plan.cuda_prebatched_cfg);
        assert_eq!(
            cuda_plan.steps[0]
                .cuda_batched_timestep_embedding
                .as_ref()
                .unwrap()
                .dims(),
            &[2, 4]
        );

        let cpu_no_cfg = vibevoice_diffusion_plan(
            &head,
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
    fn cuda_diffusion_steps_override_is_cuda_only() {
        assert_eq!(
            vibevoice_effective_diffusion_steps_for(DeviceKind::Cuda, 20, Some("10")),
            10
        );
        assert_eq!(
            vibevoice_effective_diffusion_steps_for(DeviceKind::Metal, 20, Some("10")),
            20
        );
        assert_eq!(
            vibevoice_effective_diffusion_steps_for(DeviceKind::Cpu, 20, Some("10")),
            20
        );
    }

    #[test]
    fn cuda_diffusion_steps_override_rejects_invalid_values() {
        assert_eq!(
            vibevoice_effective_diffusion_steps_for(DeviceKind::Cuda, 20, Some("0")),
            20
        );
        assert_eq!(
            vibevoice_effective_diffusion_steps_for(DeviceKind::Cuda, 20, Some("not-a-number")),
            20
        );
        assert_eq!(
            vibevoice_effective_diffusion_steps_for(DeviceKind::Cuda, 20, None),
            20
        );
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

    fn tiny_diffusion_head(device: &Device) -> VibeVoiceDiffusionHead {
        let hidden = 4;
        let latent = 2;
        let ffn = 4;
        let mut tensors = HashMap::new();
        insert_linear(
            &mut tensors,
            "noisy_images_proj.weight",
            hidden,
            latent,
            device,
            0.01,
        );
        insert_linear(
            &mut tensors,
            "cond_proj.weight",
            hidden,
            hidden,
            device,
            -0.015,
        );
        insert_linear(
            &mut tensors,
            "t_embedder.mlp.0.weight",
            hidden,
            256,
            device,
            0.002,
        );
        insert_linear(
            &mut tensors,
            "t_embedder.mlp.2.weight",
            hidden,
            hidden,
            device,
            0.02,
        );
        insert_linear(
            &mut tensors,
            "layers.0.ffn.gate_proj.weight",
            ffn,
            hidden,
            device,
            0.01,
        );
        insert_linear(
            &mut tensors,
            "layers.0.ffn.up_proj.weight",
            ffn,
            hidden,
            device,
            -0.0125,
        );
        insert_linear(
            &mut tensors,
            "layers.0.ffn.down_proj.weight",
            hidden,
            ffn,
            device,
            0.0175,
        );
        insert_linear(
            &mut tensors,
            "layers.0.adaLN_modulation.1.weight",
            3 * hidden,
            hidden,
            device,
            0.006,
        );
        tensors.insert(
            "layers.0.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0f32; hidden], (hidden,), device).unwrap(),
        );
        insert_linear(
            &mut tensors,
            "final_layer.linear.weight",
            latent,
            hidden,
            device,
            -0.02,
        );
        insert_linear(
            &mut tensors,
            "final_layer.adaLN_modulation.1.weight",
            2 * hidden,
            hidden,
            device,
            0.0075,
        );
        let cfg = crate::models::architectures::vibevoice::config::VibeVoiceDiffusionHeadConfig {
            hidden_size: hidden,
            head_layers: 1,
            head_ffn_ratio: 1.0,
            rms_norm_eps: 1e-5,
            latent_size: latent,
            speech_vae_dim: None,
            prediction_type: "v_prediction".to_string(),
            diffusion_type: "ddpm".to_string(),
            ddpm_num_steps: 1000,
            ddpm_num_inference_steps: 4,
            ddpm_beta_schedule: "cosine".to_string(),
            ddpm_batch_mul: 4,
        };
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        VibeVoiceDiffusionHead::load(cfg, vb).unwrap()
    }

    fn insert_linear(
        tensors: &mut HashMap<String, Tensor>,
        name: &str,
        rows: usize,
        cols: usize,
        device: &Device,
        scale: f32,
    ) {
        let values = (0..rows * cols)
            .map(|idx| (idx as f32 + 1.0) * scale)
            .collect::<Vec<_>>();
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(values, (rows, cols), device).unwrap(),
        );
    }
}
