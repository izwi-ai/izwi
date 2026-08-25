//! Native VibeVoice-ASR model path.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::info;

use crate::backends::{backend_kind_for_device, BackendKind, DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::{
    ClockedStateProjection, ClockedStateSelection, InputRange, InvocationTensorLease,
    StageDescriptor, WorkCost,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    InvocationStateBackingKindV2, InvocationWorkspaceLeaseSetV2, StateClock, StateGroupId,
};
use crate::kv::CacheDomainId;
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{Qwen3Model, Qwen3WeightLayout};
use crate::models::architectures::vibevoice::config::{
    VibeVoiceConfig, VibeVoicePreprocessorConfig, VibeVoiceTokenizerConfig,
};
use crate::models::architectures::vibevoice::connector::SpeechConnector;
use crate::models::architectures::vibevoice::prompt::VibeVoicePromptTokenizer;
use crate::models::architectures::vibevoice::tokenizer::{
    VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer, VibeVoiceTokenizerEncoderOutput,
};
use crate::models::architectures::vibevoice::VIBEVOICE_ASR_TOKENIZER_GROUP;
use crate::models::architectures::vibevoice::{
    vibevoice_invocation_contract, vibevoice_physical_state_spec, VibeVoicePhysicalStateSpec,
    VibeVoiceTokenizerStateDomain, VIBEVOICE_ASR_ACOUSTIC_DOMAIN, VIBEVOICE_ASR_SEMANTIC_DOMAIN,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::weights::gguf::load_model_weights;

const DEFAULT_MAX_NEW_TOKENS: usize = 768;
const DEFAULT_MAX_AUDIO_SECONDS: f32 = 60.0 * 60.0;
const CUDA_MAX_AUDIO_SECONDS_ENV: &str = "IZWI_VIBEVOICE_ASR_CUDA_MAX_AUDIO_SECS";
const TOKENIZER_STREAMING_CHUNK_SECONDS: usize = 60;
static NEXT_VIBEVOICE_ASR_STATE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_VIBEVOICE_ASR_MODEL_LOAD_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrGenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for VibeVoiceAsrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct VibeVoiceAsrSegment {
    start_time: Option<f32>,
    end_time: Option<f32>,
    speaker_id: Option<String>,
    content: String,
}

#[derive(Debug, Clone, PartialEq)]
struct VibeVoiceAsrParsedOutput {
    text: String,
    raw_text: String,
    format: &'static str,
    segments: Vec<VibeVoiceAsrSegment>,
}

#[derive(Debug, Clone)]
struct VibeVoiceAsrPreprocessStats {
    normalized: bool,
    target_db_fs: f32,
    rms_before: f32,
    gain: f32,
    clipping_divisor: f32,
}

#[derive(Debug, Clone)]
struct VibeVoiceAsrEncodeStats {
    streaming: bool,
    chunks: usize,
    chunk_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

pub struct VibeVoiceAsrModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: VibeVoiceConfig,
    preprocessor: VibeVoicePreprocessorConfig,
    tokenizer: VibeVoicePromptTokenizer,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    semantic_tokenizer: VibeVoiceSemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    language_model: Qwen3Model,
    model_identity: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VibeVoiceAsrPreparedGeometry {
    pub(crate) input_samples: usize,
    pub(crate) input_sample_rate: u32,
    pub(crate) processed_samples: usize,
    pub(crate) encoder_samples: usize,
    pub(crate) acoustic_frames: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) embedding_elements: u64,
    pub(crate) preparation_workspace_bytes: u64,
    pub(crate) retained_device_bytes: u64,
    pub(crate) retained_host_bytes: u64,
}

impl VibeVoiceAsrPreparedGeometry {
    pub(crate) fn work_cost(self) -> WorkCost {
        WorkCost::new(
            self.acoustic_frames as u64,
            self.embedding_elements,
            self.preparation_workspace_bytes,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VibeVoiceAsrPreparationDecision {
    Retained(VibeVoiceAsrPreparedGeometry),
    LegacyInvocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VibeVoiceAsrPreparationStageSeal {
    pub(crate) backend: BackendKind,
    pub(crate) dtype: String,
    pub(crate) max_work_units: u64,
    pub(crate) max_workspace_bytes: u64,
}

/// Immutable, post-sampling VibeVoice decoder input.
///
/// Freezing the mixed embeddings here is semantically important: acoustic
/// latent sampling is stochastic, so decoder retries must never rebuild this
/// tensor from audio.
#[derive(Clone)]
pub(crate) struct VibeVoiceAsrPreparedArtifact {
    model_identity: [u8; 32],
    source_identity: [u8; 32],
    prompt_identity: [u8; 32],
    prompt_ids: Arc<[u32]>,
    acoustic_input_range: Range<usize>,
    tokenizer_state_projections: Arc<[ClockedStateProjection]>,
    mixed_embeddings: Tensor,
    geometry: VibeVoiceAsrPreparedGeometry,
}

impl VibeVoiceAsrPreparedArtifact {
    pub(crate) const fn geometry(&self) -> VibeVoiceAsrPreparedGeometry {
        self.geometry
    }

    pub(crate) const fn resident_tensor_bytes(&self) -> u64 {
        self.geometry.retained_device_bytes
    }

    pub(crate) const fn resident_host_bytes(&self) -> u64 {
        self.geometry.retained_host_bytes
    }

    pub(crate) fn resident_bytes(&self) -> Result<u64> {
        self.geometry
            .retained_device_bytes
            .checked_add(self.geometry.retained_host_bytes)
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR artifact bytes overflow".into()))
    }

    pub(crate) fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    pub(crate) fn acoustic_input_range(&self) -> Range<usize> {
        self.acoustic_input_range.clone()
    }

    /// Exact immutable mapping from decoder acoustic placeholders to the
    /// padded target-rate samples consumed by both causal tokenizer encoders.
    /// The frozen mixed embeddings remain authoritative until retained tensor
    /// span execution is wired end to end.
    pub(crate) fn tokenizer_state_projections(&self) -> &[ClockedStateProjection] {
        &self.tokenizer_state_projections
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        prompt_tokens: usize,
        acoustic_input_range: Range<usize>,
        encoder_samples: usize,
        hidden: usize,
    ) -> Result<Self> {
        if prompt_tokens == 0
            || hidden == 0
            || acoustic_input_range.end > prompt_tokens
            || acoustic_input_range.is_empty()
        {
            return Err(Error::InvalidInput(
                "VibeVoice ASR test artifact geometry is invalid".into(),
            ));
        }
        let acoustic_frames = acoustic_input_range.len();
        let mixed_embeddings = Tensor::zeros(
            (1, prompt_tokens, hidden),
            DType::F32,
            &candle_core::Device::Cpu,
        )?;
        let embedding_elements = u64::try_from(mixed_embeddings.elem_count())
            .map_err(|_| Error::Overloaded("test artifact elements exceed u64".into()))?;
        let tokenizer_state_projections = Arc::from([vibevoice_tokenizer_state_projection(
            acoustic_input_range.clone(),
            encoder_samples,
        )?]);
        Ok(Self {
            model_identity: [1; 32],
            source_identity: [2; 32],
            prompt_identity: [3; 32],
            prompt_ids: vec![0; prompt_tokens].into(),
            acoustic_input_range,
            tokenizer_state_projections,
            mixed_embeddings,
            geometry: VibeVoiceAsrPreparedGeometry {
                input_samples: encoder_samples,
                input_sample_rate: 24_000,
                processed_samples: encoder_samples,
                encoder_samples,
                acoustic_frames,
                prompt_tokens,
                embedding_elements,
                preparation_workspace_bytes: 1,
                retained_device_bytes: embedding_elements * 4,
                retained_host_bytes: (prompt_tokens * size_of::<u32>()) as u64,
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VibeVoiceAsrDecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

pub(crate) struct VibeVoiceAsrDecodeState {
    state_id: u64,
    model_identity: [u8; 32],
    prompt_tokens: usize,
    next_quantum_nonce: u64,
    active_quantum: Option<u64>,
    managed_completions_drained: bool,
    cache: PhysicalPagedKvCache,
    prepared: Option<Arc<VibeVoiceAsrPreparedArtifact>>,
    prefill_progress: usize,
    unconsumed_output: Option<Tensor>,
    staged_step: Option<VibeVoiceAsrDecodeStep>,
    pos: usize,
    pending_token: Option<u32>,
    generated: Vec<u32>,
    assembled: String,
    stop_tokens: Vec<u32>,
    stop_sequences: Vec<String>,
    max_new_tokens: usize,
    finished: bool,
    stop_reason: Option<&'static str>,
    stop_token_id: Option<u32>,
    stop_sequence: Option<String>,
}

pub(crate) struct VibeVoiceAsrDecodeCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    payload: Option<VibeVoiceAsrDecodeCheckpointPayload>,
}

struct VibeVoiceAsrDecodeCheckpointPayload {
    cache: PhysicalPagedKvCache,
    prepared: Option<Arc<VibeVoiceAsrPreparedArtifact>>,
    prefill_progress: usize,
    unconsumed_output: Option<Tensor>,
    staged_step: Option<VibeVoiceAsrDecodeStep>,
    pos: usize,
    pending_token: Option<u32>,
    generated: Vec<u32>,
    assembled: String,
    finished: bool,
    stop_reason: Option<&'static str>,
    stop_token_id: Option<u32>,
    stop_sequence: Option<String>,
    managed_completions_drained: bool,
}

impl VibeVoiceAsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::VibeVoiceAsr {
            return Err(Error::InvalidInput(format!(
                "VibeVoiceAsrModel cannot load non-ASR variant {variant}"
            )));
        }
        let config = VibeVoiceConfig::load(model_dir)?;
        if config.is_tts() {
            return Err(Error::ModelLoadError(
                "VibeVoice-ASR loader received a TTS config".to_string(),
            ));
        }
        let preprocessor = VibeVoicePreprocessorConfig::load(model_dir)?;
        let dtype = std::env::var("IZWI_VIBEVOICE_ASR_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .map(|raw| {
                device.select_model_dtype_checked(
                    ModelFamily::VibeVoiceAsr,
                    Some(raw),
                    "VibeVoice ASR",
                )
            })
            .transpose()?
            .unwrap_or_else(|| device.select_model_dtype(ModelFamily::VibeVoiceAsr, None));
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
        let language_model = Qwen3Model::load_with_layout(
            config.decoder_config.clone(),
            vb,
            Qwen3WeightLayout::VIBEVOICE,
        )?;
        let model_identity = vibevoice_asr_model_identity(
            model_dir,
            dtype,
            &config,
            next_vibevoice_asr_model_load_nonce()?,
        );
        info!(
            "Loaded VibeVoice-ASR from {:?} on {:?} with dtype {:?}",
            model_dir, device.kind, dtype
        );
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            preprocessor,
            tokenizer,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            language_model,
            model_identity,
        })
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<VibeVoicePhysicalStateSpec> {
        let tokenizer_domains = [
            VibeVoiceTokenizerStateDomain::new(
                VIBEVOICE_ASR_ACOUSTIC_DOMAIN,
                StateGroupId::new(2),
                StateClock::AudioSamples,
                self.acoustic_tokenizer.encoder_state_geometry(),
            )?,
            VibeVoiceTokenizerStateDomain::new(
                VIBEVOICE_ASR_SEMANTIC_DOMAIN,
                StateGroupId::new(2),
                StateClock::AudioSamples,
                self.semantic_tokenizer.encoder_state_geometry(),
            )?,
        ];
        let contract = vibevoice_invocation_contract(
            &self.language_model,
            self.dtype,
            default_kv_page_size(),
            &[CacheDomainId::new(1)],
            &tokenizer_domains,
        )?;
        let max_context_tokens = self
            .config
            .decoder_config
            .max_position_embeddings
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "VibeVoice-ASR decoder config has no maximum context length".into(),
                )
            })?;
        vibevoice_physical_state_spec(stage_graphs, contract, max_context_tokens)
    }

    /// Run the normal-duration audio front end once and freeze its stochastic
    /// acoustic sample into immutable decoder embeddings.
    pub(crate) fn retained_preparation_decision(
        &self,
        input_samples: usize,
        input_sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<VibeVoiceAsrPreparationDecision> {
        if input_samples == 0 {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input cannot be empty".into(),
            ));
        }
        let processed_samples = resampled_sample_count(
            input_samples,
            input_sample_rate,
            self.preprocessor.target_sample_rate(),
        )?;
        let ratio = self.preprocessor.speech_tok_compress_ratio.max(1);
        let acoustic_frames = asr_placeholder_count(processed_samples, ratio);
        let encoder_samples = acoustic_frames.checked_mul(ratio).ok_or_else(|| {
            Error::Overloaded("VibeVoice ASR padded encoder length overflow".into())
        })?;
        if encoder_samples
            > tokenizer_streaming_chunk_samples(
                self.preprocessor.target_sample_rate(),
                self.preprocessor.speech_tok_compress_ratio,
            )
        {
            return Ok(VibeVoiceAsrPreparationDecision::LegacyInvocation);
        }
        let audio_seconds =
            processed_samples as f32 / self.preprocessor.target_sample_rate() as f32;
        let extra = prompt_instruction(language, prompt);
        let prepared_prompt =
            self.tokenizer
                .build_asr_prompt(audio_seconds, acoustic_frames, extra.as_deref())?;
        let prompt_tokens = prepared_prompt.input_ids.len();
        if prompt_tokens
            > self
                .config
                .decoder_config
                .max_position_embeddings
                .unwrap_or(usize::MAX)
        {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR prepared prompt exceeds decoder context".into(),
            ));
        }
        let embedding_elements = u64::try_from(prompt_tokens)
            .ok()
            .and_then(|tokens| {
                tokens.checked_mul(u64::try_from(self.language_model.hidden_size()).ok()?)
            })
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR embedding elements overflow".into()))?;
        let retained_device_bytes = embedding_elements
            .checked_mul(u64::try_from(self.dtype.size_in_bytes()).map_err(|_| {
                Error::Overloaded("VibeVoice ASR embedding dtype size exceeds u64".into())
            })?)
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR artifact bytes overflow".into()))?;
        let retained_host_bytes = u64::try_from(prompt_tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(u64::try_from(size_of::<u32>()).ok()?))
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR prompt bytes overflow".into()))?;
        let preparation_workspace_bytes = self.scalar_preparation_workspace_bytes(
            encoder_samples,
            acoustic_frames,
            prompt_tokens,
        )?;
        Ok(VibeVoiceAsrPreparationDecision::Retained(
            VibeVoiceAsrPreparedGeometry {
                input_samples,
                input_sample_rate,
                processed_samples,
                encoder_samples,
                acoustic_frames,
                prompt_tokens,
                embedding_elements,
                preparation_workspace_bytes,
                retained_device_bytes,
                retained_host_bytes,
            },
        ))
    }

    fn scalar_preparation_workspace_bytes(
        &self,
        encoder_samples: usize,
        acoustic_frames: usize,
        prompt_tokens: usize,
    ) -> Result<u64> {
        let dtype_bytes = u64::try_from(self.dtype.size_in_bytes())
            .map_err(|_| Error::Overloaded("VibeVoice ASR dtype bytes exceed u64".into()))?;
        let acoustic = tokenizer_encoder_workspace_elements(
            &self.config.acoustic_tokenizer_config,
            encoder_samples,
        )?;
        let semantic = tokenizer_encoder_workspace_elements(
            &self.config.semantic_tokenizer_config,
            encoder_samples,
        )?;
        let hidden = u64::try_from(self.language_model.hidden_size())
            .map_err(|_| Error::Overloaded("VibeVoice ASR hidden size exceeds u64".into()))?;
        let held_acoustic = u64::try_from(acoustic_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(hidden))
            .ok_or_else(|| {
                Error::Overloaded("VibeVoice ASR acoustic feature workspace overflow".into())
            })?;
        let connector_peak = held_acoustic.checked_mul(8).ok_or_else(|| {
            Error::Overloaded("VibeVoice ASR connector workspace overflow".into())
        })?;
        let prompt_peak = u64::try_from(prompt_tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(hidden))
            .and_then(|elements| elements.checked_mul(3))
            .and_then(|elements| elements.checked_add(held_acoustic))
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR prompt workspace overflow".into()))?;
        let semantic_with_acoustic = semantic
            .checked_add(held_acoustic)
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR semantic workspace overflow".into()))?;
        acoustic
            .max(semantic_with_acoustic)
            .max(connector_peak)
            .max(prompt_peak)
            .checked_mul(dtype_bytes)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR preparation workspace overflow".into()))
    }

    pub(crate) fn scalar_preparation_stage_seal(
        &self,
        backend: BackendKind,
    ) -> Result<VibeVoiceAsrPreparationStageSeal> {
        let loaded = backend_kind_for_device(&self.device.device);
        if loaded != backend {
            return Err(Error::ModelLoadError(format!(
                "VibeVoice ASR preparation backend mismatch: model={loaded:?}, adapter={backend:?}"
            )));
        }
        let encoder_samples = tokenizer_streaming_chunk_samples(
            self.preprocessor.target_sample_rate(),
            self.preprocessor.speech_tok_compress_ratio,
        );
        let acoustic_frames =
            asr_placeholder_count(encoder_samples, self.preprocessor.speech_tok_compress_ratio);
        let prompt_tokens = self
            .config
            .decoder_config
            .max_position_embeddings
            .ok_or_else(|| {
                Error::ModelLoadError("VibeVoice ASR decoder context is unbounded".into())
            })?;
        Ok(VibeVoiceAsrPreparationStageSeal {
            backend,
            dtype: format!("{:?}", self.dtype).to_ascii_lowercase(),
            max_work_units: u64::try_from(acoustic_frames)
                .map_err(|_| Error::Overloaded("VibeVoice ASR work units exceed u64".into()))?,
            max_workspace_bytes: self.scalar_preparation_workspace_bytes(
                encoder_samples,
                acoustic_frames,
                prompt_tokens,
            )?,
        })
    }

    pub(crate) fn continuous_decode_workspace_per_row_bytes(&self) -> Result<u64> {
        u64::try_from(self.language_model.hidden_size())
            .ok()
            .and_then(|hidden| hidden.checked_mul(u64::try_from(self.dtype.size_in_bytes()).ok()?))
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR decode workspace overflow".into()))
    }

    pub(crate) fn prepare_retained_artifact(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<VibeVoiceAsrPreparedArtifact> {
        let expected_geometry =
            match self.retained_preparation_decision(audio.len(), sample_rate, language, prompt)? {
                VibeVoiceAsrPreparationDecision::Retained(geometry) => geometry,
                VibeVoiceAsrPreparationDecision::LegacyInvocation => {
                    return Err(Error::InvalidInput(
                        "VibeVoice ASR long audio requires the legacy invocation path".into(),
                    ));
                }
            };
        let (processed_audio, _) = preprocess_asr_audio(audio, sample_rate, &self.preprocessor)?;
        if processed_audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input produced no samples after preprocessing".into(),
            ));
        }
        let ratio = self.preprocessor.speech_tok_compress_ratio.max(1);
        let acoustic_frames = asr_placeholder_count(processed_audio.len(), ratio);
        let encoder_samples = acoustic_frames.checked_mul(ratio).ok_or_else(|| {
            Error::Overloaded("VibeVoice ASR padded encoder length overflow".into())
        })?;
        let mut encoder_audio = processed_audio.clone();
        encoder_audio.resize(encoder_samples, 0.0);
        let speech = Tensor::from_vec(encoder_audio, (1, 1, encoder_samples), &self.device.device)?
            .to_dtype(self.dtype)?;
        // Full encode intentionally avoids invocation tensor state. The
        // resulting mixed embeddings freeze the acoustic random draw.
        let speech_features = self.encode_speech_full(&speech)?;
        if speech_features.dim(1)? != acoustic_frames {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR tokenizer produced {} frames but preparation reserved {acoustic_frames}",
                speech_features.dim(1)?
            )));
        }
        let audio_seconds =
            processed_audio.len() as f32 / self.preprocessor.target_sample_rate() as f32;
        let extra = prompt_instruction(language, prompt);
        let prepared_prompt =
            self.tokenizer
                .build_asr_prompt(audio_seconds, acoustic_frames, extra.as_deref())?;
        if prepared_prompt.input_ids.len()
            > self
                .config
                .decoder_config
                .max_position_embeddings
                .unwrap_or(usize::MAX)
        {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR prepared prompt exceeds decoder context".into(),
            ));
        }
        let input_ids = Tensor::from_vec(
            prepared_prompt.input_ids.clone(),
            (1, prepared_prompt.input_ids.len()),
            &self.device.device,
        )?;
        let token_embeddings = self.language_model.embeddings(&input_ids)?;
        let mixed_embeddings = replace_range_with_features(
            &token_embeddings,
            prepared_prompt.acoustic_input_range.clone(),
            &speech_features.to_dtype(token_embeddings.dtype())?,
        )?;
        let embedding_elements = u64::try_from(mixed_embeddings.elem_count())
            .map_err(|_| Error::Overloaded("VibeVoice ASR embedding elements exceed u64".into()))?;
        let retained_device_bytes = embedding_elements
            .checked_mul(
                u64::try_from(mixed_embeddings.dtype().size_in_bytes()).map_err(|_| {
                    Error::Overloaded("VibeVoice ASR embedding dtype size exceeds u64".into())
                })?,
            )
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR artifact bytes overflow".into()))?;
        let retained_host_bytes = u64::try_from(prepared_prompt.input_ids.len())
            .ok()
            .and_then(|tokens| tokens.checked_mul(u64::try_from(size_of::<u32>()).ok()?))
            .ok_or_else(|| Error::Overloaded("VibeVoice ASR prompt bytes overflow".into()))?;
        let geometry = VibeVoiceAsrPreparedGeometry {
            input_samples: audio.len(),
            input_sample_rate: sample_rate,
            processed_samples: processed_audio.len(),
            encoder_samples,
            acoustic_frames,
            prompt_tokens: prepared_prompt.input_ids.len(),
            embedding_elements,
            preparation_workspace_bytes: expected_geometry.preparation_workspace_bytes,
            retained_device_bytes,
            retained_host_bytes,
        };
        if geometry != expected_geometry {
            return Err(Error::InferenceError(
                "VibeVoice ASR preparation geometry changed during artifact construction".into(),
            ));
        }
        let tokenizer_state_projections = Arc::from([vibevoice_tokenizer_state_projection(
            prepared_prompt.acoustic_input_range.clone(),
            encoder_samples,
        )?]);
        Ok(VibeVoiceAsrPreparedArtifact {
            model_identity: self.model_identity,
            source_identity: vibevoice_asr_source_identity(audio, sample_rate),
            prompt_identity: vibevoice_asr_prompt_identity(language, prompt),
            prompt_ids: prepared_prompt.input_ids.into(),
            acoustic_input_range: prepared_prompt.acoustic_input_range,
            tokenizer_state_projections,
            mixed_embeddings,
            geometry,
        })
    }

    pub(crate) fn validate_retained_artifact(
        &self,
        artifact: &VibeVoiceAsrPreparedArtifact,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<()> {
        validate_vibevoice_artifact_model_identity(artifact, self.model_identity)?;
        if artifact.source_identity != vibevoice_asr_source_identity(audio, sample_rate)
            || artifact.prompt_identity != vibevoice_asr_prompt_identity(language, prompt)
            || artifact.geometry.input_samples != audio.len()
            || artifact.geometry.input_sample_rate != sample_rate
            || artifact.geometry.prompt_tokens != artifact.prompt_ids.len()
            || artifact.acoustic_input_range.end > artifact.prompt_ids.len()
        {
            return Err(Error::InvalidInput(
                "VibeVoice ASR prepared artifact has mismatched model, source, or prompt identity"
                    .into(),
            ));
        }
        validate_vibevoice_artifact_storage(artifact)
    }

    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        artifact: Arc<VibeVoiceAsrPreparedArtifact>,
        options: VibeVoiceAsrGenerationOptions,
        cache: PhysicalPagedKvCache,
    ) -> Result<VibeVoiceAsrDecodeState> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "VibeVoice ASR resumable prefill requires empty physical KV".into(),
            ));
        }
        validate_vibevoice_artifact_model_identity(&artifact, self.model_identity)?;
        validate_vibevoice_artifact_storage(&artifact)?;
        let built_in_stop_tokens = [
            self.tokenizer.specials().im_end,
            self.tokenizer.specials().endoftext,
        ];
        Ok(VibeVoiceAsrDecodeState {
            state_id: next_vibevoice_asr_state_id()?,
            model_identity: self.model_identity,
            prompt_tokens: artifact.geometry.prompt_tokens,
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
            cache,
            prepared: Some(artifact),
            prefill_progress: 0,
            unconsumed_output: None,
            staged_step: None,
            pos: 0,
            pending_token: None,
            generated: Vec::new(),
            assembled: String::new(),
            stop_tokens: collect_stop_token_ids(&built_in_stop_tokens, &options.stop_token_ids),
            stop_sequences: sanitize_stop_sequences(&options.stop_sequences),
            max_new_tokens: options.max_new_tokens.max(1),
            finished: false,
            stop_reason: None,
            stop_token_id: None,
            stop_sequence: None,
        })
    }

    /// Commit one exact prompt span. Non-final spans retain only physical KV;
    /// decoder logits become visible only after the final span.
    pub(crate) fn continue_resumable_prefill(
        &self,
        state: &mut VibeVoiceAsrDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        let prompt_tokens = state.prefill_token_count();
        if state.prefill_progress != span_start
            || span_start >= span_end
            || span_end > prompt_tokens
            || state.finished
            || state.pending_token.is_some()
            || state.unconsumed_output.is_some()
            || state.staged_step.is_some()
            || !state.generated.is_empty()
            || state.pos != span_start
            || state.cache.context_len() != span_start
        {
            return Err(Error::InvalidInput(format!(
                "VibeVoice ASR prefill span [{span_start},{span_end}) is incompatible with cursor {} and prompt length {prompt_tokens}",
                state.prefill_progress
            )));
        }
        let prepared = state.prepared.as_ref().ok_or_else(|| {
            Error::InferenceError("VibeVoice ASR retained state has no prepared embeddings".into())
        })?;
        let span = prepared
            .mixed_embeddings
            .narrow(1, span_start, span_end - span_start)?;
        let complete = span_end == prompt_tokens;
        if complete {
            let logits = self.language_model.forward_managed_with_embeds(
                &span,
                span_start,
                &mut state.cache,
                None,
            )?;
            let next = argmax_last_logits(&logits, self.device.kind.is_cuda())?;
            let step = self.finish_decode_sample(state, next)?;
            state.staged_step = Some(step);
        } else {
            self.language_model
                .forward_managed_prefill_only_with_embeds(
                    &span,
                    span_start,
                    &mut state.cache,
                    None,
                )?;
        }
        if state.cache.context_len() != span_end {
            return Err(Error::InferenceError(
                "VibeVoice ASR physical prefill cursor did not match committed span".into(),
            ));
        }
        state.prefill_progress = span_end;
        state.pos = span_end;
        state.managed_completions_drained = false;
        if complete {
            state.prepared = None;
        }
        Ok(complete)
    }

    pub(crate) fn decode_step(
        &self,
        state: &mut VibeVoiceAsrDecodeState,
    ) -> Result<VibeVoiceAsrDecodeStep> {
        if state.staged_step.is_some() {
            return Err(Error::InferenceError(
                "VibeVoice ASR staged prefill output must be drained before decode".into(),
            ));
        }
        if state.finished {
            return Ok(vibevoice_terminal_step(state));
        }
        if let Some(pending) = state.pending_token.take() {
            let token = Tensor::from_vec(vec![pending], (1, 1), &self.device.device)?;
            state.unconsumed_output = Some(self.language_model.forward_managed(
                &token,
                state.pos,
                &mut state.cache,
            )?);
            state.pos = state
                .pos
                .checked_add(1)
                .ok_or_else(|| Error::InferenceError("VibeVoice ASR position overflow".into()))?;
            state.managed_completions_drained = false;
        }
        let next = take_vibevoice_quantum_argmax(
            &mut state.unconsumed_output,
            self.device.kind.is_cuda(),
        )?;
        self.finish_decode_sample(state, next)
    }

    fn finish_decode_sample(
        &self,
        state: &mut VibeVoiceAsrDecodeState,
        next: u32,
    ) -> Result<VibeVoiceAsrDecodeStep> {
        finish_vibevoice_decode_sample(
            state,
            next,
            [
                self.tokenizer.specials().im_end,
                self.tokenizer.specials().endoftext,
            ],
            |ids| self.tokenizer.decode(ids),
        )
    }

    /// Native one-token ragged decode. Every row remains independently
    /// checkpointable even though the Qwen layers execute as one tensor batch.
    pub(crate) fn decode_step_batch(
        &self,
        states: &mut [&mut VibeVoiceAsrDecodeState],
    ) -> Result<Vec<VibeVoiceAsrDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.model_identity != self.model_identity
                || state.finished
                || state.prefill_progress != state.prefill_token_count()
                || state.prepared.is_some()
                || state.unconsumed_output.is_some()
                || state.staged_step.is_some()
                || state.pending_token.is_none()
            {
                return Err(Error::InvalidInput(
                    "VibeVoice ASR decode batch requires one live pending token per completed retained row"
                        .into(),
                ));
            }
        }
        let output = forward_vibevoice_pending_decode_batch(
            &self.language_model,
            &self.device.device,
            states,
        )?;
        let sampled = vibevoice_batch_argmax(&output)?;
        let mut steps = Vec::with_capacity(states.len());
        for (state, next) in states.iter_mut().zip(sampled) {
            steps.push(self.finish_decode_sample(state, next)?);
        }
        Ok(steps)
    }

    pub(crate) const fn supports_resumable_prefill(&self) -> bool {
        true
    }

    pub(crate) const fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt_and_options(
            audio,
            sample_rate,
            language,
            prompt,
            VibeVoiceAsrGenerationOptions::default(),
        )
    }

    pub fn transcribe_with_details_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, prompt, options);
        Err(Error::InferenceError(
            "VibeVoice ASR requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt_and_options(
            audio,
            sample_rate,
            language,
            prompt,
            VibeVoiceAsrGenerationOptions::default(),
            on_delta,
        )
    }

    pub fn transcribe_with_callback_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let _ = (audio, sample_rate, language, prompt, options, on_delta);
        Err(Error::InferenceError(
            "VibeVoice ASR requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub(crate) fn transcribe_with_callback_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_internal(
                audio,
                sample_rate,
                language,
                prompt,
                options,
                leases,
                on_delta,
            )?
            .text)
    }

    pub(crate) fn transcribe_with_details_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            prompt,
            options,
            leases,
            &mut no_op,
        )
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(vibevoice_asr_max_audio_seconds_hint(self.device.kind))
    }

    fn transcribe_internal(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: VibeVoiceAsrGenerationOptions,
        leases: &mut InvocationWorkspaceLeaseSetV2,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<VibeVoiceAsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input cannot be empty".to_string(),
            ));
        }
        let domains = leases.domains().collect::<Vec<_>>();
        if domains
            != vec![
                crate::models::architectures::vibevoice::VIBEVOICE_ASR_DECODER_DOMAIN,
                VIBEVOICE_ASR_ACOUSTIC_DOMAIN,
                VIBEVOICE_ASR_SEMANTIC_DOMAIN,
            ]
            || leases
                .lease(crate::models::architectures::vibevoice::VIBEVOICE_ASR_DECODER_DOMAIN)?
                .kind()
                != InvocationStateBackingKindV2::PagedAttention
            || leases.lease(VIBEVOICE_ASR_ACOUSTIC_DOMAIN)?.kind()
                != InvocationStateBackingKindV2::Tensor
            || leases.lease(VIBEVOICE_ASR_SEMANTIC_DOMAIN)?.kind()
                != InvocationStateBackingKindV2::Tensor
        {
            return Err(Error::InferenceError(
                "VibeVoice ASR requires exact decoder pages and acoustic/semantic tensor state"
                    .into(),
            ));
        }
        let total_started = Instant::now();
        let preprocess_started = Instant::now();
        let (processed_audio, preprocess_stats) =
            preprocess_asr_audio(audio, sample_rate, &self.preprocessor)?;
        let preprocess_ms = elapsed_ms(preprocess_started);
        if processed_audio.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR audio input produced no samples after preprocessing".to_string(),
            ));
        }
        let target_sample_rate = self.preprocessor.target_sample_rate();
        let audio_seconds = processed_audio.len() as f32 / target_sample_rate as f32;
        let compress_ratio = self.preprocessor.speech_tok_compress_ratio.max(1);
        let expected_acoustic_frames = asr_placeholder_count(processed_audio.len(), compress_ratio);
        let mut encoder_audio = processed_audio.clone();
        let encoder_samples = expected_acoustic_frames.saturating_mul(compress_ratio);
        if encoder_audio.len() < encoder_samples {
            encoder_audio.resize(encoder_samples, 0.0);
        }
        let speech = Tensor::from_vec(encoder_audio, (1, 1, encoder_samples), &self.device.device)?
            .to_dtype(self.dtype)?;
        let audio_encode_started = Instant::now();
        let (speech_features, encode_stats) = {
            let (acoustic, semantic) = leases
                .lease_pair_mut(VIBEVOICE_ASR_ACOUSTIC_DOMAIN, VIBEVOICE_ASR_SEMANTIC_DOMAIN)?;
            self.encode_speech(
                &speech,
                acoustic.typed_mut::<InvocationTensorLease>()?,
                semantic.typed_mut::<InvocationTensorLease>()?,
            )?
        };
        let physical_cache = leases
            .lease_mut(crate::models::architectures::vibevoice::VIBEVOICE_ASR_DECODER_DOMAIN)?
            .paged_cache_mut()?;
        let audio_encode_ms = elapsed_ms(audio_encode_started);
        let acoustic_frames = speech_features.dim(1)?;
        if acoustic_frames != expected_acoustic_frames {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR tokenizer produced {acoustic_frames} frames but processor reserved {expected_acoustic_frames}"
            )));
        }
        let extra = prompt_instruction(language, prompt);
        let prompt = self.tokenizer.build_asr_prompt(
            audio_seconds,
            expected_acoustic_frames,
            extra.as_deref(),
        )?;
        let input_ids = Tensor::from_vec(
            prompt.input_ids.clone(),
            (1, prompt.input_ids.len()),
            &self.device.device,
        )?;
        let input_embeds = self.language_model.embeddings(&input_ids)?;
        let input_embeds = replace_range_with_features(
            &input_embeds,
            prompt.acoustic_input_range.clone(),
            &speech_features.to_dtype(input_embeds.dtype())?,
        )?;

        if physical_cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "VibeVoice-ASR invocation cache must start empty".into(),
            ));
        }
        let decode_cache_dense_max_tokens = 0;
        let cuda_device_argmax = self.device.kind.is_cuda();
        let prefill_started = Instant::now();
        let logits = self.language_model.forward_managed_with_embeds(
            &input_embeds,
            0,
            physical_cache,
            None,
        )?;
        let prefill_ms = elapsed_ms(prefill_started);
        let mut pos = prompt.input_ids.len();
        let mut next = argmax_last_logits(&logits, cuda_device_argmax)?;
        let mut generated = Vec::new();
        let mut assembled = String::new();
        let built_in_stop_tokens = [
            self.tokenizer.specials().im_end,
            self.tokenizer.specials().endoftext,
        ];
        let stop_tokens = collect_stop_token_ids(&built_in_stop_tokens, &options.stop_token_ids);
        let stop_sequences = sanitize_stop_sequences(&options.stop_sequences);
        let max_new_tokens = options.max_new_tokens.max(1);
        let mut stop_reason = None::<&'static str>;
        let mut stop_token_id = None::<u32>;
        let mut stop_sequence = None::<String>;
        let decode_started = Instant::now();

        for _ in 0..max_new_tokens {
            if stop_tokens.contains(&next) {
                stop_reason = Some(if built_in_stop_tokens.contains(&next) {
                    "model_stop_token"
                } else {
                    "request_stop_token"
                });
                stop_token_id = Some(next);
                break;
            }
            generated.push(next);
            let decoded = self.tokenizer.decode(&generated)?;
            let (visible_decoded, matched_stop_sequence) =
                truncate_at_stop_sequence(&decoded, &stop_sequences);
            if visible_decoded.len() > assembled.len() {
                on_delta(&visible_decoded[assembled.len()..]);
            }
            assembled = visible_decoded;
            if let Some(sequence) = matched_stop_sequence {
                stop_reason = Some("stop_sequence");
                stop_sequence = Some(sequence);
                break;
            }

            let token = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            let logits = self
                .language_model
                .forward_managed(&token, pos, physical_cache)?;
            pos += 1;
            next = argmax_last_logits(&logits, cuda_device_argmax)?;
        }
        let decode_ms = elapsed_ms(decode_started);
        let reached_max_tokens = stop_reason.is_none() && generated.len() >= max_new_tokens;
        if reached_max_tokens {
            stop_reason = Some("max_tokens");
        }

        let parsed = parse_vibevoice_asr_output(&assembled);
        Ok(VibeVoiceAsrTranscriptionOutput {
            text: parsed.text.clone(),
            language: language.map(ToOwned::to_owned),
            diagnostics: Some(json!({
                "model_family": "vibevoice_asr",
                "model_dir": self.model_dir.display().to_string(),
                "audio": {
                    "input_sample_rate": sample_rate,
                    "input_samples": audio.len(),
                    "resampled_sample_rate": target_sample_rate,
                    "resampled_samples": processed_audio.len(),
                    "encoder_samples": encoder_samples,
                    "duration_seconds": audio_seconds,
                    "acoustic_frames": acoustic_frames,
                    "expected_acoustic_frames": expected_acoustic_frames,
                    "speech_tok_compress_ratio": compress_ratio,
                    "normalized": preprocess_stats.normalized,
                    "target_db_fs": preprocess_stats.target_db_fs,
                    "rms_before_normalization": preprocess_stats.rms_before,
                    "normalization_gain": preprocess_stats.gain,
                    "clipping_divisor": preprocess_stats.clipping_divisor,
                    "tokenizer_streaming": encode_stats.streaming,
                    "tokenizer_chunks": encode_stats.chunks,
                    "tokenizer_chunk_samples": encode_stats.chunk_samples,
                },
                "prompt": {
                    "tokens": prompt.prompt_token_count,
                    "acoustic_input_tokens": prompt.acoustic_input_range.end.saturating_sub(prompt.acoustic_input_range.start),
                    "language": language,
                    "extra_prompt": extra,
                },
                "decode": {
                    "generated_tokens": generated.len(),
                    "max_new_tokens": max_new_tokens,
                    "stop_reason": stop_reason,
                    "stop_token_id": stop_token_id,
                    "stop_sequence": stop_sequence,
                    "reached_max_tokens": reached_max_tokens,
                    "configured_stop_token_ids": options.stop_token_ids,
                    "configured_stop_sequences": stop_sequences,
                },
                "output": {
                    "format": parsed.format,
                    "raw_text": parsed.raw_text,
                    "segment_count": parsed.segments.len(),
                    "segments": parsed.segments,
                },
                "execution": {
                    "dtype": format!("{:?}", self.dtype),
                    "device_kind": format!("{:?}", self.device.kind),
                    "decoder_layers": self.config.decoder_config.num_hidden_layers,
                    "cuda_dense_decode_cache": decode_cache_dense_max_tokens > 0,
                    "dense_decode_max_tokens": decode_cache_dense_max_tokens,
                    "cuda_device_argmax": cuda_device_argmax,
                },
                "timings_ms": {
                    "preprocess": preprocess_ms,
                    "audio_encode": audio_encode_ms,
                    "prefill": prefill_ms,
                    "decode": decode_ms,
                    "model_total": elapsed_ms(total_started),
                }
            })),
        })
    }

    fn encode_speech(
        &self,
        speech: &Tensor,
        acoustic_state: &mut InvocationTensorLease,
        semantic_state: &mut InvocationTensorLease,
    ) -> Result<(Tensor, VibeVoiceAsrEncodeStats)> {
        let total_samples = speech.dim(2)?;
        let chunk_samples = tokenizer_streaming_chunk_samples(
            self.preprocessor.target_sample_rate(),
            self.preprocessor.speech_tok_compress_ratio,
        );
        let can_stream = total_samples > chunk_samples
            && self.config.acoustic_tokenizer_config.causal
            && self.config.semantic_tokenizer_config.causal;
        if can_stream {
            return self.encode_speech_streaming(
                speech,
                chunk_samples,
                acoustic_state,
                semantic_state,
            );
        }

        Ok((
            self.encode_speech_full(speech)?,
            VibeVoiceAsrEncodeStats {
                streaming: false,
                chunks: 1,
                chunk_samples: total_samples,
            },
        ))
    }

    fn encode_speech_full(&self, speech: &Tensor) -> Result<Tensor> {
        let acoustic = self.acoustic_tokenizer.encode(speech)?;
        let acoustic = self.acoustic_tokenizer.sample(&acoustic)?;
        let acoustic = self.acoustic_connector.forward(&acoustic)?;

        let semantic = self.semantic_tokenizer.encode(speech)?.mode();
        let semantic = self.semantic_connector.forward(&semantic)?;

        self.combine_speech_features(acoustic, semantic)
    }

    fn encode_speech_streaming(
        &self,
        speech: &Tensor,
        chunk_samples: usize,
        acoustic_state: &mut InvocationTensorLease,
        semantic_state: &mut InvocationTensorLease,
    ) -> Result<(Tensor, VibeVoiceAsrEncodeStats)> {
        let total_samples = speech.dim(2)?;
        let ranges = tokenizer_chunk_ranges(total_samples, chunk_samples);
        let mut acoustic_means = Vec::with_capacity(ranges.len());
        let mut semantic_means = Vec::with_capacity(ranges.len());
        let mut acoustic_std = None;

        for (start, len) in &ranges {
            let chunk = speech.narrow(2, *start, *len)?;
            let advance = u64::try_from(*len).map_err(|_| {
                Error::InferenceError("VibeVoice ASR chunk length exceeds u64".into())
            })?;
            let acoustic = self.acoustic_tokenizer.encode_streaming_physical(
                &chunk,
                VIBEVOICE_ASR_ACOUSTIC_DOMAIN,
                advance,
                acoustic_state,
            )?;
            acoustic_std = acoustic_std.or(acoustic.std);
            acoustic_means.push(acoustic.mean);

            let semantic = self.semantic_tokenizer.encode_streaming_physical(
                &chunk,
                VIBEVOICE_ASR_SEMANTIC_DOMAIN,
                advance,
                semantic_state,
            )?;
            semantic_means.push(semantic.mean);
        }

        let acoustic = VibeVoiceTokenizerEncoderOutput {
            mean: Tensor::cat(&acoustic_means, 1)?,
            std: acoustic_std,
        };
        let acoustic = self.acoustic_tokenizer.sample(&acoustic)?;
        let acoustic = self.acoustic_connector.forward(&acoustic)?;

        let semantic = Tensor::cat(&semantic_means, 1)?;
        let semantic = self.semantic_connector.forward(&semantic)?;
        let features = self.combine_speech_features(acoustic, semantic)?;
        Ok((
            features,
            VibeVoiceAsrEncodeStats {
                streaming: true,
                chunks: ranges.len(),
                chunk_samples,
            },
        ))
    }

    fn combine_speech_features(&self, acoustic: Tensor, semantic: Tensor) -> Result<Tensor> {
        if acoustic.dims() != semantic.dims() {
            return Err(Error::InferenceError(format!(
                "VibeVoice-ASR acoustic/semantic feature shape mismatch: {:?} vs {:?}",
                acoustic.dims(),
                semantic.dims()
            )));
        }
        acoustic.broadcast_add(&semantic).map_err(Error::from)
    }
}

impl VibeVoiceAsrDecodeState {
    pub(crate) const fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) const fn prefill_token_count(&self) -> usize {
        self.prompt_tokens
    }

    pub(crate) const fn sequence_position(&self) -> usize {
        self.pos
    }

    pub(crate) const fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn take_staged_decode_step(&mut self) -> Option<VibeVoiceAsrDecodeStep> {
        self.staged_step.take()
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let completions = self.cache.take_completed_writes();
        self.managed_completions_drained = true;
        completions
    }

    pub(crate) fn install_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        let mut checkpoint = self.begin_managed_quantum(cache)?;
        self.commit_managed_quantum(&mut checkpoint)
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<VibeVoiceAsrDecodeCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a VibeVoice ASR managed quantum is already active".into(),
            ));
        }
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "VibeVoice ASR managed completions must be drained before the next quantum".into(),
            ));
        }
        if self.staged_step.is_some() {
            return Err(Error::InferenceError(
                "VibeVoice ASR staged output must be drained before the next quantum".into(),
            ));
        }
        if self.cache.arena().id() != cache.arena().id()
            || self.cache.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a VibeVoice ASR session cannot switch managed KV authority".into(),
            ));
        }
        if cache.context_len() != self.pos {
            return Err(Error::InferenceError(format!(
                "managed VibeVoice ASR reservation starts at {}, but state is at {}",
                cache.context_len(),
                self.pos
            )));
        }
        let quantum_nonce = self.next_quantum_nonce;
        self.next_quantum_nonce = self
            .next_quantum_nonce
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("VibeVoice ASR quantum nonce overflow".into()))?;
        self.active_quantum = Some(quantum_nonce);
        Ok(VibeVoiceAsrDecodeCheckpoint {
            state_id: self.state_id,
            quantum_nonce,
            payload: Some(VibeVoiceAsrDecodeCheckpointPayload {
                cache: std::mem::replace(&mut self.cache, cache),
                prepared: self.prepared.clone(),
                prefill_progress: self.prefill_progress,
                unconsumed_output: self.unconsumed_output.clone(),
                staged_step: self.staged_step.clone(),
                pos: self.pos,
                pending_token: self.pending_token,
                generated: self.generated.clone(),
                assembled: self.assembled.clone(),
                finished: self.finished,
                stop_reason: self.stop_reason,
                stop_token_id: self.stop_token_id,
                stop_sequence: self.stop_sequence.clone(),
                managed_completions_drained: self.managed_completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut VibeVoiceAsrDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_active_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("VibeVoice ASR checkpoint was already consumed".into())
        })?;
        self.active_quantum = None;
        drop(payload);
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut VibeVoiceAsrDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_active_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("VibeVoice ASR checkpoint was already consumed".into())
        })?;
        self.cache = payload.cache;
        self.prepared = payload.prepared;
        self.prefill_progress = payload.prefill_progress;
        self.unconsumed_output = payload.unconsumed_output;
        self.staged_step = payload.staged_step;
        self.pos = payload.pos;
        self.pending_token = payload.pending_token;
        self.generated = payload.generated;
        self.assembled = payload.assembled;
        self.finished = payload.finished;
        self.stop_reason = payload.stop_reason;
        self.stop_token_id = payload.stop_token_id;
        self.stop_sequence = payload.stop_sequence;
        self.managed_completions_drained = payload.managed_completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    fn validate_active_checkpoint(&self, checkpoint: &VibeVoiceAsrDecodeCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum_nonce)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "VibeVoice ASR checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }
}

fn vibevoice_terminal_step(state: &VibeVoiceAsrDecodeState) -> VibeVoiceAsrDecodeStep {
    VibeVoiceAsrDecodeStep {
        delta: String::new(),
        text: parse_vibevoice_asr_output(&state.assembled).text,
        tokens_generated: state.generated.len(),
        finished: true,
    }
}

fn finish_vibevoice_decode_sample(
    state: &mut VibeVoiceAsrDecodeState,
    next: u32,
    built_in_stop_tokens: [u32; 2],
    decode: impl FnOnce(&[u32]) -> Result<String>,
) -> Result<VibeVoiceAsrDecodeStep> {
    if state.stop_tokens.contains(&next) {
        state.finished = true;
        state.stop_reason = Some(if built_in_stop_tokens.contains(&next) {
            "model_stop_token"
        } else {
            "request_stop_token"
        });
        state.stop_token_id = Some(next);
        return Ok(vibevoice_terminal_step(state));
    }

    state.generated.push(next);
    let decoded = decode(&state.generated)?;
    let (visible, matched) = truncate_at_stop_sequence(&decoded, &state.stop_sequences);
    let delta = if let Some(delta) = visible.strip_prefix(&state.assembled) {
        delta.to_string()
    } else {
        visible.clone()
    };
    state.assembled = visible;
    if let Some(sequence) = matched {
        state.finished = true;
        state.stop_reason = Some("stop_sequence");
        state.stop_sequence = Some(sequence);
    } else if state.generated.len() >= state.max_new_tokens {
        state.finished = true;
        state.stop_reason = Some("max_tokens");
    } else {
        state.pending_token = Some(next);
    }
    Ok(VibeVoiceAsrDecodeStep {
        delta,
        text: if state.finished {
            parse_vibevoice_asr_output(&state.assembled).text
        } else {
            state.assembled.clone()
        },
        tokens_generated: state.generated.len(),
        finished: state.finished,
    })
}

fn take_vibevoice_quantum_argmax(output: &mut Option<Tensor>, on_device: bool) -> Result<u32> {
    let output = output.take().ok_or_else(|| {
        Error::InferenceError("VibeVoice ASR decode quantum has no model output".into())
    })?;
    argmax_last_logits(&output, on_device)
}

fn forward_vibevoice_pending_decode_batch(
    model: &Qwen3Model,
    device: &candle_core::Device,
    states: &mut [&mut VibeVoiceAsrDecodeState],
) -> Result<Tensor> {
    if states.is_empty() {
        return Err(Error::InvalidInput(
            "VibeVoice ASR physical decode batch is empty".into(),
        ));
    }
    let pending = states
        .iter()
        .map(|state| {
            state.pending_token.ok_or_else(|| {
                Error::InvalidInput("VibeVoice ASR decode row has no pending token".into())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let positions = states
        .iter()
        .map(|state| {
            state
                .pos
                .checked_add(1)
                .ok_or_else(|| Error::InferenceError("VibeVoice ASR position overflow".into()))?;
            Ok(state.pos)
        })
        .collect::<Result<Vec<_>>>()?;
    let input = Tensor::from_vec(pending, (states.len(), 1), device)?;
    let output = if states.len() == 1 {
        model.forward_managed(&input, positions[0], &mut states[0].cache)?
    } else {
        let mut caches = states
            .iter_mut()
            .map(|state| &mut state.cache)
            .collect::<Vec<_>>();
        model.forward_managed_decode_batch(&input, &positions, &mut caches)?
    };
    for state in states {
        state.pending_token = None;
        state.pos += 1;
        state.managed_completions_drained = false;
    }
    Ok(output)
}

fn vibevoice_batch_argmax(output: &Tensor) -> Result<Vec<u32>> {
    let (batch, sequence, _width) = output.dims3()?;
    if batch == 0 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "VibeVoice ASR batch logits have invalid shape {:?}",
            output.dims()
        )));
    }
    output
        .narrow(1, sequence - 1, 1)?
        .squeeze(1)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()
        .map_err(Error::from)
}

/// Conservative peak live-set ceiling for one tokenizer encoder. Sequential
/// stages are compared rather than summed, and each stage uses its exact
/// downsampled frame extent. The factor covers the residual, normalization,
/// mixer, and both simultaneously-live 4x FFN tensors in one block.
fn tokenizer_encoder_workspace_elements(
    config: &VibeVoiceTokenizerConfig,
    encoder_samples: usize,
) -> Result<u64> {
    let depths = config.encoder_depths_vec()?;
    if depths.len() != config.encoder_ratios.len() + 1 {
        return Err(Error::ModelLoadError(
            "VibeVoice tokenizer workspace topology is inconsistent".into(),
        ));
    }
    let mut frames = encoder_samples;
    let mut channels = config.encoder_n_filters;
    let mut peak = checked_tensor_elements(frames, channels, 12)?;
    let mut ratios = config.encoder_ratios.clone();
    ratios.reverse();
    for (stage, ratio) in ratios.into_iter().enumerate() {
        if ratio == 0 {
            return Err(Error::ModelLoadError(
                "VibeVoice tokenizer ratio must be non-zero".into(),
            ));
        }
        let padded_frames = frames
            .checked_add(ratio.checked_mul(2).ok_or_else(|| {
                Error::Overloaded("VibeVoice tokenizer kernel width overflow".into())
            })?)
            .ok_or_else(|| Error::Overloaded("VibeVoice tokenizer frame overflow".into()))?;
        let next_frames = frames.saturating_add(ratio - 1) / ratio;
        let next_channels = config
            .encoder_n_filters
            .checked_mul(
                1usize
                    .checked_shl(u32::try_from(stage + 1).map_err(|_| {
                        Error::Overloaded("VibeVoice tokenizer stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::Overloaded("VibeVoice tokenizer channel width overflow".into())
                    })?,
            )
            .ok_or_else(|| {
                Error::Overloaded("VibeVoice tokenizer channel width overflow".into())
            })?;
        let downsample_peak = checked_tensor_elements(padded_frames, channels, 2)?
            .checked_add(checked_tensor_elements(next_frames, next_channels, 2)?)
            .ok_or_else(|| {
                Error::Overloaded("VibeVoice tokenizer downsample workspace overflow".into())
            })?;
        peak =
            peak.max(downsample_peak)
                .max(checked_tensor_elements(next_frames, next_channels, 12)?);
        frames = next_frames;
        channels = next_channels;
    }
    peak = peak.max(checked_tensor_elements(
        frames,
        config.vae_dim.max(channels),
        3,
    )?);
    Ok(peak)
}

fn checked_tensor_elements(frames: usize, channels: usize, live_factor: u64) -> Result<u64> {
    u64::try_from(frames)
        .ok()
        .and_then(|frames| frames.checked_mul(u64::try_from(channels).ok()?))
        .and_then(|elements| elements.checked_mul(live_factor))
        .ok_or_else(|| Error::Overloaded("VibeVoice tokenizer workspace elements overflow".into()))
}

fn validate_vibevoice_artifact_storage(artifact: &VibeVoiceAsrPreparedArtifact) -> Result<()> {
    let [batch, tokens, _hidden] = artifact.mixed_embeddings.dims() else {
        return Err(Error::InvalidInput(
            "VibeVoice ASR prepared embeddings are not rank three".into(),
        ));
    };
    let elements = u64::try_from(artifact.mixed_embeddings.elem_count())
        .map_err(|_| Error::Overloaded("VibeVoice ASR embedding elements exceed u64".into()))?;
    let bytes = elements
        .checked_mul(
            u64::try_from(artifact.mixed_embeddings.dtype().size_in_bytes()).map_err(|_| {
                Error::Overloaded("VibeVoice ASR embedding dtype size exceeds u64".into())
            })?,
        )
        .ok_or_else(|| Error::Overloaded("VibeVoice ASR artifact bytes overflow".into()))?;
    let expected_tokenizer_projection = vibevoice_tokenizer_state_projection(
        artifact.acoustic_input_range.clone(),
        artifact.geometry.encoder_samples,
    )?;
    if *batch != 1
        || *tokens != artifact.geometry.prompt_tokens
        || artifact.prompt_ids.len() != artifact.geometry.prompt_tokens
        || artifact.acoustic_input_range.end > *tokens
        || artifact.acoustic_input_range.end - artifact.acoustic_input_range.start
            != artifact.geometry.acoustic_frames
        || elements != artifact.geometry.embedding_elements
        || bytes != artifact.geometry.retained_device_bytes
        || artifact.tokenizer_state_projections.as_ref() != [expected_tokenizer_projection]
        || artifact.geometry.retained_host_bytes
            != u64::try_from(artifact.prompt_ids.len())
                .ok()
                .and_then(|tokens| tokens.checked_mul(u64::try_from(size_of::<u32>()).ok()?))
                .ok_or_else(|| Error::Overloaded("VibeVoice ASR prompt bytes overflow".into()))?
    {
        return Err(Error::InvalidInput(
            "VibeVoice ASR prepared artifact geometry or byte accounting is stale".into(),
        ));
    }
    Ok(())
}

fn validate_vibevoice_artifact_model_identity(
    artifact: &VibeVoiceAsrPreparedArtifact,
    expected: [u8; 32],
) -> Result<()> {
    if artifact.model_identity != expected {
        return Err(Error::InvalidInput(
            "VibeVoice ASR artifact belongs to a different loaded model instance".into(),
        ));
    }
    Ok(())
}

fn next_vibevoice_asr_state_id() -> Result<u64> {
    NEXT_VIBEVOICE_ASR_STATE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::InferenceError("VibeVoice ASR state identity overflow".into()))
}

fn next_vibevoice_asr_model_load_nonce() -> Result<u64> {
    NEXT_VIBEVOICE_ASR_MODEL_LOAD_NONCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::ModelLoadError("VibeVoice ASR load nonce overflow".into()))
}

fn vibevoice_asr_model_identity(
    model_dir: &Path,
    dtype: DType,
    config: &VibeVoiceConfig,
    load_nonce: u64,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"izwi-vibevoice-asr-model-v1");
    hasher.update(load_nonce.to_le_bytes());
    hasher.update(model_dir.as_os_str().as_encoded_bytes());
    hasher.update(format!("{dtype:?}:{config:?}").as_bytes());
    nonzero_sha256(hasher)
}

fn vibevoice_asr_source_identity(audio: &[f32], sample_rate: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"izwi-vibevoice-asr-source-v1");
    hasher.update(sample_rate.to_le_bytes());
    hasher.update((audio.len() as u64).to_le_bytes());
    for sample in audio {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    nonzero_sha256(hasher)
}

fn vibevoice_asr_prompt_identity(language: Option<&str>, prompt: Option<&str>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"izwi-vibevoice-asr-prompt-v1");
    for value in [language, prompt] {
        let value = value.unwrap_or_default().as_bytes();
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    }
    nonzero_sha256(hasher)
}

fn nonzero_sha256(hasher: Sha256) -> [u8; 32] {
    let mut identity: [u8; 32] = hasher.finalize().into();
    if identity.iter().all(|byte| *byte == 0) {
        identity[0] = 1;
    }
    identity
}

fn prompt_instruction(language: Option<&str>, prompt: Option<&str>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(language) = language.filter(|value| {
        let value = value.trim();
        !value.is_empty() && !value.eq_ignore_ascii_case("auto")
    }) {
        parts.push(format!("The spoken language is {}.", language.trim()));
    }
    if let Some(prompt) = prompt.filter(|value| !value.trim().is_empty()) {
        parts.push(prompt.trim().to_string());
    }
    (!parts.is_empty()).then(|| parts.join(" "))
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
            "VibeVoice prompt reserved {} acoustic tokens but encoder produced {feature_len}",
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

fn argmax_last_logits(logits: &Tensor, use_device_argmax: bool) -> Result<u32> {
    let seq_len = logits.dim(1)?;
    let row = logits.i((0, seq_len - 1))?;
    if use_device_argmax {
        return argmax_logits_row_device(&row);
    }
    argmax_logits_row_host(&row)
}

fn argmax_logits_row_host(row: &Tensor) -> Result<u32> {
    let row = row.to_dtype(DType::F32)?;
    let values = row.to_vec1::<f32>()?;
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| Error::InferenceError("VibeVoice-ASR logits row was empty".to_string()))
}

fn argmax_logits_row_device(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched VibeVoice-ASR logits row: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected VibeVoice-ASR logits row rank for argmax: {rank}"
            )));
        }
    };
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn collect_stop_token_ids(built_in: &[u32], requested: &[u32]) -> Vec<u32> {
    let mut stop_tokens = Vec::with_capacity(built_in.len() + requested.len());
    for token in built_in.iter().chain(requested.iter()).copied() {
        if !stop_tokens.contains(&token) {
            stop_tokens.push(token);
        }
    }
    stop_tokens
}

fn sanitize_stop_sequences(sequences: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for sequence in sequences {
        let trimmed = sequence.trim();
        if !trimmed.is_empty() && !out.iter().any(|existing: &String| existing == trimmed) {
            out.push(trimmed.to_string());
        }
    }
    out
}

fn truncate_at_stop_sequence(text: &str, stop_sequences: &[String]) -> (String, Option<String>) {
    let mut earliest: Option<(usize, &str)> = None;
    for sequence in stop_sequences {
        let Some(idx) = text.find(sequence) else {
            continue;
        };
        if earliest
            .map(|(existing_idx, _)| idx < existing_idx)
            .unwrap_or(true)
        {
            earliest = Some((idx, sequence.as_str()));
        }
    }

    if let Some((idx, sequence)) = earliest {
        (text[..idx].to_string(), Some(sequence.to_string()))
    } else {
        (text.to_string(), None)
    }
}

fn parse_vibevoice_asr_output(raw: &str) -> VibeVoiceAsrParsedOutput {
    let raw_text = cleanup_transcript_text(raw);
    for candidate in json_output_candidates(&raw_text) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) else {
            continue;
        };
        let segments = vibevoice_segments_from_value(&value);
        if !segments.is_empty() {
            return VibeVoiceAsrParsedOutput {
                text: join_segment_contents(&segments),
                raw_text,
                format: "segments",
                segments,
            };
        }
    }

    VibeVoiceAsrParsedOutput {
        text: raw_text.clone(),
        raw_text,
        format: "text",
        segments: Vec::new(),
    }
}

fn json_output_candidates(text: &str) -> Vec<String> {
    let stripped = strip_json_code_fence(text.trim());
    let mut candidates = Vec::new();
    push_unique_candidate(&mut candidates, stripped);

    if let Some(candidate) = balanced_json_slice(stripped, '[', ']') {
        push_unique_candidate(&mut candidates, candidate);
    }
    if let Some(candidate) = balanced_json_slice(stripped, '{', '}') {
        push_unique_candidate(&mut candidates, candidate);
    }

    candidates
}

fn strip_json_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let rest = rest
        .strip_prefix("json")
        .or_else(|| rest.strip_prefix("JSON"))
        .unwrap_or(rest)
        .trim_start_matches(|ch: char| ch.is_whitespace());
    rest.rsplit_once("```")
        .map(|(body, _)| body.trim())
        .unwrap_or(trimmed)
}

fn balanced_json_slice(text: &str, open: char, close: char) -> Option<&str> {
    let start = text.find(open)?;
    let end = text.rfind(close)?;
    (end >= start).then(|| text[start..=end].trim())
}

fn push_unique_candidate(candidates: &mut Vec<String>, candidate: &str) {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return;
    }
    if !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
    }
}

fn vibevoice_segments_from_value(value: &serde_json::Value) -> Vec<VibeVoiceAsrSegment> {
    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(vibevoice_segment_from_value)
            .collect(),
        serde_json::Value::Object(map) => {
            if let Some(segment) = vibevoice_segment_from_map(map) {
                return vec![segment];
            }
            for key in [
                "segments",
                "transcription",
                "transcript",
                "results",
                "utterances",
            ] {
                if let Some(segments) = get_value_case_insensitive(map, key)
                    .map(vibevoice_segments_from_value)
                    .filter(|segments| !segments.is_empty())
                {
                    return segments;
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn vibevoice_segment_from_value(value: &serde_json::Value) -> Option<VibeVoiceAsrSegment> {
    let serde_json::Value::Object(map) = value else {
        return None;
    };
    vibevoice_segment_from_map(map)
}

fn vibevoice_segment_from_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<VibeVoiceAsrSegment> {
    let content = ["Content", "content", "Text", "text", "transcript"]
        .iter()
        .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_string))
        .unwrap_or_default();
    let content = content.trim().to_string();
    if content.is_empty() {
        return None;
    }

    Some(VibeVoiceAsrSegment {
        start_time: ["Start time", "start_time", "start", "begin"]
            .iter()
            .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_seconds)),
        end_time: ["End time", "end_time", "end", "stop"]
            .iter()
            .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_seconds)),
        speaker_id: ["Speaker ID", "speaker_id", "speaker", "speaker_id"]
            .iter()
            .find_map(|key| get_value_case_insensitive(map, key).and_then(value_to_string))
            .map(|speaker| speaker.trim().to_string())
            .filter(|speaker| !speaker.is_empty()),
        content,
    })
}

fn get_value_case_insensitive<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    map.get(key).or_else(|| {
        map.iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
    })
}

fn value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn value_to_seconds(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Number(number) => number.as_f64().map(|value| value as f32),
        serde_json::Value::String(text) => parse_timestamp_seconds(text),
        _ => None,
    }
    .filter(|value| value.is_finite() && *value >= 0.0)
}

fn parse_timestamp_seconds(text: &str) -> Option<f32> {
    let trimmed = text
        .trim()
        .trim_end_matches(|ch: char| ch.eq_ignore_ascii_case(&'s'))
        .trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.contains(':') {
        let mut seconds = 0.0f32;
        for part in trimmed.split(':') {
            seconds = seconds * 60.0 + part.trim().parse::<f32>().ok()?;
        }
        return Some(seconds);
    }
    trimmed.parse::<f32>().ok()
}

fn join_segment_contents(segments: &[VibeVoiceAsrSegment]) -> String {
    segments
        .iter()
        .map(|segment| segment.content.trim())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn cleanup_transcript_text(raw: &str) -> String {
    raw.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .trim()
        .to_string()
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn preprocess_asr_audio(
    audio: &[f32],
    sample_rate: u32,
    config: &VibeVoicePreprocessorConfig,
) -> Result<(Vec<f32>, VibeVoiceAsrPreprocessStats)> {
    let mut resampled = resample_linear(audio, sample_rate, config.target_sample_rate())?;
    for sample in &mut resampled {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let stats = if config.normalize_audio {
        normalize_asr_loudness(&mut resampled, config.target_db_fs, config.eps)
    } else {
        VibeVoiceAsrPreprocessStats {
            normalized: false,
            target_db_fs: config.target_db_fs,
            rms_before: audio_rms(&resampled),
            gain: 1.0,
            clipping_divisor: 1.0,
        }
    };
    Ok((resampled, stats))
}

fn normalize_asr_loudness(
    samples: &mut [f32],
    target_db_fs: f32,
    eps: f32,
) -> VibeVoiceAsrPreprocessStats {
    if samples.is_empty() {
        return VibeVoiceAsrPreprocessStats {
            normalized: true,
            target_db_fs,
            rms_before: 0.0,
            gain: 1.0,
            clipping_divisor: 1.0,
        };
    }
    let rms = audio_rms(samples);
    let gain = 10f32.powf(target_db_fs / 20.0) / (rms + eps.max(0.0));
    for sample in samples.iter_mut() {
        *sample *= gain;
    }

    let peak = samples.iter().fold(0.0f32, |peak, &s| peak.max(s.abs()));
    let clipping_divisor = if peak > 1.0 { peak + eps.max(0.0) } else { 1.0 };
    if clipping_divisor != 1.0 {
        for sample in samples.iter_mut() {
            *sample /= clipping_divisor;
        }
    }

    VibeVoiceAsrPreprocessStats {
        normalized: true,
        target_db_fs,
        rms_before: rms,
        gain,
        clipping_divisor,
    }
}

fn audio_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|&sample| (sample as f64) * (sample as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32
}

fn asr_placeholder_count(samples: usize, speech_tok_compress_ratio: usize) -> usize {
    let ratio = speech_tok_compress_ratio.max(1);
    samples.saturating_add(ratio - 1) / ratio
}

fn vibevoice_tokenizer_state_projection(
    acoustic_input_range: Range<usize>,
    encoder_samples: usize,
) -> Result<ClockedStateProjection> {
    let primary = InputRange::new(acoustic_input_range.start, acoustic_input_range.end)?;
    let auxiliary = InputRange::new(0, encoder_samples)?;
    ClockedStateProjection::new(
        primary,
        ClockedStateSelection::new(VIBEVOICE_ASR_TOKENIZER_GROUP, StateClock::AudioSamples)?,
        auxiliary,
    )
}

fn vibevoice_asr_max_audio_seconds_hint(device_kind: DeviceKind) -> f32 {
    let cuda_override = std::env::var(CUDA_MAX_AUDIO_SECONDS_ENV).ok();
    vibevoice_asr_max_audio_seconds_hint_for(device_kind, cuda_override.as_deref())
}

fn vibevoice_asr_max_audio_seconds_hint_for(
    device_kind: DeviceKind,
    cuda_override: Option<&str>,
) -> f32 {
    if !device_kind.is_cuda() {
        return DEFAULT_MAX_AUDIO_SECONDS;
    }

    cuda_override
        .and_then(parse_positive_finite_f32)
        .unwrap_or(DEFAULT_MAX_AUDIO_SECONDS)
        .min(DEFAULT_MAX_AUDIO_SECONDS)
}

fn parse_positive_finite_f32(raw: &str) -> Option<f32> {
    raw.trim()
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn tokenizer_streaming_chunk_samples(sample_rate: u32, speech_tok_compress_ratio: usize) -> usize {
    let ratio = speech_tok_compress_ratio.max(1);
    let raw = sample_rate as usize * TOKENIZER_STREAMING_CHUNK_SECONDS;
    let aligned = raw / ratio * ratio;
    aligned.max(ratio)
}

fn tokenizer_chunk_ranges(total_samples: usize, chunk_samples: usize) -> Vec<(usize, usize)> {
    if total_samples == 0 {
        return Vec::new();
    }
    let chunk_samples = chunk_samples.max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < total_samples {
        let len = chunk_samples.min(total_samples - start);
        ranges.push((start, len));
        start = start.saturating_add(len);
    }
    ranges
}

fn resampled_sample_count(input_samples: usize, src_rate: u32, dst_rate: u32) -> Result<usize> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Audio sample rates must be non-zero".into(),
        ));
    }
    if src_rate == dst_rate || input_samples == 0 {
        return Ok(input_samples);
    }
    Ok(
        ((input_samples as f64) * (dst_rate as f64 / src_rate as f64))
            .round()
            .max(1.0) as usize,
    )
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Sample rates must be positive for VibeVoice-ASR resampling".to_string(),
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
    use candle_core::Device;

    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::architectures::qwen3::core::tiny_qwen3_model_for_test;

    fn test_managed_arena() -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = CpuKvArena::new(KvArenaConfig {
            id: KvArenaId {
                model_instance: ModelInstanceId::new(295),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: 40,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .expect("VibeVoice ASR test arena");
        (Arc::new(arena), vec![binding])
    }

    fn test_managed_cache(
        arena: Arc<dyn KvArena>,
        bindings: Vec<KvLayerBinding>,
        first_page: u32,
        context_len: usize,
    ) -> PhysicalPagedKvCache {
        let blocks = (first_page..first_page + 4)
            .map(|index| CacheBlockRef {
                arena: arena.id(),
                group: arena.config().group,
                index,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(arena, bindings, blocks, context_len)
            .expect("VibeVoice ASR test managed cache")
    }

    fn test_artifact(prompt_tokens: usize, hidden: usize) -> Arc<VibeVoiceAsrPreparedArtifact> {
        let mixed_embeddings =
            Tensor::zeros((1, prompt_tokens, hidden), DType::F32, &Device::Cpu).unwrap();
        Arc::new(VibeVoiceAsrPreparedArtifact {
            model_identity: [1; 32],
            source_identity: [2; 32],
            prompt_identity: [3; 32],
            prompt_ids: vec![0; prompt_tokens].into(),
            acoustic_input_range: 1..2,
            tokenizer_state_projections: Arc::from([vibevoice_tokenizer_state_projection(
                1..2,
                3_200,
            )
            .unwrap()]),
            mixed_embeddings,
            geometry: VibeVoiceAsrPreparedGeometry {
                input_samples: 3_200,
                input_sample_rate: 24_000,
                processed_samples: 3_200,
                encoder_samples: 3_200,
                acoustic_frames: 1,
                prompt_tokens,
                embedding_elements: (prompt_tokens * hidden) as u64,
                preparation_workspace_bytes: 4096,
                retained_device_bytes: (prompt_tokens * hidden * 4) as u64,
                retained_host_bytes: (prompt_tokens * size_of::<u32>()) as u64,
            },
        })
    }

    fn test_decode_state(
        cache: PhysicalPagedKvCache,
        prompt_tokens: usize,
        pending_token: Option<u32>,
    ) -> VibeVoiceAsrDecodeState {
        VibeVoiceAsrDecodeState {
            state_id: next_vibevoice_asr_state_id().unwrap(),
            model_identity: [1; 32],
            prompt_tokens,
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
            cache,
            prepared: None,
            prefill_progress: prompt_tokens,
            unconsumed_output: None,
            staged_step: None,
            pos: prompt_tokens,
            pending_token,
            generated: Vec::new(),
            assembled: String::new(),
            stop_tokens: vec![9, 10],
            stop_sequences: Vec::new(),
            max_new_tokens: 8,
            finished: false,
            stop_reason: None,
            stop_token_id: None,
            stop_sequence: None,
        }
    }

    fn assert_tensor_close(left: &Tensor, right: &Tensor) {
        assert_eq!(left.dims(), right.dims());
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (left, right) in left.into_iter().zip(right) {
            assert!((left - right).abs() < 1e-4, "{left} != {right}");
        }
    }

    #[test]
    fn retained_artifact_authenticates_source_prompt_and_exact_tensor_bytes() {
        let artifact = test_artifact(5, 8);
        validate_vibevoice_artifact_storage(&artifact).unwrap();
        assert_eq!(artifact.resident_tensor_bytes(), 160);
        assert_eq!(artifact.resident_host_bytes(), 20);
        assert_eq!(artifact.resident_bytes().unwrap(), 180);
        assert_eq!(artifact.geometry().work_cost(), WorkCost::new(1, 40, 4096));
        assert_ne!(
            vibevoice_asr_source_identity(&[0.0, 1.0], 24_000),
            vibevoice_asr_source_identity(&[0.0, 1.0], 16_000)
        );
        assert_ne!(
            vibevoice_asr_source_identity(&[0.0, 1.0], 24_000),
            vibevoice_asr_source_identity(&[0.0, -1.0], 24_000)
        );
        assert_ne!(
            vibevoice_asr_prompt_identity(Some("en"), Some("names")),
            vibevoice_asr_prompt_identity(Some("fr"), Some("names"))
        );

        let mut stale = (*artifact).clone();
        stale.geometry.retained_device_bytes += 4;
        assert!(validate_vibevoice_artifact_storage(&stale).is_err());
    }

    #[test]
    fn retained_artifact_rejects_a_different_model_load_instance() {
        let first_nonce = next_vibevoice_asr_model_load_nonce().unwrap();
        let second_nonce = next_vibevoice_asr_model_load_nonce().unwrap();
        assert_eq!(first_nonce.checked_add(1), Some(second_nonce));

        let artifact = test_artifact(5, 8);
        validate_vibevoice_artifact_model_identity(&artifact, [1; 32]).unwrap();
        let error = validate_vibevoice_artifact_model_identity(&artifact, [4; 32])
            .expect_err("an artifact from an earlier load must be rejected");
        assert!(error
            .to_string()
            .contains("different loaded model instance"));
    }

    #[test]
    fn production_like_tokenizer_workspace_is_a_peak_not_a_layer_sum() {
        let config: VibeVoiceTokenizerConfig = serde_json::from_value(json!({})).unwrap();
        let samples = 24_000 * 60;
        let elements = tokenizer_encoder_workspace_elements(&config, samples).unwrap();
        let input_elements = samples as u64;
        assert!(elements > input_elements);
        assert!(
            elements < input_elements * 512,
            "workspace peak {elements} is not viable relative to {input_elements} input elements"
        );
    }

    #[test]
    fn split_embedding_prefill_matches_one_shot_qwen_logits() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let ids = Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, 5), &device).unwrap();
        let embeddings = model.embeddings(&ids).unwrap();
        let (full_arena, full_bindings) = test_managed_arena();
        let (split_arena, split_bindings) = test_managed_arena();
        let mut full = test_managed_cache(full_arena, full_bindings, 0, 0);
        let mut split = test_managed_cache(split_arena, split_bindings, 0, 0);

        let full_logits = model
            .forward_managed_with_embeds(&embeddings, 0, &mut full, None)
            .unwrap();
        model
            .forward_managed_prefill_only_with_embeds(
                &embeddings.narrow(1, 0, 2).unwrap(),
                0,
                &mut split,
                None,
            )
            .unwrap();
        let split_logits = model
            .forward_managed_with_embeds(&embeddings.narrow(1, 2, 3).unwrap(), 2, &mut split, None)
            .unwrap();
        assert_eq!(full.context_len(), 5);
        assert_eq!(split.context_len(), 5);
        assert_tensor_close(
            &full_logits.i((.., 4..5, ..)).unwrap(),
            &split_logits.i((.., 2..3, ..)).unwrap(),
        );
    }

    #[test]
    fn final_prefill_samples_without_append_then_next_quantum_appends_pending_token() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut cache = test_managed_cache(arena.clone(), bindings.clone(), 0, 0);
        let ids = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device).unwrap();
        let embeddings = model.embeddings(&ids).unwrap();
        let logits = model
            .forward_managed_with_embeds(&embeddings, 0, &mut cache, None)
            .unwrap();
        assert_eq!(cache.context_len(), 3);

        let next = argmax_last_logits(&logits, false).unwrap();
        let mut state = test_decode_state(cache, 3, None);
        state.stop_tokens = vec![u32::MAX];
        let first =
            finish_vibevoice_decode_sample(&mut state, next, [u32::MAX, u32::MAX], |tokens| {
                Ok(format!("token-{}", tokens[0]))
            })
            .unwrap();
        state.staged_step = Some(first);
        state.managed_completions_drained = false;

        assert_eq!(state.sequence_position(), 3);
        assert_eq!(state.cache.context_len(), 3);
        assert_eq!(state.pending_token, Some(next));
        assert!(state.unconsumed_output.is_none());
        assert_eq!(state.take_managed_write_completions().len(), 1);
        assert!(state.take_staged_decode_step().is_some());

        let replacement = test_managed_cache(arena, bindings, 8, 3);
        let mut checkpoint = state.begin_managed_quantum(replacement).unwrap();
        {
            let mut rows = [&mut state];
            forward_vibevoice_pending_decode_batch(&model, &device, &mut rows).unwrap();
        }
        assert_eq!(state.sequence_position(), 4);
        assert_eq!(state.cache.context_len(), 4);
        assert!(state.pending_token.is_none());
        assert_eq!(state.take_managed_write_completions().len(), 1);
        state.commit_managed_quantum(&mut checkpoint).unwrap();
    }

    #[test]
    fn managed_quantum_rollback_restores_cache_token_text_and_completion_state() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut initial = test_managed_cache(arena.clone(), bindings.clone(), 0, 0);
        let prompt = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        model.forward_managed(&prompt, 0, &mut initial).unwrap();
        initial.take_completed_writes();
        let mut state = test_decode_state(initial, 2, Some(4));
        state.generated.push(4);
        state.assembled = "before".into();
        let replacement = test_managed_cache(arena, bindings, 8, 2);
        let mut checkpoint = state.begin_managed_quantum(replacement).unwrap();

        {
            let mut rows = [&mut state];
            forward_vibevoice_pending_decode_batch(&model, &device, &mut rows).unwrap();
        }
        state.generated.push(5);
        state.assembled = "after".into();
        state.finished = true;
        state.rollback_managed_quantum(&mut checkpoint).unwrap();

        assert_eq!(state.sequence_position(), 2);
        assert_eq!(state.pending_token, Some(4));
        assert_eq!(state.generated, vec![4]);
        assert_eq!(state.assembled, "before");
        assert!(!state.finished);
        assert!(state.take_managed_write_completions().is_empty());
        assert!(state.commit_managed_quantum(&mut checkpoint).is_err());
    }

    #[test]
    fn terminal_stop_transitions_do_not_append_a_pending_token() {
        let (arena, bindings) = test_managed_arena();
        let mut stop_token = test_decode_state(
            test_managed_cache(arena.clone(), bindings.clone(), 0, 0),
            0,
            None,
        );
        let step = finish_vibevoice_decode_sample(&mut stop_token, 9, [9, 10], |_| {
            panic!("stop token must not decode")
        })
        .unwrap();
        assert!(step.finished);
        assert_eq!(stop_token.stop_reason, Some("model_stop_token"));
        assert!(stop_token.pending_token.is_none());
        assert!(stop_token.take_managed_write_completions().is_empty());

        let mut stop_sequence =
            test_decode_state(test_managed_cache(arena, bindings, 8, 0), 0, None);
        stop_sequence.stop_sequences = vec![" END".into()];
        let step = finish_vibevoice_decode_sample(&mut stop_sequence, 4, [9, 10], |_| {
            Ok("hello END ignored".into())
        })
        .unwrap();
        assert!(step.finished);
        assert_eq!(step.text, "hello");
        assert_eq!(stop_sequence.generated, vec![4]);
        assert!(stop_sequence.pending_token.is_none());
        assert!(stop_sequence.take_managed_write_completions().is_empty());
    }

    #[test]
    fn ragged_native_decode_matches_scalar_and_preserves_row_isolation() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut scalar_a_cache = test_managed_cache(arena.clone(), bindings.clone(), 0, 0);
        let mut scalar_b_cache = test_managed_cache(arena.clone(), bindings.clone(), 4, 0);
        let mut batch_a_cache = test_managed_cache(arena.clone(), bindings.clone(), 8, 0);
        let mut batch_b_cache = test_managed_cache(arena, bindings, 12, 0);
        let prompt_a = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let prompt_b = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device).unwrap();
        for cache in [&mut scalar_a_cache, &mut batch_a_cache] {
            model.forward_managed(&prompt_a, 0, cache).unwrap();
            cache.take_completed_writes();
        }
        for cache in [&mut scalar_b_cache, &mut batch_b_cache] {
            model.forward_managed(&prompt_b, 0, cache).unwrap();
            cache.take_completed_writes();
        }
        let mut scalar_a = test_decode_state(scalar_a_cache, 2, Some(5));
        let mut scalar_b = test_decode_state(scalar_b_cache, 3, Some(6));
        let mut batch_a = test_decode_state(batch_a_cache, 2, Some(5));
        let mut batch_b = test_decode_state(batch_b_cache, 3, Some(6));
        let scalar_a_output = {
            let mut rows = [&mut scalar_a];
            forward_vibevoice_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };
        let scalar_b_output = {
            let mut rows = [&mut scalar_b];
            forward_vibevoice_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };
        let batch_output = {
            let mut rows = [&mut batch_a, &mut batch_b];
            forward_vibevoice_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };
        assert_tensor_close(
            &scalar_a_output,
            &batch_output.i(0).unwrap().unsqueeze(0).unwrap(),
        );
        assert_tensor_close(
            &scalar_b_output,
            &batch_output.i(1).unwrap().unsqueeze(0).unwrap(),
        );
        assert_eq!(batch_a.sequence_position(), 3);
        assert_eq!(batch_b.sequence_position(), 4);
        assert_eq!(batch_a.cache.context_len(), 3);
        assert_eq!(batch_b.cache.context_len(), 4);
        let completion_a = batch_a.take_managed_write_completions();
        let completion_b = batch_b.take_managed_write_completions();
        assert_eq!(completion_a.len(), 1);
        assert_eq!(completion_b.len(), 1);
        assert!(Arc::ptr_eq(&completion_a[0], &completion_b[0]));
    }

    #[test]
    fn ragged_batch_checkpoints_restore_every_row_after_late_semantic_failure() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut first_cache = test_managed_cache(arena.clone(), bindings.clone(), 0, 0);
        let mut second_cache = test_managed_cache(arena.clone(), bindings.clone(), 4, 0);
        let prompt = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        for cache in [&mut first_cache, &mut second_cache] {
            model.forward_managed(&prompt, 0, cache).unwrap();
            cache.take_completed_writes();
        }
        let mut first = test_decode_state(first_cache, 2, Some(5));
        let mut second = test_decode_state(second_cache, 2, Some(6));
        let mut first_checkpoint = first
            .begin_managed_quantum(test_managed_cache(arena.clone(), bindings.clone(), 8, 2))
            .unwrap();
        let mut second_checkpoint = second
            .begin_managed_quantum(test_managed_cache(arena, bindings, 12, 2))
            .unwrap();
        {
            let mut rows = [&mut first, &mut second];
            forward_vibevoice_pending_decode_batch(&model, &device, &mut rows).unwrap();
        }
        // A tokenizer/decode error on a later row is resolved by the handler
        // while both authenticated row checkpoints remain armed.
        first.generated.push(7);
        first.assembled = "mutated first".into();
        second.generated.push(8);
        second.assembled = "failed second".into();
        first
            .rollback_managed_quantum(&mut first_checkpoint)
            .unwrap();
        second
            .rollback_managed_quantum(&mut second_checkpoint)
            .unwrap();

        assert_eq!(first.sequence_position(), 2);
        assert_eq!(second.sequence_position(), 2);
        assert_eq!(first.pending_token, Some(5));
        assert_eq!(second.pending_token, Some(6));
        assert!(first.generated.is_empty());
        assert!(second.generated.is_empty());
        assert!(first.take_managed_write_completions().is_empty());
        assert!(second.take_managed_write_completions().is_empty());
    }

    #[test]
    fn resample_linear_preserves_identity_rate() {
        let audio = vec![0.0, 0.5, -0.25];
        assert_eq!(resample_linear(&audio, 24_000, 24_000).unwrap(), audio);
    }

    #[test]
    fn asr_placeholder_count_ceil_divides_samples_by_compress_ratio() {
        assert_eq!(asr_placeholder_count(0, 3_200), 0);
        assert_eq!(asr_placeholder_count(1, 3_200), 1);
        assert_eq!(asr_placeholder_count(3_200, 3_200), 1);
        assert_eq!(asr_placeholder_count(3_201, 3_200), 2);
        assert_eq!(asr_placeholder_count(9_599, 3_200), 3);
        assert_eq!(asr_placeholder_count(9_600, 3_200), 3);
    }

    #[test]
    fn tokenizer_projection_maps_placeholder_prefill_to_padded_audio_clock() {
        let projection = vibevoice_tokenizer_state_projection(4..7, 9_600).unwrap();
        assert!(projection
            .project(InputRange::new(0, 4).unwrap())
            .unwrap()
            .is_none());

        let first = projection
            .project(InputRange::new(2, 5).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(first.group(), VIBEVOICE_ASR_TOKENIZER_GROUP);
        assert_eq!(first.clock(), &StateClock::AudioSamples);
        assert_eq!(first.input(), InputRange::new(0, 3_200).unwrap());

        let tail = projection
            .project(InputRange::new(5, 9).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(tail.input(), InputRange::new(3_200, 9_600).unwrap());
    }

    #[test]
    fn tokenizer_projection_rejects_non_integral_placeholder_mapping() {
        assert!(vibevoice_tokenizer_state_projection(4..7, 9_599).is_err());
    }

    #[test]
    fn max_audio_seconds_hint_preserves_cpu_and_metal_window() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cpu, Some("30")),
            DEFAULT_MAX_AUDIO_SECONDS
        );
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Metal, Some("30")),
            DEFAULT_MAX_AUDIO_SECONDS
        );
    }

    #[test]
    fn max_audio_seconds_hint_uses_full_cuda_window_by_default() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, None),
            DEFAULT_MAX_AUDIO_SECONDS
        );
    }

    #[test]
    fn max_audio_seconds_hint_accepts_positive_cuda_override() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, Some("45.5")),
            45.5
        );
    }

    #[test]
    fn max_audio_seconds_hint_rejects_invalid_cuda_override() {
        for raw in ["", "0", "-1", "nan", "inf", "not-a-number"] {
            assert_eq!(
                vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, Some(raw)),
                DEFAULT_MAX_AUDIO_SECONDS
            );
        }
    }

    #[test]
    fn max_audio_seconds_hint_caps_cuda_override_to_model_window() {
        assert_eq!(
            vibevoice_asr_max_audio_seconds_hint_for(DeviceKind::Cuda, Some("7200")),
            DEFAULT_MAX_AUDIO_SECONDS
        );
    }

    #[test]
    fn tokenizer_streaming_chunk_size_uses_aligned_sixty_second_windows() {
        assert_eq!(tokenizer_streaming_chunk_samples(24_000, 3_200), 1_440_000);
        assert_eq!(tokenizer_streaming_chunk_samples(1_000, 3_200), 57_600);
        assert_eq!(tokenizer_streaming_chunk_samples(10, 3_200), 3_200);
    }

    #[test]
    fn tokenizer_chunk_ranges_cover_audio_without_overlap() {
        let ranges = tokenizer_chunk_ranges(10_000, 3_200);

        assert_eq!(
            ranges,
            vec![(0, 3_200), (3_200, 3_200), (6_400, 3_200), (9_600, 400)]
        );
    }

    #[test]
    fn normalize_asr_loudness_matches_reference_dbfs_formula() {
        let mut audio = vec![0.5, -0.5, 0.5, -0.5];
        let stats = normalize_asr_loudness(&mut audio, -20.0, 1e-6);

        let rms = audio_rms(&audio);
        assert!((rms - 0.1).abs() < 1e-5);
        assert!(stats.normalized);
        assert!((stats.rms_before - 0.5).abs() < 1e-6);
        assert!((stats.gain - 0.2).abs() < 1e-5);
        assert_eq!(stats.clipping_divisor, 1.0);
    }

    #[test]
    fn normalize_asr_loudness_avoids_clipping() {
        let mut audio = vec![1.0, -1.0];
        let stats = normalize_asr_loudness(&mut audio, 6.0, 1e-6);

        assert!(stats.clipping_divisor > 1.0);
        assert!(audio.iter().all(|sample| sample.abs() <= 1.0));
    }

    #[test]
    fn preprocess_asr_audio_sanitizes_and_resamples() {
        let config = VibeVoicePreprocessorConfig {
            sampling_rate: 4,
            speech_tok_compress_ratio: 2,
            normalize_audio: false,
            target_db_fs: -25.0,
            eps: 1e-6,
        };
        let (audio, stats) =
            preprocess_asr_audio(&[0.0, f32::NAN, 1.0], 2, &config).expect("preprocess");

        assert_eq!(audio.len(), 6);
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(!stats.normalized);
    }

    #[test]
    fn stop_sequences_are_trimmed_and_deduplicated() {
        let sequences = vec![" END ".to_string(), "".to_string(), "END".to_string()];
        assert_eq!(sanitize_stop_sequences(&sequences), vec!["END".to_string()]);
        assert_eq!(
            truncate_at_stop_sequence("hello END ignored", &sanitize_stop_sequences(&sequences)),
            ("hello ".to_string(), Some("END".to_string()))
        );
    }

    #[test]
    fn stop_token_ids_merge_built_ins_and_request_tokens() {
        assert_eq!(collect_stop_token_ids(&[1, 2], &[2, 3, 1]), vec![1, 2, 3]);
    }

    #[test]
    fn argmax_last_logits_preserves_host_fallback_selection() {
        let device = candle_core::Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.0f32, 3.0, 1.0, 2.0, -1.0, 5.0], (1, 2, 3), &device).unwrap();

        assert_eq!(argmax_last_logits(&logits, false).unwrap(), 2);
    }

    #[test]
    fn argmax_last_logits_can_select_on_device() {
        let device = candle_core::Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.0f32, 3.0, 1.0, 2.0, -1.0, 5.0], (1, 2, 3), &device).unwrap();

        assert_eq!(argmax_last_logits(&logits, true).unwrap(), 2);
    }

    #[test]
    fn parses_vibevoice_json_segments_into_plain_text() {
        let parsed = parse_vibevoice_asr_output(
            r#"[{"Start time": 0.0, "End time": "1.25", "Speaker ID": "Speaker 0", "Content": "Hello"}, {"Start time": "00:00:01.25", "End time": 2.0, "Speaker ID": 1, "Content": "world."}]"#,
        );

        assert_eq!(parsed.format, "segments");
        assert_eq!(parsed.text, "Hello world.");
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].speaker_id.as_deref(), Some("Speaker 0"));
        assert_eq!(parsed.segments[1].speaker_id.as_deref(), Some("1"));
        assert!((parsed.segments[1].start_time.unwrap() - 1.25).abs() < 1e-6);
    }

    #[test]
    fn parses_vibevoice_segments_inside_code_fence_and_wrapper() {
        let parsed = parse_vibevoice_asr_output(
            "```json\n{\"segments\":[{\"start\":0,\"end\":1,\"speaker\":\"A\",\"text\":\"Hi there\"}]}\n```",
        );

        assert_eq!(parsed.format, "segments");
        assert_eq!(parsed.text, "Hi there");
        assert_eq!(parsed.segments[0].end_time, Some(1.0));
    }

    #[test]
    fn parse_vibevoice_output_falls_back_to_cleaned_text() {
        let parsed = parse_vibevoice_asr_output("  plain text <|im_end|> ");

        assert_eq!(parsed.format, "text");
        assert_eq!(parsed.text, "plain text");
        assert!(parsed.segments.is_empty());
    }

    #[test]
    fn replace_range_preserves_prompt_length() {
        let device = candle_core::Device::Cpu;
        let embeds = Tensor::zeros((1, 5, 3), DType::F32, &device).unwrap();
        let features = Tensor::ones((1, 2, 3), DType::F32, &device).unwrap();
        let replaced = replace_range_with_features(&embeds, 2..4, &features).unwrap();
        assert_eq!(replaced.dims(), &[1, 5, 3]);
        assert_eq!(
            replaced.i((0, 2, ..)).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0]
        );
    }
}
