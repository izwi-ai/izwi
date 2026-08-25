//! Native Granite Speech ASR facade.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Tensor, D};
use serde_json::json;
use tracing::info;

use crate::backends::{parse_dtype_name, BackendKind, DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, StateClock, StateDType, StateDomainId, StateDomainSpec,
    StateGroupId, StateGroupSpec, StateScope, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{
    qwen3_decoder_cache_domain, Qwen3DecoderCacheGeometry,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::tokenizer::IncrementalDecoder;

mod config;
mod preprocessor;
mod prompt;
mod runtime;
mod transcript;

pub use config::{
    load_granite_speech_chat_template, GraniteSpeechAudioProcessorConfig, GraniteSpeechConfig,
    GraniteSpeechEncoderConfig, GraniteSpeechGenerationConfig, GraniteSpeechMelSpecConfig,
    GraniteSpeechProcessorConfig, GraniteSpeechProjectorConfig, GraniteSpeechTokenizerConfig,
    GraniteTextConfig,
};
pub use preprocessor::{GraniteSpeechAudioFeatures, GraniteSpeechPreprocessor};
pub use prompt::{
    GraniteSpeechPrompt, GraniteSpeechPromptOptions, GraniteSpeechPromptTokenizer,
    GraniteSpeechSpecialTokens, GraniteSpeechTask, GRANITE_SPEECH_ASR_PROMPT,
    GRANITE_SPEECH_SPEAKER_PROMPT, GRANITE_SPEECH_SYSTEM_PROMPT, GRANITE_SPEECH_TIMESTAMP_PROMPT,
};
pub use runtime::{
    GraniteSpeechAttentionDecodeProfile, GraniteSpeechAudioEmbeddingStats,
    GraniteSpeechDecodeLoopProfile, GraniteSpeechDecodeProfile, GraniteSpeechForwardProfile,
    GraniteSpeechGeneration, GraniteSpeechGenerationStats, GraniteSpeechGenerationTimings,
    GraniteSpeechLayerDecodeProfile, GraniteSpeechMlpDecodeProfile, GraniteSpeechRuntime,
};
pub use transcript::{
    parse_granite_speech_output, GraniteSpeechParsedTranscript, GraniteSpeechSegment,
    GraniteSpeechTimestampWord,
};

const REQUIRED_ARTIFACTS: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

const DEFAULT_MAX_AUDIO_SECONDS: f32 = 9.0 * 60.0;
const TIMESTAMP_MAX_AUDIO_SECONDS: f32 = 5.0 * 60.0;
static NEXT_GRANITE_SPEECH_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechAsrGenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for GraniteSpeechAsrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 768,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
}

pub struct GraniteSpeechAsrModel {
    model_identity: u64,
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: GraniteSpeechConfig,
    processor: GraniteSpeechProcessorConfig,
    generation: GraniteSpeechGenerationConfig,
    tokenizer_config: GraniteSpeechTokenizerConfig,
    chat_template: String,
    prompt_tokenizer: GraniteSpeechPromptTokenizer,
    preprocessor: GraniteSpeechPreprocessor,
    runtime: GraniteSpeechRuntime,
}

#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeechPreparedAudio {
    model_identity: u64,
    features: GraniteSpeechAudioFeatures,
    embeddings: Tensor,
    stats: GraniteSpeechAudioEmbeddingStats,
}

#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeechPreparedPromptArtifact {
    model_identity: u64,
    embeddings: Tensor,
    prompt_tokens: usize,
    audio_tokens: usize,
    language: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraniteSpeechPreparedGeometry {
    pub(crate) audio_samples: usize,
    pub(crate) encoder_frames: usize,
    pub(crate) encoder_dim: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) audio_tokens: usize,
    pub(crate) embedding_elements: u64,
    pub(crate) preparation_workspace_bytes: u64,
    pub(crate) retained_device_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GraniteSpeechPreparationBatchRow<'a> {
    pub(crate) audio: &'a [f32],
    pub(crate) sample_rate: u32,
    pub(crate) language: Option<&'a str>,
    pub(crate) prompt: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraniteSpeechPreparationBatchGeometry {
    pub(crate) batch_size: usize,
    pub(crate) padded_audio_samples_per_row: usize,
    pub(crate) padded_encoder_frames_per_row: usize,
    pub(crate) encoder_dim: usize,
    pub(crate) materialized_tensor_elements_per_row: u64,
    pub(crate) workspace_per_row_bytes: u64,
    pub(crate) max_workspace_bytes: u64,
}

impl GraniteSpeechPreparedGeometry {
    pub(crate) fn work_cost(self) -> crate::engine::WorkCost {
        crate::engine::WorkCost::new(
            self.audio_tokens as u64,
            self.embedding_elements,
            self.preparation_workspace_bytes,
        )
    }

    fn batch_useful_tensor_elements(self) -> Result<u64> {
        u64::try_from(self.encoder_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(u64::try_from(self.encoder_dim).ok()?))
            .and_then(|encoder| encoder.checked_add(self.embedding_elements))
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech useful preparation elements overflow".into())
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraniteSpeechAsrPreparationStageSeal {
    pub(crate) backend: BackendKind,
    pub(crate) dtype: String,
    pub(crate) max_work_units: u64,
    pub(crate) max_materialized_tensor_elements_per_row: u64,
    pub(crate) max_workspace_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraniteSpeechDecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

pub(crate) struct GraniteSpeechDecodeState {
    cache: PhysicalPagedKvCache,
    artifact: Arc<GraniteSpeechPreparedPromptArtifact>,
    prefill_progress: usize,
    pending_token: Option<u32>,
    pos: usize,
    generated_ids: Vec<u32>,
    incremental_decoder: IncrementalDecoder,
    rendered: String,
    published_len: usize,
    stop_holdback_bytes: usize,
    stop_tokens: Vec<u32>,
    stop_sequences: Vec<String>,
    max_new_tokens: usize,
    finished: bool,
    stop_reason: &'static str,
    stop_token: Option<u32>,
    state_id: u64,
    next_quantum_nonce: u64,
    active_quantum: Option<u64>,
    managed_completions_drained: bool,
}

struct GraniteSpeechDecodeCheckpointPayload {
    cache: PhysicalPagedKvCache,
    prefill_progress: usize,
    pending_token: Option<u32>,
    pos: usize,
    generated_ids: Vec<u32>,
    incremental_decoder: IncrementalDecoder,
    rendered: String,
    published_len: usize,
    finished: bool,
    stop_reason: &'static str,
    stop_token: Option<u32>,
    managed_completions_drained: bool,
}

pub(crate) struct GraniteSpeechDecodeCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    payload: Option<GraniteSpeechDecodeCheckpointPayload>,
}

impl GraniteSpeechDecodeState {
    pub(crate) const fn uses_managed_kv(&self) -> bool {
        true
    }

    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn prefill_token_count(&self) -> usize {
        self.artifact.prompt_tokens
    }

    pub(crate) fn sequence_position(&self) -> usize {
        self.pos
    }

    pub(crate) fn language(&self) -> Option<&str> {
        self.artifact.language.as_deref()
    }

    pub(crate) fn prepared_artifact(&self) -> Arc<GraniteSpeechPreparedPromptArtifact> {
        self.artifact.clone()
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let completions = self.cache.take_completed_writes();
        self.managed_completions_drained = true;
        completions
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<GraniteSpeechDecodeCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a Granite Speech managed quantum is already active".into(),
            ));
        }
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "Granite Speech managed KV completions must be drained before the next quantum"
                    .into(),
            ));
        }
        if self.cache.sequence_authority() != cache.sequence_authority() {
            return Err(Error::InferenceError(
                "a Granite Speech session cannot switch managed KV authority".into(),
            ));
        }
        if cache.context_len() != self.pos {
            return Err(Error::InferenceError(format!(
                "managed Granite Speech reservation starts at {}, but state is at {}",
                cache.context_len(),
                self.pos
            )));
        }
        let quantum_nonce = self.next_quantum_nonce;
        self.next_quantum_nonce = self
            .next_quantum_nonce
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("Granite Speech quantum nonce overflow".into()))?;
        self.active_quantum = Some(quantum_nonce);
        Ok(GraniteSpeechDecodeCheckpoint {
            state_id: self.state_id,
            quantum_nonce,
            payload: Some(GraniteSpeechDecodeCheckpointPayload {
                cache: std::mem::replace(&mut self.cache, cache),
                prefill_progress: self.prefill_progress,
                pending_token: self.pending_token,
                pos: self.pos,
                generated_ids: self.generated_ids.clone(),
                incremental_decoder: self.incremental_decoder.clone(),
                rendered: self.rendered.clone(),
                published_len: self.published_len,
                finished: self.finished,
                stop_reason: self.stop_reason,
                stop_token: self.stop_token,
                managed_completions_drained: self.managed_completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut GraniteSpeechDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Granite Speech checkpoint was already consumed".into())
        })?;
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut GraniteSpeechDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Granite Speech checkpoint was already consumed".into())
        })?;
        self.cache = payload.cache;
        self.prefill_progress = payload.prefill_progress;
        self.pending_token = payload.pending_token;
        self.pos = payload.pos;
        self.generated_ids = payload.generated_ids;
        self.incremental_decoder = payload.incremental_decoder;
        self.rendered = payload.rendered;
        self.published_len = payload.published_len;
        self.finished = payload.finished;
        self.stop_reason = payload.stop_reason;
        self.stop_token = payload.stop_token;
        self.managed_completions_drained = payload.managed_completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    fn validate_checkpoint(&self, checkpoint: &GraniteSpeechDecodeCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum_nonce)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Granite Speech checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }
}

impl GraniteSpeechPreparedAudio {
    pub(crate) fn features(&self) -> &GraniteSpeechAudioFeatures {
        &self.features
    }

    pub(crate) fn stats(&self) -> GraniteSpeechAudioEmbeddingStats {
        self.stats
    }
}

impl GraniteSpeechPreparedPromptArtifact {
    pub(crate) fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub(crate) fn audio_tokens(&self) -> usize {
        self.audio_tokens
    }

    pub(crate) fn resident_tensor_bytes(&self) -> Result<u64> {
        u64::try_from(self.embeddings.elem_count())
            .ok()
            .and_then(|elements| {
                elements.checked_mul(u64::try_from(self.embeddings.dtype().size_in_bytes()).ok()?)
            })
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech prepared artifact size overflow".into())
            })
    }

    pub(crate) fn resident_host_bytes(&self) -> u64 {
        self.language
            .as_ref()
            .map_or(0, |language| language.len() as u64)
    }
}

fn granite_decode_step(state: &GraniteSpeechDecodeState, delta: String) -> GraniteSpeechDecodeStep {
    GraniteSpeechDecodeStep {
        delta,
        text: state.rendered.clone(),
        tokens_generated: state.generated_ids.len(),
        finished: state.finished,
    }
}

fn stage_granite_first_decode_token(state: &mut GraniteSpeechDecodeState, token: u32) {
    state.pending_token = Some(token);
}

fn granite_batch_argmax(logits: &Tensor) -> Result<Vec<u32>> {
    let (batch, sequence, _vocab) = logits.dims3()?;
    if batch == 0 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "Granite Speech batch logits have invalid shape {:?}",
            logits.dims()
        )));
    }
    logits
        .narrow(1, sequence - 1, 1)?
        .squeeze(1)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()
        .map_err(Error::from)
}

fn truncate_granite_stop_sequence(text: &mut String, stop_sequences: &[String]) -> bool {
    for stop in stop_sequences.iter().filter(|stop| !stop.is_empty()) {
        if let Some(index) = text.find(stop) {
            text.truncate(index);
            return true;
        }
    }
    false
}

fn granite_publish_stable_text(
    state: &mut GraniteSpeechDecodeState,
    flush: bool,
) -> Result<String> {
    if state.rendered.len() < state.published_len {
        return Err(Error::InferenceError(
            "Granite Speech decoded text rewrote an already published prefix".into(),
        ));
    }
    let mut end = if flush {
        state.rendered.len()
    } else {
        state
            .rendered
            .len()
            .saturating_sub(state.stop_holdback_bytes)
    };
    while end > state.published_len && !state.rendered.is_char_boundary(end) {
        end -= 1;
    }
    if end < state.published_len {
        end = state.published_len;
    }
    let delta = state.rendered[state.published_len..end].to_string();
    state.published_len = end;
    Ok(delta)
}

#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeechPhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: Option<InferenceStateContract>,
    pub(crate) retained_max_tokens: Option<usize>,
    pub(crate) invocation: InferenceStateContract,
}

impl GraniteSpeechAsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::GraniteSpeech412BPlus {
            return Err(Error::InvalidInput(format!(
                "GraniteSpeechAsrModel cannot load non-Granite variant {variant}"
            )));
        }

        let shards = ensure_granite_speech_artifacts(model_dir)?;
        let config = GraniteSpeechConfig::load(model_dir)?;
        config.validate_plus()?;
        let processor = GraniteSpeechProcessorConfig::load(model_dir)?;
        let generation = GraniteSpeechGenerationConfig::load(model_dir)?;
        let tokenizer_config = GraniteSpeechTokenizerConfig::load(model_dir)?;
        let chat_template = load_granite_speech_chat_template(model_dir)?;
        let prompt_tokenizer =
            GraniteSpeechPromptTokenizer::load(model_dir, &config, &processor, &tokenizer_config)?;
        let preprocessor = GraniteSpeechPreprocessor::new(processor.clone())?;
        let dtype = select_granite_speech_dtype(&device, config.target_dtype_hint())?;
        let runtime = GraniteSpeechRuntime::load(&shards, &config, &device, dtype)?;

        info!(
            "Loaded Granite Speech ASR in {:?} on {:?} with dtype {:?} ({} shard files)",
            model_dir,
            device.kind,
            dtype,
            shards.len()
        );

        Ok(Self {
            model_identity: NEXT_GRANITE_SPEECH_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            processor,
            generation,
            tokenizer_config,
            chat_template,
            prompt_tokenizer,
            preprocessor,
            runtime,
        })
    }

    pub fn config(&self) -> &GraniteSpeechConfig {
        &self.config
    }

    pub(crate) const fn supports_resumable_prefill(&self) -> bool {
        true
    }

    pub(crate) const fn supports_incremental_decode(&self) -> bool {
        true
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<GraniteSpeechPhysicalStateSpec> {
        let invocation = granite_speech_invocation_contract(
            &self.config,
            self.runtime.kv_dtype(),
            default_kv_page_size(),
        )?;
        let retained = granite_speech_retained_contract(invocation.clone())?;
        granite_speech_physical_state_spec(
            stage_graphs,
            retained,
            invocation,
            self.config.text_config.max_position_embeddings,
        )
    }

    pub fn processor(&self) -> &GraniteSpeechProcessorConfig {
        &self.processor
    }

    pub fn generation_config(&self) -> &GraniteSpeechGenerationConfig {
        &self.generation
    }

    pub fn tokenizer_config(&self) -> &GraniteSpeechTokenizerConfig {
        &self.tokenizer_config
    }

    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn prepare_audio_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<GraniteSpeechAudioFeatures> {
        self.preprocessor.prepare(audio, sample_rate)
    }

    pub fn build_prompt(
        &self,
        options: &GraniteSpeechPromptOptions,
    ) -> Result<GraniteSpeechPrompt> {
        self.prompt_tokenizer.build_prompt(options)
    }

    pub(crate) fn prepare_audio_retained(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Arc<GraniteSpeechPreparedAudio>> {
        let features = self.prepare_audio_features(audio, sample_rate)?;
        validate_granite_audio_duration(features.audio_seconds)?;
        let (embeddings, stats) = self.runtime.audio_embeddings_with_stats(&features)?;
        Ok(Arc::new(GraniteSpeechPreparedAudio {
            model_identity: self.model_identity,
            features,
            embeddings,
            stats,
        }))
    }

    pub(crate) fn retained_preparation_geometry(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<GraniteSpeechPreparedGeometry> {
        let features = self.prepare_audio_features(audio, sample_rate)?;
        validate_granite_audio_duration(features.audio_seconds)?;
        let windows = features
            .encoder_frames
            .saturating_add(self.config.window_size.max(1) - 1)
            / self.config.window_size.max(1);
        let queries = self.config.window_size.max(1) / self.config.downsample_rate.max(1);
        let audio_tokens = windows.checked_mul(queries).ok_or_else(|| {
            Error::Overloaded("Granite Speech audio token geometry overflow".into())
        })?;
        if audio_tokens == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech projected zero audio tokens".into(),
            ));
        }
        let granite_prompt = self.build_prompt(&GraniteSpeechPromptOptions {
            task: GraniteSpeechTask::Asr,
            language: language.map(str::to_string),
            custom_prompt: prompt.map(str::to_string),
            ..GraniteSpeechPromptOptions::default()
        })?;
        if granite_prompt.audio_token_positions.len() != 1 {
            return Err(Error::InvalidInput(
                "Granite Speech prompt must contain exactly one audio placeholder".into(),
            ));
        }
        let prompt_tokens = granite_prompt
            .input_ids
            .len()
            .checked_add(audio_tokens.saturating_sub(1))
            .ok_or_else(|| Error::Overloaded("Granite Speech prompt geometry overflow".into()))?;
        let embedding_elements = u64::try_from(prompt_tokens)
            .ok()
            .and_then(|tokens| {
                tokens.checked_mul(u64::try_from(self.config.text_config.hidden_size).ok()?)
            })
            .ok_or_else(|| Error::Overloaded("Granite Speech artifact geometry overflow".into()))?;
        let retained_device_bytes = embedding_elements
            .checked_mul(
                u64::try_from(self.dtype.size_in_bytes()).map_err(|_| {
                    Error::Overloaded("Granite Speech dtype size exceeds u64".into())
                })?,
            )
            .ok_or_else(|| Error::Overloaded("Granite Speech artifact bytes overflow".into()))?;
        let preparation_workspace_bytes = self.preparation_workspace_bytes(
            features.encoder_frames,
            features.encoder_dim,
            prompt_tokens,
        )?;
        Ok(GraniteSpeechPreparedGeometry {
            audio_samples: audio.len(),
            encoder_frames: features.encoder_frames,
            encoder_dim: features.encoder_dim,
            prompt_tokens,
            audio_tokens,
            embedding_elements,
            preparation_workspace_bytes,
            retained_device_bytes,
        })
    }

    pub(crate) fn preparation_batch_geometry(
        &self,
        rows: &[GraniteSpeechPreparedGeometry],
    ) -> Result<GraniteSpeechPreparationBatchGeometry> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Granite Speech preparation batch geometry is empty".into(),
            ));
        }
        let padded_audio_samples_per_row =
            rows.iter().map(|row| row.audio_samples).max().unwrap_or(0);
        let padded_encoder_frames_per_row =
            rows.iter().map(|row| row.encoder_frames).max().unwrap_or(0);
        let encoder_dim = rows[0].encoder_dim;
        let max_prompt_tokens = rows.iter().map(|row| row.prompt_tokens).max().unwrap_or(0);
        if padded_audio_samples_per_row == 0
            || padded_encoder_frames_per_row == 0
            || encoder_dim == 0
            || rows.iter().any(|row| row.encoder_dim != encoder_dim)
        {
            return Err(Error::InvalidInput(
                "Granite Speech preparation rows have incompatible geometry".into(),
            ));
        }
        let input_elements = u64::try_from(padded_encoder_frames_per_row)
            .ok()
            .and_then(|frames| frames.checked_mul(u64::try_from(encoder_dim).ok()?))
            .ok_or_else(|| Error::Overloaded("Granite Speech batch input overflow".into()))?;
        let prompt_elements = u64::try_from(max_prompt_tokens)
            .ok()
            .and_then(|tokens| {
                tokens.checked_mul(u64::try_from(self.config.text_config.hidden_size).ok()?)
            })
            .ok_or_else(|| Error::Overloaded("Granite Speech batch prompt overflow".into()))?;
        let materialized_tensor_elements_per_row =
            input_elements.checked_add(prompt_elements).ok_or_else(|| {
                Error::Overloaded("Granite Speech batch materialization overflow".into())
            })?;
        let workspace_per_row_bytes = self.preparation_workspace_bytes(
            padded_encoder_frames_per_row,
            encoder_dim,
            max_prompt_tokens,
        )?;
        let max_workspace_bytes = workspace_per_row_bytes
            .checked_mul(
                u64::try_from(rows.len()).map_err(|_| {
                    Error::Overloaded("Granite Speech batch width exceeds u64".into())
                })?,
            )
            .ok_or_else(|| Error::Overloaded("Granite Speech batch workspace overflow".into()))?;
        Ok(GraniteSpeechPreparationBatchGeometry {
            batch_size: rows.len(),
            padded_audio_samples_per_row,
            padded_encoder_frames_per_row,
            encoder_dim,
            materialized_tensor_elements_per_row,
            workspace_per_row_bytes,
            max_workspace_bytes,
        })
    }

    pub(crate) fn preparation_row_cost_for_batch(
        &self,
        index: usize,
        rows: &[GraniteSpeechPreparedGeometry],
        batch: GraniteSpeechPreparationBatchGeometry,
    ) -> Result<crate::engine::WorkCost> {
        let row = rows.get(index).ok_or_else(|| {
            Error::InvalidInput("Granite Speech preparation row index is out of range".into())
        })?;
        if batch != self.preparation_batch_geometry(rows)? {
            return Err(Error::InvalidInput(
                "Granite Speech preparation batch geometry is stale or foreign".into(),
            ));
        }
        Ok(crate::engine::WorkCost::new(
            row.audio_tokens as u64,
            row.batch_useful_tensor_elements()?,
            batch.workspace_per_row_bytes,
        ))
    }

    fn preparation_workspace_bytes(
        &self,
        encoder_frames: usize,
        encoder_dim: usize,
        prompt_tokens: usize,
    ) -> Result<u64> {
        let frames = u64::try_from(encoder_frames)
            .map_err(|_| Error::Overloaded("Granite Speech encoder frames exceed u64".into()))?;
        let input_dim = u64::try_from(encoder_dim)
            .map_err(|_| Error::Overloaded("Granite Speech encoder width exceeds u64".into()))?;
        let encoder_hidden =
            u64::try_from(self.config.encoder_config.hidden_dim).map_err(|_| {
                Error::Overloaded("Granite Speech encoder hidden width exceeds u64".into())
            })?;
        let encoder_output = u64::try_from(self.config.projector_config.encoder_hidden_size)
            .map_err(|_| {
                Error::Overloaded("Granite Speech projector input width exceeds u64".into())
            })?;
        let q_hidden = u64::try_from(self.config.projector_config.hidden_size)
            .map_err(|_| Error::Overloaded("Granite Speech projector width exceeds u64".into()))?;
        let q_intermediate = u64::try_from(self.config.projector_config.intermediate_size)
            .map_err(|_| {
                Error::Overloaded("Granite Speech projector MLP width exceeds u64".into())
            })?;
        let text_hidden = u64::try_from(self.config.text_config.hidden_size)
            .map_err(|_| Error::Overloaded("Granite Speech text width exceeds u64".into()))?;
        let windows = u64::try_from(
            encoder_frames.saturating_add(self.config.window_size.max(1) - 1)
                / self.config.window_size.max(1),
        )
        .map_err(|_| Error::Overloaded("Granite Speech projector windows exceed u64".into()))?;
        let window = u64::try_from(self.config.window_size.max(1))
            .map_err(|_| Error::Overloaded("Granite Speech projector window exceeds u64".into()))?;
        let queries =
            u64::try_from(self.config.window_size.max(1) / self.config.downsample_rate.max(1))
                .map_err(|_| {
                    Error::Overloaded("Granite Speech projector queries exceed u64".into())
                })?;
        let retained_encoder_rows = u64::try_from(
            self.config
                .encoder_config
                .cat_hidden_layers
                .len()
                .saturating_add(10),
        )
        .map_err(|_| Error::Overloaded("Granite Speech encoder workspace exceeds u64".into()))?;

        // This is a conservative live-set envelope, not an allocation sum. It
        // covers preprocessing/upload, the Conformer residual/QKV/MLP live
        // set plus retained concatenation rows, QFormer window attention/MLP,
        // and prompt embedding assembly. The same formula authors both the
        // request-shaped reservation and the loaded-model ceiling.
        let encoder_elements = input_dim
            .checked_add(
                encoder_hidden
                    .checked_mul(retained_encoder_rows)
                    .ok_or_else(|| {
                        Error::Overloaded("Granite Speech encoder workspace overflow".into())
                    })?,
            )
            .and_then(|width| {
                encoder_output
                    .checked_mul(2)
                    .and_then(|output| width.checked_add(output))
            })
            .and_then(|width| frames.checked_mul(width))
            .ok_or_else(|| Error::Overloaded("Granite Speech encoder workspace overflow".into()))?;
        let projector_width = window
            .checked_mul(encoder_output)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| {
                queries
                    .checked_mul(q_hidden.checked_mul(12)?)
                    .and_then(|q| elements.checked_add(q))
            })
            .and_then(|elements| {
                queries
                    .checked_mul(q_intermediate.checked_mul(2)?)
                    .and_then(|q| elements.checked_add(q))
            })
            .and_then(|elements| {
                queries
                    .checked_mul(text_hidden)
                    .and_then(|q| elements.checked_add(q))
            })
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech projector workspace overflow".into())
            })?;
        let projector_elements = windows.checked_mul(projector_width).ok_or_else(|| {
            Error::Overloaded("Granite Speech projector workspace overflow".into())
        })?;
        let prompt_elements = u64::try_from(prompt_tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(text_hidden))
            .and_then(|elements| elements.checked_mul(3))
            .ok_or_else(|| Error::Overloaded("Granite Speech prompt workspace overflow".into()))?;
        let live_elements = encoder_elements
            .checked_add(projector_elements)
            .and_then(|elements| elements.checked_add(prompt_elements))
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech preparation workspace overflow".into())
            })?;
        let element_bytes =
            u64::try_from(self.dtype.size_in_bytes().max(std::mem::size_of::<f32>()))
                .map_err(|_| Error::Overloaded("Granite Speech dtype size exceeds u64".into()))?;
        live_elements
            .checked_mul(element_bytes)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech preparation workspace overflow".into())
            })
    }

    pub(crate) fn scalar_preparation_stage_seal(
        &self,
        backend: BackendKind,
    ) -> Result<GraniteSpeechAsrPreparationStageSeal> {
        let loaded = match self.device.kind {
            DeviceKind::Cpu => BackendKind::Cpu,
            DeviceKind::Metal => BackendKind::Metal,
            DeviceKind::Cuda => BackendKind::Cuda,
        };
        if loaded != backend {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech ASR preparation backend mismatch: model={loaded:?}, adapter={backend:?}"
            )));
        }
        let sample_rate = u64::from(self.processor.sample_rate());
        let samples = sample_rate
            .checked_mul(DEFAULT_MAX_AUDIO_SECONDS as u64)
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech maximum sample count overflow".into())
            })?;
        let hop = u64::try_from(
            self.processor
                .audio_processor
                .melspec_kwargs
                .hop_length
                .max(1),
        )
        .map_err(|_| Error::Overloaded("Granite Speech hop length exceeds u64".into()))?;
        let encoder_frames = usize::try_from(samples / hop / 2 + 2)
            .map_err(|_| Error::Overloaded("Granite Speech maximum frames exceed usize".into()))?;
        let encoder_dim = self.config.encoder_config.input_dim;
        let prompt_tokens = self.config.text_config.max_position_embeddings;
        let windows = encoder_frames.saturating_add(self.config.window_size.max(1) - 1)
            / self.config.window_size.max(1);
        let audio_tokens = windows
            .checked_mul(self.config.window_size.max(1) / self.config.downsample_rate.max(1))
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech maximum work units overflow".into())
            })?;
        let max_materialized_tensor_elements_per_row = u64::try_from(encoder_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(u64::try_from(encoder_dim).ok()?))
            .and_then(|input| {
                let prompt = u64::try_from(prompt_tokens)
                    .ok()?
                    .checked_mul(u64::try_from(self.config.text_config.hidden_size).ok()?)?;
                input.checked_add(prompt)
            })
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech maximum materialization overflow".into())
            })?;
        Ok(GraniteSpeechAsrPreparationStageSeal {
            backend,
            dtype: format!("{:?}", self.dtype).to_ascii_lowercase(),
            max_work_units: u64::try_from(audio_tokens).map_err(|_| {
                Error::Overloaded("Granite Speech maximum work units exceed u64".into())
            })?,
            max_materialized_tensor_elements_per_row,
            max_workspace_bytes: self.preparation_workspace_bytes(
                encoder_frames,
                encoder_dim,
                prompt_tokens,
            )?,
        })
    }

    pub(crate) fn prepare_prompt_artifact(
        &self,
        audio: &GraniteSpeechPreparedAudio,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
    ) -> Result<Arc<GraniteSpeechPreparedPromptArtifact>> {
        if audio.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Granite Speech prepared audio belongs to another loaded model".into(),
            ));
        }
        let granite_prompt = self.build_prompt(&GraniteSpeechPromptOptions {
            task,
            language: language.map(str::to_string),
            custom_prompt: prompt.map(str::to_string),
            prefix_text: prefix_text.map(str::to_string),
            ..GraniteSpeechPromptOptions::default()
        })?;
        let (embeddings, prompt_tokens, audio_tokens) = self.runtime.prepare_prompt_embeddings(
            &granite_prompt,
            self.prompt_tokenizer.special_tokens(),
            &audio.embeddings,
        )?;
        Ok(Arc::new(GraniteSpeechPreparedPromptArtifact {
            model_identity: self.model_identity,
            embeddings,
            prompt_tokens,
            audio_tokens,
            language: language.map(str::to_string),
        }))
    }

    pub(crate) fn prepare_prompt_artifact_batch(
        &self,
        rows: &[GraniteSpeechPreparationBatchRow<'_>],
    ) -> Result<Vec<Arc<GraniteSpeechPreparedPromptArtifact>>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Granite Speech prompt preparation batch is empty".into(),
            ));
        }
        if rows.len() == 1 {
            let row = rows[0];
            let audio = self.prepare_audio_retained(row.audio, row.sample_rate)?;
            return self
                .prepare_prompt_artifact(
                    &audio,
                    row.language,
                    GraniteSpeechTask::Asr,
                    row.prompt,
                    None,
                )
                .map(|artifact| vec![artifact]);
        }
        let mut features = Vec::with_capacity(rows.len());
        let mut prompts = Vec::with_capacity(rows.len());
        for row in rows {
            let row_features = self.prepare_audio_features(row.audio, row.sample_rate)?;
            validate_granite_audio_duration(row_features.audio_seconds)?;
            let prompt = self.build_prompt(&GraniteSpeechPromptOptions {
                task: GraniteSpeechTask::Asr,
                language: row.language.map(str::to_string),
                custom_prompt: row.prompt.map(str::to_string),
                ..GraniteSpeechPromptOptions::default()
            })?;
            if prompt.audio_token_positions.len() != 1 {
                return Err(Error::InvalidInput(
                    "Granite Speech prompt must contain exactly one audio placeholder".into(),
                ));
            }
            features.push(row_features);
            prompts.push(prompt);
        }
        let audio = self.runtime.audio_embeddings_batch_with_stats(&features)?;
        prompts
            .iter()
            .zip(audio)
            .zip(rows)
            .map(|((prompt, (audio_embeddings, _stats)), row)| {
                let (embeddings, prompt_tokens, audio_tokens) =
                    self.runtime.prepare_prompt_embeddings(
                        prompt,
                        self.prompt_tokenizer.special_tokens(),
                        &audio_embeddings,
                    )?;
                Ok(Arc::new(GraniteSpeechPreparedPromptArtifact {
                    model_identity: self.model_identity,
                    embeddings,
                    prompt_tokens,
                    audio_tokens,
                    language: row.language.map(str::to_string),
                }))
            })
            .collect()
    }

    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        artifact: Arc<GraniteSpeechPreparedPromptArtifact>,
        options: GraniteSpeechAsrGenerationOptions,
        cache: PhysicalPagedKvCache,
    ) -> Result<GraniteSpeechDecodeState> {
        if artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Granite Speech prepared prompt belongs to another loaded model".into(),
            ));
        }
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Granite Speech retained prefill must begin with an empty cache".into(),
            ));
        }
        let max_new_tokens = options.max_new_tokens.max(1);
        let required = artifact
            .prompt_tokens
            .checked_add(max_new_tokens)
            .ok_or_else(|| Error::InvalidInput("Granite Speech context length overflow".into()))?;
        if required > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Granite Speech prompt and decode require {required} KV tokens, but retained cache capacity is {}",
                cache.capacity_tokens()
            )));
        }
        let special = self.prompt_tokenizer.special_tokens();
        let mut stop_tokens = BTreeSet::from([special.eos_token_id, special.pad_token_id]);
        stop_tokens.extend(options.stop_token_ids);
        Ok(GraniteSpeechDecodeState {
            cache,
            artifact,
            prefill_progress: 0,
            pending_token: None,
            pos: 0,
            generated_ids: Vec::new(),
            incremental_decoder: self.prompt_tokenizer.incremental_decoder(),
            rendered: String::new(),
            published_len: 0,
            stop_holdback_bytes: options
                .stop_sequences
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(0)
                .saturating_sub(1),
            stop_tokens: stop_tokens.into_iter().collect(),
            stop_sequences: options.stop_sequences,
            max_new_tokens,
            finished: false,
            stop_reason: "max_tokens",
            stop_token: None,
            state_id: NEXT_GRANITE_SPEECH_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
        })
    }

    pub(crate) fn continue_resumable_prefill(
        &self,
        state: &mut GraniteSpeechDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Granite Speech retained state belongs to another loaded model".into(),
            ));
        }
        if span_start != state.prefill_progress
            || span_start >= span_end
            || span_end > state.artifact.prompt_tokens
            || state.pos != span_start
        {
            return Err(Error::InvalidInput(
                "Granite Speech prefill span is non-monotonic or out of range".into(),
            ));
        }
        let logits = self.runtime.prefill_prepared_prompt_span(
            &state.artifact.embeddings,
            span_start,
            span_end,
            &mut state.cache,
        )?;
        state.prefill_progress = span_end;
        state.pos = span_end;
        if span_end == state.artifact.prompt_tokens {
            let token = self.runtime.greedy_token(&logits)?;
            stage_granite_first_decode_token(state, token);
        }
        state.managed_completions_drained = false;
        Ok(span_end == state.artifact.prompt_tokens)
    }

    pub(crate) fn decode_step(
        &self,
        state: &mut GraniteSpeechDecodeState,
    ) -> Result<GraniteSpeechDecodeStep> {
        if state.artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Granite Speech retained state belongs to another loaded model".into(),
            ));
        }
        if state.finished {
            return Ok(granite_decode_step(state, String::new()));
        }
        if state.prefill_progress != state.artifact.prompt_tokens {
            return Err(Error::InferenceError(
                "Granite Speech decode requires completed prefill".into(),
            ));
        }
        let appended = state.pending_token.ok_or_else(|| {
            Error::InferenceError("Granite Speech decode has no staged input token".into())
        })?;
        let logits = self
            .runtime
            .decode_token(appended, state.pos, &mut state.cache)?;
        let next = self.runtime.greedy_token(&logits)?;
        state.pending_token = None;
        state.pos = state.pos.checked_add(1).ok_or_else(|| {
            Error::InferenceError("Granite Speech decode position overflow".into())
        })?;
        state.managed_completions_drained = false;
        self.finish_appended_decode_token(state, appended, next)
    }

    fn finish_appended_decode_token(
        &self,
        state: &mut GraniteSpeechDecodeState,
        appended: u32,
        next: u32,
    ) -> Result<GraniteSpeechDecodeStep> {
        if state.stop_tokens.binary_search(&appended).is_ok() {
            state.rendered.push_str(
                &self
                    .prompt_tokenizer
                    .finish_incremental_decode(&mut state.incremental_decoder)?,
            );
            truncate_granite_stop_sequence(&mut state.rendered, &state.stop_sequences);
            let delta = granite_publish_stable_text(state, true)?;
            state.pending_token = None;
            state.finished = true;
            state.stop_reason = "stop_token";
            state.stop_token = Some(appended);
            return Ok(granite_decode_step(state, delta));
        }
        state.generated_ids.push(appended);
        state.rendered.push_str(
            &self
                .prompt_tokenizer
                .decode_incrementally(&mut state.incremental_decoder, appended)?,
        );
        let mut stopped_on_sequence =
            truncate_granite_stop_sequence(&mut state.rendered, &state.stop_sequences);
        let reached_max_tokens = state.generated_ids.len() >= state.max_new_tokens;
        let stopped_on_token = state.stop_tokens.binary_search(&next).is_ok();
        if (reached_max_tokens || stopped_on_token) && !stopped_on_sequence {
            state.rendered.push_str(
                &self
                    .prompt_tokenizer
                    .finish_incremental_decode(&mut state.incremental_decoder)?,
            );
            stopped_on_sequence =
                truncate_granite_stop_sequence(&mut state.rendered, &state.stop_sequences);
        }
        let finished = stopped_on_sequence || reached_max_tokens || stopped_on_token;
        let delta = granite_publish_stable_text(state, finished)?;
        state.finished = finished;
        if stopped_on_sequence {
            state.stop_reason = "stop_sequence";
        } else if stopped_on_token {
            state.stop_reason = "stop_token";
            state.stop_token = Some(next);
        }
        state.pending_token = (!finished).then_some(next);
        Ok(granite_decode_step(state, delta))
    }

    /// One native ragged token step for retained Granite rows sharing one
    /// physical arena. Text assembly remains isolated per row.
    pub(crate) fn decode_step_batch(
        &self,
        states: &mut [&mut GraniteSpeechDecodeState],
    ) -> Result<Vec<GraniteSpeechDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.artifact.model_identity != self.model_identity
                || state.finished
                || state.prefill_progress != state.artifact.prompt_tokens
                || state.pending_token.is_none()
                || state.cache.context_len() != state.pos
            {
                return Err(Error::InvalidInput(
                    "Granite Speech decode batch requires one staged input token per completed retained row"
                        .into(),
                ));
            }
            state.pos.checked_add(1).ok_or_else(|| {
                Error::InferenceError("Granite Speech decode position overflow".into())
            })?;
        }
        let appended = states
            .iter()
            .map(|state| state.pending_token.expect("validated pending token"))
            .collect::<Vec<_>>();
        let positions = states.iter().map(|state| state.pos).collect::<Vec<_>>();
        let output = {
            let mut caches = states
                .iter_mut()
                .map(|state| &mut state.cache)
                .collect::<Vec<_>>();
            self.runtime
                .decode_tokens_batch(&appended, &positions, &mut caches)?
        };
        let sampled = granite_batch_argmax(&output)?;
        if sampled.len() != states.len() {
            return Err(Error::InferenceError(
                "Granite Speech decode batch returned the wrong row count".into(),
            ));
        }
        for state in states.iter_mut() {
            state.pending_token = None;
            state.pos += 1;
            state.managed_completions_drained = false;
        }
        let mut steps = Vec::with_capacity(states.len());
        for ((state, appended), next) in states.iter_mut().zip(appended).zip(sampled) {
            steps.push(self.finish_appended_decode_token(state, appended, next)?);
        }
        Ok(steps)
    }

    pub(crate) const fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    pub(crate) fn continuous_decode_workspace_per_row_bytes(&self) -> Result<u64> {
        u64::try_from(self.config.text_config.hidden_size)
            .ok()
            .and_then(|hidden| hidden.checked_mul(u64::try_from(self.dtype.size_in_bytes()).ok()?))
            .ok_or_else(|| Error::Overloaded("Granite Speech decode workspace overflow".into()))
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(DEFAULT_MAX_AUDIO_SECONDS)
    }

    pub fn max_timestamp_audio_seconds_hint(&self) -> Option<f32> {
        Some(TIMESTAMP_MAX_AUDIO_SECONDS)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt_and_options(
            audio,
            sample_rate,
            language,
            prompt,
            GraniteSpeechAsrGenerationOptions::default(),
        )
    }

    pub fn transcribe_with_details_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, prompt, options);
        Err(Error::InferenceError(
            "Granite Speech ASR requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub fn transcribe_with_details_and_prompt_prefix_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, prompt, prefix_text, options);
        Err(Error::InferenceError(
            "Granite Speech ASR requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub fn transcribe_with_details_task_prefix_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, task, prefix_text, options);
        Err(Error::InferenceError(
            "Granite Speech ASR requires a lifecycle-owned physical invocation cache".into(),
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
            GraniteSpeechAsrGenerationOptions::default(),
            on_delta,
        )
        .map(|output| output.text)
    }

    pub fn transcribe_with_callback_and_prompt_and_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, prompt, options, on_delta);
        Err(Error::InferenceError(
            "Granite Speech ASR requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub(crate) fn transcribe_with_details_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            GraniteSpeechTask::Asr,
            prompt,
            None,
            options,
            cache,
            &mut no_op,
            false,
        )
    }

    pub(crate) fn transcribe_with_details_and_prompt_prefix_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            GraniteSpeechTask::Asr,
            prompt,
            prefix_text,
            options,
            cache,
            &mut no_op,
            false,
        )
    }

    pub(crate) fn transcribe_with_details_task_prefix_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            task,
            None,
            prefix_text,
            options,
            cache,
            &mut no_op,
            false,
        )
    }

    pub(crate) fn transcribe_with_callback_and_prompt_and_options_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        self.transcribe_internal(
            audio,
            sample_rate,
            language,
            GraniteSpeechTask::Asr,
            prompt,
            None,
            options,
            cache,
            on_delta,
            true,
        )
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        json!({
            "family": "granite_speech_asr",
            "model_type": self.config.model_type,
            "audio_token_index": self.config.audio_token_index,
            "dtype": format!("{:?}", self.dtype),
            "device_kind": format!("{:?}", self.device.kind),
            "sample_rate": self.processor.sample_rate(),
            "n_mels": self.processor.audio_processor.melspec_kwargs.n_mels,
            "projector_downsample_rate": self.processor.audio_processor.projector_downsample_rate,
            "chat_template_bytes": self.chat_template.len(),
            "max_audio_seconds": DEFAULT_MAX_AUDIO_SECONDS,
            "max_timestamp_audio_seconds": TIMESTAMP_MAX_AUDIO_SECONDS,
        })
    }

    fn transcribe_internal(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        prompt: Option<&str>,
        prefix_text: Option<&str>,
        options: GraniteSpeechAsrGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
        on_delta: &mut dyn FnMut(&str),
        emit_deltas: bool,
    ) -> Result<GraniteSpeechAsrTranscriptionOutput> {
        let model_start = Instant::now();
        let prepare_start = Instant::now();
        let features = self.prepare_audio_features(audio, sample_rate)?;
        let mel_prepare = prepare_start.elapsed();
        validate_granite_audio_duration(features.audio_seconds)?;
        let encoder_start = Instant::now();
        let (audio_embeds, audio_stats) = self.runtime.audio_embeddings_with_stats(&features)?;
        let encoder_forward = encoder_start.elapsed();

        let prompt_options = GraniteSpeechPromptOptions {
            task,
            language: language.map(str::to_string),
            custom_prompt: prompt.map(str::to_string),
            prefix_text: prefix_text.map(str::to_string),
            ..GraniteSpeechPromptOptions::default()
        };
        let granite_prompt = self.build_prompt(&prompt_options)?;
        let special_tokens = self.prompt_tokenizer.special_tokens().clone();
        let mut decode = |ids: &[u32]| self.prompt_tokenizer.decode(ids);
        let generation = self.runtime.generate(
            &granite_prompt,
            &special_tokens,
            &audio_embeds,
            options.max_new_tokens,
            &options.stop_token_ids,
            &options.stop_sequences,
            &mut decode,
            cache,
            on_delta,
            emit_deltas,
        )?;
        let model_total = model_start.elapsed();
        let parsed = parse_granite_speech_output(&generation.text);
        let text = parsed.text.clone();
        let timings = GraniteSpeechAsrTimings {
            mel_prepare,
            encoder_forward,
            prefill: generation.stats.timings.prefill,
            decode: generation.stats.timings.decode,
            model_total,
            audio_cache_hit: false,
        };

        Ok(GraniteSpeechAsrTranscriptionOutput {
            text,
            language: language.map(str::to_string),
            diagnostics: Some(granite_diagnostics(
                &features,
                &granite_prompt,
                &generation,
                &parsed,
                self.dtype,
                &self.device,
                timings,
                audio_stats,
            )),
        })
    }
}

fn granite_speech_invocation_contract(
    config: &GraniteSpeechConfig,
    dtype: DType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let text = &config.text_config;
    let domain_id = StateDomainId::new(1);
    let domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
        domain: domain_id,
        clock: StateClock::DecoderTokens,
        num_layers: text.num_hidden_layers,
        num_query_heads: text.num_attention_heads,
        num_kv_heads: text.num_key_value_heads,
        key_head_dim: config.decoder_head_dim(),
        value_head_dim: config.decoder_head_dim(),
        sliding_window: None,
        storage_dtype: dtype,
        preferred_page_tokens,
        prefix: PrefixPolicy::Disabled,
    })?;
    let mut contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::PagedAttention(domain)],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![domain_id],
            prefix_shareable: false,
        }],
    };
    for domain in &mut contract.domains {
        let StateDomainSpec::PagedAttention(domain) = domain else {
            return Err(Error::ModelLoadError(
                "Granite Speech invocation state must be paged attention".into(),
            ));
        };
        domain.header.scope = StateScope::Invocation;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::None;
    }
    for group in &mut contract.groups {
        group.prefix_shareable = false;
    }
    contract.validate()?;
    Ok(contract)
}

fn granite_speech_retained_contract(
    mut contract: InferenceStateContract,
) -> Result<InferenceStateContract> {
    for domain in &mut contract.domains {
        let StateDomainSpec::PagedAttention(domain) = domain else {
            return Err(Error::ModelLoadError(
                "Granite Speech retained state must be paged attention".into(),
            ));
        };
        domain.header.scope = StateScope::Retained;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::Transactional;
    }
    contract.validate()?;
    Ok(contract)
}

fn granite_speech_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    retained: InferenceStateContract,
    invocation: InferenceStateContract,
    max_context_tokens: usize,
) -> Result<GraniteSpeechPhysicalStateSpec> {
    if stage_graphs.is_empty() || max_context_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Granite Speech invocation state requires stages and a non-zero text context".into(),
        ));
    }
    let normal_graphs = stage_graphs
        .iter()
        .filter(|stages| {
            stages.iter().any(|stage| {
                matches!(
                    stage.selector,
                    crate::engine::StageWorkSelector::SequencePrefill
                        | crate::engine::StageWorkSelector::SequenceDecode
                )
            })
        })
        .count();
    let atomic_graphs = stage_graphs
        .iter()
        .filter(|stages| {
            stages.len() == 1
                && stages[0].selector == crate::engine::StageWorkSelector::Atomic
                && stages[0].batch_mode == crate::engine::NativeBatchMode::None
        })
        .count();
    let pipeline_graphs = stage_graphs
        .iter()
        .filter(|stages| {
            stages.len() == 1
                && matches!(
                    stages[0].selector,
                    crate::engine::StageWorkSelector::Pipeline { ordinal: None }
                )
                && stages[0].batch_mode == crate::engine::NativeBatchMode::None
                && stages[0].shape_policy == crate::engine::StageShapePolicy::Exact
        })
        .count();
    if normal_graphs > 0 {
        let valid_normal = normal_graphs == 1
            && stage_graphs.iter().any(|stages| {
                let scalar_stage = |selector| {
                    stages.iter().any(|stage| {
                        stage.selector == selector
                            && stage.batch_mode == crate::engine::NativeBatchMode::None
                            && stage.shape_policy == crate::engine::StageShapePolicy::Exact
                            && stage.concurrency == crate::engine::ConcurrencyClass::Exclusive
                            && stage.max_batch_size == 1
                    })
                };
                let preparation = stages.iter().any(|stage| {
                    if stage.selector != crate::engine::StageWorkSelector::PreSequencePreparation {
                        return false;
                    }
                    if stage.max_batch_size == 1 {
                        return stage.batch_mode == crate::engine::NativeBatchMode::None
                            && stage.shape_policy == crate::engine::StageShapePolicy::Exact
                            && stage.concurrency == crate::engine::ConcurrencyClass::Exclusive;
                    }
                    let batch_workspace = u64::try_from(stage.max_batch_size)
                        .ok()
                        .and_then(|width| stage.workspace_per_row_bytes.checked_mul(width));
                    stage.batch_mode == crate::engine::NativeBatchMode::Static
                        && stage.shape_policy == crate::engine::StageShapePolicy::Padded
                        && stage.concurrency == crate::engine::ConcurrencyClass::Batchable
                        && stage.workspace_base_bytes == 0
                        && stage.workspace_per_row_bytes == 0
                        && stage.workspace_per_work_unit_bytes == 0
                        && batch_workspace.is_some_and(|bytes| stage.max_workspace_bytes >= bytes)
                });
                let decode = stages.iter().any(|stage| {
                    let batch_workspace = u64::try_from(stage.max_batch_size)
                        .ok()
                        .and_then(|width| stage.workspace_per_row_bytes.checked_mul(width));
                    stage.selector == crate::engine::StageWorkSelector::SequenceDecode
                        && stage.batch_mode == crate::engine::NativeBatchMode::Continuous
                        && stage.shape_policy == crate::engine::StageShapePolicy::Ragged
                        && stage.concurrency == crate::engine::ConcurrencyClass::Batchable
                        && stage.max_batch_size > 0
                        && stage.workspace_base_bytes == 0
                        && stage.workspace_per_row_bytes > 0
                        && stage.workspace_per_work_unit_bytes == 0
                        && batch_workspace.is_some_and(|bytes| stage.max_workspace_bytes >= bytes)
                });
                stages.len() == 3
                    && preparation
                    && scalar_stage(crate::engine::StageWorkSelector::SequencePrefill)
                    && decode
            });
        if !valid_normal || atomic_graphs != 1 || stage_graphs.len() != 2 {
            return Err(Error::ModelLoadError(
                "Granite Speech ASR requires authenticated preparation batching, scalar exact prefill, continuous ragged decode, and one atomic compatibility graph"
                    .into(),
            ));
        }
    } else if pipeline_graphs != stage_graphs.len() {
        return Err(Error::ModelLoadError(
            "Granite Speech invocation-only capability requires exact pipeline graphs".into(),
        ));
    }
    let max_tokens = u64::try_from(max_context_tokens)
        .map_err(|_| Error::ModelLoadError("Granite Speech context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| {
            Error::ModelLoadError("Granite Speech invocation contract is empty".into())
        })?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let uses_invocation_state = matches!(
                stage.selector,
                crate::engine::StageWorkSelector::Atomic
                    | crate::engine::StageWorkSelector::Pipeline { ordinal: None }
            );
            let mut domains = if uses_invocation_state {
                invocation
                    .domains
                    .iter()
                    .cloned()
                    .map(|state| {
                        Ok(InvocationWorkspaceDomain::State {
                            placement: state.header().placement,
                            formula: WorkspaceFormula {
                                fixed_bytes: granite_speech_paged_invocation_bytes(
                                    &state, max_tokens,
                                )?,
                                dimensions: vec![],
                                terms: vec![],
                            },
                            state,
                            capacity: InvocationStateCapacity::decoder_context(max_tokens)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            if stage.max_workspace_bytes > 0 {
                let scratch_bytes = if stage.workspace_per_row_bytes > 0 {
                    stage.workspace_per_row_bytes
                } else {
                    stage.max_workspace_bytes
                };
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError(
                            "Granite Speech execution stage count exceeds u32".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("Granite Speech scratch domain id overflow".into())
                    })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: crate::kv::v2::StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: scratch_bytes,
                        dimensions: vec![],
                        terms: vec![],
                    },
                });
            }
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: if stage.selector
                    == crate::engine::StageWorkSelector::PreSequencePreparation
                    && stage.batch_mode == crate::engine::NativeBatchMode::Static
                    && stage.workspace_per_row_bytes == 0
                {
                    InvocationLeaseScope::PerStageBatch
                } else {
                    InvocationLeaseScope::PerRow
                },
                groups: if uses_invocation_state {
                    invocation.groups.clone()
                } else {
                    Vec::new()
                },
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: invocation_stages,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let uses_retained = stage_graphs.iter().any(|stages| {
        stages.iter().any(|stage| {
            matches!(
                stage.selector,
                crate::engine::StageWorkSelector::SequencePrefill
                    | crate::engine::StageWorkSelector::SequenceDecode
            )
        })
    });
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: if uses_retained {
            RetainedStateCapability::Managed {
                contract: retained.clone(),
            }
        } else {
            RetainedStateCapability::Stateless
        },
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(GraniteSpeechPhysicalStateSpec {
        descriptor,
        retained: uses_retained.then_some(retained),
        retained_max_tokens: uses_retained.then_some(max_context_tokens),
        invocation,
    })
}

fn granite_speech_paged_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "Granite Speech invocation workspace is not paged attention".into(),
        ));
    };
    let page_tokens = u64::from(spec.page_size.preferred_tokens);
    let rounded_tokens = max_tokens
        .checked_add(page_tokens.saturating_sub(1))
        .and_then(|tokens| tokens.checked_div(page_tokens))
        .and_then(|pages| pages.checked_mul(page_tokens))
        .ok_or_else(|| Error::ModelLoadError("Granite Speech page capacity overflow".into()))?;
    let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
        let layer_elements = u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::ModelLoadError("Granite Speech KV geometry overflow".into()))?;
        total
            .checked_add(layer_elements)
            .ok_or_else(|| Error::ModelLoadError("Granite Speech KV geometry overflow".into()))
    })?;
    let element_bytes = spec
        .accepted_dtypes
        .iter()
        .map(|dtype| match dtype {
            StateDType::F32 => Ok(4_u64),
            StateDType::F16 | StateDType::Bf16 => Ok(2_u64),
            StateDType::I64 | StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
                "Granite Speech invocation paging requires a dense loaded KV dtype".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .ok_or_else(|| Error::ModelLoadError("Granite Speech KV dtype set is empty".into()))?;
    elements_per_token
        .checked_mul(rounded_tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| {
            Error::ModelLoadError("Granite Speech invocation byte bound overflow".into())
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraniteSpeechAsrTimings {
    mel_prepare: Duration,
    encoder_forward: Duration,
    prefill: Duration,
    decode: Duration,
    model_total: Duration,
    audio_cache_hit: bool,
}

fn validate_granite_audio_duration(audio_seconds: f32) -> Result<()> {
    if audio_seconds <= DEFAULT_MAX_AUDIO_SECONDS {
        return Ok(());
    }
    Err(Error::InvalidInput(format!(
        "Granite Speech ASR supports audio up to {DEFAULT_MAX_AUDIO_SECONDS:.0}s, got {audio_seconds:.1}s"
    )))
}

fn select_granite_speech_dtype(device: &DeviceProfile, config_hint: Option<&str>) -> Result<DType> {
    let explicit = std::env::var("IZWI_GRANITE_SPEECH_DTYPE")
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty());

    if let Some(raw) = explicit.as_deref() {
        return resolve_explicit_granite_speech_dtype(device, raw);
    }

    config_hint
        .map(|raw| {
            device.select_model_dtype_checked(
                ModelFamily::GraniteSpeechAsr,
                Some(raw),
                "Granite Speech ASR",
            )
        })
        .transpose()
        .map(|dtype| {
            dtype.unwrap_or_else(|| device.select_model_dtype(ModelFamily::GraniteSpeechAsr, None))
        })
}

fn resolve_explicit_granite_speech_dtype(device: &DeviceProfile, raw: &str) -> Result<DType> {
    let dtype = parse_dtype_name(raw).ok_or_else(|| {
        Error::InvalidInput(format!(
            "Invalid Granite Speech ASR dtype override {raw:?}: expected one of f32, f16, or bf16"
        ))
    })?;
    match device.kind {
        DeviceKind::Metal => match dtype {
            DType::F32 => Ok(DType::F32),
            DType::F16 if device.capabilities.supports_f16 => Ok(DType::F16),
            DType::F16 => Err(Error::InvalidInput(
                "Invalid Granite Speech ASR dtype override \"f16\": Metal device does not report F16 support"
                    .to_string(),
            )),
            DType::BF16 => Err(Error::InvalidInput(
                "Invalid Granite Speech ASR dtype override \"bf16\": BF16 is not supported on Metal"
                    .to_string(),
            )),
            _ => Err(Error::InvalidInput(format!(
                "Invalid Granite Speech ASR dtype override {raw:?}: dtype is not supported"
            ))),
        },
        _ => device.select_model_dtype_checked(
            ModelFamily::GraniteSpeechAsr,
            Some(raw),
            "Granite Speech ASR",
        ),
    }
}

fn granite_diagnostics(
    features: &GraniteSpeechAudioFeatures,
    prompt: &GraniteSpeechPrompt,
    generation: &GraniteSpeechGeneration,
    parsed: &GraniteSpeechParsedTranscript,
    dtype: DType,
    device: &DeviceProfile,
    timings: GraniteSpeechAsrTimings,
    audio_stats: GraniteSpeechAudioEmbeddingStats,
) -> serde_json::Value {
    let execution = json!({
        "dense_decode_cache": generation.stats.dense_decode_cache_enabled,
        "dense_decode_cache_configured": generation.stats.dense_decode_cache_enabled,
        "dense_head_decode_enabled": generation.stats.dense_head_decode_enabled,
        "qkv_projection_fused": generation.stats.qkv_projection_fused,
        "gate_up_projection_fused": generation.stats.gate_up_projection_fused,
        "rope_cache_precomputed": generation.stats.rope_cache_precomputed,
        "cuda_dense_decode_cache": generation.stats.dense_decode_cache_enabled,
        "cuda_device_argmax": generation.stats.cuda_device_argmax,
        "residual_branches_prescaled": generation.stats.residual_branches_prescaled,
        "f16_lm_head": generation.stats.f16_lm_head,
        "f16_qkv": generation.stats.f16_qkv,
        "f16_attention_core": generation.stats.f16_attention_core,
        "f16_mlp": generation.stats.f16_mlp,
        "f16_attention_output": generation.stats.f16_attention_output,
        "dense_decode_preallocated": generation.stats.dense_decode_preallocated,
        "dense_decode_initial_capacity": generation.stats.dense_decode_initial_capacity,
        "deferred_stop_check": generation.stats.deferred_stop_check,
        "chunked_stop_check": generation.stats.chunked_stop_check,
        "stop_check_interval": generation.stats.stop_check_interval,
        "dense_decode_max_tokens": generation.stats.dense_decode_max_tokens,
        "audio_embedding_cache_hit": timings.audio_cache_hit,
    });

    json!({
        "family": "granite_speech_asr",
        "dtype": format!("{:?}", dtype),
        "device_kind": format!("{:?}", device.kind),
        "audio_seconds": features.audio_seconds,
        "sample_rate": features.sample_rate,
        "mel_frames": features.mel_frames,
        "mel_bins": features.mel_bins,
        "encoder_frames": features.encoder_frames,
        "encoder_dim": features.encoder_dim,
        "projected_audio_tokens": generation.stats.audio_tokens,
        "prompt_tokens": generation.stats.prompt_tokens,
        "prompt_prefix_tokens": prompt.prefix_text_token_count,
        "prompt_audio_placeholders": prompt.audio_token_positions.len(),
        "prompt": {
            "prompt_tokens": generation.stats.prompt_tokens,
            "prefix_tokens": prompt.prefix_text_token_count,
            "audio_placeholders": prompt.audio_token_positions.len(),
        },
        "audio": {
            "audio_tokens": generation.stats.audio_tokens,
            "mel_frames": features.mel_frames,
            "encoder_frames": features.encoder_frames,
            "encoder_dim": features.encoder_dim,
            "conformer_context_size": audio_stats.conformer_context_size,
            "conformer_blocks": audio_stats.conformer_blocks,
            "conformer_pad_frames": audio_stats.conformer_pad_frames,
            "conformer_layers": audio_stats.conformer_layers,
            "qformer_windows": audio_stats.qformer_windows,
            "qformer_window_size": audio_stats.qformer_window_size,
            "qformer_queries_per_window": audio_stats.qformer_queries_per_window,
            "qformer_layers": audio_stats.qformer_layers,
        },
        "generated_tokens": generation.stats.generated_tokens,
        "stop_reason": generation.stats.stop_reason,
        "stop_token": generation.stats.stop_token,
        "decode": {
            "generated_tokens": generation.stats.generated_tokens,
            "max_new_tokens": generation.stats.max_new_tokens,
            "stop_reason": generation.stats.stop_reason,
            "stop_token": generation.stats.stop_token,
        },
        "token_debug": granite_token_debug_enabled().then(|| json!({
            "token_ids": generation.token_ids,
            "token_count": generation.token_ids.len(),
            "decoded_text": generation.text,
        })),
        "execution": execution,
        "decode_profile": generation
            .stats
            .decode_profile
            .as_ref()
            .map(granite_decode_profile_json),
        "timings_ms": {
            "mel_prepare": duration_ms(timings.mel_prepare),
            "encoder_forward": duration_ms(timings.encoder_forward),
            "audio_input_upload": duration_ms(audio_stats.upload),
            "audio_encoder": duration_ms(audio_stats.encoder),
            "audio_projector": duration_ms(audio_stats.projector),
            "audio_frontend_total": duration_ms(timings.mel_prepare + timings.encoder_forward),
            "prefill": duration_ms(timings.prefill),
            "decode": duration_ms(timings.decode),
            "generation_total": duration_ms(timings.prefill + timings.decode),
            "model_non_generation": duration_ms(
                timings.model_total.saturating_sub(timings.prefill + timings.decode),
            ),
            "model_total": duration_ms(timings.model_total),
        },
        "speaker_segments": parsed.segments.iter().map(|segment| {
            json!({
                "speaker": segment.speaker,
                "text": segment.text,
            })
        }).collect::<Vec<_>>(),
        "timestamp_words": parsed.timestamp_words.iter().map(|word| {
            json!({
                "word": word.word,
                "end_time_seconds": word.end_time_seconds,
            })
        }).collect::<Vec<_>>(),
    })
}

fn granite_decode_profile_json(profile: &GraniteSpeechDecodeProfile) -> serde_json::Value {
    let layer_totals = granite_layer_totals(&profile.layers);
    json!({
        "enabled": true,
        "timing_kind": profile.timing_kind,
        "steps": profile.steps,
        "layer_count": profile.layer_count,
        "step_total_ms": duration_stats_json(&profile.step_total_samples),
        "loop_totals_ms": decode_loop_profile_json(profile.totals),
        "forward_totals_ms": forward_profile_json(profile.forward),
        "decoder_totals_ms": layer_profile_json(layer_totals),
        "layers": profile.layers.iter().enumerate().map(|(idx, layer)| {
            json!({
                "index": idx,
                "timings_ms": layer_profile_json(*layer),
            })
        }).collect::<Vec<_>>(),
    })
}

fn granite_layer_totals(
    layers: &[GraniteSpeechLayerDecodeProfile],
) -> GraniteSpeechLayerDecodeProfile {
    let mut total = GraniteSpeechLayerDecodeProfile::default();
    for layer in layers {
        total.total += layer.total;
        total.input_norm += layer.input_norm;
        total.attention.qkv += layer.attention.qkv;
        total.attention.rope += layer.attention.rope;
        total.attention.cache += layer.attention.cache;
        total.attention.kernel += layer.attention.kernel;
        total.attention.output += layer.attention.output;
        total.attention.dense_head_calls += layer.attention.dense_head_calls;
        total.attention.dense_head_fused += layer.attention.dense_head_fused;
        total.attention.dense_head_fallback += layer.attention.dense_head_fallback;
        total.attention.materialized_decode_calls += layer.attention.materialized_decode_calls;
        total.attention.prefill_attention_calls += layer.attention.prefill_attention_calls;
        total.post_attention_norm += layer.post_attention_norm;
        total.mlp.gate_up += layer.mlp.gate_up;
        total.mlp.activation += layer.mlp.activation;
        total.mlp.down += layer.mlp.down;
        total.mlp.fused_silu_mul_attempts += layer.mlp.fused_silu_mul_attempts;
        total.mlp.fused_silu_mul_custom += layer.mlp.fused_silu_mul_custom;
        total.mlp.fused_silu_mul_fallback += layer.mlp.fused_silu_mul_fallback;
        total.residual += layer.residual;
    }
    total
}

fn decode_loop_profile_json(profile: GraniteSpeechDecodeLoopProfile) -> serde_json::Value {
    json!({
        "argmax": duration_ms(profile.argmax),
        "scalar_read": duration_ms(profile.scalar_read),
        "stop_check": duration_ms(profile.stop_check),
        "model_forward": duration_ms(profile.model_forward),
        "text_decode": duration_ms(profile.text_decode),
        "delta_emit": duration_ms(profile.delta_emit),
        "step_total": duration_ms(profile.step_total),
    })
}

fn forward_profile_json(profile: GraniteSpeechForwardProfile) -> serde_json::Value {
    json!({
        "token_embedding": duration_ms(profile.token_embedding),
        "rope_build": duration_ms(profile.rope_build),
        "layers_total": duration_ms(profile.layers_total),
        "final_norm": duration_ms(profile.final_norm),
        "lm_head": duration_ms(profile.lm_head),
        "lm_head_f16_calls": profile.lm_head_f16_calls,
        "lm_head_f32_calls": profile.lm_head_f32_calls,
    })
}

fn layer_profile_json(profile: GraniteSpeechLayerDecodeProfile) -> serde_json::Value {
    json!({
        "total": duration_ms(profile.total),
        "input_norm": duration_ms(profile.input_norm),
        "attention": attention_profile_json(profile.attention),
        "post_attention_norm": duration_ms(profile.post_attention_norm),
        "mlp": mlp_profile_json(profile.mlp),
        "residual": duration_ms(profile.residual),
    })
}

fn attention_profile_json(profile: GraniteSpeechAttentionDecodeProfile) -> serde_json::Value {
    json!({
        "qkv": duration_ms(profile.qkv),
        "rope": duration_ms(profile.rope),
        "cache": duration_ms(profile.cache),
        "kernel": duration_ms(profile.kernel),
        "output": duration_ms(profile.output),
        "dense_head_calls": profile.dense_head_calls,
        "dense_head_fused": profile.dense_head_fused,
        "dense_head_fallback": profile.dense_head_fallback,
        "materialized_decode_calls": profile.materialized_decode_calls,
        "prefill_attention_calls": profile.prefill_attention_calls,
    })
}

fn mlp_profile_json(profile: GraniteSpeechMlpDecodeProfile) -> serde_json::Value {
    json!({
        "gate_up": duration_ms(profile.gate_up),
        "activation": duration_ms(profile.activation),
        "down": duration_ms(profile.down),
        "fused_silu_mul_attempts": profile.fused_silu_mul_attempts,
        "fused_silu_mul_custom": profile.fused_silu_mul_custom,
        "fused_silu_mul_fallback": profile.fused_silu_mul_fallback,
    })
}

fn duration_stats_json(samples: &[Duration]) -> serde_json::Value {
    if samples.is_empty() {
        return json!({
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        });
    }
    let mut values = samples
        .iter()
        .map(|value| duration_ms(*value))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let sum = values.iter().sum::<f64>();
    json!({
        "count": values.len(),
        "avg": sum / values.len() as f64,
        "p50": percentile_sorted(&values, 0.50),
        "p95": percentile_sorted(&values, 0.95),
        "max": values.last().copied().unwrap_or(0.0),
    })
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[idx.min(values.len() - 1)]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn granite_token_debug_enabled() -> bool {
    std::env::var("IZWI_GRANITE_TOKEN_DEBUG")
        .ok()
        .or_else(|| std::env::var("IZWI_GRANITE_TOKEN_DIAGNOSTICS").ok())
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

pub fn ensure_granite_speech_artifacts(model_dir: &Path) -> Result<Vec<PathBuf>> {
    for file in REQUIRED_ARTIFACTS {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Granite Speech artifact {}",
                path.display()
            )));
        }
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let raw = fs::read_to_string(&index_path).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to read Granite Speech safetensors index {}: {err}",
            index_path.display()
        ))
    })?;
    let index: serde_json::Value = serde_json::from_str(&raw).map_err(|err| {
        Error::ModelLoadError(format!(
            "Failed to parse Granite Speech safetensors index {}: {err}",
            index_path.display()
        ))
    })?;
    let weight_map = index
        .get("weight_map")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            Error::ModelLoadError(
                "Invalid Granite Speech model.safetensors.index.json: missing weight_map"
                    .to_string(),
            )
        })?;

    let mut shard_files = BTreeSet::new();
    for value in weight_map.values() {
        let Some(file) = value.as_str() else {
            return Err(Error::ModelLoadError(
                "Invalid Granite Speech safetensors index: non-string shard filename".to_string(),
            ));
        };
        validate_shard_filename(file)?;
        shard_files.insert(file.to_string());
    }

    if shard_files.is_empty() {
        return Err(Error::ModelLoadError(
            "Granite Speech safetensors index contains no shard files".to_string(),
        ));
    }

    let mut shard_paths = Vec::with_capacity(shard_files.len());
    for file in shard_files {
        let path = model_dir.join(&file);
        if !path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Granite Speech safetensors shard {}",
                path.display()
            )));
        }
        shard_paths.push(path);
    }

    Ok(shard_paths)
}

fn validate_shard_filename(file: &str) -> Result<()> {
    let path = Path::new(file);
    let is_plain_relative = path
        .parent()
        .is_none_or(|parent| parent.as_os_str().is_empty())
        && path.file_name().is_some()
        && !path.is_absolute();
    if is_plain_relative {
        Ok(())
    } else {
        Err(Error::ModelLoadError(format!(
            "Invalid Granite Speech safetensors shard path '{file}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::backends::DeviceCapabilities;
    use crate::engine::ModelInstanceId;
    use crate::engine::{
        ExecutionMode, ExecutionProfile, NativeBatchMode, StageId, StageWorkSelector,
    };
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use uuid::Uuid;

    static GRANITE_DTYPE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn granite_test_invocation_contract() -> InferenceStateContract {
        let mut contract = crate::kv::v2::test_contract();
        for domain in &mut contract.domains {
            let StateDomainSpec::PagedAttention(domain) = domain else {
                panic!("test contract must be paged attention")
            };
            domain.header.scope = StateScope::Invocation;
            domain.header.checkpoint = CheckpointPolicy::None;
            domain.header.prefix = PrefixPolicy::Disabled;
        }
        for group in &mut contract.groups {
            group.prefix_shareable = false;
        }
        contract.validate().unwrap();
        contract
    }

    fn granite_test_stage(id: u32, selector: StageWorkSelector) -> StageDescriptor {
        let mode = if selector == StageWorkSelector::Atomic {
            ExecutionMode::Atomic
        } else {
            ExecutionMode::Sequence
        };
        let mut profile = ExecutionProfile::fail_closed(BackendKind::Cpu, None, mode);
        let batch_mode = if selector == StageWorkSelector::SequenceDecode {
            profile.max_batch_size = 4;
            NativeBatchMode::Continuous
        } else {
            profile.max_batch_size = 1;
            NativeBatchMode::None
        };
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(id),
            format!("granite.test.{id}"),
            &profile,
            batch_mode,
        );
        stage.selector = selector;
        if selector == StageWorkSelector::SequenceDecode {
            stage.workspace_per_row_bytes = 16;
            stage.max_workspace_bytes = 64;
        }
        stage
    }

    fn granite_test_static_preparation_stage(id: u32) -> StageDescriptor {
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.max_batch_size = 4;
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(id),
            format!("granite.test.{id}"),
            &profile,
            NativeBatchMode::Static,
        );
        stage.selector = StageWorkSelector::PreSequencePreparation;
        stage.max_workspace_bytes = 128;
        stage
    }

    #[test]
    fn preparation_batch_useful_elements_include_encoder_input_and_artifact() {
        let geometry = GraniteSpeechPreparedGeometry {
            audio_samples: 32,
            encoder_frames: 5,
            encoder_dim: 3,
            prompt_tokens: 7,
            audio_tokens: 2,
            embedding_elements: 28,
            preparation_workspace_bytes: 64,
            retained_device_bytes: 112,
        };
        assert_eq!(geometry.batch_useful_tensor_elements().unwrap(), 43);
    }

    #[test]
    fn physical_state_separates_retained_normal_and_atomic_compatibility_graphs() {
        let invocation = granite_test_invocation_contract();
        let retained = granite_speech_retained_contract(invocation.clone()).unwrap();
        let normal = vec![
            granite_test_stage(0, StageWorkSelector::PreSequencePreparation),
            granite_test_stage(1, StageWorkSelector::SequencePrefill),
            granite_test_stage(2, StageWorkSelector::SequenceDecode),
        ];
        let atomic = vec![granite_test_stage(0, StageWorkSelector::Atomic)];

        let spec = granite_speech_physical_state_spec(
            &[normal.as_slice(), atomic.as_slice()],
            retained,
            invocation,
            128,
        )
        .unwrap();

        assert!(spec.retained.is_some());
        assert_eq!(spec.retained_max_tokens, Some(128));
        assert!(matches!(
            spec.descriptor.retained,
            RetainedStateCapability::Managed { .. }
        ));
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("Granite test descriptor must use bounded invocation profiles")
        };
        let normal_fingerprint = stage_graph_fingerprint(&normal).unwrap();
        let normal_profile = profiles
            .iter()
            .find(|profile| profile.stage_graph_fingerprint == normal_fingerprint)
            .unwrap();
        assert!(normal_profile
            .stages
            .iter()
            .all(|stage| stage.groups.is_empty()));
        let atomic_fingerprint = stage_graph_fingerprint(&atomic).unwrap();
        let atomic_profile = profiles
            .iter()
            .find(|profile| profile.stage_graph_fingerprint == atomic_fingerprint)
            .unwrap();
        assert!(atomic_profile
            .stages
            .iter()
            .all(|stage| !stage.groups.is_empty()));
    }

    #[test]
    fn physical_state_authenticates_static_padded_preparation() {
        let invocation = granite_test_invocation_contract();
        let retained = granite_speech_retained_contract(invocation.clone()).unwrap();
        let normal = vec![
            granite_test_static_preparation_stage(0),
            granite_test_stage(1, StageWorkSelector::SequencePrefill),
            granite_test_stage(2, StageWorkSelector::SequenceDecode),
        ];
        let atomic = vec![granite_test_stage(0, StageWorkSelector::Atomic)];
        granite_speech_physical_state_spec(
            &[normal.as_slice(), atomic.as_slice()],
            retained,
            invocation,
            128,
        )
        .unwrap();
    }

    #[test]
    fn physical_state_rejects_incomplete_retained_graph() {
        let invocation = granite_test_invocation_contract();
        let retained = granite_speech_retained_contract(invocation.clone()).unwrap();
        let incomplete = vec![
            granite_test_stage(0, StageWorkSelector::PreSequencePreparation),
            granite_test_stage(1, StageWorkSelector::SequencePrefill),
        ];
        let atomic = vec![granite_test_stage(0, StageWorkSelector::Atomic)];

        assert!(granite_speech_physical_state_spec(
            &[incomplete.as_slice(), atomic.as_slice()],
            retained,
            invocation,
            128,
        )
        .is_err());
    }

    #[test]
    fn invocation_only_physical_state_authenticates_pipeline_work() {
        let invocation = granite_test_invocation_contract();
        let retained = granite_speech_retained_contract(invocation.clone()).unwrap();
        let pipeline = vec![granite_test_stage(
            0,
            StageWorkSelector::Pipeline { ordinal: None },
        )];

        let spec =
            granite_speech_physical_state_spec(&[pipeline.as_slice()], retained, invocation, 128)
                .unwrap();
        assert!(spec.retained.is_none());
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("Granite test descriptor must use bounded invocation profiles")
        };
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0]
            .stages
            .iter()
            .all(|stage| !stage.groups.is_empty()));
    }

    fn temp_model_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("granite-speech-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_required_non_weight_files(model_dir: &Path) {
        std::fs::write(model_dir.join("chat_template.jinja"), "{{ prompt }}").unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("generation_config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("processor_config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(model_dir.join("tokenizer_config.json"), "{}").unwrap();
    }

    fn test_profile(kind: DeviceKind, supports_f16: bool) -> DeviceProfile {
        DeviceProfile {
            device: candle_core::Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_f16,
                prefers_f32: kind.is_metal(),
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    fn retained_test_caches() -> (PhysicalPagedKvCache, PhysicalPagedKvCache) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena: Arc<dyn KvArena> = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: KvArenaId {
                    model_instance: ModelInstanceId::new(177),
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    generation: 1,
                },
                group: KvGroupId::new(0),
                page_tokens: 2,
                capacity_pages: 4,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .unwrap(),
        );
        let cache = || {
            let blocks = (0..4)
                .map(|index| CacheBlockRef {
                    arena: arena.id(),
                    group: arena.config().group,
                    index,
                    slot_generation: 1,
                })
                .collect();
            PhysicalPagedKvCache::new(arena.clone(), vec![binding], blocks, 0).unwrap()
        };
        (cache(), cache())
    }

    fn retained_test_state(
        cache: PhysicalPagedKvCache,
        state_id: u64,
        pending_token: Option<u32>,
    ) -> GraniteSpeechDecodeState {
        GraniteSpeechDecodeState {
            cache,
            artifact: Arc::new(GraniteSpeechPreparedPromptArtifact {
                model_identity: 9,
                embeddings: Tensor::zeros((1, 1, 4), DType::F32, &candle_core::Device::Cpu)
                    .unwrap(),
                prompt_tokens: 0,
                audio_tokens: 1,
                language: Some("en".into()),
            }),
            prefill_progress: 0,
            pending_token,
            pos: 0,
            generated_ids: vec![],
            incremental_decoder: IncrementalDecoder::new(true),
            rendered: String::new(),
            published_len: 0,
            stop_holdback_bytes: 0,
            stop_tokens: vec![2],
            stop_sequences: vec![],
            max_new_tokens: 4,
            finished: false,
            stop_reason: "max_tokens",
            stop_token: None,
            state_id,
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
        }
    }

    #[test]
    fn final_prefill_token_is_staged_without_early_publication() {
        let (cache_a, _) = retained_test_caches();
        let mut state = retained_test_state(cache_a, 21, None);

        stage_granite_first_decode_token(&mut state, 2);
        assert_eq!(state.pending_token, Some(2));
        assert!(state.generated_ids.is_empty());
        assert!(state.rendered.is_empty());
        assert!(!state.finished);
    }

    #[test]
    fn decode_checkpoint_restores_staged_token_and_cursor() {
        let (cache_a, replacement_a) = retained_test_caches();
        let mut state = retained_test_state(cache_a, 31, Some(6));

        let mut checkpoint = state.begin_managed_quantum(replacement_a).unwrap();
        state.pending_token = Some(1);
        state.pos = 1;
        state.managed_completions_drained = false;
        state.rollback_managed_quantum(&mut checkpoint).unwrap();

        assert_eq!(state.pending_token, Some(6));
        assert_eq!(state.pos, 0);
        assert!(state.managed_completions_drained);
    }

    #[test]
    fn retained_decode_checkpoint_rolls_back_staged_progress_and_is_single_use() {
        let (cache, replacement) = retained_test_caches();
        let artifact = Arc::new(GraniteSpeechPreparedPromptArtifact {
            model_identity: 9,
            embeddings: Tensor::zeros((1, 3, 4), DType::F32, &candle_core::Device::Cpu).unwrap(),
            prompt_tokens: 3,
            audio_tokens: 1,
            language: Some("en".into()),
        });
        let mut state = GraniteSpeechDecodeState {
            cache,
            artifact,
            prefill_progress: 0,
            pending_token: None,
            pos: 0,
            generated_ids: vec![],
            incremental_decoder: IncrementalDecoder::new(true),
            rendered: String::new(),
            published_len: 0,
            stop_holdback_bytes: 0,
            stop_tokens: vec![2],
            stop_sequences: vec![],
            max_new_tokens: 4,
            finished: false,
            stop_reason: "max_tokens",
            stop_token: None,
            state_id: 11,
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
        };
        let mut checkpoint = state.begin_managed_quantum(replacement).unwrap();
        state.prefill_progress = 3;
        state.pos = 3;
        state.generated_ids.push(7);
        state.pending_token = Some(7);
        state.rendered = "staged".into();
        state.finished = true;

        state.rollback_managed_quantum(&mut checkpoint).unwrap();
        assert_eq!(state.prefill_progress, 0);
        assert_eq!(state.sequence_position(), 0);
        assert!(state.generated_ids.is_empty());
        assert_eq!(state.pending_token, None);
        assert!(state.rendered.is_empty());
        assert!(!state.finished);
        assert!(state
            .rollback_managed_quantum(&mut checkpoint)
            .unwrap_err()
            .to_string()
            .contains("foreign, stale, or out of order"));

        state.stop_sequences = vec!["<stop>".into()];
        state.stop_holdback_bytes = "<stop>".len() - 1;
        state.rendered = "hello<st".into();
        let first = granite_publish_stable_text(&mut state, false).unwrap();
        state.rendered.push_str("op>");
        assert!(truncate_granite_stop_sequence(
            &mut state.rendered,
            &state.stop_sequences
        ));
        let final_delta = granite_publish_stable_text(&mut state, true).unwrap();
        assert_eq!(format!("{first}{final_delta}"), "hello");
    }

    #[test]
    fn granite_speech_dtype_config_hint_keeps_existing_metal_f32_policy() {
        let _guard = GRANITE_DTYPE_ENV_LOCK.lock().unwrap();
        std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
        let metal = test_profile(DeviceKind::Metal, true);

        assert_eq!(
            select_granite_speech_dtype(&metal, Some("torch.bfloat16")).unwrap(),
            DType::F32
        );
    }

    #[test]
    fn granite_speech_dtype_explicit_f16_can_opt_into_metal_half_precision() {
        let _guard = GRANITE_DTYPE_ENV_LOCK.lock().unwrap();
        std::env::set_var("IZWI_GRANITE_SPEECH_DTYPE", "f16");
        let metal = test_profile(DeviceKind::Metal, true);

        assert_eq!(
            select_granite_speech_dtype(&metal, None).unwrap(),
            DType::F16
        );
        std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
    }

    #[test]
    fn granite_speech_dtype_rejects_bf16_on_metal() {
        let _guard = GRANITE_DTYPE_ENV_LOCK.lock().unwrap();
        std::env::set_var("IZWI_GRANITE_SPEECH_DTYPE", "bf16");
        let metal = test_profile(DeviceKind::Metal, true);

        let err = select_granite_speech_dtype(&metal, None).unwrap_err();
        assert!(err.to_string().contains("BF16 is not supported on Metal"));
        std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
    }

    #[test]
    fn artifact_validation_requires_index_weight_map() {
        let dir = temp_model_dir();
        write_required_non_weight_files(&dir);
        std::fs::write(dir.join("model.safetensors.index.json"), "{}").unwrap();
        let err = ensure_granite_speech_artifacts(&dir).unwrap_err();
        assert!(err.to_string().contains("weight_map"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn artifact_validation_returns_unique_shards() {
        let dir = temp_model_dir();
        write_required_non_weight_files(&dir);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00003.safetensors","b":"model-00002-of-00003.safetensors","c":"model-00002-of-00003.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("model-00001-of-00003.safetensors"), []).unwrap();
        std::fs::write(dir.join("model-00002-of-00003.safetensors"), []).unwrap();
        let shards = ensure_granite_speech_artifacts(&dir).unwrap();
        assert_eq!(shards.len(), 2);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn artifact_validation_rejects_path_traversal_shards() {
        let dir = temp_model_dir();
        write_required_non_weight_files(&dir);
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"../model-00001-of-00003.safetensors"}}"#,
        )
        .unwrap();
        let err = ensure_granite_speech_artifacts(&dir).unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid Granite Speech safetensors shard path"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn audio_duration_guard_allows_nine_minutes_only() {
        assert!(validate_granite_audio_duration(DEFAULT_MAX_AUDIO_SECONDS).is_ok());
        let err = validate_granite_audio_duration(DEFAULT_MAX_AUDIO_SECONDS + 0.1).unwrap_err();
        assert!(err.to_string().contains("supports audio up to 540s"));
    }

    #[test]
    fn diagnostics_include_prompt_audio_and_rich_transcript_metadata() {
        let features = GraniteSpeechAudioFeatures {
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            audio_seconds: 0.01,
            mel_frames: 2,
            mel_bins: 80,
            encoder_frames: 1,
            encoder_dim: 160,
            projected_frames_hint: 3,
            log_mel: vec![vec![0.0; 80]; 2],
            input_features: vec![vec![0.0; 160]],
        };
        let prompt = GraniteSpeechPrompt {
            text: "<|audio|>".to_string(),
            input_ids: vec![100_352],
            audio_token_positions: vec![0],
            prefix_text_token_count: 0,
        };
        let generation = GraniteSpeechGeneration {
            token_ids: vec![1, 2],
            text: "[Speaker 1]: hello [T:045]".to_string(),
            stats: GraniteSpeechGenerationStats {
                prompt_tokens: 4,
                audio_tokens: 3,
                generated_tokens: 2,
                max_new_tokens: 128,
                stop_reason: "stop_token".to_string(),
                stop_token: Some(100_257),
                dense_decode_cache_enabled: true,
                dense_head_decode_enabled: false,
                qkv_projection_fused: true,
                gate_up_projection_fused: true,
                rope_cache_precomputed: true,
                cuda_device_argmax: false,
                residual_branches_prescaled: true,
                f16_lm_head: false,
                f16_qkv: false,
                f16_attention_core: false,
                f16_mlp: false,
                f16_attention_output: false,
                dense_decode_preallocated: true,
                dense_decode_initial_capacity: 512,
                deferred_stop_check: true,
                chunked_stop_check: false,
                stop_check_interval: 1,
                dense_decode_max_tokens: 8192,
                timings: GraniteSpeechGenerationTimings {
                    prefill: Duration::from_millis(7),
                    decode: Duration::from_millis(3),
                },
                decode_profile: Some(GraniteSpeechDecodeProfile {
                    timing_kind: "host_wall_clock_no_device_sync",
                    steps: 2,
                    layer_count: 1,
                    step_total_samples: vec![Duration::from_millis(2), Duration::from_millis(3)],
                    totals: GraniteSpeechDecodeLoopProfile {
                        argmax: Duration::from_millis(1),
                        scalar_read: Duration::from_millis(1),
                        stop_check: Duration::from_millis(1),
                        model_forward: Duration::from_millis(4),
                        text_decode: Duration::from_millis(1),
                        delta_emit: Duration::ZERO,
                        step_total: Duration::from_millis(5),
                    },
                    forward: GraniteSpeechForwardProfile {
                        token_embedding: Duration::from_millis(1),
                        rope_build: Duration::from_millis(1),
                        layers_total: Duration::from_millis(2),
                        final_norm: Duration::from_millis(1),
                        lm_head: Duration::from_millis(1),
                        lm_head_f16_calls: 2,
                        lm_head_f32_calls: 0,
                    },
                    layers: vec![GraniteSpeechLayerDecodeProfile {
                        total: Duration::from_millis(2),
                        input_norm: Duration::from_millis(1),
                        attention: GraniteSpeechAttentionDecodeProfile {
                            qkv: Duration::from_millis(1),
                            rope: Duration::from_millis(1),
                            cache: Duration::from_millis(1),
                            kernel: Duration::from_millis(1),
                            output: Duration::from_millis(1),
                            dense_head_calls: 2,
                            dense_head_fused: 2,
                            dense_head_fallback: 0,
                            materialized_decode_calls: 0,
                            prefill_attention_calls: 1,
                        },
                        post_attention_norm: Duration::from_millis(1),
                        mlp: GraniteSpeechMlpDecodeProfile {
                            gate_up: Duration::from_millis(1),
                            activation: Duration::from_millis(1),
                            down: Duration::from_millis(1),
                            fused_silu_mul_attempts: 2,
                            fused_silu_mul_custom: 2,
                            fused_silu_mul_fallback: 0,
                        },
                        residual: Duration::from_millis(1),
                    }],
                }),
            },
        };
        let timings = GraniteSpeechAsrTimings {
            mel_prepare: Duration::from_millis(1),
            encoder_forward: Duration::from_millis(2),
            prefill: generation.stats.timings.prefill,
            decode: generation.stats.timings.decode,
            model_total: Duration::from_millis(12),
            audio_cache_hit: false,
        };
        let audio_stats = GraniteSpeechAudioEmbeddingStats {
            upload: Duration::from_millis(1),
            encoder: Duration::from_millis(2),
            projector: Duration::from_millis(3),
            encoder_frames: 1,
            encoder_dim: 160,
            conformer_context_size: 200,
            conformer_blocks: 1,
            conformer_pad_frames: 199,
            conformer_layers: 16,
            qformer_windows: 1,
            qformer_window_size: 15,
            qformer_queries_per_window: 3,
            qformer_layers: 2,
        };
        let parsed = parse_granite_speech_output(&generation.text);
        let diagnostics = granite_diagnostics(
            &features,
            &prompt,
            &generation,
            &parsed,
            DType::F32,
            &DeviceProfile::cpu(),
            timings,
            audio_stats,
        );

        assert_eq!(diagnostics["projected_audio_tokens"], 3);
        assert_eq!(diagnostics["prompt_prefix_tokens"], 0);
        assert_eq!(diagnostics["prompt_audio_placeholders"], 1);
        assert_eq!(diagnostics["prompt"]["prompt_tokens"], 4);
        assert_eq!(diagnostics["audio"]["audio_tokens"], 3);
        assert_eq!(diagnostics["audio"]["conformer_context_size"], 200);
        assert_eq!(diagnostics["audio"]["conformer_pad_frames"], 199);
        assert_eq!(diagnostics["audio"]["qformer_queries_per_window"], 3);
        assert_eq!(diagnostics["decode"]["generated_tokens"], 2);
        assert_eq!(diagnostics["decode"]["max_new_tokens"], 128);
        assert_eq!(diagnostics["execution"]["dense_decode_cache"], true);
        assert_eq!(
            diagnostics["execution"]["dense_decode_cache_configured"],
            true
        );
        assert_eq!(diagnostics["execution"]["dense_head_decode_enabled"], false);
        assert_eq!(diagnostics["execution"]["qkv_projection_fused"], true);
        assert_eq!(diagnostics["execution"]["gate_up_projection_fused"], true);
        assert_eq!(diagnostics["execution"]["rope_cache_precomputed"], true);
        assert_eq!(diagnostics["execution"]["cuda_device_argmax"], false);
        assert_eq!(
            diagnostics["execution"]["residual_branches_prescaled"],
            true
        );
        assert_eq!(diagnostics["execution"]["dense_decode_preallocated"], true);
        assert_eq!(
            diagnostics["execution"]["dense_decode_initial_capacity"],
            512
        );
        assert_eq!(diagnostics["execution"]["deferred_stop_check"], true);
        assert_eq!(diagnostics["execution"]["chunked_stop_check"], false);
        assert_eq!(diagnostics["execution"]["stop_check_interval"], 1);
        assert_eq!(diagnostics["execution"]["dense_decode_max_tokens"], 8192);
        assert_eq!(diagnostics["execution"]["audio_embedding_cache_hit"], false);
        assert_eq!(diagnostics["decode_profile"]["enabled"], true);
        assert_eq!(
            diagnostics["decode_profile"]["timing_kind"],
            "host_wall_clock_no_device_sync"
        );
        assert_eq!(diagnostics["decode_profile"]["steps"], 2);
        assert_eq!(diagnostics["decode_profile"]["step_total_ms"]["count"], 2);
        assert_eq!(diagnostics["decode_profile"]["step_total_ms"]["p50"], 3.0);
        assert_eq!(
            diagnostics["decode_profile"]["loop_totals_ms"]["model_forward"],
            4.0
        );
        assert_eq!(
            diagnostics["decode_profile"]["decoder_totals_ms"]["attention"]["cache"],
            1.0
        );
        assert_eq!(
            diagnostics["decode_profile"]["forward_totals_ms"]["lm_head_f16_calls"],
            2
        );
        assert_eq!(
            diagnostics["decode_profile"]["decoder_totals_ms"]["attention"]["dense_head_fused"],
            2
        );
        assert_eq!(
            diagnostics["decode_profile"]["decoder_totals_ms"]["attention"]
                ["prefill_attention_calls"],
            1
        );
        assert_eq!(
            diagnostics["decode_profile"]["layers"][0]["timings_ms"]["mlp"]["down"],
            1.0
        );
        assert_eq!(
            diagnostics["decode_profile"]["layers"][0]["timings_ms"]["mlp"]
                ["fused_silu_mul_custom"],
            2
        );
        assert_eq!(diagnostics["timings_ms"]["prefill"], 7.0);
        assert_eq!(diagnostics["timings_ms"]["decode"], 3.0);
        assert_eq!(diagnostics["timings_ms"]["audio_input_upload"], 1.0);
        assert_eq!(diagnostics["timings_ms"]["audio_encoder"], 2.0);
        assert_eq!(diagnostics["timings_ms"]["audio_projector"], 3.0);
        assert_eq!(diagnostics["timings_ms"]["audio_frontend_total"], 3.0);
        assert_eq!(diagnostics["timings_ms"]["generation_total"], 10.0);
        assert_eq!(diagnostics["timings_ms"]["model_non_generation"], 2.0);
        assert_eq!(diagnostics["speaker_segments"][0]["speaker"], "Speaker 1");
        assert_eq!(diagnostics["timestamp_words"][0]["word"], "hello");
    }
}
