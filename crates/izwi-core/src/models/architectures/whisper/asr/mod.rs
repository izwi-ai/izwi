//! Native Whisper Large v3 Turbo ASR model loader and inference.
//!
//! This implementation follows Whisper prompting/decoding conventions used in:
//! - `whisper.cpp` (llama.cpp ecosystem): SOT/lang/task/no-timestamps prefix and
//!   timestamp suppression for text-only decode.
//! - Hugging Face `transformers`: language/task prompt handling and suppress token
//!   masks from `generation_config.json`.

#![allow(clippy::items_after_test_module)]

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, Config as WhisperConfig};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use crate::audio::{MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::backends::state::StaticAttentionLayerValue;
use crate::backends::{backend_kind_for_device, BackendKind, DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::{
    InvocationStaticAttentionLease, RetainedStaticAttentionRuntimeV2,
    RetainedStaticAttentionSequenceId, StageDescriptor, WorkCost,
};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::memory::accounting::TensorStorageAccounting;
use crate::tokenizer::Tokenizer;

use super::model::Whisper as LocalWhisper;
use super::physical::WhisperPhysicalStateSpec;

const SAMPLE_RATE: u32 = whisper::SAMPLE_RATE as u32;
const DEFAULT_MAX_NEW_TOKENS: usize = 448;
const MAX_AUDIO_SECONDS_HINT: f32 = whisper::CHUNK_LENGTH as f32;
const DEFAULT_TEMPERATURE_FALLBACK_INC: f32 = 0.2;
const DEFAULT_MAX_FALLBACK_RETRIES: usize = 1;
const DEFAULT_ADAPTIVE_MAX_NEW_TOKENS_PER_SECOND: f32 = 12.0;
const DEFAULT_ADAPTIVE_MIN_NEW_TOKENS: usize = 32;
const DEFAULT_ADAPTIVE_BUDGET_BUFFER_TOKENS: usize = 8;
const DEFAULT_SILENCE_TRIM_THRESHOLD_SCALE: f32 = 0.02;
const DEFAULT_SILENCE_TRIM_MIN_ABS: f32 = 0.0015;
const DEFAULT_SILENCE_TRIM_MARGIN_MS: usize = 120;
const DEFAULT_SILENCE_TRIM_MIN_LEADING_MS: usize = 500;
const DEFAULT_SILENCE_TRIM_MIN_TRAILING_MS: usize = 160;
const DEFAULT_SILENCE_TRIM_MIN_CLIP_SECS: f32 = 0.8;
const DEFAULT_INITIAL_PROMPT_MAX_TOKENS: usize = 224;
const DEFAULT_LOGPROB_THRESHOLD: f32 = -1.0;
const DEFAULT_NO_SPEECH_THRESHOLD: f32 = 0.6;
const REPETITION_GUARD_MIN_SPAN_TOKENS: usize = 8;
const REPETITION_GUARD_MAX_SPAN_TOKENS: usize = 96;
const REPETITION_GUARD_MIN_TOTAL_TOKENS: usize = 20;
static NEXT_WHISPER_PREPARATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WHISPER_DECODE_STATE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Deserialize, Default)]
struct WhisperGenerationConfig {
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    lang_to_id: HashMap<String, u32>,
    #[serde(default)]
    task_to_id: HashMap<String, u32>,
    #[serde(default)]
    no_timestamps_token_id: Option<u32>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    temperature_increment_on_fallback: Option<f32>,
    #[serde(default)]
    compression_ratio_threshold: Option<f32>,
    #[serde(default)]
    logprob_threshold: Option<f32>,
    #[serde(default)]
    no_speech_threshold: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct WhisperSpecialTokens {
    sot: u32,
    sot_prev: Option<u32>,
    transcribe: u32,
    eot: u32,
    blank: Option<u32>,
    no_timestamps: Option<u32>,
    no_speech: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Clone, Copy)]
pub(crate) struct WhisperAudioBatchRow<'a> {
    pub(crate) audio: &'a [f32],
    pub(crate) sample_rate: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperWindowPreparationGeometry {
    pub(crate) input_samples: usize,
    pub(crate) input_sample_rate: u32,
    pub(crate) resampled_samples: usize,
    pub(crate) useful_mel_frames: usize,
    pub(crate) materialized_mel_elements: u64,
    pub(crate) cross_memory_tokens: usize,
    pub(crate) useful_tensor_elements: u64,
    pub(crate) retained_artifact_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhisperWindowPreparationBatchGeometry {
    pub(crate) rows: usize,
    pub(crate) total_useful_tensor_elements: u64,
    pub(crate) materialized_tensor_elements_per_row: u64,
    pub(crate) workspace_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperAudioPreparationStageSeal {
    pub(crate) backend: BackendKind,
    pub(crate) dtype: String,
    pub(crate) max_batch_size: usize,
    pub(crate) max_workspace_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct WhisperPreparedWindow {
    preparation_id: u64,
    source_identity: [u8; 32],
    input_samples: usize,
    input_sample_rate: u32,
    memory_tokens: usize,
    layers: Vec<StaticAttentionLayerValue>,
}

impl WhisperPreparedWindow {
    pub(crate) const fn cross_memory_tokens(&self) -> usize {
        self.memory_tokens
    }

    pub(crate) fn resident_tensor_bytes(&self) -> Result<u64> {
        let mut accounting = TensorStorageAccounting::default();
        for layer in &self.layers {
            accounting.add_tensor(&layer.keys).ok_or_else(|| {
                Error::Overloaded("Whisper prepared key accounting overflow".into())
            })?;
            accounting.add_tensor(&layer.values).ok_or_else(|| {
                Error::Overloaded("Whisper prepared value accounting overflow".into())
            })?;
        }
        Ok(accounting.bytes())
    }

    #[cfg(test)]
    pub(crate) fn for_test(memory_tokens: usize, layers: usize, width: usize) -> Result<Self> {
        if memory_tokens == 0 || layers == 0 || width == 0 {
            return Err(Error::InvalidInput(
                "test Whisper prepared-window geometry must be non-zero".into(),
            ));
        }
        let layers = (0..layers)
            .map(|model_layer| {
                let model_layer = u32::try_from(model_layer).map_err(|_| {
                    Error::InvalidInput("test Whisper layer count exceeds u32".into())
                })?;
                Ok(StaticAttentionLayerValue {
                    model_layer,
                    keys: Tensor::zeros(
                        (memory_tokens, 1, width),
                        DType::F32,
                        &candle_core::Device::Cpu,
                    )?,
                    values: Tensor::zeros(
                        (memory_tokens, 1, width),
                        DType::F32,
                        &candle_core::Device::Cpu,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            preparation_id: 1,
            source_identity: [1; 32],
            input_samples: SAMPLE_RATE as usize,
            input_sample_rate: SAMPLE_RATE,
            memory_tokens,
            layers,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperDecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WhisperTerminalTransition {
    Accept {
        text: String,
        selected_temperature: f32,
    },
    RetryRequired {
        next_temperature: f32,
        reasons: Vec<&'static str>,
        expected_generation: u64,
        new_generation: u64,
    },
    SkipNoSpeech {
        no_speech_probability: Option<f32>,
    },
}

#[derive(Debug, Clone, Copy)]
struct WhisperPendingRetry {
    next_temperature: f32,
    expected_generation: u64,
    new_generation: u64,
    next_attempt_generation: u32,
}

pub(crate) struct WhisperDecodeState {
    state_id: u64,
    next_quantum_nonce: u64,
    active_quantum: Option<u64>,
    current_managed_generation: u64,
    managed_completions_drained: bool,
    self_kv: PhysicalPagedKvCache,
    cross_runtime: Arc<RetainedStaticAttentionRuntimeV2>,
    cross_sequence: Option<RetainedStaticAttentionSequenceId>,
    prompt: Vec<u32>,
    prefill_progress: usize,
    pending_logits: Option<Tensor>,
    generated_tokens: Vec<u32>,
    assembled: String,
    sum_logprobs: f64,
    sampled_token_count: usize,
    no_speech_prob: Option<f32>,
    ended_with_eot: bool,
    repetition_loop: bool,
    decode_steps: usize,
    best_attempt: Option<(WhisperDecodeAttempt, f32)>,
    pending_retry: Option<WhisperPendingRetry>,
    temperature: f32,
    attempt_generation: u32,
    max_steps: usize,
    finished: bool,
    rng: StdRng,
}

/// Owns a registered retained cross-attention sequence until construction of
/// [`WhisperDecodeState`] succeeds. The caller transfers ownership at the
/// model boundary; every fallible validation and installation branch must
/// therefore release through this guard rather than relying on caller cleanup.
struct WhisperCrossSequenceOwner {
    runtime: Arc<RetainedStaticAttentionRuntimeV2>,
    sequence: Option<RetainedStaticAttentionSequenceId>,
}

impl WhisperCrossSequenceOwner {
    fn new(
        runtime: Arc<RetainedStaticAttentionRuntimeV2>,
        sequence: RetainedStaticAttentionSequenceId,
    ) -> Self {
        Self {
            runtime,
            sequence: Some(sequence),
        }
    }

    fn sequence(&self) -> RetainedStaticAttentionSequenceId {
        self.sequence
            .expect("Whisper cross-sequence owner is armed until state transfer")
    }

    fn into_decode_state_parts(
        mut self,
    ) -> (
        Arc<RetainedStaticAttentionRuntimeV2>,
        RetainedStaticAttentionSequenceId,
    ) {
        let sequence = self
            .sequence
            .take()
            .expect("Whisper cross-sequence ownership transfers exactly once");
        (Arc::clone(&self.runtime), sequence)
    }
}

impl Drop for WhisperCrossSequenceOwner {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            let _ = self.runtime.release_sequence(sequence);
        }
    }
}

fn acquire_whisper_cross_sequence_owner(
    owner: WhisperCrossSequenceOwner,
    prepared: &WhisperPreparedWindow,
    expected_preparation_id: u64,
    expected_memory_tokens: usize,
    expected_layers: usize,
    self_context_len: usize,
    allocate_state_id: impl FnOnce() -> Result<u64>,
) -> Result<(WhisperCrossSequenceOwner, u64)> {
    if prepared.preparation_id != expected_preparation_id
        || prepared.memory_tokens != expected_memory_tokens
        || prepared.layers.len() != expected_layers
    {
        return Err(Error::InvalidInput(
            "Whisper prepared window belongs to another model or geometry".into(),
        ));
    }
    let state_id = allocate_state_id()?;
    if self_context_len != 0 || owner.runtime.read(owner.sequence())?.is_some() {
        return Err(Error::InvalidInput(
            "Whisper retained prefill requires empty self and cross state".into(),
        ));
    }
    Ok((owner, state_id))
}

pub(crate) struct WhisperDecodeCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    payload: Option<WhisperDecodeCheckpointPayload>,
}

struct WhisperDecodeCheckpointPayload {
    self_kv: PhysicalPagedKvCache,
    cross_sequence: Option<RetainedStaticAttentionSequenceId>,
    prefill_progress: usize,
    pending_logits: Option<Tensor>,
    generated_tokens: Vec<u32>,
    assembled: String,
    sum_logprobs: f64,
    sampled_token_count: usize,
    no_speech_prob: Option<f32>,
    ended_with_eot: bool,
    repetition_loop: bool,
    decode_steps: usize,
    best_attempt: Option<(WhisperDecodeAttempt, f32)>,
    pending_retry: Option<WhisperPendingRetry>,
    temperature: f32,
    attempt_generation: u32,
    max_steps: usize,
    finished: bool,
    rng: StdRng,
    managed_completions_drained: bool,
    current_managed_generation: u64,
}

impl WhisperDecodeState {
    pub(crate) const fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn prefill_token_count(&self) -> usize {
        self.prompt.len()
    }

    pub(crate) const fn attempt_generation(&self) -> u32 {
        self.attempt_generation
    }

    pub(crate) fn self_context_len(&self) -> usize {
        self.self_kv.context_len()
    }

    pub(crate) const fn uses_managed_kv(&self) -> bool {
        true
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let completions = self.self_kv.take_completed_writes();
        self.managed_completions_drained = true;
        completions
    }

    pub(crate) fn install_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        let mut checkpoint = self.begin_managed_quantum(cache)?;
        self.commit_managed_quantum(&mut checkpoint)?;
        Ok(())
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<WhisperDecodeCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a Whisper managed quantum is already active".into(),
            ));
        }
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "Whisper managed KV write completions must be drained before the next quantum"
                    .into(),
            ));
        }
        if self.self_kv.arena().id() != cache.arena().id()
            || self.self_kv.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Whisper session cannot switch managed KV authority".into(),
            ));
        }
        if cache.context_len() != self.self_kv.context_len() {
            return Err(Error::InferenceError(format!(
                "managed Whisper reservation starts at {}, but decode state is at {}",
                cache.context_len(),
                self.self_kv.context_len()
            )));
        }
        self.activate_managed_checkpoint(cache)
    }

    pub(crate) fn begin_managed_generation(
        &mut self,
        cache: PhysicalPagedKvCache,
        expected_generation: u64,
        new_generation: u64,
    ) -> Result<WhisperDecodeCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a Whisper managed quantum is already active".into(),
            ));
        }
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "Whisper managed KV write completions must be drained before retry generation"
                    .into(),
            ));
        }
        let pending = self.pending_retry.ok_or_else(|| {
            Error::InferenceError("Whisper has no pending managed temperature retry".into())
        })?;
        if pending.expected_generation != expected_generation
            || pending.new_generation != new_generation
            || self.current_managed_generation != expected_generation
            || expected_generation.checked_add(1) != Some(new_generation)
        {
            return Err(Error::InferenceError(
                "Whisper managed retry generation is stale or out of order".into(),
            ));
        }
        let current_id = self.self_kv.arena().id();
        let replacement_id = cache.arena().id();
        if replacement_id != current_id
            || cache.arena().config().group != self.self_kv.arena().config().group
        {
            return Err(Error::InferenceError(
                "Whisper managed retry cannot switch session KV authority".into(),
            ));
        }
        if cache.context_len() != 0 {
            return Err(Error::InferenceError(
                "Whisper managed retry generation must begin at context zero".into(),
            ));
        }
        let sequence = self.cross_sequence.ok_or_else(|| {
            Error::InferenceError("Whisper retained cross sequence was released".into())
        })?;
        self.cross_runtime.read(sequence)?.ok_or_else(|| {
            Error::InferenceError("Whisper retained cross memory is not installed".into())
        })?;

        let checkpoint = self.activate_managed_checkpoint(cache)?;
        let retry_seed = self.rng.gen::<u64>();
        self.prefill_progress = 0;
        self.pending_logits = None;
        self.generated_tokens.clear();
        self.assembled.clear();
        self.sum_logprobs = 0.0;
        self.sampled_token_count = 0;
        self.no_speech_prob = None;
        self.ended_with_eot = false;
        self.repetition_loop = false;
        self.decode_steps = 0;
        self.temperature = pending.next_temperature;
        self.attempt_generation = pending.next_attempt_generation;
        self.finished = false;
        self.rng = StdRng::seed_from_u64(retry_seed);
        self.pending_retry = None;
        self.current_managed_generation = new_generation;
        self.managed_completions_drained = true;
        Ok(checkpoint)
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut WhisperDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_active_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Whisper managed checkpoint was already consumed".into())
        })?;
        self.active_quantum = None;
        drop(payload);
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut WhisperDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_active_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Whisper managed checkpoint was already consumed".into())
        })?;
        self.self_kv = payload.self_kv;
        self.cross_sequence = payload.cross_sequence;
        self.prefill_progress = payload.prefill_progress;
        self.pending_logits = payload.pending_logits;
        self.generated_tokens = payload.generated_tokens;
        self.assembled = payload.assembled;
        self.sum_logprobs = payload.sum_logprobs;
        self.sampled_token_count = payload.sampled_token_count;
        self.no_speech_prob = payload.no_speech_prob;
        self.ended_with_eot = payload.ended_with_eot;
        self.repetition_loop = payload.repetition_loop;
        self.decode_steps = payload.decode_steps;
        self.best_attempt = payload.best_attempt;
        self.pending_retry = payload.pending_retry;
        self.temperature = payload.temperature;
        self.attempt_generation = payload.attempt_generation;
        self.max_steps = payload.max_steps;
        self.finished = payload.finished;
        self.rng = payload.rng;
        self.managed_completions_drained = payload.managed_completions_drained;
        self.current_managed_generation = payload.current_managed_generation;
        self.active_quantum = None;
        Ok(())
    }

    fn validate_active_checkpoint(&self, checkpoint: &WhisperDecodeCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum_nonce)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Whisper managed checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }

    fn activate_managed_checkpoint(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<WhisperDecodeCheckpoint> {
        let quantum_nonce = self.next_quantum_nonce;
        self.next_quantum_nonce = self.next_quantum_nonce.checked_add(1).ok_or_else(|| {
            Error::InferenceError("Whisper managed quantum nonce overflow".into())
        })?;
        self.active_quantum = Some(quantum_nonce);
        Ok(WhisperDecodeCheckpoint {
            state_id: self.state_id,
            quantum_nonce,
            payload: Some(WhisperDecodeCheckpointPayload {
                self_kv: std::mem::replace(&mut self.self_kv, cache),
                cross_sequence: self.cross_sequence,
                prefill_progress: self.prefill_progress,
                pending_logits: self.pending_logits.clone(),
                generated_tokens: self.generated_tokens.clone(),
                assembled: self.assembled.clone(),
                sum_logprobs: self.sum_logprobs,
                sampled_token_count: self.sampled_token_count,
                no_speech_prob: self.no_speech_prob,
                ended_with_eot: self.ended_with_eot,
                repetition_loop: self.repetition_loop,
                decode_steps: self.decode_steps,
                best_attempt: self.best_attempt.clone(),
                pending_retry: self.pending_retry,
                temperature: self.temperature,
                attempt_generation: self.attempt_generation,
                max_steps: self.max_steps,
                finished: self.finished,
                rng: self.rng.clone(),
                managed_completions_drained: self.managed_completions_drained,
                current_managed_generation: self.current_managed_generation,
            }),
        })
    }
}

impl Drop for WhisperDecodeState {
    fn drop(&mut self) {
        if let Some(sequence) = self.cross_sequence.take() {
            let _ = self.cross_runtime.release_sequence(sequence);
        }
    }
}

#[derive(Clone)]
struct WhisperDecodeStepSnapshot {
    pending_logits: Option<Tensor>,
    generated_tokens: Vec<u32>,
    assembled: String,
    sum_logprobs: f64,
    sampled_token_count: usize,
    no_speech_prob: Option<f32>,
    ended_with_eot: bool,
    repetition_loop: bool,
    decode_steps: usize,
    best_attempt: Option<(WhisperDecodeAttempt, f32)>,
    finished: bool,
    rng: StdRng,
    managed_completions_drained: bool,
}

impl WhisperDecodeStepSnapshot {
    fn capture(state: &WhisperDecodeState) -> Self {
        Self {
            pending_logits: state.pending_logits.clone(),
            generated_tokens: state.generated_tokens.clone(),
            assembled: state.assembled.clone(),
            sum_logprobs: state.sum_logprobs,
            sampled_token_count: state.sampled_token_count,
            no_speech_prob: state.no_speech_prob,
            ended_with_eot: state.ended_with_eot,
            repetition_loop: state.repetition_loop,
            decode_steps: state.decode_steps,
            best_attempt: state.best_attempt.clone(),
            finished: state.finished,
            rng: state.rng.clone(),
            managed_completions_drained: state.managed_completions_drained,
        }
    }

    fn restore(self, state: &mut WhisperDecodeState) {
        state.pending_logits = self.pending_logits;
        state.generated_tokens = self.generated_tokens;
        state.assembled = self.assembled;
        state.sum_logprobs = self.sum_logprobs;
        state.sampled_token_count = self.sampled_token_count;
        state.no_speech_prob = self.no_speech_prob;
        state.ended_with_eot = self.ended_with_eot;
        state.repetition_loop = self.repetition_loop;
        state.decode_steps = self.decode_steps;
        state.best_attempt = self.best_attempt;
        state.finished = self.finished;
        state.rng = self.rng;
        state.managed_completions_drained = self.managed_completions_drained;
    }
}

fn with_whisper_decode_step_transaction<T>(
    state: &mut WhisperDecodeState,
    operation: impl FnOnce(&mut WhisperDecodeState) -> Result<T>,
) -> Result<T> {
    let checkpoint = state.self_kv.logical_checkpoint();
    let snapshot = WhisperDecodeStepSnapshot::capture(state);
    match operation(state) {
        Ok(value) => Ok(value),
        Err(error) => {
            let rollback = state.self_kv.restore_logical_checkpoint(checkpoint);
            snapshot.restore(state);
            match rollback {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(Error::InferenceError(format!(
                    "Whisper decode step failed ({error}); state rollback also failed: {rollback_error}"
                ))),
            }
        }
    }
}

fn restart_whisper_temperature_attempt(
    _state: &mut WhisperDecodeState,
    temperature: f32,
) -> Result<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(Error::InvalidInput(
            "Whisper retry temperature must be finite and non-negative".into(),
        ));
    }
    Err(Error::InferenceError(
        "Whisper managed temperature retry requires a scheduler-owned retained-sequence reset"
            .into(),
    ))
}

fn next_whisper_decode_state_id() -> Result<u64> {
    NEXT_WHISPER_DECODE_STATE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| Error::InferenceError("Whisper decode-state identity overflow".into()))
}

#[derive(Debug, Clone)]
struct WhisperDecodeAttempt {
    text: String,
    avg_logprob: f32,
    no_speech_prob: Option<f32>,
    ended_with_eot: bool,
    repetition_loop: bool,
    compression_ratio: Option<f32>,
    generated_token_count: usize,
    sampled_token_count: usize,
    decode_steps: usize,
    profile: WhisperDecodeProfile,
}

#[derive(Debug, Clone, Copy, Default)]
struct WhisperDecodeProfile {
    enabled: bool,
    synchronized: bool,
    token_tensor_ms: f64,
    decoder_forward_ms: f64,
    final_linear_ms: f64,
    logits_to_host_ms: f64,
    sampling_ms: f64,
    step_total_ms: f64,
    device_greedy_steps: usize,
    device_greedy_fallbacks: usize,
    host_logits_steps: usize,
}

impl WhisperDecodeProfile {
    fn new(sync_enabled: bool) -> Self {
        Self {
            enabled: sync_enabled,
            synchronized: sync_enabled,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
struct WhisperRuntimeTuning {
    no_fallback: bool,
    max_fallback_retries: usize,
    adaptive_decode_budget: bool,
    max_new_tokens_per_second: f32,
    min_new_tokens: usize,
    max_new_tokens_cap: usize,
    decode_budget_buffer_tokens: usize,
    default_language: Option<String>,
    trim_silence: bool,
    silence_trim_threshold_scale: f32,
    silence_trim_min_abs: f32,
    silence_trim_margin_ms: usize,
    silence_trim_min_leading_ms: usize,
    silence_trim_min_trailing_ms: usize,
    silence_trim_min_clip_secs: f32,
    suppress_blank: bool,
    suppress_numerals: bool,
    initial_prompt_max_tokens: usize,
    profile_sync_timings: bool,
    device_greedy_decode: bool,
}

impl WhisperRuntimeTuning {
    fn from_env() -> Self {
        Self {
            no_fallback: env_bool("IZWI_WHISPER_NO_FALLBACK").unwrap_or(false),
            max_fallback_retries: env_usize("IZWI_WHISPER_MAX_FALLBACK_RETRIES")
                .unwrap_or(DEFAULT_MAX_FALLBACK_RETRIES),
            adaptive_decode_budget: env_bool("IZWI_WHISPER_ADAPTIVE_MAX_NEW_TOKENS")
                .unwrap_or(true),
            max_new_tokens_per_second: env_f32("IZWI_WHISPER_MAX_NEW_TOKENS_PER_SECOND")
                .unwrap_or(DEFAULT_ADAPTIVE_MAX_NEW_TOKENS_PER_SECOND),
            min_new_tokens: env_usize("IZWI_WHISPER_MIN_NEW_TOKENS")
                .unwrap_or(DEFAULT_ADAPTIVE_MIN_NEW_TOKENS),
            max_new_tokens_cap: env_usize("IZWI_WHISPER_MAX_NEW_TOKENS_CAP")
                .unwrap_or(DEFAULT_MAX_NEW_TOKENS),
            decode_budget_buffer_tokens: env_usize("IZWI_WHISPER_MAX_NEW_TOKENS_BUFFER")
                .unwrap_or(DEFAULT_ADAPTIVE_BUDGET_BUFFER_TOKENS),
            default_language: env_nonempty_string("IZWI_WHISPER_DEFAULT_LANGUAGE"),
            trim_silence: env_bool("IZWI_WHISPER_TRIM_SILENCE").unwrap_or(true),
            silence_trim_threshold_scale: env_f32("IZWI_WHISPER_SILENCE_TRIM_THRESHOLD_SCALE")
                .unwrap_or(DEFAULT_SILENCE_TRIM_THRESHOLD_SCALE),
            silence_trim_min_abs: env_f32("IZWI_WHISPER_SILENCE_TRIM_MIN_ABS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_ABS),
            silence_trim_margin_ms: env_usize("IZWI_WHISPER_SILENCE_TRIM_MARGIN_MS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MARGIN_MS),
            silence_trim_min_leading_ms: env_usize("IZWI_WHISPER_SILENCE_TRIM_MIN_LEADING_MS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_LEADING_MS),
            silence_trim_min_trailing_ms: env_usize("IZWI_WHISPER_SILENCE_TRIM_MIN_TRAILING_MS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_TRAILING_MS),
            silence_trim_min_clip_secs: env_f32("IZWI_WHISPER_SILENCE_TRIM_MIN_CLIP_SECS")
                .unwrap_or(DEFAULT_SILENCE_TRIM_MIN_CLIP_SECS),
            suppress_blank: env_bool("IZWI_WHISPER_SUPPRESS_BLANK").unwrap_or(true),
            suppress_numerals: env_bool("IZWI_WHISPER_SUPPRESS_NUMERALS").unwrap_or(false),
            initial_prompt_max_tokens: env_usize("IZWI_WHISPER_INITIAL_PROMPT_MAX_TOKENS")
                .unwrap_or(DEFAULT_INITIAL_PROMPT_MAX_TOKENS),
            profile_sync_timings: env_bool("IZWI_WHISPER_PROFILE_SYNC").unwrap_or(false),
            device_greedy_decode: env_bool("IZWI_WHISPER_DEVICE_GREEDY").unwrap_or(true),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct WhisperPromptDiagnostics {
    initial_prompt_requested: bool,
    initial_prompt_token_count: usize,
    initial_prompt_tokens_used: usize,
    initial_prompt_tokens_truncated: usize,
    initial_prompt_max_tokens: usize,
    previous_context_token_id: Option<u32>,
    rolling_context_enabled: bool,
}

#[derive(Debug, Clone)]
struct WhisperPromptPrefix {
    ids: Vec<u32>,
    diagnostics: WhisperPromptDiagnostics,
}

#[derive(Debug, Clone)]
struct WhisperLanguageResolution {
    resolved: Option<(u32, String)>,
    hint_used: bool,
    detect_ms: f64,
    strategy: &'static str,
}

enum WhisperModel {
    Local(LocalWhisper),
}

impl WhisperModel {
    fn load(vb: &VarBuilder, config: WhisperConfig) -> Result<Self> {
        LocalWhisper::load(vb, config).map(Self::Local)
    }

    fn encoder_forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Local(model) => model.encoder.forward(x),
        }
    }

    fn encoder_forward_batch(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Local(model) => model.encoder.forward_batch(x),
        }
    }

    fn prepare_cross_attention_memory(
        &self,
        audio_features: &Tensor,
    ) -> Result<Vec<StaticAttentionLayerValue>> {
        match self {
            Self::Local(model) => model.decoder.prepare_cross_attention_memory(audio_features),
        }
    }

    fn install_cross_attention_memory(
        &self,
        audio_features: &Tensor,
        source_identity: [u8; 32],
        cross_kv: &mut InvocationStaticAttentionLease,
    ) -> Result<()> {
        match self {
            Self::Local(model) => model.decoder.install_cross_attention_memory(
                audio_features,
                source_identity,
                cross_kv,
            ),
        }
    }

    fn decoder_forward_physical_at(
        &self,
        x: &Tensor,
        position_offset: usize,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &InvocationStaticAttentionLease,
    ) -> Result<Tensor> {
        match self {
            Self::Local(model) => {
                model
                    .decoder
                    .forward_physical_at(x, position_offset, self_kv, cross_kv)
            }
        }
    }

    fn decoder_forward_retained_at(
        &self,
        x: &Tensor,
        position_offset: usize,
        self_kv: &mut PhysicalPagedKvCache,
        cross_runtime: &RetainedStaticAttentionRuntimeV2,
        cross_sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<Tensor> {
        match self {
            Self::Local(model) => model.decoder.forward_retained_at(
                x,
                position_offset,
                self_kv,
                cross_runtime,
                cross_sequence,
            ),
        }
    }

    fn decoder_final_linear(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Local(model) => model.decoder.final_linear(x),
        }
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn use_cuda_whisper_dtype_shim(device: &candle_core::Device) -> bool {
    device.is_cuda()
}

fn whisper_device_diagnostics(
    device_kind: DeviceKind,
    model_dtype: DType,
    cuda_dtype_shim: bool,
) -> serde_json::Value {
    json!({
        "kind": format!("{device_kind:?}"),
        "model_dtype": format!("{model_dtype:?}"),
        "cuda_dtype_shim": cuda_dtype_shim,
        "whisper_impl": whisper_impl_name(cuda_dtype_shim),
    })
}

fn whisper_impl_name(cuda_dtype_shim: bool) -> &'static str {
    if cuda_dtype_shim {
        "local_whisper_cuda_dtype_shim"
    } else {
        "local_whisper"
    }
}

fn physical_state_required() -> Error {
    Error::InferenceError(
        "Whisper ASR requires lifecycle-owned physical invocation state".to_string(),
    )
}

fn whisper_cross_source_identity(audio: &[f32], sample_rate: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"izwi-whisper-cross-attention-v2");
    hasher.update(sample_rate.to_le_bytes());
    hasher.update((audio.len() as u64).to_le_bytes());
    for sample in audio {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    let mut identity: [u8; 32] = hasher.finalize().into();
    if identity.iter().all(|byte| *byte == 0) {
        identity[0] = 1;
    }
    identity
}

pub struct WhisperTurboAsrModel {
    preparation_id: u64,
    device: DeviceProfile,
    model_dtype: DType,
    whisper: WhisperModel,
    config: WhisperConfig,
    generation: WhisperGenerationConfig,
    tokenizer: Tokenizer,
    special: WhisperSpecialTokens,
    mel: MelSpectrogram,
    suppress_tokens: Vec<u32>,
    numeral_symbol_tokens: Vec<u32>,
    decode_mask: Tensor,
    decode_begin_mask: Tensor,
    language_token_ids: Vec<u32>,
    language_token_range: Option<(usize, usize)>,
    token_id_to_language_code: HashMap<u32, String>,
    runtime_tuning: WhisperRuntimeTuning,
    cuda_dtype_shim: bool,
}

impl WhisperTurboAsrModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_data = fs::read_to_string(config_path)?;
        let config: WhisperConfig = serde_json::from_str(&config_data)?;

        let generation = read_generation_config(model_dir)?;
        let tokenizer = Tokenizer::from_path(model_dir)?;

        let dtype_override = std::env::var("IZWI_WHISPER_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let model_dtype = match dtype_override.as_deref() {
            Some(raw) => {
                device.select_model_dtype_checked(ModelFamily::WhisperAsr, Some(raw), "Whisper")?
            }
            None => device.select_model_dtype(ModelFamily::WhisperAsr, None),
        };

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|value| value.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid Whisper safetensors index format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> = shard_files
                .iter()
                .map(|file| model_dir.join(file))
                .collect();

            unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, model_dtype, &device.device)?
            }
        } else {
            let model_path = model_dir.join("model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], model_dtype, &device.device)?
            }
        };

        let use_cuda_dtype_shim = use_cuda_whisper_dtype_shim(&device.device);
        let whisper = WhisperModel::load(&vb, config.clone())?;
        let special = resolve_special_tokens(&tokenizer, &generation)?;
        let (language_token_ids, token_id_to_language_code) =
            build_language_token_maps(&tokenizer, &generation);
        let language_token_range = contiguous_token_range(&language_token_ids);

        let mut suppress_tokens = generation.suppress_tokens.clone();
        suppress_tokens.sort_unstable();
        suppress_tokens.dedup();
        let runtime_tuning = WhisperRuntimeTuning::from_env();
        let numeral_symbol_tokens = if runtime_tuning.suppress_numerals {
            build_numeral_symbol_tokens(&tokenizer, &special)
        } else {
            Vec::new()
        };
        let (decode_mask, decode_begin_mask) = build_whisper_decode_mask_tensors(
            config.vocab_size,
            &suppress_tokens,
            &generation.begin_suppress_tokens,
            &language_token_ids,
            &special,
            runtime_tuning.suppress_blank,
            &numeral_symbol_tokens,
            model_dtype,
            &device.device,
        )?;

        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: whisper::SAMPLE_RATE,
            n_fft: whisper::N_FFT,
            win_length: None,
            hop_length: whisper::HOP_LENGTH,
            n_mels: config.num_mel_bins,
            f_min: 0.0,
            f_max: (whisper::SAMPLE_RATE / 2) as f32,
            normalize: true,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        })?;

        info!(
            "Loaded Whisper Large v3 Turbo ASR on {:?} (dtype={:?}, cuda_dtype_shim={})",
            device.kind, model_dtype, use_cuda_dtype_shim
        );

        Ok(Self {
            preparation_id: NEXT_WHISPER_PREPARATION_ID.fetch_add(1, Ordering::Relaxed),
            device,
            model_dtype,
            whisper,
            config,
            generation,
            tokenizer,
            special,
            mel,
            suppress_tokens,
            numeral_symbol_tokens,
            decode_mask,
            decode_begin_mask,
            language_token_ids,
            language_token_range,
            token_id_to_language_code,
            runtime_tuning,
            cuda_dtype_shim: use_cuda_dtype_shim,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        self.transcribe_with_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_prompt(
        &self,
        _audio: &[f32],
        _sample_rate: u32,
        _language: Option<&str>,
        _initial_prompt: Option<&str>,
    ) -> Result<String> {
        Err(physical_state_required())
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        _audio: &[f32],
        _sample_rate: u32,
        _language: Option<&str>,
        _initial_prompt: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        Err(physical_state_required())
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, None, on_delta)
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        _audio: &[f32],
        _sample_rate: u32,
        _language: Option<&str>,
        _initial_prompt: Option<&str>,
        _on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Err(physical_state_required())
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(MAX_AUDIO_SECONDS_HINT)
    }

    pub(crate) fn window_preparation_geometry(
        &self,
        audio: &[f32],
        input_sample_rate: u32,
    ) -> Result<WhisperWindowPreparationGeometry> {
        let effective_samples = self.trimmed_audio_slice(audio, input_sample_rate).len();
        self.window_preparation_geometry_for_lengths(
            audio.len(),
            effective_samples,
            input_sample_rate,
        )
    }

    fn window_preparation_geometry_for_lengths(
        &self,
        input_samples: usize,
        effective_samples: usize,
        input_sample_rate: u32,
    ) -> Result<WhisperWindowPreparationGeometry> {
        if input_samples == 0 || input_sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Whisper window preparation requires non-empty audio and a sample rate".into(),
            ));
        }
        let resampled_samples = if input_sample_rate == SAMPLE_RATE || effective_samples < 2 {
            effective_samples
        } else {
            ((effective_samples as f64) * SAMPLE_RATE as f64 / input_sample_rate as f64)
                .round()
                .max(1.0) as usize
        };
        let target_mel_frames =
            self.config
                .max_source_positions
                .checked_mul(2)
                .ok_or_else(|| {
                    Error::Overloaded("Whisper maximum mel-frame geometry overflow".into())
                })?;
        let raw_mel_frames = resampled_samples
            .checked_div(whisper::HOP_LENGTH)
            .and_then(|frames| frames.checked_add(1))
            .ok_or_else(|| Error::Overloaded("Whisper mel-frame geometry overflow".into()))?;
        let useful_mel_frames = raw_mel_frames.min(target_mel_frames);
        let mel_elements = checked_product_u64(
            &[useful_mel_frames, self.config.num_mel_bins],
            "Whisper useful mel geometry",
        )?;
        let materialized_mel_elements = checked_product_u64(
            &[target_mel_frames, self.config.num_mel_bins],
            "Whisper materialized mel geometry",
        )?;
        let cross_elements = checked_product_u64(
            &[
                self.config.decoder_layers,
                self.config.max_source_positions,
                self.config.d_model,
                2,
            ],
            "Whisper retained cross-memory geometry",
        )?;
        let dtype_bytes = u64::try_from(self.model_dtype.size_in_bytes())
            .map_err(|_| Error::Overloaded("Whisper dtype width exceeds u64".into()))?;
        let retained_artifact_bytes = cross_elements.checked_mul(dtype_bytes).ok_or_else(|| {
            Error::Overloaded("Whisper retained cross-memory bytes overflow".into())
        })?;
        Ok(WhisperWindowPreparationGeometry {
            input_samples,
            input_sample_rate,
            resampled_samples,
            useful_mel_frames,
            materialized_mel_elements,
            cross_memory_tokens: self.config.max_source_positions,
            useful_tensor_elements: mel_elements.checked_add(cross_elements).ok_or_else(|| {
                Error::Overloaded("Whisper useful tensor geometry overflow".into())
            })?,
            retained_artifact_bytes,
        })
    }

    pub(crate) fn window_preparation_batch_geometry(
        &self,
        rows: &[WhisperWindowPreparationGeometry],
    ) -> Result<WhisperWindowPreparationBatchGeometry> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Whisper window preparation batch is empty".into(),
            ));
        }
        let total_useful_tensor_elements = rows.iter().try_fold(0_u64, |total, row| {
            total
                .checked_add(row.useful_tensor_elements)
                .ok_or_else(|| Error::Overloaded("Whisper batch useful geometry overflow".into()))
        })?;
        let retained_elements = checked_product_u64(
            &[
                self.config.decoder_layers,
                self.config.max_source_positions,
                self.config.d_model,
                2,
            ],
            "Whisper materialized cross-memory geometry",
        )?;
        let materialized_tensor_elements_per_row = rows[0]
            .materialized_mel_elements
            .checked_add(retained_elements)
            .ok_or_else(|| {
                Error::Overloaded("Whisper materialized row geometry overflow".into())
            })?;
        let width = rows.len();
        let encoder_tokens = self.config.max_source_positions;
        let hidden = self.config.d_model;
        let heads = self.config.encoder_attention_heads;
        let batch_hidden = checked_product_u64(
            &[width, encoder_tokens, hidden],
            "Whisper encoder hidden geometry",
        )?;
        let attention = checked_product_u64(
            &[width, heads, encoder_tokens, encoder_tokens],
            "Whisper encoder attention geometry",
        )?;
        let ffn = checked_product_u64(
            &[width, encoder_tokens, hidden, 4],
            "Whisper encoder FFN geometry",
        )?;
        let encoder_hidden_working = batch_hidden
            .checked_mul(4)
            .ok_or_else(|| Error::Overloaded("Whisper encoder hidden workspace overflow".into()))?;
        let device_elements = checked_product_u64(&[width], "Whisper batch width")?
            .checked_mul(materialized_tensor_elements_per_row)
            .and_then(|value| value.checked_add(encoder_hidden_working))
            .and_then(|value| value.checked_add(attention))
            .and_then(|value| value.checked_add(ffn))
            .ok_or_else(|| {
                Error::Overloaded("Whisper device workspace geometry overflow".into())
            })?;
        let host_elements = rows.iter().try_fold(0_u64, |total, row| {
            total
                .checked_add(
                    u64::try_from(row.resampled_samples).map_err(|_| {
                        Error::Overloaded("Whisper resampled row exceeds u64".into())
                    })?,
                )
                .and_then(|value| value.checked_add(row.materialized_mel_elements))
                .ok_or_else(|| Error::Overloaded("Whisper host workspace geometry overflow".into()))
        })?;
        let workspace_bytes = device_elements
            .checked_mul(
                u64::try_from(self.model_dtype.size_in_bytes())
                    .map_err(|_| Error::Overloaded("Whisper dtype width exceeds u64".into()))?,
            )
            .and_then(|device| {
                host_elements
                    .checked_mul(std::mem::size_of::<f32>() as u64)
                    .and_then(|host| device.checked_add(host))
            })
            .ok_or_else(|| Error::Overloaded("Whisper batch workspace bytes overflow".into()))?;
        Ok(WhisperWindowPreparationBatchGeometry {
            rows: rows.len(),
            total_useful_tensor_elements,
            materialized_tensor_elements_per_row,
            workspace_bytes,
        })
    }

    pub(crate) fn window_preparation_row_cost_for_batch(
        &self,
        row_index: usize,
        rows: &[WhisperWindowPreparationGeometry],
        batch: &WhisperWindowPreparationBatchGeometry,
    ) -> Result<WorkCost> {
        let row = rows.get(row_index).ok_or_else(|| {
            Error::InvalidInput("Whisper preparation row index is out of range".into())
        })?;
        if rows.len() != batch.rows {
            return Err(Error::InvalidInput(
                "Whisper preparation rows disagree with batch geometry".into(),
            ));
        }
        let width = u64::try_from(batch.rows)
            .map_err(|_| Error::Overloaded("Whisper batch width exceeds u64".into()))?;
        let share = batch.workspace_bytes / width
            + u64::from((row_index as u64) < batch.workspace_bytes % width);
        Ok(WorkCost::new(
            row.useful_mel_frames as u64,
            row.useful_tensor_elements,
            share,
        ))
    }

    pub(crate) const fn window_retained_tensor_bytes(
        &self,
        row: &WhisperWindowPreparationGeometry,
    ) -> u64 {
        row.retained_artifact_bytes
    }

    pub(crate) fn window_max_batch_workspace_bytes(&self, width: usize) -> Result<u64> {
        if width == 0 {
            return Err(Error::InvalidInput(
                "Whisper preparation batch width must be non-zero".into(),
            ));
        }
        let samples = (SAMPLE_RATE as usize)
            .checked_mul(MAX_AUDIO_SECONDS_HINT as usize)
            .ok_or_else(|| Error::Overloaded("Whisper maximum window samples overflow".into()))?;
        let row = self.window_preparation_geometry_for_lengths(samples, samples, SAMPLE_RATE)?;
        self.window_preparation_batch_geometry(&vec![row; width])
            .map(|batch| batch.workspace_bytes)
    }

    pub(crate) fn window_preparation_stage_seal(
        &self,
        backend: BackendKind,
        width: usize,
    ) -> Result<WhisperAudioPreparationStageSeal> {
        let loaded = backend_kind_for_device(&self.device.device);
        if backend != loaded {
            return Err(Error::ModelLoadError(format!(
                "Whisper preparation backend mismatch: model={loaded:?}, adapter={backend:?}"
            )));
        }
        Ok(WhisperAudioPreparationStageSeal {
            backend,
            dtype: format!("{:?}", self.model_dtype).to_ascii_lowercase(),
            max_batch_size: width,
            max_workspace_bytes: self.window_max_batch_workspace_bytes(width)?,
        })
    }

    pub(crate) fn prepare_window_batch(
        &self,
        rows: &[WhisperAudioBatchRow<'_>],
    ) -> Result<Vec<WhisperPreparedWindow>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Whisper window preparation requires at least one row".into(),
            ));
        }
        let mut mels = Vec::with_capacity(rows.len());
        let mut identities = Vec::with_capacity(rows.len());
        for row in rows {
            if row.audio.is_empty() || row.sample_rate == 0 {
                return Err(Error::InvalidInput(
                    "Whisper window preparation received an empty row".into(),
                ));
            }
            let trimmed = self.trimmed_audio_slice(row.audio, row.sample_rate);
            mels.push(self.prepare_mel(trimmed, row.sample_rate)?);
            identities.push((
                whisper_cross_source_identity(trimmed, row.sample_rate),
                row.audio.len(),
                row.sample_rate,
            ));
        }
        let mel_refs = mels.iter().collect::<Vec<_>>();
        let mel_batch = Tensor::cat(&mel_refs, 0)?;
        let encoded = self.whisper.encoder_forward_batch(&mel_batch)?;
        let mut prepared = Vec::with_capacity(rows.len());
        for (index, (source_identity, input_samples, input_sample_rate)) in
            identities.into_iter().enumerate()
        {
            let features = encoded.narrow(0, index, 1)?;
            let memory_tokens = features.dim(1)?;
            let layers = self.whisper.prepare_cross_attention_memory(&features)?;
            prepared.push(WhisperPreparedWindow {
                preparation_id: self.preparation_id,
                source_identity,
                input_samples,
                input_sample_rate,
                memory_tokens,
                layers,
            });
        }
        Ok(prepared)
    }

    pub(crate) fn incremental_prompt_token_count_from_prepared_window(
        &self,
        prepared: &WhisperPreparedWindow,
        language: Option<&str>,
        initial_prompt: Option<&str>,
    ) -> Result<usize> {
        self.validate_prepared_window(prepared)?;
        let language_token = if let Some(language) = language {
            self.resolve_language_token(language)?
                .map(|(token, _)| token)
        } else if let Some(default) = self.runtime_tuning.default_language.as_deref() {
            self.resolve_language_token(default)?
                .map(|(token, _)| token)
        } else {
            self.language_token_ids.first().copied()
        };
        let initial_prompt_tokens = self.encode_initial_prompt_tokens(initial_prompt)?;
        Ok(build_whisper_prompt_prefix(
            &self.special,
            language_token,
            &initial_prompt_tokens,
            self.config.max_target_positions,
            self.runtime_tuning.initial_prompt_max_tokens,
        )?
        .ids
        .len())
    }

    pub(crate) fn begin_resumable_prefill_managed_from_prepared_window(
        &self,
        prepared: &WhisperPreparedWindow,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        max_new_tokens: Option<usize>,
        mut self_kv: PhysicalPagedKvCache,
        cross_runtime: Arc<RetainedStaticAttentionRuntimeV2>,
        cross_sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<WhisperDecodeState> {
        // Ownership transfers at entry, before any validation or state access.
        let cross_owner = WhisperCrossSequenceOwner::new(cross_runtime, cross_sequence);
        let (cross_owner, state_id) = acquire_whisper_cross_sequence_owner(
            cross_owner,
            prepared,
            self.preparation_id,
            self.config.max_source_positions,
            self.config.decoder_layers,
            self_kv.context_len(),
            next_whisper_decode_state_id,
        )?;
        let cross_sequence = cross_owner.sequence();
        cross_owner.runtime.install(
            cross_sequence,
            prepared.source_identity,
            prepared.layers.clone(),
        )?;

        let initialized: Result<(Vec<u32>, usize)> = (|| {
            let resolved_language = self.resolve_request_language_retained(
                language,
                &mut self_kv,
                cross_owner.runtime.as_ref(),
                cross_sequence,
            )?;
            // Language detection is a complete nested decoder sequence. The
            // prompt begins from a fresh self-KV generation while immutable
            // cross memory remains installed under the same sequence identity.
            if self_kv.context_len() != 0 {
                self_kv.reset_invocation()?;
            }
            let initial_prompt_tokens = self.encode_initial_prompt_tokens(initial_prompt)?;
            let prompt = build_whisper_prompt_prefix(
                &self.special,
                resolved_language.as_ref().map(|(token, _)| *token),
                &initial_prompt_tokens,
                self.config.max_target_positions,
                self.runtime_tuning.initial_prompt_max_tokens,
            )?
            .ids;
            let configured_steps = max_new_tokens.unwrap_or_else(|| {
                self.resolve_max_decode_tokens(prepared.input_samples, prepared.input_sample_rate)
            });
            let max_steps = decode_step_budget(
                prompt.len(),
                self.config.max_target_positions,
                configured_steps,
            )?;
            Ok((prompt, max_steps))
        })();
        let (prompt, max_steps) = initialized?;
        let (cross_runtime, cross_sequence) = cross_owner.into_decode_state_parts();

        Ok(WhisperDecodeState {
            state_id,
            next_quantum_nonce: 1,
            active_quantum: None,
            current_managed_generation: 1,
            managed_completions_drained: true,
            self_kv,
            cross_runtime,
            cross_sequence: Some(cross_sequence),
            prompt,
            prefill_progress: 0,
            pending_logits: None,
            generated_tokens: Vec::new(),
            assembled: String::new(),
            sum_logprobs: 0.0,
            sampled_token_count: 0,
            no_speech_prob: None,
            ended_with_eot: false,
            repetition_loop: false,
            decode_steps: 0,
            best_attempt: None,
            pending_retry: None,
            temperature: self.decode_temperatures().first().copied().unwrap_or(0.0),
            attempt_generation: 1,
            max_steps,
            finished: false,
            rng: StdRng::from_entropy(),
        })
    }

    pub(crate) fn continue_resumable_prefill(
        &self,
        state: &mut WhisperDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if span_start != state.prefill_progress
            || span_end <= span_start
            || span_end > state.prompt.len()
            || state.finished
        {
            return Err(Error::InvalidInput(
                "Whisper prefill span is non-monotonic or out of range".into(),
            ));
        }
        let sequence = state.cross_sequence.ok_or_else(|| {
            Error::InferenceError("Whisper retained cross sequence was released".into())
        })?;
        let checkpoint = state.self_kv.logical_checkpoint();
        let tokens =
            Tensor::new(&state.prompt[span_start..span_end], &self.device.device)?.unsqueeze(0)?;
        let output = match self.whisper.decoder_forward_retained_at(
            &tokens,
            span_start,
            &mut state.self_kv,
            state.cross_runtime.as_ref(),
            sequence,
        ) {
            Ok(output) => output,
            Err(error) => {
                state.self_kv.restore_logical_checkpoint(checkpoint)?;
                return Err(error);
            }
        };
        let pending_logits = if span_end == state.prompt.len() {
            let sequence_len = output.dim(1)?;
            match self
                .whisper
                .decoder_final_linear(&output.i((..1, sequence_len - 1..))?)
                .and_then(|logits| logits.i(0).map_err(Error::from))
                .and_then(|logits| logits.i(0).map_err(Error::from))
            {
                Ok(logits) => Some(logits),
                Err(error) => {
                    state.self_kv.restore_logical_checkpoint(checkpoint)?;
                    return Err(error);
                }
            }
        } else {
            None
        };
        state.prefill_progress = span_end;
        state.pending_logits = pending_logits;
        state.managed_completions_drained = false;
        Ok(span_end == state.prompt.len())
    }

    pub(crate) fn restart_temperature_attempt(
        &self,
        state: &mut WhisperDecodeState,
        temperature: f32,
    ) -> Result<()> {
        restart_whisper_temperature_attempt(state, temperature)
    }

    pub(crate) fn resolve_terminal_transition(
        &self,
        state: &mut WhisperDecodeState,
    ) -> Result<WhisperTerminalTransition> {
        if !state.finished {
            return Err(Error::InvalidInput(
                "Whisper terminal policy requires a finished decode attempt".into(),
            ));
        }
        let avg_logprob = if state.sampled_token_count > 0 {
            (state.sum_logprobs / state.sampled_token_count as f64) as f32
        } else {
            f32::NEG_INFINITY
        };
        let attempt = WhisperDecodeAttempt {
            text: state.assembled.trim().to_string(),
            avg_logprob,
            no_speech_prob: state.no_speech_prob,
            ended_with_eot: state.ended_with_eot,
            repetition_loop: state.repetition_loop,
            compression_ratio: token_compression_ratio(
                &state.generated_tokens,
                self.config.vocab_size,
            ),
            generated_token_count: state.generated_tokens.len(),
            sampled_token_count: state.sampled_token_count,
            decode_steps: state.decode_steps,
            profile: WhisperDecodeProfile::new(false),
        };
        let logprob_threshold = self
            .generation
            .logprob_threshold
            .unwrap_or(DEFAULT_LOGPROB_THRESHOLD);
        let no_speech_threshold = self
            .generation
            .no_speech_threshold
            .unwrap_or(DEFAULT_NO_SPEECH_THRESHOLD);
        let temperatures = self.decode_temperatures();
        let next_temperature = usize::try_from(state.attempt_generation)
            .ok()
            .and_then(|index| temperatures.get(index))
            .copied();
        let expected_generation = state.current_managed_generation;
        let new_generation = if next_temperature.is_some() {
            expected_generation.checked_add(1).ok_or_else(|| {
                Error::InferenceError("Whisper managed KV generation overflow".into())
            })?
        } else {
            expected_generation
        };
        let (transition, best_attempt) = whisper_terminal_policy_transition(
            attempt,
            state.temperature,
            state.best_attempt.take(),
            next_temperature,
            logprob_threshold,
            no_speech_threshold,
            self.generation.compression_ratio_threshold,
            expected_generation,
            new_generation,
        );
        state.best_attempt = best_attempt;
        state.pending_retry = match &transition {
            WhisperTerminalTransition::RetryRequired {
                next_temperature,
                expected_generation,
                new_generation,
                ..
            } => Some(WhisperPendingRetry {
                next_temperature: *next_temperature,
                expected_generation: *expected_generation,
                new_generation: *new_generation,
                next_attempt_generation: state.attempt_generation.checked_add(1).ok_or_else(
                    || Error::InferenceError("Whisper retry attempt generation overflow".into()),
                )?,
            }),
            _ => None,
        };
        match &transition {
            WhisperTerminalTransition::Accept { text, .. } => state.assembled = text.clone(),
            WhisperTerminalTransition::SkipNoSpeech { .. } => state.assembled.clear(),
            WhisperTerminalTransition::RetryRequired { .. } => {}
        }
        Ok(transition)
    }

    pub(crate) fn decode_step_retained(
        &self,
        state: &mut WhisperDecodeState,
    ) -> Result<WhisperDecodeStep> {
        if state.prefill_progress != state.prompt.len() || state.pending_logits.is_none() {
            return Err(Error::InvalidInput(
                "Whisper decode requires completed prefill and pending logits".into(),
            ));
        }
        if state.finished || state.generated_tokens.len() >= state.max_steps {
            state.finished = true;
            return Ok(WhisperDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_tokens.len(),
                finished: true,
            });
        }
        with_whisper_decode_step_transaction(state, |state| {
            self.decode_step_retained_transaction(state)
        })
    }

    fn decode_step_retained_transaction(
        &self,
        state: &mut WhisperDecodeState,
    ) -> Result<WhisperDecodeStep> {
        let logits = state
            .pending_logits
            .take()
            .expect("validated pending logits");
        let at_begin = state.generated_tokens.is_empty();
        let mut log_probs = Vec::new();
        let mut profile = WhisperDecodeProfile::new(false);
        let (next, next_logprob, step_no_speech_prob) = self.select_next_token(
            &logits,
            at_begin,
            state.temperature <= 0.0,
            state.temperature,
            &mut state.rng,
            &mut log_probs,
            &mut profile,
        )?;
        state.decode_steps = state
            .decode_steps
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("Whisper decode-step count overflow".into()))?;
        state.sampled_token_count = state
            .sampled_token_count
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("Whisper sampled-token count overflow".into()))?;
        state.sum_logprobs += f64::from(next_logprob);
        if at_begin {
            state.no_speech_prob = step_no_speech_prob;
        }
        if next == self.special.eot {
            state.ended_with_eot = true;
            state.finished = true;
        } else {
            let generated_after_step =
                state.generated_tokens.len().checked_add(1).ok_or_else(|| {
                    Error::InvalidInput("Whisper generated token count overflow".into())
                })?;
            let reaches_limit = generated_after_step >= state.max_steps;
            if !reaches_limit {
                let sequence = state.cross_sequence.ok_or_else(|| {
                    Error::InferenceError("Whisper retained cross sequence was released".into())
                })?;
                let position = state
                    .prompt
                    .len()
                    .checked_add(state.generated_tokens.len())
                    .ok_or_else(|| {
                        Error::InvalidInput("Whisper decode position overflow".into())
                    })?;
                let token = Tensor::new(&[[next]], &self.device.device)?;
                let output = self.whisper.decoder_forward_retained_at(
                    &token,
                    position,
                    &mut state.self_kv,
                    state.cross_runtime.as_ref(),
                    sequence,
                )?;
                state.managed_completions_drained = false;
                let next_logits = self
                    .whisper
                    .decoder_final_linear(&output.i((..1, 0..1))?)
                    .and_then(|logits| logits.i(0).map_err(Error::from))
                    .and_then(|logits| logits.i(0).map_err(Error::from))?;
                state.pending_logits = Some(next_logits);
            } else {
                state.finished = true;
            }
            state.generated_tokens.push(next);
            if let Some((span, repeats)) = find_suffix_token_repetition(&state.generated_tokens) {
                let trim = span.saturating_mul(repeats.saturating_sub(1));
                if trim > 0 && trim <= state.generated_tokens.len() {
                    state
                        .generated_tokens
                        .truncate(state.generated_tokens.len() - trim);
                }
                state.repetition_loop = true;
                state.finished = true;
            }
        }
        let decoded = self.decode_generated_text(&state.generated_tokens)?;
        let text = decoded.trim().to_string();
        let delta = text_delta(&state.assembled, &text);
        state.assembled = text.clone();
        Ok(WhisperDecodeStep {
            delta: delta.to_string(),
            text,
            tokens_generated: state.generated_tokens.len(),
            finished: state.finished,
        })
    }

    pub(crate) const fn supports_resumable_prefill(&self) -> bool {
        true
    }

    pub(crate) const fn supports_continuous_decode_batch(&self) -> bool {
        false
    }

    fn validate_prepared_window(&self, prepared: &WhisperPreparedWindow) -> Result<()> {
        if prepared.preparation_id != self.preparation_id
            || prepared.memory_tokens != self.config.max_source_positions
            || prepared.layers.len() != self.config.decoder_layers
        {
            return Err(Error::InvalidInput(
                "Whisper prepared window belongs to another model or geometry".into(),
            ));
        }
        Ok(())
    }

    fn resolve_request_language_retained(
        &self,
        language: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_runtime: &RetainedStaticAttentionRuntimeV2,
        cross_sequence: RetainedStaticAttentionSequenceId,
    ) -> Result<Option<(u32, String)>> {
        if let Some(language) = language {
            return self.resolve_language_token(language);
        }
        if let Some(default) = self.runtime_tuning.default_language.as_deref() {
            return self.resolve_language_token(default);
        }
        if self.language_token_ids.is_empty() {
            return Ok(None);
        }
        self_kv.reset_invocation()?;
        let tokens = Tensor::new(&[[self.special.sot]], &self.device.device)?;
        let output = self.whisper.decoder_forward_retained_at(
            &tokens,
            0,
            self_kv,
            cross_runtime,
            cross_sequence,
        )?;
        let logits = self
            .whisper
            .decoder_final_linear(&output.i(..1)?)?
            .i(0)?
            .i(0)?;
        let resolved = if let Some(language) = self.detect_language_token_on_cuda(&logits)? {
            Some(language)
        } else {
            let logits = tensor_to_f32_vec1(&logits)?;
            self.language_token_ids
                .iter()
                .filter_map(|token| logits.get(*token as usize).map(|score| (*token, *score)))
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .and_then(|(token, _)| {
                    self.token_id_to_language_code
                        .get(&token)
                        .cloned()
                        .map(|code| (token, code))
                })
        };
        Ok(resolved)
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<WhisperPhysicalStateSpec> {
        super::physical::whisper_physical_state_spec(
            &self.config,
            self.model_dtype,
            self.device.kind.into(),
            stage_graphs,
        )
    }

    pub(crate) fn transcribe_with_details_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &mut InvocationStaticAttentionLease,
    ) -> Result<AsrTranscriptionOutput> {
        self.transcribe_impl(
            audio,
            sample_rate,
            language,
            initial_prompt,
            self_kv,
            cross_kv,
        )
    }

    pub(crate) fn transcribe_with_callback_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &mut InvocationStaticAttentionLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_streaming(
            audio,
            sample_rate,
            language,
            initial_prompt,
            self_kv,
            cross_kv,
            on_delta,
        )
    }

    fn transcribe_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &mut InvocationStaticAttentionLease,
    ) -> Result<AsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let trimmed_audio = self.trimmed_audio_slice(audio, sample_rate);
        let request_started = Instant::now();
        let mel_started = Instant::now();
        let mel = self.prepare_mel(trimmed_audio, sample_rate)?;
        let mel_prepare_ms = mel_started.elapsed().as_secs_f64() * 1000.0;

        self.synchronize_profile_device()?;
        let encoder_started = Instant::now();
        let audio_features = self.whisper.encoder_forward(&mel)?;
        self.whisper.install_cross_attention_memory(
            &audio_features,
            whisper_cross_source_identity(trimmed_audio, sample_rate),
            cross_kv,
        )?;
        self.synchronize_profile_device()?;
        let encoder_forward_ms = elapsed_ms(encoder_started);

        let language_resolution =
            self.resolve_request_language(&self.whisper, language, self_kv, cross_kv)?;
        let resolved_language = language_resolution.resolved.clone();
        let language_hint_used = language_resolution.hint_used;
        let language_detect_ms = language_resolution.detect_ms;

        let initial_prompt_tokens = self.encode_initial_prompt_tokens(initial_prompt)?;
        let prompt_prefix = build_whisper_prompt_prefix(
            &self.special,
            resolved_language
                .as_ref()
                .map(|(language_token, _language_code)| *language_token),
            &initial_prompt_tokens,
            self.config.max_target_positions,
            self.runtime_tuning.initial_prompt_max_tokens,
        )?;
        let prompt = prompt_prefix.ids;
        let prompt_diagnostics = prompt_prefix.diagnostics;

        let max_steps = decode_step_budget(
            prompt.len(),
            self.config.max_target_positions,
            self.resolve_max_decode_tokens(audio.len(), sample_rate),
        )?;
        let temperatures = self.decode_temperatures();
        let logprob_threshold = self
            .generation
            .logprob_threshold
            .unwrap_or(DEFAULT_LOGPROB_THRESHOLD);
        let no_speech_threshold = self
            .generation
            .no_speech_threshold
            .unwrap_or(DEFAULT_NO_SPEECH_THRESHOLD);
        let compression_ratio_threshold = self.generation.compression_ratio_threshold;

        let decode_started = Instant::now();
        let mut attempted_temperatures = Vec::with_capacity(temperatures.len());
        let mut attempt_diagnostics = Vec::with_capacity(temperatures.len());
        let mut fallback_reasons = Vec::<&'static str>::new();
        let mut best_attempt: Option<WhisperDecodeAttempt> = None;
        let mut selected_temperature = temperatures.first().copied().unwrap_or(0.0);
        let mut best_temperature = selected_temperature;
        for (idx, temperature) in temperatures.iter().copied().enumerate() {
            attempted_temperatures.push(temperature);
            let attempt = self.decode_attempt(
                &self.whisper,
                &prompt,
                max_steps,
                temperature,
                self_kv,
                cross_kv,
            )?;
            let no_speech_skip =
                should_skip_as_no_speech(&attempt, logprob_threshold, no_speech_threshold);
            let retry_reasons =
                decode_retry_reasons(&attempt, logprob_threshold, compression_ratio_threshold);
            attempt_diagnostics.push(whisper_attempt_diagnostics(
                temperature,
                &attempt,
                &retry_reasons,
                no_speech_skip,
            ));

            if no_speech_skip {
                record_unique_reason(&mut fallback_reasons, "no_speech");
                best_attempt = Some(WhisperDecodeAttempt {
                    text: String::new(),
                    ..attempt
                });
                selected_temperature = temperature;
                break;
            }

            let is_last_temperature = idx + 1 == temperatures.len();
            let should_retry = !is_last_temperature && !retry_reasons.is_empty();
            if !should_retry {
                if best_attempt
                    .as_ref()
                    .map(|best| is_better_attempt(&attempt, best))
                    .unwrap_or(true)
                {
                    best_attempt = Some(attempt);
                    best_temperature = temperature;
                }
                selected_temperature = best_temperature;
                break;
            }

            record_unique_reasons(&mut fallback_reasons, &retry_reasons);
            if best_attempt
                .as_ref()
                .map(|best| is_better_attempt(&attempt, best))
                .unwrap_or(true)
            {
                best_attempt = Some(attempt);
                best_temperature = temperature;
            }
        }
        let decode_secs = decode_started.elapsed().as_secs_f64();
        let decode_ms = decode_secs * 1000.0;

        let final_attempt = best_attempt.unwrap_or_else(|| WhisperDecodeAttempt {
            text: String::new(),
            avg_logprob: f32::NEG_INFINITY,
            no_speech_prob: None,
            ended_with_eot: false,
            repetition_loop: false,
            compression_ratio: None,
            generated_token_count: 0,
            sampled_token_count: 0,
            decode_steps: 0,
            profile: WhisperDecodeProfile::new(self.runtime_tuning.profile_sync_timings),
        });

        let text = final_attempt.text.trim().to_string();
        let language = resolved_language.map(|(_token_id, code)| code);
        let model_total_ms = request_started.elapsed().as_secs_f64() * 1000.0;
        let generated_tokens_per_second = if decode_secs > 0.0 {
            Some(final_attempt.generated_token_count as f64 / decode_secs)
        } else {
            None
        };
        let diagnostics = json!({
            "model_family": "whisper_asr",
            "device": whisper_device_diagnostics(
                self.device.kind,
                self.model_dtype,
                self.cuda_dtype_shim,
            ),
            "fallback_attempts": attempted_temperatures.len(),
            "attempted_temperatures": attempted_temperatures,
            "fallback_policy": {
                "no_fallback": self.runtime_tuning.no_fallback,
                "max_fallback_retries": self.runtime_tuning.max_fallback_retries,
                "max_attempts": self.runtime_tuning.max_fallback_retries.saturating_add(1),
            },
            "decode_budget": {
                "adaptive_enabled": self.runtime_tuning.adaptive_decode_budget,
                "max_new_tokens_per_second": self.runtime_tuning.max_new_tokens_per_second,
                "min_new_tokens": self.runtime_tuning.min_new_tokens,
                "max_new_tokens_cap": self.runtime_tuning.max_new_tokens_cap,
                "buffer_tokens": self.runtime_tuning.decode_budget_buffer_tokens,
                "resolved_max_new_tokens": self.resolve_max_decode_tokens(audio.len(), sample_rate),
                "audio_seconds": if sample_rate > 0 {
                    audio.len() as f32 / sample_rate as f32
                } else {
                    0.0
                },
            },
            "audio_window": {
                "trim_silence": self.runtime_tuning.trim_silence,
                "input_samples": audio.len(),
                "effective_samples": trimmed_audio.len(),
                "trimmed_samples": audio.len().saturating_sub(trimmed_audio.len()),
            },
            "language_resolution": {
                "strategy": language_resolution.strategy,
                "default_language": self.runtime_tuning.default_language,
            },
            "prompt": {
                "initial_prompt_requested": prompt_diagnostics.initial_prompt_requested,
                "initial_prompt_token_count": prompt_diagnostics.initial_prompt_token_count,
                "initial_prompt_tokens_used": prompt_diagnostics.initial_prompt_tokens_used,
                "initial_prompt_tokens_truncated": prompt_diagnostics.initial_prompt_tokens_truncated,
                "initial_prompt_max_tokens": prompt_diagnostics.initial_prompt_max_tokens,
                "previous_context_token_id": prompt_diagnostics.previous_context_token_id,
                "rolling_context_enabled": prompt_diagnostics.rolling_context_enabled,
            },
            "logit_filters": {
                "suppress_blank": self.runtime_tuning.suppress_blank,
                "blank_token_id": self.special.blank,
                "suppress_numerals": self.runtime_tuning.suppress_numerals,
                "numeral_symbol_token_count": self.numeral_symbol_tokens.len(),
                "device_greedy_decode": self.runtime_tuning.device_greedy_decode,
                "device_greedy_active": self.runtime_tuning.device_greedy_decode
                    && !self.device.device.is_cpu(),
            },
            "selected_temperature": selected_temperature,
            "language_hint_used": language_hint_used,
            "fallback_reasons": fallback_reasons,
            "decode_attempts": attempt_diagnostics,
            "decode": {
                "ended_with_eot": final_attempt.ended_with_eot,
                "repetition_loop": final_attempt.repetition_loop,
                "avg_logprob": final_attempt.avg_logprob,
                "no_speech_prob": final_attempt.no_speech_prob,
                "compression_ratio": final_attempt.compression_ratio,
                "decode_steps": final_attempt.decode_steps,
                "generated_tokens": final_attempt.generated_token_count,
                "generated_token_count": final_attempt.generated_token_count,
                "sampled_token_count": final_attempt.sampled_token_count,
                "generated_tokens_per_second": generated_tokens_per_second,
                "profile": whisper_decode_profile_diagnostics(&final_attempt.profile),
            },
            "profiling": {
                "sync_enabled": self.runtime_tuning.profile_sync_timings,
                "timing_mode": if self.runtime_tuning.profile_sync_timings {
                    "device_synchronized"
                } else {
                    "wall_clock"
                },
            },
            "timings_ms": {
                "mel_prepare": mel_prepare_ms,
                "encoder_forward": encoder_forward_ms,
                "language_detect": language_detect_ms,
                "decode": decode_ms,
                "model_total": model_total_ms,
            }
        });

        Ok(AsrTranscriptionOutput {
            text,
            language,
            diagnostics: Some(diagnostics),
        })
    }

    fn transcribe_streaming(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &mut InvocationStaticAttentionLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let trimmed_audio = self.trimmed_audio_slice(audio, sample_rate);
        let mel = self.prepare_mel(trimmed_audio, sample_rate)?;
        let audio_features = self.whisper.encoder_forward(&mel)?;
        self.whisper.install_cross_attention_memory(
            &audio_features,
            whisper_cross_source_identity(trimmed_audio, sample_rate),
            cross_kv,
        )?;

        let language_resolution =
            self.resolve_request_language(&self.whisper, language, self_kv, cross_kv)?;
        let resolved_language = language_resolution.resolved;

        let initial_prompt_tokens = self.encode_initial_prompt_tokens(initial_prompt)?;
        let prompt = build_whisper_prompt_prefix(
            &self.special,
            resolved_language
                .as_ref()
                .map(|(language_token, _language_code)| *language_token),
            &initial_prompt_tokens,
            self.config.max_target_positions,
            self.runtime_tuning.initial_prompt_max_tokens,
        )?
        .ids;

        let max_steps = decode_step_budget(
            prompt.len(),
            self.config.max_target_positions,
            self.resolve_max_decode_tokens(audio.len(), sample_rate),
        )?;
        let temperatures = self.decode_temperatures();
        let logprob_threshold = self
            .generation
            .logprob_threshold
            .unwrap_or(DEFAULT_LOGPROB_THRESHOLD);
        let no_speech_threshold = self
            .generation
            .no_speech_threshold
            .unwrap_or(DEFAULT_NO_SPEECH_THRESHOLD);
        let compression_ratio_threshold = self.generation.compression_ratio_threshold;

        let first_temperature = temperatures.first().copied().unwrap_or(0.0);
        let first_attempt = self.decode_attempt_streaming(
            &self.whisper,
            &prompt,
            max_steps,
            first_temperature,
            self_kv,
            cross_kv,
            on_delta,
        )?;

        if should_skip_as_no_speech(&first_attempt, logprob_threshold, no_speech_threshold) {
            return Ok(String::new());
        }

        let mut best_attempt = first_attempt;
        let mut should_retry = temperatures.len() > 1
            && (best_attempt.text.trim().is_empty()
                || should_retry_decode(
                    &best_attempt,
                    logprob_threshold,
                    compression_ratio_threshold,
                ));
        if !should_retry {
            return Ok(best_attempt.text.trim().to_string());
        }

        for (idx, temperature) in temperatures.iter().copied().enumerate().skip(1) {
            let attempt = self.decode_attempt(
                &self.whisper,
                &prompt,
                max_steps,
                temperature,
                self_kv,
                cross_kv,
            )?;

            if should_skip_as_no_speech(&attempt, logprob_threshold, no_speech_threshold) {
                return Ok(String::new());
            }

            let is_last_temperature = idx + 1 == temperatures.len();
            should_retry = !is_last_temperature
                && (attempt.text.trim().is_empty()
                    || should_retry_decode(
                        &attempt,
                        logprob_threshold,
                        compression_ratio_threshold,
                    ));
            if !should_retry {
                if is_better_attempt(&attempt, &best_attempt) {
                    best_attempt = attempt;
                }
                break;
            }

            if is_better_attempt(&attempt, &best_attempt) {
                best_attempt = attempt;
            }
        }

        Ok(best_attempt.text.trim().to_string())
    }

    fn prepare_mel(&self, audio: &[f32], sample_rate: u32) -> Result<Tensor> {
        let mono_16khz = if sample_rate == SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, SAMPLE_RATE)
        };

        let mut mel_spec = self.mel.compute(&mono_16khz)?;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        // Whisper encoder downsamples by 2 before positional embeddings.
        let max_input_frames = self.config.max_source_positions.saturating_mul(2).max(1);
        let n_mels = self.config.num_mel_bins;
        pad_or_trim_mel_frames(&mut mel_spec, n_mels, max_input_frames);
        let frames = mel_spec.len();
        let mut flat = vec![0f32; frames * n_mels];
        for (frame_idx, frame) in mel_spec.iter().enumerate() {
            for mel_idx in 0..n_mels {
                flat[mel_idx * frames + frame_idx] = frame[mel_idx];
            }
        }

        let mel = Tensor::from_vec(flat, (1, n_mels, frames), &self.device.device)?;
        if mel.dtype() != self.model_dtype {
            return Ok(mel.to_dtype(self.model_dtype)?);
        }
        Ok(mel)
    }

    fn synchronize_profile_device(&self) -> Result<()> {
        if self.runtime_tuning.profile_sync_timings {
            self.device.device.synchronize()?;
        }
        Ok(())
    }

    fn mask_for_logits(&self, mask: &Tensor, logits: &Tensor) -> Result<Tensor> {
        if mask.dtype() == logits.dtype() {
            Ok(mask.clone())
        } else {
            mask.to_dtype(logits.dtype()).map_err(Error::from)
        }
    }

    fn masked_decode_logits(&self, logits: &Tensor, at_begin: bool) -> Result<Tensor> {
        let base_mask = self.mask_for_logits(&self.decode_mask, logits)?;
        let mut masked = logits.broadcast_add(&base_mask)?;
        if at_begin {
            let begin_mask = self.mask_for_logits(&self.decode_begin_mask, logits)?;
            masked = masked.broadcast_add(&begin_mask)?;
        }
        Ok(masked)
    }

    fn greedy_decode_step_on_device(
        &self,
        logits: &Tensor,
        at_begin: bool,
        inv_temperature: f32,
    ) -> Result<(u32, f32, Option<f32>)> {
        let masked = self.masked_decode_logits(logits, at_begin)?;
        greedy_decode_step_from_masked_logits(
            &masked,
            &self.special,
            self.config.vocab_size,
            inv_temperature,
        )
    }

    fn select_next_token<R: Rng + ?Sized>(
        &self,
        logits: &Tensor,
        at_begin: bool,
        deterministic: bool,
        temperature: f32,
        rng: &mut R,
        log_probs_buf: &mut Vec<f32>,
        profile: &mut WhisperDecodeProfile,
    ) -> Result<(u32, f32, Option<f32>)> {
        let inv_temperature = 1.0f32 / temperature.max(1e-6);
        if deterministic && self.runtime_tuning.device_greedy_decode && !self.device.device.is_cpu()
        {
            match self.greedy_decode_step_on_device(logits, at_begin, inv_temperature) {
                Ok(step) => {
                    profile.device_greedy_steps = profile.device_greedy_steps.saturating_add(1);
                    return Ok(step);
                }
                Err(err) => {
                    profile.device_greedy_fallbacks =
                        profile.device_greedy_fallbacks.saturating_add(1);
                    debug!("Whisper device greedy decode fell back to host logits: {err}");
                }
            }
        }

        let mut logits_vec = if profile.enabled {
            let started = Instant::now();
            let logits_vec = tensor_to_f32_vec1(logits)?;
            profile.logits_to_host_ms += elapsed_ms(started);
            logits_vec
        } else {
            tensor_to_f32_vec1(logits)?
        };
        profile.host_logits_steps = profile.host_logits_steps.saturating_add(1);
        self.apply_decode_constraints(&mut logits_vec, at_begin);
        if deterministic {
            if let Some(logsumexp) = scaled_logsumexp(&logits_vec, inv_temperature) {
                let (next, best_logit) = best_finite_logit(&logits_vec, self.special.eot);
                let next_scaled_logit = best_logit * inv_temperature;
                let next_logprob = if next_scaled_logit.is_finite() {
                    next_scaled_logit - logsumexp
                } else {
                    f32::NEG_INFINITY
                };
                let no_speech_prob = self.special.no_speech.and_then(|token_id| {
                    probability_for_token_from_logits(
                        &logits_vec,
                        token_id,
                        logsumexp,
                        inv_temperature,
                    )
                });
                Ok((next, next_logprob, no_speech_prob))
            } else {
                Ok((self.special.eot, f32::NEG_INFINITY, None))
            }
        } else {
            logits_to_log_probs_in_place(&logits_vec, temperature, log_probs_buf);
            let no_speech_prob = self
                .special
                .no_speech
                .and_then(|token_id| probability_for_token(log_probs_buf, token_id));
            let (next, next_logprob) =
                sample_token_from_log_probs(log_probs_buf, temperature, self.special.eot, rng);
            Ok((next, next_logprob, no_speech_prob))
        }
    }

    fn resolve_language_token(&self, language: &str) -> Result<Option<(u32, String)>> {
        let normalized = language.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Ok(None);
        }

        let language_code = if let Some(code) = normalized
            .strip_prefix("<|")
            .and_then(|inner| inner.strip_suffix("|>"))
        {
            code.to_string()
        } else if has_whisper_language_token(
            &self.generation.lang_to_id,
            &normalized,
            &self.tokenizer,
        ) {
            normalized
        } else if let Some(code) = language_name_to_code(&normalized) {
            code.to_string()
        } else if let Some(code) = language_alias_to_code(&normalized) {
            code.to_string()
        } else {
            return Err(Error::InvalidInput(format!(
                "Unsupported Whisper language '{}'",
                language
            )));
        };

        let token = format!("<|{}|>", language_code);
        let token_id = self
            .generation
            .lang_to_id
            .get(&token)
            .copied()
            .or_else(|| self.tokenizer.token_to_id(&token))
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Whisper model does not support language token '{}'",
                    token
                ))
            })?;

        Ok(Some((token_id, language_code)))
    }

    fn detect_language_token(
        &self,
        whisper: &WhisperModel,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &InvocationStaticAttentionLease,
    ) -> Result<Option<(u32, String)>> {
        if self.language_token_ids.is_empty() {
            return Ok(None);
        }

        self_kv.reset_invocation()?;
        let tokens = Tensor::new(&[[self.special.sot]], &self.device.device)?;
        let ys = whisper.decoder_forward_physical_at(&tokens, 0, self_kv, cross_kv)?;
        let logits = whisper.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        if let Some(language) = self.detect_language_token_on_cuda(&logits)? {
            return Ok(Some(language));
        }
        let logits_vec = tensor_to_f32_vec1(&logits)?;

        let mut best_token: Option<u32> = None;
        let mut best_score = f32::NEG_INFINITY;
        for token_id in &self.language_token_ids {
            let idx = *token_id as usize;
            if idx >= logits_vec.len() {
                continue;
            }
            let score = logits_vec[idx];
            if score > best_score {
                best_score = score;
                best_token = Some(*token_id);
            }
        }

        let Some(token_id) = best_token else {
            return Ok(None);
        };
        let Some(code) = self.token_id_to_language_code.get(&token_id).cloned() else {
            return Ok(None);
        };

        Ok(Some((token_id, code)))
    }

    fn detect_language_token_on_cuda(&self, logits: &Tensor) -> Result<Option<(u32, String)>> {
        if !logits.device().is_cuda() {
            return Ok(None);
        }

        let Some((start, len)) = self.language_token_range else {
            return Ok(None);
        };
        let Some(end) = start.checked_add(len) else {
            return Ok(None);
        };
        if end > logits.dim(0)? {
            return Ok(None);
        }

        let language_logits = logits.narrow(0, start, len)?;
        let offset = language_logits.argmax(0)?.to_scalar::<u32>()? as usize;
        let token_id = start.saturating_add(offset) as u32;
        let Some(code) = self.token_id_to_language_code.get(&token_id).cloned() else {
            return Ok(None);
        };

        Ok(Some((token_id, code)))
    }

    fn decode_generated_text(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(token_ids)
    }

    fn encode_initial_prompt_tokens(&self, initial_prompt: Option<&str>) -> Result<Vec<u32>> {
        let Some(prompt) = initial_prompt
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
        else {
            return Ok(Vec::new());
        };
        self.tokenizer.encode(prompt)
    }

    fn decode_temperatures(&self) -> Vec<f32> {
        // Mirrors whisper.cpp/transformers temperature fallback ladder.
        let start = self.generation.temperature.unwrap_or(0.0).clamp(0.0, 1.0);
        let inc = self
            .generation
            .temperature_increment_on_fallback
            .unwrap_or(DEFAULT_TEMPERATURE_FALLBACK_INC);
        capped_decode_temperatures(
            start,
            inc,
            self.runtime_tuning.no_fallback,
            self.runtime_tuning.max_fallback_retries,
        )
    }

    fn resolve_max_decode_tokens(&self, audio_len_samples: usize, sample_rate: u32) -> usize {
        let configured_max = self.generation.max_length.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        let cap = configured_max.min(self.runtime_tuning.max_new_tokens_cap.max(1));
        if !self.runtime_tuning.adaptive_decode_budget || sample_rate == 0 {
            return cap;
        }

        let audio_secs = audio_len_samples as f32 / sample_rate as f32;
        adaptive_decode_budget(
            audio_secs,
            cap,
            self.runtime_tuning.max_new_tokens_per_second,
            self.runtime_tuning.min_new_tokens,
            self.runtime_tuning.decode_budget_buffer_tokens,
        )
    }

    fn resolve_request_language(
        &self,
        whisper: &WhisperModel,
        language: Option<&str>,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &InvocationStaticAttentionLease,
    ) -> Result<WhisperLanguageResolution> {
        if let Some(language) = language {
            let resolved = self.resolve_language_token(language)?;
            return Ok(WhisperLanguageResolution {
                resolved,
                hint_used: true,
                detect_ms: 0.0,
                strategy: "hint",
            });
        }

        if let Some(default_language) = self.runtime_tuning.default_language.as_deref() {
            let resolved = self.resolve_language_token(default_language)?;
            return Ok(WhisperLanguageResolution {
                resolved,
                hint_used: false,
                detect_ms: 0.0,
                strategy: "default",
            });
        }

        self.synchronize_profile_device()?;
        let detect_started = Instant::now();
        let resolved = self.detect_language_token(whisper, self_kv, cross_kv)?;
        self.synchronize_profile_device()?;
        let detect_ms = elapsed_ms(detect_started);
        Ok(WhisperLanguageResolution {
            resolved,
            hint_used: false,
            detect_ms,
            strategy: "detected",
        })
    }

    fn trimmed_audio_slice<'a>(&self, audio: &'a [f32], sample_rate: u32) -> &'a [f32] {
        if !self.runtime_tuning.trim_silence {
            return audio;
        }

        let (start, end) = trimmed_audio_bounds(
            audio,
            sample_rate,
            self.runtime_tuning.silence_trim_threshold_scale,
            self.runtime_tuning.silence_trim_min_abs,
            self.runtime_tuning.silence_trim_margin_ms,
            self.runtime_tuning.silence_trim_min_leading_ms,
            self.runtime_tuning.silence_trim_min_trailing_ms,
            self.runtime_tuning.silence_trim_min_clip_secs,
        );
        &audio[start..end]
    }
}

fn capped_decode_temperatures(
    start: f32,
    temperature_inc: f32,
    no_fallback: bool,
    max_fallback_retries: usize,
) -> Vec<f32> {
    if no_fallback {
        return vec![start];
    }

    let mut temperatures = Vec::new();
    if temperature_inc <= 0.0 {
        temperatures.push(start);
    } else {
        let mut t = start;
        while t <= 1.0 + 1e-6 {
            temperatures.push((t * 100.0).round() / 100.0);
            t += temperature_inc;
        }
    }

    if temperatures.is_empty() {
        temperatures.push(start);
    }

    let max_attempts = max_fallback_retries.saturating_add(1);
    if temperatures.len() > max_attempts {
        temperatures.truncate(max_attempts);
    }
    temperatures
}

fn adaptive_decode_budget(
    audio_secs: f32,
    configured_cap: usize,
    tokens_per_second: f32,
    min_new_tokens: usize,
    buffer_tokens: usize,
) -> usize {
    if configured_cap == 0 {
        return 1;
    }
    let tps = tokens_per_second.max(0.0);
    let scaled = (audio_secs.max(0.0) * tps).ceil() as usize;
    let proposed = scaled
        .saturating_add(buffer_tokens)
        .max(min_new_tokens)
        .max(1);
    proposed.min(configured_cap.max(1))
}

fn build_whisper_prompt_prefix(
    special: &WhisperSpecialTokens,
    language_token: Option<u32>,
    initial_prompt_tokens: &[u32],
    max_target_positions: usize,
    initial_prompt_max_tokens: usize,
) -> Result<WhisperPromptPrefix> {
    let mut controls = Vec::with_capacity(4);
    controls.push(special.sot);
    if let Some(language_token) = language_token {
        controls.push(language_token);
    }
    controls.push(special.transcribe);
    if let Some(no_timestamps) = special.no_timestamps {
        controls.push(no_timestamps);
    }

    if controls.len() >= max_target_positions {
        return Err(Error::InvalidInput(format!(
            "Whisper prompt controls length {} exceeds decoder context {}",
            controls.len(),
            max_target_positions
        )));
    }

    let initial_prompt_requested = !initial_prompt_tokens.is_empty();
    let base_context_budget = max_target_positions
        .saturating_sub(controls.len())
        .saturating_sub(1);
    let can_use_previous_context =
        initial_prompt_requested && special.sot_prev.is_some() && base_context_budget > 1;
    let previous_context_tokens = usize::from(can_use_previous_context);
    let available_for_context = base_context_budget.saturating_sub(previous_context_tokens);
    let prompt_budget = available_for_context.min(initial_prompt_max_tokens);
    let initial_prompt_tokens_used = initial_prompt_tokens.len().min(prompt_budget);
    let initial_prompt_tokens_truncated = initial_prompt_tokens
        .len()
        .saturating_sub(initial_prompt_tokens_used);
    let previous_context_token_id = if initial_prompt_tokens_used > 0 && can_use_previous_context {
        special.sot_prev
    } else {
        None
    };

    let mut ids =
        Vec::with_capacity(previous_context_tokens + initial_prompt_tokens_used + controls.len());
    if let Some(token_id) = previous_context_token_id {
        ids.push(token_id);
    }
    if initial_prompt_tokens_used > 0 {
        let start = initial_prompt_tokens.len() - initial_prompt_tokens_used;
        ids.extend_from_slice(&initial_prompt_tokens[start..]);
    }
    ids.extend(controls);

    Ok(WhisperPromptPrefix {
        ids,
        diagnostics: WhisperPromptDiagnostics {
            initial_prompt_requested,
            initial_prompt_token_count: initial_prompt_tokens.len(),
            initial_prompt_tokens_used,
            initial_prompt_tokens_truncated,
            initial_prompt_max_tokens,
            previous_context_token_id,
            rolling_context_enabled: false,
        },
    })
}

fn trimmed_audio_bounds(
    audio: &[f32],
    sample_rate: u32,
    threshold_scale: f32,
    min_abs: f32,
    margin_ms: usize,
    min_leading_ms: usize,
    min_trailing_ms: usize,
    min_clip_secs: f32,
) -> (usize, usize) {
    if audio.is_empty() || sample_rate == 0 {
        return (0, audio.len());
    }

    let clip_secs = audio.len() as f32 / sample_rate as f32;
    if clip_secs < min_clip_secs.max(0.0) {
        return (0, audio.len());
    }

    let peak = audio.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak <= 0.0 {
        return (0, audio.len());
    }

    let threshold = (peak * threshold_scale.max(0.0)).max(min_abs.max(0.0));
    let Some(first) = audio.iter().position(|sample| sample.abs() >= threshold) else {
        return (0, audio.len());
    };
    let Some(last) = audio.iter().rposition(|sample| sample.abs() >= threshold) else {
        return (0, audio.len());
    };

    let margin = sample_rate as usize * margin_ms / 1000;
    let mut start = first.saturating_sub(margin);
    let mut end = (last.saturating_add(margin).saturating_add(1)).min(audio.len());

    let min_leading_samples =
        ((min_leading_ms as u64).saturating_mul(sample_rate as u64) / 1000) as usize;
    let min_trailing_samples =
        ((min_trailing_ms as u64).saturating_mul(sample_rate as u64) / 1000) as usize;
    if start < min_leading_samples {
        start = 0;
    }
    if audio.len().saturating_sub(end) < min_trailing_samples {
        end = audio.len();
    }

    if end <= start {
        return (0, audio.len());
    }
    if start == 0 && end == audio.len() {
        return (0, audio.len());
    }
    (start, end)
}

fn pad_or_trim_mel_frames(mel_spec: &mut Vec<Vec<f32>>, n_mels: usize, target_frames: usize) {
    if mel_spec.len() > target_frames {
        mel_spec.truncate(target_frames);
    } else if mel_spec.len() < target_frames {
        mel_spec.resize_with(target_frames, || vec![0.0; n_mels]);
    }
}

impl WhisperTurboAsrModel {
    fn decode_attempt(
        &self,
        whisper: &WhisperModel,
        prompt_prefix: &[u32],
        max_steps: usize,
        temperature: f32,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &InvocationStaticAttentionLease,
    ) -> Result<WhisperDecodeAttempt> {
        self_kv.reset_invocation()?;
        let mut rng = rand::thread_rng();
        let deterministic = temperature <= 0.0;
        let mut prompt = prompt_prefix.to_vec();
        let mut generated_tokens = Vec::<u32>::new();
        let mut sum_logprobs = 0.0f64;
        let mut sampled_token_count = 0usize;
        let mut no_speech_prob: Option<f32> = None;
        let mut ended_with_eot = false;
        let mut repetition_loop = false;
        let mut decode_steps = 0usize;
        let profile_enabled = self.runtime_tuning.profile_sync_timings;
        let mut profile = WhisperDecodeProfile::new(profile_enabled);
        let mut log_probs_buf = Vec::<f32>::new();

        for step_idx in 0..max_steps {
            decode_steps = decode_steps.saturating_add(1);
            let mut single_token = [0u32; 1];
            let (input_tokens, position_offset) = if step_idx == 0 {
                (prompt.as_slice(), 0)
            } else {
                single_token[0] = *prompt.last().ok_or_else(|| {
                    Error::InferenceError("Whisper decode prompt is empty".to_string())
                })?;
                (&single_token[..], prompt.len().saturating_sub(1))
            };
            self.synchronize_profile_device()?;
            let step_started = if profile_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let tokens_t = if profile_enabled {
                let started = Instant::now();
                let tokens = Tensor::new(input_tokens, &self.device.device)?.unsqueeze(0)?;
                self.synchronize_profile_device()?;
                profile.token_tensor_ms += elapsed_ms(started);
                tokens
            } else {
                Tensor::new(input_tokens, &self.device.device)?.unsqueeze(0)?
            };
            let ys = if profile_enabled {
                let started = Instant::now();
                let ys = whisper.decoder_forward_physical_at(
                    &tokens_t,
                    position_offset,
                    self_kv,
                    cross_kv,
                )?;
                self.synchronize_profile_device()?;
                profile.decoder_forward_ms += elapsed_ms(started);
                ys
            } else {
                whisper.decoder_forward_physical_at(
                    &tokens_t,
                    position_offset,
                    self_kv,
                    cross_kv,
                )?
            };
            let (_, seq_len, _) = ys.dims3()?;
            let logits = if profile_enabled {
                let started = Instant::now();
                let logits = whisper
                    .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                    .i(0)?
                    .i(0)?;
                self.synchronize_profile_device()?;
                profile.final_linear_ms += elapsed_ms(started);
                logits
            } else {
                whisper
                    .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                    .i(0)?
                    .i(0)?
            };

            let sampling_started = if profile_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let (next, next_logprob, step_no_speech_prob) = self.select_next_token(
                &logits,
                step_idx == 0,
                deterministic,
                temperature,
                &mut rng,
                &mut log_probs_buf,
                &mut profile,
            )?;
            if let Some(started) = sampling_started {
                profile.sampling_ms += elapsed_ms(started);
            }

            if step_idx == 0 {
                no_speech_prob = step_no_speech_prob;
            }
            sum_logprobs += next_logprob as f64;
            sampled_token_count = sampled_token_count.saturating_add(1);
            if let Some(started) = step_started {
                profile.step_total_ms += elapsed_ms(started);
            }

            if next == self.special.eot {
                ended_with_eot = true;
                break;
            }

            generated_tokens.push(next);
            prompt.push(next);

            if let Some((span, repeats)) = find_suffix_token_repetition(&generated_tokens) {
                let trim = span.saturating_mul(repeats.saturating_sub(1));
                if trim > 0 && trim <= generated_tokens.len() {
                    generated_tokens.truncate(generated_tokens.len() - trim);
                }
                repetition_loop = true;
                break;
            }
        }

        let text = self
            .decode_generated_text(&generated_tokens)?
            .trim()
            .to_string();
        let avg_logprob = if sampled_token_count > 0 {
            (sum_logprobs / sampled_token_count as f64) as f32
        } else {
            f32::NEG_INFINITY
        };
        let compression_ratio = token_compression_ratio(&generated_tokens, self.config.vocab_size);

        Ok(WhisperDecodeAttempt {
            text,
            avg_logprob,
            no_speech_prob,
            ended_with_eot,
            repetition_loop,
            compression_ratio,
            generated_token_count: generated_tokens.len(),
            sampled_token_count,
            decode_steps,
            profile,
        })
    }

    fn decode_attempt_streaming(
        &self,
        whisper: &WhisperModel,
        prompt_prefix: &[u32],
        max_steps: usize,
        temperature: f32,
        self_kv: &mut PhysicalPagedKvCache,
        cross_kv: &InvocationStaticAttentionLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<WhisperDecodeAttempt> {
        self_kv.reset_invocation()?;
        let mut rng = rand::thread_rng();
        let deterministic = temperature <= 0.0;
        let mut prompt = prompt_prefix.to_vec();
        let mut generated_tokens = Vec::<u32>::new();
        let mut sum_logprobs = 0.0f64;
        let mut sampled_token_count = 0usize;
        let mut no_speech_prob: Option<f32> = None;
        let mut ended_with_eot = false;
        let mut repetition_loop = false;
        let mut streamed_text = String::new();
        let mut log_probs_buf = Vec::<f32>::new();
        let mut decode_steps = 0usize;
        let mut profile = WhisperDecodeProfile::new(false);

        for step_idx in 0..max_steps {
            decode_steps = decode_steps.saturating_add(1);
            let mut single_token = [0u32; 1];
            let (input_tokens, position_offset) = if step_idx == 0 {
                (prompt.as_slice(), 0)
            } else {
                single_token[0] = *prompt.last().ok_or_else(|| {
                    Error::InferenceError("Whisper decode prompt is empty".to_string())
                })?;
                (&single_token[..], prompt.len().saturating_sub(1))
            };
            let tokens_t = Tensor::new(input_tokens, &self.device.device)?.unsqueeze(0)?;
            let ys = whisper.decoder_forward_physical_at(
                &tokens_t,
                position_offset,
                self_kv,
                cross_kv,
            )?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = whisper
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let (next, next_logprob, step_no_speech_prob) = self.select_next_token(
                &logits,
                step_idx == 0,
                deterministic,
                temperature,
                &mut rng,
                &mut log_probs_buf,
                &mut profile,
            )?;

            if step_idx == 0 {
                no_speech_prob = step_no_speech_prob;
            }
            sum_logprobs += next_logprob as f64;
            sampled_token_count = sampled_token_count.saturating_add(1);

            if next == self.special.eot {
                ended_with_eot = true;
                break;
            }

            generated_tokens.push(next);
            prompt.push(next);

            let decoded = self.decode_generated_text(&generated_tokens)?;
            let trimmed = decoded.trim();
            let delta = text_delta(&streamed_text, trimmed);
            if !delta.is_empty() {
                on_delta(delta);
                streamed_text.clear();
                streamed_text.push_str(trimmed);
            }

            if let Some((span, repeats)) = find_suffix_token_repetition(&generated_tokens) {
                let trim = span.saturating_mul(repeats.saturating_sub(1));
                if trim > 0 && trim <= generated_tokens.len() {
                    generated_tokens.truncate(generated_tokens.len() - trim);
                }
                repetition_loop = true;
                break;
            }
        }

        let text = self
            .decode_generated_text(&generated_tokens)?
            .trim()
            .to_string();
        let avg_logprob = if sampled_token_count > 0 {
            (sum_logprobs / sampled_token_count as f64) as f32
        } else {
            f32::NEG_INFINITY
        };
        let compression_ratio = token_compression_ratio(&generated_tokens, self.config.vocab_size);

        Ok(WhisperDecodeAttempt {
            text,
            avg_logprob,
            no_speech_prob,
            ended_with_eot,
            repetition_loop,
            compression_ratio,
            generated_token_count: generated_tokens.len(),
            sampled_token_count,
            decode_steps,
            profile,
        })
    }

    fn apply_decode_constraints(&self, logits: &mut [f32], at_begin: bool) {
        apply_whisper_decode_constraints(
            logits,
            at_begin,
            &self.suppress_tokens,
            &self.generation.begin_suppress_tokens,
            &self.language_token_ids,
            &self.special,
            self.runtime_tuning.suppress_blank,
            &self.numeral_symbol_tokens,
        );
    }
}

fn read_generation_config(model_dir: &Path) -> Result<WhisperGenerationConfig> {
    let generation_path = model_dir.join("generation_config.json");
    if !generation_path.exists() {
        return Ok(WhisperGenerationConfig::default());
    }
    let generation_data = fs::read_to_string(generation_path)?;
    Ok(serde_json::from_str::<WhisperGenerationConfig>(
        &generation_data,
    )?)
}

fn resolve_special_tokens(
    tokenizer: &Tokenizer,
    generation: &WhisperGenerationConfig,
) -> Result<WhisperSpecialTokens> {
    let sot = tokenizer.token_to_id(whisper::SOT_TOKEN).ok_or_else(|| {
        Error::TokenizationError("Missing <|startoftranscript|> token".to_string())
    })?;
    let sot_prev = tokenizer.token_to_id("<|startofprev|>");
    let transcribe = tokenizer
        .token_to_id(whisper::TRANSCRIBE_TOKEN)
        .or_else(|| generation.task_to_id.get("transcribe").copied())
        .ok_or_else(|| Error::TokenizationError("Missing <|transcribe|> token".to_string()))?;
    let eot = tokenizer
        .token_to_id(whisper::EOT_TOKEN)
        .or(generation.eos_token_id)
        .ok_or_else(|| Error::TokenizationError("Missing <|endoftext|> token".to_string()))?;
    let blank = tokenizer
        .encode(" ")
        .ok()
        .and_then(|ids| (ids.len() == 1).then_some(ids[0]))
        .or_else(|| tokenizer.token_to_id(" "));
    let no_timestamps = generation
        .no_timestamps_token_id
        .or_else(|| tokenizer.token_to_id(whisper::NO_TIMESTAMPS_TOKEN));
    let no_speech = whisper::NO_SPEECH_TOKENS
        .iter()
        .find_map(|token| tokenizer.token_to_id(token));

    Ok(WhisperSpecialTokens {
        sot,
        sot_prev,
        transcribe,
        eot,
        blank,
        no_timestamps,
        no_speech,
    })
}

fn build_numeral_symbol_tokens(tokenizer: &Tokenizer, special: &WhisperSpecialTokens) -> Vec<u32> {
    let special_ids: HashSet<u32> = [
        Some(special.sot),
        special.sot_prev,
        Some(special.transcribe),
        Some(special.eot),
        special.blank,
        special.no_timestamps,
        special.no_speech,
    ]
    .into_iter()
    .flatten()
    .collect();

    let mut tokens: Vec<u32> = tokenizer
        .vocab()
        .into_iter()
        .filter_map(|(token, token_id)| {
            if special_ids.contains(&token_id) || is_whisper_control_token(&token) {
                return None;
            }
            token_contains_numeral_or_symbol(&token).then_some(token_id)
        })
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn is_whisper_control_token(token: &str) -> bool {
    token.starts_with("<|") && token.ends_with("|>")
}

fn token_contains_numeral_or_symbol(token: &str) -> bool {
    token.chars().any(|ch| {
        ch.is_numeric()
            || matches!(
                ch,
                '$' | '%'
                    | '+'
                    | '='
                    | '#'
                    | '@'
                    | '*'
                    | '/'
                    | '\\'
                    | '<'
                    | '>'
                    | '^'
                    | '_'
                    | '~'
                    | '&'
            )
    })
}

fn build_language_token_maps(
    tokenizer: &Tokenizer,
    generation: &WhisperGenerationConfig,
) -> (Vec<u32>, HashMap<u32, String>) {
    let mut token_to_lang = HashMap::new();
    let mut lang_ids = Vec::new();

    if generation.lang_to_id.is_empty() {
        for (code, _name) in WHISPER_LANGUAGES {
            let token = format!("<|{}|>", code);
            if let Some(token_id) = tokenizer.token_to_id(&token) {
                lang_ids.push(token_id);
                token_to_lang.insert(token_id, (*code).to_string());
            }
        }
    } else {
        for (token, token_id) in &generation.lang_to_id {
            if let Some(code) = token
                .strip_prefix("<|")
                .and_then(|inner| inner.strip_suffix("|>"))
            {
                lang_ids.push(*token_id);
                token_to_lang.insert(*token_id, code.to_string());
            }
        }
    }

    lang_ids.sort_unstable();
    lang_ids.dedup();
    (lang_ids, token_to_lang)
}

fn contiguous_token_range(token_ids: &[u32]) -> Option<(usize, usize)> {
    let first = *token_ids.first()? as usize;
    for (offset, token_id) in token_ids.iter().enumerate() {
        if *token_id as usize != first + offset {
            return None;
        }
    }
    Some((first, token_ids.len()))
}

fn has_whisper_language_token(
    generation_lang_to_id: &HashMap<String, u32>,
    code: &str,
    tokenizer: &Tokenizer,
) -> bool {
    let token = format!("<|{}|>", code);
    generation_lang_to_id.contains_key(&token) || tokenizer.token_to_id(&token).is_some()
}

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|raw| {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_nonempty_string(key: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn tensor_to_f32_vec1(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = if tensor.dtype() == DType::F32 {
        tensor.clone()
    } else {
        tensor.to_dtype(DType::F32)?
    };
    tensor.to_vec1::<f32>().map_err(Error::from)
}

fn greedy_decode_step_from_masked_logits(
    masked: &Tensor,
    special: &WhisperSpecialTokens,
    vocab_size: usize,
    inv_temperature: f32,
) -> Result<(u32, f32, Option<f32>)> {
    let masked_f32 = if masked.dtype() == DType::F32 {
        masked.clone()
    } else {
        masked.to_dtype(DType::F32)?
    };
    let scaled = if (inv_temperature - 1.0).abs() <= f32::EPSILON {
        masked_f32.clone()
    } else {
        (&masked_f32 * inv_temperature as f64)?
    };
    let next = masked.argmax(0)?.to_scalar::<u32>()?;
    let max_scaled = scaled.max(0)?;
    let logsumexp = scaled
        .broadcast_sub(&max_scaled)?
        .exp()?
        .sum_all()?
        .log()?
        .broadcast_add(&max_scaled)?;
    let next_scaled_logit = scaled.i(next as usize)?;
    let no_speech_scaled_logit = special
        .no_speech
        .filter(|token_id| (*token_id as usize) < vocab_size)
        .and_then(|token_id| scaled.i(token_id as usize).ok());
    let mut stat_tensors = vec![max_scaled, logsumexp, next_scaled_logit];
    if let Some(no_speech_scaled_logit) = no_speech_scaled_logit {
        stat_tensors.push(no_speech_scaled_logit);
    }
    let stat_refs: Vec<&Tensor> = stat_tensors.iter().collect();
    let stats = Tensor::stack(&stat_refs, 0)?.to_vec1::<f32>()?;
    let max_scaled_value = stats[0];
    if !max_scaled_value.is_finite() {
        return Ok((special.eot, f32::NEG_INFINITY, None));
    }
    let logsumexp = stats[1];
    let next_scaled_logit = stats[2];
    let next_logprob = if next_scaled_logit.is_finite() && logsumexp.is_finite() {
        next_scaled_logit - logsumexp
    } else {
        f32::NEG_INFINITY
    };
    let no_speech_prob = stats.get(3).and_then(|scaled_logit| {
        if scaled_logit.is_finite() && logsumexp.is_finite() {
            Some((*scaled_logit - logsumexp).exp())
        } else {
            None
        }
    });
    Ok((next, next_logprob, no_speech_prob))
}

fn build_whisper_decode_mask_tensors(
    vocab_size: usize,
    suppress_tokens: &[u32],
    begin_suppress_tokens: &[u32],
    language_token_ids: &[u32],
    special: &WhisperSpecialTokens,
    suppress_blank: bool,
    numeral_symbol_tokens: &[u32],
    dtype: DType,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let mut base = vec![0f32; vocab_size];
    for token_id in suppress_tokens {
        mask_token_value(&mut base, *token_id);
    }
    mask_token_value(&mut base, special.sot);
    mask_token_value(&mut base, special.transcribe);
    for token_id in language_token_ids {
        mask_token_value(&mut base, *token_id);
    }
    for token_id in numeral_symbol_tokens {
        mask_token_value(&mut base, *token_id);
    }
    if let Some(no_timestamps_token_id) = special.no_timestamps {
        mask_token_value(&mut base, no_timestamps_token_id);
        let timestamp_begin = no_timestamps_token_id.saturating_add(1) as usize;
        if timestamp_begin < base.len() {
            base[timestamp_begin..].fill(f32::NEG_INFINITY);
        }
    }

    let mut begin = vec![0f32; vocab_size];
    for token_id in begin_suppress_tokens {
        mask_token_value(&mut begin, *token_id);
    }
    if suppress_blank {
        mask_token_value(&mut begin, special.eot);
        if let Some(blank_token_id) = special.blank {
            mask_token_value(&mut begin, blank_token_id);
        }
    }

    let base = Tensor::from_vec(base, (vocab_size,), device)?;
    let begin = Tensor::from_vec(begin, (vocab_size,), device)?;
    let base = if dtype == DType::F32 {
        base
    } else {
        base.to_dtype(dtype)?
    };
    let begin = if dtype == DType::F32 {
        begin
    } else {
        begin.to_dtype(dtype)?
    };
    Ok((base, begin))
}

fn apply_whisper_decode_constraints(
    logits: &mut [f32],
    at_begin: bool,
    suppress_tokens: &[u32],
    begin_suppress_tokens: &[u32],
    language_token_ids: &[u32],
    special: &WhisperSpecialTokens,
    suppress_blank: bool,
    numeral_symbol_tokens: &[u32],
) {
    for token_id in suppress_tokens {
        mask_token(logits, *token_id);
    }
    if at_begin {
        for token_id in begin_suppress_tokens {
            mask_token(logits, *token_id);
        }
        if suppress_blank {
            mask_token(logits, special.eot);
            if let Some(blank_token_id) = special.blank {
                mask_token(logits, blank_token_id);
            }
        }
    }

    mask_token(logits, special.sot);
    mask_token(logits, special.transcribe);
    for token_id in language_token_ids {
        mask_token(logits, *token_id);
    }
    for token_id in numeral_symbol_tokens {
        mask_token(logits, *token_id);
    }

    if let Some(no_timestamps_token_id) = special.no_timestamps {
        // whisper.cpp / transformers text-only decode behavior.
        mask_token(logits, no_timestamps_token_id);
        let timestamp_begin = no_timestamps_token_id.saturating_add(1) as usize;
        if timestamp_begin < logits.len() {
            logits[timestamp_begin..].fill(f32::NEG_INFINITY);
        }
    }
}

fn mask_token(logits: &mut [f32], token_id: u32) {
    let idx = token_id as usize;
    if idx < logits.len() {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn mask_token_value(mask: &mut [f32], token_id: u32) {
    let idx = token_id as usize;
    if idx < mask.len() {
        mask[idx] = f32::NEG_INFINITY;
    }
}

fn logits_to_log_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(logits.len());
    logits_to_log_probs_in_place(logits, temperature, &mut out);
    out
}

fn logits_to_log_probs_in_place(logits: &[f32], temperature: f32, out: &mut Vec<f32>) {
    let inv_temperature = if temperature > 0.0 {
        1.0 / temperature
    } else {
        1.0
    };

    out.clear();
    out.resize(logits.len(), f32::NEG_INFINITY);

    let Some(logsumexp) = scaled_logsumexp(logits, inv_temperature) else {
        return;
    };

    for (idx, logit) in logits.iter().enumerate() {
        if logit.is_finite() {
            out[idx] = (*logit * inv_temperature) - logsumexp;
        }
    }
}

fn scaled_logsumexp(logits: &[f32], inv_temperature: f32) -> Option<f32> {
    let mut max_scaled = f32::NEG_INFINITY;
    for logit in logits {
        if !logit.is_finite() {
            continue;
        }
        let scaled = *logit * inv_temperature;
        if scaled > max_scaled {
            max_scaled = scaled;
        }
    }

    if !max_scaled.is_finite() {
        return None;
    }

    let mut sum_exp = 0.0f64;
    for logit in logits {
        if !logit.is_finite() {
            continue;
        }
        let scaled = *logit * inv_temperature;
        sum_exp += (scaled - max_scaled).exp() as f64;
    }

    if sum_exp <= 0.0 {
        return None;
    }

    Some(max_scaled + (sum_exp as f32).ln())
}

fn best_finite_logit(logits: &[f32], fallback_token: u32) -> (u32, f32) {
    let mut best_idx = fallback_token as usize;
    let mut best_logit = f32::NEG_INFINITY;
    for (idx, logit) in logits.iter().enumerate() {
        if *logit > best_logit {
            best_idx = idx;
            best_logit = *logit;
        }
    }
    if !best_logit.is_finite() {
        return (fallback_token, f32::NEG_INFINITY);
    }
    (best_idx as u32, best_logit)
}

fn probability_for_token_from_logits(
    logits: &[f32],
    token_id: u32,
    logsumexp: f32,
    inv_temperature: f32,
) -> Option<f32> {
    let idx = token_id as usize;
    if idx >= logits.len() {
        return None;
    }
    let logit = logits[idx];
    if !logit.is_finite() {
        return None;
    }
    Some((logit * inv_temperature - logsumexp).exp())
}

fn probability_for_token(log_probs: &[f32], token_id: u32) -> Option<f32> {
    let idx = token_id as usize;
    if idx >= log_probs.len() {
        return None;
    }
    let log_prob = log_probs[idx];
    if !log_prob.is_finite() {
        return None;
    }
    Some(log_prob.exp())
}

fn sample_token_from_log_probs<R: Rng + ?Sized>(
    log_probs: &[f32],
    temperature: f32,
    fallback_token: u32,
    rng: &mut R,
) -> (u32, f32) {
    if temperature <= 0.0 {
        let mut best_idx = fallback_token as usize;
        let mut best_logprob = f32::NEG_INFINITY;
        for (idx, logprob) in log_probs.iter().enumerate() {
            if *logprob > best_logprob {
                best_idx = idx;
                best_logprob = *logprob;
            }
        }
        if !best_logprob.is_finite() {
            return (fallback_token, f32::NEG_INFINITY);
        }
        return (best_idx as u32, best_logprob);
    }

    let mut sum = 0.0f64;
    for logprob in log_probs {
        if logprob.is_finite() {
            sum += logprob.exp() as f64;
        }
    }

    if sum <= 0.0 {
        return (fallback_token, f32::NEG_INFINITY);
    }

    let mut threshold = rng.gen_range(0.0..sum);
    for (idx, logprob) in log_probs.iter().enumerate() {
        if !logprob.is_finite() {
            continue;
        }
        threshold -= logprob.exp() as f64;
        if threshold <= 0.0 {
            return (idx as u32, *logprob);
        }
    }

    let mut best_idx = fallback_token as usize;
    let mut best_logprob = f32::NEG_INFINITY;
    for (idx, logprob) in log_probs.iter().enumerate() {
        if *logprob > best_logprob {
            best_idx = idx;
            best_logprob = *logprob;
        }
    }
    (best_idx as u32, best_logprob)
}

fn token_compression_ratio(tokens: &[u32], vocab_size: usize) -> Option<f32> {
    if tokens.is_empty() || vocab_size == 0 {
        return None;
    }

    let width = ((vocab_size as f64).log2().floor() as usize / 8).saturating_add(1);
    let mut raw = Vec::with_capacity(tokens.len() * width);
    for token in tokens {
        let value = *token as u64;
        for byte in 0..width {
            raw.push(((value >> (8 * byte)) & 0xFF) as u8);
        }
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw).ok()?;
    let compressed = encoder.finish().ok()?;
    if compressed.is_empty() {
        return None;
    }
    Some(raw.len() as f32 / compressed.len() as f32)
}

fn should_skip_as_no_speech(
    attempt: &WhisperDecodeAttempt,
    logprob_threshold: f32,
    no_speech_threshold: f32,
) -> bool {
    attempt.avg_logprob < logprob_threshold
        && attempt
            .no_speech_prob
            .map(|prob| prob > no_speech_threshold)
            .unwrap_or(false)
}

fn decode_retry_reasons(
    attempt: &WhisperDecodeAttempt,
    logprob_threshold: f32,
    compression_ratio_threshold: Option<f32>,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if attempt.repetition_loop {
        reasons.push("repetition_loop");
    }
    if !attempt.ended_with_eot {
        reasons.push("missing_eot");
    }
    if attempt.avg_logprob < logprob_threshold {
        reasons.push("low_logprob");
    }
    if let (Some(ratio), Some(threshold)) = (attempt.compression_ratio, compression_ratio_threshold)
    {
        if ratio > threshold {
            reasons.push("compression_ratio");
        }
    }
    if has_low_word_diversity(&attempt.text) {
        reasons.push("low_word_diversity");
    }
    reasons
}

fn whisper_terminal_policy_transition(
    attempt: WhisperDecodeAttempt,
    temperature: f32,
    mut best: Option<(WhisperDecodeAttempt, f32)>,
    next_temperature: Option<f32>,
    logprob_threshold: f32,
    no_speech_threshold: f32,
    compression_ratio_threshold: Option<f32>,
    expected_generation: u64,
    new_generation: u64,
) -> (
    WhisperTerminalTransition,
    Option<(WhisperDecodeAttempt, f32)>,
) {
    if should_skip_as_no_speech(&attempt, logprob_threshold, no_speech_threshold) {
        return (
            WhisperTerminalTransition::SkipNoSpeech {
                no_speech_probability: attempt.no_speech_prob,
            },
            None,
        );
    }
    let retry_reasons =
        decode_retry_reasons(&attempt, logprob_threshold, compression_ratio_threshold);
    if best
        .as_ref()
        .map(|(current, _)| is_better_attempt(&attempt, current))
        .unwrap_or(true)
    {
        best = Some((attempt.clone(), temperature));
    }
    if !retry_reasons.is_empty() {
        if let Some(next_temperature) = next_temperature {
            return (
                WhisperTerminalTransition::RetryRequired {
                    next_temperature,
                    reasons: retry_reasons,
                    expected_generation,
                    new_generation,
                },
                best,
            );
        }
    }
    let (selected, selected_temperature) = best.unwrap_or((attempt, temperature));
    (
        WhisperTerminalTransition::Accept {
            text: selected.text.trim().to_string(),
            selected_temperature,
        },
        None,
    )
}

fn should_retry_decode(
    attempt: &WhisperDecodeAttempt,
    logprob_threshold: f32,
    compression_ratio_threshold: Option<f32>,
) -> bool {
    // Fallback criteria aligned with upstream Whisper implementations:
    // repetition/unfinished decode, low avg logprob, and optional compression ratio.
    !decode_retry_reasons(attempt, logprob_threshold, compression_ratio_threshold).is_empty()
}

fn record_unique_reason(reasons: &mut Vec<&'static str>, reason: &'static str) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

fn record_unique_reasons(reasons: &mut Vec<&'static str>, new_reasons: &[&'static str]) {
    for reason in new_reasons {
        record_unique_reason(reasons, reason);
    }
}

fn whisper_attempt_diagnostics(
    temperature: f32,
    attempt: &WhisperDecodeAttempt,
    retry_reasons: &[&'static str],
    no_speech_skip: bool,
) -> serde_json::Value {
    json!({
        "temperature": temperature,
        "avg_logprob": attempt.avg_logprob,
        "no_speech_prob": attempt.no_speech_prob,
        "ended_with_eot": attempt.ended_with_eot,
        "repetition_loop": attempt.repetition_loop,
        "compression_ratio": attempt.compression_ratio,
        "decode_steps": attempt.decode_steps,
        "generated_token_count": attempt.generated_token_count,
        "sampled_token_count": attempt.sampled_token_count,
        "retry_reasons": retry_reasons,
        "no_speech": no_speech_skip,
        "profile": whisper_decode_profile_diagnostics(&attempt.profile),
    })
}

fn whisper_decode_profile_diagnostics(profile: &WhisperDecodeProfile) -> serde_json::Value {
    let measured_ms = profile.token_tensor_ms
        + profile.decoder_forward_ms
        + profile.final_linear_ms
        + profile.logits_to_host_ms
        + profile.sampling_ms;
    json!({
        "enabled": profile.enabled,
        "synchronized": profile.synchronized,
        "token_tensor_ms": profile.token_tensor_ms,
        "decoder_forward_ms": profile.decoder_forward_ms,
        "final_linear_ms": profile.final_linear_ms,
        "logits_to_host_ms": profile.logits_to_host_ms,
        "sampling_ms": profile.sampling_ms,
        "step_total_ms": profile.step_total_ms,
        "unattributed_ms": (profile.step_total_ms - measured_ms).max(0.0),
        "device_greedy_steps": profile.device_greedy_steps,
        "device_greedy_fallbacks": profile.device_greedy_fallbacks,
        "host_logits_steps": profile.host_logits_steps,
    })
}

fn has_low_word_diversity(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect();
    if words.len() < 8 {
        return false;
    }

    let unique = words.iter().collect::<HashSet<_>>().len();
    (unique as f32 / words.len() as f32) < 0.6
}

fn is_better_attempt(
    candidate: &WhisperDecodeAttempt,
    current_best: &WhisperDecodeAttempt,
) -> bool {
    if candidate.ended_with_eot != current_best.ended_with_eot {
        return candidate.ended_with_eot;
    }
    if candidate.repetition_loop != current_best.repetition_loop {
        return !candidate.repetition_loop;
    }
    if candidate.avg_logprob != current_best.avg_logprob {
        return candidate.avg_logprob > current_best.avg_logprob;
    }
    candidate.text.len() > current_best.text.len()
}

fn text_delta<'a>(previous: &str, current: &'a str) -> &'a str {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta;
    }

    let mut shared_prefix_bytes = 0usize;
    for (left, right) in previous.chars().zip(current.chars()) {
        if left != right {
            break;
        }
        shared_prefix_bytes += right.len_utf8();
    }

    &current[shared_prefix_bytes..]
}

fn find_suffix_token_repetition(ids: &[u32]) -> Option<(usize, usize)> {
    if ids.len() < REPETITION_GUARD_MIN_TOTAL_TOKENS {
        return None;
    }

    let max_span = (ids.len() / 2).min(REPETITION_GUARD_MAX_SPAN_TOKENS);
    if max_span < REPETITION_GUARD_MIN_SPAN_TOKENS {
        return None;
    }

    for span in (REPETITION_GUARD_MIN_SPAN_TOKENS..=max_span).rev() {
        let tail_start = ids.len() - span;
        let tail = &ids[tail_start..];
        let mut repeats = 1usize;

        while ids.len() >= span.saturating_mul(repeats + 1) {
            let start = ids.len() - span * (repeats + 1);
            let end = start + span;
            if &ids[start..end] == tail {
                repeats += 1;
            } else {
                break;
            }
        }

        if repeats >= 2 {
            return Some((span, repeats));
        }
    }

    None
}

fn decode_step_budget(
    prompt_len: usize,
    max_target_positions: usize,
    generation_max_length: usize,
) -> Result<usize> {
    if max_target_positions == 0 || prompt_len >= max_target_positions {
        return Err(Error::InvalidInput(format!(
            "Whisper decode prompt length {} exceeds decoder context {}",
            prompt_len, max_target_positions
        )));
    }

    let prompt_budget = max_target_positions - prompt_len;

    // Whisper decoder positional embeddings are bounded by max_target_positions.
    // Keep generated tokens within remaining context budget to avoid narrow() overflow.
    Ok(generation_max_length.max(1).min(prompt_budget))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use rand::{Rng, SeedableRng};

    use super::{
        acquire_whisper_cross_sequence_owner, adaptive_decode_budget,
        apply_whisper_decode_constraints, best_finite_logit, build_whisper_decode_mask_tensors,
        build_whisper_prompt_prefix, capped_decode_temperatures, contiguous_token_range,
        decode_retry_reasons, decode_step_budget, find_suffix_token_repetition,
        greedy_decode_step_from_masked_logits, has_low_word_diversity, logits_to_log_probs,
        logits_to_log_probs_in_place, pad_or_trim_mel_frames, probability_for_token_from_logits,
        scaled_logsumexp, tensor_to_f32_vec1, text_delta, token_contains_numeral_or_symbol,
        trimmed_audio_bounds, use_cuda_whisper_dtype_shim, whisper_cross_source_identity,
        whisper_decode_profile_diagnostics, whisper_device_diagnostics, whisper_impl_name,
        whisper_terminal_policy_transition, with_whisper_decode_step_transaction,
        WhisperDecodeAttempt, WhisperDecodeProfile, WhisperDecodeState, WhisperPreparedWindow,
        WhisperSpecialTokens, WhisperTerminalTransition,
    };
    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::backends::BackendKind;
    use crate::engine::{ModelInstanceId, RetainedStaticAttentionRuntimeV2};
    use crate::error::{Error, Result};
    use crate::kv::v2::{
        CheckpointPolicy, InferenceStateContract, KeyEncoding, PlacementPolicy, PrefixPolicy,
        StateClock, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
        StateGroupSpec, StateScope, StaticAttentionDomainSpec, StaticAttentionLayerSpec,
        CURRENT_INFERENCE_STATE_ABI,
    };
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::shared::attention::physical::PhysicalPagedKvCache;

    #[test]
    fn whisper_dtype_shim_is_cuda_only() {
        assert!(!use_cuda_whisper_dtype_shim(&candle_core::Device::Cpu));
    }

    #[test]
    fn cross_attention_source_identity_authenticates_audio_and_rate() {
        let audio = [0.0, -0.25, 0.5, 1.0];
        let identity = whisper_cross_source_identity(&audio, 16_000);
        assert_ne!(identity, [0; 32]);
        assert_eq!(
            identity,
            whisper_cross_source_identity(&audio, 16_000),
            "the same encoder source must have a stable identity"
        );
        assert_ne!(
            identity,
            whisper_cross_source_identity(&audio, 48_000),
            "sample rate is part of the encoder source"
        );
        assert_ne!(
            identity,
            whisper_cross_source_identity(&[0.0, -0.25, 0.5, 0.75], 16_000),
            "sample bits are part of the encoder source"
        );
    }

    #[test]
    fn whisper_device_diagnostics_keep_cpu_and_metal_on_local_f32() {
        for kind in [
            crate::backends::DeviceKind::Cpu,
            crate::backends::DeviceKind::Metal,
        ] {
            let diagnostics = whisper_device_diagnostics(kind, candle_core::DType::F32, false);
            let expected_kind = format!("{kind:?}");

            assert_eq!(
                diagnostics.get("kind").and_then(|value| value.as_str()),
                Some(expected_kind.as_str())
            );
            assert_eq!(
                diagnostics
                    .get("model_dtype")
                    .and_then(|value| value.as_str()),
                Some("F32")
            );
            assert_eq!(
                diagnostics
                    .get("cuda_dtype_shim")
                    .and_then(|value| value.as_bool()),
                Some(false)
            );
            assert_eq!(
                diagnostics
                    .get("whisper_impl")
                    .and_then(|value| value.as_str()),
                Some("local_whisper")
            );
        }
    }

    #[test]
    fn whisper_impl_name_marks_cuda_dtype_shim_explicitly() {
        assert_eq!(whisper_impl_name(false), "local_whisper");
        assert_eq!(whisper_impl_name(true), "local_whisper_cuda_dtype_shim");
    }

    #[test]
    fn whisper_mel_frames_pad_or_trim_to_model_window() {
        let mut short = vec![vec![1.0, 2.0]];
        pad_or_trim_mel_frames(&mut short, 2, 3);
        assert_eq!(short.len(), 3);
        assert_eq!(short[0], vec![1.0, 2.0]);
        assert_eq!(short[1], vec![0.0, 0.0]);
        assert_eq!(short[2], vec![0.0, 0.0]);

        let mut long = vec![vec![1.0], vec![2.0], vec![3.0]];
        pad_or_trim_mel_frames(&mut long, 1, 2);
        assert_eq!(long, vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn whisper_decode_profile_reports_synchronized_stage_timings() {
        let profile = WhisperDecodeProfile {
            enabled: true,
            synchronized: true,
            token_tensor_ms: 1.0,
            decoder_forward_ms: 2.0,
            final_linear_ms: 3.0,
            logits_to_host_ms: 4.0,
            sampling_ms: 5.0,
            step_total_ms: 20.0,
            device_greedy_steps: 2,
            device_greedy_fallbacks: 1,
            host_logits_steps: 3,
        };

        let diagnostics = whisper_decode_profile_diagnostics(&profile);

        assert_eq!(
            diagnostics.get("enabled").and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            diagnostics
                .get("synchronized")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            diagnostics
                .get("unattributed_ms")
                .and_then(|value| value.as_f64()),
            Some(5.0)
        );
        assert_eq!(
            diagnostics
                .get("device_greedy_steps")
                .and_then(|value| value.as_u64()),
            Some(2)
        );
    }

    #[test]
    fn device_decode_masks_match_host_constraints() {
        let special = WhisperSpecialTokens {
            sot: 0,
            sot_prev: None,
            transcribe: 6,
            eot: 1,
            blank: Some(5),
            no_timestamps: Some(8),
            no_speech: Some(4),
        };
        let suppress_tokens = vec![2];
        let begin_suppress_tokens = vec![3];
        let language_token_ids = vec![4];
        let numeral_symbol_tokens = vec![7];
        let (base_mask, begin_mask) = build_whisper_decode_mask_tensors(
            12,
            &suppress_tokens,
            &begin_suppress_tokens,
            &language_token_ids,
            &special,
            true,
            &numeral_symbol_tokens,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .expect("decode masks");
        let logits: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let logits_t = candle_core::Tensor::from_vec(
            logits.clone(),
            (logits.len(),),
            &candle_core::Device::Cpu,
        )
        .expect("logits");

        let mut host = logits.clone();
        apply_whisper_decode_constraints(
            &mut host,
            true,
            &suppress_tokens,
            &begin_suppress_tokens,
            &language_token_ids,
            &special,
            true,
            &numeral_symbol_tokens,
        );
        let device = logits_t
            .broadcast_add(&base_mask)
            .expect("base mask")
            .broadcast_add(&begin_mask)
            .expect("begin mask")
            .to_vec1::<f32>()
            .expect("masked logits");

        assert_eq!(device, host);
    }

    #[test]
    fn device_greedy_decode_matches_host_greedy_selection() {
        let special = WhisperSpecialTokens {
            sot: 0,
            sot_prev: None,
            transcribe: 6,
            eot: 1,
            blank: Some(5),
            no_timestamps: Some(8),
            no_speech: Some(4),
        };
        let mut logits = vec![0.0, -1.0, 3.0, 8.0, 2.0, 4.0, 1.0, 7.0, 0.0, 5.0, 6.0, 9.0];
        apply_whisper_decode_constraints(&mut logits, false, &[], &[], &[], &special, true, &[]);
        let inv_temperature = 1.0;
        let logsumexp = scaled_logsumexp(&logits, inv_temperature).expect("logsumexp");
        let (host_next, host_best_logit) = best_finite_logit(&logits, special.eot);
        let host_logprob = host_best_logit * inv_temperature - logsumexp;
        let host_no_speech = probability_for_token_from_logits(
            &logits,
            special.no_speech.unwrap(),
            logsumexp,
            inv_temperature,
        );
        let logits_t = candle_core::Tensor::from_vec(logits, (12,), &candle_core::Device::Cpu)
            .expect("masked logits");

        let (device_next, device_logprob, device_no_speech) =
            greedy_decode_step_from_masked_logits(&logits_t, &special, 12, inv_temperature)
                .expect("device greedy");

        assert_eq!(device_next, host_next);
        assert!((device_logprob - host_logprob).abs() < 1e-5);
        assert!(
            (device_no_speech.expect("device no speech") - host_no_speech.expect("host no speech"))
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn contiguous_token_range_detects_dense_language_ids() {
        assert_eq!(contiguous_token_range(&[10, 11, 12]), Some((10, 3)));
        assert_eq!(contiguous_token_range(&[10, 12]), None);
        assert_eq!(contiguous_token_range(&[]), None);
    }

    #[test]
    fn tensor_to_f32_vec1_accepts_f16_logits() {
        let logits =
            candle_core::Tensor::from_vec(vec![1.0f32, -2.0, 3.5], (3,), &candle_core::Device::Cpu)
                .expect("logits")
                .to_dtype(candle_core::DType::F16)
                .expect("f16 logits");

        let values = tensor_to_f32_vec1(&logits).expect("f32 host copy");
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-3);
        assert!((values[1] + 2.0).abs() < 1e-3);
        assert!((values[2] - 3.5).abs() < 1e-3);
    }

    #[test]
    fn decode_step_budget_clamps_generation_to_remaining_context() {
        let budget = decode_step_budget(4, 448, 448).expect("budget");
        assert_eq!(budget, 444);
    }

    #[test]
    fn decode_step_budget_rejects_prompt_overflow() {
        assert!(decode_step_budget(448, 448, 448).is_err());
        assert!(decode_step_budget(449, 448, 448).is_err());
    }

    #[test]
    fn detects_suffix_token_repetition() {
        let mut ids = Vec::new();
        ids.extend(1u32..=12);
        ids.extend(1u32..=12);
        let repetition = find_suffix_token_repetition(&ids);
        assert_eq!(repetition, Some((12, 2)));
    }

    #[test]
    fn ignores_short_or_non_repeating_suffixes() {
        let ids: Vec<u32> = (1..=16).collect();
        assert_eq!(find_suffix_token_repetition(&ids), None);
    }

    #[test]
    fn in_place_log_probs_match_allocating_variant() {
        let logits = vec![0.25f32, 0.75, -1.0, f32::NEG_INFINITY, 2.0];
        let expected = logits_to_log_probs(&logits, 0.7);
        let mut out = Vec::new();
        logits_to_log_probs_in_place(&logits, 0.7, &mut out);
        assert_eq!(expected.len(), out.len());
        for (left, right) in expected.iter().zip(out.iter()) {
            if left.is_finite() || right.is_finite() {
                assert!((left - right).abs() < 1e-5, "{left} != {right}");
            } else {
                assert!(!left.is_finite() && !right.is_finite());
            }
        }
    }

    #[test]
    fn decode_constraints_suppress_blank_only_at_begin() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: None,
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };
        let mut begin_logits = vec![0.0f32; 10];
        apply_whisper_decode_constraints(
            &mut begin_logits,
            true,
            &[],
            &[],
            &[],
            &special,
            true,
            &[],
        );
        assert_eq!(begin_logits[3], f32::NEG_INFINITY);
        assert_eq!(begin_logits[4], f32::NEG_INFINITY);

        let mut next_logits = vec![0.0f32; 10];
        apply_whisper_decode_constraints(
            &mut next_logits,
            false,
            &[],
            &[],
            &[],
            &special,
            true,
            &[],
        );
        assert!(next_logits[3].is_finite());
        assert!(next_logits[4].is_finite());
    }

    #[test]
    fn decode_constraints_mask_numeral_symbol_tokens() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: None,
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };
        let mut logits = vec![0.0f32; 10];
        apply_whisper_decode_constraints(&mut logits, false, &[], &[], &[], &special, false, &[5]);
        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert_eq!(logits[7], f32::NEG_INFINITY);
        assert_eq!(logits[8], f32::NEG_INFINITY);
        assert_eq!(logits[9], f32::NEG_INFINITY);
    }

    #[test]
    fn numeral_symbol_filter_detects_digits_and_symbols() {
        assert!(token_contains_numeral_or_symbol("12"));
        assert!(token_contains_numeral_or_symbol("$"));
        assert!(!token_contains_numeral_or_symbol("word"));
    }

    #[test]
    fn text_delta_uses_prefix_fast_path() {
        assert_eq!(text_delta("the quick", "the quick brown"), " brown");
    }

    #[test]
    fn text_delta_handles_midstring_rewrites() {
        assert_eq!(text_delta("hello wrld", "hello world"), "orld");
    }

    #[test]
    fn capped_decode_temperatures_limits_retry_count() {
        let temps = capped_decode_temperatures(0.0, 0.2, false, 1);
        assert_eq!(temps, vec![0.0, 0.2]);
    }

    #[test]
    fn capped_decode_temperatures_respects_no_fallback() {
        let temps = capped_decode_temperatures(0.4, 0.2, true, 8);
        assert_eq!(temps, vec![0.4]);
    }

    #[test]
    fn adaptive_decode_budget_scales_with_audio_duration() {
        let budget = adaptive_decode_budget(3.6, 448, 12.0, 32, 8);
        assert_eq!(budget, 52);
    }

    #[test]
    fn adaptive_decode_budget_respects_cap_and_minimum() {
        let budget = adaptive_decode_budget(0.4, 40, 2.0, 32, 8);
        assert_eq!(budget, 32);

        let capped = adaptive_decode_budget(30.0, 120, 12.0, 32, 8);
        assert_eq!(capped, 120);
    }

    #[test]
    fn whisper_prompt_prefix_keeps_default_controls_unchanged() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: Some(9),
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };

        let prefix = build_whisper_prompt_prefix(&special, Some(5), &[], 12, 4).expect("prefix");

        assert_eq!(prefix.ids, vec![1, 5, 2, 7]);
        assert!(!prefix.diagnostics.initial_prompt_requested);
        assert_eq!(prefix.diagnostics.initial_prompt_tokens_used, 0);
    }

    #[test]
    fn whisper_prompt_prefix_truncates_initial_prompt_tail() {
        let special = WhisperSpecialTokens {
            sot: 1,
            sot_prev: Some(9),
            transcribe: 2,
            eot: 3,
            blank: Some(4),
            no_timestamps: Some(7),
            no_speech: None,
        };

        let prefix = build_whisper_prompt_prefix(&special, Some(5), &[10, 11, 12, 13, 14], 12, 3)
            .expect("prefix");

        assert_eq!(prefix.ids, vec![9, 12, 13, 14, 1, 5, 2, 7]);
        assert!(prefix.diagnostics.initial_prompt_requested);
        assert_eq!(prefix.diagnostics.initial_prompt_token_count, 5);
        assert_eq!(prefix.diagnostics.initial_prompt_tokens_used, 3);
        assert_eq!(prefix.diagnostics.initial_prompt_tokens_truncated, 2);
        assert_eq!(prefix.diagnostics.previous_context_token_id, Some(9));
    }

    #[test]
    fn trimmed_audio_bounds_removes_leading_and_trailing_silence() {
        let sr = 16_000u32;
        let mut audio = vec![0.0f32; 8_000];
        audio.extend(vec![0.2f32; 16_000]);
        audio.extend(vec![0.0f32; 8_000]);

        let (start, end) = trimmed_audio_bounds(&audio, sr, 0.02, 0.0015, 120, 300, 120, 0.8);
        assert!(start > 0);
        assert!(end < audio.len());
        assert!(end > start);
    }

    #[test]
    fn trimmed_audio_bounds_keeps_short_clips_untouched() {
        let sr = 16_000u32;
        let audio = vec![0.0f32; 4_000];
        let (start, end) = trimmed_audio_bounds(&audio, sr, 0.02, 0.0015, 120, 300, 120, 0.8);
        assert_eq!(start, 0);
        assert_eq!(end, audio.len());
    }

    #[test]
    fn trimmed_audio_bounds_preserves_short_leading_silence() {
        let sr = 16_000u32;
        let mut audio = vec![0.0f32; 6_000];
        audio.extend(vec![0.2f32; 16_000]);
        audio.extend(vec![0.0f32; 6_000]);

        let (start, end) = trimmed_audio_bounds(&audio, sr, 0.02, 0.0015, 120, 500, 120, 0.8);
        assert_eq!(start, 0);
        assert!(end < audio.len());
    }

    #[test]
    fn low_word_diversity_flags_repetitive_output() {
        assert!(has_low_word_diversity(
            "The quick quick brown fox fox jumps jumps over the little the little"
        ));
    }

    #[test]
    fn low_word_diversity_allows_normal_transcript() {
        assert!(!has_low_word_diversity(
            "The quick brown fox jumps over the lazy dog"
        ));
    }

    #[test]
    fn decode_retry_reasons_are_structured() {
        let attempt = WhisperDecodeAttempt {
            text: "same same same same same same same same".to_string(),
            avg_logprob: -2.0,
            no_speech_prob: Some(0.1),
            ended_with_eot: false,
            repetition_loop: true,
            compression_ratio: Some(3.0),
            generated_token_count: 8,
            sampled_token_count: 9,
            decode_steps: 9,
            profile: WhisperDecodeProfile::default(),
        };

        let reasons = decode_retry_reasons(&attempt, -1.0, Some(2.4));

        assert!(reasons.contains(&"repetition_loop"));
        assert!(reasons.contains(&"missing_eot"));
        assert!(reasons.contains(&"low_logprob"));
        assert!(reasons.contains(&"compression_ratio"));
        assert!(reasons.contains(&"low_word_diversity"));
    }

    fn terminal_attempt(
        text: &str,
        avg_logprob: f32,
        no_speech_prob: Option<f32>,
        ended_with_eot: bool,
    ) -> WhisperDecodeAttempt {
        WhisperDecodeAttempt {
            text: text.into(),
            avg_logprob,
            no_speech_prob,
            ended_with_eot,
            repetition_loop: false,
            compression_ratio: Some(1.0),
            generated_token_count: 2,
            sampled_token_count: 3,
            decode_steps: 3,
            profile: WhisperDecodeProfile::default(),
        }
    }

    #[test]
    fn terminal_policy_no_speech_precedes_temperature_retry() {
        let (transition, best) = whisper_terminal_policy_transition(
            terminal_attempt("hallucination", -2.0, Some(0.9), false),
            0.0,
            None,
            Some(0.2),
            -1.0,
            0.6,
            Some(2.4),
            1,
            2,
        );
        assert_eq!(
            transition,
            WhisperTerminalTransition::SkipNoSpeech {
                no_speech_probability: Some(0.9)
            }
        );
        assert!(best.is_none());
    }

    #[test]
    fn terminal_policy_requests_configured_retry_and_retains_best_attempt() {
        let attempt = terminal_attempt("first", -2.0, Some(0.1), false);
        let (transition, best) = whisper_terminal_policy_transition(
            attempt.clone(),
            0.0,
            None,
            Some(0.2),
            -1.0,
            0.6,
            Some(2.4),
            1,
            2,
        );
        let WhisperTerminalTransition::RetryRequired {
            next_temperature,
            reasons,
            expected_generation,
            new_generation,
        } = transition
        else {
            panic!("low-quality attempt must require retry");
        };
        assert_eq!(next_temperature, 0.2);
        assert_eq!((expected_generation, new_generation), (1, 2));
        assert!(reasons.contains(&"missing_eot"));
        assert!(reasons.contains(&"low_logprob"));
        assert_eq!(best.expect("retained best").0.text, attempt.text);
    }

    #[test]
    fn terminal_policy_final_attempt_selects_better_prior_attempt() {
        let prior = terminal_attempt("better transcript", -0.2, Some(0.1), true);
        let final_attempt = terminal_attempt("worse", -2.0, Some(0.1), false);
        let (transition, best) = whisper_terminal_policy_transition(
            final_attempt,
            0.2,
            Some((prior, 0.0)),
            None,
            -1.0,
            0.6,
            Some(2.4),
            1,
            2,
        );
        assert_eq!(
            transition,
            WhisperTerminalTransition::Accept {
                text: "better transcript".into(),
                selected_temperature: 0.0,
            }
        );
        assert!(best.is_none());
    }

    fn early_failure_cross_runtime(
        model: ModelInstanceId,
    ) -> Arc<RetainedStaticAttentionRuntimeV2> {
        let static_domain = StateDomainId::new(1);
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::StaticAttention(
                StaticAttentionDomainSpec {
                    header: StateDomainHeader {
                        id: static_domain,
                        scope: StateScope::Retained,
                        clock: StateClock::EncoderTokens,
                        placement: PlacementPolicy::BackendLocal,
                        prefix: PrefixPolicy::Disabled,
                        checkpoint: CheckpointPolicy::Transactional,
                    },
                    layers: vec![StaticAttentionLayerSpec {
                        model_layer: 0,
                        query_heads: 1,
                        kv_heads: 1,
                        key_head_dim: 2,
                        value_head_dim: 2,
                        key_encoding: KeyEncoding::Raw,
                    }],
                    max_memory_tokens: 2,
                    accepted_dtypes: vec![StateDType::F32],
                },
            )],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![static_domain],
                prefix_shareable: false,
            }],
        };
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: Some(StateDType::F32),
                },
            )
            .expect("test static state plan"),
        );
        Arc::new(
            RetainedStaticAttentionRuntimeV2::new(
                model,
                1,
                &contract,
                plan,
                static_domain,
                1,
                Device::Cpu,
            )
            .expect("test static runtime"),
        )
    }

    fn assert_cross_sequence_released_for_reuse(runtime: &RetainedStaticAttentionRuntimeV2) {
        let replacement = runtime
            .register_sequence()
            .expect("failed prefill entry must release capacity for immediate reuse");
        runtime
            .release_sequence(replacement)
            .expect("replacement sequence release");
    }

    #[test]
    fn managed_prefill_early_failures_release_cross_sequence_for_reuse() {
        let prepared = WhisperPreparedWindow::for_test(2, 1, 2).expect("prepared window");

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(991));
        let sequence = runtime.register_sequence().expect("validation sequence");
        let error = acquire_whisper_cross_sequence_owner(
            super::WhisperCrossSequenceOwner::new(runtime.clone(), sequence),
            &prepared,
            2,
            2,
            1,
            0,
            || Ok(1),
        )
        .err()
        .expect("foreign prepared identity must fail");
        assert!(error.to_string().contains("another model or geometry"));
        assert_cross_sequence_released_for_reuse(runtime.as_ref());

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(992));
        let sequence = runtime.register_sequence().expect("state-id sequence");
        let error = acquire_whisper_cross_sequence_owner(
            super::WhisperCrossSequenceOwner::new(runtime.clone(), sequence),
            &prepared,
            1,
            2,
            1,
            0,
            || Err(Error::InferenceError("forced state-id failure".into())),
        )
        .err()
        .expect("state identity allocation must fail deterministically");
        assert!(error.to_string().contains("forced state-id failure"));
        assert_cross_sequence_released_for_reuse(runtime.as_ref());

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(993));
        let sequence = runtime.register_sequence().expect("self-context sequence");
        let error = acquire_whisper_cross_sequence_owner(
            super::WhisperCrossSequenceOwner::new(runtime.clone(), sequence),
            &prepared,
            1,
            2,
            1,
            1,
            || Ok(1),
        )
        .err()
        .expect("nonempty self context must fail");
        assert!(error
            .to_string()
            .contains("requires empty self and cross state"));
        assert_cross_sequence_released_for_reuse(runtime.as_ref());

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(994));
        let sequence = runtime.register_sequence().expect("cross-read sequence");
        runtime
            .install(sequence, prepared.source_identity, prepared.layers.clone())
            .expect("preinstall cross memory");
        let error = acquire_whisper_cross_sequence_owner(
            super::WhisperCrossSequenceOwner::new(runtime.clone(), sequence),
            &prepared,
            1,
            2,
            1,
            0,
            || Ok(1),
        )
        .err()
        .expect("nonempty cross read must fail");
        assert!(error
            .to_string()
            .contains("requires empty self and cross state"));
        assert_cross_sequence_released_for_reuse(runtime.as_ref());

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(995));
        let sequence = runtime
            .register_sequence()
            .expect("install-failure sequence");
        let incompatible =
            WhisperPreparedWindow::for_test(2, 1, 3).expect("incompatible prepared window");
        let (owner, _) = acquire_whisper_cross_sequence_owner(
            super::WhisperCrossSequenceOwner::new(runtime.clone(), sequence),
            &incompatible,
            1,
            2,
            1,
            0,
            || Ok(1),
        )
        .expect("entry validation before forced install failure");
        let error = owner
            .runtime
            .install(
                owner.sequence(),
                incompatible.source_identity,
                incompatible.layers.clone(),
            )
            .expect_err("incompatible layer width must fail installation");
        assert!(!error.to_string().is_empty());
        drop(owner);
        assert_cross_sequence_released_for_reuse(runtime.as_ref());

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(996));
        let sequence = runtime
            .register_sequence()
            .expect("post-install failure sequence");
        let (owner, _) = acquire_whisper_cross_sequence_owner(
            super::WhisperCrossSequenceOwner::new(runtime.clone(), sequence),
            &prepared,
            1,
            2,
            1,
            0,
            || Ok(1),
        )
        .expect("post-install entry");
        owner
            .runtime
            .install(
                owner.sequence(),
                prepared.source_identity,
                prepared.layers.clone(),
            )
            .expect("install before simulated prompt initialization failure");
        drop(owner);
        assert_cross_sequence_released_for_reuse(runtime.as_ref());

        let runtime = early_failure_cross_runtime(ModelInstanceId::new(997));
        let sequence = runtime
            .register_sequence()
            .expect("foreign-model rejection sequence");
        let error =
            crate::models::registry::reject_foreign_whisper_prefill(runtime.clone(), sequence)
                .err()
                .expect("foreign model must reject Whisper prefill");
        assert!(error.to_string().contains("another ASR model"));
        assert_cross_sequence_released_for_reuse(runtime.as_ref());
    }

    fn transactional_test_state() -> WhisperDecodeState {
        let state_id = super::next_whisper_decode_state_id().expect("test state identity");
        let model = ModelInstanceId::new(90 + state_id);
        let arena_id = KvArenaId {
            model_instance: model,
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 1,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .expect("test KV arena"),
        );
        let self_kv = PhysicalPagedKvCache::new(
            arena,
            vec![binding],
            vec![CacheBlockRef {
                arena: arena_id,
                group,
                index: 0,
                slot_generation: 1,
            }],
            0,
        )
        .expect("test physical cache");

        let static_domain = StateDomainId::new(1);
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::StaticAttention(
                StaticAttentionDomainSpec {
                    header: StateDomainHeader {
                        id: static_domain,
                        scope: StateScope::Retained,
                        clock: StateClock::EncoderTokens,
                        placement: PlacementPolicy::BackendLocal,
                        prefix: PrefixPolicy::Disabled,
                        checkpoint: CheckpointPolicy::Transactional,
                    },
                    layers: vec![StaticAttentionLayerSpec {
                        model_layer: 0,
                        query_heads: 1,
                        kv_heads: 1,
                        key_head_dim: 2,
                        value_head_dim: 2,
                        key_encoding: KeyEncoding::Raw,
                    }],
                    max_memory_tokens: 2,
                    accepted_dtypes: vec![StateDType::F32],
                },
            )],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![static_domain],
                prefix_shareable: false,
            }],
        };
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: Some(StateDType::F32),
                },
            )
            .expect("test static state plan"),
        );
        let cross_runtime = Arc::new(
            RetainedStaticAttentionRuntimeV2::new(
                model,
                1,
                &contract,
                plan,
                static_domain,
                1,
                Device::Cpu,
            )
            .expect("test static runtime"),
        );
        let cross_sequence = cross_runtime
            .register_sequence()
            .expect("test static sequence");
        WhisperDecodeState {
            state_id,
            next_quantum_nonce: 1,
            active_quantum: None,
            current_managed_generation: 1,
            managed_completions_drained: true,
            self_kv,
            cross_runtime,
            cross_sequence: Some(cross_sequence),
            prompt: vec![1],
            prefill_progress: 1,
            pending_logits: Some(
                Tensor::from_vec(vec![0.25_f32, 0.75], (2,), &Device::Cpu).expect("test logits"),
            ),
            generated_tokens: vec![],
            assembled: String::new(),
            sum_logprobs: 0.0,
            sampled_token_count: 0,
            no_speech_prob: None,
            ended_with_eot: false,
            repetition_loop: false,
            decode_steps: 0,
            best_attempt: None,
            pending_retry: None,
            temperature: 1.0,
            attempt_generation: 1,
            max_steps: 3,
            finished: false,
            rng: rand::rngs::StdRng::seed_from_u64(7),
        }
    }

    fn append_test_decode_token(state: &mut WhisperDecodeState, token: u32) -> Result<()> {
        let mut prepared = state
            .self_kv
            .prepare_append(state.self_kv.context_len(), 1)?;
        let qkv = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
        state
            .self_kv
            .write_and_attend(0, &mut prepared, &qkv, &qkv, &qkv, 1.0)?;
        state.self_kv.commit_prepared(prepared)?;
        state.managed_completions_drained = false;
        state.pending_logits = None;
        state.generated_tokens.push(token);
        state.assembled = format!("token-{token}");
        Ok(())
    }

    fn replacement_test_cache(
        state: &WhisperDecodeState,
        context_len: usize,
    ) -> PhysicalPagedKvCache {
        PhysicalPagedKvCache::new(
            state.self_kv.arena().clone(),
            vec![KvLayerBinding {
                model_layer: 0,
                physical_layer: 0,
            }],
            state.self_kv.blocks.clone(),
            context_len,
        )
        .expect("replacement physical cache")
    }

    fn generation_test_cache(
        state: &WhisperDecodeState,
        _managed_generation: u64,
        context_len: usize,
    ) -> PhysicalPagedKvCache {
        let arena_id = state.self_kv.arena().id();
        let group = state.self_kv.arena().config().group;
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 1,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .expect("retry-generation arena"),
        );
        PhysicalPagedKvCache::new(
            arena,
            vec![binding],
            vec![CacheBlockRef {
                arena: arena_id,
                group,
                index: 0,
                slot_generation: 1,
            }],
            context_len,
        )
        .expect("retry-generation cache")
    }

    fn install_test_cross_memory(state: &WhisperDecodeState) {
        let sequence = state.cross_sequence.expect("test cross sequence");
        state
            .cross_runtime
            .install(
                sequence,
                [7; 32],
                vec![crate::backends::state::StaticAttentionLayerValue {
                    model_layer: 0,
                    keys: Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)
                        .expect("test cross keys"),
                    values: Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)
                        .expect("test cross values"),
                }],
            )
            .expect("install test cross memory");
    }

    #[test]
    fn post_append_text_decode_failure_rolls_back_for_exact_retry() {
        let mut state = transactional_test_state();
        let original_logits = state
            .pending_logits
            .as_ref()
            .expect("initial logits")
            .to_vec1::<f32>()
            .expect("host logits");
        let mut expected_rng = state.rng.clone();
        let expected_sample = expected_rng.gen::<u32>();

        let error = with_whisper_decode_step_transaction(&mut state, |state| {
            let sampled = state.rng.gen::<u32>();
            append_test_decode_token(state, sampled)?;
            state.finished = true;
            Err::<(), _>(Error::InferenceError(
                "forced post-append text decode failure".into(),
            ))
        })
        .expect_err("forced text decode failure");
        assert!(error
            .to_string()
            .contains("forced post-append text decode failure"));
        assert_eq!(state.self_kv.context_len(), 0);
        assert!(state.generated_tokens.is_empty());
        assert!(state.assembled.is_empty());
        assert!(!state.finished);
        assert_eq!(
            state
                .pending_logits
                .as_ref()
                .expect("restored logits")
                .to_vec1::<f32>()
                .expect("restored host logits"),
            original_logits
        );

        let retried_sample = with_whisper_decode_step_transaction(&mut state, |state| {
            let sampled = state.rng.gen::<u32>();
            append_test_decode_token(state, sampled)?;
            Ok(sampled)
        })
        .expect("exact retry");
        assert_eq!(retried_sample, expected_sample);
        assert_eq!(state.self_kv.context_len(), 1);
        assert_eq!(state.generated_tokens, vec![expected_sample]);
        assert_eq!(state.assembled, format!("token-{expected_sample}"));
    }

    #[test]
    fn managed_quantum_rollback_restores_reservation_semantics_and_static_identity() {
        let mut state = transactional_test_state();
        let static_sequence = state.cross_sequence;
        let original_logits = state
            .pending_logits
            .as_ref()
            .expect("initial logits")
            .to_vec1::<f32>()
            .expect("initial host logits");
        let replacement = replacement_test_cache(&state, 0);
        let mut checkpoint = state
            .begin_managed_quantum(replacement)
            .expect("begin managed quantum");

        append_test_decode_token(&mut state, 41).expect("append inside quantum");
        state.prefill_progress = 0;
        state.temperature = 0.7;
        state.attempt_generation = 9;
        state.max_steps = 1;
        state.finished = true;
        let _ = state.rng.gen::<u32>();
        state
            .rollback_managed_quantum(&mut checkpoint)
            .expect("rollback managed quantum");

        assert_eq!(state.self_kv.context_len(), 0);
        assert_eq!(state.cross_sequence, static_sequence);
        assert_eq!(state.prefill_progress, 1);
        assert!(state.generated_tokens.is_empty());
        assert!(state.assembled.is_empty());
        assert_eq!(state.temperature, 1.0);
        assert_eq!(state.attempt_generation, 1);
        assert_eq!(state.max_steps, 3);
        assert!(!state.finished);
        assert_eq!(
            state
                .pending_logits
                .as_ref()
                .expect("restored logits")
                .to_vec1::<f32>()
                .expect("restored host logits"),
            original_logits
        );
    }

    #[test]
    fn managed_quantum_commit_exposes_authenticated_completions_once() {
        let mut state = transactional_test_state();
        let static_sequence = state.cross_sequence;
        let replacement = replacement_test_cache(&state, 0);
        let mut checkpoint = state
            .begin_managed_quantum(replacement)
            .expect("begin managed quantum");
        append_test_decode_token(&mut state, 17).expect("append inside quantum");
        let completions = state.take_managed_write_completions();
        assert_eq!(completions.len(), 1);
        assert!(state.take_managed_write_completions().is_empty());
        state
            .commit_managed_quantum(&mut checkpoint)
            .expect("commit managed quantum");
        assert_eq!(state.self_kv.context_len(), 1);
        assert_eq!(state.cross_sequence, static_sequence);

        let terminal = replacement_test_cache(&state, 1);
        let mut checkpoint = state
            .begin_managed_quantum(terminal)
            .expect("begin zero-append terminal quantum");
        state.finished = true;
        assert!(state.take_managed_write_completions().is_empty());
        state
            .commit_managed_quantum(&mut checkpoint)
            .expect("commit zero-append terminal quantum");
        assert_eq!(state.self_kv.context_len(), 1);
        assert_eq!(state.cross_sequence, static_sequence);
    }

    #[test]
    fn managed_quantum_requires_exact_continuation_position() {
        let mut state = transactional_test_state();
        let replacement = replacement_test_cache(&state, 1);
        let error = match state.begin_managed_quantum(replacement) {
            Ok(_) => panic!("future reservation must be rejected"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("reservation starts at 1, but decode state is at 0"));
        assert_eq!(state.self_kv.context_len(), 0);
    }

    #[test]
    fn managed_quantum_rejects_nested_and_undrained_replacement() {
        let mut state = transactional_test_state();
        let replacement = replacement_test_cache(&state, 0);
        let mut checkpoint = state
            .begin_managed_quantum(replacement)
            .expect("begin managed quantum");
        let nested = replacement_test_cache(&state, 0);
        let error = match state.begin_managed_quantum(nested) {
            Ok(_) => panic!("nested managed quantum must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("already active"));

        append_test_decode_token(&mut state, 23).expect("append inside quantum");
        state
            .commit_managed_quantum(&mut checkpoint)
            .expect("commit managed quantum");
        let undrained = replacement_test_cache(&state, 1);
        let error = match state.begin_managed_quantum(undrained) {
            Ok(_) => panic!("undrained completions must reject the next quantum"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("must be drained"));
        assert_eq!(state.take_managed_write_completions().len(), 1);
        let drained = replacement_test_cache(&state, 1);
        let mut checkpoint = state
            .begin_managed_quantum(drained)
            .expect("drained continuation");
        state
            .rollback_managed_quantum(&mut checkpoint)
            .expect("rollback drained continuation");
    }

    #[test]
    fn managed_quantum_rejects_wrong_row_checkpoint_without_consuming_either() {
        let mut first = transactional_test_state();
        let mut second = transactional_test_state();
        let first_replacement = replacement_test_cache(&first, 0);
        let mut first_checkpoint = first
            .begin_managed_quantum(first_replacement)
            .expect("first quantum");
        let second_replacement = replacement_test_cache(&second, 0);
        let mut second_checkpoint = second
            .begin_managed_quantum(second_replacement)
            .expect("second quantum");

        let error = second
            .commit_managed_quantum(&mut first_checkpoint)
            .expect_err("foreign checkpoint must be rejected");
        assert!(error
            .to_string()
            .contains("foreign, stale, or out of order"));
        first
            .rollback_managed_quantum(&mut first_checkpoint)
            .expect("foreign rejection must not consume first checkpoint");
        second
            .rollback_managed_quantum(&mut second_checkpoint)
            .expect("foreign rejection must not clear second active quantum");
    }

    #[test]
    fn managed_quantum_checkpoint_is_single_use_and_out_of_order_safe() {
        let mut state = transactional_test_state();
        let first_replacement = replacement_test_cache(&state, 0);
        let mut first = state
            .begin_managed_quantum(first_replacement)
            .expect("first quantum");
        state
            .commit_managed_quantum(&mut first)
            .expect("commit first quantum");
        let stale = state
            .commit_managed_quantum(&mut first)
            .expect_err("committed checkpoint must be stale");
        assert!(stale
            .to_string()
            .contains("foreign, stale, or out of order"));

        let second_replacement = replacement_test_cache(&state, 0);
        let mut second = state
            .begin_managed_quantum(second_replacement)
            .expect("second quantum");
        let out_of_order = state
            .rollback_managed_quantum(&mut first)
            .expect_err("older checkpoint must not roll back a newer quantum");
        assert!(out_of_order
            .to_string()
            .contains("foreign, stale, or out of order"));
        state
            .rollback_managed_quantum(&mut second)
            .expect("newer active checkpoint remains valid");
    }

    #[test]
    fn managed_generation_resets_attempt_and_preserves_static_cross_identity() {
        let mut state = transactional_test_state();
        install_test_cross_memory(&state);
        let sequence = state.cross_sequence.expect("cross sequence");
        let cross_before = state
            .cross_runtime
            .read(sequence)
            .expect("read cross memory")
            .expect("installed cross memory");
        state.generated_tokens = vec![9, 10];
        state.assembled = "bad attempt".into();
        state.sum_logprobs = -8.0;
        state.sampled_token_count = 2;
        state.finished = true;
        state.best_attempt = Some((terminal_attempt("best so far", -0.5, Some(0.1), true), 0.0));
        state.pending_retry = Some(super::WhisperPendingRetry {
            next_temperature: 0.2,
            expected_generation: 1,
            new_generation: 2,
            next_attempt_generation: 2,
        });
        let replacement = generation_test_cache(&state, 2, 0);
        let mut checkpoint = state
            .begin_managed_generation(replacement, 1, 2)
            .expect("begin retry generation");

        assert_eq!(state.self_kv.arena().id().generation, 1);
        assert_eq!(state.current_managed_generation, 2);
        assert_eq!(state.self_kv.context_len(), 0);
        assert_eq!(state.prefill_progress, 0);
        assert!(state.pending_logits.is_none());
        assert!(state.generated_tokens.is_empty());
        assert!(state.assembled.is_empty());
        assert_eq!(state.temperature, 0.2);
        assert_eq!(state.attempt_generation, 2);
        assert!(!state.finished);
        assert!(state.pending_retry.is_none());
        assert_eq!(
            state.best_attempt.as_ref().expect("preserved best").0.text,
            "best so far"
        );
        assert_eq!(
            state
                .cross_runtime
                .read(sequence)
                .expect("read preserved cross")
                .expect("preserved cross memory"),
            cross_before
        );

        state
            .rollback_managed_quantum(&mut checkpoint)
            .expect("rollback retry generation");
        assert_eq!(state.self_kv.arena().id().generation, 1);
        assert_eq!(state.current_managed_generation, 1);
        assert_eq!(state.generated_tokens, vec![9, 10]);
        assert_eq!(state.assembled, "bad attempt");
        assert!(state.pending_retry.is_some());
        assert_eq!(state.cross_sequence, Some(sequence));
    }

    #[test]
    fn managed_generation_rejects_no_pending_and_stale_generations() {
        let mut state = transactional_test_state();
        install_test_cross_memory(&state);
        let no_pending = generation_test_cache(&state, 2, 0);
        let error = match state.begin_managed_generation(no_pending, 1, 2) {
            Ok(_) => panic!("retry generation requires pending policy transition"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("no pending"));

        state.pending_retry = Some(super::WhisperPendingRetry {
            next_temperature: 0.2,
            expected_generation: 1,
            new_generation: 2,
            next_attempt_generation: 2,
        });
        let stale = generation_test_cache(&state, 2, 0);
        let error = match state.begin_managed_generation(stale, 0, 2) {
            Ok(_) => panic!("stale expected generation must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("stale or out of order"));
        assert_eq!(state.self_kv.arena().id().generation, 1);
        assert_eq!(state.current_managed_generation, 1);
        assert!(state.pending_retry.is_some());
    }

    #[test]
    fn managed_generation_rejects_wrong_authority_and_nonzero_context() {
        let mut state = transactional_test_state();
        install_test_cross_memory(&state);
        state.pending_retry = Some(super::WhisperPendingRetry {
            next_temperature: 0.2,
            expected_generation: 1,
            new_generation: 2,
            next_attempt_generation: 2,
        });
        let nonzero = generation_test_cache(&state, 2, 1);
        let error = match state.begin_managed_generation(nonzero, 1, 2) {
            Ok(_) => panic!("retry generation must start at zero"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("context zero"));

        let mut foreign = transactional_test_state();
        install_test_cross_memory(&foreign);
        foreign.pending_retry = state.pending_retry;
        let wrong_authority = generation_test_cache(&foreign, 2, 0);
        let error = match state.begin_managed_generation(wrong_authority, 1, 2) {
            Ok(_) => panic!("foreign retry cache must be rejected"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("cannot switch session KV authority"));
        assert_eq!(state.self_kv.arena().id().generation, 1);
        assert_eq!(state.current_managed_generation, 1);
    }
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;

    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    out
}

fn checked_product_u64(values: &[usize], label: &str) -> Result<u64> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(
                u64::try_from(*value)
                    .map_err(|_| Error::Overloaded(format!("{label} exceeds u64")))?,
            )
            .ok_or_else(|| Error::Overloaded(format!("{label} overflow")))
    })
}

fn language_name_to_code(language: &str) -> Option<&'static str> {
    WHISPER_LANGUAGES
        .iter()
        .find(|(_code, name)| *name == language)
        .map(|(code, _name)| *code)
}

fn language_alias_to_code(language: &str) -> Option<&'static str> {
    match language {
        "burmese" => Some("my"),
        "valencian" => Some("ca"),
        "flemish" => Some("nl"),
        "haitian" => Some("ht"),
        "letzeburgesch" => Some("lb"),
        "pushto" => Some("ps"),
        "panjabi" => Some("pa"),
        "moldavian" | "moldovan" => Some("ro"),
        "sinhalese" => Some("si"),
        "castilian" => Some("es"),
        "mandarin" => Some("zh"),
        _ => None,
    }
}

// Mirrors Whisper multilingual language table from upstream implementations.
const WHISPER_LANGUAGES: [(&str, &str); 100] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];
