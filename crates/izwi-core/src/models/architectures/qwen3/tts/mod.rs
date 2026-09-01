//! Native Qwen3-TTS model loader and inference.
//!
//! This module provides native Rust implementation for Qwen3-TTS models,
//! supporting both CustomVoice (preset speakers) and voice cloning modes.

mod config;
mod predictor;
mod rope;
mod speech_tokenizer;
mod talker;
mod tokenizer;

pub use config::Qwen3TtsConfig;
pub use predictor::{
    code_predictor_physical_context_tokens, CodePredictor, CodePredictorPhysicalCache,
    CODE_PREDICTOR_PHYSICAL_PREFILL_TOKENS,
};
pub use speech_tokenizer::SpeechTokenizerDecoder;
pub use talker::{TalkerModel, TalkerPhysicalCache};
pub use tokenizer::{SpeakerReference, TtsSpecialTokens, TtsTokenizer};

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::engine::{StageDescriptor, StageWorkSelector};
use crate::error::{Error, Result};
use crate::kv::v2::{
    minimum_physical_bytes_for_capacity, stage_graph_fingerprint, BoundedShape,
    CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract, InvocationLeaseScope,
    InvocationStageWorkspace, InvocationStateCapacity, InvocationWorkspaceDomain,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy, PositionSemantics,
    PrefixPolicy, RetainedStateCapability, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::qwen3::core::{
    qwen3_decoder_cache_domain, Qwen3DecoderCacheGeometry,
};
use crate::models::shared::attention::paged::{default_kv_page_size, KvCacheQuantization};
use crate::models::shared::sampling::{
    bounded_device_sampling_candidates, device_candidates_cover_top_p, sample_device_candidates,
};
use crate::models::shared::state::exact_stage_scratch_domain;

const NEWLINE_TOKEN_ID: u32 = 198;
const ENV_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM: &str = "IZWI_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM";
const MIN_QWEN_TTS_TOKENS_BEFORE_EOS: usize = 8;
const MAX_VOICE_CLONE_REFERENCE_FRAMES: usize = 320;
const QWEN3_TTS_MODEL_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);

fn validate_qwen3_tts_tensor_state_shapes(
    memory: &Tensor,
    pad: &Tensor,
    hidden: Option<&Tensor>,
    logits: Option<&Tensor>,
) -> Result<()> {
    let (memory_batch, memory_sequence, memory_width) = memory.dims3()?;
    let (pad_batch, pad_sequence, pad_width) = pad.dims3()?;
    if memory_batch != 1
        || memory_sequence == 0
        || pad_batch != 1
        || pad_sequence != 1
        || pad_width != memory_width
    {
        return Err(Error::InferenceError(
            "Qwen3-TTS retained tensor state has a non-canonical memory/pad shape".into(),
        ));
    }
    if let Some(hidden) = hidden {
        let (batch, sequence, width) = hidden.dims3()?;
        if batch != 1 || sequence != 1 || width != memory_width {
            return Err(Error::InferenceError(
                "Qwen3-TTS retained hidden state has a non-canonical shape".into(),
            ));
        }
    }
    if let Some(logits) = logits {
        let (batch, sequence, vocabulary) = logits.dims3()?;
        if batch != 1 || sequence != 1 || vocabulary == 0 {
            return Err(Error::InferenceError(
                "Qwen3-TTS retained logits have a non-canonical shape".into(),
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn transformer_workspace_upper_bound_bytes(
    query_tokens: usize,
    attention_span: usize,
    hidden: usize,
    intermediate: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocabulary: usize,
    element_bytes: usize,
) -> Result<u64> {
    let value = |input: usize, label: &str| {
        u64::try_from(input)
            .map_err(|_| Error::Overloaded(format!("Qwen3-TTS {label} exceeds u64")))
    };
    let query_tokens = value(query_tokens.max(1), "query span")?;
    let attention_span = value(attention_span.max(1), "attention span")?;
    let hidden = value(hidden, "hidden width")?;
    let intermediate = value(intermediate, "intermediate width")?;
    let query_heads = value(query_heads, "query head count")?;
    let kv_heads = value(kv_heads, "KV head count")?;
    let head_dim = value(head_dim, "head width")?;
    let vocabulary = value(vocabulary, "vocabulary width")?;
    let multiply = |left: u64, right: u64| {
        left.checked_mul(right)
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS transformer workspace overflow".into()))
    };
    let hidden_buffers = multiply(multiply(query_tokens, hidden)?, 6)?;
    let ffn_buffers = multiply(multiply(query_tokens, intermediate)?, 3)?;
    let qkv_heads = query_heads
        .checked_add(kv_heads.checked_mul(2).ok_or_else(|| {
            Error::Overloaded("Qwen3-TTS transformer head geometry overflow".into())
        })?)
        .ok_or_else(|| Error::Overloaded("Qwen3-TTS transformer head geometry overflow".into()))?;
    let qkv = multiply(multiply(multiply(query_tokens, qkv_heads)?, head_dim)?, 2)?;
    let attention = multiply(
        multiply(multiply(query_heads, query_tokens)?, attention_span)?,
        2,
    )?;
    let elements = hidden_buffers
        .checked_add(ffn_buffers)
        .and_then(|total| total.checked_add(qkv))
        .and_then(|total| total.checked_add(attention))
        .and_then(|total| total.checked_add(vocabulary))
        .ok_or_else(|| Error::Overloaded("Qwen3-TTS transformer workspace overflow".into()))?;
    multiply(
        elements,
        value(element_bytes.max(std::mem::size_of::<f32>()), "dtype width")?,
    )
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3TtsPhysicalStateSpec {
    pub(crate) retained: InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) predictor_contract: InferenceStateContract,
}

/// Runtime generation settings for semantic token sampling.
#[derive(Debug, Clone)]
pub struct TtsGenerationParams {
    /// Semantic token temperature. <= 0 means greedy.
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k sampling cutoff. 0 means disabled.
    pub top_k: usize,
    /// Repetition penalty for previously sampled semantic tokens.
    pub repetition_penalty: f32,
    /// Maximum generated codec frames. `0` means auto (model maximum).
    pub max_frames: usize,
}

impl Default for TtsGenerationParams {
    fn default() -> Self {
        // Mirrors the official generation_config defaults.
        Self {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
            max_frames: crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES,
        }
    }
}

/// Runtime configuration for progressive TTS audio emission.
#[derive(Debug, Clone, Copy)]
pub struct TtsStreamingConfig {
    /// Minimum codec frames before emitting first audio chunk.
    pub min_frames_before_stream: usize,
    /// Minimum newly generated codec frames before decoding again.
    pub decode_interval_frames: usize,
    /// Keep a small decode lookahead to reduce boundary artifacts.
    pub decode_lookahead_frames: usize,
}

impl Default for TtsStreamingConfig {
    fn default() -> Self {
        Self {
            min_frames_before_stream: 6,
            decode_interval_frames: 4,
            decode_lookahead_frames: 2,
        }
    }
}

impl TtsStreamingConfig {
    /// Decode audio only at completion.
    ///
    /// This avoids repeatedly decoding the full codec timeline for non-streaming
    /// generation paths, which materially improves long-form performance.
    pub fn final_only() -> Self {
        Self {
            min_frames_before_stream: usize::MAX,
            decode_interval_frames: usize::MAX,
            decode_lookahead_frames: 0,
        }
    }
}

/// Device-side TTS prompt material prepared once before scheduler chunking.
///
/// The embedding tensor is immutable across resumptions; only its exact
/// `[span_start, span_end)` view is executed by each prefill quantum.
#[derive(Clone)]
pub struct PreparedTtsDecodePrefill {
    prefill_embeds: Tensor,
    trailing_text_hidden: Tensor,
    retained_sequence_memory: Tensor,
    tts_pad_embed: Tensor,
    params: TtsGenerationParams,
}

impl PreparedTtsDecodePrefill {
    pub fn prefill_tokens(&self) -> Result<usize> {
        self.prefill_embeds.dim(1).map_err(Error::from)
    }

    pub fn max_frames(&self) -> usize {
        self.params.max_frames.max(1)
    }
}

/// In-progress prepared-embedding prefill over scheduler-owned talker pages.
pub struct PhysicalTtsPrefillState {
    prepared: PreparedTtsDecodePrefill,
    talker_cache: TalkerPhysicalCache,
    stream_config: TtsStreamingConfig,
    progress: usize,
    total_tokens: usize,
    last_hidden: Option<Tensor>,
    last_logits: Option<Tensor>,
    tensor_sequence: Option<PhysicalStateSequenceId>,
}

pub(crate) struct PhysicalTtsPrefillManagedCheckpoint {
    talker_cache: TalkerPhysicalCache,
    prepared: PreparedTtsDecodePrefill,
    stream_config: TtsStreamingConfig,
    progress: usize,
    total_tokens: usize,
    last_hidden: Option<Tensor>,
    last_logits: Option<Tensor>,
    tensor_sequence: Option<PhysicalStateSequenceId>,
}

impl PhysicalTtsPrefillState {
    pub fn prefill_progress(&self) -> usize {
        self.progress
    }

    pub fn prefill_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn is_complete(&self) -> bool {
        self.progress == self.total_tokens
            && self.last_hidden.is_some()
            && self.last_logits.is_some()
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        self.talker_cache.take_completed_writes()
    }

    pub(crate) fn install_retained_talker_reservation(
        &mut self,
        cache: TalkerPhysicalCache,
    ) -> Result<()> {
        if self.talker_cache.arena().id() != cache.arena().id()
            || self.talker_cache.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Qwen3-TTS prefill cannot switch retained talker authority".into(),
            ));
        }
        if cache.context_len() != self.progress {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS prefill reservation starts at {}, but prepared progress is {}",
                cache.context_len(),
                self.progress
            )));
        }
        self.talker_cache = cache;
        Ok(())
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsPrefillManagedCheckpoint> {
        if self.talker_cache.arena().id() != cache.arena().id()
            || self.talker_cache.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Qwen3-TTS prefill cannot switch retained talker authority".into(),
            ));
        }
        if cache.context_len() != self.progress {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS prefill reservation starts at {}, but prepared progress is {}",
                cache.context_len(),
                self.progress
            )));
        }
        let checkpoint = PhysicalTtsPrefillManagedCheckpoint {
            talker_cache: std::mem::replace(&mut self.talker_cache, cache),
            prepared: self.prepared.clone(),
            stream_config: self.stream_config,
            progress: self.progress,
            total_tokens: self.total_tokens,
            last_hidden: self.last_hidden.clone(),
            last_logits: self.last_logits.clone(),
            tensor_sequence: self.tensor_sequence,
        };
        Ok(checkpoint)
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: PhysicalTtsPrefillManagedCheckpoint,
    ) {
        *self = checkpoint.into_state();
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(Error::InferenceError(
                "Qwen3-TTS prefill tensor-state sequence identity changed".into(),
            ));
        }
        self.tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn stage_tensor_state(
        &self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        self.tensor_sequence.ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS prefill has no tensor sequence".into())
        })?;
        validate_qwen3_tts_tensor_state_shapes(
            &self.prepared.retained_sequence_memory,
            &self.prepared.tts_pad_embed,
            self.last_hidden.as_ref(),
            self.last_logits.as_ref(),
        )?;
        let transaction = PhysicalStateTransactionId::new(transaction)?;
        let expected_cursor = arena
            .read_transaction_base(transaction, QWEN3_TTS_MODEL_STATE_DOMAIN)?
            .map_or(0, |snapshot| snapshot.cursor);
        let target_cursor = self.talker_cache.context_len() as u64;
        let components = [
            Some(self.prepared.retained_sequence_memory.clone()),
            Some(self.prepared.tts_pad_embed.clone()),
            self.last_hidden.clone(),
            self.last_logits.clone(),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, tensor)| StateComponentValue {
            component: StateComponentId::new((index + 1) as u32),
            tensor,
        })
        .collect();
        arena.stage_replace(
            transaction,
            QWEN3_TTS_MODEL_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            components,
        )?;
        Ok(())
    }

    pub fn into_retained_talker_cache(self) -> TalkerPhysicalCache {
        self.talker_cache
    }
}

impl PhysicalTtsPrefillManagedCheckpoint {
    pub(crate) fn into_state(self) -> PhysicalTtsPrefillState {
        PhysicalTtsPrefillState {
            prepared: self.prepared,
            talker_cache: self.talker_cache,
            stream_config: self.stream_config,
            progress: self.progress,
            total_tokens: self.total_tokens,
            last_hidden: self.last_hidden,
            last_logits: self.last_logits,
            tensor_sequence: self.tensor_sequence,
        }
    }
}

/// Incremental Qwen3-TTS state backed by scheduler-owned physical talker pages.
///
/// Predictor KV is intentionally absent: every frame receives a fresh
/// [`CodePredictorPhysicalCache`] invocation workspace from the caller.
pub struct PhysicalTtsDecodeState {
    talker_cache: TalkerPhysicalCache,
    text_vocab_size: u32,
    acoustic_vocab_size: u32,
    semantic_vocab_size: u32,
    trailing_text_hidden: Option<Tensor>,
    retained_sequence_memory: Option<Tensor>,
    prefill_tokens: usize,
    trailing_text_len: usize,
    tts_pad_embed: Option<Tensor>,
    max_frames: usize,
    frame_idx: usize,
    offset: usize,
    all_code_groups: Vec<Vec<u32>>,
    semantic_history: Vec<u32>,
    last_hidden: Option<Tensor>,
    last_logits: Option<Tensor>,
    tensor_sequence: Option<PhysicalStateSequenceId>,
    rng: SimpleRng,
    params: TtsGenerationParams,
    stream_config: TtsStreamingConfig,
    emitted_frames: usize,
    emitted_samples: usize,
    decode_raw_token_scratch: Vec<Vec<u32>>,
    finished: bool,
}

/// Row-local continuation snapshot used to make the multi-row convenience API
/// atomic at its `Result<Vec<_>>` boundary. Physical KV has its own logical
/// checkpoint; this value covers every mutable non-KV field touched by talker
/// commit or codec finalization.
struct PhysicalTtsDecodeBatchCheckpoint {
    frame_idx: usize,
    offset: usize,
    all_code_group_lengths: Vec<usize>,
    semantic_history: Vec<u32>,
    last_hidden: Option<Tensor>,
    last_logits: Option<Tensor>,
    retained_sequence_memory: Option<Tensor>,
    trailing_text_hidden: Option<Tensor>,
    tts_pad_embed: Option<Tensor>,
    tensor_sequence: Option<PhysicalStateSequenceId>,
    rng: SimpleRng,
    emitted_frames: usize,
    emitted_samples: usize,
    decode_raw_token_scratch: Vec<Vec<u32>>,
    finished: bool,
}

pub(crate) struct PhysicalTtsManagedQuantumCheckpoint {
    talker_cache: TalkerPhysicalCache,
    continuation: PhysicalTtsDecodeBatchCheckpoint,
}

/// Consuming terminal handoff for a physical TTS session.
///
/// Taking this value drains the model state exactly once and returns ownership
/// of the retained talker page view to the executor for release or reuse.
pub struct PhysicalTtsDecodeCompletion {
    pub talker_cache: TalkerPhysicalCache,
    pub codec_groups: Vec<Vec<u32>>,
    pub frames_generated: usize,
    pub emitted_samples: usize,
}

impl PhysicalTtsDecodeState {
    fn batch_checkpoint(&self) -> PhysicalTtsDecodeBatchCheckpoint {
        PhysicalTtsDecodeBatchCheckpoint {
            frame_idx: self.frame_idx,
            offset: self.offset,
            all_code_group_lengths: self.all_code_groups.iter().map(Vec::len).collect(),
            semantic_history: self.semantic_history.clone(),
            last_hidden: self.last_hidden.clone(),
            last_logits: self.last_logits.clone(),
            retained_sequence_memory: self.retained_sequence_memory.clone(),
            trailing_text_hidden: self.trailing_text_hidden.clone(),
            tts_pad_embed: self.tts_pad_embed.clone(),
            tensor_sequence: self.tensor_sequence,
            rng: self.rng,
            emitted_frames: self.emitted_frames,
            emitted_samples: self.emitted_samples,
            decode_raw_token_scratch: self.decode_raw_token_scratch.clone(),
            finished: self.finished,
        }
    }

    fn restore_batch_checkpoint(&mut self, checkpoint: PhysicalTtsDecodeBatchCheckpoint) {
        self.frame_idx = checkpoint.frame_idx;
        self.offset = checkpoint.offset;
        for (group, length) in self
            .all_code_groups
            .iter_mut()
            .zip(checkpoint.all_code_group_lengths)
        {
            group.truncate(length);
        }
        self.semantic_history = checkpoint.semantic_history;
        self.last_hidden = checkpoint.last_hidden;
        self.last_logits = checkpoint.last_logits;
        self.retained_sequence_memory = checkpoint.retained_sequence_memory;
        self.trailing_text_hidden = checkpoint.trailing_text_hidden;
        self.tts_pad_embed = checkpoint.tts_pad_embed;
        self.tensor_sequence = checkpoint.tensor_sequence;
        self.rng = checkpoint.rng;
        self.emitted_frames = checkpoint.emitted_frames;
        self.emitted_samples = checkpoint.emitted_samples;
        self.decode_raw_token_scratch = checkpoint.decode_raw_token_scratch;
        self.finished = checkpoint.finished;
    }

    pub fn talker_context_len(&self) -> usize {
        self.talker_cache.context_len()
    }

    pub fn frames_generated(&self) -> usize {
        self.all_code_groups.first().map(Vec::len).unwrap_or(0)
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        self.talker_cache.take_completed_writes()
    }

    pub(crate) fn install_retained_talker_reservation(
        &mut self,
        cache: TalkerPhysicalCache,
    ) -> Result<()> {
        if self.talker_cache.arena().id() != cache.arena().id()
            || self.talker_cache.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Qwen3-TTS session cannot switch retained talker authority".to_string(),
            ));
        }
        if cache.context_len() != self.offset {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS talker reservation starts at {}, but decode state is at {}",
                cache.context_len(),
                self.offset
            )));
        }
        self.talker_cache = cache;
        Ok(())
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsManagedQuantumCheckpoint> {
        if self.talker_cache.arena().id() != cache.arena().id()
            || self.talker_cache.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Qwen3-TTS session cannot switch retained talker authority".into(),
            ));
        }
        if cache.context_len() != self.offset {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS talker reservation starts at {}, but decode state is at {}",
                cache.context_len(),
                self.offset
            )));
        }
        Ok(PhysicalTtsManagedQuantumCheckpoint {
            talker_cache: std::mem::replace(&mut self.talker_cache, cache),
            continuation: self.batch_checkpoint(),
        })
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: PhysicalTtsManagedQuantumCheckpoint,
    ) {
        self.talker_cache = checkpoint.talker_cache;
        self.restore_batch_checkpoint(checkpoint.continuation);
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(Error::InferenceError(
                "Qwen3-TTS tensor-state sequence identity changed".into(),
            ));
        }
        self.tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self.tensor_sequence.ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS physical state has no tensor sequence".into())
        })?;
        let snapshot = arena
            .read(sequence, QWEN3_TTS_MODEL_STATE_DOMAIN)?
            .ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS tensor state has no committed snapshot".into())
            })?;
        if snapshot.components.len() != 4 {
            return Err(Error::InferenceError(
                "Qwen3-TTS tensor snapshot has incomplete component coverage".into(),
            ));
        }
        let mut tensors = snapshot
            .components
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if value.component != StateComponentId::new((index + 1) as u32) {
                    return Err(Error::InferenceError(
                        "Qwen3-TTS tensor snapshot has non-canonical components".into(),
                    ));
                }
                value.tensor.clone().ok_or_else(|| {
                    Error::InferenceError("Qwen3-TTS required tensor component is absent".into())
                })
            });
        let retained_sequence_memory = tensors.next().expect("four components")?;
        let tts_pad_embed = tensors.next().expect("four components")?;
        let last_hidden = tensors.next().expect("four components")?;
        let last_logits = tensors.next().expect("four components")?;
        validate_qwen3_tts_tensor_state_shapes(
            &retained_sequence_memory,
            &tts_pad_embed,
            Some(&last_hidden),
            Some(&last_logits),
        )?;
        self.trailing_text_hidden = Some(retained_sequence_memory.narrow(
            1,
            self.prefill_tokens,
            self.trailing_text_len,
        )?);
        self.retained_sequence_memory = Some(retained_sequence_memory);
        self.tts_pad_embed = Some(tts_pad_embed);
        self.last_hidden = Some(last_hidden);
        self.last_logits = Some(last_logits);
        Ok(())
    }

    pub(crate) fn stage_tensor_state(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        validate_qwen3_tts_tensor_state_shapes(
            self.retained_sequence_memory.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS tensor memory was not hydrated".into())
            })?,
            self.tts_pad_embed.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS tensor pad was not hydrated".into())
            })?,
            self.last_hidden.as_ref(),
            self.last_logits.as_ref(),
        )?;
        let transaction = PhysicalStateTransactionId::new(transaction)?;
        let expected_cursor = arena
            .read_transaction_base(transaction, QWEN3_TTS_MODEL_STATE_DOMAIN)?
            .map_or(0, |snapshot| snapshot.cursor);
        let target_cursor = self.talker_cache.context_len() as u64;
        let components = [
            self.retained_sequence_memory.clone(),
            self.tts_pad_embed.clone(),
            self.last_hidden.clone(),
            self.last_logits.clone(),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, tensor)| {
            let tensor = tensor.ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS tensor state was not hydrated".into())
            })?;
            Ok(StateComponentValue {
                component: StateComponentId::new(u32::try_from(index + 1).map_err(|_| {
                    Error::InferenceError("Qwen3-TTS component id overflow".into())
                })?),
                tensor: Some(tensor),
            })
        })
        .collect::<Result<Vec<_>>>()?;
        arena.stage_replace(
            transaction,
            QWEN3_TTS_MODEL_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            components,
        )?;
        self.clear_tensor_handles();
        Ok(())
    }

    pub(crate) fn clear_tensor_handles(&mut self) {
        self.trailing_text_hidden = None;
        self.retained_sequence_memory = None;
        self.tts_pad_embed = None;
        self.last_hidden = None;
        self.last_logits = None;
    }

    /// Consume an in-flight or terminal state during cancellation and return
    /// its retained talker page view without manufacturing a completion.
    pub fn into_retained_talker_cache(self) -> TalkerPhysicalCache {
        self.talker_cache
    }

    /// Consume a terminal state and return the retained physical resources.
    pub fn into_completion(self) -> Result<PhysicalTtsDecodeCompletion> {
        if !self.finished {
            return Err(Error::InvalidInput(
                "Qwen3-TTS physical state cannot drain before completion".to_string(),
            ));
        }
        let frames_generated = self.all_code_groups.first().map(Vec::len).unwrap_or(0);
        Ok(PhysicalTtsDecodeCompletion {
            talker_cache: self.talker_cache,
            codec_groups: self.all_code_groups,
            frames_generated,
            emitted_samples: self.emitted_samples,
        })
    }
}

fn run_tts_decode_batch_transaction<T>(
    states: &mut [&mut PhysicalTtsDecodeState],
    predictor_caches: &mut [&mut CodePredictorPhysicalCache],
    operation: impl FnOnce(
        &mut [&mut PhysicalTtsDecodeState],
        &mut [&mut CodePredictorPhysicalCache],
    ) -> Result<T>,
) -> Result<T> {
    let row_checkpoints = states
        .iter()
        .map(|state| state.batch_checkpoint())
        .collect::<Vec<_>>();
    let talker_checkpoints = states
        .iter()
        .map(|state| state.talker_cache.logical_checkpoint())
        .collect::<Vec<_>>();
    let talker_cursors = states
        .iter()
        .map(|state| state.talker_cache.context_len())
        .collect::<Vec<_>>();
    let predictor_checkpoints = predictor_caches
        .iter()
        .map(|cache| cache.logical_checkpoint())
        .collect::<Vec<_>>();
    let predictor_cursors = predictor_caches
        .iter()
        .map(|cache| cache.context_len())
        .collect::<Vec<_>>();

    let error = match operation(states, predictor_caches) {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };

    let mut rollback_error = None;
    for (((state, row_checkpoint), talker_checkpoint), initial_cursor) in states
        .iter_mut()
        .zip(row_checkpoints)
        .zip(talker_checkpoints)
        .zip(talker_cursors)
    {
        if state.talker_cache.context_len() != initial_cursor {
            if let Err(rollback) = state
                .talker_cache
                .restore_logical_checkpoint(talker_checkpoint)
            {
                rollback_error.get_or_insert(rollback);
            }
        }
        state.restore_batch_checkpoint(row_checkpoint);
    }
    for ((cache, predictor_checkpoint), initial_cursor) in predictor_caches
        .iter_mut()
        .zip(predictor_checkpoints)
        .zip(predictor_cursors)
    {
        if cache.context_len() != initial_cursor {
            if let Err(rollback) = cache.restore_logical_checkpoint(predictor_checkpoint) {
                rollback_error.get_or_insert(rollback);
            }
        }
    }
    if let Some(rollback) = rollback_error {
        Err(Error::InferenceError(format!(
            "Qwen3-TTS decode batch failed: {error}; rollback also failed: {rollback}"
        )))
    } else {
        Err(error)
    }
}

#[derive(Debug, Clone)]
pub struct TtsDecodeStep {
    pub samples: Vec<f32>,
    pub frames_generated: usize,
    pub finished: bool,
    pub sampling_ms: f64,
    pub decode_ms: f64,
    pub codec_ms: f64,
    /// True only when this row executed predictor and talker kernels.
    pub executed_model_row: bool,
}

/// Row-local semantic decision and immutable inputs for the independently
/// schedulable predictor stage. Construction never mutates decode state.
pub struct PreparedTtsPredictorStage {
    semantic_token: u32,
    semantic_embed: Tensor,
    talker_hidden: Tensor,
    source_logits: Tensor,
    text_addition: Tensor,
    expected_talker_context: usize,
    expected_frame_idx: usize,
    expected_rng: SimpleRng,
    expected_semantic_history: Vec<u32>,
    expected_tensor_sequence: Option<PhysicalStateSequenceId>,
    next_rng: SimpleRng,
    next_semantic_history: Vec<u32>,
    sampling_ms: f64,
}

/// Result of row-local frame preparation. Terminal rows bypass predictor and
/// talker work but retain enough information to reproduce scalar RNG/EOS and
/// codec-finalization semantics.
pub enum PreparedTtsFrameStage {
    Predictor(PreparedTtsPredictorStage),
    Terminal(PreparedTtsTerminalStage),
}

enum TtsTerminalReason {
    AlreadyFinished,
    FrameLimit,
    SemanticEos,
}

pub struct PreparedTtsTerminalStage {
    reason: TtsTerminalReason,
    expected_talker_context: usize,
    expected_frame_idx: usize,
    expected_rng: SimpleRng,
    expected_semantic_history: Vec<u32>,
    next_rng: SimpleRng,
    sampling_ms: f64,
}

impl PreparedTtsPredictorStage {
    pub fn semantic_token(&self) -> u32 {
        self.semantic_token
    }

    pub fn talker_context(&self) -> usize {
        self.expected_talker_context
    }
}

/// Completed predictor output waiting for one native talker batch.
pub struct PreparedTtsTalkerStage {
    predictor: PreparedTtsPredictorStage,
    acoustic_codes: Vec<u32>,
    step_input: Tensor,
    predictor_ms: f64,
}

/// Timing handoff from the transactional talker stage to codec emission.
/// Codec/vocoder work intentionally happens only after this state commit.
#[derive(Debug, Clone, Copy)]
pub struct TtsTalkerStageCompletion {
    sampling_ms: f64,
    predictor_ms: f64,
    talker_ms: f64,
}

/// Inputs required to authorize one scheduler-owned Qwen3-TTS decode session.
///
/// This preflight performs tokenization and request-shape arithmetic only. It
/// does not create a talker or predictor cache, so the executor can reserve the
/// immutable exact-session lease before the first retained tensor allocation.
#[derive(Debug, Clone, Copy)]
pub struct TtsSessionCacheRequest<'a> {
    pub text: &'a str,
    pub reference: Option<&'a SpeakerReference>,
    pub language: Option<&'a str>,
    pub instruct: Option<&'a str>,
    pub uses_preset_speaker: bool,
    pub max_frames: usize,
}

/// Exact host-derived sequence shape used for scheduler admission.
///
/// Computing this layout performs tokenization and audio-length arithmetic but
/// does not allocate device tensors or execute either decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3TtsPhysicalSessionLayout {
    pub prefill_tokens: usize,
    pub max_frames: usize,
}

impl TtsGenerationParams {
    /// Convert external generation config to TTS sampling params.
    pub fn from_generation_config(cfg: &crate::runtime::GenerationConfig) -> Self {
        let opts = &cfg.options;
        Self {
            temperature: opts.temperature.max(0.0),
            top_p: opts.top_p.clamp(0.0, 1.0),
            top_k: if opts.top_k == 0 { 50 } else { opts.top_k },
            repetition_penalty: opts.repetition_penalty.max(1.0),
            max_frames: if opts.max_tokens == 0 {
                crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES
            } else {
                opts.max_tokens
                    .clamp(16, crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
            },
        }
    }
}

/// Qwen3-TTS Model for speech synthesis
pub struct Qwen3TtsModel {
    /// Device configuration
    device: DeviceProfile,
    /// Primary transformer data type for inference.
    dtype: DType,
    /// Data type used by the acoustic code predictor.
    code_predictor_dtype: DType,
    /// Data type used by the speech tokenizer decoder.
    speech_tokenizer_dtype: DType,
    /// Storage dtype used by managed talker KV pages.
    talker_state_dtype: DType,
    /// Storage dtype used by invocation-scoped predictor KV pages.
    predictor_state_dtype: DType,
    /// Tokenizer for text and codec tokens
    tokenizer: TtsTokenizer,
    /// Special token IDs
    specials: TtsSpecialTokens,
    /// Main talker (LLM) model
    talker: TalkerModel,
    /// Code predictor for multi-codebook generation
    code_predictor: CodePredictor,
    /// Speech tokenizer decoder for codec to audio conversion
    speech_tokenizer: SpeechTokenizerDecoder,
    /// Model configuration
    config: Qwen3TtsConfig,
    /// Decode-time KV page size.
    kv_page_size: usize,
    /// KV cache quantization mode.
    kv_quantization: KvCacheQuantization,
}

impl Qwen3TtsModel {
    /// Target two-domain contract for the talker and code predictor.
    ///
    /// The loaded adapter remains opaque until both cache owners consume one
    /// engine transaction and return domain-complete write receipts.
    pub(crate) fn managed_inference_state_contract(&self) -> Result<InferenceStateContract> {
        qwen3_tts_inference_state_contract(
            &self.config,
            self.talker_state_dtype,
            self.predictor_state_dtype,
            self.kv_page_size,
        )
    }

    /// Author the complete retained-talker + invocation-predictor state for
    /// the exact execution graphs frozen by the loaded adapter draft.
    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Qwen3TtsPhysicalStateSpec> {
        if stage_graphs.is_empty() {
            return Err(Error::ModelLoadError(
                "Qwen3 TTS physical state has no execution graph".to_string(),
            ));
        }
        let full = self.managed_inference_state_contract()?;
        let mut retained_base = InferenceStateContract {
            abi: full.abi,
            domains: vec![full.domains[0].clone()],
            groups: vec![full.groups[0].clone()],
        };
        let StateDomainSpec::PagedAttention(talker) = &mut retained_base.domains[0] else {
            unreachable!("Qwen3 TTS talker is paged attention")
        };
        talker.header.prefix = PrefixPolicy::Disabled;
        talker.header.checkpoint = CheckpointPolicy::Transactional;
        retained_base.groups[0].prefix_shareable = false;
        retained_base.validate()?;
        let retained = qwen3_tts_retained_state_contract(retained_base, &self.config, self.dtype)?;

        let mut predictor_contract = InferenceStateContract {
            abi: full.abi,
            domains: vec![full.domains[1].clone()],
            groups: vec![full.groups[1].clone()],
        };
        let StateDomainSpec::PagedAttention(predictor) = &mut predictor_contract.domains[0] else {
            unreachable!("Qwen3 TTS predictor is paged attention")
        };
        predictor.header.scope = StateScope::Invocation;
        predictor.header.prefix = PrefixPolicy::Disabled;
        predictor.header.checkpoint = CheckpointPolicy::None;
        predictor_contract.groups[0].prefix_shareable = false;
        predictor_contract.validate()?;
        let predictor_domain = predictor_contract.domains[0].clone();
        let predictor_group = predictor_contract.groups[0].clone();
        let max_invocation_domain_id = predictor_domain.id().get();
        let predictor_workspace_tokens = u64::try_from(self.physical_predictor_workspace_tokens())
            .map_err(|_| {
                Error::ModelLoadError("Qwen3-TTS predictor workspace exceeds u64".into())
            })?;
        let predictor_capacity = InvocationStateCapacity::PagedTokens {
            max_tokens: predictor_workspace_tokens,
        };
        let predictor_physical_bytes =
            minimum_physical_bytes_for_capacity(&predictor_domain, predictor_capacity)?;

        let mut profiles = Vec::with_capacity(stage_graphs.len());
        for stages in stage_graphs {
            let mut invocation_stages = stages
                .iter()
                .enumerate()
                .map(|(index, stage)| {
                    let decode = stage.selector == StageWorkSelector::SequenceDecode;
                    let lease_scope = if decode || stage.max_batch_size == 1 {
                        InvocationLeaseScope::PerRow
                    } else {
                        InvocationLeaseScope::PerStageBatch
                    };
                    let mut domains = decode
                        .then(|| InvocationWorkspaceDomain::State {
                            state: predictor_domain.clone(),
                            capacity: predictor_capacity,
                            placement: predictor_domain.header().placement,
                            formula: WorkspaceFormula {
                                fixed_bytes: predictor_physical_bytes,
                                dimensions: vec![],
                                terms: vec![],
                            },
                        })
                        .into_iter()
                        .collect::<Vec<_>>();
                    let scratch_id = max_invocation_domain_id
                        .checked_add(u32::try_from(index + 1).map_err(|_| {
                            Error::ModelLoadError("Qwen3 TTS stage count exceeds u32".into())
                        })?)
                        .ok_or_else(|| {
                            Error::ModelLoadError("Qwen3 TTS scratch domain id overflow".into())
                        })?;
                    if let Some(scratch) = exact_stage_scratch_domain(
                        stage,
                        StateDomainId::new(scratch_id),
                        lease_scope,
                    )? {
                        domains.push(scratch);
                    }
                    Ok(InvocationStageWorkspace {
                        stage: stage.id,
                        lease_scope,
                        groups: decode
                            .then(|| predictor_group.clone())
                            .into_iter()
                            .collect(),
                        domains,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            invocation_stages.sort_unstable_by_key(|stage| stage.stage);
            profiles.push(InvocationWorkspaceProfile {
                stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
                stages: invocation_stages,
            });
        }
        profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
        profiles.dedup();
        let descriptor = CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Managed {
                contract: retained.clone(),
            },
            invocation: InvocationWorkspaceSet::Bounded { profiles },
        };
        for stages in stage_graphs {
            descriptor.validate_against_stages(stages)?;
        }
        Ok(Qwen3TtsPhysicalStateSpec {
            retained,
            retained_max_tokens: self.config.talker_config.max_position_embeddings,
            descriptor,
            predictor_contract,
        })
    }
}

fn qwen3_tts_state_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        other => Err(Error::ModelLoadError(format!(
            "Qwen3 TTS retained model state does not support {other:?}"
        ))),
    }
}

fn qwen3_tts_retained_state_contract(
    mut retained: InferenceStateContract,
    config: &Qwen3TtsConfig,
    dtype: DType,
) -> Result<InferenceStateContract> {
    let state_dtype = qwen3_tts_state_dtype(dtype)?;
    let hidden = config.talker_config.hidden_size;
    let max_sequence = config.talker_config.max_position_embeddings;
    let vocab = config.talker_config.vocab_size;
    retained
        .domains
        .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: QWEN3_TTS_MODEL_STATE_DOMAIN,
                scope: StateScope::Retained,
                clock: StateClock::Custom("qwen3_tts_talker_tokens".into()),
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::Transactional,
            },
            components: vec![
                qwen3_tts_state_component(
                    1,
                    TensorRole::EncoderMemory,
                    &[("batch", 1), ("sequence", max_sequence), ("hidden", hidden)],
                    state_dtype,
                )?,
                qwen3_tts_state_component(
                    2,
                    TensorRole::RetainedEmbedding,
                    &[("batch", 1), ("sequence", 1), ("hidden", hidden)],
                    state_dtype,
                )?,
                qwen3_tts_state_component(
                    3,
                    TensorRole::RecurrentHidden,
                    &[("batch", 1), ("sequence", 1), ("hidden", hidden)],
                    state_dtype,
                )?,
                qwen3_tts_state_component(
                    4,
                    TensorRole::RetainedLogits,
                    &[("batch", 1), ("sequence", 1), ("vocabulary", vocab)],
                    state_dtype,
                )?,
            ],
        }));
    retained.groups[0]
        .domains
        .push(QWEN3_TTS_MODEL_STATE_DOMAIN);
    retained.groups[0].prefix_shareable = false;
    retained.validate()?;
    Ok(retained)
}

fn qwen3_tts_state_component(
    id: u32,
    role: TensorRole,
    dimensions: &[(&str, usize)],
    dtype: StateDType,
) -> Result<TensorComponentSpec> {
    let dimensions = dimensions
        .iter()
        .map(|(axis, extent)| {
            let axis = match *axis {
                "batch" => ShapeAxis::Batch,
                "sequence" => ShapeAxis::Sequence,
                "hidden" => ShapeAxis::Hidden,
                other => ShapeAxis::Custom(other.into()),
            };
            Ok(ShapeDimension {
                axis,
                extent: ShapeExtent::RuntimeBounded {
                    min: 1,
                    max: u64::try_from(*extent).map_err(|_| {
                        Error::ModelLoadError("Qwen3 TTS state extent exceeds u64".into())
                    })?,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TensorComponentSpec {
        id: StateComponentId::new(id),
        role,
        shape: BoundedShape { dimensions },
        accepted_dtypes: vec![dtype],
    })
}

fn qwen3_tts_inference_state_contract(
    config: &Qwen3TtsConfig,
    talker_dtype: DType,
    predictor_dtype: DType,
    page_tokens: usize,
) -> Result<InferenceStateContract> {
    let talker = &config.talker_config;
    let predictor = &talker.code_predictor_config;
    let talker_domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
        domain: StateDomainId::new(1),
        clock: StateClock::Custom("qwen3_tts_talker_tokens".into()),
        num_layers: talker.num_hidden_layers,
        num_query_heads: talker.num_attention_heads,
        num_kv_heads: talker.num_key_value_heads,
        key_head_dim: talker.head_dim,
        value_head_dim: talker.head_dim,
        sliding_window: None,
        storage_dtype: talker_dtype,
        preferred_page_tokens: page_tokens,
        prefix: PrefixPolicy::CommittedPages {
            positions: PositionSemantics::Absolute,
        },
    })?;
    let predictor_domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
        domain: StateDomainId::new(2),
        clock: StateClock::Custom("qwen3_tts_predictor_tokens".into()),
        num_layers: predictor.num_hidden_layers,
        num_query_heads: predictor.num_attention_heads,
        num_kv_heads: predictor.num_key_value_heads,
        key_head_dim: predictor.head_dim,
        value_head_dim: predictor.head_dim,
        sliding_window: None,
        storage_dtype: predictor_dtype,
        preferred_page_tokens: page_tokens,
        prefix: PrefixPolicy::Disabled,
    })?;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![
            StateDomainSpec::PagedAttention(talker_domain),
            StateDomainSpec::PagedAttention(predictor_domain),
        ],
        groups: vec![
            StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: true,
            },
            StateGroupSpec {
                id: StateGroupId::new(2),
                domains: vec![StateDomainId::new(2)],
                prefix_shareable: false,
            },
        ],
    };
    contract.validate()?;
    Ok(contract)
}

#[derive(Debug, Clone, Serialize)]
pub struct Qwen3TtsDiagnostics {
    pub model_family: &'static str,
    pub model_type: String,
    pub model_size: String,
    pub device_kind: String,
    pub talker_dtype: String,
    pub code_predictor_dtype: String,
    pub speech_tokenizer_dtype: String,
    pub kv_page_size: usize,
    pub kv_quantization: String,
    pub speaker_count: usize,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub num_code_groups: usize,
    /// Whether bounded candidate selection can execute on the selected device.
    pub device_sampling: bool,
    /// Backward-compatible CUDA-specific capability flag.
    pub cuda_sampling: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen3TtsDTypePlan {
    talker: DType,
    code_predictor: DType,
    speech_tokenizer: DType,
}

fn select_qwen3_tts_dtypes(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
    is_custom_voice_model: bool,
    is_voice_clone_model: bool,
) -> Result<Qwen3TtsDTypePlan> {
    if let Some(raw) = dtype_override.map(str::trim).filter(|raw| !raw.is_empty()) {
        let dtype =
            device.select_model_dtype_checked(ModelFamily::Qwen3Tts, Some(raw), "Qwen3 TTS")?;
        return Ok(Qwen3TtsDTypePlan {
            talker: dtype,
            code_predictor: dtype,
            speech_tokenizer: dtype,
        });
    }

    let legacy_dtype = if is_custom_voice_model || is_voice_clone_model {
        // CustomVoice and Base/voice-clone checkpoints require F32 for
        // intelligible audio. In particular, selecting F16 for every Metal
        // component produces structurally valid but corrupted speech.
        DType::F32
    } else if device.kind.is_metal() {
        // VoiceDesign remains safe at F16 and benefits from lower residency.
        DType::F16
    } else {
        device.select_model_dtype(ModelFamily::Qwen3Tts, None)
    };

    if device.kind.is_cuda() {
        let transformer_dtype = device.select_model_dtype(ModelFamily::Qwen3Tts, None);
        let speech_tokenizer_dtype = if is_custom_voice_model || is_voice_clone_model {
            DType::F32
        } else {
            legacy_dtype
        };
        Ok(Qwen3TtsDTypePlan {
            talker: transformer_dtype,
            code_predictor: transformer_dtype,
            speech_tokenizer: speech_tokenizer_dtype,
        })
    } else {
        Ok(Qwen3TtsDTypePlan {
            talker: legacy_dtype,
            code_predictor: legacy_dtype,
            speech_tokenizer: legacy_dtype,
        })
    }
}

fn select_qwen3_tts_state_dtypes(
    device: &DeviceProfile,
    compute: Qwen3TtsDTypePlan,
    kv_cache_dtype: &str,
) -> Result<(DType, DType)> {
    if !device.kind.is_metal() || compute.talker != DType::F32 {
        return Ok((compute.talker, compute.code_predictor));
    }
    let storage = match kv_cache_dtype.trim().to_ascii_lowercase().as_str() {
        "float16" | "fp16" | "f16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        "float32" | "fp32" | "f32" => DType::F32,
        value => {
            return Err(Error::InvalidInput(format!(
                "Qwen3 TTS managed state does not support KV dtype `{value}`"
            )))
        }
    };
    Ok((storage, storage))
}

fn qwen_tts_uses_device_sampling(device: &DeviceProfile) -> bool {
    device.kind.is_cuda() || device.kind.is_metal()
}

fn qwen_tts_allows_eos(frames_generated: usize) -> bool {
    frames_generated >= MIN_QWEN_TTS_TOKENS_BEFORE_EOS
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TtsSessionCacheLayout {
    prefill_tokens: usize,
    max_frames: usize,
}

impl TtsSessionCacheLayout {
    fn talker_cache_tokens(self) -> Option<usize> {
        self.prefill_tokens.checked_add(self.max_frames)
    }
}

fn cache_layout_overflow() -> Error {
    Error::Overloaded("Qwen3-TTS session cache layout overflow".to_string())
}

fn conditioned_prefill_tokens(
    prompt_tokens: usize,
    has_language: bool,
    has_speaker: bool,
    instruct_tokens: usize,
) -> Option<usize> {
    // build_conditioned_prefill_embeddings retains three role tokens, all but
    // the final codec-BOS token from the codec prefix, optional instruction
    // tokens, and the first target-text token when present.
    let codec_prefix_tokens =
        (if has_language { 6usize } else { 5usize }).checked_add(usize::from(has_speaker))?;
    3usize
        .checked_add(codec_prefix_tokens.checked_sub(1)?)?
        .checked_add(instruct_tokens)?
        .checked_add(usize::from(prompt_tokens > 0))
}

fn resolve_session_cache_layout(
    max_position_embeddings: usize,
    prefill_tokens: usize,
    requested_max_frames: usize,
) -> Result<TtsSessionCacheLayout> {
    let first_decode_position = prefill_tokens
        .checked_add(1)
        .ok_or_else(cache_layout_overflow)?;
    let context_budget = max_position_embeddings
        .checked_sub(first_decode_position)
        .filter(|budget| *budget > 0)
        .ok_or_else(|| {
            Error::InferenceError(
                "Qwen3-TTS prompt exceeds model context window; no room for audio generation"
                    .to_string(),
            )
        })?;
    let max_frames = if requested_max_frames == 0 {
        context_budget
    } else {
        requested_max_frames.max(1).min(context_budget)
    };
    Ok(TtsSessionCacheLayout {
        prefill_tokens,
        max_frames,
    })
}

fn standard_session_cache_layout(
    max_position_embeddings: usize,
    prompt_tokens: usize,
    has_language: bool,
    has_speaker: bool,
    instruct_tokens: usize,
    requested_max_frames: usize,
) -> Result<TtsSessionCacheLayout> {
    let prefill_tokens =
        conditioned_prefill_tokens(prompt_tokens, has_language, has_speaker, instruct_tokens)
            .ok_or_else(cache_layout_overflow)?;
    resolve_session_cache_layout(
        max_position_embeddings,
        prefill_tokens,
        requested_max_frames,
    )
}

fn voice_clone_session_cache_layout(
    max_position_embeddings: usize,
    prompt_tokens: usize,
    reference_prompt_tokens: usize,
    reference_frames: usize,
    has_language: bool,
    requested_max_frames: usize,
) -> Result<TtsSessionCacheLayout> {
    let target_text_tokens = prompt_tokens.checked_sub(8).filter(|count| *count > 0);
    let reference_text_tokens = reference_prompt_tokens
        .checked_sub(5)
        .filter(|count| *count > 0);
    let (target_text_tokens, reference_text_tokens) =
        match (target_text_tokens, reference_text_tokens) {
            (Some(target), Some(reference)) => (target, reference),
            _ => {
                return Err(Error::InvalidInput(
                    "Voice cloning requires non-empty target/reference transcript tokens"
                        .to_string(),
                ))
            }
        };
    if reference_frames == 0 {
        return Err(Error::ModelError(
            "Voice cloning reference encoder produced no conditioning tokens".to_string(),
        ));
    }

    let text_tokens = target_text_tokens
        .checked_add(reference_text_tokens)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(cache_layout_overflow)?;
    let codec_tokens = reference_frames
        .min(MAX_VOICE_CLONE_REFERENCE_FRAMES)
        .checked_add(1)
        .ok_or_else(cache_layout_overflow)?;
    let base_prefill_tokens =
        conditioned_prefill_tokens(0, has_language, false, 0).ok_or_else(cache_layout_overflow)?;
    let prefill_tokens = base_prefill_tokens
        .checked_add(text_tokens.max(codec_tokens))
        .ok_or_else(cache_layout_overflow)?;

    resolve_session_cache_layout(
        max_position_embeddings,
        prefill_tokens,
        requested_max_frames,
    )
}

impl Qwen3TtsModel {
    /// Load a Qwen3-TTS model from the specified directory
    pub fn load(
        model_dir: &Path,
        device: DeviceProfile,
        kv_page_size: usize,
        kv_cache_dtype: &str,
    ) -> Result<Self> {
        info!("Loading Qwen3-TTS model from {:?}", model_dir);

        // Load configuration
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Qwen3TtsConfig = serde_json::from_str(&config_str)?;

        info!("Model type: {}", config.tts_model_type);
        info!("Model size: {}", config.tts_model_size);

        let model_type_normalized = config
            .tts_model_type
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_'], "");
        let is_custom_voice_model = model_type_normalized == "customvoice";
        let is_voice_clone_model = model_type_normalized == "base"
            || model_type_normalized == "voiceclone"
            || model_type_normalized == "voicecloning"
            || config.talker_config.spk_id.is_empty();
        let dtype_override = std::env::var("IZWI_QWEN_TTS_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype_plan = select_qwen3_tts_dtypes(
            &device,
            dtype_override.as_deref(),
            is_custom_voice_model,
            is_voice_clone_model,
        )?;
        let (talker_state_dtype, predictor_state_dtype) =
            select_qwen3_tts_state_dtypes(&device, dtype_plan, kv_cache_dtype)?;

        // Load tokenizer
        let specials = TtsSpecialTokens::from_configs(&config, &config.talker_config);
        let tokenizer = TtsTokenizer::load(model_dir, specials.clone(), &config.talker_config)?;

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let talker_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                dtype_plan.talker,
                &device.device,
            )?
        };
        let code_predictor_vb = if dtype_plan.code_predictor == dtype_plan.talker {
            talker_vb.clone()
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&weights_path),
                    dtype_plan.code_predictor,
                    &device.device,
                )?
            }
        };

        // Load talker model
        info!("Loading talker model...");
        let talker = TalkerModel::load(config.talker_config.clone(), talker_vb.pp("talker"))?;

        // Load code predictor
        info!("Loading code predictor...");
        let num_code_groups = config.talker_config.num_code_groups;
        // For 1.7B model, codec embeddings use talker.text_hidden_size (2048)
        // For 0.6B model, codec embeddings use code_predictor.hidden_size (1024)
        // Detect 1.7B by checking if talker.hidden_size differs from code_predictor.hidden_size
        let mut code_predictor_config = config.talker_config.code_predictor_config.clone();
        if config.talker_config.hidden_size != code_predictor_config.hidden_size {
            // 1.7B case: codec embeddings use text_hidden_size dimension
            code_predictor_config.text_hidden_size = Some(config.talker_config.text_hidden_size);
        }
        let code_predictor = CodePredictor::load(
            code_predictor_config,
            code_predictor_vb.pp("talker.code_predictor"),
            num_code_groups,
        )?;

        // Load speech tokenizer decoder
        info!("Loading speech tokenizer decoder...");
        let speech_tokenizer_path = model_dir.join("speech_tokenizer");
        let speech_tokenizer = SpeechTokenizerDecoder::load(
            &speech_tokenizer_path,
            device.device.clone(),
            dtype_plan.speech_tokenizer,
        )?;

        info!(
            "Qwen3-TTS model loaded successfully on {:?} (talker {:?}, predictor {:?}, speech tokenizer {:?}, talker state {:?}, predictor state {:?})",
            device.kind,
            dtype_plan.talker,
            dtype_plan.code_predictor,
            dtype_plan.speech_tokenizer,
            talker_state_dtype,
            predictor_state_dtype,
        );
        let kv_quantization = KvCacheQuantization::from_dtype_hint(kv_cache_dtype);

        Ok(Self {
            device,
            dtype: dtype_plan.talker,
            code_predictor_dtype: dtype_plan.code_predictor,
            speech_tokenizer_dtype: dtype_plan.speech_tokenizer,
            talker_state_dtype,
            predictor_state_dtype,
            tokenizer,
            specials,
            talker,
            code_predictor,
            speech_tokenizer,
            config,
            kv_page_size: kv_page_size.max(1),
            kv_quantization,
        })
    }

    /// Resolve the exact talker prefill and bounded output shape before the
    /// scheduler allocates retained physical pages.
    pub fn physical_session_layout(
        &self,
        request: TtsSessionCacheRequest<'_>,
    ) -> Result<Qwen3TtsPhysicalSessionLayout> {
        let layout = self.session_cache_layout(request)?;
        Ok(Qwen3TtsPhysicalSessionLayout {
            prefill_tokens: layout.prefill_tokens,
            max_frames: layout.max_frames,
        })
    }

    fn session_cache_layout(
        &self,
        request: TtsSessionCacheRequest<'_>,
    ) -> Result<TtsSessionCacheLayout> {
        let prompt_ids = self.encode_assistant_prompt_ids(request.text)?;
        let has_language = self.resolve_language_id(request.language).is_some();
        if let Some(reference) = request.reference {
            let reference_prompt_ids = self.encode_reference_prompt_ids(reference.text.as_str())?;
            let reference_frames = self
                .speech_tokenizer
                .reference_frame_upper_bound(reference.audio_samples.len(), reference.sample_rate)?
                .min(MAX_VOICE_CLONE_REFERENCE_FRAMES);
            voice_clone_session_cache_layout(
                self.config.talker_config.max_position_embeddings,
                prompt_ids.len(),
                reference_prompt_ids.len(),
                reference_frames,
                has_language,
                request.max_frames,
            )
        } else {
            let instruct_tokens = self
                .encode_instruction_ids(request.instruct)?
                .map_or(0, |tokens| tokens.len());
            standard_session_cache_layout(
                self.config.talker_config.max_position_embeddings,
                prompt_ids.len(),
                has_language,
                request.uses_preset_speaker,
                instruct_tokens,
                request.max_frames,
            )
        }
    }

    /// Exact token capacity for each fresh per-frame predictor workspace.
    pub fn physical_predictor_workspace_tokens(&self) -> usize {
        self.code_predictor.physical_context_tokens_per_frame()
    }

    pub fn supports_resumable_prefill(&self) -> bool {
        true
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    pub fn continuous_decode_is_tensor_batched(&self) -> bool {
        true
    }

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        let predictor = &self.config.talker_config.code_predictor_config;
        let predictor_span = self.physical_predictor_workspace_tokens();
        let transformer = transformer_workspace_upper_bound_bytes(
            predictor_span,
            predictor_span,
            predictor.hidden_size,
            predictor.intermediate_size,
            predictor.num_attention_heads,
            predictor.num_key_value_heads,
            predictor.head_dim,
            predictor.vocab_size,
            self.code_predictor_dtype.size_in_bytes(),
        )?;
        let row_staging = self
            .config
            .talker_config
            .hidden_size
            .checked_mul(8)
            .and_then(|elements| elements.checked_mul(self.dtype.size_in_bytes()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS batch row staging overflow".into()))?;
        transformer
            .checked_add(row_staging)
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS batch workspace overflow".into()))
    }

    pub fn resumable_prefill_workspace_bytes(
        &self,
        prefill_tokens: usize,
        max_frames: usize,
        has_reference: bool,
    ) -> Result<(u64, u64)> {
        let sequence = prefill_tokens
            .checked_add(max_frames)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS prefill sequence overflow".into()))?;
        let host_elements = sequence
            .checked_mul(8)
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS prefill host overflow".into()))?;
        let host = u64::try_from(host_elements)
            .ok()
            .and_then(|value| value.checked_mul(std::mem::size_of::<u32>() as u64))
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS prefill host overflow".into()))?;
        let talker = &self.config.talker_config;
        let transformer = transformer_workspace_upper_bound_bytes(
            prefill_tokens,
            prefill_tokens,
            talker.hidden_size,
            talker.intermediate_size,
            talker.num_attention_heads,
            talker.num_key_value_heads,
            talker.head_dim,
            talker.vocab_size,
            self.dtype.size_in_bytes(),
        )?;
        let embedding_elements = sequence
            .checked_mul(talker.hidden_size)
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| {
                prefill_tokens
                    .checked_mul(talker.text_hidden_size)
                    .and_then(|text| value.checked_add(text))
            })
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS prefill embedding overflow".into()))?;
        let embedding_bytes = u64::try_from(embedding_elements)
            .ok()
            .and_then(|value| value.checked_mul(self.dtype.size_in_bytes() as u64))
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS prefill embedding overflow".into()))?;
        let mut accelerator = transformer
            .checked_add(embedding_bytes)
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS prefill tensor overflow".into()))?;
        if has_reference {
            accelerator = accelerator
                .checked_add(
                    self.speech_tokenizer
                        .reference_encode_temporary_upper_bound_bytes(
                            30,
                            self.speech_tokenizer_dtype.size_in_bytes(),
                        )?,
                )
                .ok_or_else(|| {
                    Error::Overloaded("Qwen3-TTS reference workspace overflow".into())
                })?;
        }
        Ok((host.max(1), accelerator.max(1)))
    }

    /// Frame-bounded scheduler accounting for one continuously decoded row.
    /// Host memory includes retained codec history, the rollback-safe codec
    /// scratch clone, bounded semantic history, and commit-fenced waveform
    /// copies. Backend memory includes both talker/predictor batching and the
    /// terminal speech-tokenizer/vocoder peak.
    pub fn continuous_decode_bounded_workspace_per_row_bytes(
        &self,
        prefill_tokens: usize,
        max_frames: usize,
    ) -> Result<(u64, u64)> {
        let frames = max_frames.max(1);
        let groups = self.config.talker_config.num_code_groups.max(1);
        let samples = self.speech_tokenizer.decoded_sample_upper_bound(frames)?;
        let checked_bytes = |elements: usize, element_bytes: usize, label: &str| {
            elements
                .checked_mul(element_bytes)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| Error::Overloaded(format!("Qwen3-TTS {label} workspace overflow")))
        };
        let codec_elements = groups
            .checked_mul(frames)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS codec history overflow".into()))?;
        let codec_host = checked_bytes(codec_elements, std::mem::size_of::<u32>(), "codec host")?;
        let semantic_host = checked_bytes(
            frames.min(256),
            std::mem::size_of::<u32>(),
            "semantic history",
        )?;
        let sampling_host =
            checked_bytes(self.config.talker_config.vocab_size, 32, "sampling vectors")?;
        // Decoder output, row result, and commit-fenced stream staging can
        // coexist. The accumulated final output is reserved by RuntimeService,
        // not duplicated in this per-quantum workspace.
        let waveform_host = checked_bytes(
            samples
                .checked_mul(3)
                .ok_or_else(|| Error::Overloaded("Qwen3-TTS waveform bound overflow".into()))?,
            std::mem::size_of::<f32>(),
            "waveform host",
        )?;
        let vector_headers = groups
            .checked_mul(3)
            .and_then(|count| count.checked_mul(std::mem::size_of::<Vec<u32>>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS vector metadata overflow".into()))?;
        let host_bytes = codec_host
            .checked_add(semantic_host)
            .and_then(|bytes| bytes.checked_add(waveform_host))
            .and_then(|bytes| bytes.checked_add(sampling_host))
            .and_then(|bytes| bytes.checked_add(vector_headers))
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS host workspace overflow".into()))?;

        let batch = self.continuous_decode_batch_workspace_per_row_bytes()?;
        let codec = self.speech_tokenizer.decode_temporary_upper_bound_bytes(
            frames,
            self.speech_tokenizer_dtype.size_in_bytes(),
        )?;
        let context = prefill_tokens
            .checked_add(frames)
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS context workspace overflow".into()))?;
        let talker = &self.config.talker_config;
        let talker_workspace = transformer_workspace_upper_bound_bytes(
            1,
            context,
            talker.hidden_size,
            talker.intermediate_size,
            talker.num_attention_heads,
            talker.num_key_value_heads,
            talker.head_dim,
            talker.vocab_size,
            self.dtype.size_in_bytes(),
        )?;
        let accelerator_bytes = batch
            .checked_add(codec)
            .and_then(|bytes| bytes.checked_add(talker_workspace))
            .ok_or_else(|| Error::Overloaded("Qwen3-TTS accelerator workspace overflow".into()))?;
        Ok((host_bytes, accelerator_bytes))
    }

    pub fn diagnostics(&self) -> Qwen3TtsDiagnostics {
        Qwen3TtsDiagnostics {
            model_family: "qwen3_tts",
            model_type: self.config.tts_model_type.clone(),
            model_size: self.config.tts_model_size.clone(),
            device_kind: format!("{:?}", self.device.kind),
            talker_dtype: format!("{:?}", self.dtype),
            code_predictor_dtype: format!("{:?}", self.code_predictor_dtype),
            speech_tokenizer_dtype: format!("{:?}", self.speech_tokenizer_dtype),
            kv_page_size: self.kv_page_size,
            kv_quantization: format!("{:?}", self.kv_quantization).to_ascii_lowercase(),
            speaker_count: self.config.talker_config.spk_id.len(),
            vocab_size: self.config.talker_config.vocab_size,
            text_vocab_size: self.config.talker_config.text_vocab_size,
            num_code_groups: self.config.talker_config.num_code_groups,
            device_sampling: qwen_tts_uses_device_sampling(&self.device),
            cuda_sampling: self.device.kind.is_cuda(),
        }
    }

    pub fn start_physical_decode_with_voice_clone_params(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsDecodeState> {
        let mut prefill = self.begin_physical_prefill_with_voice_clone_params(
            text,
            reference,
            language,
            params,
            stream_config,
            talker_cache,
        )?;
        let total = prefill.prefill_tokens();
        self.continue_physical_prefill(&mut prefill, 0, total)?;
        self.finish_physical_prefill(prefill)
    }

    pub fn begin_physical_prefill_with_voice_clone_params(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsPrefillState> {
        let prepared = self.prepare_voice_clone_decode(text, reference, language, params)?;
        self.begin_physical_prefill(prepared, stream_config, talker_cache)
    }

    pub fn start_physical_decode_with_speaker_params(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsDecodeState> {
        let mut prefill = self.begin_physical_prefill_with_speaker_params(
            text,
            speaker,
            language,
            instruct,
            params,
            stream_config,
            talker_cache,
        )?;
        let total = prefill.prefill_tokens();
        self.continue_physical_prefill(&mut prefill, 0, total)?;
        self.finish_physical_prefill(prefill)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn begin_physical_prefill_with_speaker_params(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsPrefillState> {
        let prepared = self.prepare_speaker_decode(text, speaker, language, instruct, params)?;
        self.begin_physical_prefill(prepared, stream_config, talker_cache)
    }

    pub fn start_physical_decode_with_text_params(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsDecodeState> {
        let mut prefill = self.begin_physical_prefill_with_text_params(
            text,
            language,
            instruct,
            params,
            stream_config,
            talker_cache,
        )?;
        let total = prefill.prefill_tokens();
        self.continue_physical_prefill(&mut prefill, 0, total)?;
        self.finish_physical_prefill(prefill)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn begin_physical_prefill_with_text_params(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsPrefillState> {
        let prepared = self.prepare_text_decode(text, language, instruct, params)?;
        self.begin_physical_prefill(prepared, stream_config, talker_cache)
    }

    fn prepared_decode_prefill(
        &self,
        prefill_embeds: Tensor,
        trailing_text_hidden: Tensor,
        tts_pad_embed: Tensor,
        params: TtsGenerationParams,
    ) -> Result<PreparedTtsDecodePrefill> {
        let prefill_tokens = prefill_embeds.dim(1)?;
        let trailing_tokens = trailing_text_hidden.dim(1)?;
        let retained_tokens = prefill_tokens
            .checked_add(trailing_tokens)
            .ok_or_else(|| Error::InferenceError("Qwen3-TTS retained sequence overflow".into()))?;
        if retained_tokens > self.config.talker_config.max_position_embeddings {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS prepared and trailing memory require {retained_tokens} tokens, exceeding context {}",
                self.config.talker_config.max_position_embeddings
            )));
        }
        let retained_sequence_memory = Tensor::cat(&[&prefill_embeds, &trailing_text_hidden], 1)?;
        let prefill_embeds = retained_sequence_memory.narrow(1, 0, prefill_tokens)?;
        let trailing_text_hidden =
            retained_sequence_memory.narrow(1, prefill_tokens, trailing_tokens)?;
        Ok(PreparedTtsDecodePrefill {
            prefill_embeds,
            trailing_text_hidden,
            retained_sequence_memory,
            tts_pad_embed,
            params,
        })
    }

    fn prepare_voice_clone_decode(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<PreparedTtsDecodePrefill> {
        let ref_codec_tokens = self.encode_reference_audio(reference)?;
        if ref_codec_tokens.is_empty() || ref_codec_tokens[0].is_empty() {
            return Err(Error::ModelError(
                "Voice cloning reference encoder produced no conditioning tokens".to_string(),
            ));
        }

        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let ref_prompt_ids = self.encode_reference_prompt_ids(reference.text.as_str())?;
        let target_text_ids: Vec<u32> = if prompt_ids.len() > 8 {
            prompt_ids[3..prompt_ids.len() - 5].to_vec()
        } else {
            Vec::new()
        };
        let reference_text_ids: Vec<u32> = if ref_prompt_ids.len() > 5 {
            ref_prompt_ids[3..ref_prompt_ids.len() - 2].to_vec()
        } else {
            Vec::new()
        };
        if target_text_ids.is_empty() || reference_text_ids.is_empty() {
            return Err(Error::InvalidInput(
                "Voice cloning requires non-empty target/reference transcript tokens".to_string(),
            ));
        }

        let language_id = self.resolve_language_id(language);
        let base_prefill =
            self.build_conditioned_prefill_embeddings(&[], None, language_id, None)?;
        let tts_pad_embed = self
            .talker
            .get_projected_special_embed(self.specials.tts_pad_token_id)?;
        let tts_eos_embed = self
            .talker
            .get_projected_special_embed(self.specials.tts_eos_token_id)?;
        let (icl_embed, trailing_text_hidden) = self.build_voice_clone_icl_embeddings(
            &target_text_ids,
            &reference_text_ids,
            &ref_codec_tokens,
            &tts_pad_embed,
            &tts_eos_embed,
            false,
        )?;
        let prefill_embeds = Tensor::cat(&[&base_prefill, &icl_embed], 1)?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Voice-clone prompt exceeds model context window".to_string(),
            ));
        }
        let resolved_max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        self.prepared_decode_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            TtsGenerationParams {
                max_frames: resolved_max_frames,
                ..params.clone()
            },
        )
    }

    fn prepare_speaker_decode(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<PreparedTtsDecodePrefill> {
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let speaker_id = self.tokenizer.get_speaker_id(speaker).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unknown speaker '{speaker}'. Available speakers: {}",
                self.tokenizer
                    .available_speakers()
                    .into_iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;
        let prefill_embeds = self.build_conditioned_prefill_embeddings(
            &prompt_ids,
            Some(speaker_id),
            language_id,
            instruct_ids.as_deref(),
        )?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let resolved_max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let (trailing_text_hidden, _, tts_pad_embed) =
            self.build_trailing_text_embeddings_from_prompt(&prompt_ids, resolved_max_frames)?;
        self.prepared_decode_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            TtsGenerationParams {
                max_frames: resolved_max_frames,
                ..params.clone()
            },
        )
    }

    fn prepare_text_decode(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<PreparedTtsDecodePrefill> {
        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;
        let prefill_embeds = self.build_conditioned_prefill_embeddings(
            &prompt_ids,
            None,
            language_id,
            instruct_ids.as_deref(),
        )?;
        let prefill_len = prefill_embeds.dim(1)?;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let resolved_max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let (trailing_text_hidden, _, tts_pad_embed) =
            self.build_trailing_text_embeddings_from_prompt(&prompt_ids, resolved_max_frames)?;
        self.prepared_decode_prefill(
            prefill_embeds,
            trailing_text_hidden,
            tts_pad_embed,
            TtsGenerationParams {
                max_frames: resolved_max_frames,
                ..params.clone()
            },
        )
    }

    /// Prepare one row-local semantic decision without mutating retained state.
    ///
    /// Terminal rows carry their row-local RNG transition and bypass
    /// predictor/talker work; codec finalization remains outside that transaction.
    pub fn prepare_tts_predictor_stage(
        &self,
        state: &PhysicalTtsDecodeState,
    ) -> Result<PreparedTtsFrameStage> {
        if state.finished {
            return Ok(PreparedTtsFrameStage::Terminal(PreparedTtsTerminalStage {
                reason: TtsTerminalReason::AlreadyFinished,
                expected_talker_context: state.talker_cache.context_len(),
                expected_frame_idx: state.frame_idx,
                expected_rng: state.rng,
                expected_semantic_history: state.semantic_history.clone(),
                next_rng: state.rng,
                sampling_ms: 0.0,
            }));
        }
        if state.frame_idx >= state.max_frames {
            return Ok(PreparedTtsFrameStage::Terminal(PreparedTtsTerminalStage {
                reason: TtsTerminalReason::FrameLimit,
                expected_talker_context: state.talker_cache.context_len(),
                expected_frame_idx: state.frame_idx,
                expected_rng: state.rng,
                expected_semantic_history: state.semantic_history.clone(),
                next_rng: state.rng,
                sampling_ms: 0.0,
            }));
        }
        if state.offset != state.talker_cache.context_len() {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical state offset {} diverged from retained talker cursor {}",
                state.offset,
                state.talker_cache.context_len()
            )));
        }
        let sampling_started = Instant::now();
        let allow_eos = qwen_tts_allows_eos(state.frames_generated());
        let mut next_rng = state.rng;
        let last_logits = state.last_logits.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS last logits are not hydrated".into())
        })?;
        let semantic_token = sample_semantic(
            &last_logits.i((0, 0))?,
            state.semantic_vocab_size,
            self.specials.codec_eos_token_id,
            allow_eos,
            &state.params,
            &state.semantic_history,
            &mut next_rng,
            qwen_tts_uses_device_sampling(&self.device),
        )?;
        if allow_eos && semantic_token == self.specials.codec_eos_token_id {
            return Ok(PreparedTtsFrameStage::Terminal(PreparedTtsTerminalStage {
                reason: TtsTerminalReason::SemanticEos,
                expected_talker_context: state.talker_cache.context_len(),
                expected_frame_idx: state.frame_idx,
                expected_rng: state.rng,
                expected_semantic_history: state.semantic_history.clone(),
                next_rng,
                sampling_ms: sampling_started.elapsed().as_secs_f64() * 1000.0,
            }));
        }
        let mut next_semantic_history = state.semantic_history.clone();
        next_semantic_history.push(semantic_token);
        if next_semantic_history.len() > 256 {
            let drain = next_semantic_history.len() - 256;
            next_semantic_history.drain(0..drain);
        }
        let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;
        let talker_hidden = state.last_hidden.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS last hidden state is not hydrated".into())
        })?;
        let text_addition = if state.frame_idx < state.trailing_text_len {
            state
                .trailing_text_hidden
                .as_ref()
                .ok_or_else(|| {
                    Error::InferenceError("Qwen3-TTS trailing text state is not hydrated".into())
                })?
                .i((.., state.frame_idx..state.frame_idx + 1, ..))?
        } else {
            state.tts_pad_embed.clone().ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS pad embedding is not hydrated".into())
            })?
        };
        Ok(PreparedTtsFrameStage::Predictor(
            PreparedTtsPredictorStage {
                semantic_token,
                semantic_embed,
                talker_hidden: talker_hidden.clone(),
                source_logits: last_logits.clone(),
                text_addition,
                expected_talker_context: state.talker_cache.context_len(),
                expected_frame_idx: state.frame_idx,
                expected_rng: state.rng,
                expected_semantic_history: state.semantic_history.clone(),
                expected_tensor_sequence: state.tensor_sequence,
                next_rng,
                next_semantic_history,
                sampling_ms: sampling_started.elapsed().as_secs_f64() * 1000.0,
            },
        ))
    }

    /// Run the independently batchable predictor/codebook stage.
    ///
    /// All rows must be at fresh invocation cursor zero and share exact model
    /// geometry. Predictor writes roll back to their entry cursors if any later
    /// acoustic embedding or step-input construction fails.
    pub fn tts_predictor_stage_batch_physical(
        &self,
        rows: Vec<PreparedTtsPredictorStage>,
        caches: &mut [&mut CodePredictorPhysicalCache],
    ) -> Result<Vec<PreparedTtsTalkerStage>> {
        if rows.is_empty() || rows.len() != caches.len() {
            return Err(Error::InvalidInput(
                "Qwen3-TTS predictor stage requires one non-empty cache row per prepared row"
                    .into(),
            ));
        }
        let talker_refs = rows
            .iter()
            .map(|row| &row.talker_hidden)
            .collect::<Vec<_>>();
        let semantic_refs = rows
            .iter()
            .map(|row| &row.semantic_embed)
            .collect::<Vec<_>>();
        let talker_hidden = Tensor::cat(&talker_refs, 0)?;
        let semantic_embeds = Tensor::cat(&semantic_refs, 0)?;
        let checkpoints = caches
            .iter()
            .map(|cache| cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let initial_cursors = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let predictor_started = Instant::now();
        let acoustic = self.code_predictor.generate_acoustic_codes_physical_batch(
            &talker_hidden,
            &semantic_embeds,
            caches,
        )?;
        let predictor_ms = predictor_started.elapsed().as_secs_f64() * 1000.0;
        let result = rows
            .into_iter()
            .zip(acoustic)
            .map(|(predictor, acoustic_codes)| {
                if acoustic_codes.len() != self.code_predictor.num_acoustic_groups() {
                    return Err(Error::InferenceError(format!(
                        "Qwen3-TTS predictor row returned {} acoustic groups, expected {}",
                        acoustic_codes.len(),
                        self.code_predictor.num_acoustic_groups()
                    )));
                }
                let acoustic_embed_sum = self
                    .code_predictor
                    .get_acoustic_embeddings_sum(&acoustic_codes)?;
                let step_input = predictor
                    .semantic_embed
                    .broadcast_add(&acoustic_embed_sum)?
                    .broadcast_add(&predictor.text_addition)?;
                Ok(PreparedTtsTalkerStage {
                    predictor,
                    acoustic_codes,
                    step_input,
                    predictor_ms,
                })
            })
            .collect::<Result<Vec<_>>>();
        if let Err(error) = result {
            let mut rollback_error = None;
            for (row, cache) in caches.iter_mut().enumerate() {
                if cache.context_len() != initial_cursors[row] {
                    if let Err(rollback) =
                        cache.restore_logical_checkpoint(checkpoints[row].clone())
                    {
                        rollback_error.get_or_insert(rollback);
                    }
                }
            }
            return if let Some(rollback) = rollback_error {
                Err(Error::InferenceError(format!(
                    "Qwen3-TTS predictor stage failed: {error}; rollback also failed: {rollback}"
                )))
            } else {
                Err(error)
            };
        }
        result
    }

    /// Commit one native ragged talker batch and then update row continuation
    /// tensors/history. Every fallible tensor operation completes before either
    /// the talker cursors or continuation fields become visible.
    pub fn tts_talker_stage_batch_physical(
        &self,
        states: &mut [&mut PhysicalTtsDecodeState],
        rows: Vec<PreparedTtsTalkerStage>,
    ) -> Result<Vec<TtsTalkerStageCompletion>> {
        Self::tts_talker_stage_batch_with_model(&self.talker, states, rows)
    }

    fn tts_talker_stage_batch_with_model(
        talker: &TalkerModel,
        states: &mut [&mut PhysicalTtsDecodeState],
        rows: Vec<PreparedTtsTalkerStage>,
    ) -> Result<Vec<TtsTalkerStageCompletion>> {
        if states.is_empty() || states.len() != rows.len() {
            return Err(Error::InvalidInput(
                "Qwen3-TTS talker stage requires matching non-empty state and predictor rows"
                    .into(),
            ));
        }
        let mut combined_tokens = Vec::with_capacity(rows.len());
        for (state, row) in states.iter().zip(&rows) {
            if state.finished
                || state.frame_idx != row.predictor.expected_frame_idx
                || state.offset != row.predictor.expected_talker_context
                || state.talker_cache.context_len() != row.predictor.expected_talker_context
                || state.rng.state != row.predictor.expected_rng.state
                || state.semantic_history != row.predictor.expected_semantic_history
                || state.tensor_sequence != row.predictor.expected_tensor_sequence
                || state
                    .last_hidden
                    .as_ref()
                    .is_none_or(|hidden| hidden.id() != row.predictor.talker_hidden.id())
                || state
                    .last_logits
                    .as_ref()
                    .is_none_or(|logits| logits.id() != row.predictor.source_logits.id())
                || state.all_code_groups.len() != row.acoustic_codes.len() + 1
            {
                return Err(Error::InvalidInput(
                    "Qwen3-TTS talker stage row no longer matches its prepared continuation".into(),
                ));
            }
            let mut tokens = Vec::with_capacity(row.acoustic_codes.len() + 1);
            tokens.push(
                state
                    .text_vocab_size
                    .checked_add(row.predictor.semantic_token)
                    .ok_or_else(|| {
                        Error::InvalidInput("Qwen3-TTS semantic token offset overflow".into())
                    })?,
            );
            for (acoustic_idx, group_token) in row.acoustic_codes.iter().enumerate() {
                let group_idx = u32::try_from(acoustic_idx + 1).map_err(|_| {
                    Error::InvalidInput("Qwen3-TTS acoustic group index overflow".into())
                })?;
                let offset = group_idx
                    .checked_mul(state.acoustic_vocab_size)
                    .and_then(|offset| offset.checked_add(state.text_vocab_size))
                    .and_then(|offset| offset.checked_add(*group_token))
                    .ok_or_else(|| {
                        Error::InvalidInput("Qwen3-TTS acoustic token offset overflow".into())
                    })?;
                tokens.push(offset);
            }
            combined_tokens.push(tokens);
        }
        let input_refs = rows.iter().map(|row| &row.step_input).collect::<Vec<_>>();
        let inputs = Tensor::cat(&input_refs, 0)?;
        let checkpoints = states
            .iter()
            .map(|state| state.talker_cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let initial_cursors = states
            .iter()
            .map(|state| state.talker_cache.context_len())
            .collect::<Vec<_>>();
        let talker_started = Instant::now();
        let output = {
            let mut caches = states
                .iter_mut()
                .map(|state| &mut state.talker_cache)
                .collect::<Vec<_>>();
            talker.generate_physical_step_batch_with_embeds(&inputs, &mut caches)?
        };
        let talker_ms = talker_started.elapsed().as_secs_f64() * 1000.0;
        let continuation_tensors = (0..states.len())
            .map(|row| {
                Ok((
                    output.hidden_states.i(row)?.unsqueeze(0)?,
                    output.logits.i(row)?.unsqueeze(0)?,
                ))
            })
            .collect::<Result<Vec<_>>>();
        let continuation_tensors = match continuation_tensors {
            Ok(tensors) => tensors,
            Err(error) => {
                let mut rollback_error = None;
                for (row, state) in states.iter_mut().enumerate() {
                    if state.talker_cache.context_len() != initial_cursors[row] {
                        if let Err(rollback) = state
                            .talker_cache
                            .restore_logical_checkpoint(checkpoints[row].clone())
                        {
                            rollback_error.get_or_insert(rollback);
                        }
                    }
                }
                return if let Some(rollback) = rollback_error {
                    Err(Error::InferenceError(format!(
                        "Qwen3-TTS talker continuation failed: {error}; rollback also failed: {rollback}"
                    )))
                } else {
                    Err(error)
                };
            }
        };

        let mut completions = Vec::with_capacity(states.len());
        for (((state, row), tokens), (new_hidden, new_logits)) in states
            .iter_mut()
            .zip(rows)
            .zip(combined_tokens)
            .zip(continuation_tensors)
        {
            state.rng = row.predictor.next_rng;
            state.semantic_history = row.predictor.next_semantic_history;
            for (group, token) in state.all_code_groups.iter_mut().zip(tokens) {
                group.push(token);
            }
            state.last_hidden = Some(new_hidden);
            state.last_logits = Some(new_logits);
            state.frame_idx += 1;
            state.offset = state.talker_cache.context_len();
            completions.push(TtsTalkerStageCompletion {
                sampling_ms: row.predictor.sampling_ms,
                predictor_ms: row.predictor_ms,
                talker_ms,
            });
        }
        Ok(completions)
    }

    /// Perform codec emission after a successful talker transaction.
    pub fn finish_tts_talker_stage_physical(
        &self,
        state: &mut PhysicalTtsDecodeState,
        completion: TtsTalkerStageCompletion,
    ) -> Result<TtsDecodeStep> {
        let codec_started = Instant::now();
        let mut samples = self.collect_incremental_audio_physical(state, false)?;
        if state.frame_idx >= state.max_frames {
            let final_samples = self.collect_incremental_audio_physical(state, true)?;
            state.finished = true;
            samples.extend(final_samples);
        }
        let codec_ms = codec_started.elapsed().as_secs_f64() * 1000.0;
        Ok(TtsDecodeStep {
            samples,
            frames_generated: state.frames_generated(),
            finished: state.finished,
            sampling_ms: completion.sampling_ms,
            decode_ms: completion.predictor_ms + completion.talker_ms,
            codec_ms,
            executed_model_row: true,
        })
    }

    /// Finalize a terminal row without touching predictor or talker KV.
    pub fn finish_tts_terminal_stage_physical(
        &self,
        state: &mut PhysicalTtsDecodeState,
        terminal: PreparedTtsTerminalStage,
    ) -> Result<TtsDecodeStep> {
        if state.frame_idx != terminal.expected_frame_idx
            || state.talker_cache.context_len() != terminal.expected_talker_context
            || state.offset != terminal.expected_talker_context
            || state.rng.state != terminal.expected_rng.state
            || state.semantic_history != terminal.expected_semantic_history
        {
            return Err(Error::InvalidInput(
                "Qwen3-TTS terminal row no longer matches its prepared continuation".into(),
            ));
        }
        if matches!(terminal.reason, TtsTerminalReason::AlreadyFinished) {
            return Ok(TtsDecodeStep {
                samples: Vec::new(),
                frames_generated: state.frames_generated(),
                finished: true,
                sampling_ms: 0.0,
                decode_ms: 0.0,
                codec_ms: 0.0,
                executed_model_row: false,
            });
        }
        let codec_started = Instant::now();
        let samples = self.collect_incremental_audio_physical(state, true)?;
        if matches!(terminal.reason, TtsTerminalReason::SemanticEos) {
            state.rng = terminal.next_rng;
        }
        state.finished = true;
        Ok(TtsDecodeStep {
            samples,
            frames_generated: state.frames_generated(),
            finished: true,
            sampling_ms: terminal.sampling_ms,
            decode_ms: 0.0,
            codec_ms: codec_started.elapsed().as_secs_f64() * 1000.0,
            executed_model_row: false,
        })
    }

    /// Execute a changing set of TTS rows while preserving input/output order.
    /// Terminal/EOS rows are finalized independently, live rows form one native
    /// predictor/talker batch, and a single live row uses scalar model kernels.
    pub fn tts_decode_step_batch_physical(
        &self,
        states: &mut [&mut PhysicalTtsDecodeState],
        predictor_caches: &mut [&mut CodePredictorPhysicalCache],
    ) -> Result<Vec<TtsDecodeStep>> {
        if states.is_empty() || states.len() != predictor_caches.len() {
            return Err(Error::InvalidInput(
                "Qwen3-TTS decode batch requires matching non-empty state and predictor rows"
                    .into(),
            ));
        }
        if states.len() == 1 {
            return self
                .tts_decode_step_physical(states[0], predictor_caches[0])
                .map(|step| vec![step]);
        }
        run_tts_decode_batch_transaction(states, predictor_caches, |states, predictor_caches| {
            let prepared = states
                .iter()
                .map(|state| self.prepare_tts_predictor_stage(state))
                .collect::<Result<Vec<_>>>()?;
            let row_count = states.len();
            let mut live_mask = vec![false; row_count];
            let mut live_indices = Vec::new();
            let mut predictor_rows = Vec::new();
            let mut terminal_rows = (0..row_count).map(|_| None).collect::<Vec<_>>();
            for (row, stage) in prepared.into_iter().enumerate() {
                match stage {
                    PreparedTtsFrameStage::Predictor(stage) => {
                        live_mask[row] = true;
                        live_indices.push(row);
                        predictor_rows.push(stage);
                    }
                    PreparedTtsFrameStage::Terminal(stage) => {
                        // Scalar EOS sampling validates its fresh invocation
                        // workspace before deciding not to use it. Preserve that
                        // contract, while frame-limit/already-finished rows remain
                        // workspace-free exactly like the scalar path.
                        if matches!(stage.reason, TtsTerminalReason::SemanticEos) {
                            self.code_predictor
                                .validate_physical_workspace(predictor_caches[row])?;
                        }
                        terminal_rows[row] = Some(stage);
                    }
                }
            }
            let mut results = vec![None; row_count];
            if !predictor_rows.is_empty() {
                let mut live_caches = predictor_caches
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(row, cache)| live_mask[row].then_some(&mut **cache))
                    .collect::<Vec<_>>();
                let talker_rows =
                    self.tts_predictor_stage_batch_physical(predictor_rows, &mut live_caches)?;
                let mut live_states = states
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(row, state)| live_mask[row].then_some(&mut **state))
                    .collect::<Vec<_>>();
                let completions =
                    self.tts_talker_stage_batch_physical(&mut live_states, talker_rows)?;
                for ((row, state), completion) in live_indices
                    .iter()
                    .copied()
                    .zip(live_states.iter_mut())
                    .zip(completions)
                {
                    results[row] = Some(self.finish_tts_talker_stage_physical(state, completion)?);
                }
                drop(live_states);
                drop(live_caches);
            }
            for (row, (state, terminal)) in states.iter_mut().zip(terminal_rows).enumerate() {
                if let Some(terminal) = terminal {
                    results[row] = Some(self.finish_tts_terminal_stage_physical(state, terminal)?);
                }
            }
            results
                .into_iter()
                .map(|step| {
                    step.ok_or_else(|| {
                        Error::InferenceError("Qwen3-TTS batch omitted an output row".into())
                    })
                })
                .collect()
        })
    }

    /// Execute one physical TTS quantum using a fresh predictor workspace.
    ///
    /// The retained talker cache lives in `state`; `predictor_cache` must be a
    /// cursor-zero invocation cache reserved for this frame. Predictor pages
    /// are never stored in the returned state and may be released immediately
    /// after this call, including terminal EOS calls where they remain unused.
    pub fn tts_decode_step_physical(
        &self,
        state: &mut PhysicalTtsDecodeState,
        predictor_cache: &mut CodePredictorPhysicalCache,
    ) -> Result<TtsDecodeStep> {
        if state.finished {
            return Ok(TtsDecodeStep {
                samples: Vec::new(),
                frames_generated: state.frames_generated(),
                finished: true,
                sampling_ms: 0.0,
                decode_ms: 0.0,
                codec_ms: 0.0,
                executed_model_row: false,
            });
        }

        if state.frame_idx >= state.max_frames {
            let codec_started = Instant::now();
            let final_samples = self.collect_incremental_audio_physical(state, true)?;
            state.finished = true;
            return Ok(TtsDecodeStep {
                samples: final_samples,
                frames_generated: state.frames_generated(),
                finished: true,
                sampling_ms: 0.0,
                decode_ms: 0.0,
                codec_ms: codec_started.elapsed().as_secs_f64() * 1000.0,
                executed_model_row: false,
            });
        }
        if state.offset != state.talker_cache.context_len() {
            return Err(Error::InferenceError(format!(
                "Qwen3-TTS physical state offset {} diverged from retained talker cursor {}",
                state.offset,
                state.talker_cache.context_len()
            )));
        }
        self.code_predictor
            .validate_physical_workspace(predictor_cache)?;

        let step_start = Instant::now();
        let allow_eos = qwen_tts_allows_eos(state.frames_generated());
        let semantic_start = Instant::now();
        let mut next_rng = state.rng;
        let last_logits = state.last_logits.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS last logits are not hydrated".into())
        })?;
        let semantic_token = sample_semantic(
            &last_logits.i((0, 0))?,
            state.semantic_vocab_size,
            self.specials.codec_eos_token_id,
            allow_eos,
            &state.params,
            &state.semantic_history,
            &mut next_rng,
            qwen_tts_uses_device_sampling(&self.device),
        )?;
        let semantic_ms = semantic_start.elapsed().as_secs_f64() * 1000.0;

        if allow_eos && semantic_token == self.specials.codec_eos_token_id {
            debug!(
                frames_generated = state.frames_generated(),
                device = ?self.device.kind,
                "Qwen3-TTS physical decode reached semantic EOS"
            );
            let codec_started = Instant::now();
            let final_samples = self.collect_incremental_audio_physical(state, true)?;
            state.rng = next_rng;
            state.finished = true;
            return Ok(TtsDecodeStep {
                samples: final_samples,
                frames_generated: state.frames_generated(),
                finished: true,
                sampling_ms: semantic_ms,
                decode_ms: 0.0,
                codec_ms: codec_started.elapsed().as_secs_f64() * 1000.0,
                executed_model_row: false,
            });
        }

        let mut next_semantic_history = state.semantic_history.clone();
        next_semantic_history.push(semantic_token);
        if next_semantic_history.len() > 256 {
            let drain = next_semantic_history.len() - 256;
            next_semantic_history.drain(0..drain);
        }

        let predictor_start = Instant::now();
        let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;
        let last_hidden = state.last_hidden.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3-TTS last hidden state is not hydrated".into())
        })?;
        let acoustic_codes = self.code_predictor.generate_acoustic_codes_physical(
            last_hidden,
            &semantic_embed,
            predictor_cache,
        )?;
        let acoustic_embed_sum = self
            .code_predictor
            .get_acoustic_embeddings_sum(&acoustic_codes)?;
        let predictor_ms = predictor_start.elapsed().as_secs_f64() * 1000.0;

        let text_addition = if state.frame_idx < state.trailing_text_len {
            state
                .trailing_text_hidden
                .as_ref()
                .ok_or_else(|| {
                    Error::InferenceError("Qwen3-TTS trailing text state is not hydrated".into())
                })?
                .i((.., state.frame_idx..state.frame_idx + 1, ..))?
        } else {
            state.tts_pad_embed.clone().ok_or_else(|| {
                Error::InferenceError("Qwen3-TTS pad embedding is not hydrated".into())
            })?
        };
        let step_input = semantic_embed
            .broadcast_add(&acoustic_embed_sum)?
            .broadcast_add(&text_addition)?;

        let talker_start = Instant::now();
        let (new_hidden, new_logits) = self
            .talker
            .generate_physical_step_with_embed(&step_input, &mut state.talker_cache)?;
        let talker_ms = talker_start.elapsed().as_secs_f64() * 1000.0;

        state.rng = next_rng;
        state.semantic_history = next_semantic_history;
        state.all_code_groups[0].push(state.text_vocab_size + semantic_token);
        for (acoustic_idx, &group_token) in acoustic_codes.iter().enumerate() {
            let group_idx = acoustic_idx + 1;
            if group_idx < state.all_code_groups.len() {
                let combined_token = state.text_vocab_size
                    + group_token
                    + (group_idx as u32 * state.acoustic_vocab_size);
                state.all_code_groups[group_idx].push(combined_token);
            }
        }
        state.last_hidden = Some(new_hidden);
        state.last_logits = Some(new_logits);
        state.frame_idx += 1;
        state.offset = state.talker_cache.context_len();

        let audio_start = Instant::now();
        let mut samples = self.collect_incremental_audio_physical(state, false)?;
        if state.frame_idx >= state.max_frames {
            let final_samples = self.collect_incremental_audio_physical(state, true)?;
            state.finished = true;
            samples.extend(final_samples);
        }
        let audio_ms = audio_start.elapsed().as_secs_f64() * 1000.0;

        if self.device.kind.is_cuda() {
            debug!(
                frame_idx = state.frame_idx,
                semantic_ms,
                predictor_ms,
                talker_ms,
                audio_ms,
                total_ms = step_start.elapsed().as_secs_f64() * 1000.0,
                emitted_samples = samples.len(),
                "Qwen3-TTS physical CUDA decode step timings"
            );
        }

        let total_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        Ok(TtsDecodeStep {
            samples,
            frames_generated: state.frames_generated(),
            finished: state.finished,
            sampling_ms: semantic_ms,
            decode_ms: (total_ms - semantic_ms - audio_ms).max(0.0),
            codec_ms: audio_ms,
            executed_model_row: true,
        })
    }

    fn resolve_language_id(&self, language: Option<&str>) -> Option<u32> {
        let normalized = language.map(str::trim).filter(|s| !s.is_empty());
        match normalized {
            Some(lang) if lang.eq_ignore_ascii_case("auto") => None,
            Some(lang) => Some(self.tokenizer.get_language_id(lang)),
            None => None,
        }
    }

    fn encode_instruction_ids(&self, instruct: Option<&str>) -> Result<Option<Vec<u32>>> {
        let Some(text) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };
        // Align instruction prompt shape with upstream VoiceDesign/CustomVoice API.
        let prompt = format!("<|im_start|>user\n{text}<|im_end|>\n");
        let ids = self.tokenizer.encode_text(&prompt, None)?;
        if ids.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ids))
        }
    }

    fn encode_assistant_prompt_ids(&self, text: &str) -> Result<Vec<u32>> {
        // Mirror upstream prompting:
        // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let prompt = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        self.tokenizer.encode_text(&prompt, None)
    }

    fn encode_reference_prompt_ids(&self, reference_text: &str) -> Result<Vec<u32>> {
        // Mirror upstream voice-clone reference prompt:
        // <|im_start|>assistant\n{reference_text}<|im_end|>\n
        let prompt = format!("<|im_start|>assistant\n{reference_text}<|im_end|>\n");
        self.tokenizer.encode_text(&prompt, None)
    }

    fn build_voice_clone_icl_embeddings(
        &self,
        target_text_ids: &[u32],
        reference_text_ids: &[u32],
        ref_codec_tokens: &[Vec<u32>],
        tts_pad_embed: &Tensor,
        tts_eos_embed: &Tensor,
        non_streaming_mode: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut all_text_ids = Vec::with_capacity(reference_text_ids.len() + target_text_ids.len());
        all_text_ids.extend_from_slice(reference_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        let text_embed = self.talker.get_projected_text_embeddings(&all_text_ids)?;
        let text_embed = Tensor::cat(&[&text_embed, tts_eos_embed], 1)?;

        let codec_embed = self.build_ref_codec_embeddings(ref_codec_tokens)?;

        let text_lens = text_embed.dim(1)?;
        let codec_lens = codec_embed.dim(1)?;
        if codec_lens == 0 {
            return Err(Error::ModelError(
                "Reference codec conditioning is empty".to_string(),
            ));
        }

        if non_streaming_mode {
            let codec_pad_ids = vec![self.specials.codec_pad_id; text_lens];
            let codec_pad_embed = self.talker.get_codec_embedding_batch(&codec_pad_ids)?;
            let icl_input = text_embed.broadcast_add(&codec_pad_embed)?;
            let icl_input =
                Tensor::cat(&[&icl_input, &codec_embed.broadcast_add(tts_pad_embed)?], 1)?;
            return Ok((icl_input, tts_pad_embed.clone()));
        }

        if text_lens > codec_lens {
            let text_prefix = text_embed.i((.., ..codec_lens, ..))?;
            let trailing = text_embed.i((.., codec_lens.., ..))?;
            let icl_input = text_prefix.broadcast_add(&codec_embed)?;
            Ok((icl_input, trailing))
        } else {
            let mut padded_parts: Vec<Tensor> = vec![text_embed];
            for _ in 0..codec_lens.saturating_sub(text_lens) {
                padded_parts.push(tts_pad_embed.clone());
            }
            let padded_refs: Vec<&Tensor> = padded_parts.iter().collect();
            let padded_text = Tensor::cat(&padded_refs, 1)?;
            let icl_input = padded_text.broadcast_add(&codec_embed)?;
            Ok((icl_input, tts_pad_embed.clone()))
        }
    }

    fn build_ref_codec_embeddings(&self, ref_codec_tokens: &[Vec<u32>]) -> Result<Tensor> {
        let num_code_groups = self.config.talker_config.num_code_groups;
        if ref_codec_tokens.len() < num_code_groups {
            return Err(Error::InvalidInput(format!(
                "Reference codec groups mismatch: got {}, expected at least {}",
                ref_codec_tokens.len(),
                num_code_groups
            )));
        }

        let num_acoustic_groups = self.code_predictor.num_acoustic_groups();
        let usable_groups = (1 + num_acoustic_groups).min(num_code_groups);
        let mut frame_len = usize::MAX;
        for group in ref_codec_tokens.iter().take(usable_groups) {
            frame_len = frame_len.min(group.len());
        }
        if frame_len == usize::MAX || frame_len == 0 {
            return Err(Error::ModelError(
                "Reference codec conditioning has no usable frames".to_string(),
            ));
        }

        frame_len = frame_len.min(MAX_VOICE_CLONE_REFERENCE_FRAMES);

        let codec_vocab = self.tokenizer.codec_vocab_size() as u32;
        let mut semantic_codes = Vec::with_capacity(frame_len);
        let mut acoustic_embed_steps = Vec::with_capacity(frame_len);

        for frame_idx in 0..frame_len {
            semantic_codes.push(ref_codec_tokens[0][frame_idx] % codec_vocab);

            let mut acoustic_codes = Vec::with_capacity(num_acoustic_groups);
            for group_idx in 0..num_acoustic_groups {
                let source_group = group_idx + 1;
                let code = ref_codec_tokens
                    .get(source_group)
                    .and_then(|group| group.get(frame_idx))
                    .copied()
                    .unwrap_or(0)
                    % codec_vocab;
                acoustic_codes.push(code);
            }
            acoustic_embed_steps.push(
                self.code_predictor
                    .get_acoustic_embeddings_sum(&acoustic_codes)?,
            );
        }

        let semantic_embed = self.talker.get_codec_embedding_batch(&semantic_codes)?;
        let acoustic_refs: Vec<&Tensor> = acoustic_embed_steps.iter().collect();
        let acoustic_embed = Tensor::cat(&acoustic_refs, 1)?;
        let codec_embed = semantic_embed.broadcast_add(&acoustic_embed)?;

        let codec_bos = self
            .talker
            .get_codec_embedding_batch(&[self.specials.codec_bos_id])?;
        Tensor::cat(&[&codec_bos, &codec_embed], 1).map_err(Error::from)
    }

    /// Bind immutable prepared embeddings to a fresh retained talker cache.
    /// No transformer work occurs until [`Self::continue_physical_prefill`].
    pub fn begin_physical_prefill(
        &self,
        prepared: PreparedTtsDecodePrefill,
        stream_config: TtsStreamingConfig,
        talker_cache: TalkerPhysicalCache,
    ) -> Result<PhysicalTtsPrefillState> {
        if talker_cache.context_len() != 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical prefill requires a fresh talker cursor, got {}",
                talker_cache.context_len()
            )));
        }
        let acoustic_groups = self.code_predictor.num_acoustic_groups();
        if acoustic_groups == 0 {
            return Err(Error::InvalidInput(
                "Qwen3-TTS physical prefill requires at least one acoustic group".to_string(),
            ));
        }
        let expected_code_groups = acoustic_groups.saturating_add(1);
        if self.config.talker_config.num_code_groups != expected_code_groups {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical prefill has {} codec groups, expected {expected_code_groups}",
                self.config.talker_config.num_code_groups
            )));
        }
        let total_tokens = prepared.prefill_tokens()?;
        if total_tokens == 0 {
            return Err(Error::InvalidInput(
                "Qwen3-TTS prepared prefill cannot be empty".into(),
            ));
        }
        Ok(PhysicalTtsPrefillState {
            prepared,
            talker_cache,
            stream_config,
            progress: 0,
            total_tokens,
            last_hidden: None,
            last_logits: None,
            tensor_sequence: None,
        })
    }

    /// Execute one exact prepared-embedding span. The logical span cursor and
    /// physical talker cursor must agree; a failed span changes neither.
    pub fn continue_physical_prefill(
        &self,
        state: &mut PhysicalTtsPrefillState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.progress != span_start
            || span_start >= span_end
            || span_end > state.total_tokens
            || state.last_hidden.is_some()
            || state.last_logits.is_some()
            || state.talker_cache.context_len() != span_start
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS resumable prefill span [{span_start},{span_end}) is incompatible with cursor {} and total {}",
                state.progress, state.total_tokens
            )));
        }
        let span = state
            .prepared
            .prefill_embeds
            .narrow(1, span_start, span_end - span_start)?;
        let complete = span_end == state.total_tokens;
        let prefill_start = Instant::now();
        let checkpoint = state.talker_cache.logical_checkpoint();
        let transition = (|| {
            let output = self.talker.prefill_physical_span_with_embeds(
                &span,
                span_start,
                &mut state.talker_cache,
                None,
                complete,
            )?;
            if state.talker_cache.context_len() != span_end {
                return Err(Error::InferenceError(format!(
                    "Qwen3-TTS physical prefill ended at cursor {}, expected {span_end}",
                    state.talker_cache.context_len()
                )));
            }
            match (complete, output) {
                (true, Some(continuation)) => Ok(Some(continuation)),
                (false, None) => Ok(None),
                _ => Err(Error::InferenceError(
                    "Qwen3-TTS prefill span returned inconsistent continuation tensors".into(),
                )),
            }
        })();
        let continuation = match transition {
            Ok(continuation) => continuation,
            Err(error) => {
                if state.talker_cache.context_len() != span_start {
                    state.talker_cache.restore_logical_checkpoint(checkpoint)?;
                }
                return Err(error);
            }
        };
        if let Some((last_hidden, last_logits)) = continuation {
            state.last_hidden = Some(last_hidden);
            state.last_logits = Some(last_logits);
        }
        state.progress = span_end;
        if self.device.kind.is_cuda() {
            debug!(
                span_start,
                span_end,
                complete,
                talker_dtype = ?self.dtype,
                predictor_dtype = ?self.code_predictor_dtype,
                speech_tokenizer_dtype = ?self.speech_tokenizer_dtype,
                prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
                "Qwen3-TTS physical CUDA prefill timings"
            );
        }
        Ok(complete)
    }

    /// Consume a completed resumable prefill and materialize decode-only
    /// continuation state. No codec/vocoder work occurs at this boundary.
    pub fn finish_physical_prefill(
        &self,
        state: PhysicalTtsPrefillState,
    ) -> Result<PhysicalTtsDecodeState> {
        if !state.is_complete() || state.talker_cache.context_len() != state.total_tokens {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS cannot finish prefill at logical/physical cursors {}/{} of {}",
                state.progress,
                state.talker_cache.context_len(),
                state.total_tokens
            )));
        }
        let PhysicalTtsPrefillState {
            prepared,
            talker_cache,
            stream_config,
            total_tokens,
            last_hidden,
            last_logits,
            ..
        } = state;
        let trailing_text_len = prepared.trailing_text_hidden.dim(1)?;
        Ok(PhysicalTtsDecodeState {
            talker_cache,
            text_vocab_size: self.tokenizer.text_vocab_size() as u32,
            acoustic_vocab_size: self.tokenizer.codec_vocab_size() as u32,
            semantic_vocab_size: (self.config.talker_config.vocab_size as u32).saturating_sub(1024),
            trailing_text_hidden: Some(prepared.trailing_text_hidden),
            retained_sequence_memory: Some(prepared.retained_sequence_memory),
            prefill_tokens: total_tokens,
            trailing_text_len,
            tts_pad_embed: Some(prepared.tts_pad_embed),
            max_frames: prepared.params.max_frames.max(1),
            frame_idx: 0,
            offset: total_tokens,
            all_code_groups: vec![Vec::new(); self.config.talker_config.num_code_groups],
            semantic_history: Vec::new(),
            last_hidden,
            last_logits,
            tensor_sequence: None,
            rng: SimpleRng::new(),
            params: prepared.params,
            stream_config,
            emitted_frames: 0,
            emitted_samples: 0,
            decode_raw_token_scratch: Vec::new(),
            finished: false,
        })
    }

    fn collect_incremental_audio_physical(
        &self,
        state: &mut PhysicalTtsDecodeState,
        force: bool,
    ) -> Result<Vec<f32>> {
        self.collect_incremental_audio_parts(
            &state.all_code_groups,
            state.stream_config,
            &mut state.emitted_frames,
            &mut state.emitted_samples,
            &mut state.decode_raw_token_scratch,
            force,
        )
    }

    fn collect_incremental_audio_parts(
        &self,
        codec_groups: &[Vec<u32>],
        stream_config: TtsStreamingConfig,
        emitted_frames: &mut usize,
        emitted_samples: &mut usize,
        decode_raw_token_scratch: &mut Vec<Vec<u32>>,
        force: bool,
    ) -> Result<Vec<f32>> {
        let total_frames = codec_groups.first().map(Vec::len).unwrap_or(0);
        if total_frames == 0 {
            return Ok(Vec::new());
        }

        if !force {
            if total_frames < stream_config.min_frames_before_stream {
                return Ok(Vec::new());
            }
            let newly_generated = total_frames.saturating_sub(*emitted_frames);
            if newly_generated < stream_config.decode_interval_frames {
                return Ok(Vec::new());
            }
        }

        let lookahead = if force {
            0
        } else {
            stream_config.decode_lookahead_frames
        };
        let target_frames = total_frames.saturating_sub(lookahead);
        if target_frames <= *emitted_frames {
            return Ok(Vec::new());
        }

        for group in codec_groups {
            if group.len() < target_frames {
                return Ok(Vec::new());
            }
        }

        if self.device.kind.is_cuda() && !force && qwen_tts_cuda_chunked_codec_stream_enabled() {
            let (samples, next_emitted_frames, next_emitted_samples) = self
                .decode_cuda_stream_chunk(
                    codec_groups,
                    *emitted_frames,
                    *emitted_samples,
                    target_frames,
                    stream_config,
                    decode_raw_token_scratch,
                )?;
            *emitted_frames = next_emitted_frames;
            *emitted_samples = next_emitted_samples;
            return Ok(samples);
        }

        self.fill_raw_codec_scratch(codec_groups, target_frames, decode_raw_token_scratch)?;
        let decoded = self.decode_raw_codec_tokens(decode_raw_token_scratch)?;
        if decoded.len() <= *emitted_samples {
            *emitted_frames = target_frames;
            return Ok(Vec::new());
        }

        let new_samples = decoded[*emitted_samples..].to_vec();
        *emitted_samples = decoded.len();
        *emitted_frames = target_frames;
        Ok(new_samples)
    }

    fn build_conditioned_prefill_embeddings(
        &self,
        prompt_ids: &[u32],
        speaker_id: Option<u32>,
        language_id: Option<u32>,
        instruct_ids: Option<&[u32]>,
    ) -> Result<Tensor> {
        let role_prefix = self.talker.get_projected_text_embeddings(&[
            self.specials.im_start_token_id,
            self.specials.assistant_token_id,
            NEWLINE_TOKEN_ID,
        ])?;

        let mut codec_prefix_ids = Vec::new();
        if let Some(language_id) = language_id {
            codec_prefix_ids.extend_from_slice(&[
                self.specials.codec_think_id,
                self.specials.codec_think_bos_id,
                language_id,
                self.specials.codec_think_eos_id,
            ]);
        } else {
            codec_prefix_ids.extend_from_slice(&[
                self.specials.codec_nothink_id,
                self.specials.codec_think_bos_id,
                self.specials.codec_think_eos_id,
            ]);
        }
        if let Some(speaker_id) = speaker_id {
            codec_prefix_ids.push(speaker_id);
        }
        codec_prefix_ids.push(self.specials.codec_pad_id);
        codec_prefix_ids.push(self.specials.codec_bos_id);

        let prefix_len = codec_prefix_ids.len();
        if prefix_len < 2 {
            return Err(Error::InferenceError(
                "Invalid codec prefix while building conditioned prefill".to_string(),
            ));
        }

        let codec_prefix = self.talker.get_codec_embedding_batch(&codec_prefix_ids)?;
        let codec_without_last = codec_prefix.i((.., ..prefix_len - 1, ..))?;

        let mut tts_overlay_ids = vec![self.specials.tts_pad_token_id; prefix_len - 2];
        tts_overlay_ids.push(self.specials.tts_bos_token_id);
        let tts_overlay = self
            .talker
            .get_projected_text_embeddings(&tts_overlay_ids)?;
        let codec_hidden = tts_overlay.broadcast_add(&codec_without_last)?;

        let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;

        if let Some(instruct_ids) = instruct_ids {
            if !instruct_ids.is_empty() {
                let instruct_hidden = self.talker.get_projected_text_embeddings(instruct_ids)?;
                hidden = Tensor::cat(&[&instruct_hidden, &hidden], 1)?;
            }
        }

        let first_text_id = if prompt_ids.len() > 3 {
            Some(prompt_ids[3])
        } else {
            prompt_ids.first().copied()
        };

        if let Some(first_text_id) = first_text_id {
            let first_text_proj = self
                .talker
                .get_projected_text_embeddings(&[first_text_id])?;
            let codec_bos_embed = codec_prefix.i((.., prefix_len - 1..prefix_len, ..))?;
            let first_combined = first_text_proj.broadcast_add(&codec_bos_embed)?;
            hidden = Tensor::cat(&[&hidden, &first_combined], 1)?;
        }

        Ok(hidden)
    }

    fn build_trailing_text_embeddings_from_prompt(
        &self,
        prompt_ids: &[u32],
        max_frames: usize,
    ) -> Result<(Tensor, usize, Tensor)> {
        // Upstream uses prompt layout:
        // [role(3), first_text(1), trailing_text(...), trailer(5)]
        let trailing_ids: &[u32] = if prompt_ids.len() > 9 {
            &prompt_ids[4..prompt_ids.len() - 5]
        } else if prompt_ids.len() > 4 {
            &prompt_ids[4..]
        } else {
            &[]
        };

        let trailing = if !trailing_ids.is_empty() {
            let trailing_end = max_frames.min(trailing_ids.len());
            let remaining = self
                .talker
                .get_projected_text_embeddings(&trailing_ids[..trailing_end])?;
            let eos_embed = self
                .talker
                .get_projected_special_embed(self.specials.tts_eos_token_id)?;
            Tensor::cat(&[&remaining, &eos_embed], 1)?
        } else {
            self.talker
                .get_projected_special_embed(self.specials.tts_eos_token_id)?
        };
        let trailing_len = trailing.dim(1)?;
        let tts_pad = self
            .talker
            .get_projected_special_embed(self.specials.tts_pad_token_id)?;
        Ok((trailing, trailing_len, tts_pad))
    }

    /// Encode reference audio to codec tokens for voice cloning
    fn encode_reference_audio(&self, reference: &SpeakerReference) -> Result<Vec<Vec<u32>>> {
        let codec_tokens = self
            .speech_tokenizer
            .encode_reference_audio(&reference.audio_samples, reference.sample_rate)?;
        let num_groups = codec_tokens.len();
        let num_frames = codec_tokens.first().map(|g| g.len()).unwrap_or(0);
        debug!(
            "Reference audio encoded for voice cloning ({} groups x {} frames, transcript chars: {})",
            num_groups,
            num_frames,
            reference.text.len()
        );
        Ok(codec_tokens)
    }

    /// Convert codec tokens to audio waveform
    fn codec_to_audio(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        let target_frames = codec_tokens.first().map(|g| g.len()).unwrap_or(0);
        let mut raw_codec_tokens: Vec<Vec<u32>> = Vec::new();
        self.fill_raw_codec_scratch(codec_tokens, target_frames, &mut raw_codec_tokens)?;
        self.decode_raw_codec_tokens(&raw_codec_tokens)
    }

    fn decode_cuda_stream_chunk(
        &self,
        codec_tokens: &[Vec<u32>],
        emitted_frames: usize,
        emitted_samples: usize,
        target_frames: usize,
        stream_config: TtsStreamingConfig,
        scratch: &mut Vec<Vec<u32>>,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let context_frames = emitted_frames.min(
            stream_config
                .decode_lookahead_frames
                .max(stream_config.decode_interval_frames)
                .max(4),
        );
        let start_frame = emitted_frames.saturating_sub(context_frames);
        let chunk_frames = target_frames.saturating_sub(start_frame);
        if chunk_frames == 0 {
            return Ok((Vec::new(), emitted_frames, emitted_samples));
        }

        self.fill_raw_codec_scratch_range(codec_tokens, start_frame, target_frames, scratch)?;
        let decoded = self.decode_raw_codec_tokens(scratch)?;
        let skip_samples = decoded
            .len()
            .saturating_mul(emitted_frames.saturating_sub(start_frame))
            / chunk_frames;
        let new_samples = decoded.get(skip_samples..).unwrap_or(&[]).to_vec();
        let emitted_samples = emitted_samples.saturating_add(new_samples.len());
        Ok((new_samples, target_frames, emitted_samples))
    }

    fn fill_raw_codec_scratch(
        &self,
        codec_tokens: &[Vec<u32>],
        target_frames: usize,
        scratch: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        self.fill_raw_codec_scratch_range(codec_tokens, 0, target_frames, scratch)
    }

    fn fill_raw_codec_scratch_range(
        &self,
        codec_tokens: &[Vec<u32>],
        start_frame: usize,
        end_frame: usize,
        scratch: &mut Vec<Vec<u32>>,
    ) -> Result<()> {
        if codec_tokens.is_empty() || end_frame <= start_frame {
            scratch.clear();
            return Ok(());
        }

        let target_frames = end_frame - start_frame;
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let codec_vocab_size = self.tokenizer.codec_vocab_size() as u32;

        if scratch.len() != codec_tokens.len() {
            scratch.resize_with(codec_tokens.len(), Vec::new);
        }

        for (group_idx, group_tokens) in codec_tokens.iter().enumerate() {
            if group_tokens.len() < end_frame {
                return Err(Error::InvalidInput(
                    "Insufficient codec frames for requested decode slice".to_string(),
                ));
            }
            let raw_tokens = &mut scratch[group_idx];
            raw_tokens.clear();
            raw_tokens.reserve(target_frames);

            for &token in group_tokens.iter().take(end_frame).skip(start_frame) {
                raw_tokens.push(raw_codec_token(
                    token,
                    group_idx,
                    text_vocab_size,
                    codec_vocab_size,
                ));
            }
        }
        Ok(())
    }

    fn decode_raw_codec_tokens(&self, raw_codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        if raw_codec_tokens.is_empty() || raw_codec_tokens[0].is_empty() {
            return Ok(Vec::new());
        }
        let mut audio = self.speech_tokenizer.decode(raw_codec_tokens)?;
        normalize_audio(&mut audio);
        Ok(audio)
    }

    /// List available preset speakers
    pub fn available_speakers(&self) -> Vec<&String> {
        self.tokenizer.available_speakers()
    }

    /// List available languages
    pub fn available_languages(&self) -> Vec<&String> {
        self.tokenizer.available_languages()
    }

    /// Get the model configuration
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }
}

fn raw_codec_token(
    token: u32,
    group_idx: usize,
    text_vocab_size: u32,
    codec_vocab_size: u32,
) -> u32 {
    if group_idx == 0 {
        if token >= text_vocab_size {
            token - text_vocab_size
        } else {
            token
        }
    } else {
        let offset = text_vocab_size + (group_idx as u32 * codec_vocab_size);
        if token >= offset {
            token - offset
        } else {
            token
        }
    }
}

pub(crate) fn qwen_tts_cuda_chunked_codec_stream_enabled() -> bool {
    qwen_tts_cuda_chunked_codec_stream_enabled_from(
        std::env::var(ENV_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM)
            .ok()
            .as_deref(),
    )
}

fn qwen_tts_cuda_chunked_codec_stream_enabled_from(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Argmax sampling for greedy decoding
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS logits rank for argmax: {rank}"
            )))
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

fn logit_scalar_f32(logits: &Tensor, idx: u32) -> Result<f32> {
    logits
        .i(idx as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
        .map_err(Error::from)
}

fn argmax_semantic(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };

    let vocab_len = logits.dim(0)?;
    let semantic_len = (semantic_vocab_size as usize).min(vocab_len);
    let mut best_idx = if semantic_len > 0 {
        let semantic_logits = logits.narrow(0, 0, semantic_len)?;
        Some(argmax(&semantic_logits)?)
    } else {
        None
    };
    let best_val = if let Some(idx) = best_idx {
        logit_scalar_f32(&logits, idx)?
    } else {
        f32::NEG_INFINITY
    };

    let eos_idx = eos_token_id as usize;
    if allow_eos && eos_idx < vocab_len && eos_idx >= semantic_len {
        let eos_val = logit_scalar_f32(&logits, eos_token_id)?;
        if eos_val > best_val {
            best_idx = Some(eos_token_id);
        }
    }

    if let Some(idx) = best_idx {
        Ok(idx)
    } else if allow_eos {
        Ok(eos_token_id)
    } else {
        Ok(0)
    }
}

fn argmax_semantic_reference(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx: Option<usize> = None;
    let mut max_val = f32::NEG_INFINITY;

    for (idx, &val) in values.iter().enumerate() {
        let token_id = idx as u32;
        let allowed = token_id < semantic_vocab_size || (allow_eos && token_id == eos_token_id);
        if !allowed {
            continue;
        }
        if val > max_val {
            max_val = val;
            max_idx = Some(idx);
        }
    }

    if let Some(idx) = max_idx {
        Ok(idx as u32)
    } else if allow_eos {
        Ok(eos_token_id)
    } else {
        Ok(0)
    }
}

fn sample_semantic(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
    params: &TtsGenerationParams,
    history: &[u32],
    rng: &mut SimpleRng,
    prefer_device_sampling: bool,
) -> Result<u32> {
    if !prefer_device_sampling {
        return sample_semantic_reference(
            logits,
            semantic_vocab_size,
            eos_token_id,
            allow_eos,
            params,
            history,
            rng,
        );
    }

    // Greedy fallback stays on device until the selected scalar is copied back.
    if params.temperature <= 1e-5 {
        return argmax_semantic(logits, semantic_vocab_size, eos_token_id, allow_eos);
    }

    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };
    let vocab_len = logits.dim(0)?;
    let semantic_len = (semantic_vocab_size as usize).min(vocab_len);
    let (sampling_vocab, allowed_mask) =
        semantic_sampling_vocab_and_mask(vocab_len, semantic_len, eos_token_id, allow_eos);
    if let Some(candidates) = bounded_device_sampling_candidates(
        &logits,
        sampling_vocab,
        params.top_k,
        params.temperature,
        history,
        params.repetition_penalty,
        0.0,
        Some(&allowed_mask),
    )? {
        if device_candidates_cover_top_p(&candidates, params.top_p) {
            if let Some(token) = sample_device_candidates(&candidates, params.top_p, rng.next_f32())
            {
                return Ok(token);
            }
        }
    }
    let mut values = collect_semantic_sampling_values(&logits, semantic_len, params, history)?;

    let eos_idx = eos_token_id as usize;
    if allow_eos && eos_idx < vocab_len && eos_idx >= semantic_len {
        values.push((eos_token_id, logit_scalar_f32(&logits, eos_token_id)?));
    }

    // Repetition penalty over recent semantic history.
    if params.repetition_penalty > 1.0 && !history.is_empty() {
        let seen: HashSet<u32> = history.iter().copied().collect();
        for (token_id, v) in values.iter_mut() {
            if !seen.contains(token_id) {
                continue;
            }
            if !v.is_finite() {
                continue;
            }
            if *v > 0.0 {
                *v /= params.repetition_penalty;
            } else {
                *v *= params.repetition_penalty;
            }
        }
    }

    let temperature = params.temperature.max(1e-5);
    for (_, v) in values.iter_mut() {
        if v.is_finite() {
            *v /= temperature;
        }
    }

    let mut candidates: Vec<(u32, f32)> = values
        .into_iter()
        .filter(|(_, value)| value.is_finite())
        .collect();
    if candidates.is_empty() {
        return Ok(if allow_eos { eos_token_id } else { 0 });
    }

    // Top-k filtering.
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.truncate(params.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f32)> = candidates
        .iter()
        .map(|(token_id, value)| (*token_id, (*value - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_semantic(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p filtering over normalized probabilities.
    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = params.top_p.max(1e-6);
        let mut cumsum = 0.0f32;
        let mut keep = 0usize;
        for (_, p) in probs.iter() {
            cumsum += *p;
            keep += 1;
            if cumsum >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (token_id, p) in probs.iter() {
        acc += *p;
        if r <= acc {
            return Ok(*token_id);
        }
    }

    // Numerical fallback: pick max probability candidate.
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx)
        .or(Some(if allow_eos { eos_token_id } else { 0 }))
        .ok_or_else(|| Error::InferenceError("Failed to sample semantic token".to_string()))
}

fn semantic_sampling_vocab_and_mask(
    vocab_len: usize,
    semantic_len: usize,
    eos_token_id: u32,
    allow_eos: bool,
) -> (usize, Vec<bool>) {
    let eos_index = eos_token_id as usize;
    let sampling_vocab = if allow_eos && eos_index < vocab_len {
        semantic_len.max(eos_index.saturating_add(1))
    } else {
        semantic_len
    };
    let allowed = (0..sampling_vocab)
        .map(|index| index < semantic_len || (allow_eos && index == eos_index))
        .collect();
    (sampling_vocab, allowed)
}

fn sample_semantic_reference(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
    params: &TtsGenerationParams,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match logits.rank() {
        1 => logits,
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3-TTS semantic logits shape: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3-TTS semantic logits rank: {rank}"
            )))
        }
    };
    let mut values = logits.to_vec1::<f32>()?;

    // Token suppression: keep semantic range and optional EOS only.
    for (idx, v) in values.iter_mut().enumerate() {
        let token_id = idx as u32;
        let allowed = token_id < semantic_vocab_size || (allow_eos && token_id == eos_token_id);
        if !allowed {
            *v = f32::NEG_INFINITY;
        }
    }

    // Repetition penalty over recent semantic history.
    if params.repetition_penalty > 1.0 && !history.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history {
            let idx = token as usize;
            if idx < seen.len() {
                seen[idx] = true;
            }
        }
        for (idx, seen_flag) in seen.iter().enumerate() {
            if !*seen_flag {
                continue;
            }
            let v = &mut values[idx];
            if !v.is_finite() {
                continue;
            }
            if *v > 0.0 {
                *v /= params.repetition_penalty;
            } else {
                *v *= params.repetition_penalty;
            }
        }
    }

    // Greedy fallback when sampling is effectively disabled.
    if params.temperature <= 1e-5 {
        return argmax_semantic_reference(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }

    let temperature = params.temperature.max(1e-5);
    for v in values.iter_mut() {
        if v.is_finite() {
            *v /= temperature;
        }
    }

    let mut candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &v)| if v.is_finite() { Some(idx) } else { None })
        .collect();
    if candidates.is_empty() {
        return Ok(if allow_eos { eos_token_id } else { 0 });
    }

    // Top-k filtering.
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
        candidates.truncate(params.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|&idx| values[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| (idx, (values[idx] - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_semantic_reference(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p filtering over normalized probabilities.
    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = params.top_p.max(1e-6);
        let mut cumsum = 0.0f32;
        let mut keep = 0usize;
        for (_, p) in probs.iter() {
            cumsum += *p;
            keep += 1;
            if cumsum >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (idx, p) in probs.iter() {
        acc += *p;
        if r <= acc {
            return Ok(*idx as u32);
        }
    }

    // Numerical fallback: pick max probability candidate.
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx as u32)
        .or(Some(if allow_eos { eos_token_id } else { 0 }))
        .ok_or_else(|| Error::InferenceError("Failed to sample semantic token".to_string()))
}

fn collect_semantic_sampling_values(
    logits: &Tensor,
    semantic_len: usize,
    params: &TtsGenerationParams,
    history: &[u32],
) -> Result<Vec<(u32, f32)>> {
    if semantic_len == 0 {
        return Ok(Vec::new());
    }

    let semantic_logits = logits.narrow(0, 0, semantic_len)?;
    if params.top_k > 0 && params.top_k < semantic_len {
        let penalty_extra = if params.repetition_penalty > 1.0 && !history.is_empty() {
            history
                .iter()
                .copied()
                .filter(|token_id| (*token_id as usize) < semantic_len)
                .collect::<HashSet<_>>()
                .len()
        } else {
            0
        };
        let prefetch = params.top_k.saturating_add(penalty_extra).min(semantic_len);
        let (sorted_values, sorted_indices) = semantic_logits.sort_last_dim(false)?;
        let values = sorted_values.narrow(0, 0, prefetch)?.to_vec1::<f32>()?;
        let indices = sorted_indices
            .narrow(0, 0, prefetch)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        return Ok(indices.into_iter().zip(values).collect());
    }

    Ok(semantic_logits
        .to_vec1::<f32>()?
        .into_iter()
        .enumerate()
        .map(|(idx, value)| (idx as u32, value))
        .collect())
}

#[derive(Clone, Copy)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
    }
}

fn normalize_audio(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    // Drop non-finite values and remove DC offset.
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
            continue;
        }
        sum += *s as f64;
        count += 1;
    }
    if count > 0 {
        let mean = (sum / count as f64) as f32;
        for s in samples.iter_mut() {
            *s -= mean;
        }
    }

    // Peak normalize to avoid hard clipping in WAV encoder.
    let mut peak = 0.0f32;
    for &s in samples.iter() {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
    }
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }

    // Keep output loudness within a practical band for WAV playback.
    let mut power = 0.0f64;
    for &s in samples.iter() {
        power += (s as f64) * (s as f64);
    }
    let rms = (power / samples.len() as f64).sqrt() as f32;
    let max_rms = 0.12f32;
    let min_rms = 0.04f32;
    if rms > max_rms && rms > 1e-6 {
        let scale = max_rms / rms;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    } else if rms < min_rms && rms > 1e-6 {
        let scale = (min_rms / rms).min(8.0);
        for s in samples.iter_mut() {
            *s *= scale;
        }

        // Re-apply peak guard after boosting.
        let mut peak = 0.0f32;
        for &s in samples.iter() {
            let a = s.abs();
            if a > peak {
                peak = a;
            }
        }
        if peak > 0.95 {
            let scale = 0.95 / peak;
            for s in samples.iter_mut() {
                *s *= scale;
            }
        }
    }
}

/// Load a Qwen3-TTS model
pub fn load_model(model_path: &Path, device: DeviceProfile) -> Result<Qwen3TtsModel> {
    let kv_cache_dtype =
        std::env::var("IZWI_KV_CACHE_DTYPE").unwrap_or_else(|_| "float16".to_string());
    Qwen3TtsModel::load(model_path, device, default_kv_page_size(), &kv_cache_dtype)
}

#[cfg(test)]
mod tests {
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::backends::{BackendKind, DeviceCapabilities, DeviceKind};
    use crate::engine::{
        plan_managed_state_capacity, ManagedStateCapacityRequest, ModelInstanceId,
    };

    use super::config::{CodePredictorConfig, TalkerConfig};
    use super::*;

    fn dtype_test_profile(
        kind: DeviceKind,
        supports_bf16: bool,
        supports_f16: bool,
    ) -> DeviceProfile {
        DeviceProfile {
            device: candle_core::Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_bf16,
                supports_f16,
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    fn cache_test_config() -> Qwen3TtsConfig {
        Qwen3TtsConfig {
            architectures: vec!["Qwen3TTSForConditionalGeneration".to_string()],
            model_type: "qwen3_tts".to_string(),
            tokenizer_type: "qwen3_tts_tokenizer_12hz".to_string(),
            tts_model_size: "0b6".to_string(),
            tts_model_type: "custom_voice".to_string(),
            assistant_token_id: 77091,
            im_end_token_id: 151645,
            im_start_token_id: 151644,
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            talker_config: TalkerConfig {
                model_type: "qwen3_tts_talker".to_string(),
                hidden_size: 1024,
                intermediate_size: 3072,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                max_position_embeddings: 32768,
                vocab_size: 3072,
                text_vocab_size: 151936,
                text_hidden_size: 2048,
                num_code_groups: 16,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                hidden_act: "silu".to_string(),
                use_cache: true,
                position_id_per_seconds: 13,
                rope_scaling: None,
                sliding_window: None,
                code_predictor_config: CodePredictorConfig {
                    model_type: "qwen3_tts_talker_code_predictor".to_string(),
                    hidden_size: 1024,
                    intermediate_size: 3072,
                    num_hidden_layers: 5,
                    num_attention_heads: 16,
                    num_key_value_heads: 8,
                    head_dim: 128,
                    max_position_embeddings: 65536,
                    vocab_size: 2048,
                    num_code_groups: 16,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1_000_000.0,
                    hidden_act: "silu".to_string(),
                    use_cache: true,
                    layer_types: vec![],
                    text_hidden_size: None,
                },
                codec_bos_id: 2149,
                codec_eos_token_id: 2150,
                codec_think_id: 2154,
                codec_nothink_id: 2155,
                codec_pad_id: 2148,
                codec_think_bos_id: 2156,
                codec_think_eos_id: 2157,
                spk_id: std::collections::HashMap::new(),
                spk_is_dialect: std::collections::HashMap::new(),
                codec_language_id: std::collections::HashMap::new(),
            },
        }
    }

    #[test]
    fn target_managed_contract_keeps_talker_and_predictor_domains_distinct() {
        let contract =
            qwen3_tts_inference_state_contract(&cache_test_config(), DType::F16, DType::F32, 32)
                .expect("target contract");
        assert_eq!(contract.domains.len(), 2);

        let StateDomainSpec::PagedAttention(talker) = &contract.domains[0] else {
            panic!("talker domain must be paged attention");
        };
        let StateDomainSpec::PagedAttention(predictor) = &contract.domains[1] else {
            panic!("predictor domain must be paged attention");
        };
        assert_eq!(talker.header.id, StateDomainId::new(1));
        assert_eq!(talker.layers.len(), 28);
        assert_eq!(talker.accepted_dtypes, vec![StateDType::F16]);
        assert_eq!(predictor.header.id, StateDomainId::new(2));
        assert_eq!(predictor.layers.len(), 5);
        assert_eq!(predictor.accepted_dtypes, vec![StateDType::F32]);
    }

    #[test]
    fn predictor_width_two_plan_uses_exact_physical_kv_bytes() {
        let full =
            qwen3_tts_inference_state_contract(&cache_test_config(), DType::F16, DType::F32, 32)
                .unwrap();
        let mut predictor = full.domains[1].clone();
        let StateDomainSpec::PagedAttention(spec) = &mut predictor else {
            panic!("predictor must use paged attention")
        };
        spec.header.scope = StateScope::Invocation;
        spec.header.prefix = PrefixPolicy::Disabled;
        spec.header.checkpoint = CheckpointPolicy::None;
        let placement = spec.header.placement;
        let capacity = InvocationStateCapacity::PagedTokens { max_tokens: 16 };
        let per_row = minimum_physical_bytes_for_capacity(&predictor, capacity).unwrap();
        assert_eq!(per_row, 1_310_720);
        assert_eq!(per_row.checked_mul(2).unwrap(), 2_621_440);

        let workspace = InvocationWorkspaceDomain::State {
            state: predictor,
            capacity,
            placement,
            formula: WorkspaceFormula {
                fixed_bytes: per_row,
                dimensions: vec![],
                terms: vec![],
            },
        };
        assert_eq!(workspace.maximum_bytes().unwrap(), per_row);
    }

    #[test]
    fn transformer_workspace_bound_tracks_span_and_geometry_and_fails_closed() {
        let short = transformer_workspace_upper_bound_bytes(1, 32, 1024, 3072, 16, 8, 128, 3072, 2)
            .unwrap();
        let long = transformer_workspace_upper_bound_bytes(64, 64, 1024, 3072, 16, 8, 128, 3072, 2)
            .unwrap();
        let wider =
            transformer_workspace_upper_bound_bytes(64, 64, 1024, 6144, 16, 8, 128, 3072, 2)
                .unwrap();
        assert!(long > short);
        assert!(wider > long);
        assert!(transformer_workspace_upper_bound_bytes(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
        )
        .is_err());
    }

    fn retained_state_test_contract() -> InferenceStateContract {
        let full =
            qwen3_tts_inference_state_contract(&cache_test_config(), DType::F32, DType::F32, 32)
                .expect("managed contract");
        let mut retained = InferenceStateContract {
            abi: full.abi,
            domains: vec![full.domains[0].clone()],
            groups: vec![full.groups[0].clone()],
        };
        let StateDomainSpec::PagedAttention(talker) = &mut retained.domains[0] else {
            panic!("talker domain must be paged attention");
        };
        talker.header.prefix = PrefixPolicy::Disabled;
        talker.header.checkpoint = CheckpointPolicy::Transactional;
        retained.groups[0].prefix_shareable = false;
        qwen3_tts_retained_state_contract(retained, &cache_test_config(), DType::F32)
            .expect("retained state contract")
    }

    #[test]
    fn retained_state_contract_groups_talker_pages_and_all_decode_tensors() {
        let contract = retained_state_test_contract();
        assert_eq!(contract.domains.len(), 2);
        assert_eq!(contract.groups.len(), 1);
        assert_eq!(
            contract.groups[0].domains,
            vec![StateDomainId::new(1), QWEN3_TTS_MODEL_STATE_DOMAIN]
        );
        assert!(!contract.groups[0].prefix_shareable);

        let StateDomainSpec::Tensor(tensor) = &contract.domains[1] else {
            panic!("second retained domain must be tensor state");
        };
        assert_eq!(
            tensor.header.clock,
            StateClock::Custom("qwen3_tts_talker_tokens".into())
        );
        assert_eq!(tensor.header.checkpoint, CheckpointPolicy::Transactional);
        assert_eq!(tensor.header.prefix, PrefixPolicy::Disabled);
        assert_eq!(tensor.components.len(), 4);
        assert!(tensor.components.iter().all(|component| {
            component.shape.dimensions.first().is_some_and(|dimension| {
                dimension.axis == ShapeAxis::Batch
                    && dimension.extent == ShapeExtent::RuntimeBounded { min: 1, max: 1 }
            })
        }));
        assert_eq!(
            tensor
                .components
                .iter()
                .map(|component| component.role.clone())
                .collect::<Vec<_>>(),
            vec![
                TensorRole::EncoderMemory,
                TensorRole::RetainedEmbedding,
                TensorRole::RecurrentHidden,
                TensorRole::RetainedLogits,
            ]
        );
        assert!(tensor
            .components
            .iter()
            .all(|component| component.accepted_dtypes == vec![StateDType::F32]));
    }

    #[test]
    fn retained_tensor_shape_guard_requires_fixed_batch_axis() {
        let memory = Tensor::zeros((2, 4, 8), DType::F32, &candle_core::Device::Cpu).unwrap();
        let pad = Tensor::zeros((1, 1, 8), DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(validate_qwen3_tts_tensor_state_shapes(&memory, &pad, None, None).is_err());

        let memory = Tensor::zeros((1, 4, 8), DType::F32, &candle_core::Device::Cpu).unwrap();
        let hidden = Tensor::zeros((1, 1, 7), DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(
            validate_qwen3_tts_tensor_state_shapes(&memory, &pad, Some(&hidden), None).is_err()
        );
    }

    #[test]
    fn cuda_context_pages_do_not_multiply_qwen_tts_retained_rows() {
        let full =
            qwen3_tts_inference_state_contract(&cache_test_config(), DType::F16, DType::F32, 64)
                .expect("managed contract");
        let mut retained = InferenceStateContract {
            abi: full.abi,
            domains: vec![full.domains[0].clone()],
            groups: vec![full.groups[0].clone()],
        };
        let StateDomainSpec::PagedAttention(talker) = &mut retained.domains[0] else {
            panic!("talker domain must be paged attention");
        };
        talker.header.prefix = PrefixPolicy::Disabled;
        talker.header.checkpoint = CheckpointPolicy::Transactional;
        retained.groups[0].prefix_shareable = false;
        let retained =
            qwen3_tts_retained_state_contract(retained, &cache_test_config(), DType::F16)
                .expect("retained state contract");
        let state_plan = negotiate_state_plan(
            &retained,
            &StateBackendPlanRequest {
                // Capacity planning is backend-pure. CPU lets this regression
                // verify CUDA geometry on a host without an NVIDIA device.
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(64),
                storage_dtype_hint: Some(StateDType::F16),
            },
        )
        .expect("state plan");
        let (allocation, tensor) = plan_managed_state_capacity(
            &state_plan,
            ModelInstanceId::new(804),
            ManagedStateCapacityRequest {
                total_paged_pages: 512,
                logical_token_reach: Some(32_768),
                retained_sequence_rows: 16,
                staged_transaction_rows: 16,
            },
        )
        .expect("capacity plan");
        let tensor = tensor.expect("Qwen3 TTS has retained tensor state");

        assert_eq!(tensor.sequence_capacity(), 16);
        assert_eq!(tensor.transaction_capacity(), 16);
        assert_eq!(tensor.per_sequence_bytes(), 67_119_104);
        assert_eq!(tensor.authorized_bytes(), 2_147_811_328);

        // Before the shared page/row fix, the 512 pages needed by one 32K
        // context were treated as 512 retained sequences and added to the 16
        // transaction rows. That 35.4 GB state claim alone can reject a model
        // load on a 48 GB L40S after weights are resident.
        let legacy_authorization = tensor.per_sequence_bytes() * (512 + 16);
        assert_eq!(legacy_authorization, 35_438_886_912);
        assert_eq!(
            allocation
                .group_capacity(
                    state_plan.paged_attention[0].group,
                    state_plan.paged_attention[0].domain,
                )
                .expect("paged capacity")
                .strategy
                .maximum_blocks(),
            512
        );
        let non_paged = &state_plan.non_paged[0];
        assert_eq!(
            allocation
                .group_capacity(non_paged.group(), non_paged.domain())
                .expect("tensor capacity")
                .strategy
                .maximum_blocks(),
            32
        );
    }

    #[test]
    fn retained_tensor_arena_roundtrip_abort_and_release_are_transactional() {
        let contract = retained_state_test_contract();
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(32),
                storage_dtype_hint: Some(StateDType::F32),
            },
        )
        .expect("state plan");
        let capacity = crate::backends::state::TensorStateCapacity::for_plan(&plan, 1, 1)
            .expect("tensor capacity");
        let arena = TensorStateArena::new(Arc::new(plan), capacity, candle_core::Device::Cpu)
            .expect("tensor arena");
        let sequence = PhysicalStateSequenceId::new(7).unwrap();
        arena.register(sequence).unwrap();

        let values = |scalar: f32| {
            (1..=4)
                .map(|component| StateComponentValue {
                    component: StateComponentId::new(component),
                    tensor: Some(
                        Tensor::from_slice(&[scalar], 1, &candle_core::Device::Cpu).unwrap(),
                    ),
                })
                .collect::<Vec<_>>()
        };
        let committed = PhysicalStateTransactionId::new(11).unwrap();
        arena.begin(committed, sequence).unwrap();
        arena
            .stage_replace(committed, QWEN3_TTS_MODEL_STATE_DOMAIN, 0, 9, values(1.0))
            .unwrap();
        arena.commit(committed, 9).unwrap();
        let snapshot = arena
            .read(sequence, QWEN3_TTS_MODEL_STATE_DOMAIN)
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.cursor, 9);
        assert_eq!(snapshot.components.len(), 4);

        let aborted = PhysicalStateTransactionId::new(12).unwrap();
        arena.begin(aborted, sequence).unwrap();
        arena
            .stage_replace(aborted, QWEN3_TTS_MODEL_STATE_DOMAIN, 9, 10, values(2.0))
            .unwrap();
        arena.abort(aborted).unwrap();
        let snapshot = arena
            .read(sequence, QWEN3_TTS_MODEL_STATE_DOMAIN)
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.cursor, 9);
        assert!(snapshot.components.iter().all(|component| {
            component.tensor.as_ref().unwrap().to_vec1::<f32>().unwrap() == vec![1.0]
        }));

        arena.release(sequence).unwrap();
        assert!(arena.read(sequence, QWEN3_TTS_MODEL_STATE_DOMAIN).is_err());
    }

    #[test]
    fn non_final_prefill_stages_canonical_tensor_state_without_head_outputs() {
        let mut contract = retained_state_test_contract();
        let StateDomainSpec::Tensor(domain) = &mut contract.domains[1] else {
            panic!("second retained domain must be tensor state")
        };
        for component in &mut domain.components {
            for dimension in &mut component.shape.dimensions {
                let max = match dimension.axis {
                    ShapeAxis::Sequence => {
                        if component.id == StateComponentId::new(1) {
                            8
                        } else {
                            1
                        }
                    }
                    ShapeAxis::Hidden => 4,
                    _ => 8,
                };
                dimension.extent = ShapeExtent::RuntimeBounded { min: 1, max };
            }
        }
        contract.validate().unwrap();
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(2),
                storage_dtype_hint: Some(StateDType::F32),
            },
        )
        .unwrap();
        let capacity = crate::backends::state::TensorStateCapacity::for_plan(&plan, 1, 1).unwrap();
        let arena =
            TensorStateArena::new(Arc::new(plan), capacity, candle_core::Device::Cpu).unwrap();
        let sequence = PhysicalStateSequenceId::new(71).unwrap();
        let transaction = PhysicalStateTransactionId::new(72).unwrap();
        arena.register(sequence).unwrap();
        arena.begin(transaction, sequence).unwrap();

        let device = candle_core::Device::Cpu;
        let retained_sequence_memory = Tensor::zeros((1, 4, 4), DType::F32, &device).unwrap();
        let prepared = PreparedTtsDecodePrefill {
            prefill_embeds: retained_sequence_memory.narrow(1, 0, 2).unwrap(),
            trailing_text_hidden: retained_sequence_memory.narrow(1, 2, 2).unwrap(),
            retained_sequence_memory,
            tts_pad_embed: Tensor::zeros((1, 1, 4), DType::F32, &device).unwrap(),
            params: TtsGenerationParams::default(),
        };
        let (cache_arena, bindings) = talker::tests::test_arena(770);
        let mut state = PhysicalTtsPrefillState {
            prepared,
            talker_cache: talker::tests::test_cache(cache_arena, &bindings, 0),
            stream_config: TtsStreamingConfig::final_only(),
            progress: 0,
            total_tokens: 2,
            last_hidden: None,
            last_logits: None,
            tensor_sequence: None,
        };
        state.bind_tensor_sequence(sequence.get()).unwrap();
        state.stage_tensor_state(&arena, transaction.get()).unwrap();
        arena.commit(transaction, 0).unwrap();
        let snapshot = arena
            .read(sequence, QWEN3_TTS_MODEL_STATE_DOMAIN)
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.cursor, 0);
        assert!(snapshot.components[0].tensor.is_some());
        assert!(snapshot.components[1].tensor.is_some());
        assert!(snapshot.components[2].tensor.is_none());
        assert!(snapshot.components[3].tensor.is_none());
    }

    fn transactional_decode_state(cache: TalkerPhysicalCache, seed: u64) -> PhysicalTtsDecodeState {
        let device = candle_core::Device::Cpu;
        PhysicalTtsDecodeState {
            talker_cache: cache,
            text_vocab_size: 32,
            acoustic_vocab_size: 8,
            semantic_vocab_size: 8,
            trailing_text_hidden: Some(Tensor::full(seed as f32, (1, 1, 4), &device).unwrap()),
            retained_sequence_memory: Some(Tensor::full(seed as f32, (1, 1, 4), &device).unwrap()),
            prefill_tokens: 0,
            trailing_text_len: 1,
            tts_pad_embed: Some(Tensor::full(seed as f32 + 1.0, (1, 1, 4), &device).unwrap()),
            max_frames: 8,
            frame_idx: 0,
            offset: 0,
            all_code_groups: vec![Vec::new(); 4],
            semantic_history: vec![seed as u32],
            last_hidden: Some(Tensor::full(seed as f32 + 2.0, (1, 1, 4), &device).unwrap()),
            last_logits: Some(Tensor::full(seed as f32 + 3.0, (1, 1, 8), &device).unwrap()),
            tensor_sequence: None,
            rng: SimpleRng { state: seed },
            params: TtsGenerationParams {
                max_frames: 8,
                ..Default::default()
            },
            stream_config: TtsStreamingConfig::final_only(),
            emitted_frames: 0,
            emitted_samples: 0,
            decode_raw_token_scratch: Vec::new(),
            finished: false,
        }
    }

    #[test]
    fn managed_quantum_rollback_restores_all_tensor_continuation_handles() {
        let (arena, bindings) = talker::tests::test_arena(771);
        let mut state =
            transactional_decode_state(talker::tests::test_cache(arena.clone(), &bindings, 0), 61);
        let tensor_ids = [
            state.retained_sequence_memory.as_ref().unwrap().id(),
            state.trailing_text_hidden.as_ref().unwrap().id(),
            state.tts_pad_embed.as_ref().unwrap().id(),
            state.last_hidden.as_ref().unwrap().id(),
            state.last_logits.as_ref().unwrap().id(),
        ];
        let checkpoint = state
            .begin_managed_quantum(talker::tests::test_cache(arena, &bindings, 0))
            .unwrap();
        state.clear_tensor_handles();
        state.frame_idx = 3;
        state.rng.state = 999;
        state.rollback_managed_quantum(checkpoint);
        assert_eq!(state.frame_idx, 0);
        assert_eq!(state.rng.state, 61);
        assert_eq!(
            [
                state.retained_sequence_memory.as_ref().unwrap().id(),
                state.trailing_text_hidden.as_ref().unwrap().id(),
                state.tts_pad_embed.as_ref().unwrap().id(),
                state.last_hidden.as_ref().unwrap().id(),
                state.last_logits.as_ref().unwrap().id(),
            ],
            tensor_ids
        );
    }

    fn prepared_talker_stage(
        state: &PhysicalTtsDecodeState,
        semantic_token: u32,
    ) -> PreparedTtsTalkerStage {
        let device = candle_core::Device::Cpu;
        let mut next_history = state.semantic_history.clone();
        next_history.push(semantic_token);
        PreparedTtsTalkerStage {
            predictor: PreparedTtsPredictorStage {
                semantic_token,
                semantic_embed: Tensor::zeros((1, 1, 4), DType::F32, &device).unwrap(),
                talker_hidden: state.last_hidden.as_ref().unwrap().clone(),
                source_logits: state.last_logits.as_ref().unwrap().clone(),
                text_addition: Tensor::zeros((1, 1, 4), DType::F32, &device).unwrap(),
                expected_talker_context: state.talker_cache.context_len(),
                expected_frame_idx: state.frame_idx,
                expected_rng: state.rng,
                expected_semantic_history: state.semantic_history.clone(),
                expected_tensor_sequence: state.tensor_sequence,
                next_rng: SimpleRng {
                    state: state.rng.state.wrapping_add(1),
                },
                next_semantic_history: next_history,
                sampling_ms: 1.0,
            },
            acoustic_codes: vec![1, 2, 3],
            step_input: Tensor::full(0.25f32, (1, 1, 4), &device).unwrap(),
            predictor_ms: 2.0,
        }
    }

    #[test]
    fn failed_talker_batch_preserves_every_cursor_and_continuation_tensor() {
        let device = candle_core::Device::Cpu;
        let talker = talker::tests::tiny_talker(&device);
        let (arena_a, bindings_a) = talker::tests::test_arena(721);
        let (arena_b, bindings_b) = talker::tests::test_arena(722);
        let mut state_a =
            transactional_decode_state(talker::tests::test_cache(arena_a, &bindings_a, 0), 41);
        let mut state_b =
            transactional_decode_state(talker::tests::test_cache(arena_b, &bindings_b, 0), 43);
        let rows = vec![
            prepared_talker_stage(&state_a, 2),
            prepared_talker_stage(&state_b, 3),
        ];
        let snapshot = [&state_a, &state_b].map(|state| {
            (
                state.talker_cache.context_len(),
                state.frame_idx,
                state.offset,
                state.rng.state,
                state.semantic_history.clone(),
                state.all_code_groups.clone(),
                state
                    .last_hidden
                    .as_ref()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                state
                    .last_logits
                    .as_ref()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
        });
        let mut states = [&mut state_a, &mut state_b];

        assert!(
            Qwen3TtsModel::tts_talker_stage_batch_with_model(&talker, &mut states, rows,).is_err()
        );
        for (state, expected) in [&state_a, &state_b].into_iter().zip(snapshot) {
            assert_eq!(state.talker_cache.context_len(), expected.0);
            assert_eq!(state.frame_idx, expected.1);
            assert_eq!(state.offset, expected.2);
            assert_eq!(state.rng.state, expected.3);
            assert_eq!(state.semantic_history, expected.4);
            assert_eq!(state.all_code_groups, expected.5);
            assert_eq!(
                state
                    .last_hidden
                    .as_ref()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                expected.6
            );
            assert_eq!(
                state
                    .last_logits
                    .as_ref()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                expected.7
            );
        }
    }

    fn simulate_batch_model_progress(
        talker: &TalkerModel,
        state: &mut PhysicalTtsDecodeState,
        predictor_cache: &mut CodePredictorPhysicalCache,
        token: u32,
    ) -> Result<()> {
        let input = Tensor::full(token as f32 / 8.0, (1, 1, 4), &candle_core::Device::Cpu)?;
        let (last_hidden, last_logits) =
            talker.generate_physical_step_with_embed(&input, &mut state.talker_cache)?;
        let _ = talker.generate_physical_step_with_embed(&input, predictor_cache)?;

        state.frame_idx += 1;
        state.offset = state.talker_cache.context_len();
        state.rng.state = state.rng.state.wrapping_add(1);
        state.semantic_history.push(token);
        for group in &mut state.all_code_groups {
            group.push(token);
        }
        state.last_hidden = Some(last_hidden);
        state.last_logits = Some(last_logits);
        Ok(())
    }

    fn simulate_batch_codec_finalization(state: &mut PhysicalTtsDecodeState, token: u32) {
        state.emitted_frames += 1;
        state.emitted_samples += 80;
        state.decode_raw_token_scratch.push(vec![token]);
    }

    #[test]
    fn late_codec_failure_rolls_back_every_row_before_ordered_retry() {
        let talker = talker::tests::tiny_talker(&candle_core::Device::Cpu);
        let (arena, bindings) = talker::tests::test_arena(723);
        let mut state_a =
            transactional_decode_state(talker::tests::test_cache(arena.clone(), &bindings, 0), 47);
        let mut state_b =
            transactional_decode_state(talker::tests::test_cache(arena.clone(), &bindings, 6), 53);
        let mut predictor_a = talker::tests::test_cache(arena.clone(), &bindings, 12);
        let mut predictor_b = talker::tests::test_cache(arena, &bindings, 18);
        let initial_hidden_ids = [
            state_a.last_hidden.as_ref().unwrap().id(),
            state_b.last_hidden.as_ref().unwrap().id(),
        ];
        let initial_logits_ids = [
            state_a.last_logits.as_ref().unwrap().id(),
            state_b.last_logits.as_ref().unwrap().id(),
        ];
        let initial_rng = [state_a.rng.state, state_b.rng.state];

        let mut states = [&mut state_a, &mut state_b];
        let mut predictors = [&mut predictor_a, &mut predictor_b];
        let failed: Result<Vec<u32>> =
            run_tts_decode_batch_transaction(&mut states, &mut predictors, |states, predictors| {
                for row in 0..states.len() {
                    simulate_batch_model_progress(
                        &talker,
                        states[row],
                        predictors[row],
                        u32::try_from(row + 1).unwrap(),
                    )?;
                }
                // Row zero has already produced audio when row one's codec
                // fails. The public Vec-returning call cannot expose row zero,
                // so the complete physical batch must become retryable.
                simulate_batch_codec_finalization(states[0], 1);
                Err(Error::InferenceError(
                    "injected second-row codec failure".into(),
                ))
            });
        let failure = failed.unwrap_err().to_string();
        assert!(
            failure.contains("injected second-row codec failure"),
            "unexpected failure: {failure}"
        );
        drop(states);
        drop(predictors);

        for (row, state) in [&mut state_a, &mut state_b].into_iter().enumerate() {
            assert_eq!(state.talker_cache.context_len(), 0);
            assert_eq!(state.frame_idx, 0);
            assert_eq!(state.offset, 0);
            assert_eq!(state.rng.state, initial_rng[row]);
            assert_eq!(state.semantic_history.len(), 1);
            assert!(state.all_code_groups.iter().all(Vec::is_empty));
            assert_eq!(
                state.last_hidden.as_ref().unwrap().id(),
                initial_hidden_ids[row]
            );
            assert_eq!(
                state.last_logits.as_ref().unwrap().id(),
                initial_logits_ids[row]
            );
            assert_eq!(state.emitted_frames, 0);
            assert_eq!(state.emitted_samples, 0);
            assert!(state.decode_raw_token_scratch.is_empty());
            assert!(!state.finished);
            assert!(state.take_managed_write_completions().is_empty());
        }
        assert_eq!(
            (predictor_a.context_len(), predictor_b.context_len()),
            (0, 0)
        );
        assert!(predictor_a.take_completed_writes().is_empty());
        assert!(predictor_b.take_completed_writes().is_empty());

        let mut states = [&mut state_a, &mut state_b];
        let mut predictors = [&mut predictor_a, &mut predictor_b];
        let outputs =
            run_tts_decode_batch_transaction(&mut states, &mut predictors, |states, predictors| {
                for row in 0..states.len() {
                    let token = u32::try_from(row + 1).unwrap();
                    simulate_batch_model_progress(&talker, states[row], predictors[row], token)?;
                    simulate_batch_codec_finalization(states[row], token);
                }
                Ok(vec![10, 20])
            })
            .unwrap();
        assert_eq!(outputs, vec![10, 20]);
        drop(states);
        drop(predictors);

        for state in [&state_a, &state_b] {
            assert_eq!(state.talker_cache.context_len(), 1);
            assert_eq!(state.frame_idx, 1);
            assert_eq!(state.offset, 1);
            assert_eq!(state.semantic_history.len(), 2);
            assert!(state.all_code_groups.iter().all(|group| group.len() == 1));
            assert_eq!(state.emitted_frames, 1);
            assert_eq!(state.emitted_samples, 80);
            assert_eq!(state.decode_raw_token_scratch.len(), 1);
        }
        assert_eq!(
            (predictor_a.context_len(), predictor_b.context_len()),
            (1, 1)
        );
    }

    #[test]
    fn test_special_tokens_creation() {
        let main_config = cache_test_config();

        let specials = TtsSpecialTokens::from_configs(&main_config, &main_config.talker_config);
        assert_eq!(specials.codec_bos_id, 2149);
        assert_eq!(specials.codec_eos_token_id, 2150);
    }

    #[test]
    fn session_cache_layout_scales_with_request_instead_of_model_context() {
        let tiny = standard_session_cache_layout(32_768, 12, false, true, 0, 16).unwrap();
        let larger = standard_session_cache_layout(32_768, 240, true, true, 32, 512).unwrap();

        assert_eq!(tiny.prefill_tokens, 9);
        assert_eq!(tiny.max_frames, 16);
        assert_eq!(tiny.talker_cache_tokens(), Some(25));
        assert!(larger.prefill_tokens > tiny.prefill_tokens);
        assert!(larger.max_frames > tiny.max_frames);
        assert!(larger.talker_cache_tokens().unwrap() > tiny.talker_cache_tokens().unwrap());
        assert!(larger.talker_cache_tokens().unwrap() < 32_768);
    }

    #[test]
    fn session_cache_layout_caps_frames_and_voice_reference_conditioning() {
        let capped = standard_session_cache_layout(128, 30, true, true, 10, usize::MAX)
            .expect("context-capped layout");
        assert_eq!(capped.prefill_tokens, 20);
        assert_eq!(capped.max_frames, 107);
        assert_eq!(capped.talker_cache_tokens(), Some(127));

        let small_reference =
            voice_clone_session_cache_layout(2_048, 20, 15, 8, false, 64).unwrap();
        let large_reference =
            voice_clone_session_cache_layout(2_048, 20, 15, 200, false, 64).unwrap();
        let capped_reference =
            voice_clone_session_cache_layout(2_048, 20, 15, usize::MAX, false, 64).unwrap();
        let explicit_cap = voice_clone_session_cache_layout(
            2_048,
            20,
            15,
            MAX_VOICE_CLONE_REFERENCE_FRAMES,
            false,
            64,
        )
        .unwrap();
        assert!(large_reference.prefill_tokens > small_reference.prefill_tokens);
        assert_eq!(capped_reference, explicit_cap);
    }

    #[test]
    fn session_cache_layout_fails_closed_on_overflow() {
        let layout_error =
            standard_session_cache_layout(usize::MAX, 1, false, false, usize::MAX, 1)
                .expect_err("instruction-token overflow must fail");
        assert!(matches!(layout_error, Error::Overloaded(_)));
    }

    #[test]
    fn cuda_base_tts_uses_half_transformers_without_changing_decoder_default() {
        let profile = dtype_test_profile(DeviceKind::Cuda, true, true);
        let plan = select_qwen3_tts_dtypes(&profile, None, false, true).unwrap();

        assert_eq!(plan.talker, DType::BF16);
        assert_eq!(plan.code_predictor, DType::BF16);
        assert_eq!(plan.speech_tokenizer, DType::F32);
    }

    #[test]
    fn cuda_custom_tts_falls_back_to_f16_transformers_without_bf16() {
        let profile = dtype_test_profile(DeviceKind::Cuda, false, true);
        let plan = select_qwen3_tts_dtypes(&profile, None, true, false).unwrap();

        assert_eq!(plan.talker, DType::F16);
        assert_eq!(plan.code_predictor, DType::F16);
        assert_eq!(plan.speech_tokenizer, DType::F32);
    }

    #[test]
    fn qwen_tts_default_dtype_policy_preserves_correctness_sensitive_modes() {
        let cpu = dtype_test_profile(DeviceKind::Cpu, false, false);
        let cpu_custom = select_qwen3_tts_dtypes(&cpu, None, true, false).unwrap();
        assert_eq!(
            cpu_custom,
            Qwen3TtsDTypePlan {
                talker: DType::F32,
                code_predictor: DType::F32,
                speech_tokenizer: DType::F32,
            }
        );

        let metal = dtype_test_profile(DeviceKind::Metal, false, false);
        let metal_custom = select_qwen3_tts_dtypes(&metal, None, true, false).unwrap();
        assert_eq!(
            metal_custom,
            Qwen3TtsDTypePlan {
                talker: DType::F32,
                code_predictor: DType::F32,
                speech_tokenizer: DType::F32,
            }
        );

        let metal_base = select_qwen3_tts_dtypes(&metal, None, false, true).unwrap();
        assert_eq!(metal_base, metal_custom);

        let metal_voice_design = select_qwen3_tts_dtypes(&metal, None, false, false).unwrap();
        assert_eq!(
            metal_voice_design,
            Qwen3TtsDTypePlan {
                talker: DType::F16,
                code_predictor: DType::F16,
                speech_tokenizer: DType::F16,
            }
        );
    }

    #[test]
    fn metal_f32_compute_uses_configured_dense_kv_storage() {
        let metal = dtype_test_profile(DeviceKind::Metal, false, false);
        let compute = select_qwen3_tts_dtypes(&metal, None, true, false).unwrap();
        assert_eq!(compute.talker, DType::F32);
        assert_eq!(
            select_qwen3_tts_state_dtypes(&metal, compute, "float16").unwrap(),
            (DType::F16, DType::F16)
        );
        assert_eq!(
            select_qwen3_tts_state_dtypes(&metal, compute, "float32").unwrap(),
            (DType::F32, DType::F32)
        );

        let cpu = dtype_test_profile(DeviceKind::Cpu, false, false);
        let cpu_compute = select_qwen3_tts_dtypes(&cpu, None, true, false).unwrap();
        assert_eq!(
            select_qwen3_tts_state_dtypes(&cpu, cpu_compute, "float16").unwrap(),
            (DType::F32, DType::F32)
        );
    }

    #[test]
    fn explicit_qwen_tts_dtype_override_applies_to_all_components() {
        let profile = dtype_test_profile(DeviceKind::Cuda, true, true);
        let plan = select_qwen3_tts_dtypes(&profile, Some("f32"), true, false).unwrap();

        assert_eq!(
            plan,
            Qwen3TtsDTypePlan {
                talker: DType::F32,
                code_predictor: DType::F32,
                speech_tokenizer: DType::F32,
            }
        );
    }

    #[test]
    fn qwen_tts_optimized_sampling_is_accelerator_neutral() {
        let cpu = dtype_test_profile(DeviceKind::Cpu, false, false);
        let metal = dtype_test_profile(DeviceKind::Metal, false, false);
        let cuda = dtype_test_profile(DeviceKind::Cuda, true, true);

        assert!(!qwen_tts_uses_device_sampling(&cpu));
        assert!(qwen_tts_uses_device_sampling(&metal));
        assert!(qwen_tts_uses_device_sampling(&cuda));
    }

    #[test]
    fn qwen_tts_eos_gate_matches_reference_minimum() {
        assert!(!qwen_tts_allows_eos(
            MIN_QWEN_TTS_TOKENS_BEFORE_EOS.saturating_sub(1)
        ));
        assert!(qwen_tts_allows_eos(MIN_QWEN_TTS_TOKENS_BEFORE_EOS));
        assert!(qwen_tts_allows_eos(MIN_QWEN_TTS_TOKENS_BEFORE_EOS + 1));
    }

    #[test]
    fn semantic_argmax_masks_invalid_tokens_and_gates_eos() {
        let logits = Tensor::new(
            vec![0.0f32, 2.0, 4.0, 1.0, 100.0, 3.0, 5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();

        assert_eq!(argmax_semantic(&logits, 3, 6, false).unwrap(), 2);
        assert_eq!(argmax_semantic(&logits, 3, 6, true).unwrap(), 6);
    }

    #[test]
    fn greedy_semantic_sampling_uses_device_argmax_path() {
        let logits = Tensor::new(
            vec![0.0f32, 2.0, 4.0, 1.0, 100.0, 3.0, 5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let params = TtsGenerationParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        assert_eq!(
            sample_semantic(&logits, 3, 6, true, &params, &[], &mut rng, true).unwrap(),
            6
        );
    }

    #[test]
    fn reference_semantic_sampling_suppresses_invalid_tokens() {
        let logits = Tensor::new(
            vec![0.0f32, 2.0, 4.0, 1.0, 100.0, 3.0, 5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let params = TtsGenerationParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        assert_eq!(
            sample_semantic(&logits, 3, 6, false, &params, &[], &mut rng, false).unwrap(),
            2
        );
        assert_eq!(
            sample_semantic(&logits, 3, 6, true, &params, &[], &mut rng, false).unwrap(),
            6
        );
    }

    #[test]
    fn semantic_sampling_reference_and_device_paths_match_simple_top_k() {
        let logits = Tensor::new(
            vec![10.0f32, 9.0, 8.0, 0.0, -5.0],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let params = TtsGenerationParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let mut reference_rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };
        let mut device_rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        let reference = sample_semantic(
            &logits,
            5,
            99,
            false,
            &params,
            &[0],
            &mut reference_rng,
            false,
        )
        .unwrap();
        let device =
            sample_semantic(&logits, 5, 99, false, &params, &[0], &mut device_rng, true).unwrap();

        assert_eq!(reference, device);
    }

    #[test]
    fn semantic_device_sampling_mask_keeps_only_semantic_tokens_and_optional_eos() {
        let (vocab, mask) = semantic_sampling_vocab_and_mask(12, 4, 9, true);
        assert_eq!(vocab, 10);
        assert_eq!(
            mask,
            vec![true, true, true, true, false, false, false, false, false, true]
        );

        let (vocab, mask) = semantic_sampling_vocab_and_mask(12, 4, 9, false);
        assert_eq!(vocab, 4);
        assert_eq!(mask, vec![true; 4]);

        let (vocab, mask) = semantic_sampling_vocab_and_mask(5, 4, 9, true);
        assert_eq!(vocab, 4);
        assert_eq!(mask, vec![true; 4]);
    }

    #[test]
    fn top_k_semantic_sampling_keeps_penalty_replacement_candidates() {
        let logits = Tensor::new(vec![10.0f32, 9.0, 0.0, -5.0], &candle_core::Device::Cpu).unwrap();
        let params = TtsGenerationParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut rng = SimpleRng {
            state: 0x1234_5678_9abc_def0,
        };

        assert_eq!(
            sample_semantic(&logits, 4, 99, false, &params, &[0], &mut rng, true).unwrap(),
            1
        );
    }

    #[test]
    fn raw_codec_token_mapping_preserves_unoffset_tokens() {
        assert_eq!(raw_codec_token(151_936 + 7, 0, 151_936, 2048), 7);
        assert_eq!(raw_codec_token(7, 0, 151_936, 2048), 7);
        assert_eq!(
            raw_codec_token(151_936 + (3 * 2048) + 19, 3, 151_936, 2048),
            19
        );
        assert_eq!(raw_codec_token(19, 3, 151_936, 2048), 19);
    }

    #[test]
    fn cuda_chunked_codec_streaming_is_explicit_opt_in() {
        assert!(!qwen_tts_cuda_chunked_codec_stream_enabled_from(None));
        for raw in ["", "0", "false", "off", "no"] {
            assert!(!qwen_tts_cuda_chunked_codec_stream_enabled_from(Some(raw)));
        }
        for raw in ["1", "true", "YES", " on "] {
            assert!(qwen_tts_cuda_chunked_codec_stream_enabled_from(Some(raw)));
        }
    }
}
