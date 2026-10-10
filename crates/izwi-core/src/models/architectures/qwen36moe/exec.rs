//! Qwen3.6-MoE chat execution core: tokenizer, prompt rendering, hybrid
//! decode state, MTP speculation, and sampling. Forked from the dense
//! `qwen35` family and owned by `qwen36moe` alone.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;

use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::backends::kv::CandleAcceleratorKvArena;
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::InferenceStateContract;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};
use crate::models::shared::speculative_sampling::{
    propose_speculative_draft, verify_speculative_proposals, SpeculativeDraft,
};
use crate::models::shared::sampling::{
    bounded_device_sampling_candidates, device_candidates_cover_top_p, sample_device_candidates,
};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::text::{Qwen36MoeFfnGeometry, Qwen36TextModel, Qwen36TextRuntimeState};

mod timing;

const IMAGE_PAD_PLACEHOLDER: &str = "<|image_pad|>";
const VIDEO_PAD_PLACEHOLDER: &str = "<|video_pad|>";
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;
const MAX_PREFILL_CHUNK_SIZE: usize = 2048;

fn qwen35_prefill_chunk_size() -> usize {
    std::env::var("IZWI_QWEN35_PREFILL_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE)
        .min(MAX_PREFILL_CHUNK_SIZE)
}

fn next_prefill_segment_end(
    prompt_ids: &[u32],
    cursor: usize,
    span_end: usize,
    image_pad: u32,
    chunk_size: usize,
) -> Result<(usize, bool)> {
    if cursor >= span_end || span_end > prompt_ids.len() || chunk_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 prefill segment bounds are invalid".into(),
        ));
    }
    let image = prompt_ids[cursor] == image_pad;
    let boundary = (cursor + 1..span_end)
        .find(|index| (prompt_ids[*index] == image_pad) != image)
        .unwrap_or(span_end);
    Ok((boundary.min(cursor.saturating_add(chunk_size)), image))
}
/// Fully prepared text-only prefill input. The runtime carries this exact
/// artifact into the executor so prompt rendering and encoding happen once.
#[derive(Debug, Clone)]
pub struct Qwen36PreparedPrompt {
    prompt_ids: Vec<u32>,
    prompt_positions: Vec<[usize; 3]>,
    next_text_position: usize,
}

impl Qwen36PreparedPrompt {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    pub(crate) fn prompt_positions(&self) -> &[[usize; 3]] {
        &self.prompt_positions
    }
}

fn resolve_prepared_prompt<F>(
    prepared: Option<&Qwen36PreparedPrompt>,
    prepare: F,
) -> Result<Qwen36PreparedPrompt>
where
    F: FnOnce() -> Result<Qwen36PreparedPrompt>,
{
    match prepared {
        Some(prepared) => Ok(prepared.clone()),
        None => prepare(),
    }
}

fn initial_penalty_history(
    prompt_ids: &[u32],
    max_new_tokens: usize,
    track_history: bool,
) -> Vec<u32> {
    if !track_history {
        return Vec::new();
    }

    let mut history = Vec::with_capacity(prompt_ids.len().saturating_add(max_new_tokens.max(1)));
    history.extend_from_slice(prompt_ids);
    history
}

fn expected_physical_decode_cursor(prefill_progress: usize, tokens_generated: usize) -> usize {
    prefill_progress.saturating_add(tokens_generated.saturating_sub(1))
}

/// Shared decodability gate for every continuous-batch entry path (scalar
/// rounds and speculative envelopes): the executor only grants multi-row
/// quanta to rows whose prefill bootstrap already ran, so an undecodable row
/// is a contract breach, not a degrade.
fn validate_continuous_decode_rows(states: &[&mut ChatDecodeState]) -> Result<()> {
    for (row, state) in states.iter().enumerate() {
        let expected_physical_cursor =
            expected_physical_decode_cursor(state.prefill_progress, state.tokens_generated);
        let mut reasons: Vec<String> = Vec::new();
        if state.finished {
            reasons.push("finished".into());
        }
        if state.tokens_generated >= state.max_new_tokens {
            reasons.push("max_new_tokens reached".into());
        }
        if state.unconsumed_output.is_some() {
            reasons.push("unconsumed prefill output".into());
        }
        if state.pending_token.is_none() {
            reasons.push("no pending token".into());
        }
        if state.physical_kv.context_len() != expected_physical_cursor {
            reasons.push(format!(
                "physical cursor {} != expected {expected_physical_cursor}",
                state.physical_kv.context_len()
            ));
        }
        if !reasons.is_empty() {
            return Err(Error::InvalidInput(format!(
                "continuous chat batch row {row} is not decodable: {}",
                reasons.join("; ")
            )));
        }
    }
    Ok(())
}

/// DS9.4 sub-batch reborrows: gather a subset of rows out of a batch slice
/// while keeping every row independently mutable. Row ids are ascending.
fn take_state_subset<'a>(
    states: &'a mut [&mut ChatDecodeState],
    ids: &[usize],
) -> Vec<&'a mut ChatDecodeState> {
    let mut rest: &mut [&mut ChatDecodeState] = states;
    let mut out = Vec::with_capacity(ids.len());
    for (position, &id) in ids.iter().enumerate() {
        let (_skipped, tail) = rest.split_at_mut(id - position);
        let (taken, remainder) = tail.split_at_mut(1);
        out.push(&mut *taken[0]);
        rest = remainder;
    }
    out
}

/// CPU-only prompt journal for replay: prompt ids, their rope positions, and
/// the first decode text position.
#[derive(Clone)]
pub(crate) struct Qwen36PromptJournal {
    prompt_ids: Vec<u32>,
    prompt_positions: Vec<[usize; 3]>,
    next_text_position: usize,
}

/// Durable CPU-only continuation record. It deliberately owns no tensors,
/// cache views, device events, or physical sequence identities.
#[derive(Clone)]
pub(crate) struct Qwen36ReplayCheckpoint {
    prompt_journal: Qwen36PromptJournal,
    generated_ids: Vec<u32>,
    appended_tokens: usize,
    prefill_progress: usize,
    pending_token: Option<u32>,
    history_ids: Vec<u32>,
    decoder: IncrementalDecoder,
    tokens_generated: usize,
    track_history: bool,
    assembled: String,
    max_new_tokens: usize,
    next_text_position: usize,
    config: ChatGenerationConfig,
    rng: SimpleRng,
    draft_rng: SimpleRng,
    adaptive_mtp: crate::models::architectures::qwen36moe::mtp::AdaptiveMtp,
}

impl Qwen36ReplayCheckpoint {
    pub(crate) fn replay_tokens(&self) -> usize {
        self.appended_tokens
    }
}

pub struct ChatDecodeState {
    text_state: Qwen36TextRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    /// Model output awaiting sampling inside the current executor quantum.
    /// This slot is drained before the state is returned to the executor.
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    history_ids: Vec<u32>,
    decoder: IncrementalDecoder,
    tokens_generated: usize,
    track_history: bool,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
    next_text_position: usize,
    /// Scheduler-visible prompt cursor, independent of text position IDs.
    prefill_progress: usize,
    config: ChatGenerationConfig,
    rng: SimpleRng,
    /// Derived draft stream: speculative proposals must not perturb the
    /// target's draw sequence, so sampled drafting draws from its own fork.
    draft_rng: SimpleRng,
    /// DS9.3: logprob entries produced by the current decode step, drained
    /// by the registry right after the step. Cleared at each sample.
    pub(crate) pending_logprobs: Vec<crate::engine::TokenLogprob>,
    /// DS9.2: per-request constrained-decoding runtime, present only when
    /// the request asked for `response_format: json_object`.
    grammar: Option<crate::models::shared::sampling::GrammarRuntime>,
    /// MTP draft cache (one paged attention layer at model_layer =
    /// block_count) and the latest draft anchor — present only while a
    /// session decodes through the MTP quantum path.
    mtp_cache: Option<PhysicalPagedKvCache>,
    mtp_anchor_hidden: Option<Tensor>,
    /// DS9.4 per-request depth controller and its pending device-timed
    /// rounds. The controller only trains where device event timing is
    /// available; elsewhere the configured depth holds.
    adaptive_mtp: crate::models::architectures::qwen36moe::mtp::AdaptiveMtp,
    mtp_timings: Vec<timing::PendingRound>,
    /// Suspension support: the prompt journal, the append-only generated
    /// token journal (bounded by max_new_tokens), and a pending replay being
    /// recomputed span by span.
    prompt_journal: Qwen36PromptJournal,
    generated_ids: Vec<u32>,
    replay: Option<std::sync::Arc<Qwen36ReplayCheckpoint>>,
}

impl ChatDecodeState {
    /// The MTP cache cursor for this session, when it decoded through the
    /// MTP path. Always equal to `next_text_position` while MTP is active.
    pub(crate) fn mtp_cache_cursor(&self) -> Option<usize> {
        self.mtp_cache.as_ref().map(|cache| cache.context_len())
    }

    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    /// Committed output tokens for the session so far.
    pub(crate) fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    /// Whether the session reached a stop condition or its output cap.
    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    /// Tokens still to recompute before decode resumes, when a replay is
    /// pending on this session.
    pub(crate) fn replay_tokens(&self) -> Option<usize> {
        self.replay.as_ref().map(|saved| saved.appended_tokens)
    }

    /// Capture the durable CPU continuation record. Caller must fence the
    /// completed step before releasing the physical state. Grammar sessions
    /// cannot suspend: their recompute needs inputs the journal deliberately
    /// does not carry.
    pub(crate) fn replay_checkpoint(&self) -> Result<Qwen36ReplayCheckpoint> {
        if let Some(saved) = &self.replay {
            return Ok((**saved).clone());
        }
        if self.finished {
            return Err(Error::InvalidInput(
                "cannot suspend a finished Qwen3.5 sequence".into(),
            ));
        }
        if self.grammar.is_some() {
            return Err(Error::InvalidInput(
                "cannot suspend a grammar-constrained Qwen3.5 session".into(),
            ));
        }
        let appended_tokens = self.physical_kv.context_len();
        let known = self
            .prompt_journal
            .prompt_ids
            .len()
            .saturating_add(self.generated_ids.len());
        if appended_tokens > known || appended_tokens < self.prefill_progress {
            return Err(Error::InferenceError(
                "Qwen3.5 replay journal does not cover cache cursor".into(),
            ));
        }
        Ok(Qwen36ReplayCheckpoint {
            prompt_journal: self.prompt_journal.clone(),
            generated_ids: self.generated_ids.clone(),
            appended_tokens,
            prefill_progress: self.prefill_progress,
            pending_token: self.pending_token,
            history_ids: self.history_ids.clone(),
            decoder: self.decoder.clone(),
            tokens_generated: self.tokens_generated,
            track_history: self.track_history,
            assembled: self.assembled.clone(),
            max_new_tokens: self.max_new_tokens,
            next_text_position: self.next_text_position,
            config: self.config.clone(),
            rng: self.rng.clone(),
            draft_rng: self.draft_rng.clone(),
            adaptive_mtp: self.adaptive_mtp.clone(),
        })
    }

    pub(crate) fn uses_physical_kv(&self) -> bool {
        true
    }

    pub(crate) fn install_physical_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        let current = &self.physical_kv;
        if current.arena().id() != cache.arena().id()
            || current.context_len() != cache.context_len()
        {
            return Err(Error::InferenceError(
                "Qwen3.5 physical KV reservation does not continue the session".into(),
            ));
        }
        self.physical_kv = cache;
        Ok(())
    }

    pub(crate) fn take_physical_write_completions(
        &mut self,
    ) -> Vec<std::sync::Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        self.physical_kv.take_completed_writes()
    }

    pub(crate) fn begin_shared_step_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<Qwen36SharedStepCheckpoint> {
        self.begin_shared_step_quantum_with_mtp(cache, None)
    }

    /// Continuous-batch transaction checkpoint. The managed MTP cache, when
    /// the family loaded a draft head, joins the same transaction: the
    /// reservation must continue the session cursor and the rollback
    /// restores both caches.
    pub(crate) fn begin_shared_step_quantum_with_mtp(
        &mut self,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<Qwen36SharedStepCheckpoint> {
        if self.physical_kv.arena().id() != cache.arena().id()
            || self.physical_kv.context_len() != cache.context_len()
        {
            return Err(Error::InferenceError(
                "Qwen3.5 shared-step KV reservation does not continue the session".into(),
            ));
        }
        if let Some(mtp) = &mtp_cache {
            // The MTP domain may live in its own arena; continuity is a
            // cursor contract, not an arena identity contract.
            let expected = self
                .mtp_cache
                .as_ref()
                .map(|cache| cache.context_len())
                .unwrap_or(0);
            if mtp.context_len() != expected {
                return Err(Error::InferenceError(format!(
                    "Qwen3.5 MTP reservation cursor {} does not continue the session cursor {expected}",
                    mtp.context_len()
                )));
            }
        }
        Ok(Qwen36SharedStepCheckpoint {
            text_state: self.text_state.clone(),
            physical_kv: std::mem::replace(&mut self.physical_kv, cache),
            mtp_cache: std::mem::replace(&mut self.mtp_cache, mtp_cache),
            unconsumed_output: self.unconsumed_output.clone(),
            pending_token: self.pending_token,
            history_ids: self.history_ids.clone(),
            decoder: self.decoder.clone(),
            tokens_generated: self.tokens_generated,
            assembled: self.assembled.clone(),
            finished: self.finished,
            next_text_position: self.next_text_position,
            rng: self.rng.clone(),
            draft_rng: self.draft_rng.clone(),
            adaptive_mtp: self.adaptive_mtp.clone(),
            mtp_timings: self.mtp_timings.clone(),
            replay: self.replay.clone(),
        })
    }

    pub(crate) fn rollback_shared_step_quantum(&mut self, checkpoint: Qwen36SharedStepCheckpoint) {
        self.text_state = checkpoint.text_state;
        self.physical_kv = checkpoint.physical_kv;
        self.mtp_cache = checkpoint.mtp_cache;
        self.unconsumed_output = checkpoint.unconsumed_output;
        self.pending_token = checkpoint.pending_token;
        self.history_ids = checkpoint.history_ids;
        self.decoder = checkpoint.decoder;
        self.tokens_generated = checkpoint.tokens_generated;
        self.assembled = checkpoint.assembled;
        self.finished = checkpoint.finished;
        self.next_text_position = checkpoint.next_text_position;
        self.rng = checkpoint.rng;
        self.draft_rng = checkpoint.draft_rng;
        self.adaptive_mtp
            .restore_from_checkpoint(checkpoint.adaptive_mtp);
        self.mtp_timings = checkpoint.mtp_timings;
        self.replay = checkpoint.replay;
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .physical_tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(Error::InferenceError(
                "Qwen3.5 tensor-state sequence identity changed".into(),
            ));
        }
        self.physical_tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self.physical_tensor_sequence.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 physical state has no tensor sequence".into())
        })?;
        self.text_state.restore_tensor_domains(arena, sequence)
    }

    pub(crate) fn stage_tensor_state(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        let target_cursor = self.physical_kv.context_len() as u64;
        self.text_state.stage_tensor_domains(
            arena,
            PhysicalStateTransactionId::new(transaction)?,
            target_cursor,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub input_tokens_committed: usize,
    pub finished: bool,
}

/// One drafted block awaiting verification. Greedy rounds carry bare token
/// ids (exact prefix matching against the target argmax); sampled rounds
/// carry each proposal with its full draft distribution so the shared
/// verifier can run lossless rejection sampling.
pub(crate) enum DraftBlock {
    Greedy(Vec<u32>),
    Stochastic(Vec<SpeculativeDraft>),
}

impl DraftBlock {
    fn token_ids(&self) -> Vec<u32> {
        match self {
            DraftBlock::Greedy(tokens) => tokens.clone(),
            DraftBlock::Stochastic(proposals) => {
                proposals.iter().map(|proposal| proposal.token_id).collect()
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            DraftBlock::Greedy(tokens) => tokens.len(),
            DraftBlock::Stochastic(proposals) => proposals.len(),
        }
    }
}

pub(crate) struct Qwen36SharedStepCheckpoint {
    text_state: Qwen36TextRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    mtp_cache: Option<PhysicalPagedKvCache>,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    history_ids: Vec<u32>,
    decoder: IncrementalDecoder,
    tokens_generated: usize,
    assembled: String,
    finished: bool,
    next_text_position: usize,
    rng: SimpleRng,
    draft_rng: SimpleRng,
    adaptive_mtp: crate::models::architectures::qwen36moe::mtp::AdaptiveMtp,
    mtp_timings: Vec<timing::PendingRound>,
    replay: Option<std::sync::Arc<Qwen36ReplayCheckpoint>>,
}

#[derive(Debug, Clone)]
pub struct Qwen36TextConfig {
    pub architecture: String,
    pub block_count: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rope_dimension_sections: Vec<usize>,
    pub rope_dimension_count: usize,
    pub rope_freq_base: f64,
    pub attention_layer_norm_rms_epsilon: f64,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
    pub full_attention_interval: usize,
    /// Sparse-expert geometry; `None` for the dense GGUF family, `Some`
    /// when every layer's feed-forward is the sparse MoE block.
    pub moe_ffn: Option<Qwen36MoeFfnGeometry>,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    im_start: u32,
    im_end: u32,
    image_pad: u32,
    video_pad: u32,
    eos: u32,
    eos_alt: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

pub(crate) struct Qwen36Tokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    literal_special_tokens: Vec<(String, u32)>,
    chat_template: String,
    default_enable_thinking: bool,
    bos_token: Option<String>,
}

#[derive(Debug)]
struct GgufTokenizerMetadata {
    tokens: Vec<String>,
    token_types: Vec<u32>,
    merges: Vec<String>,
    pre_tokenizer: Option<String>,
    chat_template: String,
    eos_token_id: Option<u32>,
}

impl Qwen36Tokenizer {
    pub(crate) fn load(
        model_dir: &Path,
        variant: ModelVariant,
        loader: &GgufLoader,
    ) -> Result<Self> {
        let gguf_meta = parse_gguf_tokenizer_metadata(loader)?;
        let config = load_tokenizer_config_file(model_dir)?;
        let mut inner = match Tokenizer::from_path(model_dir) {
            Ok(inner) => inner,
            Err(_) => Tokenizer::from_gguf_bpe(
                &gguf_meta.tokens,
                &gguf_meta.merges,
                gguf_meta.pre_tokenizer.as_deref(),
                false,
            )?,
        };
        inner.register_gguf_token_types(&gguf_meta.tokens, &gguf_meta.token_types)?;
        let vocab_size = inner.vocab_size();

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        if let Some(cfg) = &config {
            for (id, entry) in &cfg.added_tokens_decoder {
                if let Ok(parsed) = id.parse::<u32>() {
                    token_to_id.insert(entry.content.clone(), parsed);
                }
            }
        }
        for (idx, token) in gguf_meta.tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| {
                Error::TokenizationError(format!("GGUF tokenizer id out of range: {idx}"))
            })?;
            token_to_id.entry(token.clone()).or_insert(id);
        }

        let id_for = |token: &str| token_to_id.get(token).copied();
        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let image_pad = id_for("<|image_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|image_pad|> token id".to_string())
        })?;
        let video_pad = id_for("<|video_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|video_pad|> token id".to_string())
        })?;

        let eos = config
            .as_ref()
            .and_then(|cfg| cfg.eos_token.as_deref())
            .and_then(id_for)
            .or(gguf_meta.eos_token_id)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");

        let chat_template = config
            .as_ref()
            .and_then(|cfg| cfg.chat_template.clone())
            .unwrap_or_else(|| gguf_meta.chat_template.clone());
        let default_enable_thinking = resolve_default_enable_thinking(&chat_template, variant);

        let mut literal_special_tokens: Vec<(String, u32)> = gguf_meta
            .tokens
            .iter()
            .zip(&gguf_meta.token_types)
            .enumerate()
            .filter_map(|(id, (token, token_type))| {
                matches!(token_type, 3 | 4).then_some((token.clone(), id as u32))
            })
            .collect();
        if let Some(cfg) = &config {
            literal_special_tokens.extend(cfg.added_tokens_decoder.iter().filter_map(
                |(id, entry)| id.parse::<u32>().ok().map(|id| (entry.content.clone(), id)),
            ));
        }
        literal_special_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });
        literal_special_tokens.dedup_by(|(left, _), (right, _)| left == right);

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                image_pad,
                video_pad,
                eos,
                eos_alt,
            },
            literal_special_tokens,
            chat_template,
            default_enable_thinking,
            bos_token: config.and_then(|cfg| cfg.bos_token),
        })
    }

    /// HF-native load path for checkpoints without a GGUF tokenizer
    /// (the qwen36moe FP8 safetensors bundle): `tokenizer.json` supplies
    /// the vocabulary and `tokenizer_config.json` the specials/template.
    pub(crate) fn load_hf(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        let config = load_tokenizer_config_file(model_dir)?;
        let inner = Tokenizer::from_path_requiring_tokenizer_json(model_dir)?;
        let vocab_size = inner.vocab_size();

        let id_for = |token: &str| inner.token_to_id(token);
        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let image_pad = id_for("<|image_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|image_pad|> token id".to_string())
        })?;
        let video_pad = id_for("<|video_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|video_pad|> token id".to_string())
        })?;

        let eos = config
            .as_ref()
            .and_then(|cfg| cfg.eos_token.as_deref())
            .and_then(id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");

        let chat_template = config
            .as_ref()
            .and_then(|cfg| cfg.chat_template.clone())
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Missing tokenizer chat template: no tokenizer_config.json chat_template"
                        .to_string(),
                )
            })?;
        let default_enable_thinking = resolve_default_enable_thinking(&chat_template, variant);

        let mut literal_special_tokens: Vec<(String, u32)> = config
            .as_ref()
            .map(|cfg| {
                cfg.added_tokens_decoder
                    .iter()
                    .filter_map(|(id, entry)| {
                        id.parse::<u32>().ok().map(|id| (entry.content.clone(), id))
                    })
                    .collect()
            })
            .unwrap_or_default();
        literal_special_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });
        literal_special_tokens.dedup_by(|(left, _), (right, _)| left == right);

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                image_pad,
                video_pad,
                eos,
                eos_alt,
            },
            literal_special_tokens,
            chat_template,
            default_enable_thinking,
            bos_token: config.and_then(|cfg| cfg.bos_token),
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        if self.literal_special_tokens.is_empty() {
            return self.inner.encode(text);
        }

        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let tail = &text[offset..];
            let mut next_match: Option<(usize, &str, u32)> = None;
            for (token, token_id) in &self.literal_special_tokens {
                if let Some(rel_idx) = tail.find(token) {
                    let candidate = (rel_idx, token.as_str(), *token_id);
                    match next_match {
                        None => next_match = Some(candidate),
                        Some((best_idx, best_token, _)) => {
                            if rel_idx < best_idx
                                || (rel_idx == best_idx && token.len() > best_token.len())
                            {
                                next_match = Some(candidate);
                            }
                        }
                    }
                }
            }

            let Some((rel_idx, matched_token, matched_id)) = next_match else {
                ids.extend(self.inner.encode(tail)?);
                break;
            };

            if rel_idx > 0 {
                ids.extend(self.inner.encode(&tail[..rel_idx])?);
            }
            ids.push(matched_id);
            offset += rel_idx + matched_token.len();
        }

        Ok(ids)
    }

    fn decode_token_delta(
        &self,
        decoder: &mut IncrementalDecoder,
        token_id: u32,
    ) -> Result<String> {
        if token_id as usize >= self.vocab_size {
            return Ok(String::new());
        }
        self.inner.decode_incrementally(decoder, token_id)
    }

    fn finish_decode(&self, decoder: &mut IncrementalDecoder) -> Result<String> {
        self.inner.finish_incremental_decode(decoder)
    }
}

/// Text-only execution core for the qwen36moe family: tokenizer, hybrid
/// trunk configuration, and the decode-state machinery the
/// [`Qwen36MoeChatModel`](super::chat::Qwen36MoeChatModel) wrapper drives.
/// Forked from the dense `qwen35` exec so neither family's changes can reach
/// the other.
pub(crate) struct Qwen36ChatExec {
    pub(crate) variant: ModelVariant,
    pub(crate) tokenizer: Qwen36Tokenizer,
    pub(crate) text_config: Qwen36TextConfig,
    pub(crate) text_model: Qwen36TextModel,
    /// MTP draft head when the checkpoint carries a validated manifest and
    /// the load policy enabled it. `None` keeps every path byte-identical
    /// to the non-MTP behavior.
    pub(crate) mtp_head: Option<crate::models::architectures::qwen36moe::mtp::Qwen36MtpHead>,
    /// Speculative rounds actually executed — telemetry and test evidence
    /// that the draft/verify path ran rather than the scalar fallback.
    pub(crate) mtp_speculative_rounds: std::sync::atomic::AtomicU64,
    /// DS9.4 speculative-envelope rounds executed across continuous rows.
    pub(crate) mtp_envelope_rounds: std::sync::atomic::AtomicU64,
    /// Whether the adaptive depth controller may train (the performance
    /// knob); the device check and head presence narrow it further.
    pub(crate) mtp_adaptive: bool,
    /// Persistent KV storage dtype — the MTP cache arena follows it.
    pub(crate) kv_storage_dtype: DType,
}

impl Qwen36ChatExec {
    pub(crate) fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub(crate) fn text_config(&self) -> &Qwen36TextConfig {
        &self.text_config
    }

    pub(crate) fn max_context_tokens(&self) -> Result<usize> {
        if self.text_config.context_length == 0 {
            return Err(Error::ModelLoadError(
                "Qwen3.5 checkpoint has a zero context length".into(),
            ));
        }
        Ok(self.text_config.context_length)
    }

    /// Hybrid retained-state contract shared by loading, scheduling, and the
    /// native model adapter.
    pub(crate) fn managed_composite_cache_contract(
        &self,
        attention_dtype: DType,
        preferred_page_tokens: usize,
    ) -> Result<InferenceStateContract> {
        crate::models::architectures::qwen36moe::cache::qwen35_composite_cache_contract_with_mtp(
            &self.text_config,
            attention_dtype,
            preferred_page_tokens,
            self.mtp_head.is_some(),
        )
    }

    pub(crate) fn chat_template(&self) -> &str {
        &self.tokenizer.chat_template
    }

    pub(crate) fn default_enable_thinking(&self) -> bool {
        self.tokenizer.default_enable_thinking
    }

    pub(crate) fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.prompt_token_ids_with_config(messages, &ChatGenerationConfig::default())
    }

    pub(crate) fn prompt_token_ids_with_config(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Vec<u32>> {
        Ok(self.prepare_text_prompt(messages, config)?.prompt_ids)
    }

    /// Text-only prompt preparation: ChatML render with the thinking
    /// contract, byte-safe encoding, and uniform per-token text positions.
    /// Callers that accept media must reject it before invoking this.
    pub(crate) fn prepare_text_prompt(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Qwen36PreparedPrompt> {
        let prompt = render_prompt(messages, config, self.default_enable_thinking())?;
        if prompt.contains(VIDEO_PAD_PLACEHOLDER) {
            return Err(Error::InvalidInput(
                "Qwen3.5 video inputs are not implemented yet".to_string(),
            ));
        }
        if prompt.contains(IMAGE_PAD_PLACEHOLDER) {
            return Err(Error::InvalidInput(
                "Qwen3.5 image placeholders require paired media inputs".to_string(),
            ));
        }
        let prompt_ids = self.tokenizer.encode_text(&prompt)?;
        let prompt_positions = build_text_positions(prompt_ids.len());
        Ok(Qwen36PreparedPrompt {
            next_text_position: prompt_positions.len(),
            prompt_ids,
            prompt_positions,
        })
    }

    pub(crate) fn supports_incremental_decode(&self) -> bool {
        true
    }

    pub(crate) fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    pub(crate) fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        let cfg = &self.text_config;
        let hidden = u64::try_from(cfg.embedding_length).ok();
        let ff = u64::try_from(cfg.feed_forward_length).ok();
        let q = cfg
            .attention_head_count
            .checked_mul(cfg.attention_key_length)
            .and_then(|width| width.checked_mul(2))
            .and_then(|width| u64::try_from(width).ok());
        let kv = cfg
            .attention_head_count_kv
            .checked_mul(cfg.attention_key_length)
            .and_then(|width| u64::try_from(width).ok());
        let conv = cfg
            .ssm_group_count
            .checked_mul(cfg.ssm_state_size)
            .and_then(|width| width.checked_mul(2))
            .and_then(|width| width.checked_add(cfg.ssm_inner_size))
            .and_then(|width| u64::try_from(width).ok());
        hidden
            .and_then(|hidden| hidden.checked_mul(8))
            .and_then(|base| base.checked_add(ff?.checked_mul(2)?))
            .and_then(|base| base.checked_add(q?))
            .and_then(|base| base.checked_add(kv?.checked_mul(2)?))
            .and_then(|base| base.checked_add(conv?))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| {
                Error::InvalidInput("Qwen3.5 continuous decode workspace overflow".into())
            })
    }

    pub(crate) fn begin_resumable_prefill_state_physical(
        &self,
        prepared: &Qwen36PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<ChatDecodeState> {
        if prepared.prompt_ids.is_empty() || cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 physical prefill requires a non-empty prompt and an empty reservation"
                    .into(),
            ));
        }
        let track_history =
            config.repetition_penalty > 1.0 || config.presence_penalty.abs() > f32::EPSILON;
        let mut rng = SimpleRng::new(config.seed);
        let draft_rng = rng.fork();
        Ok(ChatDecodeState {
            text_state: self.text_model.new_state(),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: None,
            history_ids: initial_penalty_history(
                &prepared.prompt_ids,
                max_new_tokens,
                track_history,
            ),
            decoder: IncrementalDecoder::new(true),
            tokens_generated: 0,
            track_history,
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            next_text_position: prepared.next_text_position,
            prefill_progress: 0,
            pending_logprobs: Vec::new(),
            config: config.clone(),
            rng,
            draft_rng,
            grammar: self.grammar_runtime(config),
            mtp_cache,
            mtp_anchor_hidden: None,
            adaptive_mtp: self.new_adaptive_mtp(),
            mtp_timings: Vec::new(),
            prompt_journal: Qwen36PromptJournal {
                prompt_ids: prepared.prompt_ids.clone(),
                prompt_positions: prepared.prompt_positions.clone(),
                next_text_position: prepared.next_text_position,
            },
            generated_ids: Vec::new(),
            replay: None,
        })
    }

    /// Fresh depth controller for a new session: it trains only where device
    /// event timing is available (CUDA) and the performance knob opted in;
    /// elsewhere the configured draft depth holds for the whole request.
    fn new_adaptive_mtp(&self) -> crate::models::architectures::qwen36moe::mtp::AdaptiveMtp {
        let enabled = self.mtp_adaptive
            && self.mtp_head.is_some()
            && self.text_model.device().is_cuda();
        crate::models::architectures::qwen36moe::mtp::AdaptiveMtp::new(
            enabled,
            self.mtp_head
                .as_ref()
                .map(|head| head.draft_depth())
                .unwrap_or(1),
        )
    }

    /// DS9.2 grammar runtime for this request, or `None` when the request
    /// did not ask for constrained decoding. Always-sampleable ids mirror
    /// `is_stop_token` so the decode loop can still finish inside the mask.
    fn grammar_runtime(
        &self,
        config: &ChatGenerationConfig,
    ) -> Option<crate::models::shared::sampling::GrammarRuntime> {
        if !config.constrain_json_object {
            return None;
        }
        Some(crate::models::shared::sampling::GrammarRuntime::new(
            std::sync::Arc::new(self.tokenizer.inner.clone()),
            [
                Some(self.tokenizer.specials.im_end),
                Some(self.tokenizer.specials.eos),
                self.tokenizer.specials.eos_alt,
            ]
            .into_iter()
            .flatten()
            .collect(),
        ))
    }

    pub(crate) fn continue_resumable_prefill_physical(
        &self,
        state: &mut ChatDecodeState,
        prepared: &Qwen36PreparedPrompt,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.prefill_progress != span_start
            || span_start >= span_end
            || span_end > prepared.prompt_ids.len()
            || state.finished
            || state.unconsumed_output.is_some()
            || state.pending_token.is_some()
            || state.tokens_generated != 0
            || state.physical_kv.context_len() != span_start
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 resumable prefill span [{span_start},{span_end}) is incompatible with cursor {} and prompt length {}",
                state.prefill_progress,
                prepared.prompt_ids.len()
            )));
        }
        let complete = span_end == prepared.prompt_ids.len();
        let mut logits = None;
        let mut cursor = span_start;
        let chunk_size = qwen35_prefill_chunk_size();
        while cursor < span_end {
            let (segment_end, image_segment) = next_prefill_segment_end(
                &prepared.prompt_ids,
                cursor,
                span_end,
                self.tokenizer.specials.image_pad,
                chunk_size,
            )?;
            if image_segment {
                // Text-only family: prepared prompts never carry vision rows.
                return Err(Error::InvalidInput(
                    "Qwen3.5 image placeholder has no vision input".into(),
                ));
            }
            let compute_logits = complete && segment_end == span_end;
            logits = self.text_model.prefill_token_ids_physical(
                &prepared.prompt_ids[cursor..segment_end],
                &prepared.prompt_positions[cursor..segment_end],
                &mut state.text_state,
                &mut state.physical_kv,
                compute_logits,
            )?;
            cursor = segment_end;
        }
        if state.physical_kv.context_len() != span_end {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 resumable prefill committed physical cursor {} instead of {span_end}",
                state.physical_kv.context_len()
            )));
        }
        state.prefill_progress = span_end;
        if complete {
            state.unconsumed_output = Some(logits.ok_or_else(|| {
                Error::InferenceError("Qwen3.5 final prefill span produced no logits".into())
            })?);
        }
        Ok(complete)
    }

    pub(crate) fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.replay.is_some() {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode cannot run before replay completes".into(),
            ));
        }
        if state.finished || state.tokens_generated >= state.max_new_tokens {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.clone(),
                tokens_generated: state.tokens_generated,
                input_tokens_committed: 0,
                finished: true,
            });
        }

        let mut input_tokens_committed = 0usize;
        if let Some(pending) = state.pending_token.take() {
            state.unconsumed_output = Some(self.text_model.forward_token_id_at_physical(
                pending,
                [state.next_text_position; 3],
                &mut state.text_state,
                &mut state.physical_kv,
            )?);
            state.next_text_position += 1;
            input_tokens_committed = 1;
        }

        let history: &[u32] = if state.track_history {
            &state.history_ids
        } else {
            &[]
        };
        state.pending_logprobs.clear();
        let next = {
            let output = state.unconsumed_output.take().ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3.5 decode quantum has no unconsumed model output".to_string(),
                )
            })?;
            let (token, raw_logprobs) = if let Some(grammar) = state.grammar.as_mut() {
                grammar.sample_token(
                    &output,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?
            } else {
                sample_next_token_with_logprobs(
                    &output,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?
            };
            if let Some(raw) = raw_logprobs {
                let entry = crate::models::shared::sampling::resolve_token_logprob(
                    &self.tokenizer.inner,
                    &raw,
                )?;
                if !self.is_stop_token(token, &state.config) {
                    state.pending_logprobs.push(entry);
                }
            }
            token
        };
        if self.is_stop_token(next, &state.config) {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.clone(),
                tokens_generated: state.tokens_generated,
                input_tokens_committed,
                finished: true,
            });
        }

        let mut delta = self
            .tokenizer
            .decode_token_delta(&mut state.decoder, next)?;
        if state.track_history {
            state.history_ids.push(next);
        }
        state.generated_ids.push(next);
        state.tokens_generated = state.tokens_generated.saturating_add(1);
        state.assembled.push_str(&delta);
        state.pending_token = Some(next);
        if state.tokens_generated >= state.max_new_tokens {
            state.finished = true;
            let suffix = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&suffix);
            delta.push_str(&suffix);
        }
        let final_text = if state.finished {
            state.assembled.clone()
        } else {
            String::new()
        };

        Ok(ChatDecodeStep {
            delta,
            text: final_text,
            tokens_generated: state.tokens_generated,
            input_tokens_committed,
            finished: state.finished,
        })
    }

    pub(crate) fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        validate_continuous_decode_rows(states)?;
        let mut token_ids = Vec::with_capacity(states.len());
        let mut positions = Vec::with_capacity(states.len());
        let mut text_states = Vec::with_capacity(states.len());
        let mut caches = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            token_ids.push(state.pending_token.take().expect("pending token checked"));
            positions.push([state.next_text_position; 3]);
            text_states.push(&mut state.text_state);
            caches.push(&mut state.physical_kv);
        }
        let logits = self.text_model.forward_token_ids_batch_at_physical(
            &token_ids,
            &positions,
            &mut text_states,
            &mut caches,
        )?;
        drop(text_states);
        drop(caches);

        // An all-greedy CUDA batch reads every row's token back at once.
        let vocab_size = self.tokenizer.vocab_size;
        let batch_greedy = if logits.device().is_cuda()
            && vocab_size > 0
            && states.iter().all(|state| {
                state.grammar.is_none()
                    && !state.config.logprobs
                    && deterministic_greedy(&state.config)
            }) {
            let rows = logits.i((.., 0))?;
            let rows = if vocab_size < rows.dim(1)? {
                rows.narrow(1, 0, vocab_size)?
            } else {
                rows
            };
            Some(device_greedy_rows(&rows)?)
        } else {
            None
        };

        let mut sampled = Vec::with_capacity(states.len());
        for (row, state) in states.iter_mut().enumerate() {
            let history = if state.track_history {
                state.history_ids.as_slice()
            } else {
                &[]
            };
            state.pending_logprobs.clear();
            let row_logits = logits.i((row, 0))?;
            let (token, raw_logprobs) = if let Some(tokens) = &batch_greedy {
                // A row without a finite logit takes the reporting slow path.
                match tokens[row] {
                    Some(token) => (token, None),
                    None => (argmax_clamped(&row_logits, vocab_size)?, None),
                }
            } else if let Some(grammar) = state.grammar.as_mut() {
                grammar.sample_token(
                    &row_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?
            } else {
                sample_next_token_with_logprobs(
                    &row_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?
            };
            let entry = raw_logprobs
                .map(|raw| {
                    crate::models::shared::sampling::resolve_token_logprob(
                        &self.tokenizer.inner,
                        &raw,
                    )
                })
                .transpose()?;
            sampled.push((token, entry));
            state.next_text_position = state.next_text_position.saturating_add(1);
        }
        let mut steps = Vec::with_capacity(states.len());
        for (state, (next, logprob_entry)) in states.iter_mut().zip(sampled) {
            let is_stop = self.is_stop_token(next, &state.config);
            if let Some(entry) = logprob_entry {
                if !is_stop {
                    state.pending_logprobs.push(entry);
                }
            }
            if state.track_history && !is_stop {
                state.history_ids.push(next);
            }
            if !is_stop {
                state.pending_token = Some(next);
            }
            let delta = self.publish_token(state, next)?;
            steps.push(ChatDecodeStep {
                delta,
                text: if state.finished {
                    state.assembled.clone()
                } else {
                    String::new()
                },
                tokens_generated: state.tokens_generated,
                input_tokens_committed: 1,
                finished: state.finished,
            });
        }
        Ok(steps)
    }

    fn publish_token(&self, state: &mut ChatDecodeState, token: u32) -> Result<String> {
        // Append-only replay journal: every published token, bounded by the
        // output cap. A trailing stop token is journaled but never counted.
        state.generated_ids.push(token);
        if self.is_stop_token(token, &state.config) {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(delta);
        }
        let mut delta = self
            .tokenizer
            .decode_token_delta(&mut state.decoder, token)?;
        state.tokens_generated = state.tokens_generated.saturating_add(1);
        state.assembled.push_str(&delta);
        if state.tokens_generated >= state.max_new_tokens {
            state.finished = true;
            let suffix = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&suffix);
            delta.push_str(&suffix);
        }
        Ok(delta)
    }

    fn is_stop_token(&self, token_id: u32, config: &ChatGenerationConfig) -> bool {
        token_id == self.tokenizer.specials.im_end
            || token_id == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(token_id)
            || config.stop_token_ids.contains(&token_id)
    }
}

impl Qwen36ChatExec {
    /// Whether the MTP quantum path may drive this decode step: a loaded
    /// draft head plus an unconstrained request. Greedy requests accept by
    /// exact prefix match; sampled requests (any temperature, penalties,
    /// top-k/top-p) accept through the shared lossless rejection sampler,
    /// which applies the same transforms to draft proposals and target
    /// rows. Grammar and logprobs stay scalar-only: constrained decoding
    /// must mask every sample, and logprob requests need the scalar
    /// sampler's raw-logit accounting for tokens speculation never shows.
    fn mtp_active(&self, state: &ChatDecodeState) -> bool {
        let config = &state.config;
        self.mtp_head.is_some()
            && state.mtp_cache.is_some()
            && state.grammar.is_none()
            && !config.logprobs
    }

    /// Greedy argmax over one logits row, clamped to the tokenizer vocab.
    /// Uses the scalar sampler's `argmax_clamped` so tie-breaking and
    /// non-finite handling are bit-identical to the non-MTP path.
    fn greedy_token(&self, logits: &Tensor) -> Result<u32> {
        argmax_clamped(&logits.flatten_all()?, self.tokenizer.vocab_size)
    }

    /// Ensure the per-session MTP cache exists. V1 allocates a standalone
    /// one-layer arena sized for the model's full context; the managed MTP
    /// KV domain (the qwen3.8 contract shape) replaces this when the
    /// continuous-batch envelope lands.
    fn ensure_mtp_cache(&self, state: &mut ChatDecodeState) -> Result<()> {
        if state.mtp_cache.is_some() {
            return Ok(());
        }
        let device = self.text_model.device().clone();
        let cfg = &self.text_config;
        let model_layer = u32::try_from(cfg.block_count)
            .map_err(|_| Error::InvalidInput("Qwen3.5 MTP layer id exceeds u32".into()))?;
        let backend = if device.is_cuda() {
            BackendKind::Cuda
        } else if device.is_metal() {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        };
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let id = KvArenaId {
            model_instance: crate::engine::ModelInstanceId::new(nanos),
            backend,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let page_tokens = 64usize;
        let config = KvArenaConfig {
            id,
            group,
            page_tokens: page_tokens as u32,
            capacity_pages: (cfg.context_length.max(1) as u32).div_ceil(page_tokens as u32),
            growth: None,
            dtype: self.kv_storage_dtype,
            layers: vec![KvLayerConfig {
                binding: KvLayerBinding {
                    model_layer,
                    physical_layer: 0,
                },
                num_kv_heads: cfg.attention_head_count_kv as u32,
                key_head_dim: cfg.attention_key_length as u32,
                value_head_dim: cfg.attention_value_length as u32,
            }],
        };
        let binding = KvLayerBinding {
            model_layer,
            physical_layer: 0,
        };
        let blocks = (0..config.capacity_pages)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let arena: std::sync::Arc<dyn KvArena> = if backend != BackendKind::Cpu {
            std::sync::Arc::new(CandleAcceleratorKvArena::new_mutation_only(config, device)?)
        } else {
            std::sync::Arc::new(CpuKvArena::new(config)?)
        };
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let arena: std::sync::Arc<dyn KvArena> = std::sync::Arc::new(CpuKvArena::new(config)?);
        state.mtp_cache = Some(PhysicalPagedKvCache::new(arena, vec![binding], blocks, 0)?);
        Ok(())
    }

    /// One decode quantum (up to `input_budget` committed tokens). With an
    /// MTP head and a qualifying request this runs speculative rounds —
    /// draft `depth` tokens through the head, verify them with the target,
    /// commit the accepted prefix; the fallback loops the scalar step.
    pub(crate) fn decode_quantum(
        &self,
        state: &mut ChatDecodeState,
        input_budget: usize,
    ) -> Result<ChatDecodeStep> {
        if state.replay.is_some() {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode cannot run before replay completes".into(),
            ));
        }
        let Some(head) = self.mtp_head.as_ref() else {
            return self.decode_quantum_scalar(state, input_budget);
        };
        self.ensure_mtp_cache(state)?;
        if !self.mtp_active(state) {
            return self.decode_quantum_scalar(state, input_budget);
        }
        if state.finished || state.tokens_generated >= state.max_new_tokens {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.clone(),
                tokens_generated: state.tokens_generated,
                input_tokens_committed: 0,
                finished: true,
            });
        }

        let mut delta = String::new();

        // Bootstrap: publish the first token by sampling the stored prefill
        // logits — no KV commit, exactly like the scalar decode step's
        // prologue. The pending token is what the first round forwards.
        let mut published_bootstrap = false;
        if state.pending_token.is_none() {
            published_bootstrap = true;
            let output = state.unconsumed_output.take().ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3.5 MTP quantum has neither pending token nor prefill output".into(),
                )
            })?;
            // Same penalty history as every other sampled step: the prompt
            // ids are already in it, and the bootstrap token is no exception.
            let history: &[u32] = if state.track_history {
                state.history_ids.as_slice()
            } else {
                &[]
            };
            let next = sample_next_token(
                &logits_last_row(&output)?,
                self.tokenizer.vocab_size,
                &state.config,
                history,
                &mut state.rng,
            )?;
            // Publish through the shared choke point so the cap, stop, and
            // replay-journal handling cannot drift from the scalar step.
            let step_delta = self.publish_token(state, next)?;
            if !self.is_stop_token(next, &state.config) {
                state.pending_token = Some(next);
            }
            delta.push_str(&step_delta);
        }

        // The bootstrap published one token, exactly like the scalar
        // quantum's first step — it counts against the budget.
        let mut committed = usize::from(published_bootstrap);
        let budget = input_budget.max(1);
        // The anchor is seeded by the first scalar tail: it forwards the
        // pending token, samples the next one, and writes the pair
        // (embed(next), h(pos)) — exactly the qwen3.8 bootstrap shape.
        let mut anchor_ready = state.mtp_anchor_hidden.is_some();
        while committed < budget && !state.finished {
            // Output-capped remaining slice: the controller's observation
            // budget is the round's own remaining slice, not the raw grant.
            let remaining = (budget - committed).min(
                state
                    .max_new_tokens
                    .saturating_sub(state.tokens_generated)
                    .max(1),
            );
            self.observe_completed_mtp_timings(state);
            let timer = if state.adaptive_mtp.can_train(remaining) {
                timing::RoundTimer::start(self.text_model.device())
            } else {
                None
            };
            let depth = if anchor_ready {
                state.adaptive_mtp.depth(remaining)
            } else {
                0
            };
            let round_tokens = if depth == 0 {
                delta.push_str(&self.mtp_scalar_tail(state, head)?);
                anchor_ready = true;
                1
            } else {
                let (tokens, round_delta) = self.mtp_speculative_round(state, head, depth)?;
                delta.push_str(&round_delta);
                tokens
            };
            committed += round_tokens;
            self.observe_completed_mtp_timings(state);
            if let Some(pending) =
                timer.and_then(|timer| timer.finish(depth, round_tokens, remaining))
            {
                if state.mtp_timings.len() == 4 {
                    state.mtp_timings.remove(0);
                }
                state.mtp_timings.push(pending);
            }
        }
        Ok(ChatDecodeStep {
            delta,
            text: state.assembled.clone(),
            tokens_generated: state.tokens_generated,
            input_tokens_committed: committed,
            finished: state.finished,
        })
    }

    fn decode_quantum_scalar(
        &self,
        state: &mut ChatDecodeState,
        input_budget: usize,
    ) -> Result<ChatDecodeStep> {
        if state.replay.is_some() {
            return Err(Error::InvalidInput(
                "Qwen3.5 decode cannot run before replay completes".into(),
            ));
        }
        let mut delta = String::new();
        let mut committed = 0usize;
        for _ in 0..input_budget.max(1) {
            let step = self.decode_step(state)?;
            delta.push_str(&step.delta);
            committed += step.input_tokens_committed;
            if step.finished {
                break;
            }
        }
        Ok(ChatDecodeStep {
            delta,
            text: state.assembled.clone(),
            tokens_generated: state.tokens_generated,
            input_tokens_committed: committed,
            finished: state.finished,
        })
    }

    /// Scalar tail: forward the pending token, sample, and write the MTP
    /// pair `(embed(next), h(pos))` at position pos.
    fn mtp_scalar_tail(
        &self,
        state: &mut ChatDecodeState,
        head: &crate::models::architectures::qwen36moe::mtp::Qwen36MtpHead,
    ) -> Result<String> {
        let pending = state.pending_token.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 MTP scalar tail has no pending token".into())
        })?;
        let position = state.next_text_position;
        let hidden = self.text_model.forward_token_id_hidden_at_physical(
            pending,
            [position; 3],
            &mut state.text_state,
            &mut state.physical_kv,
        )?;
        let normalized_hidden = self.text_model.normalize_hidden(&hidden)?;
        let logits = self.text_model.forward_hidden_to_logits(&hidden)?;
        let history = if state.track_history {
            state.history_ids.as_slice()
        } else {
            &[]
        };
        state.pending_logprobs.clear();
        let next = sample_next_token(
            &logits,
            self.tokenizer.vocab_size,
            &state.config,
            history,
            &mut state.rng,
        )?;
        if state.track_history {
            state.history_ids.push(next);
        }
        let mtp_cache = state
            .mtp_cache
            .as_mut()
            .ok_or_else(|| Error::InferenceError("Qwen3.5 MTP cache lost".into()))?;
        let embedding = self.text_model.embed_token_ids(&[next])?;
        let anchor = head.forward_step(&embedding, &normalized_hidden, [position; 3], mtp_cache)?;
        state.mtp_anchor_hidden = Some(anchor);
        state.pending_token = Some(next);
        state.next_text_position += 1;
        let mut delta = String::new();
        delta.push_str(&self.publish_token(state, next)?);
        Ok(delta)
    }

    /// Speculative round: draft `depth` tokens, verify with the target, and
    /// commit the accepted prefix. Returns (committed tokens, their text).
    fn mtp_speculative_round(
        &self,
        state: &mut ChatDecodeState,
        head: &crate::models::architectures::qwen36moe::mtp::Qwen36MtpHead,
        depth: usize,
    ) -> Result<(usize, String)> {
        self.mtp_speculative_rounds
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let drafted = self.mtp_draft(state, head, depth)?;
        self.mtp_verify_and_commit(state, head, &drafted)
    }

    /// Draft `depth` tokens provisionally through the head; the draft's MTP
    /// rows are discarded before verification rewrites the canonical pairs.
    /// Greedy requests keep the no-distribution fast path; sampled requests
    /// propose through the shared lossless sampler on the row's own draft
    /// RNG stream, retaining each proposal's distribution for verification.
    fn mtp_draft(
        &self,
        state: &mut ChatDecodeState,
        head: &crate::models::architectures::qwen36moe::mtp::Qwen36MtpHead,
        depth: usize,
    ) -> Result<DraftBlock> {
        let position = state.next_text_position;
        let anchor = state.mtp_anchor_hidden.clone().ok_or_else(|| {
            Error::InferenceError("Qwen3.5 MTP round has no anchor".into())
        })?;
        let stochastic = state.config.temperature > 1e-5;
        let mut draft_history = if state.track_history {
            state.history_ids.clone()
        } else {
            Vec::new()
        };
        let mut draft_proposals = Vec::with_capacity(depth);

        let continuation_positions: Vec<[usize; 3]> =
            (0..depth - 1).map(|offset| [position + offset; 3]).collect();
        let mtp_cache = state
            .mtp_cache
            .as_mut()
            .ok_or_else(|| Error::InferenceError("Qwen3.5 MTP cache lost".into()))?;
        let checkpoint = mtp_cache.logical_checkpoint();
        let draft_rng_checkpoint = state.draft_rng.clone();
        let vocab_size = self.tokenizer.vocab_size;
        let drafted = head.draft_recurrently(
            &self.text_model,
            &anchor,
            depth,
            &continuation_positions,
            mtp_cache,
            |logits| {
                if !stochastic {
                    return crate::models::architectures::qwen36moe::mtp::greedy_argmax(
                        logits,
                        vocab_size,
                    );
                }
                let mut values = logits_to_vec(&logits.i((0, 0))?)?;
                truncate_logits_to_vocab(&mut values, vocab_size);
                let proposal = propose_speculative_draft(
                    &values,
                    &state.config,
                    &mut draft_history,
                    &mut state.draft_rng,
                )?;
                let token_id = proposal.token_id;
                draft_proposals.push(proposal);
                Ok(token_id)
            },
        );
        match drafted {
            Ok(tokens) => {
                // On success the advanced draft stream stays — proposals
                // consumed it by design. Failures restore the checkpoint
                // below, so a retried round redraws the same proposals.
                mtp_cache.restore_logical_checkpoint(checkpoint)?;
                Ok(if stochastic {
                    DraftBlock::Stochastic(draft_proposals)
                } else {
                    DraftBlock::Greedy(tokens)
                })
            }
            Err(error) => {
                mtp_cache.restore_logical_checkpoint(checkpoint)?;
                // Transactional RNG: a failed round never moves the draft
                // stream, so the retried round redraws the same proposals.
                state.draft_rng = draft_rng_checkpoint;
                Err(error)
            }
        }
    }

    /// Verify a pre-drafted block against the target and commit the accepted
    /// prefix. Shared by the solo quantum and the continuous speculative
    /// envelope, so a row's round semantics are identical wherever it runs.
    fn mtp_verify_and_commit(
        &self,
        state: &mut ChatDecodeState,
        head: &crate::models::architectures::qwen36moe::mtp::Qwen36MtpHead,
        drafted: &DraftBlock,
    ) -> Result<(usize, String)> {
        let depth = drafted.len();
        let drafted_tokens = drafted.token_ids();
        let pending = state.pending_token.ok_or_else(|| {
            Error::InferenceError("Qwen3.5 MTP round has no pending token".into())
        })?;
        let position = state.next_text_position;

        // Sequential target verification: one forward per candidate token,
        // snapshotting linear state after each so any accepted prefix can be
        // installed without re-running weights.
        let target_inputs = std::iter::once(pending)
            .chain(drafted_tokens.iter().copied())
            .collect::<Vec<_>>();
        let mut verify_hiddens = Vec::with_capacity(target_inputs.len());
        let mut verify_logits = Vec::with_capacity(target_inputs.len());
        let mut snapshots = Vec::with_capacity(target_inputs.len() + 1);
        snapshots.push(state.text_state.snapshot_linear_states()?);
        for (offset, &token) in target_inputs.iter().enumerate() {
            let hidden = self.text_model.forward_token_id_hidden_at_physical(
                token,
                [position + offset; 3],
                &mut state.text_state,
                &mut state.physical_kv,
            )?;
            let normalized = self.text_model.normalize_hidden(&hidden)?;
            let logits = self.text_model.project_hidden_span(&hidden)?;
            verify_hiddens.push(normalized);
            verify_logits.push(logits);
            snapshots.push(state.text_state.snapshot_linear_states()?);
        }

        // The first emitted token lands at position+1: the pending token
        // itself is never re-emitted. Greedy rounds accept by exact prefix
        // match against the target argmax; sampled rounds run lossless
        // rejection sampling over host logits rows (penalties, temperature,
        // top-k/top-p applied identically to draft and target rows).
        let emitted = if let DraftBlock::Stochastic(proposals) = drafted {
            let mut host_rows = Vec::with_capacity(verify_logits.len());
            for logits in &verify_logits {
                let mut values = logits_to_vec(&logits_last_row(logits)?)?;
                truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                host_rows.push(values);
            }
            let mut verification_history = if state.track_history {
                state.history_ids.clone()
            } else {
                Vec::new()
            };
            verify_speculative_proposals(
                proposals,
                &host_rows,
                &state.config,
                &mut verification_history,
                &mut state.rng,
            )?
            .emitted_tokens
        } else {
            let mut emitted = Vec::with_capacity(depth + 1);
            for row in 0..=depth {
                let token = self.greedy_token(&verify_logits[row])?;
                emitted.push(token);
                if row == depth || token != drafted_tokens[row] {
                    break;
                }
            }
            emitted
        };
        let remaining_outputs = state.max_new_tokens.saturating_sub(state.tokens_generated);
        let mut kept: Vec<u32> = Vec::with_capacity(emitted.len());
        for &token in &emitted {
            if kept.len() >= remaining_outputs {
                break;
            }
            kept.push(token);
            if self.is_stop_token(token, &state.config) {
                break;
            }
        }
        let count = kept.len();
        if count == 0 {
            state.finished = true;
            return Ok((0, String::new()));
        }

        // Commit only the accepted prefix: install the linear states from
        // the matching snapshot and drop the rejected KV rows. Full
        // attention keeps its already-written rows; they sit beyond the
        // truncated cursor and are rewritten by later steps.
        state.text_state.restore_linear_states(&snapshots[count])?;
        state.physical_kv.truncate_verified_prefix(position + count)?;

        // Rebuild the draft state over the canonical commit: pair i = (embed
        // of kept[i] — the token at position+i+1 — , the target hidden at
        // position+i).
        let mtp_cache = state
            .mtp_cache
            .as_mut()
            .ok_or_else(|| Error::InferenceError("Qwen3.5 MTP cache lost".into()))?;
        let mut anchor = state.mtp_anchor_hidden.clone().ok_or_else(|| {
            Error::InferenceError("Qwen3.5 MTP anchor lost".into())
        })?;
        for (row, &token) in kept.iter().enumerate() {
            let embedding = self.text_model.embed_token_ids(&[token])?;
            anchor = head.forward_step(
                &embedding,
                &verify_hiddens[row],
                [position + row; 3],
                mtp_cache,
            )?;
        }
        state.mtp_anchor_hidden = Some(anchor);
        state.pending_token = kept.last().copied();
        state.next_text_position += count;
        if state.track_history {
            state.history_ids.extend_from_slice(&kept);
        }
        let mut delta = String::new();
        for &token in &kept {
            delta.push_str(&self.publish_token(state, token)?);
            if state.finished {
                break;
            }
        }
        Ok((count, delta))
    }

    /// Whether this exec carries a loaded MTP draft head — the load-time
    /// gate behind the continuous speculative envelope profile.
    pub(crate) fn has_mtp_head(&self) -> bool {
        self.mtp_head.is_some()
    }

    /// Rebuild a decode state from a durable checkpoint over fresh cache
    /// reservations. No device state is restored: the appended KV rows and
    /// the MTP draft domain are recomputed by replay spans before decode
    /// resumes.
    pub(crate) fn begin_replay_state_physical(
        &self,
        saved: &Qwen36ReplayCheckpoint,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<ChatDecodeState> {
        if cache.context_len() != 0
            || mtp_cache
                .as_ref()
                .is_some_and(|cache| cache.context_len() != 0)
            || self.mtp_head.is_some() != mtp_cache.is_some()
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 replay requires fresh matching cache reservations".into(),
            ));
        }
        Ok(ChatDecodeState {
            text_state: self.text_model.new_state(),
            physical_kv: cache,
            mtp_cache,
            mtp_anchor_hidden: None,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: saved.pending_token,
            history_ids: saved.history_ids.clone(),
            decoder: saved.decoder.clone(),
            tokens_generated: saved.tokens_generated,
            track_history: saved.track_history,
            assembled: saved.assembled.clone(),
            max_new_tokens: saved.max_new_tokens,
            finished: false,
            next_text_position: saved.next_text_position,
            prefill_progress: saved.prefill_progress,
            config: saved.config.clone(),
            rng: saved.rng.clone(),
            draft_rng: saved.draft_rng.clone(),
            pending_logprobs: Vec::new(),
            grammar: None,
            adaptive_mtp: saved.adaptive_mtp.clone(),
            mtp_timings: Vec::new(),
            prompt_journal: saved.prompt_journal.clone(),
            generated_ids: saved.generated_ids.clone(),
            replay: (saved.appended_tokens > 0).then(|| std::sync::Arc::new(saved.clone())),
        })
    }

    /// Rebuild one scheduler span without sampling or emitting output. The
    /// span recomputes target rows over the prompt-plus-generated journal
    /// and, with a loaded head, rebuilds the MTP draft pairs whose
    /// successors are already known. The pending token is the last span's
    /// MTP successor; it is not forwarded to the target cache.
    pub(crate) fn continue_replay_physical(
        &self,
        state: &mut ChatDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        let saved = state
            .replay
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Qwen3.5 state has no pending replay".into()))?;
        if state.physical_kv.context_len() != span_start
            || span_end <= span_start
            || span_end > saved.appended_tokens
        {
            return Err(Error::InvalidInput(
                "Qwen3.5 replay span does not continue its append cursor".into(),
            ));
        }
        let prompt_len = saved.prompt_journal.prompt_ids.len();
        let known_len = prompt_len + saved.generated_ids.len();
        let token_at = |index: usize| {
            if index < prompt_len {
                saved.prompt_journal.prompt_ids.get(index).copied()
            } else if index <= known_len.saturating_sub(1) {
                saved.generated_ids.get(index - prompt_len).copied()
            } else {
                None
            }
        };
        // The last journaled token is the pending one: it is the MTP
        // successor at the append cursor but is not forwarded by the target.
        let ids: Vec<_> = (span_start..span_end).filter_map(token_at).collect();
        let positions: Vec<_> = (span_start..span_end)
            .map(|index| {
                if index < prompt_len {
                    saved.prompt_journal.prompt_positions[index]
                } else {
                    [saved.prompt_journal.next_text_position + index - prompt_len; 3]
                }
            })
            .collect();
        if ids.len() != positions.len() {
            return Err(Error::InferenceError(
                "Qwen3.5 replay journal has a gap inside the scheduled span".into(),
            ));
        }
        let prompt_complete = saved.prefill_progress == prompt_len;
        let chunk_size = qwen35_prefill_chunk_size();
        for start in (span_start..span_end).step_by(chunk_size) {
            let end = (start + chunk_size).min(span_end);
            let chunk_ids = &ids[start - span_start..end - span_start];
            let chunk_positions = &positions[start - span_start..end - span_start];
            let output = self.text_model.prefill_token_ids_with_hidden_physical(
                chunk_ids,
                chunk_positions,
                &mut state.text_state,
                &mut state.physical_kv,
                end == saved.appended_tokens && saved.pending_token.is_none(),
            )?;
            if let (Some(head), Some(mtp)) = (&self.mtp_head, state.mtp_cache.as_mut()) {
                // The original session's prefill writes no MTP pairs — pairs
                // exist only for decode rows (sequence index >= prompt_len),
                // each pairing a row's post-norm hidden with its successor's
                // embedding. The pending token supplies the final successor
                // without being forwarded to the target cache.
                let pair_rows: Vec<usize> = (start..end)
                    .filter(|&index| index >= prompt_len)
                    .collect();
                let count = pair_rows.len();
                if count > 0 {
                    let successors = pair_rows
                        .iter()
                        .map(|&index| {
                            token_at(index + 1).ok_or_else(|| {
                                Error::InferenceError(
                                    "Qwen3.5 replay journal lost an MTP successor".into(),
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let embeddings = self.text_model.embed_token_ids(&successors)?;
                    // Pair rows are the chunk's trailing decode rows; their
                    // hiddens start at the first pair row's span offset.
                    let hidden_offset = pair_rows[0] - span_start;
                    let predecessors = self.text_model.normalize_hidden(
                        &output.hidden_states.narrow(1, hidden_offset, count)?,
                    )?;
                    let mut last_hidden = None;
                    for (row, &index) in pair_rows.iter().enumerate() {
                        let hidden = head.forward_step(
                            &embeddings.narrow(1, row, 1)?,
                            &predecessors.narrow(1, row, 1)?,
                            positions[index - span_start],
                            mtp,
                        )?;
                        last_hidden = Some(hidden);
                    }
                    if end == saved.appended_tokens && prompt_complete {
                        // The final pair seeds the resumed draft anchor
                        // exactly like a scalar tail would.
                        state.mtp_anchor_hidden = last_hidden;
                    }
                }
            }
            if end == saved.appended_tokens && prompt_complete && saved.pending_token.is_none() {
                state.unconsumed_output = output.logits;
            }
        }
        let finished = span_end == saved.appended_tokens;
        if finished {
            state.replay = None;
        }
        Ok(finished)
    }

    /// Drain resolved device-event timings into the depth controller. The
    /// bounded pending queue also rides the quantum checkpoint, so a
    /// cancelled quantum restores observations and policy together.
    fn observe_completed_mtp_timings(&self, state: &mut ChatDecodeState) {
        let mut index = 0;
        while index < state.mtp_timings.len() {
            if let Some(elapsed) = state.mtp_timings[index].try_elapsed() {
                let completed = state.mtp_timings.remove(index);
                state.adaptive_mtp.observe(
                    completed.depth,
                    completed.committed,
                    elapsed,
                    completed.budget,
                );
            } else {
                index += 1;
            }
        }
    }

    /// DS9.4: one shared speculative envelope for continuous rows. Every row
    /// must be decodable exactly as the scalar batch requires (the executor
    /// only grants envelope quanta to rows whose prefill bootstrap already
    /// ran) and MTP-resident. A row without a seeded anchor sits out its
    /// first round as a scalar tail, and any row whose request disqualifies
    /// speculation (grammar, logprobs, sampling penalties) collapses the
    /// whole call to scalar rounds — the scalar batch is the only path that
    /// can serve those configs. Verification and commit stay per row on each
    /// row's own physical cache; only the draft advances batch.
    pub(crate) fn decode_speculative_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
        input_budget: usize,
    ) -> Result<Vec<ChatDecodeStep>> {
        let head = self.mtp_head.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3.5 speculative batch requires a loaded MTP head".into())
        })?;
        let state_count = states.len();
        if state_count == 0 {
            return Ok(Vec::new());
        }
        validate_continuous_decode_rows(states)?;
        let mut deltas = vec![String::new(); state_count];
        let mut committed = vec![0usize; state_count];
        let mut finished = vec![false; state_count];
        let mut active: Vec<usize> = (0..state_count).collect();
        active.retain(|&row| !finished[row] && committed[row] < input_budget);
        // Rows share one MTP arena whenever their caches were cut from the
        // same pool (the managed-domain split does this); the batched draft
        // advance needs it, standalone per-session arenas advance in turn.
        let shared_mtp_arena = active.iter().all(|&row| {
            states[row]
                .mtp_cache
                .as_ref()
                .is_some_and(|cache| {
                    std::sync::Arc::ptr_eq(
                        cache.arena(),
                        states[active[0]]
                            .mtp_cache
                            .as_ref()
                            .expect("eligibility checked mtp cache")
                            .arena(),
                    )
                })
        });

        while !active.is_empty() {
            let all_eligible = active.iter().all(|&row| self.mtp_active(states[row]));
            if !all_eligible {
                let mut subset = take_state_subset(states, &active);
                let steps = self.decode_step_batch(&mut subset)?;
                for (position, &row) in active.iter().enumerate() {
                    deltas[row].push_str(&steps[position].delta);
                    committed[row] += steps[position].input_tokens_committed;
                    finished[row] = steps[position].finished;
                }
                active.retain(|&row| !finished[row] && committed[row] < input_budget);
                continue;
            }

            // Per-row depths mirror the solo quantum: a round commits the
            // pending token plus up to `depth` drafted tokens, and the
            // anchor must be seeded by a scalar tail before the row drafts.
            // Per-row (rather than a homogeneous minimum) keeps every row's
            // round sequence identical to its solo quantum's.
            let depths: Vec<usize> = active
                .iter()
                .map(|&row| {
                    let state = &*states[row];
                    let remaining_budget = input_budget - committed[row];
                    let remaining_output = state
                        .max_new_tokens
                        .saturating_sub(state.tokens_generated)
                        .max(1);
                    if state.mtp_anchor_hidden.is_none() {
                        0
                    } else {
                        state
                            .adaptive_mtp
                            .depth(remaining_output.min(remaining_budget))
                            .min(remaining_budget)
                            .min(remaining_output)
                    }
                })
                .collect();
            let max_depth = depths.iter().copied().max().unwrap_or(0);
            if max_depth == 0 {
                // Every active row is on its scalar tail (budget down to one
                // token, anchor not yet seeded, or a numerically disabled
                // controller): one committed token per row, each seeding its
                // MTP pair.
                for &row in &active {
                    let delta = self.mtp_scalar_tail(states[row], head)?;
                    deltas[row].push_str(&delta);
                    committed[row] += 1;
                    finished[row] = states[row].finished;
                }
                active.retain(|&row| !finished[row] && committed[row] < input_budget);
                continue;
            }

            let round_started = std::time::Instant::now();
            self.mtp_envelope_rounds
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Per-row MTP checkpoints: the draft's KV writes are provisional
            // and rolled back before verification rewrites the positions.
            let mut mtp_checkpoints = Vec::with_capacity(active.len());
            for &row in &active {
                let state = &mut *states[row];
                let mtp = state.mtp_cache.as_mut().ok_or_else(|| {
                    Error::InferenceError("Qwen3.5 speculative row lost its MTP cache".into())
                })?;
                mtp_checkpoints.push(mtp.logical_checkpoint());
            }
            let mut currents: Vec<Tensor> = Vec::with_capacity(active.len());
            for &row in &active {
                let state = &*states[row];
                let anchor = state.mtp_anchor_hidden.as_ref().ok_or_else(|| {
                    Error::InferenceError("Qwen3.5 speculative row has no recurrent anchor".into())
                })?;
                currents.push(anchor.clone());
            }

            // Batched recurrent draft: one MTP-layer forward per draft step
            // across the rows still drafting at that depth. Greedy rows
            // select through the same argmax the solo draft uses; sampled
            // rows propose through the shared lossless sampler on their own
            // draft RNG stream. A non-finite draft logits row falls back to
            // a scalar round and disables that row's speculation for the
            // rest of its run.
            let mut draft_tokens = vec![Vec::new(); active.len()];
            let mut draft_proposals = vec![Vec::new(); active.len()];
            let mut draft_histories = vec![Vec::new(); active.len()];
            for (position, &row) in active.iter().enumerate() {
                if states[row].track_history {
                    draft_histories[position] = states[row].history_ids.clone();
                }
            }
            let mut draft_error_row: Option<usize> = None;
            'draft: for step_index in 0..max_depth {
                let drafting: Vec<usize> = (0..active.len())
                    .filter(|&position| depths[position] > step_index)
                    .collect();
                let mut tokens_step = Vec::with_capacity(drafting.len());
                for &position in &drafting {
                    let row = active[position];
                    let logits =
                        self.text_model
                            .project_with_shared_lm_head(&currents[position])?;
                    let mut values = logits_to_vec(&logits.i((0, 0))?)?;
                    truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                    if !values.iter().any(|value| value.is_finite()) {
                        draft_error_row = Some(position);
                        break 'draft;
                    }
                    if states[row].config.temperature > 1e-5 {
                        let state = &mut *states[row];
                        let proposal = propose_speculative_draft(
                            &values,
                            &state.config,
                            &mut draft_histories[position],
                            &mut state.draft_rng,
                        )?;
                        let token_id = proposal.token_id;
                        draft_proposals[position].push(proposal);
                        tokens_step.push(token_id);
                    } else {
                        tokens_step.push(argmax_values(&values)?);
                    }
                }
                for (fill, &position) in drafting.iter().enumerate() {
                    draft_tokens[position].push(tokens_step[fill]);
                }
                if step_index + 1 < max_depth {
                    let advancing: Vec<usize> = (0..active.len())
                        .filter(|&position| depths[position] > step_index + 1)
                        .collect();
                    if advancing.is_empty() {
                        continue;
                    }
                    let advance_tokens = advancing
                        .iter()
                        .map(|&position| draft_tokens[position][step_index])
                        .collect::<Vec<_>>();
                    let positions = advancing
                        .iter()
                        .map(|&position| {
                            let row = active[position];
                            [states[row].next_text_position + step_index; 3]
                        })
                        .collect::<Vec<_>>();
                    let mut embeddings = Vec::with_capacity(advancing.len());
                    for &token in &advance_tokens {
                        embeddings.push(self.text_model.embed_token_ids(&[token])?);
                    }
                    let embeddings = Tensor::cat(&embeddings, 0)?;
                    let hiddens = Tensor::cat(
                        &advancing
                            .iter()
                            .map(|&position| &currents[position])
                            .collect::<Vec<_>>(),
                        0,
                    )?;
                    if shared_mtp_arena {
                        let mut subset = take_state_subset(states, &active);
                        let mut caches = Vec::with_capacity(advancing.len());
                        for (position, state) in subset.iter_mut().enumerate() {
                            if !advancing.contains(&position) {
                                continue;
                            }
                            let mtp = state.mtp_cache.as_mut().ok_or_else(|| {
                                Error::InferenceError(
                                    "Qwen3.5 speculative row lost its MTP cache".into(),
                                )
                            })?;
                            caches.push(mtp);
                        }
                        let next = head.forward_steps_batch(
                            &embeddings,
                            &hiddens,
                            &positions,
                            caches.as_mut_slice(),
                        )?;
                        for (fill, &position) in advancing.iter().enumerate() {
                            currents[position] = next.i(fill)?.unsqueeze(0)?;
                        }
                    } else {
                        // Standalone per-session MTP arenas cannot share one
                        // lowered slot map — advance the rows one at a time.
                        for (fill, &position) in advancing.iter().enumerate() {
                            let row = active[position];
                            let state = &mut *states[row];
                            let mtp = state.mtp_cache.as_mut().ok_or_else(|| {
                                Error::InferenceError(
                                    "Qwen3.5 speculative row lost its MTP cache".into(),
                                )
                            })?;
                            let next = head.forward_step(
                                &embeddings.i(fill)?.unsqueeze(0)?,
                                &currents[position],
                                positions[fill],
                                mtp,
                            )?;
                            currents[position] = next;
                        }
                    }
                }
            }

            for (position, &row) in active.iter().enumerate() {
                let state = &mut *states[row];
                let mtp = state.mtp_cache.as_mut().ok_or_else(|| {
                    Error::InferenceError("Qwen3.5 speculative row lost its MTP cache".into())
                })?;
                mtp.restore_logical_checkpoint(mtp_checkpoints[position].clone())?;
                if draft_error_row == Some(position) {
                    state.adaptive_mtp.disable_after_nonfinite_draft();
                }
            }
            if draft_error_row.is_some() {
                // The envelope falls back to a scalar round together; the
                // offending row's controller is disabled for the rest of its
                // run and its depth() will keep reporting zero.
                let mut subset = take_state_subset(states, &active);
                let steps = self.decode_step_batch(&mut subset)?;
                for (position, &row) in active.iter().enumerate() {
                    deltas[row].push_str(&steps[position].delta);
                    committed[row] += steps[position].input_tokens_committed;
                    finished[row] = steps[position].finished;
                }
                active.retain(|&row| !finished[row] && committed[row] < input_budget);
                continue;
            }

            // Greedy rows carry bare ids; sampled rows carry their proposal
            // distributions into the shared verifier.
            let draft_blocks: Vec<DraftBlock> = (0..active.len())
                .map(|position| {
                    if draft_proposals[position].is_empty() {
                        DraftBlock::Greedy(std::mem::take(&mut draft_tokens[position]))
                    } else {
                        DraftBlock::Stochastic(std::mem::take(&mut draft_proposals[position]))
                    }
                })
                .collect();

            // Verification and commit stay per row on each row's own caches.
            for (position, &row) in active.iter().enumerate() {
                let state = &mut *states[row];
                if depths[position] == 0 {
                    let delta = self.mtp_scalar_tail(state, head)?;
                    deltas[row].push_str(&delta);
                    committed[row] += 1;
                    // Scheduler-limited tails train the controller only when
                    // the observation budget is the round's own remaining
                    // slice, matching the solo quantum's timer gating.
                    let remaining = (input_budget - committed[row]).min(
                        state
                            .max_new_tokens
                            .saturating_sub(state.tokens_generated)
                            .max(1),
                    );
                    if state.adaptive_mtp.can_train(remaining) {
                        state
                            .adaptive_mtp
                            .observe(0, 1, round_started.elapsed(), remaining);
                    }
                } else {
                    let (tokens, delta) =
                        self.mtp_verify_and_commit(state, head, &draft_blocks[position])?;
                    committed[row] += tokens;
                    deltas[row].push_str(&delta);
                    let remaining = (input_budget - committed[row]).min(
                        state
                            .max_new_tokens
                            .saturating_sub(state.tokens_generated)
                            .max(1),
                    );
                    if state.adaptive_mtp.can_train(remaining) {
                        state.adaptive_mtp.observe(
                            depths[position],
                            tokens,
                            round_started.elapsed(),
                            remaining,
                        );
                    }
                }
                finished[row] = state.finished;
            }
            active.retain(|&row| !finished[row] && committed[row] < input_budget);
        }

        let mut steps = Vec::with_capacity(state_count);
        for row in 0..state_count {
            let state = &*states[row];
            steps.push(ChatDecodeStep {
                delta: std::mem::take(&mut deltas[row]),
                text: if state.finished {
                    state.assembled.clone()
                } else {
                    String::new()
                },
                tokens_generated: state.tokens_generated,
                input_tokens_committed: committed[row],
                finished: state.finished,
            });
        }
        Ok(steps)
    }
}

fn build_text_positions(token_count: usize) -> Vec<[usize; 3]> {
    (0..token_count).map(|idx| [idx; 3]).collect()
}

fn render_fallback_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        });
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_prompt(
    messages: &[ChatMessage],
    config: &ChatGenerationConfig,
    default_enable_thinking: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 chat prompt requires at least one message".to_string(),
        ));
    }

    let mut prompt = String::new();
    let leading_system =
        matches!(messages.first(), Some(message) if message.role == ChatRole::System);
    let system_content = if leading_system {
        messages[0].content.trim()
    } else {
        ""
    };

    if !config.request.tools.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in &config.request.tools {
            prompt.push('\n');
            prompt.push_str(&serde_json::to_string(tool)?);
        }
        prompt.push_str("\n</tools>");
        prompt.push_str(TOOL_PROMPT_SUFFIX);
        if !system_content.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(system_content);
        }
        prompt.push_str("<|im_end|>\n");
    } else if leading_system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system_content);
        prompt.push_str("<|im_end|>\n");
    }

    let last_query_index = last_query_index(messages)?;
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System {
            if index != 0 {
                return Err(Error::InvalidInput(
                    "Qwen3.5 system message must be the first message".to_string(),
                ));
            }
            continue;
        }

        match message.role {
            ChatRole::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(message.content.trim());
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                let (reasoning_content, content) = split_assistant_reasoning(&message.content);
                prompt.push_str("<|im_start|>assistant\n");
                if index > last_query_index {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning_content.trim());
                    prompt.push_str("\n</think>\n\n");
                    prompt.push_str(content.trim_start());
                } else {
                    prompt.push_str(content.trim());
                }
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::System => {}
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    if config
        .request
        .enable_thinking
        .unwrap_or(default_enable_thinking)
    {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    Ok(prompt)
}

fn last_query_index(messages: &[ChatMessage]) -> Result<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            (message.role == ChatRole::User && !is_tool_response(&message.content)).then_some(index)
        })
        .ok_or_else(|| {
            Error::InvalidInput("Qwen3.5 prompt requires at least one user query".to_string())
        })
}

fn is_tool_response(content: &str) -> bool {
    let content = content.trim();
    content.starts_with("<tool_response>") && content.ends_with("</tool_response>")
}

fn split_assistant_reasoning(content: &str) -> (&str, &str) {
    let Some(end_idx) = content.find("</think>") else {
        return ("", content);
    };
    let reasoning_prefix = &content[..end_idx];
    let reasoning = reasoning_prefix
        .rsplit_once("<think>")
        .map(|(_, reasoning)| reasoning)
        .unwrap_or(reasoning_prefix);
    let answer = content[(end_idx + "</think>".len())..].trim_start_matches('\n');
    (reasoning.trim_matches('\n'), answer)
}

const TOOL_PROMPT_SUFFIX: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

fn resolve_default_enable_thinking(_chat_template: &str, variant: ModelVariant) -> bool {
    // Qwen3.6-35B-A3B ships thinking default-on; the empty think block is
    // emitted when a request disables it.
    matches!(variant, ModelVariant::Qwen36Moe35BA3BFp8)
}

fn parse_gguf_tokenizer_metadata(loader: &GgufLoader) -> Result<GgufTokenizerMetadata> {
    Ok(GgufTokenizerMetadata {
        tokens: required_string_array(loader, "tokenizer.ggml.tokens")?,
        token_types: required_u32_array(loader, "tokenizer.ggml.token_type")?,
        merges: required_string_array(loader, "tokenizer.ggml.merges")?,
        pre_tokenizer: loader.get_metadata_string("tokenizer.ggml.pre"),
        chat_template: loader
            .get_metadata_string("tokenizer.chat_template")
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Missing or invalid GGUF metadata: tokenizer.chat_template".to_string(),
                )
            })?,
        eos_token_id: loader
            .get_metadata_u64("tokenizer.ggml.eos_token_id")
            .and_then(|value| u32::try_from(value).ok()),
    })
}

fn load_tokenizer_config_file(model_dir: &Path) -> Result<Option<TokenizerConfigFile>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_str = fs::read_to_string(config_path)?;
    let config: TokenizerConfigFile = serde_json::from_str(&config_str)?;
    Ok(Some(config))
}

fn parse_text_config(loader: &GgufLoader) -> Result<Qwen36TextConfig> {
    Ok(Qwen36TextConfig {
        architecture: loader
            .get_metadata_string("general.architecture")
            .unwrap_or_else(|| "qwen35".to_string()),
        block_count: required_usize(loader, "qwen35.block_count")?,
        context_length: required_usize(loader, "qwen35.context_length")?,
        embedding_length: required_usize(loader, "qwen35.embedding_length")?,
        feed_forward_length: required_usize(loader, "qwen35.feed_forward_length")?,
        attention_head_count: required_usize(loader, "qwen35.attention.head_count")?,
        attention_head_count_kv: required_usize(loader, "qwen35.attention.head_count_kv")?,
        attention_key_length: required_usize(loader, "qwen35.attention.key_length")?,
        attention_value_length: required_usize(loader, "qwen35.attention.value_length")?,
        rope_dimension_sections: required_usize_array(loader, "qwen35.rope.dimension_sections")?,
        rope_dimension_count: required_usize(loader, "qwen35.rope.dimension_count")?,
        rope_freq_base: required_f64(loader, "qwen35.rope.freq_base")?,
        attention_layer_norm_rms_epsilon: required_f64(
            loader,
            "qwen35.attention.layer_norm_rms_epsilon",
        )?,
        ssm_conv_kernel: required_usize(loader, "qwen35.ssm.conv_kernel")?,
        ssm_state_size: required_usize(loader, "qwen35.ssm.state_size")?,
        ssm_group_count: required_usize(loader, "qwen35.ssm.group_count")?,
        ssm_time_step_rank: required_usize(loader, "qwen35.ssm.time_step_rank")?,
        ssm_inner_size: required_usize(loader, "qwen35.ssm.inner_size")?,
        full_attention_interval: required_usize(loader, "qwen35.full_attention_interval")?,
        moe_ffn: None,
    })
}

pub(crate) fn required_usize(loader: &GgufLoader, key: &str) -> Result<usize> {
    loader
        .get_metadata_u64(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

pub(crate) fn required_f64(loader: &GgufLoader, key: &str) -> Result<f64> {
    let value = loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    Ok(value)
}

pub(crate) fn required_usize_array(loader: &GgufLoader, key: &str) -> Result<Vec<usize>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    let GgufValue::Array(items) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF array metadata for {key}"
        )));
    };

    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(raw) = gguf_to_u64(item) else {
            return Err(Error::ModelLoadError(format!(
                "Expected integer array values for {key}"
            )));
        };
        let value = usize::try_from(raw).map_err(|_| {
            Error::ModelLoadError(format!("Array value out of range for {key}: {raw}"))
        })?;
        values.push(value);
    }
    Ok(values)
}

fn required_u32_array(loader: &GgufLoader, key: &str) -> Result<Vec<u32>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    let GgufValue::Array(items) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF array metadata for {key}"
        )));
    };

    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(raw) = gguf_to_u64(item) else {
            return Err(Error::ModelLoadError(format!(
                "Expected integer array values for {key}"
            )));
        };
        let value = u32::try_from(raw).map_err(|_| {
            Error::ModelLoadError(format!("Array value out of range for {key}: {raw}"))
        })?;
        values.push(value);
    }
    Ok(values)
}

fn required_string_array(loader: &GgufLoader, key: &str) -> Result<Vec<String>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    let GgufValue::Array(items) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF array metadata for {key}"
        )));
    };

    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(raw) = gguf_to_string(item) else {
            return Err(Error::ModelLoadError(format!(
                "Expected string array values for {key}"
            )));
        };
        values.push(raw);
    }
    Ok(values)
}

fn gguf_to_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U64(n) => Some(*n),
        GgufValue::I64(n) => Some(*n as u64),
        GgufValue::U32(n) => Some(*n as u64),
        GgufValue::I32(n) => Some(*n as u64),
        GgufValue::U16(n) => Some(*n as u64),
        GgufValue::I16(n) => Some(*n as u64),
        GgufValue::U8(n) => Some(*n as u64),
        GgufValue::I8(n) => Some(*n as u64),
        _ => None,
    }
}

fn gguf_to_string(value: &GgufValue) -> Option<String> {
    match value {
        GgufValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F64(n) => Some(*n),
        GgufValue::F32(n) => Some(*n as f64),
        GgufValue::U64(n) => Some(*n as f64),
        GgufValue::I64(n) => Some(*n as f64),
        GgufValue::U32(n) => Some(*n as f64),
        GgufValue::I32(n) => Some(*n as f64),
        GgufValue::U16(n) => Some(*n as f64),
        GgufValue::I16(n) => Some(*n as f64),
        GgufValue::U8(n) => Some(*n as f64),
        GgufValue::I8(n) => Some(*n as f64),
        _ => None,
    }
}

fn take_quantum_sample(
    output: &mut Option<Tensor>,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let output = output.take().ok_or_else(|| {
        Error::InferenceError("Qwen3.5 decode quantum has no unconsumed model output".to_string())
    })?;
    sample_next_token(&output, vocab_size, config, history, rng)
}

/// DS9.3: sample a token and, when the request asked for logprobs, resolve
/// the raw-distribution stats on host. Sampling math is unchanged.
fn sample_next_token_with_logprobs(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<(
    u32,
    Option<crate::models::shared::sampling::RawTokenLogprobs>,
)> {
    if !config.logprobs {
        let token = sample_next_token(logits, vocab_size, config, history, rng)?;
        return Ok((token, None));
    }
    let raw_values = logits_to_vec(logits)?;
    let mut values = raw_values.clone();
    truncate_logits_to_vocab(&mut values, vocab_size);
    if values.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 sampler received no in-vocabulary logits".to_string(),
        ));
    }
    let (logsumexp, top) =
        crate::models::shared::sampling::raw_logprobs_stats(&values, config.top_logprobs)?;
    let token = sample_next_token(logits, vocab_size, config, history, rng)?;
    let chosen_raw = raw_values
        .get(token as usize)
        .copied()
        .ok_or_else(|| Error::InferenceError("sampled token outside raw row".into()))?;
    if !chosen_raw.is_finite() {
        return Err(Error::InferenceError(
            "sampled token has a non-finite raw logit".into(),
        ));
    }
    Ok((
        token,
        Some(crate::models::shared::sampling::RawTokenLogprobs {
            token,
            logprob: chosen_raw - logsumexp,
            top,
        }),
    ))
}

fn sample_next_token(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 sampler received vocab_size=0".to_string(),
        ));
    }

    // Fast path for deterministic greedy decode (bench/default path):
    // avoid copying full logits tensors to CPU each token.
    if deterministic_greedy(config) {
        return argmax_clamped(logits, vocab_size);
    }

    if let Some(candidates) = bounded_device_sampling_candidates(
        logits,
        vocab_size,
        config.top_k,
        config.temperature,
        history,
        config.repetition_penalty,
        config.presence_penalty,
        None,
    )? {
        if device_candidates_cover_top_p(&candidates, config.top_p) {
            if let Some(sampled) =
                sample_device_candidates(&candidates, config.top_p, rng.next_f32())
            {
                return Ok(sampled);
            }
        }
    }

    let mut values = logits_to_vec(logits)?;
    truncate_logits_to_vocab(&mut values, vocab_size);

    if config.repetition_penalty > 1.0 && !history.is_empty() {
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
            let value = &mut values[idx];
            if !value.is_finite() {
                continue;
            }
            if *value > 0.0 {
                *value /= config.repetition_penalty;
            } else {
                *value *= config.repetition_penalty;
            }
        }
    }

    if config.presence_penalty.abs() > f32::EPSILON && !history.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history {
            let idx = token as usize;
            if idx < seen.len() {
                seen[idx] = true;
            }
        }

        for (idx, seen_flag) in seen.iter().enumerate() {
            if *seen_flag && values[idx].is_finite() {
                values[idx] -= config.presence_penalty;
            }
        }
    }

    if config.temperature <= 1e-5 {
        return argmax_values(&values);
    }

    let temperature = config.temperature.max(1e-5);
    for value in &mut values {
        if value.is_finite() {
            *value /= temperature;
        }
    }

    let mut candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.is_finite().then_some(idx))
        .collect();
    if candidates.is_empty() {
        return argmax_values(&values);
    }

    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
        candidates.truncate(config.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|&idx| values[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| (idx, (values[idx] - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_values(&values);
    }
    for (_, prob) in &mut probs {
        *prob /= sum;
    }

    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = config.top_p.max(1e-6);
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, prob) in &probs {
            cumulative += *prob;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, prob)| *prob).sum();
        if sum > 0.0 {
            for (_, prob) in &mut probs {
                *prob /= sum;
            }
        }
    }

    let sample = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (idx, prob) in &probs {
        cumulative += *prob;
        if sample <= cumulative {
            return Ok(*idx as u32);
        }
    }

    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx as u32)
        .ok_or_else(|| Error::InferenceError("Failed to sample Qwen3.5 token".to_string()))
}

/// Normalize model logits — `[1, seq, vocab]`, `[seq, vocab]`, or
/// `[vocab]` — into the flat last-row vector the samplers consume. The
/// stored prefill output is rank 3; forward paths return rank 1.
fn logits_last_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        1 => Ok(logits.clone()),
        2 => Ok(logits.clone()),
        3 => {
            let (batch, sequence, _) = logits.dims3()?;
            if batch != 1 || sequence == 0 {
                return Err(Error::InvalidInput(format!(
                    "Qwen3.5 sampler received malformed logits {batch}x{sequence}"
                )));
            }
            logits.i((0, sequence - 1)).map_err(Error::from)
        }
        rank => Err(Error::InferenceError(format!(
            "Unexpected Qwen3.5 logits rank for sampling: {rank}"
        ))),
    }
}

fn logits_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.5 logits shape for sampling: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.5 logits rank for sampling: {rank}"
            )))
        }
    };

    logits
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(Error::from)
}

fn truncate_logits_to_vocab(values: &mut Vec<f32>, vocab_size: usize) {
    if vocab_size < values.len() {
        values.truncate(vocab_size);
    }
}

fn no_valid_logits_error(values: &[f32]) -> Error {
    let mut nan = 0usize;
    let mut positive_infinity = 0usize;
    let mut negative_infinity = 0usize;
    for value in values {
        if value.is_nan() {
            nan = nan.saturating_add(1);
        } else if *value == f32::INFINITY {
            positive_infinity = positive_infinity.saturating_add(1);
        } else if *value == f32::NEG_INFINITY {
            negative_infinity = negative_infinity.saturating_add(1);
        }
    }
    Error::InferenceError(format!(
        "No valid Qwen3.5 logits to sample: 0 finite, {nan} NaN, \
         {positive_infinity} +Inf, {negative_infinity} -Inf across {} in-vocabulary logits",
        values.len()
    ))
}

fn argmax_values(values: &[f32]) -> Result<u32> {
    let mut max_idx = None;
    let mut max_value = f32::NEG_INFINITY;

    for (idx, value) in values.iter().enumerate() {
        if value.is_finite() && *value > max_value {
            max_value = *value;
            max_idx = Some(idx);
        }
    }

    max_idx
        .map(|idx| idx as u32)
        .ok_or_else(|| no_valid_logits_error(values))
}

/// Plain greedy decode: no temperature, penalties, top-k or top-p.
fn deterministic_greedy(config: &ChatGenerationConfig) -> bool {
    config.temperature <= 1e-5
        && (config.repetition_penalty - 1.0).abs() <= f32::EPSILON
        && config.presence_penalty.abs() <= f32::EPSILON
        && config.top_k == 0
        && config.top_p >= 1.0
}

/// Greedy tokens for `[rows, vocab]` logits with one readback. The device
/// kernel returns each row's highest finite logit, lowest index on ties (the
/// semantics of [`argmax_values`]), and whether the row had one. `None` marks
/// a row without a finite logit.
fn device_greedy_rows(logits: &Tensor) -> Result<Vec<Option<u32>>> {
    let packed = crate::kernels::cuda::sampling::greedy_rows(logits)?.to_vec2::<u32>()?;
    Ok(packed
        .into_iter()
        .map(|row| (row[1] != 0).then_some(row[0]))
        .collect())
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.5 logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.5 logits rank for argmax: {rank}"
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

fn argmax_clamped(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 argmax received vocab_size=0".to_string(),
        ));
    }

    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.5 logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.5 logits rank for argmax: {rank}"
            )))
        }
    };

    let cols = logits.dim(0)?;
    let clamped = if vocab_size < cols {
        logits.narrow(0, 0, vocab_size)?
    } else {
        logits
    };
    if clamped.device().is_cuda() {
        // One readback instead of argmax + selected-logit reads.
        if let Some(token) = device_greedy_rows(&clamped.unsqueeze(0)?)?[0] {
            return Ok(token);
        }
        let values = clamped.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        return argmax_values(&values);
    }
    let selected = argmax(&clamped)?;
    let selected_logit = clamped
        .i(selected as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    if selected_logit.is_finite() {
        return Ok(selected);
    }

    // Some device argmax kernels do not define useful ordering for NaNs. This
    // slow path runs only after the selected value is non-finite: it recovers a
    // finite candidate when one exists and otherwise returns useful counts for
    // the exact in-vocabulary row in every sampling mode.
    let values = clamped.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    argmax_values(&values)
}

#[derive(Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        } else {
            seed
        };
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    /// Derive the draft RNG stream from the target stream: speculative
    /// proposals must not perturb the target's draw sequence.
    fn fork(&mut self) -> Self {
        let seed = (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32());
        Self::new(seed)
    }

    fn next_u32(&mut self) -> u32 {
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

impl rand::RngCore for SimpleRng {
    fn next_u32(&mut self) -> u32 {
        SimpleRng::next_u32(self)
    }

    fn next_u64(&mut self) -> u64 {
        (u64::from(SimpleRng::next_u32(self)) << 32) | u64::from(SimpleRng::next_u32(self))
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(std::mem::size_of::<u32>()) {
            let bytes = SimpleRng::next_u32(self).to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl crate::models::shared::sampling::GrammarRng for SimpleRng {
    fn draw_unit(&mut self) -> f32 {
        self.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::chat::ChatRequestConfig;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    fn device_greedy_rows_share_the_host_fallback_semantics() {
        let rows = [
            vec![1.0f32, 3.0, 3.0, -2.0],              // tie: lowest index
            vec![f32::NAN, 0.5, f32::INFINITY, 0.25],  // non-finite skipped
            vec![-1.0, f32::NEG_INFINITY, -0.5, -0.5], // tie after -inf
            vec![f32::NAN, f32::INFINITY, f32::NAN, f32::NEG_INFINITY], // none finite
        ];
        let logits =
            Tensor::from_vec(rows.concat(), (rows.len(), 4), &candle_core::Device::Cpu).unwrap();
        let tokens = device_greedy_rows(&logits).unwrap();
        for (row, (values, token)) in rows.iter().zip(&tokens).enumerate() {
            assert_eq!(*token, argmax_values(values).ok(), "row {row}");
        }
        assert_eq!(tokens, vec![Some(1), Some(1), Some(2), None]);
        // BF16 logits (the CUDA trunk dtype) select the same tokens.
        let bf16 = device_greedy_rows(&logits.to_dtype(DType::BF16).unwrap()).unwrap();
        assert_eq!(bf16, tokens);
    }

    fn byte_level_char_for_test(byte: u8) -> char {
        let mut bytes: Vec<u8> = (b'!'..=b'~')
            .chain(b'\xA1'..=b'\xAC')
            .chain(b'\xAE'..=b'\xFF')
            .collect();
        let mut codepoints: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
        let mut next = 0u32;
        for candidate in 0..=u8::MAX {
            if !bytes.contains(&candidate) {
                bytes.push(candidate);
                codepoints.push(256 + next);
                next += 1;
            }
        }
        let index = bytes
            .iter()
            .position(|candidate| *candidate == byte)
            .expect("byte-level alphabet contains every byte");
        char::from_u32(codepoints[index]).expect("byte-level codepoint is valid")
    }

    fn synthetic_qwen35_tokenizer() -> Qwen36Tokenizer {
        let mut tokens: Vec<String> = (0..=u8::MAX)
            .map(|byte| byte_level_char_for_test(byte).to_string())
            .collect();
        let im_start = tokens.len() as u32;
        tokens.push("<|im_start|>".to_string());
        let im_end = tokens.len() as u32;
        tokens.push("<|im_end|>".to_string());
        let image_pad = tokens.len() as u32;
        tokens.push("<|image_pad|>".to_string());
        let video_pad = tokens.len() as u32;
        tokens.push("<|video_pad|>".to_string());
        let eos_alt = tokens.len() as u32;
        tokens.push("<|endoftext|>".to_string());
        let think_open = tokens.len() as u32;
        tokens.push("<think>".to_string());
        let think_close = tokens.len() as u32;
        tokens.push("</think>".to_string());

        let mut token_types = vec![1; 256];
        token_types.extend([3, 3, 3, 3, 3, 4, 4]);
        let mut inner =
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).expect("tokenizer");
        inner
            .register_gguf_token_types(&tokens, &token_types)
            .expect("atomic tokens");

        Qwen36Tokenizer {
            vocab_size: inner.vocab_size(),
            inner,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                image_pad,
                video_pad,
                eos: im_end,
                eos_alt: Some(eos_alt),
            },
            literal_special_tokens: vec![
                ("<|im_start|>".to_string(), im_start),
                ("<|endoftext|>".to_string(), eos_alt),
                ("<|image_pad|>".to_string(), image_pad),
                ("<|video_pad|>".to_string(), video_pad),
                ("<|im_end|>".to_string(), im_end),
                ("</think>".to_string(), think_close),
                ("<think>".to_string(), think_open),
            ],
            chat_template: String::new(),
            default_enable_thinking: false,
            bos_token: None,
        }
    }

    #[test]
    fn prepared_prompt_reuse_skips_fetch_and_vision_encode_builder() {
        let prepared = Qwen36PreparedPrompt {
            prompt_ids: vec![1, 2, 3],
            prompt_positions: build_text_positions(3),
            next_text_position: 3,
        };
        let preparation_calls = AtomicUsize::new(0);
        let resolved = resolve_prepared_prompt(Some(&prepared), || {
            preparation_calls.fetch_add(1, AtomicOrdering::Relaxed);
            Err(Error::InferenceError(
                "fetch/vision encode should not run".to_string(),
            ))
        })
        .unwrap();

        assert_eq!(resolved.prompt_ids(), prepared.prompt_ids());
        assert_eq!(preparation_calls.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn penalty_history_starts_with_exact_prompt_ids() {
        let prompt_ids = vec![248_000, 17, 23, 248_001];
        let mut history = initial_penalty_history(&prompt_ids, 8, true);
        assert_eq!(history, prompt_ids);
        history.push(99);
        assert_eq!(&history[..prompt_ids.len()], prompt_ids.as_slice());
        assert!(initial_penalty_history(&prompt_ids, 8, false).is_empty());
    }

    #[test]
    fn prefill_chunk_size_is_bounded_and_rejects_zero() {
        let _guard = crate::env_test_lock().lock().expect("env lock");
        std::env::remove_var("IZWI_QWEN35_PREFILL_CHUNK_SIZE");
        assert_eq!(qwen35_prefill_chunk_size(), DEFAULT_PREFILL_CHUNK_SIZE);
        std::env::set_var("IZWI_QWEN35_PREFILL_CHUNK_SIZE", "0");
        assert_eq!(qwen35_prefill_chunk_size(), DEFAULT_PREFILL_CHUNK_SIZE);
        std::env::set_var("IZWI_QWEN35_PREFILL_CHUNK_SIZE", "64");
        assert_eq!(qwen35_prefill_chunk_size(), 64);
        std::env::set_var("IZWI_QWEN35_PREFILL_CHUNK_SIZE", "999999");
        assert_eq!(qwen35_prefill_chunk_size(), MAX_PREFILL_CHUNK_SIZE);
        std::env::remove_var("IZWI_QWEN35_PREFILL_CHUNK_SIZE");
    }

    #[test]
    fn multimodal_prefill_segmentation_is_partition_invariant() {
        let image_pad = 99;
        let prompt = [1, 2, image_pad, image_pad, image_pad, 3, 4, image_pad, 5];
        let classify = |spans: &[(usize, usize)]| {
            let mut rows = Vec::new();
            for &(start, end) in spans {
                let mut cursor = start;
                while cursor < end {
                    let (segment_end, image) =
                        next_prefill_segment_end(&prompt, cursor, end, image_pad, 2)
                            .expect("valid segment");
                    rows.extend((cursor..segment_end).map(|index| (index, image)));
                    cursor = segment_end;
                }
            }
            rows
        };

        let monolithic = classify(&[(0, prompt.len())]);
        let scheduler_chunked = classify(&[(0, 3), (3, 6), (6, prompt.len())]);
        assert_eq!(scheduler_chunked, monolithic);
        assert_eq!(
            monolithic
                .iter()
                .filter(|(_, image)| *image)
                .map(|(index, _)| *index)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 7]
        );
        assert!(next_prefill_segment_end(&prompt, 0, prompt.len(), image_pad, 0).is_err());
    }

    #[test]
    fn multimodal_text_positions_do_not_define_the_physical_decode_cursor() {
        let prompt_tokens = 6;
        let next_text_position = 4;
        assert_ne!(prompt_tokens, next_text_position);
        assert_eq!(expected_physical_decode_cursor(prompt_tokens, 0), 6);
        assert_eq!(expected_physical_decode_cursor(prompt_tokens, 1), 6);
        assert_eq!(expected_physical_decode_cursor(prompt_tokens, 2), 7);
    }

    #[test]
    fn resolve_default_thinking_depends_on_variant_not_template_signature() {
        let small = "{%- if add_generation_prompt %}{%- if enable_thinking is defined and enable_thinking is true %}<think>\n{%- endif %}{%- endif %}";
        assert!(resolve_default_enable_thinking(
            small,
            ModelVariant::Qwen36Moe35BA3BFp8
        ));
    }

    #[test]
    fn explicit_thinking_override_wins_over_variant_default() {
        let messages = [ChatMessage {
            role: ChatRole::User,
            content: "Answer briefly.".to_string(),
        }];
        let enable = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(true),
                tools: Vec::new(),
                media_inputs: Vec::new(),
                ..ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };
        let disable = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: Vec::new(),
                ..ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };

        assert!(render_prompt(&messages, &enable, false)
            .expect("enable thinking")
            .ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(render_prompt(&messages, &disable, true)
            .expect("disable thinking")
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn render_prompt_injects_tool_system_preamble() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(true),
                tools: vec![serde_json::json!({"type":"function","function":{"name":"lookup"}})],
                media_inputs: Vec::new(),
                ..ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: "Be precise.".to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hi".to_string(),
                },
            ],
            &config,
            false,
        )
        .expect("prompt should render");

        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\":\"lookup\""));
        assert!(prompt.contains("Be precise."));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn render_prompt_can_force_closed_think_block() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: Vec::new(),
                ..ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[ChatMessage {
                role: ChatRole::User,
                content: "Answer briefly.".to_string(),
            }],
            &config,
            true,
        )
        .expect("prompt should render");

        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn split_assistant_reasoning_handles_implicit_open_pattern() {
        assert_eq!(
            split_assistant_reasoning("reasoning first</think>\nFinal answer"),
            ("reasoning first", "Final answer")
        );
    }

    #[test]
    fn render_prompt_strips_prior_assistant_reasoning_from_history() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(true),
                tools: Vec::new(),
                media_inputs: Vec::new(),
                ..ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[
                ChatMessage {
                    role: ChatRole::User,
                    content: "First question".to_string(),
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "reasoning first</think>\nFinal answer".to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Follow-up".to_string(),
                },
            ],
            &config,
            true,
        )
        .expect("prompt should render");

        assert!(prompt.contains("<|im_start|>assistant\nFinal answer<|im_end|>\n"));
        assert!(!prompt.contains("reasoning first"));
    }

    #[test]
    fn multi_turn_rendering_preserves_unicode_assistant_content() {
        let previous_answer = "café 中文 🌍 👨‍👩‍👧‍👦";
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                tools: Vec::new(),
                media_inputs: Vec::new(),
                ..ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };
        let prompt = render_prompt(
            &[
                ChatMessage {
                    role: ChatRole::User,
                    content: "First turn".to_string(),
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: previous_answer.to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Second turn".to_string(),
                },
            ],
            &config,
            false,
        )
        .expect("render multi-turn prompt");

        assert!(prompt.contains(&format!(
            "<|im_start|>assistant\n{previous_answer}<|im_end|>\n"
        )));
        assert!(!prompt.contains('\u{fffd}'));

        let tokenizer = synthetic_qwen35_tokenizer();
        let ids = tokenizer
            .encode_text(&prompt)
            .expect("encode rendered prompt");
        assert!(ids.contains(&tokenizer.specials.im_start));
        assert!(ids.contains(&tokenizer.specials.im_end));
        assert!(ids.contains(&tokenizer.inner.token_to_id("<think>").unwrap()));
        assert!(ids.contains(&tokenizer.inner.token_to_id("</think>").unwrap()));
        assert_eq!(
            tokenizer
                .inner
                .decode_with_special_tokens(&ids)
                .expect("decode rendered ids"),
            prompt
        );
    }

    #[test]
    fn sample_next_token_masks_logits_above_vocab_limit() {
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 12.0, 9.0],
            (5,),
            &candle_core::Device::Cpu,
        )
        .expect("logits");
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig::default(),
            logprobs: false,
            top_logprobs: 0,
            constrain_json_object: false,
        };
        let mut rng = SimpleRng::new(7);
        let token = sample_next_token(&logits, 3, &config, &[], &mut rng).expect("sample token");
        assert_eq!(token, 2);
    }

    #[test]
    fn sampling_consumes_qwen35_quantum_output_without_changing_result() {
        let output = Tensor::from_vec(
            vec![0.1f32, 1.2, 0.4, 0.8],
            (1, 4),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let config = ChatGenerationConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 3,
            repetition_penalty: 1.1,
            presence_penalty: 0.2,
            stop_token_ids: Vec::new(),
            seed: 17,
            request: ChatRequestConfig::default(),
            logprobs: false,
            top_logprobs: 0,
            constrain_json_object: false,
        };
        let history = [1u32];
        let mut direct_rng = SimpleRng::new(17);
        let expected = sample_next_token(&output, 4, &config, &history, &mut direct_rng).unwrap();
        let mut quantum_rng = SimpleRng::new(17);
        let mut unconsumed = Some(output);

        assert_eq!(
            take_quantum_sample(&mut unconsumed, 4, &config, &history, &mut quantum_rng).unwrap(),
            expected
        );
        assert!(unconsumed.is_none());
        assert!(
            take_quantum_sample(&mut unconsumed, 4, &config, &history, &mut quantum_rng).is_err()
        );
    }

    #[test]
    fn sample_next_token_errors_when_vocab_limit_is_zero() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], (3,), &candle_core::Device::Cpu)
            .expect("logits");
        let config = ChatGenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig::default(),
            logprobs: false,
            top_logprobs: 0,
            constrain_json_object: false,
        };
        let mut rng = SimpleRng::new(7);
        let result = sample_next_token(&logits, 0, &config, &[], &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn sampler_reports_non_finite_counts_for_greedy_and_probabilistic_modes() {
        let logits = Tensor::from_vec(
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 42.0],
            (4,),
            &candle_core::Device::Cpu,
        )
        .expect("logits");

        for temperature in [0.0, 0.7] {
            let config = ChatGenerationConfig {
                temperature,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                stop_token_ids: Vec::new(),
                seed: 7,
                request: ChatRequestConfig::default(),
                logprobs: false,
                top_logprobs: 0,
                constrain_json_object: false,
            };
            let mut rng = SimpleRng::new(7);
            let error = sample_next_token(&logits, 3, &config, &[], &mut rng)
                .expect_err("all in-vocabulary logits are non-finite");
            let message = error.to_string();
            assert!(message.contains("0 finite"));
            assert!(message.contains("1 NaN"));
            assert!(message.contains("1 +Inf"));
            assert!(message.contains("1 -Inf"));
            assert!(message.contains("3 in-vocabulary logits"));
        }
    }
}
