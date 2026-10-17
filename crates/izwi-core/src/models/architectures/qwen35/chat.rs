//! Native Qwen3.5 chat model loader and text generation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::{debug, info};

use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::kv::v2::InferenceStateContract;
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};
use crate::models::shared::sampling::{
    bounded_device_sampling_candidates_with_history, device_candidates_cover_top_p,
    sample_device_candidates, PenaltyHistory,
};
use crate::models::shared::telemetry::{record_host_read, thread_host_reads};
use crate::models::shared::weights::gguf::{GgufLoader, GgufModelInfo};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::cache::qwen35_composite_cache_contract;
use super::fast_path::Qwen35CudaSwitches;
use super::telemetry::{Qwen35Telemetry, SamplerPath};
use super::text::{Qwen35TextModel, Qwen35TextRuntimeState};
use super::vision::{PreparedVisionInputs, Qwen35VisionModel};

const IMAGE_PAD_PLACEHOLDER: &str = "<|image_pad|>";
const VIDEO_PAD_PLACEHOLDER: &str = "<|video_pad|>";
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;
const MAX_PREFILL_CHUNK_SIZE: usize = 2048;
/// Prefill span requested from the scheduler when a request prefills alone
/// on an accelerator. Each engine step adds fixed host work (state staging,
/// KV commit, scheduling), and with chunked prefill on (the CUDA default)
/// the engine's threshold cuts prompts into 192-token steps. The model still
/// runs the span in `IZWI_QWEN35_PREFILL_CHUNK_SIZE` forward chunks.
const ACCELERATOR_PREFERRED_PREFILL_TOKENS: usize = 1024;

/// Preferred isolated prefill span for a backend; 0 keeps the engine's
/// threshold. CPU prefill is compute-bound per token, so a wider step only
/// delays cancellation and new arrivals.
fn preferred_prefill_tokens_for(backend: BackendKind) -> usize {
    match backend {
        BackendKind::Cuda | BackendKind::Metal => ACCELERATOR_PREFERRED_PREFILL_TOKENS,
        BackendKind::Cpu => 0,
    }
}

/// Decode tokens per engine step on accelerators, as Qwen3.6 uses. Every
/// step pays fixed host work (scheduling, state staging and the KV commit)
/// that a multi-token quantum spreads over its tokens. Streams then deliver
/// up to this many tokens at a time.
const ACCELERATOR_DECODE_QUANTUM: usize = 8;
const MAX_DECODE_QUANTUM: usize = 16;

/// Decode tokens per engine step for a request decoding alone:
/// `IZWI_QWEN35_DECODE_QUANTUM` (1..=16) when set, else 8 on CUDA and Metal
/// and 1 on the CPU, where a slow token makes long bursts.
fn preferred_decode_tokens_for(backend: BackendKind) -> usize {
    let configured = std::env::var("IZWI_QWEN35_DECODE_QUANTUM")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0);
    configured.map_or(
        match backend {
            BackendKind::Cuda | BackendKind::Metal => ACCELERATOR_DECODE_QUANTUM,
            BackendKind::Cpu => 1,
        },
        |value| value.min(MAX_DECODE_QUANTUM),
    )
}

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
/// Fully prepared Qwen3.5 prefill input. The runtime carries this exact
/// artifact into the executor so media loading and vision encoding happen once.
#[derive(Debug, Clone)]
pub struct Qwen35PreparedPrompt {
    prompt_ids: Vec<u32>,
    prompt_positions: Vec<[usize; 3]>,
    next_text_position: usize,
    vision_inputs: Option<PreparedVisionInputs>,
}

impl Qwen35PreparedPrompt {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    pub(crate) fn prompt_positions(&self) -> &[[usize; 3]] {
        &self.prompt_positions
    }
}

fn resolve_prepared_prompt<F>(
    prepared: Option<&Qwen35PreparedPrompt>,
    prepare: F,
) -> Result<Qwen35PreparedPrompt>
where
    F: FnOnce() -> Result<Qwen35PreparedPrompt>,
{
    match prepared {
        Some(prepared) => Ok(prepared.clone()),
        None => prepare(),
    }
}

/// Initial token history for the sampling penalties. The prompt is kept
/// only when the repetition penalty reads it, because the presence penalty
/// reads generated tokens alone (vLLM semantics, which Qwen's published
/// sampling profiles assume).
fn initial_penalty_history(
    prompt_ids: &[u32],
    max_new_tokens: usize,
    track_history: bool,
    include_prompt: bool,
) -> Vec<u32> {
    if !track_history {
        return Vec::new();
    }

    let prompt = if include_prompt { prompt_ids } else { &[] };
    let mut history = Vec::with_capacity(prompt.len().saturating_add(max_new_tokens.max(1)));
    history.extend_from_slice(prompt);
    history
}

/// Penalty histories for the next sample. `history_ids` starts with
/// `prompt_len` prompt tokens (zero when the prompt is not tracked) and
/// continues with the generated tokens.
fn penalty_history(
    track_history: bool,
    history_ids: &[u32],
    prompt_len: usize,
) -> PenaltyHistory<'_> {
    if !track_history {
        return PenaltyHistory::uniform(&[]);
    }
    PenaltyHistory {
        repetition: history_ids,
        presence: &history_ids[prompt_len.min(history_ids.len())..],
    }
}

fn expected_physical_decode_cursor(prefill_progress: usize, tokens_generated: usize) -> usize {
    prefill_progress.saturating_add(tokens_generated.saturating_sub(1))
}

pub struct ChatDecodeState {
    text_state: Qwen35TextRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    /// Model output awaiting sampling inside the current executor quantum.
    /// This slot is drained before the state is returned to the executor.
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    /// Append-only journal of the emitted tokens, which a capacity replay
    /// writes back after the prompt. Step rollback keeps only its length.
    generated_ids: Vec<u32>,
    history_ids: Vec<u32>,
    /// Number of prompt tokens at the start of `history_ids`.
    penalty_prompt_len: usize,
    decoder: IncrementalDecoder,
    tokens_generated: usize,
    track_history: bool,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
    next_text_position: usize,
    /// Scheduler-visible prompt cursor, independent of text position IDs.
    prefill_progress: usize,
    /// Number of prepared vision rows consumed by committed prompt spans.
    prefill_vision_progress: usize,
    config: ChatGenerationConfig,
    rng: SimpleRng,
    /// DS9.3: logprob entries produced by the current decode step, drained
    /// by the registry right after the step. Cleared at each sample.
    pub(crate) pending_logprobs: Vec<crate::engine::TokenLogprob>,
    /// DS9.2: per-request constrained-decoding runtime, present only when
    /// the request asked for `response_format: json_object`.
    grammar: Option<crate::models::shared::sampling::GrammarRuntime>,
    /// The loaded model's decode counters.
    telemetry: Arc<Qwen35Telemetry>,
    /// Cache length a capacity replay is rebuilding; the state decodes again
    /// once its KV pages and recurrent state reach it.
    replay_end: Option<usize>,
}

/// CPU continuation record of a session released for capacity: the
/// generated-token journal (the prompt is the request's prepared prompt) and
/// the sampler, grammar and text state at a quantum boundary. It owns no
/// tensors; the released KV pages and DeltaNet/convolution state are rebuilt
/// by replaying the prompt and every generated token but the pending one.
#[derive(Clone)]
pub(crate) struct Qwen35ReplayCheckpoint {
    replay_tokens: usize,
    pending_token: u32,
    generated_ids: Vec<u32>,
    history_ids: Vec<u32>,
    penalty_prompt_len: usize,
    decoder: IncrementalDecoder,
    tokens_generated: usize,
    track_history: bool,
    assembled: String,
    max_new_tokens: usize,
    next_text_position: usize,
    prefill_progress: usize,
    config: ChatGenerationConfig,
    rng: SimpleRng,
    grammar: Option<crate::models::shared::sampling::GrammarRuntime>,
}

impl Qwen35ReplayCheckpoint {
    pub(crate) fn replay_tokens(&self) -> usize {
        self.replay_tokens
    }
}

impl ChatDecodeState {
    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn uses_physical_kv(&self) -> bool {
        true
    }

    /// The replay a capacity suspension still has to rebuild, if any.
    pub(crate) fn replay_tokens(&self) -> Option<usize> {
        self.replay_end
    }

    /// Capture the CPU continuation record at a quantum boundary, before the
    /// caller releases the KV pages and recurrent state.
    pub(crate) fn replay_checkpoint(&self) -> Result<Qwen35ReplayCheckpoint> {
        let replay_tokens = self
            .replay_end
            .unwrap_or_else(|| self.physical_kv.context_len());
        let pending_token = self.pending_token.filter(|_| {
            !self.finished
                && self.unconsumed_output.is_none()
                && self.generated_ids.len() == self.tokens_generated
                && replay_tokens + 1 == self.prefill_progress + self.tokens_generated
        });
        let Some(pending_token) = pending_token else {
            return Err(Error::InvalidInput(
                "a Qwen3.5 session suspends only between decode quanta".into(),
            ));
        };
        Ok(Qwen35ReplayCheckpoint {
            replay_tokens,
            pending_token,
            generated_ids: self.generated_ids.clone(),
            history_ids: self.history_ids.clone(),
            penalty_prompt_len: self.penalty_prompt_len,
            decoder: self.decoder.clone(),
            tokens_generated: self.tokens_generated,
            track_history: self.track_history,
            assembled: self.assembled.clone(),
            max_new_tokens: self.max_new_tokens,
            next_text_position: self.next_text_position,
            prefill_progress: self.prefill_progress,
            config: self.config.clone(),
            rng: self.rng.clone(),
            grammar: self.grammar.clone(),
        })
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
    ) -> Result<Qwen35SharedStepCheckpoint> {
        if self.physical_kv.arena().id() != cache.arena().id()
            || self.physical_kv.context_len() != cache.context_len()
        {
            return Err(Error::InferenceError(
                "Qwen3.5 shared-step KV reservation does not continue the session".into(),
            ));
        }
        Ok(Qwen35SharedStepCheckpoint {
            text_state: self.text_state.clone(),
            physical_kv: std::mem::replace(&mut self.physical_kv, cache),
            unconsumed_output: self.unconsumed_output.clone(),
            pending_token: self.pending_token,
            generated_len: self.generated_ids.len(),
            history_ids: self.history_ids.clone(),
            decoder: self.decoder.clone(),
            tokens_generated: self.tokens_generated,
            assembled: self.assembled.clone(),
            finished: self.finished,
            next_text_position: self.next_text_position,
            rng: self.rng.clone(),
        })
    }

    pub(crate) fn rollback_shared_step_quantum(&mut self, checkpoint: Qwen35SharedStepCheckpoint) {
        self.text_state = checkpoint.text_state;
        self.physical_kv = checkpoint.physical_kv;
        self.unconsumed_output = checkpoint.unconsumed_output;
        self.pending_token = checkpoint.pending_token;
        self.generated_ids.truncate(checkpoint.generated_len);
        self.history_ids = checkpoint.history_ids;
        self.decoder = checkpoint.decoder;
        self.tokens_generated = checkpoint.tokens_generated;
        self.assembled = checkpoint.assembled;
        self.finished = checkpoint.finished;
        self.next_text_position = checkpoint.next_text_position;
        self.rng = checkpoint.rng;
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
        let conv_ring_stacks = self.text_state.stage_tensor_domains(
            arena,
            PhysicalStateTransactionId::new(transaction)?,
            target_cursor,
        )?;
        self.telemetry.record_conv_ring_stacks(conv_ring_stacks);
        Ok(())
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

pub(crate) struct Qwen35SharedStepCheckpoint {
    text_state: Qwen35TextRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    generated_len: usize,
    history_ids: Vec<u32>,
    decoder: IncrementalDecoder,
    tokens_generated: usize,
    assembled: String,
    finished: bool,
    next_text_position: usize,
    rng: SimpleRng,
}

#[derive(Debug, Clone)]
pub struct Qwen35TextConfig {
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

pub(crate) struct Qwen35Tokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    literal_special_tokens: SpecialTokenMatcher,
    chat_template: String,
    default_enable_thinking: bool,
    bos_token: Option<String>,
}

/// Literal special tokens matched in one pass over a prompt. At each byte
/// only the tokens starting with that byte are tried, longest first, so a
/// match is the earliest position and then the longest token there. Prompts
/// with thousands of `<|image_pad|>` tokens stay linear; a per-match search
/// of the remaining text for every special token was quadratic.
#[derive(Debug, Clone)]
struct SpecialTokenMatcher {
    tokens: Vec<(String, u32)>,
    /// Indices into `tokens` by first byte, longest token first.
    by_first_byte: Vec<Vec<usize>>,
}

/// A prompt piece: plain text for the BPE tokenizer, or a special token id.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PromptSegment<'a> {
    Text(&'a str),
    Special(u32),
}

impl SpecialTokenMatcher {
    /// Tokens sorted longest first (then lexically) with duplicates removed;
    /// the first id listed for a duplicated string wins.
    fn new(mut tokens: Vec<(String, u32)>) -> Self {
        tokens.retain(|(token, _)| !token.is_empty());
        tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });
        tokens.dedup_by(|(left, _), (right, _)| left == right);
        let mut by_first_byte = vec![Vec::new(); 256];
        for (index, (token, _)) in tokens.iter().enumerate() {
            by_first_byte[usize::from(token.as_bytes()[0])].push(index);
        }
        Self {
            tokens,
            by_first_byte,
        }
    }

    fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The first special token at or after byte `start`, as
    /// `(byte offset, byte length, id)`. Tokens are valid UTF-8 and begin
    /// with a leading byte, so every match starts on a char boundary.
    fn next_match(&self, text: &str, start: usize) -> Option<(usize, usize, u32)> {
        let bytes = text.as_bytes();
        (start..bytes.len()).find_map(|offset| {
            self.by_first_byte[usize::from(bytes[offset])]
                .iter()
                .map(|index| &self.tokens[*index])
                .find(|(token, _)| bytes[offset..].starts_with(token.as_bytes()))
                .map(|(token, id)| (offset, token.len(), *id))
        })
    }

    fn segments<'a>(&self, text: &'a str) -> Vec<PromptSegment<'a>> {
        let mut segments = Vec::new();
        let mut offset = 0usize;
        while let Some((start, len, id)) = self.next_match(text, offset) {
            if start > offset {
                segments.push(PromptSegment::Text(&text[offset..start]));
            }
            segments.push(PromptSegment::Special(id));
            offset = start + len;
        }
        if offset < text.len() {
            segments.push(PromptSegment::Text(&text[offset..]));
        }
        segments
    }
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

impl Qwen35Tokenizer {
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
        let literal_special_tokens = SpecialTokenMatcher::new(literal_special_tokens);

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
        for segment in self.literal_special_tokens.segments(text) {
            match segment {
                PromptSegment::Text(text) => ids.extend(self.inner.encode(text)?),
                PromptSegment::Special(id) => ids.push(id),
            }
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

/// Execution core the dense `Qwen35ChatModel` drives: tokenizer, hybrid
/// trunk configuration, and the decode-state machinery. The wrapper adds
/// vision encoding on top.
pub(crate) struct Qwen35ChatExec {
    pub(crate) variant: ModelVariant,
    pub(crate) tokenizer: Qwen35Tokenizer,
    pub(crate) text_config: Qwen35TextConfig,
    pub(crate) text_model: Qwen35TextModel,
}

pub struct Qwen35ChatModel {
    device_kind: BackendKind,
    exec: Qwen35ChatExec,
    text_checkpoint: GgufModelInfo,
    projector_checkpoint: GgufModelInfo,
    vision_model: Qwen35VisionModel,
}

impl Qwen35ChatExec {
    pub(crate) fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub(crate) fn text_config(&self) -> &Qwen35TextConfig {
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
        qwen35_composite_cache_contract(&self.text_config, attention_dtype, preferred_page_tokens)
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
    ) -> Result<Qwen35PreparedPrompt> {
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
        Ok(Qwen35PreparedPrompt {
            next_text_position: prompt_positions.len(),
            prompt_ids,
            prompt_positions,
            vision_inputs: None,
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
        prepared: &Qwen35PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if prepared.prompt_ids.is_empty() || cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 physical prefill requires a non-empty prompt and an empty reservation"
                    .into(),
            ));
        }
        let track_history =
            config.repetition_penalty > 1.0 || config.presence_penalty.abs() > f32::EPSILON;
        let include_prompt = track_history && config.repetition_penalty > 1.0;
        Ok(ChatDecodeState {
            text_state: self.text_model.new_state(),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: None,
            generated_ids: Vec::with_capacity(max_new_tokens.max(1)),
            history_ids: initial_penalty_history(
                &prepared.prompt_ids,
                max_new_tokens,
                track_history,
                include_prompt,
            ),
            penalty_prompt_len: if include_prompt {
                prepared.prompt_ids.len()
            } else {
                0
            },
            decoder: IncrementalDecoder::new(config.skip_special_tokens),
            tokens_generated: 0,
            track_history,
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            next_text_position: prepared.next_text_position,
            prefill_progress: 0,
            pending_logprobs: Vec::new(),
            prefill_vision_progress: 0,
            config: config.clone(),
            rng: SimpleRng::new(config.seed),
            grammar: self.grammar_runtime(config),
            telemetry: self.text_model.telemetry().clone(),
            replay_end: None,
        })
    }

    /// Start rebuilding a suspended session in an empty reservation and a
    /// fresh recurrent state. The state decodes again after
    /// [`Self::continue_replay`] reaches the end of the checkpoint's replay.
    pub(crate) fn begin_replay_state_physical(
        &self,
        saved: &Qwen35ReplayCheckpoint,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if cache.context_len() != 0 || saved.replay_tokens == 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 replay requires an empty reservation and a non-empty journal".into(),
            ));
        }
        Ok(ChatDecodeState {
            text_state: self.text_model.new_state(),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: Some(saved.pending_token),
            generated_ids: saved.generated_ids.clone(),
            history_ids: saved.history_ids.clone(),
            penalty_prompt_len: saved.penalty_prompt_len,
            decoder: saved.decoder.clone(),
            tokens_generated: saved.tokens_generated,
            track_history: saved.track_history,
            assembled: saved.assembled.clone(),
            max_new_tokens: saved.max_new_tokens,
            finished: false,
            next_text_position: saved.next_text_position,
            prefill_progress: saved.prefill_progress,
            // The replayed prompt consumes its vision rows again.
            prefill_vision_progress: 0,
            config: saved.config.clone(),
            rng: saved.rng.clone(),
            pending_logprobs: Vec::new(),
            grammar: saved.grammar.clone(),
            telemetry: self.text_model.telemetry().clone(),
            replay_end: Some(saved.replay_tokens),
        })
    }

    /// Write `[span_start, span_end)` of the replay journal (the prepared
    /// prompt, vision rows included, then the generated tokens at their text
    /// positions) into the KV pages and recurrent state without sampling or
    /// emitting. Returns whether the replay is complete.
    pub(crate) fn continue_replay(
        &self,
        state: &mut ChatDecodeState,
        prepared: &Qwen35PreparedPrompt,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        let replay_end = state
            .replay_end
            .ok_or_else(|| Error::InvalidInput("Qwen3.5 state has no pending replay".into()))?;
        let prompt_len = prepared.prompt_ids.len();
        if span_start != state.physical_kv.context_len()
            || span_start >= span_end
            || span_end > replay_end
            || prompt_len != state.prefill_progress
            || replay_end > prompt_len + state.generated_ids.len()
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 replay span [{span_start},{span_end}) does not continue cursor {} of {replay_end}",
                state.physical_kv.context_len()
            )));
        }
        if span_start < prompt_len {
            self.prefill_prompt_segments(
                state,
                prepared,
                span_start,
                span_end.min(prompt_len),
                false,
            )?;
            if span_end >= prompt_len {
                self.check_prompt_vision_consumed(state, prepared)?;
            }
        }
        let chunk_size = qwen35_prefill_chunk_size();
        let mut cursor = span_start.max(prompt_len);
        while cursor < span_end {
            let end = span_end.min(cursor.saturating_add(chunk_size));
            let positions = (cursor..end)
                .map(|index| [prepared.next_text_position + index - prompt_len; 3])
                .collect::<Vec<_>>();
            self.text_model.prefill_token_ids_physical(
                &state.generated_ids[cursor - prompt_len..end - prompt_len],
                &positions,
                &mut state.text_state,
                &mut state.physical_kv,
                false,
            )?;
            cursor = end;
        }
        if state.physical_kv.context_len() != span_end {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 replay committed physical cursor {} instead of {span_end}",
                state.physical_kv.context_len()
            )));
        }
        let complete = span_end == replay_end;
        if complete {
            state.replay_end = None;
        }
        Ok(complete)
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
        prepared: &Qwen35PreparedPrompt,
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
        let logits =
            self.prefill_prompt_segments(state, prepared, span_start, span_end, complete)?;
        if state.physical_kv.context_len() != span_end {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 resumable prefill committed physical cursor {} instead of {span_end}",
                state.physical_kv.context_len()
            )));
        }
        state.prefill_progress = span_end;
        if complete {
            self.check_prompt_vision_consumed(state, prepared)?;
            state.unconsumed_output = Some(logits.ok_or_else(|| {
                Error::InferenceError("Qwen3.5 final prefill span produced no logits".into())
            })?);
        }
        Ok(complete)
    }

    /// Prefill `[span_start, span_end)` of the prepared prompt, image
    /// placeholders from their vision rows, in chunks; logits only for the
    /// span's last token and only when `logits_at_end`.
    fn prefill_prompt_segments(
        &self,
        state: &mut ChatDecodeState,
        prepared: &Qwen35PreparedPrompt,
        span_start: usize,
        span_end: usize,
        logits_at_end: bool,
    ) -> Result<Option<Tensor>> {
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
                let vision = prepared.vision_inputs.as_ref().ok_or_else(|| {
                    Error::InvalidInput("Qwen3.5 image placeholder has no vision input".into())
                })?;
                let image_count = segment_end - cursor;
                let embeddings = vision
                    .embeddings
                    .narrow(0, state.prefill_vision_progress, image_count)?
                    .reshape((1, image_count, self.text_model.hidden_size()))?;
                let compute_logits = logits_at_end && segment_end == span_end;
                logits = self.text_model.prefill_input_embeddings_physical(
                    &embeddings,
                    &prepared.prompt_positions[cursor..segment_end],
                    &mut state.text_state,
                    &mut state.physical_kv,
                    compute_logits,
                )?;
                state.prefill_vision_progress += image_count;
                cursor = segment_end;
                continue;
            }
            let compute_logits = logits_at_end && segment_end == span_end;
            logits = self.text_model.prefill_token_ids_physical(
                &prepared.prompt_ids[cursor..segment_end],
                &prepared.prompt_positions[cursor..segment_end],
                &mut state.text_state,
                &mut state.physical_kv,
                compute_logits,
            )?;
            cursor = segment_end;
        }
        Ok(logits)
    }

    fn check_prompt_vision_consumed(
        &self,
        state: &ChatDecodeState,
        prepared: &Qwen35PreparedPrompt,
    ) -> Result<()> {
        let prepared_vision_rows = prepared
            .vision_inputs
            .as_ref()
            .map(|vision| vision.embeddings.dim(0))
            .transpose()?
            .unwrap_or(0);
        if state.prefill_vision_progress != prepared_vision_rows {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 prefill consumed {} of {prepared_vision_rows} prepared vision rows",
                state.prefill_vision_progress
            )));
        }
        Ok(())
    }

    pub(crate) fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.replay_end.is_some() {
            return Err(Error::InferenceError(
                "Qwen3.5 decode cannot run before its replay completes".into(),
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
            state.telemetry.record_solo_decode_step();
        }

        let history = penalty_history(
            state.track_history,
            &state.history_ids,
            state.penalty_prompt_len,
        );
        state.pending_logprobs.clear();
        let next = {
            let output = state.unconsumed_output.take().ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3.5 decode quantum has no unconsumed model output".to_string(),
                )
            })?;
            let reads_before = thread_host_reads();
            let (token, raw_logprobs, path) = if let Some(grammar) = state.grammar.as_mut() {
                let (token, raw_logprobs) = grammar.sample_token_with_history(
                    &output,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?;
                (token, raw_logprobs, SamplerPath::Grammar)
            } else {
                sample_next_token_with_logprobs(
                    &output,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?
            };
            state
                .telemetry
                .record_sample(path, thread_host_reads().wrapping_sub(reads_before));
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
        for state in states.iter() {
            let expected_physical_cursor =
                expected_physical_decode_cursor(state.prefill_progress, state.tokens_generated);
            if state.finished
                || state.tokens_generated >= state.max_new_tokens
                || state.unconsumed_output.is_some()
                || state.pending_token.is_none()
                || state.replay_end.is_some()
                || state.physical_kv.context_len() != expected_physical_cursor
            {
                return Err(Error::InvalidInput(
                    "continuous chat batch contains a non-decodable Qwen3.5 state".into(),
                ));
            }
        }
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
        if let Some(state) = states.first() {
            state.telemetry.record_batched_decode_step(states.len());
        }

        let mut sampled = Vec::with_capacity(states.len());
        for (row, state) in states.iter_mut().enumerate() {
            let history = penalty_history(
                state.track_history,
                &state.history_ids,
                state.penalty_prompt_len,
            );
            state.pending_logprobs.clear();
            let row_logits = logits.i((row, 0))?;
            let reads_before = thread_host_reads();
            let (token, raw_logprobs, path) = if let Some(grammar) = state.grammar.as_mut() {
                let (token, raw_logprobs) = grammar.sample_token_with_history(
                    &row_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?;
                (token, raw_logprobs, SamplerPath::Grammar)
            } else {
                sample_next_token_with_logprobs(
                    &row_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?
            };
            state
                .telemetry
                .record_sample(path, thread_host_reads().wrapping_sub(reads_before));
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
        if self.is_stop_token(token, &state.config) {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(delta);
        }
        let mut delta = self
            .tokenizer
            .decode_token_delta(&mut state.decoder, token)?;
        state.generated_ids.push(token);
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

impl InferenceStateContractProvider for Qwen35ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        let dtype = match self.device_kind {
            BackendKind::Cuda => DType::F16,
            BackendKind::Cpu | BackendKind::Metal => DType::F32,
        };
        Ok(InferenceStateCapability::Managed(
            self.managed_composite_cache_contract(dtype, default_kv_page_size())?,
        ))
    }
}

impl Qwen35ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        Self::load_with_performance(
            model_dir,
            variant,
            device,
            &crate::performance::PerformanceConfig::default(),
        )
    }

    /// Load and resolve the fused fast paths under the global performance
    /// switches (`cuda.mode`, `cuda.fused_decode`; CUDA only) and the
    /// `IZWI_QWEN35_*` switches.
    pub fn load_with_performance(
        model_dir: &Path,
        variant: ModelVariant,
        device: DeviceProfile,
        performance: &crate::performance::PerformanceConfig,
    ) -> Result<Self> {
        let gguf_path = model_dir.join(qwen35_gguf_filename(variant)?);
        let mmproj_path = model_dir.join("mmproj-F16.gguf");

        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }
        if !mmproj_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 projector checkpoint not found: {}",
                mmproj_path.display()
            )));
        }

        let backend = BackendKind::from(device.kind);
        let text_loader = GgufLoader::from_path_with_backend(&gguf_path, backend)?;
        let text_checkpoint = text_loader.get_model_info();
        let architecture = text_checkpoint
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        if architecture != "qwen35" {
            return Err(Error::ModelLoadError(format!(
                "Expected general.architecture=qwen35 for {}, found {}",
                gguf_path.display(),
                architecture
            )));
        }

        let text_config = parse_text_config(&text_loader)?;
        debug!(variant = %variant, ?text_config, "Resolved Qwen3.5 text configuration");
        let tokenizer = Qwen35Tokenizer::load(model_dir, variant, &text_loader)?;
        let mut text_model = Qwen35TextModel::load(&text_loader, &text_config, &device.device)?;
        text_model.resolve_fast_paths(
            Qwen35CudaSwitches::from_performance(backend, &performance.cuda),
            false,
        );

        let projector_loader = GgufLoader::from_path_with_backend(&mmproj_path, backend)?;
        let projector_checkpoint = projector_loader.get_model_info();
        let vision_model =
            Qwen35VisionModel::load(&projector_loader, &device.device, text_model.hidden_size())?;

        info!(
            "Loaded Qwen3.5 chat assets for {} on {:?} ({} text tensors, {} projector tensors)",
            variant.display_name(),
            device.kind,
            text_loader.tensor_count(),
            projector_loader.tensor_count()
        );

        Ok(Self {
            device_kind: backend,
            exec: Qwen35ChatExec {
                variant,
                tokenizer,
                text_config,
                text_model,
            },
            text_checkpoint,
            projector_checkpoint,
            vision_model,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.exec.variant()
    }

    pub fn text_config(&self) -> &Qwen35TextConfig {
        self.exec.text_config()
    }

    pub fn max_context_tokens(&self) -> Result<usize> {
        self.exec.max_context_tokens()
    }

    /// Prefill span for an isolated request (see
    /// [`crate::engine::ExecutionProfile::preferred_prefill_tokens`]).
    pub fn preferred_prefill_tokens(&self) -> usize {
        preferred_prefill_tokens_for(self.device_kind)
    }

    /// Decode tokens per engine step for a request decoding alone. The
    /// scheduler grants it only to an isolated row with no waiting work;
    /// shared batches stay at one token per row.
    pub fn preferred_decode_tokens(&self) -> usize {
        preferred_decode_tokens_for(self.device_kind)
    }

    /// Whether the decode quantum outlives the request's soft scheduling SLA.
    /// Interactive and streaming chat pass that SLA (about 0.7 s) early in
    /// almost every stream, after which the scheduler would grant one token
    /// per step. The quantum still yields to waiting or co-batched work and
    /// to hard deadlines.
    pub fn sustained_decode_quantum(&self) -> bool {
        self.preferred_decode_tokens() > 1
    }

    /// Hybrid retained-state contract shared by loading, scheduling, and the
    /// native model adapter.
    pub(crate) fn managed_composite_cache_contract(
        &self,
        attention_dtype: DType,
        preferred_page_tokens: usize,
    ) -> Result<InferenceStateContract> {
        self.exec
            .managed_composite_cache_contract(attention_dtype, preferred_page_tokens)
    }

    pub fn chat_template(&self) -> &str {
        self.exec.chat_template()
    }

    pub fn default_enable_thinking(&self) -> bool {
        self.exec.default_enable_thinking()
    }

    pub fn text_checkpoint(&self) -> &GgufModelInfo {
        &self.text_checkpoint
    }

    pub fn projector_checkpoint(&self) -> &GgufModelInfo {
        &self.projector_checkpoint
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.prompt_token_ids_with_config(messages, &ChatGenerationConfig::default())
    }

    pub fn prompt_token_ids_with_config(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Vec<u32>> {
        Ok(self
            .prepare_prompt_for_execution(messages, config)?
            .prompt_ids)
    }

    pub fn prepare_prompt_for_execution(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Qwen35PreparedPrompt> {
        self.prepare_prompt(messages, config)
    }

    pub fn supports_incremental_decode(&self) -> bool {
        self.exec.supports_incremental_decode()
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        self.exec.supports_continuous_decode_batch()
    }

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        self.exec.continuous_decode_batch_workspace_per_row_bytes()
    }

    pub fn device_kind(&self) -> BackendKind {
        self.device_kind
    }

    /// Serving diagnostics for the admin model API: device, storage and
    /// scheduling preferences, and cumulative decode counters.
    pub fn runtime_diagnostics(&self) -> serde_json::Value {
        let kv_dtype = match self.device_kind {
            BackendKind::Cuda => DType::F16,
            BackendKind::Cpu | BackendKind::Metal => DType::F32,
        };
        serde_json::json!({
            "family": "qwen35_chat",
            "device": self.device_kind.as_str(),
            "compute_dtype": "F32",
            "kv_dtype": format!("{kv_dtype:?}"),
            "scheduling": {
                "preferred_prefill_tokens": self.preferred_prefill_tokens(),
                "prefill_chunk_tokens": qwen35_prefill_chunk_size(),
                "preferred_decode_tokens": self.preferred_decode_tokens(),
                "sustained_decode_quantum": self.sustained_decode_quantum(),
            },
            "gdn_decode": self.exec.text_model.gdn_decode_summary(),
            "projections": self.exec.text_model.projection_summary(),
            "decode_graph": self.exec.text_model.decode_graph_summary(),
            "counters": self.exec.text_model.telemetry().snapshot(),
        })
    }

    pub(crate) fn start_decode_state_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen35PreparedPrompt>,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        let prepared = resolve_prepared_prompt(prepared, || self.prepare_prompt(messages, config))?;
        let mut state = self.exec.begin_resumable_prefill_state_physical(
            &prepared,
            max_new_tokens,
            config,
            cache,
        )?;
        self.exec.continue_resumable_prefill_physical(
            &mut state,
            &prepared,
            0,
            prepared.prompt_ids.len(),
        )?;
        Ok(state)
    }

    pub(crate) fn begin_resumable_prefill_state_physical(
        &self,
        prepared: &Qwen35PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        self.exec
            .begin_resumable_prefill_state_physical(prepared, max_new_tokens, config, cache)
    }

    pub(crate) fn continue_resumable_prefill_physical(
        &self,
        state: &mut ChatDecodeState,
        prepared: &Qwen35PreparedPrompt,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        self.exec
            .continue_resumable_prefill_physical(state, prepared, span_start, span_end)
    }

    pub(crate) fn begin_replay_state_physical(
        &self,
        saved: &Qwen35ReplayCheckpoint,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        self.exec.begin_replay_state_physical(saved, cache)
    }

    pub(crate) fn continue_replay(
        &self,
        state: &mut ChatDecodeState,
        prepared: &Qwen35PreparedPrompt,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        self.exec
            .continue_replay(state, prepared, span_start, span_end)
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        self.exec.decode_step(state)
    }

    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        self.exec.decode_step_batch(states)
    }

    fn prepare_prompt(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Qwen35PreparedPrompt> {
        let prompt = render_prompt(messages, config, self.default_enable_thinking())?;
        let image_placeholders = prompt.matches(IMAGE_PAD_PLACEHOLDER).count();
        let video_placeholders = prompt.matches(VIDEO_PAD_PLACEHOLDER).count();
        if video_placeholders > 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 video inputs are not implemented yet".to_string(),
            ));
        }

        let Some(vision_inputs) = self
            .vision_model
            .encode_media(&config.request.media_inputs)?
        else {
            return self.exec.prepare_text_prompt(messages, config);
        };

        if image_placeholders == 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 media inputs require <|image_pad|> placeholders in the rendered prompt"
                    .to_string(),
            ));
        }
        if image_placeholders != vision_inputs.token_counts.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 prompt/media mismatch: rendered {image_placeholders} image placeholders for {} media inputs",
                vision_inputs.token_counts.len()
            )));
        }

        let expanded_prompt = expand_image_placeholders(&prompt, &vision_inputs.token_counts)?;
        let prompt_ids = self.exec.tokenizer.encode_text(&expanded_prompt)?;
        let (prompt_positions, next_text_position) = build_prompt_positions(
            &prompt_ids,
            &vision_inputs,
            self.exec.tokenizer.specials.image_pad,
            self.exec.tokenizer.specials.video_pad,
            self.vision_model.spatial_merge_size(),
        )?;
        Ok(Qwen35PreparedPrompt {
            prompt_ids,
            prompt_positions,
            next_text_position,
            vision_inputs: Some(vision_inputs),
        })
    }
}

fn build_text_positions(token_count: usize) -> Vec<[usize; 3]> {
    (0..token_count).map(|idx| [idx; 3]).collect()
}

fn expand_image_placeholders(prompt: &str, token_counts: &[usize]) -> Result<String> {
    let segments: Vec<&str> = prompt.split(IMAGE_PAD_PLACEHOLDER).collect();
    if segments.len() != token_counts.len() + 1 {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 prompt/media mismatch while expanding image placeholders: found {} placeholders for {} media inputs",
            segments.len().saturating_sub(1),
            token_counts.len()
        )));
    }

    let mut expanded = String::with_capacity(
        prompt.len()
            + token_counts
                .iter()
                .map(|count| IMAGE_PAD_PLACEHOLDER.len() * count.saturating_sub(1))
                .sum::<usize>(),
    );
    for (idx, segment) in segments.iter().enumerate() {
        expanded.push_str(segment);
        if let Some(&token_count) = token_counts.get(idx) {
            if token_count == 0 {
                return Err(Error::InvalidInput(
                    "Qwen3.5 image inputs must contribute at least one LLM token".to_string(),
                ));
            }
            expanded.push_str(&IMAGE_PAD_PLACEHOLDER.repeat(token_count));
        }
    }
    Ok(expanded)
}

fn build_prompt_positions(
    prompt_ids: &[u32],
    vision_inputs: &PreparedVisionInputs,
    image_pad_id: u32,
    video_pad_id: u32,
    spatial_merge_size: usize,
) -> Result<(Vec<[usize; 3]>, usize)> {
    let mut prompt_positions = Vec::with_capacity(prompt_ids.len());
    let mut current_pos = 0usize;
    let mut idx = 0usize;
    let mut image_idx = 0usize;

    while idx < prompt_ids.len() {
        let token_id = prompt_ids[idx];
        if token_id == image_pad_id {
            let Some(grid) = vision_inputs.grids.get(image_idx).copied() else {
                return Err(Error::InvalidInput(
                    "Qwen3.5 prompt consumed more image placeholder runs than media inputs"
                        .to_string(),
                ));
            };
            let expected_tokens = *vision_inputs.token_counts.get(image_idx).ok_or_else(|| {
                Error::InvalidInput(
                    "Qwen3.5 image token counts are missing for a prepared media input".to_string(),
                )
            })?;
            let run_start = idx;
            while idx < prompt_ids.len() && prompt_ids[idx] == image_pad_id {
                idx += 1;
            }
            let actual_tokens = idx - run_start;
            if actual_tokens != expected_tokens {
                return Err(Error::InvalidInput(format!(
                    "Qwen3.5 image placeholder count mismatch for media input {}: prompt has {}, encoder expects {}",
                    image_idx + 1,
                    actual_tokens,
                    expected_tokens
                )));
            }
            prompt_positions.extend(vision_position_ids(current_pos, grid, spatial_merge_size)?);
            current_pos += grid[1].max(grid[2]) / spatial_merge_size;
            image_idx += 1;
            continue;
        }
        if token_id == video_pad_id {
            return Err(Error::InvalidInput(
                "Qwen3.5 video inputs are not implemented yet".to_string(),
            ));
        }

        let run_start = idx;
        while idx < prompt_ids.len()
            && prompt_ids[idx] != image_pad_id
            && prompt_ids[idx] != video_pad_id
        {
            idx += 1;
        }
        let text_len = idx - run_start;
        prompt_positions.extend((0..text_len).map(|offset| {
            let position = current_pos + offset;
            [position; 3]
        }));
        current_pos += text_len;
    }

    if image_idx != vision_inputs.grids.len() {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 prompt consumed {} image placeholder runs for {} media inputs",
            image_idx,
            vision_inputs.grids.len()
        )));
    }
    if prompt_positions.len() != prompt_ids.len() {
        return Err(Error::InferenceError(format!(
            "Qwen3.5 prompt position count mismatch: {} positions for {} tokens",
            prompt_positions.len(),
            prompt_ids.len()
        )));
    }

    Ok((prompt_positions, current_pos))
}

fn vision_position_ids(
    start_position: usize,
    grid_thw: [usize; 3],
    spatial_merge_size: usize,
) -> Result<Vec<[usize; 3]>> {
    if spatial_merge_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 spatial merge size must be non-zero".to_string(),
        ));
    }

    let llm_grid_t = grid_thw[0];
    let llm_grid_h = grid_thw[1] / spatial_merge_size;
    let llm_grid_w = grid_thw[2] / spatial_merge_size;
    let image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w;
    let mut positions = Vec::with_capacity(image_seq_length);
    for _temporal_idx in 0..llm_grid_t {
        for height_idx in 0..llm_grid_h {
            for width_idx in 0..llm_grid_w {
                positions.push([
                    start_position,
                    start_position + height_idx,
                    start_position + width_idx,
                ]);
            }
        }
    }
    Ok(positions)
}

fn render_fallback_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(match message.role {
            ChatRole::System => "system",
            ChatRole::User | ChatRole::Tool => "user",
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
            // The template's tool turn: a user turn wrapping the result.
            ChatRole::Tool => {
                prompt.push_str("<|im_start|>user\n<tool_response>\n");
                prompt.push_str(message.content.trim());
                prompt.push_str("\n</tool_response><|im_end|>\n");
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

fn qwen35_gguf_filename(variant: ModelVariant) -> Result<&'static str> {
    match variant {
        ModelVariant::Qwen3508BGguf => Ok("Qwen3.5-0.8B-Q4_K_M.gguf"),
        ModelVariant::Qwen352BGguf => Ok("Qwen3.5-2B-Q4_K_M.gguf"),
        ModelVariant::Qwen354BGguf => Ok("Qwen3.5-4B-Q4_K_M.gguf"),
        ModelVariant::Qwen359BGguf => Ok("Qwen3.5-9B-Q4_K_M.gguf"),
        _ => Err(Error::ModelLoadError(format!(
            "Unsupported Qwen3.5 GGUF chat variant: {variant}"
        ))),
    }
}

fn resolve_default_enable_thinking(_chat_template: &str, variant: ModelVariant) -> bool {
    matches!(
        variant,
        ModelVariant::Qwen354BGguf
            | ModelVariant::Qwen359BGguf
    )
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

fn parse_text_config(loader: &GgufLoader) -> Result<Qwen35TextConfig> {
    Ok(Qwen35TextConfig {
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
    history: PenaltyHistory<'_>,
    rng: &mut SimpleRng,
) -> Result<u32> {
    let output = output.take().ok_or_else(|| {
        Error::InferenceError("Qwen3.5 decode quantum has no unconsumed model output".to_string())
    })?;
    sample_next_token(&output, vocab_size, config, history, rng)
}

/// DS9.3: sample a token and, when the request asked for logprobs, resolve
/// the raw-distribution stats on host. Sampling math is unchanged. Also
/// returns the sampler path that chose the token.
fn sample_next_token_with_logprobs(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: PenaltyHistory<'_>,
    rng: &mut SimpleRng,
) -> Result<(
    u32,
    Option<crate::models::shared::sampling::RawTokenLogprobs>,
    SamplerPath,
)> {
    if !config.logprobs {
        let (token, path) = sample_next_token_traced(logits, vocab_size, config, history, rng)?;
        return Ok((token, None, path));
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
    let (token, path) = sample_next_token_traced(logits, vocab_size, config, history, rng)?;
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
        path,
    ))
}

fn sample_next_token(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: PenaltyHistory<'_>,
    rng: &mut SimpleRng,
) -> Result<u32> {
    sample_next_token_traced(logits, vocab_size, config, history, rng).map(|(token, _)| token)
}

/// [`sample_next_token`] plus the sampler path that chose the token.
fn sample_next_token_traced(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: PenaltyHistory<'_>,
    rng: &mut SimpleRng,
) -> Result<(u32, SamplerPath)> {
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "Qwen3.5 sampler received vocab_size=0".to_string(),
        ));
    }

    // Fast path for deterministic greedy decode (bench/default path):
    // avoid copying full logits tensors to CPU each token.
    let deterministic_greedy = config.temperature <= 1e-5
        && (config.repetition_penalty - 1.0).abs() <= f32::EPSILON
        && config.presence_penalty.abs() <= f32::EPSILON
        && config.top_k == 0
        && config.top_p >= 1.0;
    if deterministic_greedy {
        return argmax_clamped(logits, vocab_size).map(|token| (token, SamplerPath::Greedy));
    }

    if let Some(candidates) = bounded_device_sampling_candidates_with_history(
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
                return Ok((sampled, SamplerPath::Device));
            }
        }
    }

    sample_on_host(logits, vocab_size, config, history, rng).map(|token| (token, SamplerPath::Host))
}

/// Exact host sampling over the full logits row: penalties, temperature,
/// top-k, top-p, then the draw.
fn sample_on_host(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: PenaltyHistory<'_>,
    rng: &mut SimpleRng,
) -> Result<u32> {
    let mut values = logits_to_vec(logits)?;
    truncate_logits_to_vocab(&mut values, vocab_size);

    if config.repetition_penalty > 1.0 && !history.repetition.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history.repetition {
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

    if config.presence_penalty.abs() > f32::EPSILON && !history.presence.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history.presence {
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

    let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    record_host_read(DType::F32, values.len());
    Ok(values)
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
    // One readback carries the argmax and the logit it selected. Indices
    // stay exact in F32 (vocabularies are far below 2^24).
    let index = clamped.argmax(D::Minus1)?.reshape(1)?;
    let selected_logit = clamped.to_dtype(DType::F32)?.gather(&index, 0)?;
    let packed =
        Tensor::cat(&[&index.to_dtype(DType::F32)?, &selected_logit], 0)?.to_vec1::<f32>()?;
    record_host_read(DType::F32, packed.len());
    let (selected, selected_logit) = (packed[0] as u32, packed[1]);
    if selected_logit.is_finite() {
        return Ok(selected);
    }

    // Some device argmax kernels do not define useful ordering for NaNs. This
    // slow path runs only after the selected value is non-finite: it recovers a
    // finite candidate when one exists and otherwise returns useful counts for
    // the exact in-vocabulary row in every sampling mode.
    let values = clamped.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    record_host_read(DType::F32, values.len());
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

impl crate::models::shared::sampling::GrammarRng for SimpleRng {
    fn draw_unit(&mut self) -> f32 {
        self.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::DeviceProfile;
    use crate::models::shared::chat::ChatRequestConfig;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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

    fn synthetic_qwen35_tokenizer() -> Qwen35Tokenizer {
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

        Qwen35Tokenizer {
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
            literal_special_tokens: SpecialTokenMatcher::new(vec![
                ("<|im_start|>".to_string(), im_start),
                ("<|endoftext|>".to_string(), eos_alt),
                ("<|image_pad|>".to_string(), image_pad),
                ("<|video_pad|>".to_string(), video_pad),
                ("<|im_end|>".to_string(), im_end),
                ("</think>".to_string(), think_close),
                ("<think>".to_string(), think_open),
            ]),
            chat_template: String::new(),
            default_enable_thinking: false,
            bos_token: None,
        }
    }

    /// The previous quadratic matcher, kept as the reference: the earliest
    /// occurrence of any token, the longest token on a tie.
    fn reference_segments<'a>(tokens: &[(String, u32)], text: &'a str) -> Vec<PromptSegment<'a>> {
        let mut segments = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let tail = &text[offset..];
            let best = tokens
                .iter()
                .filter_map(|(token, id)| tail.find(token.as_str()).map(|at| (at, token, *id)))
                .min_by(|left, right| left.0.cmp(&right.0).then(right.1.len().cmp(&left.1.len())));
            let Some((at, token, id)) = best else {
                segments.push(PromptSegment::Text(tail));
                break;
            };
            if at > 0 {
                segments.push(PromptSegment::Text(&tail[..at]));
            }
            segments.push(PromptSegment::Special(id));
            offset += at + token.len();
        }
        segments
    }

    #[test]
    fn special_token_matcher_agrees_with_the_quadratic_reference() {
        let tokens = vec![
            ("<|im_start|>".to_string(), 1),
            ("<|im_end|>".to_string(), 2),
            ("<|image_pad|>".to_string(), 3),
            ("<think>".to_string(), 4),
            ("</think>".to_string(), 5),
            ("<|im".to_string(), 6),
            ("é<|".to_string(), 7),
        ];
        let matcher = SpecialTokenMatcher::new(tokens.clone());
        let pieces = [
            "<|im_start|>",
            "<|im_end|>",
            "<|image_pad|>",
            "<think>",
            "</think>",
            "<|im",
            "<|",
            "<",
            "|",
            ">",
            "im_",
            "end",
            "think",
            "/",
            "é",
            "日本",
            " ",
            "a",
            "\n",
        ];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for case in 0..2_000 {
            let mut text = String::new();
            for _ in 0..(case % 40) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                text.push_str(pieces[(state % pieces.len() as u64) as usize]);
            }
            assert_eq!(
                matcher.segments(&text),
                reference_segments(&tokens, &text),
                "{text:?}"
            );
        }
        // The longest token wins where several start at one byte.
        assert_eq!(
            matcher.segments("x<|im_end|>"),
            vec![PromptSegment::Text("x"), PromptSegment::Special(2)]
        );
        assert!(matcher.segments("").is_empty());
    }

    #[test]
    fn special_token_matcher_stays_linear_on_long_image_prompts() {
        let matcher = SpecialTokenMatcher::new(vec![
            ("<|image_pad|>".to_string(), 3),
            ("<|im_end|>".to_string(), 2),
        ]);
        let text = format!("start{}end", "<|image_pad|>".repeat(20_000));
        let segments = matcher.segments(&text);
        assert_eq!(segments.len(), 20_002);
        assert_eq!(segments[1], PromptSegment::Special(3));
    }

    fn local_model_dir(name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join("Library/Application Support/izwi/models")
            .join(name)
    }

    #[test]
    fn prepared_prompt_reuse_skips_fetch_and_vision_encode_builder() {
        let prepared = Qwen35PreparedPrompt {
            prompt_ids: vec![1, 2, 3],
            prompt_positions: build_text_positions(3),
            next_text_position: 3,
            vision_inputs: None,
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
        let mut history = initial_penalty_history(&prompt_ids, 8, true, true);
        assert_eq!(history, prompt_ids);
        history.push(99);
        assert_eq!(&history[..prompt_ids.len()], prompt_ids.as_slice());
        assert!(initial_penalty_history(&prompt_ids, 8, false, true).is_empty());
        assert!(initial_penalty_history(&prompt_ids, 8, true, false).is_empty());
    }

    #[test]
    fn presence_penalty_reads_generated_tokens_while_repetition_reads_the_prompt() {
        let prompt_ids = [248_046u32, 13, 7];
        let mut history = initial_penalty_history(&prompt_ids, 8, true, true);
        history.extend([5, 13]);
        let split = penalty_history(true, &history, prompt_ids.len());
        assert_eq!(split.repetition, &[248_046, 13, 7, 5, 13]);
        assert_eq!(split.presence, &[5, 13]);

        let generated_only = vec![5u32, 13];
        let presence_only = penalty_history(true, &generated_only, 0);
        assert_eq!(presence_only.repetition, &[5, 13]);
        assert_eq!(presence_only.presence, &[5, 13]);

        let untracked = penalty_history(false, &history, prompt_ids.len());
        assert!(untracked.repetition.is_empty() && untracked.presence.is_empty());
    }

    #[test]
    fn presence_penalty_spares_tokens_that_only_appear_in_the_prompt() {
        // Token 0 stands for `<|im_end|>`: it ends every prompt turn but has
        // not been generated. A presence penalty must not push it below
        // token 1, which appears nowhere. Token 2 was generated.
        let logits = Tensor::from_vec(vec![10.0f32, 9.0, -50.0], (3,), &candle_core::Device::Cpu)
            .expect("logits");
        let config = ChatGenerationConfig {
            temperature: 1e-3,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 2.0,
            stop_token_ids: Vec::new(),
            seed: 7,
            request: ChatRequestConfig::default(),
            logprobs: false,
            top_logprobs: 0,
            constrain_json_object: false,
            skip_special_tokens: true,
        };
        let history = [0u32, 2];
        let mut rng = SimpleRng::new(7);
        let split = PenaltyHistory {
            repetition: &history,
            presence: &history[1..],
        };
        assert_eq!(
            sample_next_token(&logits, 3, &config, split, &mut rng).unwrap(),
            0,
            "prompt-only token keeps its logit"
        );
        let mut rng = SimpleRng::new(7);
        let legacy = PenaltyHistory::uniform(&history);
        assert_eq!(
            sample_next_token(&logits, 3, &config, legacy, &mut rng).unwrap(),
            1,
            "a penalized token 0 falls below token 1"
        );
    }

    #[test]
    fn decode_quantum_is_eight_on_accelerators_and_one_on_cpu() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("IZWI_QWEN35_DECODE_QUANTUM");
        assert_eq!(preferred_decode_tokens_for(BackendKind::Cuda), 8);
        assert_eq!(preferred_decode_tokens_for(BackendKind::Metal), 8);
        assert_eq!(preferred_decode_tokens_for(BackendKind::Cpu), 1);
        for (value, expected) in [("1", 1), ("4", 4), ("99", MAX_DECODE_QUANTUM), ("0", 8)] {
            std::env::set_var("IZWI_QWEN35_DECODE_QUANTUM", value);
            assert_eq!(
                preferred_decode_tokens_for(BackendKind::Cuda),
                expected,
                "{value}"
            );
        }
        std::env::set_var("IZWI_QWEN35_DECODE_QUANTUM", "3");
        assert_eq!(preferred_decode_tokens_for(BackendKind::Cpu), 3);
        std::env::remove_var("IZWI_QWEN35_DECODE_QUANTUM");
    }

    #[test]
    fn accelerators_prefer_wide_isolated_prefill_spans() {
        for backend in [BackendKind::Cuda, BackendKind::Metal] {
            let span = preferred_prefill_tokens_for(backend);
            assert_eq!(span, ACCELERATOR_PREFERRED_PREFILL_TOKENS, "{backend:?}");
            assert_eq!(span % DEFAULT_PREFILL_CHUNK_SIZE, 0, "{backend:?}");
            assert!(span <= MAX_PREFILL_CHUNK_SIZE, "{backend:?}");
        }
        assert_eq!(preferred_prefill_tokens_for(BackendKind::Cpu), 0);
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
        let large = "{%- if add_generation_prompt %}{%- if enable_thinking is defined and enable_thinking is false %}{%- else %}<think>\n{%- endif %}{%- endif %}";

        assert!(!resolve_default_enable_thinking(
            large,
            ModelVariant::Qwen3508BGguf
        ));
        assert!(!resolve_default_enable_thinking(
            large,
            ModelVariant::Qwen352BGguf
        ));
        assert!(resolve_default_enable_thinking(
            small,
            ModelVariant::Qwen354BGguf
        ));
        assert!(resolve_default_enable_thinking(
            small,
            ModelVariant::Qwen359BGguf
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
            skip_special_tokens: true,
        };
        let mut rng = SimpleRng::new(7);
        let token = sample_next_token(&logits, 3, &config, PenaltyHistory::uniform(&[]), &mut rng)
            .expect("sample token");
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
            skip_special_tokens: true,
        };
        let history = PenaltyHistory::uniform(&[1u32]);
        let mut direct_rng = SimpleRng::new(17);
        let expected = sample_next_token(&output, 4, &config, history, &mut direct_rng).unwrap();
        let mut quantum_rng = SimpleRng::new(17);
        let mut unconsumed = Some(output);

        assert_eq!(
            take_quantum_sample(&mut unconsumed, 4, &config, history, &mut quantum_rng).unwrap(),
            expected
        );
        assert!(unconsumed.is_none());
        assert!(
            take_quantum_sample(&mut unconsumed, 4, &config, history, &mut quantum_rng).is_err()
        );
    }

    #[test]
    fn greedy_sampling_reads_the_index_and_its_logit_back_once() {
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
            skip_special_tokens: true,
        };
        let sample = |values: Vec<f32>, vocab: usize| {
            let logits = Tensor::from_vec(values.clone(), values.len(), &candle_core::Device::Cpu)
                .expect("logits");
            let before = thread_host_reads();
            let (token, path) = sample_next_token_traced(
                &logits,
                vocab,
                &config,
                PenaltyHistory::uniform(&[]),
                &mut SimpleRng::new(7),
            )
            .expect("sample");
            (token, path, thread_host_reads() - before)
        };

        // Finite winner: one packed readback. The vocabulary limit hides
        // the larger out-of-vocabulary logit.
        assert_eq!(
            sample(vec![0.5, 3.0, -1.0, 9.0], 3),
            (1, SamplerPath::Greedy, 1)
        );
        // A non-finite winner falls back to one full-row read.
        assert_eq!(
            sample(vec![0.5, f32::INFINITY, 2.0, -1.0], 4),
            (2, SamplerPath::Greedy, 2)
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
            skip_special_tokens: true,
        };
        let mut rng = SimpleRng::new(7);
        let result = sample_next_token(&logits, 0, &config, PenaltyHistory::uniform(&[]), &mut rng);
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
                skip_special_tokens: true,
            };
            let mut rng = SimpleRng::new(7);
            let error =
                sample_next_token(&logits, 3, &config, PenaltyHistory::uniform(&[]), &mut rng)
                    .expect_err("all in-vocabulary logits are non-finite");
            let message = error.to_string();
            assert!(message.contains("0 finite"));
            assert!(message.contains("1 NaN"));
            assert!(message.contains("1 +Inf"));
            assert!(message.contains("1 -Inf"));
            assert!(message.contains("3 in-vocabulary logits"));
        }
    }

    #[test]
    fn load_local_qwen35_assets_smoke_if_available() {
        let model_dir = local_model_dir("Qwen3.5-4B");
        if !model_dir.exists() {
            return;
        }

        let model =
            Qwen35ChatModel::load(&model_dir, ModelVariant::Qwen354BGguf, DeviceProfile::cpu())
                .expect("qwen3.5 assets should load");

        assert_eq!(model.variant(), ModelVariant::Qwen354BGguf);
        assert_eq!(model.text_config().architecture, "qwen35");
        assert_eq!(model.text_config().full_attention_interval, 4);
        assert_eq!(
            model.default_enable_thinking(),
            resolve_default_enable_thinking(model.chat_template(), ModelVariant::Qwen354BGguf)
        );
        assert_eq!(
            model.text_checkpoint().architecture.as_deref(),
            Some("qwen35")
        );
        assert!(model
            .projector_checkpoint()
            .path
            .ends_with("mmproj-F16.gguf"));
    }

    #[test]
    fn expand_image_placeholders_repeats_each_media_slot() {
        let prompt = concat!(
            "<|vision_start|><|image_pad|><|vision_end|>",
            " then ",
            "<|vision_start|><|image_pad|><|vision_end|>"
        );
        let expanded = expand_image_placeholders(prompt, &[4, 2]).expect("expand");
        assert_eq!(expanded.matches(IMAGE_PAD_PLACEHOLDER).count(), 6);
        assert!(expanded.contains(
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
        ));
        assert!(expanded.contains("then <|vision_start|><|image_pad|><|image_pad|><|vision_end|>"));
    }

    #[test]
    fn build_prompt_positions_matches_qwen_multimodal_rope_layout() {
        let image_pad_id = 248056;
        let prompt_ids = vec![
            11,
            image_pad_id,
            image_pad_id,
            image_pad_id,
            image_pad_id,
            12,
        ];
        let vision_inputs = PreparedVisionInputs {
            embeddings: Tensor::zeros((4, 8), DType::F32, &candle_core::Device::Cpu)
                .expect("embeddings"),
            grids: vec![[1, 4, 4]],
            token_counts: vec![4],
        };

        let (positions, next_text_position) =
            build_prompt_positions(&prompt_ids, &vision_inputs, image_pad_id, 248057, 2)
                .expect("positions");

        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3],
            ]
        );
        assert_eq!(next_text_position, 4);
    }

    /// A tiny synthetic Qwen3.5 text trunk: four layers (DeltaNet 0 and 2,
    /// gated full attention 1 and 3) with dense feed-forwards, written as a
    /// `qwen35` GGUF and loaded through the production loader, plus the
    /// synthetic byte-level tokenizer.
    fn tiny_exec(device: &candle_core::Device) -> Qwen35ChatExec {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};

        const HIDDEN: usize = 32;
        const FF: usize = 16;
        let mut seed = 0x5135_8A0E_u64;
        let mut tensors = Vec::<(String, QTensor)>::new();
        let mut push = |name: String, shape: Vec<usize>| {
            let values = (0..shape.iter().product::<usize>())
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2
                })
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, shape, &candle_core::Device::Cpu).unwrap();
            tensors.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
        };
        let tokenizer = synthetic_qwen35_tokenizer();
        push(
            "token_embd.weight".into(),
            vec![tokenizer.vocab_size, HIDDEN],
        );
        push("output_norm.weight".into(), vec![HIDDEN]);
        for layer in 0..4 {
            let prefix = format!("blk.{layer}");
            push(format!("{prefix}.attn_norm.weight"), vec![HIDDEN]);
            push(format!("{prefix}.post_attention_norm.weight"), vec![HIDDEN]);
            push(format!("{prefix}.ffn_gate.weight"), vec![FF, HIDDEN]);
            push(format!("{prefix}.ffn_up.weight"), vec![FF, HIDDEN]);
            push(format!("{prefix}.ffn_down.weight"), vec![HIDDEN, FF]);
            if layer % 2 == 1 {
                // Gated attention: q fuses the per-head output gate.
                push(format!("{prefix}.attn_q.weight"), vec![64, HIDDEN]);
                push(format!("{prefix}.attn_k.weight"), vec![16, HIDDEN]);
                push(format!("{prefix}.attn_v.weight"), vec![16, HIDDEN]);
                push(format!("{prefix}.attn_output.weight"), vec![HIDDEN, HIDDEN]);
                push(format!("{prefix}.attn_q_norm.weight"), vec![8]);
                push(format!("{prefix}.attn_k_norm.weight"), vec![8]);
            } else {
                // DeltaNet: 2 key heads of state 8 (q and k), 4 value heads.
                push(format!("{prefix}.attn_qkv.weight"), vec![64, HIDDEN]);
                push(format!("{prefix}.attn_gate.weight"), vec![32, HIDDEN]);
                push(format!("{prefix}.ssm_beta.weight"), vec![4, HIDDEN]);
                push(format!("{prefix}.ssm_alpha.weight"), vec![4, HIDDEN]);
                push(format!("{prefix}.ssm_dt.bias"), vec![4]);
                push(format!("{prefix}.ssm_a"), vec![4]);
                push(format!("{prefix}.ssm_conv1d.weight"), vec![64, 4]);
                push(format!("{prefix}.ssm_norm.weight"), vec![8]);
                push(format!("{prefix}.ssm_out.weight"), vec![HIDDEN, 32]);
            }
        }
        let value = |value: u64| gguf_file::Value::U64(value);
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("qwen35".into()),
            ),
            ("qwen35.block_count", value(4)),
            ("qwen35.context_length", value(256)),
            ("qwen35.embedding_length", value(HIDDEN as u64)),
            ("qwen35.feed_forward_length", value(FF as u64)),
            ("qwen35.attention.head_count", value(4)),
            ("qwen35.attention.head_count_kv", value(2)),
            ("qwen35.attention.key_length", value(8)),
            ("qwen35.attention.value_length", value(8)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(1e-5),
            ),
            (
                "qwen35.rope.dimension_sections",
                gguf_file::Value::Array(vec![value(2), value(2), value(2)]),
            ),
            ("qwen35.rope.dimension_count", value(8)),
            ("qwen35.rope.freq_base", gguf_file::Value::F64(10_000.0)),
            ("qwen35.ssm.conv_kernel", value(4)),
            ("qwen35.ssm.state_size", value(8)),
            ("qwen35.ssm.group_count", value(2)),
            ("qwen35.ssm.time_step_rank", value(4)),
            ("qwen35.ssm.inner_size", value(32)),
            ("qwen35.full_attention_interval", value(2)),
        ];
        let directory =
            std::env::temp_dir().join(format!("izwi-qwen35-replay-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("tiny-qwen35.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        gguf_file::write(
            &mut file,
            &metadata
                .iter()
                .map(|(name, value)| (*name, value))
                .collect::<Vec<_>>(),
            &tensors
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        drop(file);
        let backend = if device.is_metal() {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        };
        let loader = GgufLoader::from_path_with_backend(&path, backend).unwrap();
        let text_config = parse_text_config(&loader).unwrap();
        let text_model = Qwen35TextModel::load(&loader, &text_config, device).unwrap();
        drop(loader);
        std::fs::remove_dir_all(directory).unwrap();
        Qwen35ChatExec {
            variant: ModelVariant::Qwen3508BGguf,
            tokenizer,
            text_config,
            text_model,
        }
    }

    /// Run the pending token's forward as `decode_step` does and return the
    /// logits it leaves for sampling.
    fn pending_logits(exec: &Qwen35ChatExec, state: &mut ChatDecodeState) -> Vec<f32> {
        let pending = state.pending_token.take().expect("pending token");
        let logits = exec
            .text_model
            .forward_token_id_at_physical(
                pending,
                [state.next_text_position; 3],
                &mut state.text_state,
                &mut state.physical_kv,
            )
            .unwrap();
        state.next_text_position += 1;
        state.unconsumed_output = Some(logits.clone());
        logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// D3: a hybrid Qwen3.5 session (DeltaNet recurrence, convolution ring
    /// and full-attention KV) released for capacity mid-generation and
    /// rebuilt in a fresh cache and recurrent state by replaying its prompt
    /// and generated tokens over the original span boundaries resumes
    /// exactly: bit-identical next-step logits and the same greedy and
    /// seeded streams. A one-span replay runs other op shapes and is held to
    /// a tolerance.
    fn assert_replay_resumes_exactly(device: &candle_core::Device) {
        const MAX_NEW_TOKENS: usize = 24;
        const SUSPEND_AT: usize = 7;
        let exec = tiny_exec(device);
        let prompt_ids = vec![104u32, 101, 108, 108, 111, 33];
        let prepared = Qwen35PreparedPrompt {
            prompt_positions: (0..prompt_ids.len()).map(|index| [index; 3]).collect(),
            next_text_position: prompt_ids.len(),
            prompt_ids: prompt_ids.clone(),
            vision_inputs: None,
        };
        let profile = DeviceProfile {
            device: device.clone(),
            kind: if device.is_metal() {
                crate::backends::DeviceKind::Metal
            } else {
                crate::backends::DeviceKind::Cpu
            },
            capabilities: Default::default(),
            memory_pool: None,
        };
        let sampled = ChatGenerationConfig {
            temperature: 1.5,
            seed: 11,
            ..ChatGenerationConfig::default()
        };
        for config in [ChatGenerationConfig::default(), sampled] {
            for one_span in [false, true] {
                let mut caches = super::trace_tests::trace_caches(&exec.text_config, &profile, 3);
                let fresh = caches.pop().unwrap();
                let mut rows = caches
                    .into_iter()
                    .map(|cache| {
                        let mut state = exec
                            .begin_resumable_prefill_state_physical(
                                &prepared,
                                MAX_NEW_TOKENS,
                                &config,
                                cache,
                            )
                            .unwrap();
                        exec.continue_resumable_prefill_physical(
                            &mut state,
                            &prepared,
                            0,
                            prompt_ids.len(),
                        )
                        .unwrap();
                        state
                    })
                    .collect::<Vec<_>>();
                let mut suspended = rows.pop().unwrap();
                let mut baseline = rows.pop().unwrap();
                // After the prefill's own token the suspended row decodes
                // through the continuous batch, so both token-publishing
                // paths keep the replay journal.
                for step in 0..SUSPEND_AT {
                    let expected = exec.decode_step(&mut baseline).unwrap();
                    let actual = if step == 0 {
                        exec.decode_step(&mut suspended).unwrap()
                    } else {
                        exec.decode_step_batch(&mut [&mut suspended])
                            .unwrap()
                            .remove(0)
                    };
                    assert_eq!(actual.delta, expected.delta);
                }
                assert!(
                    !baseline.finished,
                    "the fixture ended before the suspension"
                );
                let checkpoint = suspended.replay_checkpoint().unwrap();
                assert_eq!(
                    checkpoint.replay_tokens(),
                    prompt_ids.len() + SUSPEND_AT - 1
                );
                drop(suspended);
                let mut replayed = exec
                    .begin_replay_state_physical(&checkpoint, fresh)
                    .unwrap();
                assert!(exec.decode_step(&mut replayed).is_err());
                let spans = if one_span {
                    vec![(0, checkpoint.replay_tokens())]
                } else {
                    std::iter::once((0, prompt_ids.len()))
                        .chain(
                            (prompt_ids.len()..checkpoint.replay_tokens())
                                .map(|start| (start, start + 1)),
                        )
                        .collect()
                };
                for (index, (start, end)) in spans.iter().copied().enumerate() {
                    let complete = exec
                        .continue_replay(&mut replayed, &prepared, start, end)
                        .unwrap();
                    assert_eq!(complete, index + 1 == spans.len());
                }
                assert_eq!(replayed.next_text_position, baseline.next_text_position);
                let expected = pending_logits(&exec, &mut baseline);
                let actual = pending_logits(&exec, &mut replayed);
                if one_span {
                    let error = expected
                        .iter()
                        .zip(&actual)
                        .map(|(left, right)| {
                            let error = (left - right).abs();
                            if error.is_finite() {
                                error
                            } else {
                                f32::INFINITY
                            }
                        })
                        .fold(0.0, f32::max);
                    assert!(error < 1e-3, "one-span replay logits differ by {error}");
                    continue;
                }
                assert_eq!(actual, expected, "rebuilt next-step logits");
                while !baseline.finished {
                    let step = exec.decode_step(&mut baseline).unwrap();
                    let resumed = exec.decode_step(&mut replayed).unwrap();
                    assert_eq!(resumed.delta, step.delta);
                    assert_eq!(resumed.text, step.text);
                }
                assert!(replayed.finished);
                assert_eq!(replayed.generated_ids, baseline.generated_ids);
            }
        }
    }

    #[test]
    fn replayed_session_resumes_exactly_on_cpu() {
        assert_replay_resumes_exactly(&candle_core::Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn replayed_session_resumes_exactly_on_metal() {
        let Some((device, _serial)) = crate::backends::metal_test_device_or_skip(
            "qwen35 replayed_session_resumes_exactly_on_metal",
        ) else {
            return;
        };
        assert_replay_resumes_exactly(&device);
    }
}

#[cfg(test)]
#[path = "chat_trace_tests.rs"]
mod trace_tests;
