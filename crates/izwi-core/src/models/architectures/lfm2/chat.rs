//! LFM2/LFM2.5 GGUF text-chat model loader and generation.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::info;

use super::diagnostics;
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::BackendKind;
use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};
use crate::models::shared::sampling::ChatSampler;
use crate::models::shared::telemetry::record_prefill_sequence_span;
use crate::models::shared::weights::gguf::GgufLoader;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::backbone::{Lfm2ShortConvRuntimeState, QuantizedLfm2Backbone};
use super::config::{parse_lfm2_backbone_config, Lfm2BackboneConfig};
use super::physical::lfm2_managed_cache_contract;

pub struct ChatDecodeState {
    shortconv: Lfm2ShortConvRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    greedy: bool,
    decoder: IncrementalDecoder,
    stop_reason: Option<&'static str>,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
    position: usize,
    prefill_progress: usize,
    /// DS9.3: logprob entries produced by the current decode step, drained
    /// by the registry right after the step. Cleared at each sample.
    pub(crate) pending_logprobs: Vec<crate::engine::TokenLogprob>,
}

pub(crate) struct Lfm2ChatDecodeCheckpoint {
    shortconv: Lfm2ShortConvRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    greedy: bool,
    decoder: IncrementalDecoder,
    stop_reason: Option<&'static str>,
    assembled: String,
    finished: bool,
    position: usize,
    prefill_progress: usize,
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub input_tokens_committed: usize,
    pub finished: bool,
}

impl ChatDecodeState {
    pub(crate) fn stop_reason(&self) -> Option<&'static str> {
        self.stop_reason
    }

    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<Lfm2ChatDecodeCheckpoint> {
        if cache.arena().id() != self.physical_kv.arena().id()
            || cache.context_len() != self.physical_kv.context_len()
        {
            return Err(Error::InferenceError(
                "LFM2 managed reservation does not continue the session".into(),
            ));
        }
        Ok(Lfm2ChatDecodeCheckpoint {
            shortconv: self.shortconv.clone(),
            physical_kv: std::mem::replace(&mut self.physical_kv, cache),
            unconsumed_output: self.unconsumed_output.clone(),
            pending_token: self.pending_token,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            greedy: self.greedy,
            decoder: self.decoder.clone(),
            stop_reason: self.stop_reason,
            assembled: self.assembled.clone(),
            finished: self.finished,
            position: self.position,
            prefill_progress: self.prefill_progress,
        })
    }

    pub(crate) fn install_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        if cache.arena().id() != self.physical_kv.arena().id()
            || cache.context_len() != self.physical_kv.context_len()
        {
            return Err(Error::InferenceError(
                "LFM2 managed reservation does not continue the session".into(),
            ));
        }
        self.physical_kv = cache;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(&mut self, checkpoint: Lfm2ChatDecodeCheckpoint) {
        self.shortconv = checkpoint.shortconv;
        self.physical_kv = checkpoint.physical_kv;
        self.unconsumed_output = checkpoint.unconsumed_output;
        self.pending_token = checkpoint.pending_token;
        self.generated_ids = checkpoint.generated_ids;
        self.sampler = checkpoint.sampler;
        self.greedy = checkpoint.greedy;
        self.decoder = checkpoint.decoder;
        self.stop_reason = checkpoint.stop_reason;
        self.assembled = checkpoint.assembled;
        self.finished = checkpoint.finished;
        self.position = checkpoint.position;
        self.prefill_progress = checkpoint.prefill_progress;
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .physical_tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(Error::InferenceError(
                "LFM2 tensor-state sequence identity changed".into(),
            ));
        }
        self.physical_tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self.physical_tensor_sequence.ok_or_else(|| {
            Error::InferenceError("LFM2 physical state has no tensor sequence".into())
        })?;
        self.shortconv.restore(arena, sequence)
    }

    pub(crate) fn stage_tensor_state(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        self.shortconv.stage(
            arena,
            PhysicalStateTransactionId::new(transaction)?,
            self.physical_kv.context_len() as u64,
        )
    }

    pub(crate) fn take_physical_write_completions(
        &mut self,
    ) -> Vec<std::sync::Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        self.physical_kv.take_completed_writes()
    }
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    bos: Option<u32>,
    im_start: u32,
    im_end: u32,
    eos: u32,
    eos_alt: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
    #[serde(default)]
    special: bool,
}

struct ChatTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    non_generatable: Vec<u32>,
}

struct PromptScaffoldTokens {
    /// The generation prompt's text after `<|im_start|>`.
    assistant_header: Vec<u32>,
    /// The separator after each turn's `<|im_end|>`.
    newline: Vec<u32>,
}

impl PromptScaffoldTokens {
    fn load(tokenizer: &ChatTokenizer) -> Result<Self> {
        Ok(Self {
            assistant_header: tokenizer.encode_text("assistant\n")?,
            newline: tokenizer.encode_text("\n")?,
        })
    }
}

fn template_role(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

impl ChatTokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

        for (id, token) in &config.added_tokens_decoder {
            let expected = id.parse::<u32>().map_err(|_| {
                Error::TokenizationError(format!("LFM2 invalid tokenizer metadata ID {id}"))
            })?;
            if inner.token_to_id(&token.content) != Some(expected) {
                return Err(Error::TokenizationError(format!(
                    "LFM2 tokenizer metadata disagrees with vocabulary for ID {expected}"
                )));
            }
        }

        let id_for = |token: &str| -> Option<u32> {
            config.added_tokens_decoder.iter().find_map(|(id, entry)| {
                if entry.content == token {
                    id.parse().ok()
                } else {
                    None
                }
            })
        };

        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let bos = config
            .bos_token
            .as_deref()
            .and_then(id_for)
            .or_else(|| id_for("<|startoftext|>"));
        let eos = config
            .eos_token
            .as_deref()
            .and_then(id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                bos,
                im_start,
                im_end,
                eos,
                eos_alt,
            },
            non_generatable: config
                .added_tokens_decoder
                .iter()
                .filter(|(_, token)| {
                    token.special
                        || matches!(
                            token.content.as_str(),
                            "<|pad|>" | "<|startoftext|>" | "<|im_start|>"
                        )
                })
                .filter_map(|(id, _)| id.parse::<u32>().ok())
                .filter(|id| *id != im_end && *id != eos && Some(*id) != eos_alt)
                .collect(),
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    fn validate_selection(&self, token: u32) -> Result<()> {
        if token as usize >= self.vocab_size || self.non_generatable.contains(&token) {
            return Err(Error::InferenceError(format!(
                "LFM2 invalid generated token {token}: out-of-vocabulary or non-generatable control token"
            )));
        }
        Ok(())
    }

    fn validate_logits(&self, logits: &Tensor) -> Result<()> {
        diagnostics::validate_finite(logits, logits_stage())?;
        self.validate_logits_width(logits)
    }

    fn validate_logits_width(&self, logits: &Tensor) -> Result<()> {
        if logits.dim(D::Minus1)? < self.vocab_size {
            return Err(Error::InferenceError(
                "LFM2 logits are smaller than tokenizer vocabulary".into(),
            ));
        }
        Ok(())
    }
}

pub struct Lfm2ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
    /// Additive logits mask: 0 for generatable ids, -inf for control tokens
    /// and ids past the tokenizer vocabulary. Applied before greedy argmax
    /// and before sampling, so neither path can select an id the decoder
    /// must reject.
    generation_mask: Tensor,
    prompt_scaffold: PromptScaffoldTokens,
    config: Lfm2BackboneConfig,
    text_model: QuantizedLfm2Backbone,
}

impl Lfm2ChatModel {
    pub fn max_context_tokens(&self) -> Result<usize> {
        if self.config.context_length == 0 {
            return Err(Error::ModelLoadError(
                "LFM2 checkpoint has a zero context length".into(),
            ));
        }
        Ok(self.config.context_length)
    }

    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_path = model_dir.join(lfm2_chat_gguf_name(variant)?);
        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }

        let tokenizer = ChatTokenizer::load(model_dir)?;
        let loader =
            GgufLoader::from_path_with_backend(&gguf_path, BackendKind::from(device.kind))?;
        let mut config = parse_lfm2_backbone_config(&loader)?;
        config.context_length = lfm25_chat_context_limit(config.context_length);
        let text_model = QuantizedLfm2Backbone::load(&loader, config.clone(), &device.device)?;
        if tokenizer.vocab_size == 0 || tokenizer.vocab_size > text_model.vocab_size() {
            return Err(Error::ModelLoadError(
                "LFM2 tokenizer vocabulary exceeds model embeddings".into(),
            ));
        }
        let prompt_scaffold = PromptScaffoldTokens::load(&tokenizer)?;
        let generation_mask = generation_mask(
            text_model.vocab_size(),
            tokenizer.vocab_size,
            &tokenizer.non_generatable,
            &device.device,
        )?;

        info!(
            "Loaded LFM2 GGUF chat model on {:?} from {}",
            device.kind,
            gguf_path.display()
        );
        Ok(Self {
            device,
            tokenizer,
            generation_mask,
            prompt_scaffold,
            config,
            text_model,
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    /// One decode row's live F32 activations: the token's per-layer
    /// activations plus its full-vocabulary logits row.
    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        let vocab = u64::try_from(self.text_model.vocab_size())
            .map_err(|_| Error::InvalidInput("LFM2 vocabulary exceeds u64".into()))?;
        self.prefill_workspace_per_token_bytes()?
            .checked_add(vocab.checked_mul(4).ok_or_else(workspace_overflow)?)
            .ok_or_else(workspace_overflow)
    }

    /// Live F32 activations of one prefill token in the widest block: the
    /// residual, its norm and the mixer output (3 x hidden), the ShortConv
    /// in-projection (3 x hidden), and the MLP gate, up and gated product
    /// (3 x ffn). Attention scratch is the KV arena's own workspace.
    pub(crate) fn prefill_workspace_per_token_bytes(&self) -> Result<u64> {
        let hidden = self.config.embedding_length as u64;
        let ffn = self
            .config
            .feed_forward_length
            .unwrap_or(self.config.embedding_length * 4) as u64;
        hidden
            .checked_mul(6)
            .and_then(|bytes| bytes.checked_add(ffn.checked_mul(3)?))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(workspace_overflow)
    }

    /// Prefill span: long prompts run in spans of this many tokens on every
    /// backend, bounding activation memory (a 32K prompt in one pass held
    /// ~4.6 GB), and an isolated request uses the whole span rather than the
    /// engine's 192-token chunk. CPU spans are shorter because CPU prefill is
    /// matrix-vector bound and long spans only add latency spikes.
    pub fn preferred_prefill_tokens(&self) -> usize {
        match self.device_kind() {
            BackendKind::Cpu => 512,
            BackendKind::Metal | BackendKind::Cuda => 1024,
        }
    }

    pub fn device_kind(&self) -> BackendKind {
        BackendKind::from(self.device.kind)
    }

    pub(crate) fn begin_resumable_prefill_state_managed(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if prompt_ids.is_empty() || cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "LFM2 managed prefill requires a non-empty prompt and empty cache".into(),
            ));
        }
        validate_generation_config(config)?;
        if config
            .stop_token_ids
            .iter()
            .any(|&id| id as usize >= self.tokenizer.vocab_size)
        {
            return Err(Error::InvalidInput(
                "LFM2 configured stop ID exceeds vocabulary".into(),
            ));
        }
        Ok(ChatDecodeState {
            shortconv: self.text_model.new_shortconv_state(),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: None,
            generated_ids: Vec::with_capacity(max_new_tokens.max(1)),
            sampler: ChatSampler::new(config.clone(), prompt_ids).with_json_object_constraint(
                std::sync::Arc::new(self.tokenizer.inner.clone()),
                vec![
                    Some(self.tokenizer.specials.im_end),
                    Some(self.tokenizer.specials.eos),
                    self.tokenizer.specials.eos_alt,
                ]
                .into_iter()
                .flatten()
                .collect(),
            ),
            // A JSON-constrained row must sample through the grammar, so it
            // never takes the batched argmax shortcut.
            greedy: is_plain_greedy(config) && !config.constrain_json_object,
            decoder: IncrementalDecoder::new(true),
            stop_reason: None,
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            position: 0,
            prefill_progress: 0,
            pending_logprobs: Vec::new(),
        })
    }

    pub(crate) fn continue_resumable_prefill_managed(
        &self,
        state: &mut ChatDecodeState,
        prompt_ids: &[u32],
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.prefill_progress != span_start
            || span_start >= span_end
            || span_end > prompt_ids.len()
            || state.physical_kv.context_len() != span_start
            || state.shortconv.cursor() != span_start as u64
            || state.unconsumed_output.is_some()
            || state.pending_token.is_some()
            || state.finished
        {
            return Err(Error::InvalidInput(
                "LFM2 resumable prefill span does not match its retained state".into(),
            ));
        }
        record_prefill_sequence_span(span_end - span_start);
        let input = Tensor::from_slice(
            &prompt_ids[span_start..span_end],
            (1, span_end - span_start),
            &self.device.device,
        )?;
        let complete = span_end == prompt_ids.len();
        let logits = self.text_model.forward_tokens_retained(
            &input,
            span_start,
            &mut state.physical_kv,
            &mut state.shortconv,
            complete,
        )?;
        state.prefill_progress = span_end;
        state.position = span_end;
        if complete {
            state.unconsumed_output = Some(logits.ok_or_else(|| {
                Error::InferenceError("LFM2 final prefill span produced no logits".into())
            })?);
        }
        Ok(complete)
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            let reason = *state.stop_reason.get_or_insert("length");
            let delta = self
                .tokenizer
                .inner
                .finish_incremental_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            validate_terminal_text(&state.assembled, reason, state.generated_ids.len())?;
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                input_tokens_committed: 0,
                finished: true,
            });
        }
        let mut committed = 0usize;
        if let Some(pending) = state.pending_token.take() {
            let input = Tensor::from_slice(&[pending], (1, 1), &self.device.device)?;
            state.unconsumed_output = self.text_model.forward_tokens_retained(
                &input,
                state.position,
                &mut state.physical_kv,
                &mut state.shortconv,
                true,
            )?;
            state.position = state.position.saturating_add(1);
            committed = 1;
        }
        let logits = state.unconsumed_output.take().ok_or_else(|| {
            Error::InferenceError("LFM2 decode state has no sampleable output".into())
        })?;
        self.tokenizer.validate_logits(&logits)?;
        let logits = self.mask_non_generatable(&logits)?;
        let (next, entry) = self.sample_token(state, &logits)?;
        self.accept_token(state, next, committed, entry)
    }

    /// Mask control tokens out of finite, validated logits (`[V]` or `[B, V]`).
    fn mask_non_generatable(&self, logits: &Tensor) -> Result<Tensor> {
        if logits.dim(D::Minus1)? != self.generation_mask.dim(0)? {
            return Err(Error::InferenceError(
                "LFM2 logits width does not match the generation mask".into(),
            ));
        }
        logits
            .broadcast_add(&self.generation_mask)
            .map_err(Error::from)
    }

    /// DS9.3: sample one token, resolving raw logprobs into the public
    /// payload when the request asked for them. `accept_token` records the
    /// entry only when the token is actually emitted.
    fn sample_token(
        &self,
        state: &mut ChatDecodeState,
        logits: &Tensor,
    ) -> Result<(u32, Option<crate::engine::TokenLogprob>)> {
        state.pending_logprobs.clear();
        if !state.sampler.wants_logprobs() {
            let token = state.sampler.sample(logits, self.tokenizer.vocab_size)?;
            return Ok((token, None));
        }
        let (token, raw) = state
            .sampler
            .sample_with_logprobs(logits, self.tokenizer.vocab_size)?;
        let entry = raw
            .map(|raw| {
                crate::models::shared::sampling::resolve_token_logprob(&self.tokenizer.inner, &raw)
            })
            .transpose()?;
        Ok((token, entry))
    }

    fn accept_token(
        &self,
        state: &mut ChatDecodeState,
        next: u32,
        committed: usize,
        logprob_entry: Option<crate::engine::TokenLogprob>,
    ) -> Result<ChatDecodeStep> {
        if diagnostics::enabled() && state.generated_ids.len() < 64 {
            tracing::info!(
                position = state.position,
                generated = state.generated_ids.len(),
                token_id = next,
                "LFM2 selected token (bounded diagnostic trace)"
            );
        }
        if next as usize >= self.tokenizer.vocab_size {
            return Err(Error::InferenceError(format!(
                "LFM2 out-of-vocabulary selected ID {next}"
            )));
        }
        let reason = if next == self.tokenizer.specials.im_end
            || next == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(next)
        {
            Some("eos")
        } else if state.sampler.is_configured_stop(next) {
            Some("configured_stop")
        } else {
            None
        };
        let mut delta = String::new();
        if reason.is_none() {
            self.tokenizer.validate_selection(next)?;
            delta = self
                .tokenizer
                .inner
                .decode_incrementally(&mut state.decoder, next)?;
            state.generated_ids.push(next);
            state.assembled.push_str(&delta);
            if let Some(entry) = logprob_entry {
                state.pending_logprobs.push(entry);
            }
        }
        state.stop_reason = reason.or_else(|| {
            if should_check_repetition_loop(state.generated_ids.len())
                && has_token_repetition_loop(&state.generated_ids)
            {
                Some("repetition")
            } else if state.generated_ids.len() >= state.max_new_tokens {
                Some("length")
            } else {
                None
            }
        });
        if let Some(reason) = state.stop_reason {
            let suffix = self
                .tokenizer
                .inner
                .finish_incremental_decode(&mut state.decoder)?;
            state.assembled.push_str(&suffix);
            delta.push_str(&suffix);
            state.finished = true;
            validate_terminal_text(&state.assembled, reason, state.generated_ids.len())?;
        } else {
            state.pending_token = Some(next);
        }
        Ok(ChatDecodeStep {
            delta,
            text: if state.finished {
                state.assembled.trim().to_string()
            } else {
                String::new()
            },
            tokens_generated: state.generated_ids.len(),
            input_tokens_committed: committed,
            finished: state.finished,
        })
    }

    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.finished
                || state.generated_ids.len() >= state.max_new_tokens
                || state.unconsumed_output.is_some()
                || state.pending_token.is_none()
                || state.physical_kv.context_len() != state.position
                || state.shortconv.cursor() != state.position as u64
            {
                return Err(Error::InvalidInput(
                    "continuous LFM2 batch contains a non-decodable state".into(),
                ));
            }
        }
        let mut tokens = Vec::with_capacity(states.len());
        let mut positions = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            tokens.push(state.pending_token.take().expect("pending token checked"));
            positions.push(state.position);
        }
        // The step's KV writes and ShortConv states are committed only after
        // the next tokens are read back: by then the device has drained, so
        // the whole step costs one synchronization.
        let (logits, pending) = {
            let shortconv = states.iter().map(|state| &state.shortconv).collect::<Vec<_>>();
            let caches = states.iter().map(|state| &state.physical_kv).collect::<Vec<_>>();
            self.text_model.forward_token_ids_retained_batch_deferred(
                &tokens,
                &positions,
                &shortconv,
                &caches,
            )?
        };
        let next_tokens = match self.select_batch_tokens(states, &logits) {
            Ok(next_tokens) => next_tokens,
            Err(error) => return Err(pending.abort_with(error)),
        };
        {
            let mut shortconv = Vec::with_capacity(states.len());
            let mut caches = Vec::with_capacity(states.len());
            for state in states.iter_mut() {
                let state = &mut **state;
                shortconv.push(&mut state.shortconv);
                caches.push(&mut state.physical_kv);
            }
            pending.commit(&mut shortconv, &mut caches)?;
        }
        let mut steps = Vec::with_capacity(states.len());
        for (state, (next, logprob_entry)) in states.iter_mut().zip(next_tokens) {
            state.position = state.position.saturating_add(1);
            steps.push(self.accept_token(state, next, 1, logprob_entry)?);
        }
        Ok(steps)
    }

    /// Next token per row from finite, width-checked logits. Plain greedy
    /// rows read their argmax and the finite flag back in one transfer;
    /// sampled rows sample first and then read the flag from an idle device,
    /// so either way the step synchronizes once. No token is accepted from
    /// non-finite logits.
    fn select_batch_tokens(
        &self,
        states: &mut [&mut ChatDecodeState],
        logits: &Tensor,
    ) -> Result<Vec<(u32, Option<crate::engine::TokenLogprob>)>> {
        self.tokenizer.validate_logits_width(logits)?;
        let finite = diagnostics::finite_flag(logits)?;
        let masked = self.mask_non_generatable(logits)?;
        // DS9.3: rows collecting logprobs must sample through the resolving
        // path, which forfeits the batched argmax for the whole envelope.
        let wants_logprobs = states.iter().any(|state| state.sampler.wants_logprobs());
        if states.iter().all(|state| state.greedy) && !wants_logprobs {
            let (ids, all_finite) = packed_greedy_selection(
                &masked.narrow(D::Minus1, 0, self.tokenizer.vocab_size)?,
                &finite,
            )?;
            if !all_finite {
                return Err(diagnostics::non_finite_error(logits, logits_stage()));
            }
            return Ok(ids.into_iter().map(|id| (id, None)).collect());
        }
        let mut sampled = Vec::with_capacity(states.len());
        for (row, state) in states.iter_mut().enumerate() {
            let row_logits = masked.i(row)?;
            sampled.push(self.sample_token(state, &row_logits)?);
        }
        if finite.to_vec1::<f32>()?[0] == 0.0 {
            return Err(diagnostics::non_finite_error(logits, logits_stage()));
        }
        Ok(sampled)
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.build_prompt(messages, &[])
    }

    /// Prompt ids with the request's tool definitions in the system turn.
    pub fn prompt_token_ids_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> Result<Vec<u32>> {
        self.build_prompt(messages, tools)
    }

    /// Render `messages` exactly like the checkpoint's chat template:
    ///
    /// * one BOS;
    /// * a leading system message becomes the system turn; tools append
    ///   `List of tools: [...]` to it (after a newline when it has text), each
    ///   tool serialized like Python's `json.dumps`; the turn is emitted only
    ///   when non-empty;
    /// * every other turn is `<|im_start|>{role}\n{content}<|im_end|>\n` with
    ///   its content verbatim (empty turns included), except that an
    ///   assistant turn before the last one keeps only the text after its
    ///   final `</think>`, trimmed;
    /// * the generation prompt `<|im_start|>assistant\n`.
    ///
    /// The text between two special tokens is tokenized as one string, as
    /// the template's rendering is, so a content's leading characters merge
    /// with its role header exactly as they would there.
    fn build_prompt(&self, messages: &[ChatMessage], tools: &[serde_json::Value]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        let (system, turns) = match messages.split_first() {
            Some((first, rest)) if matches!(first.role, ChatRole::System) => {
                (Some(first.content.as_str()), rest)
            }
            _ => (None, messages),
        };
        let mut system = system.unwrap_or_default().to_string();
        if !tools.is_empty() {
            if !system.is_empty() {
                system.push('\n');
            }
            system.push_str("List of tools: [");
            for (index, tool) in tools.iter().enumerate() {
                if index > 0 {
                    system.push_str(", ");
                }
                match tool {
                    serde_json::Value::String(text) => system.push_str(text),
                    other => write_python_json(other, &mut system),
                }
            }
            system.push(']');
        }
        if !system.is_empty() {
            self.append_prompt_turn(&mut ids, "system", &system)?;
        }

        let last_assistant_index = turns
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));
        for (index, message) in turns.iter().enumerate() {
            let content = if matches!(message.role, ChatRole::Assistant)
                && Some(index) != last_assistant_index
            {
                strip_past_assistant_thinking(&message.content)
            } else {
                Cow::Borrowed(message.content.as_str())
            };
            self.append_prompt_turn(&mut ids, template_role(&message.role), &content)?;
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend_from_slice(&self.prompt_scaffold.assistant_header);

        Ok(ids)
    }

    fn append_prompt_turn(&self, ids: &mut Vec<u32>, role: &str, content: &str) -> Result<()> {
        ids.push(self.tokenizer.specials.im_start);
        ids.extend(self.tokenizer.encode_text(&format!("{role}\n{content}"))?);
        ids.push(self.tokenizer.specials.im_end);
        ids.extend_from_slice(&self.prompt_scaffold.newline);
        Ok(())
    }

}

impl InferenceStateContractProvider for Lfm2ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        Ok(InferenceStateCapability::Managed(
            lfm2_managed_cache_contract(&self.config)?,
        ))
    }
}

fn validate_generation_config(config: &ChatGenerationConfig) -> Result<()> {
    if !config.temperature.is_finite()
        || config.temperature < 0.0
        || !config.top_p.is_finite()
        || !(0.0..=1.0).contains(&config.top_p)
        || config.top_p == 0.0
        || !config.repetition_penalty.is_finite()
        || config.repetition_penalty < 1.0
        || !config.presence_penalty.is_finite()
    {
        return Err(Error::InvalidInput(
            "LFM2 invalid sampling configuration (repetition penalty must be >= 1)".into(),
        ));
    }
    Ok(())
}

fn is_plain_greedy(config: &ChatGenerationConfig) -> bool {
    config.temperature <= 1e-5
        && config.top_k == 0
        && config.top_p >= 1.0
        && (config.repetition_penalty - 1.0).abs() <= f32::EPSILON
        && config.presence_penalty.abs() <= f32::EPSILON
}

/// Serialize like Python's `json.dumps(value, ensure_ascii=False)`, which the
/// chat template's `tojson` uses: `", "` and `": "` separators, non-ASCII
/// kept. Object keys follow `serde_json`'s map order.
fn write_python_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::Value::String(key.clone()).to_string());
                out.push_str(": ");
                write_python_json(item, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn lfm2_chat_gguf_name(variant: ModelVariant) -> Result<&'static str> {
    match variant {
        ModelVariant::Lfm2512BInstructGguf => Ok("LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
        ModelVariant::Lfm2512BThinkingGguf => Ok("LFM2.5-1.2B-Thinking-Q4_K_M.gguf"),
        _ => Err(Error::ModelLoadError(format!(
            "Unsupported LFM2 GGUF chat variant: {variant}"
        ))),
    }
}

/// What a chat load reads and builds, for load admission.
pub(crate) struct Lfm2ChatLoadLayout {
    pub(crate) gguf: std::path::PathBuf,
    /// (positions, head dim) of the backbone's RoPE tables.
    pub(crate) rope_table: (usize, usize),
}

pub(crate) fn lfm2_chat_load_layout(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<Lfm2ChatLoadLayout> {
    let gguf = model_dir.join(lfm2_chat_gguf_name(variant)?);
    let mut config = parse_lfm2_backbone_config(&GgufLoader::from_path(&gguf)?)?;
    config.context_length = lfm25_chat_context_limit(config.context_length);
    Ok(Lfm2ChatLoadLayout {
        rope_table: super::backbone::rope_table_geometry(&config)?,
        gguf,
    })
}

fn logits_stage() -> &'static str {
    if diagnostics::enabled() {
        "LFM2 raw logits before sampling (activation diagnostics enabled)"
    } else {
        "LFM2 raw logits before sampling (activation diagnostics disabled; set IZWI_LFM2_DIAGNOSTICS=1 in the server process)"
    }
}

fn workspace_overflow() -> Error {
    Error::InvalidInput("LFM2 workspace estimate overflow".into())
}

/// Context the LFM2.5-1.2B chat checkpoints support.
///
/// The GGUF metadata says 128,000, but Liquid's model cards and docs give
/// 32,768, and vLLM's recipe caps the model there; quality past it is
/// unverified. Capping at load also bounds the RoPE tables and every
/// backend's KV reservation (CUDA otherwise reserved a 128K arena).
const LFM25_CHAT_CONTEXT_TOKENS: usize = 32_768;

fn lfm25_chat_context_limit(checkpoint_context: usize) -> usize {
    checkpoint_context.min(LFM25_CHAT_CONTEXT_TOKENS)
}

fn generation_mask(
    logits_width: usize,
    tokenizer_vocab: usize,
    non_generatable: &[u32],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut mask = vec![0f32; logits_width];
    for value in mask.iter_mut().skip(tokenizer_vocab) {
        *value = f32::NEG_INFINITY;
    }
    for &id in non_generatable {
        if let Some(value) = mask.get_mut(id as usize) {
            *value = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask, logits_width, device).map_err(Error::from)
}

fn validate_terminal_text(text: &str, reason: &str, generated: usize) -> Result<()> {
    if text.trim().is_empty() {
        return Err(Error::InferenceError(format!(
            "LFM2 generation did not produce a valid answer: stop_reason={reason}, generated_tokens={generated}, raw_chars={}", text.chars().count()
        )));
    }
    Ok(())
}

/// Per-row argmax of `[B, V]` logits read back with a device-side `[1]`
/// finite flag in one transfer: `(tokens, all_finite)`. Ties pick the first
/// maximum. Token ids are below 2^24, so the F32 round trip is exact.
fn packed_greedy_selection(logits: &Tensor, finite: &Tensor) -> Result<(Vec<u32>, bool)> {
    let (rows, _) = logits.dims2()?;
    let ids = logits.argmax(D::Minus1)?.to_dtype(DType::F32)?;
    let packed = Tensor::cat(&[&ids, finite], 0)?.to_vec1::<f32>()?;
    if packed.len() != rows + 1 {
        return Err(Error::InferenceError(
            "LFM2 packed greedy readback has the wrong length".into(),
        ));
    }
    Ok((
        packed[..rows].iter().map(|id| *id as u32).collect(),
        packed[rows] != 0.0,
    ))
}

/// The template's `content.split("</think>")[-1] | trim` for past assistant
/// turns; content without `</think>` is kept verbatim.
fn strip_past_assistant_thinking(input: &str) -> Cow<'_, str> {
    if let Some((_reasoning, tail)) = input.rsplit_once("</think>") {
        Cow::Owned(tail.trim().to_string())
    } else {
        Cow::Borrowed(input)
    }
}

/// Tokens of exactly periodic output that count as a degenerate loop.
///
/// The guard exists because a looping reply (most often a reasoning model's
/// `<think>` block) would otherwise run to `max_tokens`, which defaults to the
/// remaining context. It must never cut legitimate repetitive output short,
/// so it fires only when the whole recent window repeats one short span.
const REPETITION_WINDOW: usize = 256;
/// Longest repeating span the guard looks for.
const REPETITION_MAX_PERIOD: usize = 64;
const REPETITION_CHECK_INTERVAL: usize = 16;

fn should_check_repetition_loop(len: usize) -> bool {
    len >= REPETITION_WINDOW && len.is_multiple_of(REPETITION_CHECK_INTERVAL)
}

/// True when the last [`REPETITION_WINDOW`] ids repeat with a period of at
/// most [`REPETITION_MAX_PERIOD`].
fn has_token_repetition_loop(ids: &[u32]) -> bool {
    if ids.len() < REPETITION_WINDOW {
        return false;
    }
    let window = &ids[ids.len() - REPETITION_WINDOW..];
    (1..=REPETITION_MAX_PERIOD)
        .any(|period| (period..window.len()).all(|i| window[i] == window[i - period]))
}

#[cfg(test)]
mod tests {
    use super::{
        has_token_repetition_loop, should_check_repetition_loop, strip_past_assistant_thinking,
    };

    #[test]
    fn lfm2_packed_greedy_tokens_pick_the_first_maximum_per_row() {
        use candle_core::{Device, Tensor};
        let logits = Tensor::from_vec(
            vec![1f32, 4., 2., 4., -3., -2., -1., -4., 9., 2., 1., 0.],
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let finite = super::diagnostics::finite_flag(&logits).unwrap();
        assert_eq!(
            super::packed_greedy_selection(&logits, &finite).unwrap(),
            (vec![1, 2, 0], true)
        );
        assert_eq!(
            super::packed_greedy_selection(&logits.narrow(0, 1, 1).unwrap(), &finite).unwrap(),
            (vec![2], true)
        );
        let bad = Tensor::from_vec(vec![1f32, f32::NAN], (1, 2), &Device::Cpu).unwrap();
        let bad_finite = super::diagnostics::finite_flag(&bad).unwrap();
        assert!(!super::packed_greedy_selection(&bad, &bad_finite).unwrap().1);
    }

    #[test]
    fn tool_json_uses_python_json_dumps_spacing() {
        let tool = serde_json::json!({
            "function": {"name": "get_weather", "parameters": {"required": ["city"], "x": [1, 2.5, true, null]}},
            "type": "function",
            "u": "é\n\"q\""
        });
        let mut out = String::new();
        super::write_python_json(&tool, &mut out);
        assert_eq!(
            out,
            r#"{"function": {"name": "get_weather", "parameters": {"required": ["city"], "x": [1, 2.5, true, null]}}, "type": "function", "u": "é\n\"q\""}"#
        );
    }

    #[test]
    fn lfm25_chat_context_is_capped_at_the_model_card_length() {
        assert_eq!(super::lfm25_chat_context_limit(128_000), 32_768);
        assert_eq!(super::lfm25_chat_context_limit(32), 32);
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_only_tail_after_close_tag() {
        let input = "<think>reasoning</think>\nFinal answer";
        assert_eq!(strip_past_assistant_thinking(input), "Final answer");
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_unclosed_content_verbatim() {
        let input = "  <think>still reasoning\n";
        assert_eq!(strip_past_assistant_thinking(input), input);
    }

    #[test]
    fn detects_a_window_of_periodic_output() {
        let phrase: Vec<u32> = (1..=40).collect();
        let mut ids = vec![42; 16];
        while ids.len() < 16 + 256 {
            ids.extend(phrase.iter().copied());
        }
        assert!(has_token_repetition_loop(&ids));
    }

    #[test]
    fn legitimate_repetitive_lists_are_not_loops() {
        // "list 40 zeros": a period-2 run of 80 tokens after a preamble.
        let mut ids: Vec<u32> = (100..300).collect();
        for _ in 0..40 {
            ids.extend([7_u32, 8]);
        }
        assert!(!has_token_repetition_loop(&ids));
        // A table whose rows share a 6-token separator but differ in content.
        let mut table = Vec::new();
        for row in 0..60_u32 {
            table.extend([1, 2, 3, 1000 + row, 4, 5]);
        }
        assert!(!has_token_repetition_loop(&table));
        // A period longer than the guard looks for.
        let long: Vec<u32> = (0..300).map(|i| i % 65).collect();
        assert!(!has_token_repetition_loop(&long));
    }

    #[test]
    fn repetition_loop_check_interval_skips_between_boundaries() {
        assert!(!should_check_repetition_loop(255));
        assert!(should_check_repetition_loop(256));
        assert!(!should_check_repetition_loop(257));
        assert!(should_check_repetition_loop(272));
    }
}

#[cfg(test)]
#[path = "chat_integrity_tests.rs"]
mod integrity_tests;

#[cfg(test)]
#[path = "chat_reference_tests.rs"]
mod reference_tests;

#[cfg(test)]
#[path = "chat_bench_tests.rs"]
mod bench_tests;
