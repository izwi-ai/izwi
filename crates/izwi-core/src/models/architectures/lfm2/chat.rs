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
    system_header: Vec<u32>,
    user_header: Vec<u32>,
    assistant_header: Vec<u32>,
    newline: Vec<u32>,
}

impl PromptScaffoldTokens {
    fn load(tokenizer: &ChatTokenizer) -> Result<Self> {
        Ok(Self {
            system_header: tokenizer.encode_text("system\n")?,
            user_header: tokenizer.encode_text("user\n")?,
            assistant_header: tokenizer.encode_text("assistant\n")?,
            newline: tokenizer.encode_text("\n")?,
        })
    }

    fn role_header(&self, role: &ChatRole) -> &[u32] {
        match role {
            ChatRole::System => &self.system_header,
            ChatRole::User => &self.user_header,
            ChatRole::Assistant => &self.assistant_header,
        }
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
        let stage = if diagnostics::enabled() {
            "LFM2 raw logits before sampling (activation diagnostics enabled)"
        } else {
            "LFM2 raw logits before sampling (activation diagnostics disabled; set IZWI_LFM2_DIAGNOSTICS=1 in the server process)"
        };
        diagnostics::validate_finite(logits, stage)?;
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
        let gguf_name = match variant {
            ModelVariant::Lfm2512BInstructGguf => "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            ModelVariant::Lfm2512BThinkingGguf => "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
            _ => {
                return Err(Error::ModelLoadError(format!(
                    "Unsupported LFM2 GGUF chat variant: {variant}"
                )));
            }
        };
        let gguf_path = model_dir.join(gguf_name);
        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }

        let tokenizer = ChatTokenizer::load(model_dir)?;
        let loader =
            GgufLoader::from_path_with_backend(&gguf_path, BackendKind::from(device.kind))?;
        let config = parse_lfm2_backbone_config(&loader)?;
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

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        u64::try_from(self.config.embedding_length)
            .ok()
            .and_then(|hidden| hidden.checked_mul(4))
            .ok_or_else(|| Error::InvalidInput("LFM2 decode workspace overflow".into()))
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
        let mut shortconv = Vec::with_capacity(states.len());
        let mut caches = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            tokens.push(state.pending_token.take().expect("pending token checked"));
            positions.push(state.position);
            shortconv.push(&mut state.shortconv);
            caches.push(&mut state.physical_kv);
        }
        let logits = self.text_model.forward_token_ids_retained_batch(
            &tokens,
            &positions,
            &mut shortconv,
            &mut caches,
        )?;
        drop(shortconv);
        drop(caches);
        self.tokenizer.validate_logits(&logits)?;
        let logits = self.mask_non_generatable(&logits)?;
        // DS9.3: rows collecting logprobs must sample through the resolving
        // path, which forfeits the batched argmax for the whole envelope.
        let wants_logprobs = states.iter().any(|state| state.sampler.wants_logprobs());
        let next_tokens = if states.iter().all(|state| state.greedy) && !wants_logprobs {
            argmax_batch(&logits.narrow(D::Minus1, 0, self.tokenizer.vocab_size)?)?
                .into_iter()
                .map(|token| (token, None))
                .collect::<Vec<_>>()
        } else {
            let mut sampled = Vec::with_capacity(states.len());
            for (row, state) in states.iter_mut().enumerate() {
                let row_logits = logits.i(row)?;
                sampled.push(self.sample_token(state, &row_logits)?);
            }
            sampled
        };
        let mut steps = Vec::with_capacity(states.len());
        for (state, (next, logprob_entry)) in states.iter_mut().zip(next_tokens) {
            state.position = state.position.saturating_add(1);
            steps.push(self.accept_token(state, next, 1, logprob_entry)?);
        }
        Ok(steps)
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.build_prompt(messages)
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        let last_assistant_index = messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));

        for (prompt_index, message) in messages.iter().enumerate() {
            self.append_prompt_message(
                &mut ids,
                prompt_index,
                &message.role,
                message.content.as_str(),
                last_assistant_index,
            )?;
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend_from_slice(&self.prompt_scaffold.assistant_header);

        Ok(ids)
    }

    fn append_prompt_message(
        &self,
        ids: &mut Vec<u32>,
        prompt_index: usize,
        role: &ChatRole,
        content: &str,
        last_assistant_index: Option<usize>,
    ) -> Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let normalized =
            if matches!(role, ChatRole::Assistant) && Some(prompt_index) != last_assistant_index {
                strip_past_assistant_thinking(trimmed)
            } else {
                Cow::Borrowed(trimmed)
            };
        if normalized.is_empty() {
            return Ok(());
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend_from_slice(self.prompt_scaffold.role_header(role));
        ids.extend(self.tokenizer.encode_text(normalized.as_ref())?);
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

fn argmax_batch(logits: &Tensor) -> Result<Vec<u32>> {
    logits.dims2()?;
    logits
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()
        .map_err(Error::from)
}

fn strip_past_assistant_thinking(input: &str) -> Cow<'_, str> {
    if let Some((_reasoning, tail)) = input.rsplit_once("</think>") {
        Cow::Owned(tail.trim().to_string())
    } else {
        Cow::Borrowed(input.trim())
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
        assert_eq!(super::argmax_batch(&logits).unwrap(), vec![1, 2, 0]);
        assert_eq!(
            super::argmax_batch(&logits.narrow(0, 1, 1).unwrap()).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_only_tail_after_close_tag() {
        let input = "<think>reasoning</think>\nFinal answer";
        assert_eq!(strip_past_assistant_thinking(input), "Final answer");
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_unclosed_content() {
        let input = "<think>still reasoning";
        assert_eq!(
            strip_past_assistant_thinking(input),
            "<think>still reasoning"
        );
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
