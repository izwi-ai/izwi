//! Inkling 2 as a served chat model: prompt, sampling and reply text around
//! the trunk, on engine-owned state.
//!
//! A sequence's keys and values live in the engine's paged cache and its
//! convolution histories in the engine's tensor-state arena (`state`). The
//! trunk runs a span at a time: a prompt span during prefill, one token per
//! decode step. Each span is one transaction: the cache commit and the
//! convolution advance happen together after the trunk succeeds.

use std::path::Path;
use std::sync::Arc;

use candle_core::Tensor;
use tracing::info;

use super::checkpoint::Inkling2Checkpoint;
use super::config::Inkling2Config;
use super::prompt::{forced_call, render_prompt};
use super::reply::{Inkling2Reply, Inkling2ReplyTokens};
use super::state::{
    kv_dtype, managed_contract, paged_geometry, Inkling2ConvRuntimeState, Inkling2PagedContext,
};
use super::trunk::Inkling2Trunk;
use crate::backends::kv::KvWriteBatchCompletion;
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{replay_journal_span, ChatGenerationConfig, ChatMessage};
use crate::models::shared::sampling::{resolve_token_logprob, ChatSampler};
use crate::tokenizer::Tokenizer;

/// Context a sequence may reach when the checkpoint allows more. Every held
/// token costs keys and values in all 48 layer applications, so the
/// published 1M-token limit is not served.
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 32_768;

fn chat_error(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("Inkling 2 chat: {message}"))
}

pub struct ChatDecodeState {
    conv: Inkling2ConvRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    /// Logits of the last prefilled token, `[vocab]`, not sampled yet.
    unconsumed_logits: Option<Tensor>,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    reply: Inkling2Reply,
    max_new_tokens: usize,
    finished: bool,
    /// Tokens the trunk has consumed.
    position: usize,
    /// Scheduler-visible prompt cursor.
    prefill_progress: usize,
    /// Cache length a capacity replay is rebuilding.
    replay_end: Option<usize>,
    /// Logprob entries of the current decode step, drained by the registry.
    pub(crate) pending_logprobs: Vec<crate::engine::TokenLogprob>,
}

pub(crate) struct ChatDecodeCheckpoint {
    conv: Inkling2ConvRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    unconsumed_logits: Option<Tensor>,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    reply: Inkling2Reply,
    finished: bool,
    position: usize,
    prefill_progress: usize,
}

/// CPU continuation record of a session released for capacity. It owns no
/// tensors; the released state is rebuilt by replaying the prompt and every
/// generated token but the pending one.
#[derive(Clone)]
pub(crate) struct ChatReplayCheckpoint {
    replay_tokens: usize,
    pending_token: u32,
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    reply: Inkling2Reply,
    max_new_tokens: usize,
}

impl ChatReplayCheckpoint {
    pub(crate) fn replay_tokens(&self) -> usize {
        self.replay_tokens
    }
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    /// The whole reply; only a terminal step carries it.
    pub text: String,
    pub tokens_generated: usize,
    pub input_tokens_committed: usize,
    pub finished: bool,
}

impl ChatDecodeState {
    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn replay_tokens(&self) -> Option<usize> {
        self.replay_end
    }

    pub(crate) fn replay_checkpoint(&self) -> Result<ChatReplayCheckpoint> {
        let replay_tokens = self.replay_end.unwrap_or(self.position);
        let pending_token = self.pending_token.filter(|_| {
            !self.finished
                && self.unconsumed_logits.is_none()
                && replay_tokens + 1 == self.prefill_progress + self.generated_ids.len()
        });
        let Some(pending_token) = pending_token else {
            return Err(Error::InvalidInput(
                "an Inkling 2 session suspends only between decode quanta".into(),
            ));
        };
        Ok(ChatReplayCheckpoint {
            replay_tokens,
            pending_token,
            prefill_progress: self.prefill_progress,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            reply: self.reply.clone(),
            max_new_tokens: self.max_new_tokens,
        })
    }

    fn continues(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.arena().id() != self.physical_kv.arena().id()
            || cache.context_len() != self.physical_kv.context_len()
        {
            return Err(chat_error(
                "a managed reservation does not continue the session",
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeCheckpoint> {
        self.continues(&cache)?;
        Ok(ChatDecodeCheckpoint {
            conv: self.conv.clone(),
            physical_kv: std::mem::replace(&mut self.physical_kv, cache),
            unconsumed_logits: self.unconsumed_logits.clone(),
            pending_token: self.pending_token,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            reply: self.reply.clone(),
            finished: self.finished,
            position: self.position,
            prefill_progress: self.prefill_progress,
        })
    }

    pub(crate) fn install_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        self.continues(&cache)?;
        self.physical_kv = cache;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(&mut self, checkpoint: ChatDecodeCheckpoint) {
        self.conv = checkpoint.conv;
        self.physical_kv = checkpoint.physical_kv;
        self.unconsumed_logits = checkpoint.unconsumed_logits;
        self.pending_token = checkpoint.pending_token;
        self.generated_ids = checkpoint.generated_ids;
        self.sampler = checkpoint.sampler;
        self.reply = checkpoint.reply;
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
            return Err(chat_error("the tensor-state sequence identity changed"));
        }
        self.physical_tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self
            .physical_tensor_sequence
            .ok_or_else(|| chat_error("the physical state has no tensor sequence"))?;
        self.conv.restore(arena, sequence)
    }

    pub(crate) fn stage_tensor_state(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        self.conv.stage(
            arena,
            PhysicalStateTransactionId::new(transaction)?,
            self.physical_kv.context_len() as u64,
        )
    }

    pub(crate) fn take_physical_write_completions(&mut self) -> Vec<Arc<KvWriteBatchCompletion>> {
        self.physical_kv.take_completed_writes()
    }

    #[cfg(test)]
    pub(crate) fn unconsumed_logits_for_test(&self) -> Option<&Tensor> {
        self.unconsumed_logits.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn generated_ids_for_test(&self) -> &[u32] {
        &self.generated_ids
    }

    #[cfg(test)]
    pub(crate) fn finished_for_test(&self) -> bool {
        self.finished
    }

    /// The reservation the engine would hand back for the next quantum.
    #[cfg(test)]
    pub(crate) fn cache_view_for_test(&self) -> PhysicalPagedKvCache {
        let layers = self.physical_kv.arena().config().layers.len() as u32;
        PhysicalPagedKvCache::new(
            self.physical_kv.arena().clone(),
            (0..layers)
                .map(|layer| crate::kv::KvLayerBinding {
                    model_layer: layer,
                    physical_layer: layer,
                })
                .collect(),
            self.physical_kv.blocks.clone(),
            self.physical_kv.context_len(),
        )
        .unwrap()
    }
}

pub struct Inkling2ChatModel {
    variant: ModelVariant,
    device: DeviceProfile,
    config: Inkling2Config,
    tokenizer: Tokenizer,
    reply_tokens: Inkling2ReplyTokens,
    trunk: Inkling2Trunk,
    max_context_tokens: usize,
}

impl InferenceStateContractProvider for Inkling2ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        Ok(InferenceStateCapability::Managed(managed_contract(
            &self.config,
            kv_dtype(self.device_kind()),
        )?))
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
            "Inkling 2 invalid sampling configuration (repetition penalty must be >= 1)".into(),
        ));
    }
    Ok(())
}

impl Inkling2ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let config = Inkling2Config::load(model_dir)?;
        let tokenizer = Tokenizer::from_path_requiring_tokenizer_json(model_dir)?;
        let checkpoint = Inkling2Checkpoint::open(model_dir, &config)?;
        Self::from_parts(variant, device, config, tokenizer, &checkpoint)
    }

    pub(crate) fn from_parts(
        variant: ModelVariant,
        device: DeviceProfile,
        config: Inkling2Config,
        tokenizer: Tokenizer,
        checkpoint: &Inkling2Checkpoint,
    ) -> Result<Self> {
        let reply_tokens = Inkling2ReplyTokens::load(&tokenizer)?;
        if reply_tokens.end_sampling != config.eos_token_id {
            return Err(Error::ModelLoadError(format!(
                "Inkling 2: the tokenizer's end-of-sampling token is {}, the config's {}",
                reply_tokens.end_sampling, config.eos_token_id
            )));
        }
        if tokenizer.vocab_size() > config.unpadded_vocab {
            return Err(Error::ModelLoadError(
                "Inkling 2: the tokenizer vocabulary exceeds the unembedding".into(),
            ));
        }
        let trunk = Inkling2Trunk::load(checkpoint, &config, &device.device)?;
        let max_context_tokens = config.max_context_tokens.min(DEFAULT_MAX_CONTEXT_TOKENS);
        info!(
            "Loaded Inkling 2 chat model on {:?} from {} ({} layer applications, {:.1} GiB of packed experts, {} context tokens)",
            device.kind,
            checkpoint.dir().display(),
            config.application_count(),
            trunk.expert_resident_bytes() as f64 / (1u64 << 30) as f64,
            max_context_tokens
        );
        Ok(Self {
            variant,
            device,
            config,
            tokenizer,
            reply_tokens,
            trunk,
            max_context_tokens,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    /// A model whose replies never end on a token, so a test on random
    /// weights generates as many tokens as it asks for.
    #[cfg(test)]
    pub(crate) fn without_turn_end_for_test(mut self) -> Self {
        self.reply_tokens.end_sampling = u32::MAX;
        self.reply_tokens.other_roles = [u32::MAX; 3];
        self
    }

    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    pub fn device_kind(&self) -> BackendKind {
        BackendKind::from(self.device.kind)
    }

    pub fn max_context_tokens(&self) -> Result<usize> {
        Ok(self.max_context_tokens)
    }

    pub fn runtime_device_kind(&self) -> String {
        format!("{:?}", self.device.kind).to_ascii_lowercase()
    }

    /// Activations are F32 on every backend (the composed trunk).
    pub fn runtime_compute_dtype(&self) -> Option<String> {
        Some("f32".to_string())
    }

    pub fn runtime_diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "family": "inkling2_chat",
            "trunk": "composed",
            "experts": "packed, host",
            "layer_applications": self.config.application_count(),
            "expert_resident_bytes": self.trunk.expert_resident_bytes(),
            "max_context_tokens": self.max_context_tokens,
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    /// Rows of a continuous batch run one after another inside the step.
    pub fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    /// Prefill span. Long spans mostly add latency: the experts run per
    /// token on the host.
    pub fn preferred_prefill_tokens(&self) -> usize {
        match self.device_kind() {
            BackendKind::Cpu => 128,
            BackendKind::Metal | BackendKind::Cuda => 256,
        }
    }

    /// Live F32 activations of one prefill token: the depth-residual stack
    /// and layer temporaries (about 16 hidden vectors), the query, key and
    /// value projections, and one expert's gate and up rows.
    pub(crate) fn prefill_workspace_per_token_bytes(&self) -> u64 {
        let hidden = self.config.hidden as u64;
        let attention = (self.config.full.query_width() + 2 * self.config.full.kv_width()) as u64;
        (16 * hidden + attention + 3 * self.config.expert_width as u64) * 4
    }

    /// One decode row: a token's activations and its logits row.
    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        Ok(self.prefill_workspace_per_token_bytes() + self.config.unpadded_vocab as u64 * 4)
    }

    pub fn prompt_token_ids(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Vec<u32>> {
        if !config.request.media_inputs.is_empty() {
            return Err(Error::InvalidInput(
                "Inkling 2 image input is not available yet".to_string(),
            ));
        }
        render_prompt(messages, &config.request)?.encode(&self.tokenizer)
    }

    /// Runs `tokens` at positions `start..` and commits their keys, values
    /// and convolution state. Returns the last token's logits (`[vocab]`)
    /// when asked.
    fn forward_span(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut PhysicalPagedKvCache,
        conv: &mut Inkling2ConvRuntimeState,
        want_logits: bool,
    ) -> Result<Option<Tensor>> {
        if tokens.is_empty() || cache.context_len() != start || conv.cursor() != start as u64 {
            return Err(chat_error(format!(
                "a span at {start} does not continue the state (cache {}, convolution {})",
                cache.context_len(),
                conv.cursor()
            )));
        }
        cache.validate_sparse_model_layers(&paged_geometry(&self.config))?;
        let embeddings = self.trunk.embed(tokens)?;
        let mut working = conv.conv()?.clone();
        let mut prepared = cache.prepare_append(start, tokens.len())?;
        let hidden = {
            let mut context = Inkling2PagedContext {
                cache: &*cache,
                prepared: &mut prepared,
            };
            self.trunk
                .forward_span(&embeddings, start, &mut working, &mut context, None)
        };
        let hidden = match hidden {
            Ok(hidden) => hidden,
            Err(error) => {
                cache.abort_prepared(prepared)?;
                return Err(error);
            }
        };
        cache.commit_prepared(prepared)?;
        conv.advance(working, tokens.len());
        if !want_logits {
            return Ok(None);
        }
        let last = hidden.narrow(0, tokens.len() - 1, 1)?;
        Ok(Some(self.trunk.logits(&last)?.squeeze(0)?))
    }

    fn new_state(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        let forced = forced_call(&config.request)?;
        Ok(ChatDecodeState {
            conv: Inkling2ConvRuntimeState::new(self.trunk.new_conv_state()?),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_logits: None,
            pending_token: None,
            generated_ids: Vec::new(),
            sampler: ChatSampler::new(config.clone(), prompt_ids),
            reply: Inkling2Reply::new(
                self.reply_tokens.clone(),
                forced.as_ref().map(|(_, json)| json.as_str()),
            ),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            position: 0,
            prefill_progress: 0,
            replay_end: None,
            pending_logprobs: Vec::new(),
        })
    }

    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        // Prefix sharing is off for this family: a state starts empty.
        if prompt_ids.is_empty() || cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Inkling 2 managed prefill requires a non-empty prompt and an empty reservation"
                    .into(),
            ));
        }
        validate_generation_config(config)?;
        if config
            .stop_token_ids
            .iter()
            .any(|id| *id as usize >= self.config.unpadded_vocab)
        {
            return Err(Error::InvalidInput(
                "Inkling 2 configured stop ID exceeds vocabulary".into(),
            ));
        }
        self.new_state(prompt_ids, max_new_tokens, config, cache)
    }

    pub(crate) fn continue_resumable_prefill(
        &self,
        state: &mut ChatDecodeState,
        prompt_ids: &[u32],
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.prefill_progress != span_start
            || span_start >= span_end
            || span_end > prompt_ids.len()
            || state.position != span_start
            || state.unconsumed_logits.is_some()
            || state.pending_token.is_some()
            || !state.generated_ids.is_empty()
            || state.finished
        {
            return Err(Error::InvalidInput(format!(
                "Inkling 2 resumable prefill span [{span_start},{span_end}) does not match cursor {}",
                state.prefill_progress
            )));
        }
        let complete = span_end == prompt_ids.len();
        let logits = self.forward_span(
            &prompt_ids[span_start..span_end],
            span_start,
            &mut state.physical_kv,
            &mut state.conv,
            complete,
        )?;
        state.prefill_progress = span_end;
        state.position = span_end;
        if complete {
            state.unconsumed_logits = logits;
        }
        Ok(complete)
    }

    /// Starts a session and prefills its whole prompt.
    pub fn start_decode_managed(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        let mut state =
            self.begin_resumable_prefill_managed(prompt_ids, max_new_tokens, config, cache)?;
        self.continue_resumable_prefill(&mut state, prompt_ids, 0, prompt_ids.len())?;
        Ok(state)
    }

    /// Starts rebuilding a suspended session in an empty reservation.
    pub(crate) fn begin_replay_managed(
        &self,
        saved: &ChatReplayCheckpoint,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if cache.context_len() != 0 || saved.replay_tokens == 0 {
            return Err(Error::InvalidInput(
                "Inkling 2 replay requires an empty reservation and a non-empty journal".into(),
            ));
        }
        Ok(ChatDecodeState {
            conv: Inkling2ConvRuntimeState::new(self.trunk.new_conv_state()?),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_logits: None,
            pending_token: Some(saved.pending_token),
            generated_ids: saved.generated_ids.clone(),
            sampler: saved.sampler.clone(),
            reply: saved.reply.clone(),
            max_new_tokens: saved.max_new_tokens,
            finished: false,
            position: 0,
            prefill_progress: saved.prefill_progress,
            replay_end: Some(saved.replay_tokens),
            pending_logprobs: Vec::new(),
        })
    }

    /// Writes `[span_start, span_end)` of the replay journal (the prompt,
    /// then the generated tokens) without sampling. Returns whether the
    /// replay is complete.
    pub(crate) fn continue_replay(
        &self,
        state: &mut ChatDecodeState,
        prompt_ids: &[u32],
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        let replay_end = state
            .replay_end
            .ok_or_else(|| Error::InvalidInput("Inkling 2 state has no pending replay".into()))?;
        if span_start != state.position
            || span_start >= span_end
            || span_end > replay_end
            || prompt_ids.len() != state.prefill_progress
        {
            return Err(Error::InvalidInput(format!(
                "Inkling 2 replay span [{span_start},{span_end}) does not continue cursor {} of {replay_end}",
                state.position
            )));
        }
        let ids = replay_journal_span(prompt_ids, &state.generated_ids, span_start, span_end)?;
        self.forward_span(
            &ids,
            span_start,
            &mut state.physical_kv,
            &mut state.conv,
            false,
        )?;
        state.position = span_end;
        let complete = span_end == replay_end;
        if complete {
            state.replay_end = None;
        }
        Ok(complete)
    }

    fn finish(
        &self,
        state: &mut ChatDecodeState,
        mut delta: String,
        committed: usize,
    ) -> Result<ChatDecodeStep> {
        state.finished = true;
        delta.push_str(&state.reply.finish(&self.tokenizer)?);
        Ok(ChatDecodeStep {
            delta,
            text: state.reply.text().trim().to_string(),
            tokens_generated: state.generated_ids.len(),
            input_tokens_committed: committed,
            finished: true,
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.replay_end.is_some() {
            return Err(chat_error("decode cannot run before the replay completes"));
        }
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            return self.finish(state, String::new(), 0);
        }
        let mut committed = 0usize;
        if let Some(token) = state.pending_token.take() {
            state.unconsumed_logits = self.forward_span(
                &[token],
                state.position,
                &mut state.physical_kv,
                &mut state.conv,
                true,
            )?;
            state.position += 1;
            committed = 1;
        }
        let logits = state
            .unconsumed_logits
            .take()
            .ok_or_else(|| chat_error("the decode state has no logits to sample"))?;
        state.pending_logprobs.clear();
        let vocab = self.config.unpadded_vocab;
        let (next, entry) = if state.sampler.wants_logprobs() {
            let (token, raw) = state.sampler.sample_with_logprobs(&logits, vocab)?;
            let entry = raw
                .map(|raw| resolve_token_logprob(&self.tokenizer, &raw))
                .transpose()?;
            (token, entry)
        } else {
            (state.sampler.sample(&logits, vocab)?, None)
        };
        if next as usize >= vocab {
            return Err(chat_error(format!(
                "sampled token {next} is outside the vocabulary"
            )));
        }
        if state.sampler.is_configured_stop(next) {
            return self.finish(state, String::new(), committed);
        }
        // Every sampled token advances the reply and, unless it ends the
        // turn, the model's own state on the next step, shown or not.
        let delta = state.reply.push(&self.tokenizer, next)?;
        if state.reply.finished() {
            return self.finish(state, delta, committed);
        }
        state.generated_ids.push(next);
        state.pending_token = Some(next);
        if let Some(entry) = entry {
            state.pending_logprobs.push(entry);
        }
        if state.generated_ids.len() >= state.max_new_tokens {
            return self.finish(state, delta, committed);
        }
        Ok(ChatDecodeStep {
            delta,
            text: String::new(),
            tokens_generated: state.generated_ids.len(),
            input_tokens_committed: committed,
            finished: false,
        })
    }

    /// One decode step of every row. Rows run one after another; a row's
    /// result does not depend on the others.
    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        if states.iter().any(|state| {
            state.finished
                || state.generated_ids.len() >= state.max_new_tokens
                || state.unconsumed_logits.is_some()
                || state.pending_token.is_none()
                || state.replay_end.is_some()
        }) {
            return Err(Error::InvalidInput(
                "Inkling 2 continuous batch contains a non-decodable state".into(),
            ));
        }
        states
            .iter_mut()
            .map(|state| self.decode_step(state))
            .collect()
    }
}
