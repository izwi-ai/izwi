//! Native Qwen3 text-chat model loader and generation.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Tensor};
#[cfg(test)]
use candle_core::{IndexOp, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use crate::backends::kv::KvWriteBatchCompletion;
use crate::backends::BackendKind;
use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::kv::v2::StateDomainId;
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{Qwen3Config, Qwen3Model};
use crate::models::shared::attention::mixed::{MixedRowLayout, MixedRowSpan};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{
    replay_journal_span, ChatGenerationConfig, ChatMessage, ChatRole,
};
use crate::models::shared::config::checkpoint_dtype_from_config_json;
use crate::models::shared::sampling::{sample_chat_rows, ChatSampler};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

pub struct ChatDecodeState {
    cache: PhysicalPagedKvCache,
    /// Model output that has not yet been sampled by the current executor
    /// quantum. The first decode step consumes prefill output in the same
    /// invocation; later quanta compute and consume their output locally.
    unconsumed_output: Option<Tensor>,
    pos: usize,
    /// Token already emitted to the caller but not yet appended to KV. The
    /// next decode quantum consumes exactly this token, keeping physical KV
    /// progress aligned with the scheduler's input range.
    pending_token: Option<u32>,
    /// Scheduler-visible prompt cursor. This stays logical even when a final
    /// first span reuses an engine-owned physical prefix.
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    decoder: IncrementalDecoder,
    max_new_tokens: usize,
    finished: bool,
    /// Cache length a capacity replay is rebuilding; the state decodes again
    /// once its cache reaches it.
    replay_end: Option<usize>,
    /// DS9.3: logprob entries produced by the current decode step, drained
    /// by the registry right after the step. Cleared at each step entry.
    pub(crate) pending_logprobs: Vec<crate::engine::TokenLogprob>,
}

/// CPU continuation record of a session released for capacity: the
/// generated-token journal (the prompt is the request's own) and the sampler
/// and text state at a quantum boundary. It owns no tensors; the released
/// cache is rebuilt by replaying the prompt and every generated token but the
/// pending one.
#[derive(Clone)]
pub(crate) struct ChatReplayCheckpoint {
    replay_tokens: usize,
    pending_token: u32,
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    decoder: IncrementalDecoder,
    max_new_tokens: usize,
}

impl ChatReplayCheckpoint {
    pub(crate) fn replay_tokens(&self) -> usize {
        self.replay_tokens
    }
}

pub(crate) struct ChatDecodeCheckpoint {
    cache: PhysicalPagedKvCache,
    unconsumed_output: Option<Tensor>,
    pos: usize,
    pending_token: Option<u32>,
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    decoder: IncrementalDecoder,
    finished: bool,
}

impl ChatDecodeState {
    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub fn uses_managed_kv(&self) -> bool {
        true
    }

    /// The replay a capacity suspension still has to rebuild, if any.
    pub(crate) fn replay_tokens(&self) -> Option<usize> {
        self.replay_end
    }

    /// Capture the CPU continuation record at a quantum boundary, before the
    /// caller releases the cache.
    pub(crate) fn replay_checkpoint(&self) -> Result<ChatReplayCheckpoint> {
        let replay_tokens = self.replay_end.unwrap_or(self.pos);
        let pending_token = self.pending_token.filter(|_| {
            !self.finished
                && self.unconsumed_output.is_none()
                && replay_tokens + 1 == self.prefill_progress + self.generated_ids.len()
        });
        let Some(pending_token) = pending_token else {
            return Err(Error::InvalidInput(
                "a Qwen3 session suspends only between decode quanta".into(),
            ));
        };
        Ok(ChatReplayCheckpoint {
            replay_tokens,
            pending_token,
            prefill_progress: self.prefill_progress,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            decoder: self.decoder.clone(),
            max_new_tokens: self.max_new_tokens,
        })
    }

    pub(crate) fn take_managed_write_completions(&mut self) -> Vec<Arc<KvWriteBatchCompletion>> {
        self.cache.take_completed_writes()
    }

    pub(crate) fn install_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        let checkpoint = self.begin_managed_quantum(cache)?;
        drop(checkpoint);
        Ok(())
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeCheckpoint> {
        let current = &self.cache;
        if current.arena().id() != cache.arena().id()
            || current.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Qwen3 session cannot switch managed KV authority".to_string(),
            ));
        }
        if cache.context_len() != self.pos {
            return Err(Error::InferenceError(format!(
                "managed Qwen3 reservation starts at {}, but decode state is at {}",
                cache.context_len(),
                self.pos
            )));
        }
        let checkpoint = ChatDecodeCheckpoint {
            cache: std::mem::replace(&mut self.cache, cache),
            unconsumed_output: self.unconsumed_output.clone(),
            pos: self.pos,
            pending_token: self.pending_token,
            prefill_progress: self.prefill_progress,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            decoder: self.decoder.clone(),
            finished: self.finished,
        };
        Ok(checkpoint)
    }

    pub(crate) fn rollback_managed_quantum(&mut self, checkpoint: ChatDecodeCheckpoint) {
        self.cache = checkpoint.cache;
        self.unconsumed_output = checkpoint.unconsumed_output;
        self.pos = checkpoint.pos;
        self.pending_token = checkpoint.pending_token;
        self.prefill_progress = checkpoint.prefill_progress;
        self.generated_ids = checkpoint.generated_ids;
        self.sampler = checkpoint.sampler;
        self.decoder = checkpoint.decoder;
        self.finished = checkpoint.finished;
    }
}

/// One prefill row of a mixed step: the prompt span `[span_start,
/// span_end)` that continues `state`'s resumable prefill.
pub(crate) struct MixedPrefillRow<'s, 'p> {
    pub(crate) state: &'s mut ChatDecodeState,
    pub(crate) prompt: &'p [u32],
    pub(crate) span_start: usize,
    pub(crate) span_end: usize,
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
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
    eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct ChatTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl ChatTokenizer {
    fn load(model_dir: &Path, expected_vocab_size: Option<usize>) -> Result<Self> {
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, expected_vocab_size)?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

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
                im_start,
                im_end,
                eos,
                eos_alt,
            },
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    /// The text `id` adds to the generated sequence. Ids past the vocabulary
    /// add nothing, as a whole-sequence decode skips them.
    fn decode_next(&self, decoder: &mut IncrementalDecoder, id: u32) -> Result<String> {
        if id as usize >= self.vocab_size {
            return Ok(String::new());
        }
        self.inner.decode_incrementally(decoder, id)
    }
}

pub struct Qwen3ChatModel {
    device: DeviceProfile,
    compute_dtype: DType,
    tokenizer: ChatTokenizer,
    text_model: Qwen3Model,
    /// DS1.6 catalog verdict for committed prefix reuse on the load backend.
    prefix_reuse: bool,
}

impl InferenceStateContractProvider for Qwen3ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        if !self.text_model.supports_managed_kv_execution() {
            return Err(Error::ModelLoadError(
                "native Qwen3 attention geometry is unsupported by the physical runtime"
                    .to_string(),
            ));
        }
        Ok(InferenceStateCapability::Managed(
            self.text_model.managed_inference_state_contract(
                StateDomainId::new(1),
                self.compute_dtype,
                default_kv_page_size(),
                self.prefix_reuse,
            )?,
        ))
    }
}

impl Qwen3ChatModel {
    pub fn max_context_tokens(&self) -> Result<usize> {
        self.text_model.context_length().ok_or_else(|| {
            Error::ModelLoadError(
                "Qwen3 checkpoint is missing a positive max_position_embeddings value".into(),
            )
        })
    }

    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device: DeviceProfile,
        prefix_reuse: bool,
    ) -> Result<Self> {
        if variant.is_qwen_chat_gguf() {
            return Self::load_gguf(model_dir, variant, device, prefix_reuse);
        }
        Self::load_safetensors(model_dir, device, prefix_reuse)
    }

    fn load_safetensors(
        model_dir: &Path,
        device: DeviceProfile,
        prefix_reuse: bool,
    ) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config = parse_qwen3_config(&config_str)?;
        let checkpoint_dtype = checkpoint_dtype_from_config_json(&config_str);

        let tokenizer = ChatTokenizer::load(model_dir, Some(config.vocab_size))?;
        let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype = select_qwen3_dense_dtype(&device, dtype_override.as_deref(), checkpoint_dtype)?;

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(&index_path)?;
            let index: Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> =
                shard_files.iter().map(|f| model_dir.join(f)).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)? }
        } else {
            let weights_path = model_dir.join("model.safetensors");
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? }
        };

        let text_model = Qwen3Model::load(config, vb)?;

        info!(
            "Loaded Qwen3 chat model on {:?} with dtype {:?}",
            device.kind, dtype
        );

        Ok(Self {
            device,
            compute_dtype: dtype,
            tokenizer,
            text_model,
            prefix_reuse,
        })
    }

    fn load_gguf(
        model_dir: &Path,
        variant: ModelVariant,
        device: DeviceProfile,
        prefix_reuse: bool,
    ) -> Result<Self> {
        let gguf_name = match variant {
            ModelVariant::Qwen314BGguf => "Qwen3-14B-Q4_K_M.gguf",
            ModelVariant::Qwen3Moe30bA3bGguf => "Qwen3-30B-A3B-Q4_K_M.gguf",
            _ => {
                return Err(Error::ModelLoadError(format!(
                    "Unsupported GGUF chat variant: {variant}"
                )))
            }
        };
        let gguf_path = model_dir.join(gguf_name);
        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }

        let loader =
            GgufLoader::from_path_with_backend(&gguf_path, BackendKind::from(device.kind))?;
        let config = parse_qwen3_gguf_config(&loader)?;
        let tokenizer = ChatTokenizer::load(model_dir, Some(config.vocab_size))?;
        let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype = select_qwen3_dense_dtype(&device, dtype_override.as_deref(), None)?;
        let text_model = Qwen3Model::load_gguf_text(config, &loader, &device.device, dtype, "")?;

        info!(
            "Loaded native physical Qwen3 GGUF chat model on {:?} with dtype {:?} from {}",
            device.kind,
            dtype,
            gguf_path.display()
        );

        Ok(Self {
            device,
            compute_dtype: dtype,
            tokenizer,
            text_model,
            prefix_reuse,
        })
    }

    pub fn runtime_device_kind(&self) -> String {
        format!("{:?}", self.device.kind).to_ascii_lowercase()
    }

    pub fn runtime_compute_dtype(&self) -> Option<String> {
        Some(format!("{:?}", self.compute_dtype).to_ascii_lowercase())
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        let mut no_op = |_delta: &str| {};
        self.generate_with_callback(messages, max_new_tokens, &mut no_op)
    }

    pub fn generate_with_callback(
        &self,
        _messages: &[ChatMessage],
        _max_new_tokens: usize,
        _on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        Err(Error::InvalidInput(
            "Qwen3 generation requires scheduler-owned physical state".to_string(),
        ))
    }

    /// Start a native safetensors session directly in authoritative physical
    /// KV pages supplied by the engine, prefilling the request's sealed prompt
    /// (the tokens a capacity replay writes back). The prompt is written into
    /// the arena during prefill; no model-owned prompt history survives this
    /// boundary.
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

    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if prompt_ids.is_empty() || cache.context_len() >= prompt_ids.len() {
            return Err(Error::InvalidInput(
                "Qwen3 resumable prefill requires at least one private prompt token".into(),
            ));
        }
        // DS1.5: a managed prefix attach starts the logical cursor at the
        // attached physical cursor.
        let pos = cache.context_len();
        Ok(ChatDecodeState {
            cache,
            unconsumed_output: None,
            pos,
            pending_token: None,
            prefill_progress: pos,
            generated_ids: Vec::new(),
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
            decoder: IncrementalDecoder::new(config.skip_special_tokens),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            replay_end: None,
            pending_logprobs: Vec::new(),
        })
    }

    /// Start rebuilding a suspended session in an empty reservation. The
    /// state decodes again after [`Self::continue_replay`] reaches the end
    /// of the checkpoint's replay.
    pub(crate) fn begin_replay_managed(
        &self,
        saved: &ChatReplayCheckpoint,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if cache.context_len() != 0 || saved.replay_tokens == 0 {
            return Err(Error::InvalidInput(
                "Qwen3 replay requires an empty reservation and a non-empty journal".into(),
            ));
        }
        Ok(ChatDecodeState {
            cache,
            unconsumed_output: None,
            pos: 0,
            pending_token: Some(saved.pending_token),
            prefill_progress: saved.prefill_progress,
            generated_ids: saved.generated_ids.clone(),
            sampler: saved.sampler.clone(),
            decoder: saved.decoder.clone(),
            max_new_tokens: saved.max_new_tokens,
            finished: false,
            replay_end: Some(saved.replay_tokens),
            pending_logprobs: Vec::new(),
        })
    }

    /// Write `[span_start, span_end)` of the replay journal (the prompt, then
    /// the generated tokens) into the cache without sampling or emitting.
    /// Returns whether the replay is complete.
    pub(crate) fn continue_replay(
        &self,
        state: &mut ChatDecodeState,
        prompt_ids: &[u32],
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        let replay_end = state
            .replay_end
            .ok_or_else(|| Error::InvalidInput("Qwen3 state has no pending replay".into()))?;
        if span_start != state.pos
            || span_start != state.cache.context_len()
            || span_start >= span_end
            || span_end > replay_end
            || prompt_ids.len() != state.prefill_progress
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3 replay span [{span_start},{span_end}) does not continue cursor {} of {replay_end}",
                state.pos
            )));
        }
        let ids = replay_journal_span(prompt_ids, &state.generated_ids, span_start, span_end)?;
        let input_ids = Tensor::from_vec(ids, (1, span_end - span_start), &self.device.device)?;
        let embeds = self.text_model.embeddings(&input_ids)?;
        self.text_model.forward_managed_prefill_only_with_embeds(
            &embeds,
            span_start,
            &mut state.cache,
            None,
        )?;
        state.pos = span_end;
        let complete = span_end == replay_end;
        if complete {
            state.replay_end = None;
        }
        Ok(complete)
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
            || state.finished
            || state.unconsumed_output.is_some()
            || state.pending_token.is_some()
            || !state.generated_ids.is_empty()
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3 resumable prefill span [{span_start},{span_end}) is incompatible with cursor {} and prompt length {}",
                state.prefill_progress,
                prompt_ids.len()
            )));
        }
        let physical_start = state.cache.context_len();
        let first_span = span_start == 0;
        if state.pos != physical_start
            || (!first_span && physical_start != span_start)
            || (first_span && physical_start >= span_end)
        {
            return Err(Error::InferenceError(format!(
                "Qwen3 resumable prefill physical cursor {physical_start} is incompatible with logical span [{span_start},{span_end})"
            )));
        }
        let input_ids = Tensor::from_slice(
            &prompt_ids[physical_start..span_end],
            (1, span_end - physical_start),
            &self.device.device,
        )?;
        let complete = span_end == prompt_ids.len();
        let output = if complete {
            Some(
                self.text_model
                    .forward_managed(&input_ids, physical_start, &mut state.cache)?,
            )
        } else {
            let embeds = self.text_model.embeddings(&input_ids)?;
            self.text_model.forward_managed_hidden_with_embeds(
                &embeds,
                physical_start,
                &mut state.cache,
                None,
            )?;
            None
        };
        if state.cache.context_len() != span_end {
            return Err(Error::InferenceError(format!(
                "Qwen3 resumable prefill committed physical cursor {} instead of {span_end}",
                state.cache.context_len()
            )));
        }
        state.pos = span_end;
        state.prefill_progress = span_end;
        state.unconsumed_output = output;
        Ok(complete)
    }

    /// DS9.3: sample one token, resolving raw logprobs into the public
    /// payload when the request asked for them. The caller records the
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

    fn record_logprob(state: &mut ChatDecodeState, entry: Option<crate::engine::TokenLogprob>) {
        if let Some(entry) = entry {
            state.pending_logprobs.push(entry);
        }
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        let text_model = &self.text_model;

        if state.replay_end.is_some() {
            return Err(Error::InferenceError(
                "Qwen3 decode cannot run before its replay completes".into(),
            ));
        }
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            return self.finish_decode(state, String::new());
        }

        if let Some(pending) = state.pending_token.take() {
            let next_tensor = Tensor::from_vec(vec![pending], (1, 1), &self.device.device)?;
            state.unconsumed_output =
                Some(text_model.forward_managed(&next_tensor, state.pos, &mut state.cache)?);
            state.pos += 1;
        }

        let output = state.unconsumed_output.take().ok_or_else(|| {
            Error::InferenceError("Qwen3 decode quantum has no unconsumed model output".into())
        })?;
        let (next, logprob_entry) = self.sample_token(state, &output)?;
        self.accept_token(state, next, logprob_entry)
    }

    /// Decode one token for a changing set of independent chat sessions in a
    /// single model forward. Session-owned caches remain separate so rows may
    /// enter or leave at every scheduler quantum boundary.
    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        let text_model = &self.text_model;
        if states.is_empty() {
            return Ok(Vec::new());
        }
        if states
            .iter()
            .any(|state| state.unconsumed_output.is_some() || state.replay_end.is_some())
        {
            return Err(Error::InferenceError(
                "continuous Qwen3 decode crossed a quantum with unconsumed model output or replay"
                    .to_string(),
            ));
        }
        if states
            .iter()
            .any(|state| state.finished || state.generated_ids.len() >= state.max_new_tokens)
        {
            return Err(Error::InvalidInput(
                "Continuous chat batch contains a terminal decode state".to_string(),
            ));
        }

        let pending_tokens = states
            .iter_mut()
            .map(|state| {
                state.pending_token.take().ok_or_else(|| {
                    Error::InferenceError(
                        "continuous Qwen3 decode state has no scheduled input token".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let positions = states.iter().map(|state| state.pos).collect::<Vec<_>>();
        let input_ids = Tensor::from_vec(pending_tokens, (states.len(), 1), &self.device.device)?;
        let mut caches = states
            .iter_mut()
            .map(|state| &mut state.cache)
            .collect::<Vec<_>>();
        let next_logits =
            text_model.forward_managed_decode_batch(&input_ids, &positions, &mut caches)?;
        for state in states.iter_mut() {
            state.pos = state.pos.saturating_add(1);
            state.pending_logprobs.clear();
        }

        // Every row samples from one readback of the whole step.
        let next_logits = next_logits.reshape((states.len(), ()))?;
        let mut samplers = states
            .iter_mut()
            .map(|state| &mut state.sampler)
            .collect::<Vec<_>>();
        let sampled = sample_chat_rows(&mut samplers, &next_logits, self.tokenizer.vocab_size)?;
        states
            .iter_mut()
            .zip(sampled)
            .map(|(state, (next, raw))| {
                let entry = raw
                    .map(|raw| {
                        crate::models::shared::sampling::resolve_token_logprob(
                            &self.tokenizer.inner,
                            &raw,
                        )
                    })
                    .transpose()?;
                self.accept_token(state, next, entry)
            })
            .collect()
    }

    /// One mixed step: `decode` rows take one token each and every
    /// `prefill` row continues its resumable prefill over `[span_start,
    /// span_end)` of its prompt, in one model forward and one sampling
    /// readback. Returns each decode row's step, and for each prefill row its
    /// bootstrap token when the span ends the prompt. A step of only decode
    /// rows, or of one prefill row, is exactly the split path's call.
    pub(crate) fn step_mixed(
        &self,
        decode: &mut [&mut ChatDecodeState],
        prefill: &mut [MixedPrefillRow<'_, '_>],
    ) -> Result<(Vec<ChatDecodeStep>, Vec<Option<ChatDecodeStep>>)> {
        match (decode.is_empty(), &mut *prefill) {
            (_, []) => return Ok((self.decode_step_batch(decode)?, Vec::new())),
            (true, [row]) => {
                let complete = self.continue_resumable_prefill(
                    &mut *row.state,
                    row.prompt,
                    row.span_start,
                    row.span_end,
                )?;
                let bootstrap = complete
                    .then(|| self.decode_step(&mut *row.state))
                    .transpose()?;
                return Ok((Vec::new(), vec![bootstrap]));
            }
            _ => {}
        }
        if decode.iter().any(|state| {
            state.unconsumed_output.is_some()
                || state.pending_token.is_none()
                || state.finished
                || state.generated_ids.len() >= state.max_new_tokens
        }) {
            return Err(Error::InvalidInput(
                "Qwen3 mixed step contains a non-decodable decode row".to_string(),
            ));
        }
        let mut prefill_spans = Vec::with_capacity(prefill.len());
        let mut prefill_ids = Vec::new();
        for row in prefill.iter() {
            let state = &*row.state;
            let physical_start = state.cache.context_len();
            if state.prefill_progress != row.span_start
                || row.span_start >= row.span_end
                || row.span_end > row.prompt.len()
                || state.finished
                || state.unconsumed_output.is_some()
                || state.pending_token.is_some()
                || !state.generated_ids.is_empty()
                || state.pos != physical_start
                || physical_start >= row.span_end
                || (row.span_start != 0 && physical_start != row.span_start)
            {
                return Err(Error::InvalidInput(format!(
                    "Qwen3 mixed prefill span [{},{}) is incompatible with cursor {} and prompt length {}",
                    row.span_start,
                    row.span_end,
                    state.prefill_progress,
                    row.prompt.len()
                )));
            }
            prefill_spans.push(MixedRowSpan {
                start_pos: physical_start,
                q_len: row.span_end - physical_start,
            });
            prefill_ids.extend_from_slice(&row.prompt[physical_start..row.span_end]);
        }
        let mut spans = Vec::with_capacity(decode.len() + prefill.len());
        let mut input_ids = Vec::with_capacity(decode.len() + prefill_ids.len());
        for state in decode.iter_mut() {
            spans.push(MixedRowSpan {
                start_pos: state.pos,
                q_len: 1,
            });
            input_ids.push(state.pending_token.take().expect("pending token checked"));
        }
        spans.extend(prefill_spans);
        input_ids.extend(prefill_ids);
        let completes = prefill
            .iter()
            .map(|row| row.span_end == row.prompt.len())
            .collect::<Vec<_>>();
        let sample_rows = (0..decode.len())
            .chain(
                completes
                    .iter()
                    .enumerate()
                    .filter(|(_, complete)| **complete)
                    .map(|(row, _)| decode.len() + row),
            )
            .collect::<Vec<_>>();
        let layout = MixedRowLayout::new(spans, decode.len())?;
        let logits = {
            let mut caches = decode
                .iter_mut()
                .map(|state| &mut state.cache)
                .chain(prefill.iter_mut().map(|row| &mut row.state.cache))
                .collect::<Vec<_>>();
            self.text_model
                .forward_managed_mixed(&input_ids, layout, &mut caches, &sample_rows)?
        };
        for state in decode.iter_mut() {
            state.pos += 1;
            state.pending_logprobs.clear();
        }
        for row in prefill.iter_mut() {
            row.state.pos = row.span_end;
            row.state.prefill_progress = row.span_end;
            row.state.pending_logprobs.clear();
        }

        let Some(logits) = logits else {
            return Ok((Vec::new(), prefill.iter().map(|_| None).collect()));
        };
        // Decode rows and final spans sample from one readback.
        let sampled = {
            let mut samplers = decode
                .iter_mut()
                .map(|state| &mut state.sampler)
                .chain(
                    prefill
                        .iter_mut()
                        .zip(&completes)
                        .filter(|(_, complete)| **complete)
                        .map(|(row, _)| &mut row.state.sampler),
                )
                .collect::<Vec<_>>();
            sample_chat_rows(&mut samplers, &logits, self.tokenizer.vocab_size)?
        };
        let mut sampled = sampled.into_iter();
        let mut accept = |state: &mut ChatDecodeState| {
            let (next, raw) = sampled
                .next()
                .ok_or_else(|| Error::InferenceError("Qwen3 mixed step lost a row".into()))?;
            let entry = raw
                .map(|raw| {
                    crate::models::shared::sampling::resolve_token_logprob(
                        &self.tokenizer.inner,
                        &raw,
                    )
                })
                .transpose()?;
            self.accept_token(state, next, entry)
        };
        let decode_steps = decode
            .iter_mut()
            .map(|state| accept(state))
            .collect::<Result<Vec<_>>>()?;
        let prefill_steps = prefill
            .iter_mut()
            .zip(completes)
            .map(|(row, complete)| complete.then(|| accept(&mut *row.state)).transpose())
            .collect::<Result<Vec<_>>>()?;
        Ok((decode_steps, prefill_steps))
    }

    /// Emit `next`, or end the row on a stop token. Text is decoded
    /// incrementally, so a step costs the same however long the reply is.
    fn accept_token(
        &self,
        state: &mut ChatDecodeState,
        next: u32,
        logprob_entry: Option<crate::engine::TokenLogprob>,
    ) -> Result<ChatDecodeStep> {
        if next == self.tokenizer.specials.im_end
            || next == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(next)
            || state.sampler.is_configured_stop(next)
        {
            return self.finish_decode(state, String::new());
        }

        Self::record_logprob(state, logprob_entry);
        state.generated_ids.push(next);
        state.pending_token = Some(next);
        let delta = self.tokenizer.decode_next(&mut state.decoder, next)?;
        if state.generated_ids.len() >= state.max_new_tokens {
            return self.finish_decode(state, delta);
        }
        Ok(ChatDecodeStep {
            delta,
            text: String::new(),
            tokens_generated: state.generated_ids.len(),
            finished: false,
        })
    }

    /// Finish the row: flush text the decoder still holds (an incomplete
    /// UTF-8 sequence decodes as a whole-sequence decode renders it) and
    /// return the whole text, which only a terminal step carries.
    fn finish_decode(
        &self,
        state: &mut ChatDecodeState,
        mut delta: String,
    ) -> Result<ChatDecodeStep> {
        state.finished = true;
        let suffix = self
            .tokenizer
            .inner
            .finish_incremental_decode(&mut state.decoder)?;
        delta.push_str(&suffix);
        Ok(ChatDecodeStep {
            delta,
            text: state.decoder.decoded_text().trim().to_string(),
            tokens_generated: state.generated_ids.len(),
            finished: true,
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    /// Additional per-row tensor materialization used only by the continuous
    /// path when ragged attention rows are concatenated back into one batch.
    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        continuous_decode_collation_workspace_bytes(
            self.text_model.hidden_size(),
            self.compute_dtype.size_in_bytes(),
        )
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

        let mut prompt_messages = messages.to_vec();
        if !matches!(
            prompt_messages.first().map(|m| &m.role),
            Some(ChatRole::System)
        ) {
            prompt_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are a helpful assistant.".to_string(),
                },
            );
        }

        let mut ids = Vec::new();
        for message in &prompt_messages {
            let content = if matches!(message.role, ChatRole::Assistant) {
                strip_think_blocks(message.content.trim())
            } else {
                message.content.trim().to_string()
            };

            if content.is_empty() {
                continue;
            }

            ids.push(self.tokenizer.specials.im_start);
            ids.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            ids.extend(self.tokenizer.encode_text(&content)?);
            ids.push(self.tokenizer.specials.im_end);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(ids)
    }
}

fn strip_think_blocks(input: &str) -> String {
    let mut output = input.to_string();
    let open = "<think>";
    let close = "</think>";

    if let Some(close_idx) = output.find(close) {
        let has_open_before_close = output[..close_idx].contains(open);
        if !has_open_before_close {
            let start = close_idx + close.len();
            output = output[start..].to_string();
        }
    }

    while let Some(start) = output.find(open) {
        let search_from = start + open.len();
        if let Some(end_rel) = output[search_from..].find(close) {
            let end = search_from + end_rel + close.len();
            output.replace_range(start..end, "");
            continue;
        }

        output.truncate(start);
        break;
    }

    output.replace(close, " ").trim().to_string()
}

fn parse_qwen3_config(config_str: &str) -> Result<Qwen3Config> {
    let value: Value = serde_json::from_str(config_str)?;
    if let Some(text_config) = value.get("text_config") {
        serde_json::from_value(text_config.clone()).map_err(Error::from)
    } else {
        serde_json::from_value(value).map_err(Error::from)
    }
}

fn parse_qwen3_gguf_config(loader: &GgufLoader) -> Result<Qwen3Config> {
    // Qwen3-MoE checkpoints carry the "qwen3moe" architecture id with MoE
    // expert keys; dense checkpoints use "qwen3". Accept both prefixes so one
    // parser serves the family.
    let arch = ["qwen3moe", "qwen3"]
        .into_iter()
        .find(|arch| loader.get_metadata_u64(&format!("{arch}.block_count")).is_some())
        .ok_or_else(|| {
            Error::ModelLoadError("Missing or invalid GGUF metadata: qwen3.block_count".to_string())
        })?;
    let required_usize = |key: &str| {
        loader
            .get_metadata_u64(&format!("{arch}.{key}"))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {arch}.{key}"))
            })
    };
    let optional_usize = |key: &str| {
        loader
            .get_metadata_u64(&format!("{arch}.{key}"))
            .and_then(|value| usize::try_from(value).ok())
    };
    let required_f64 = |key: &str| {
        loader
            .get_metadata_f64(&format!("{arch}.{key}"))
            .ok_or_else(|| {
                Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {arch}.{key}"))
            })
    };
    let vocab_size = loader
        .get_metadata_array_len("tokenizer.ggml.tokens")
        .ok_or_else(|| {
            Error::ModelLoadError(
                "Missing or invalid GGUF metadata: tokenizer.ggml.tokens".to_string(),
            )
        })?;

    let num_experts = optional_usize("expert_count");
    let num_experts_per_tok = optional_usize("expert_used_count");
    let moe_intermediate_size = optional_usize("expert_feed_forward_length");
    let norm_topk_prob = loader
        .get_metadata_f64(&format!("{arch}.norm_topk_prob"))
        .map(|value| value != 0.0);

    Ok(Qwen3Config {
        hidden_size: required_usize("embedding_length")?,
        intermediate_size: required_usize("feed_forward_length")?,
        num_attention_heads: required_usize("attention.head_count")?,
        num_hidden_layers: required_usize("block_count")?,
        num_key_value_heads: required_usize("attention.head_count_kv")?,
        max_position_embeddings: Some(required_usize("context_length")?),
        head_dim: loader
            .get_metadata_u64(&format!("{arch}.attention.key_length"))
            .and_then(|value| usize::try_from(value).ok()),
        rms_norm_eps: required_f64("attention.layer_norm_rms_epsilon")?,
        rope_theta: required_f64("rope.freq_base")?,
        vocab_size,
        lm_head_size: None,
        tie_word_embeddings: !loader.has_tensor("output.weight"),
        rope_scaling: None,
        sliding_window: None,
        use_sliding_window: false,
        ada_rms_norm_t_cond: false,
        ada_rms_norm_t_cond_dim: 0,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        norm_topk_prob,
    })
}

fn select_qwen3_dense_dtype(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
    checkpoint_dtype: Option<DType>,
) -> Result<DType> {
    match dtype_override.map(str::trim) {
        Some(raw) if !raw.is_empty() => {
            device.select_model_dtype_checked(ModelFamily::Qwen3Chat, Some(raw), "Qwen3 chat")
        }
        _ if device.kind.is_metal() => {
            // Keep chat memory/latency practical on Apple Silicon.
            Ok(DType::F16)
        }
        _ => {
            Ok(device.select_model_dtype_with_checkpoint(ModelFamily::Qwen3Chat, checkpoint_dtype))
        }
    }
}

fn continuous_decode_collation_workspace_bytes(
    hidden_size: usize,
    dtype_bytes: usize,
) -> Result<u64> {
    u64::try_from(hidden_size)
        .ok()
        .and_then(|hidden| hidden.checked_mul(u64::try_from(dtype_bytes).ok()?))
        .ok_or_else(|| {
            Error::Overloaded("continuous decode workspace estimate overflow".to_string())
        })
}

#[cfg(test)]
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected logits rank for argmax: {rank}"
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

#[cfg(test)]
fn argmax_last_sequence_output(output: &Tensor) -> Result<u32> {
    let (batch, sequence, _width) = output.dims3()?;
    if batch != 1 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "Qwen3 decode output expects [1,sequence,width], got {:?}",
            output.dims()
        )));
    }
    argmax(&output.i((0, sequence - 1))?)
}

#[cfg(test)]
fn take_quantum_argmax(output: &mut Option<Tensor>) -> Result<u32> {
    let output = output.take().ok_or_else(|| {
        Error::InferenceError("Qwen3 decode quantum has no unconsumed model output".to_string())
    })?;
    argmax_last_sequence_output(&output)
}

#[cfg(test)]
mod tests {
    use super::{
        argmax_last_sequence_output, continuous_decode_collation_workspace_bytes,
        select_qwen3_dense_dtype, strip_think_blocks, take_quantum_argmax,
    };
    use crate::backends::{DeviceCapabilities, DeviceKind, DeviceProfile};
    use candle_core::{DType, Device, IndexOp, Tensor};

    #[test]
    fn strip_think_blocks_handles_explicit_tags() {
        let stripped = strip_think_blocks("<think>reasoning</think>\nFinal answer");
        assert_eq!(stripped, "Final answer");
    }

    #[test]
    fn strip_think_blocks_handles_implicit_open_pattern() {
        let stripped = strip_think_blocks("reasoning only</think>\nFinal answer");
        assert_eq!(stripped, "Final answer");
    }

    #[test]
    fn continuous_decode_workspace_tracks_loaded_hidden_size_and_dtype() {
        assert_eq!(
            continuous_decode_collation_workspace_bytes(1_024, 2).unwrap(),
            2_048
        );
        assert_eq!(
            continuous_decode_collation_workspace_bytes(3_584, 4).unwrap(),
            14_336
        );
    }

    #[test]
    fn sampling_consumes_quantum_output_without_changing_argmax() {
        let output = Tensor::from_vec(vec![0.5f32, 7.0, 2.0], (1, 1, 3), &Device::Cpu).unwrap();
        let expected = argmax_last_sequence_output(&output).unwrap();
        let mut unconsumed = Some(output);

        assert_eq!(take_quantum_argmax(&mut unconsumed).unwrap(), expected);
        assert!(unconsumed.is_none());
        assert!(take_quantum_argmax(&mut unconsumed).is_err());
    }

    #[test]
    fn continuous_rows_are_sampled_from_batch_output_without_state_tensors() {
        let batch = Tensor::from_vec(
            vec![0.0f32, 4.0, 1.0, 9.0, 3.0, 2.0],
            (2, 1, 3),
            &Device::Cpu,
        )
        .unwrap();
        let sampled = (0..2)
            .map(|row| {
                let row_output = batch.i(row).unwrap().unsqueeze(0).unwrap();
                argmax_last_sequence_output(&row_output).unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(sampled, vec![1, 0]);
    }

    #[test]
    fn qwen3_dense_cuda_uses_checkpoint_dtype() {
        let profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let dtype = select_qwen3_dense_dtype(&profile, None, Some(DType::F16)).unwrap();
        assert_eq!(dtype, DType::F16);
    }
}

/// The tiny Qwen3 chat model: eight whole-character pieces, and specials
/// past the vocabulary, so a row ends only at its token budget.
#[cfg(test)]
pub(crate) fn tiny_chat_model_for_test(device: &candle_core::Device) -> Qwen3ChatModel {
    stream_tests::tiny_chat_model(device, stream_tests::ASCII_PIECES)
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use crate::backends::{DeviceCapabilities, DeviceKind};
    use crate::models::architectures::qwen3::core::tiny_qwen3_model_for_test;
    use crate::models::shared::attention::physical::test_physical_caches_on;
    use crate::models::shared::telemetry::thread_host_reads;
    use candle_core::Device;

    const PROMPT: [u32; 3] = [0, 1, 7];

    /// Byte-level pieces for the tiny model's eight ids: "a", " ", the two
    /// bytes of "é", the three bytes of "€", and "b".
    const SPLIT_PIECES: &str = r#"{"a":0,"Ġ":1,"Ã":2,"©":3,"â":4,"Ĥ":5,"¬":6,"b":7}"#;
    /// Eight whole-character pieces.
    pub(super) const ASCII_PIECES: &str = r#"{"a":0,"Ġ":1,"b":2,"c":3,"Ġd":4,"e":5,"f":6,"g":7}"#;

    /// The tiny model with a byte-level WordLevel tokenizer over `pieces`.
    /// The specials sit past the vocabulary, so a row ends only at its token
    /// budget.
    pub(super) fn tiny_chat_model(device: &Device, pieces: &str) -> Qwen3ChatModel {
        let inner = Tokenizer::from_hf_json_bytes(
            format!(
                r#"{{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},
            "model":{{"type":"WordLevel","unk_token":"a","vocab":{pieces}}}
        }}"#
            )
            .as_bytes(),
        )
        .unwrap();
        Qwen3ChatModel {
            device: DeviceProfile {
                device: device.clone(),
                kind: if device.is_metal() {
                    DeviceKind::Metal
                } else {
                    DeviceKind::Cpu
                },
                capabilities: DeviceCapabilities::default(),
                memory_pool: None,
            },
            compute_dtype: DType::F32,
            tokenizer: ChatTokenizer {
                inner,
                vocab_size: 8,
                specials: SpecialTokenIds {
                    im_start: 8,
                    im_end: 9,
                    eos: 9,
                    eos_alt: None,
                },
            },
            text_model: tiny_qwen3_model_for_test(device),
            prefix_reuse: false,
        }
    }

    fn prefilled_rows(
        model: &Qwen3ChatModel,
        device: &Device,
        configs: &[ChatGenerationConfig],
        max_new_tokens: usize,
    ) -> Vec<ChatDecodeState> {
        let capacity = PROMPT.len() + max_new_tokens + 1;
        test_physical_caches_on(device, 1, 1, 2, capacity, configs.len())
            .into_iter()
            .zip(configs)
            .map(|(cache, config)| {
                let mut state = model
                    .begin_resumable_prefill_managed(&PROMPT, max_new_tokens, config, cache)
                    .unwrap();
                model
                    .continue_resumable_prefill(&mut state, &PROMPT, 0, PROMPT.len())
                    .unwrap();
                state
            })
            .collect()
    }

    /// Each row's generated ids, streamed deltas and terminal text, decoded
    /// through the continuous batch after one single-row step per row.
    fn generate(
        model: &Qwen3ChatModel,
        device: &Device,
        configs: &[ChatGenerationConfig],
        max_new_tokens: usize,
    ) -> Vec<(Vec<u32>, Vec<String>, String)> {
        let mut states = prefilled_rows(model, device, configs, max_new_tokens);
        let mut deltas = vec![Vec::new(); states.len()];
        let mut texts = vec![None; states.len()];
        let mut record = |row: usize, step: ChatDecodeStep| {
            deltas[row].push(step.delta);
            if step.finished {
                texts[row] = Some(step.text);
            } else {
                assert!(step.text.is_empty(), "a non-terminal step copied the text");
            }
        };
        for (row, state) in states.iter_mut().enumerate() {
            record(row, model.decode_step(state).unwrap());
        }
        loop {
            let (rows, mut live): (Vec<usize>, Vec<&mut ChatDecodeState>) = states
                .iter_mut()
                .enumerate()
                .filter(|(_, state)| !state.finished)
                .unzip();
            if live.is_empty() {
                break;
            }
            let steps = model.decode_step_batch(&mut live).unwrap();
            for (row, step) in rows.into_iter().zip(steps) {
                record(row, step);
            }
        }
        states
            .into_iter()
            .zip(deltas)
            .zip(texts)
            .map(|((state, deltas), text)| (state.generated_ids, deltas, text.unwrap()))
            .collect()
    }

    /// The deltas the old path streamed: the whole sequence decoded again
    /// after every token and diffed against the previous decode.
    fn whole_sequence_deltas(model: &Qwen3ChatModel, ids: &[u32]) -> Vec<String> {
        let mut previous = String::new();
        (1..=ids.len())
            .map(|end| {
                let current = model.tokenizer.inner.decode(&ids[..end]).unwrap();
                let delta = match current.strip_prefix(previous.as_str()) {
                    Some(delta) => delta.to_string(),
                    None => {
                        let common = previous
                            .chars()
                            .zip(current.chars())
                            .take_while(|(a, b)| a == b)
                            .count();
                        current.chars().skip(common).collect()
                    }
                };
                previous = current;
                delta
            })
            .collect()
    }

    /// A greedy and two seeded 2,048-token runs through the continuous
    /// batch stream exactly the whole-sequence decode, and step for step
    /// what the old whole-sequence path streamed wherever that was
    /// well-formed. Returns how many characters the runs split across
    /// tokens.
    fn assert_stream_matches_whole_sequence_decode(device: &Device, pieces: &str) -> usize {
        let model = tiny_chat_model(device, pieces);
        let sampled = |seed, top_k| ChatGenerationConfig {
            temperature: 1.5,
            top_k,
            seed,
            ..ChatGenerationConfig::default()
        };
        let configs = [
            ChatGenerationConfig::default(),
            sampled(11, 0),
            sampled(12, 6),
        ];
        let max_new_tokens = 2_048;
        let mut split_characters = 0;
        let rows = generate(&model, device, &configs, max_new_tokens);
        for (row, (ids, deltas, text)) in rows.iter().enumerate() {
            assert_eq!(ids.len(), max_new_tokens, "row {row} ended early");
            assert_eq!(deltas.len(), ids.len(), "row {row}: one step per token");
            let whole = model.tokenizer.inner.decode(ids).unwrap();
            assert_eq!(deltas.concat(), whole, "row {row}: stream != whole decode");
            assert_eq!(text, whole.trim(), "row {row}: terminal text");
            // The old stream sent a U+FFFD for a split character before
            // the character itself; elsewhere both streams agree per step.
            let old = whole_sequence_deltas(&model, ids);
            if old.iter().all(|delta| !delta.contains('\u{FFFD}')) {
                assert_eq!(deltas, &old, "row {row}: per-step deltas");
            }
            split_characters += whole.matches(['é', '€']).count();
        }
        split_characters
    }

    fn assert_streams_on(device: &Device) {
        assert_eq!(
            assert_stream_matches_whole_sequence_decode(device, ASCII_PIECES),
            0
        );
        assert!(
            assert_stream_matches_whole_sequence_decode(device, SPLIT_PIECES) > 0,
            "the runs never split a character across tokens"
        );
    }

    /// Run the pending token's forward as `decode_step` does and return
    /// the logits it leaves for sampling.
    fn pending_logits(model: &Qwen3ChatModel, state: &mut ChatDecodeState) -> Vec<f32> {
        let pending = state.pending_token.take().expect("pending token");
        let input = Tensor::from_vec(vec![pending], (1, 1), &model.device.device).unwrap();
        let logits = model
            .text_model
            .forward_managed(&input, state.pos, &mut state.cache)
            .unwrap();
        state.pos += 1;
        state.unconsumed_output = Some(logits.clone());
        logits.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// D2: a session released for capacity after `SUSPEND_AT` tokens and
    /// rebuilt in a fresh cache by replaying its prompt and generated tokens
    /// over the original span boundaries (the prompt, then one decode-shaped
    /// span per token) resumes exactly: the rebuilt next-step logits equal an
    /// unpreempted run's bit for bit, and the greedy and seeded streams match
    /// delta for delta. A replay in one span runs other op shapes, so it is
    /// held to a tolerance.
    fn assert_replay_resumes_exactly(device: &Device) {
        const MAX_NEW_TOKENS: usize = 24;
        const SUSPEND_AT: usize = 7;
        let model = tiny_chat_model(device, ASCII_PIECES);
        let sampled = ChatGenerationConfig {
            temperature: 1.5,
            seed: 11,
            ..ChatGenerationConfig::default()
        };
        for config in [ChatGenerationConfig::default(), sampled] {
            for one_span in [false, true] {
                let mut caches =
                    test_physical_caches_on(device, 1, 1, 2, PROMPT.len() + MAX_NEW_TOKENS + 1, 3);
                let fresh = caches.pop().unwrap();
                let mut rows = caches
                    .into_iter()
                    .map(|cache| {
                        let mut state = model
                            .begin_resumable_prefill_managed(
                                &PROMPT,
                                MAX_NEW_TOKENS,
                                &config,
                                cache,
                            )
                            .unwrap();
                        model
                            .continue_resumable_prefill(&mut state, &PROMPT, 0, PROMPT.len())
                            .unwrap();
                        state
                    })
                    .collect::<Vec<_>>();
                let mut suspended = rows.pop().unwrap();
                let mut baseline = rows.pop().unwrap();
                let mut deltas = Vec::new();
                for _ in 0..SUSPEND_AT {
                    let step = model.decode_step(&mut baseline).unwrap();
                    assert_eq!(model.decode_step(&mut suspended).unwrap().delta, step.delta);
                    deltas.push(step.delta);
                }
                let checkpoint = suspended.replay_checkpoint().unwrap();
                assert_eq!(checkpoint.replay_tokens(), PROMPT.len() + SUSPEND_AT - 1);
                drop(suspended);
                let mut replayed = model.begin_replay_managed(&checkpoint, fresh).unwrap();
                assert!(model.decode_step(&mut replayed).is_err());
                let spans = if one_span {
                    vec![(0, checkpoint.replay_tokens())]
                } else {
                    std::iter::once((0, PROMPT.len()))
                        .chain(
                            (PROMPT.len()..checkpoint.replay_tokens())
                                .map(|start| (start, start + 1)),
                        )
                        .collect()
                };
                for (index, (start, end)) in spans.iter().copied().enumerate() {
                    let complete = model
                        .continue_replay(&mut replayed, &PROMPT, start, end)
                        .unwrap();
                    assert_eq!(complete, index + 1 == spans.len());
                }
                let expected = pending_logits(&model, &mut baseline);
                let actual = pending_logits(&model, &mut replayed);
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
                    assert!(error < 1e-4, "one-span replay logits differ by {error}");
                    continue;
                }
                assert_eq!(actual, expected, "rebuilt next-step logits");
                while !baseline.finished {
                    let step = model.decode_step(&mut baseline).unwrap();
                    assert_eq!(model.decode_step(&mut replayed).unwrap().delta, step.delta);
                    deltas.push(step.delta);
                }
                assert!(replayed.finished);
                assert_eq!(replayed.generated_ids, baseline.generated_ids);
                assert_eq!(deltas.len(), MAX_NEW_TOKENS);
            }
        }
    }

    #[test]
    fn replayed_session_resumes_exactly_on_cpu() {
        assert_replay_resumes_exactly(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn replayed_session_resumes_exactly_on_metal() {
        let Some((device, _serial)) = crate::models::shared::metal_test_device_or_skip(
            "qwen3 replayed_session_resumes_exactly_on_metal",
        ) else {
            return;
        };
        assert_replay_resumes_exactly(&device);
    }

    #[test]
    fn incremental_stream_matches_whole_sequence_decode_on_cpu() {
        assert_streams_on(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn incremental_stream_matches_whole_sequence_decode_on_metal() {
        let Some((device, _serial)) = crate::models::shared::metal_test_device_or_skip(
            "qwen3 incremental_stream_matches_whole_sequence_decode_on_metal",
        ) else {
            return;
        };
        assert_streams_on(&device);
    }

    /// Two request streams on one Metal device, run concurrently, read back
    /// exactly what they read back alone, on every iteration: a tiny model
    /// decoding through the accelerator KV arena, and greedy sampling of
    /// uploaded logits (Phase 2.2 saw this sampler read back token 0 while a
    /// model decoded on another thread). Each stream's device work runs on
    /// the device thread, interleaved step by step as the engine interleaves
    /// requests. Ignored soak:
    /// `cargo test -p izwi-core --features metal --lib concurrent_streams_on_one_metal_device -- --ignored`
    /// (`IZWI_DEVICE_SOAK_ITERATIONS`, default 50).
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "Metal soak; run explicitly"]
    fn concurrent_streams_on_one_metal_device_match_a_serial_run() {
        use crate::models::shared::sampling::ChatSampler;

        let Some((device, _serial)) = crate::models::shared::metal_test_device_or_skip(
            "qwen3 concurrent_streams_on_one_metal_device_match_a_serial_run",
        ) else {
            return;
        };
        let iterations = std::env::var("IZWI_DEVICE_SOAK_ITERATIONS")
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(50);
        let device_thread =
            crate::backends::DeviceThread::for_device(&device).expect("device thread");
        let on_device = |work: &mut (dyn FnMut() + Send)| {
            device_thread.run_sync(work).expect("device job");
        };
        let model = tiny_chat_model(&device, ASCII_PIECES);
        let configs = [ChatGenerationConfig::default()];
        let decode_stream = || {
            let mut state = None;
            on_device(&mut || state = prefilled_rows(&model, &device, &configs, 40).pop());
            let mut state = state.expect("prefilled row");
            for _ in 0..32 {
                on_device(&mut || {
                    model.decode_step(&mut state).expect("decode step");
                });
            }
            state.generated_ids.clone()
        };
        // Unstructured logits whose greedy pick the host knows exactly.
        const WIDTH: usize = 1_500;
        let logits = |step: u64| {
            let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ step.wrapping_mul(0x2545_F491);
            (0..WIDTH)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 24) as f32 * 8.0 - 4.0
                })
                .collect::<Vec<_>>()
        };
        let sampler_stream = || {
            let mut sampler = ChatSampler::new(ChatGenerationConfig::default(), &[]);
            (0..64)
                .map(|step| {
                    let mut token = None;
                    let values = logits(step);
                    on_device(&mut || {
                        let logits = Tensor::from_vec(values.clone(), WIDTH, &device).unwrap();
                        token = Some(sampler.sample(&logits, WIDTH).expect("sample"));
                    });
                    token.expect("sampled token")
                })
                .collect::<Vec<_>>()
        };
        let greedy = (0..64)
            .map(|step| {
                let values = logits(step);
                (0..WIDTH)
                    .max_by(|&left, &right| values[left].total_cmp(&values[right]))
                    .expect("non-empty logits") as u32
            })
            .collect::<Vec<_>>();
        let expected_decode = decode_stream();
        assert_eq!(expected_decode.len(), 32);
        assert_eq!(sampler_stream(), greedy);
        for iteration in 0..iterations {
            let (decoded, sampled) = std::thread::scope(|scope| {
                let decode = scope.spawn(decode_stream);
                let sample = scope.spawn(sampler_stream);
                (
                    decode.join().expect("decode stream"),
                    sample.join().expect("sampler stream"),
                )
            });
            assert_eq!(decoded, expected_decode, "iteration {iteration}: decode stream");
            assert_eq!(sampled, greedy, "iteration {iteration}: sampler stream");
        }
    }

    /// A step of a multi-row batch reads the device back once, whatever mix
    /// of greedy, sampled and logprob rows it holds.
    fn assert_one_readback_per_batch_step(device: &Device) {
        let model = tiny_chat_model(device, ASCII_PIECES);
        let configs = [
            ChatGenerationConfig::default(),
            ChatGenerationConfig {
                temperature: 0.8,
                top_k: 4,
                seed: 5,
                ..ChatGenerationConfig::default()
            },
            ChatGenerationConfig {
                logprobs: true,
                top_logprobs: 2,
                ..ChatGenerationConfig::default()
            },
        ];
        let mut states = prefilled_rows(&model, device, &configs, 32);
        for state in &mut states {
            model.decode_step(state).unwrap();
        }
        for _ in 0..16 {
            let mut rows = states.iter_mut().collect::<Vec<_>>();
            let before = thread_host_reads();
            let steps = model.decode_step_batch(&mut rows).unwrap();
            assert_eq!(thread_host_reads() - before, 1, "readbacks per step");
            assert_eq!(steps.len(), 3);
            assert_eq!(states[2].pending_logprobs.len(), 1);
        }
    }

    #[test]
    fn continuous_batch_reads_back_once_per_step_on_cpu() {
        assert_one_readback_per_batch_step(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn continuous_batch_reads_back_once_per_step_on_metal() {
        let Some((device, _serial)) = crate::models::shared::metal_test_device_or_skip(
            "qwen3 continuous_batch_reads_back_once_per_step_on_metal",
        ) else {
            return;
        };
        assert_one_readback_per_batch_step(&device);
    }

    /// A mixed step whose op shapes equal a split call's (only decode rows,
    /// or one prefill row) is that call: same tokens, logprobs and state.
    #[test]
    fn mixed_steps_with_split_shapes_are_the_split_calls() {
        let device = Device::Cpu;
        let model = tiny_chat_model(&device, ASCII_PIECES);
        let configs = [
            ChatGenerationConfig {
                temperature: 0.8,
                seed: 3,
                logprobs: true,
                top_logprobs: 2,
                ..ChatGenerationConfig::default()
            },
            ChatGenerationConfig::default(),
        ];
        let summary = |state: &ChatDecodeState| {
            (
                state.pos,
                state.prefill_progress,
                state.pending_token,
                state.generated_ids.clone(),
                state.cache.context_len(),
            )
        };
        let mut split = prefilled_rows(&model, &device, &configs, 8);
        let mut mixed = prefilled_rows(&model, &device, &configs, 8);
        for (split, mixed) in split.iter_mut().zip(&mut mixed) {
            model.decode_step(split).unwrap();
            model.decode_step(mixed).unwrap();
        }
        for _ in 0..3 {
            let expected = model
                .decode_step_batch(&mut split.iter_mut().collect::<Vec<_>>())
                .unwrap();
            let (actual, prefill) = model
                .step_mixed(&mut mixed.iter_mut().collect::<Vec<_>>(), &mut [])
                .unwrap();
            assert!(prefill.is_empty());
            for (row, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(expected.delta, actual.delta);
                assert_eq!(summary(&split[row]), summary(&mixed[row]));
                assert_eq!(split[row].pending_logprobs, mixed[row].pending_logprobs);
            }
        }

        let caches = test_physical_caches_on(&device, 1, 1, 2, 16, 2);
        let mut states = caches
            .into_iter()
            .map(|cache| {
                model
                    .begin_resumable_prefill_managed(&PROMPT, 8, &configs[0], cache)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let (mut split, mut mixed) = (states.remove(0), states.remove(0));
        for (start, end) in [(0, 2), (2, PROMPT.len())] {
            let complete = model
                .continue_resumable_prefill(&mut split, &PROMPT, start, end)
                .unwrap();
            let expected = complete.then(|| model.decode_step(&mut split).unwrap());
            let (decode, mut prefill) = model
                .step_mixed(
                    &mut [],
                    &mut [MixedPrefillRow {
                        state: &mut mixed,
                        prompt: &PROMPT,
                        span_start: start,
                        span_end: end,
                    }],
                )
                .unwrap();
            assert!(decode.is_empty());
            let actual = prefill.remove(0);
            assert_eq!(
                expected.map(|step| step.delta),
                actual.map(|step| step.delta)
            );
            assert_eq!(summary(&split), summary(&mixed));
            assert_eq!(split.pending_logprobs, mixed.pending_logprobs);
        }
    }
}
