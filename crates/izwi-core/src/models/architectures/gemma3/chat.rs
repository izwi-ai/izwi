//! Native Gemma 3 text-chat model loader and generation.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Tensor};
#[cfg(test)]
use candle_core::{IndexOp, D};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3::Config as Gemma3Config;
use serde_json::Value;
use tracing::{info, warn};

use crate::backends::kv::KvWriteBatchCompletion;
use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::kv::v2::StateDomainId;
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::architectures::gemma3::core::Gemma3PhysicalModel;
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{
    replay_journal_span, ChatGenerationConfig, ChatMessage, ChatRole,
};
use crate::models::shared::config::checkpoint_dtype_from_config_json;
use crate::models::shared::sampling::{sample_chat_rows, ChatSampler};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

pub struct ChatDecodeState {
    cache: PhysicalPagedKvCache,
    unconsumed_logits: Option<Tensor>,
    position: usize,
    pending_token: Option<u32>,
    /// Scheduler-visible prompt cursor, separate from a reused physical
    /// prefix position on the first scheduler-visible span.
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    text: ReplyText,
    stagnant_steps: usize,
    max_new_tokens: usize,
    finished: bool,
    /// Cache length a capacity replay is rebuilding; the state decodes again
    /// once its cache reaches it.
    replay_end: Option<usize>,
    /// DS9.3: logprob entries produced by the current decode step, drained
    /// by the registry right after the step. Cleared at each sample.
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
    text: ReplyText,
    stagnant_steps: usize,
    max_new_tokens: usize,
}

impl ChatReplayCheckpoint {
    pub(crate) fn replay_tokens(&self) -> usize {
        self.replay_tokens
    }
}

pub(crate) struct ChatDecodeCheckpoint {
    cache: PhysicalPagedKvCache,
    unconsumed_logits: Option<Tensor>,
    position: usize,
    pending_token: Option<u32>,
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    text: ReplyText,
    stagnant_steps: usize,
    finished: bool,
}

impl ChatDecodeState {
    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    /// The replay a capacity suspension still has to rebuild, if any.
    pub(crate) fn replay_tokens(&self) -> Option<usize> {
        self.replay_end
    }

    /// Capture the CPU continuation record at a quantum boundary, before the
    /// caller releases the cache.
    pub(crate) fn replay_checkpoint(&self) -> Result<ChatReplayCheckpoint> {
        let replay_tokens = self.replay_end.unwrap_or(self.position);
        let pending_token = self.pending_token.filter(|_| {
            !self.finished
                && self.unconsumed_logits.is_none()
                && replay_tokens + 1 == self.prefill_progress + self.generated_ids.len()
        });
        let Some(pending_token) = pending_token else {
            return Err(Error::InvalidInput(
                "a Gemma session suspends only between decode quanta".into(),
            ));
        };
        Ok(ChatReplayCheckpoint {
            replay_tokens,
            pending_token,
            prefill_progress: self.prefill_progress,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            text: self.text.clone(),
            stagnant_steps: self.stagnant_steps,
            max_new_tokens: self.max_new_tokens,
        })
    }

    pub(crate) fn install_physical_reservation(
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
        if self.cache.arena().id() != cache.arena().id()
            || self.cache.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Gemma session cannot switch physical KV authority".into(),
            ));
        }
        if cache.context_len() != self.position {
            return Err(Error::InferenceError(format!(
                "physical Gemma reservation starts at {}, but decode state is at {}",
                cache.context_len(),
                self.position
            )));
        }
        let checkpoint = ChatDecodeCheckpoint {
            cache: std::mem::replace(&mut self.cache, cache),
            unconsumed_logits: self.unconsumed_logits.clone(),
            position: self.position,
            pending_token: self.pending_token,
            prefill_progress: self.prefill_progress,
            generated_ids: self.generated_ids.clone(),
            sampler: self.sampler.clone(),
            text: self.text.clone(),
            stagnant_steps: self.stagnant_steps,
            finished: self.finished,
        };
        Ok(checkpoint)
    }

    pub(crate) fn rollback_managed_quantum(&mut self, checkpoint: ChatDecodeCheckpoint) {
        self.cache = checkpoint.cache;
        self.unconsumed_logits = checkpoint.unconsumed_logits;
        self.position = checkpoint.position;
        self.pending_token = checkpoint.pending_token;
        self.prefill_progress = checkpoint.prefill_progress;
        self.generated_ids = checkpoint.generated_ids;
        self.sampler = checkpoint.sampler;
        self.text = checkpoint.text;
        self.stagnant_steps = checkpoint.stagnant_steps;
        self.finished = checkpoint.finished;
    }

    pub(crate) fn take_physical_write_completions(&mut self) -> Vec<Arc<KvWriteBatchCompletion>> {
        self.cache.take_completed_writes()
    }
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
    bos: Option<u32>,
    eos: u32,
    start_of_turn: u32,
    end_of_turn: u32,
}

struct GemmaTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl GemmaTokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
        let vocab_size = inner.vocab_size();

        let token_id = |token: &str| -> Result<u32> {
            inner.token_to_id(token).ok_or_else(|| {
                Error::TokenizationError(format!("Missing Gemma special token: {token}"))
            })
        };

        let start_of_turn = token_id("<start_of_turn>")?;
        let end_of_turn = token_id("<end_of_turn>")?;
        let eos = inner.token_to_id("<eos>").unwrap_or(end_of_turn);
        let bos = inner.token_to_id("<bos>");

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                bos,
                eos,
                start_of_turn,
                end_of_turn,
            },
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    /// The text `id` adds to the reply. Ids past the vocabulary add
    /// nothing, as a whole-sequence decode skips them.
    fn decode_next(&self, text: &mut ReplyText, id: u32) -> Result<String> {
        if id as usize >= self.vocab_size {
            return Ok(String::new());
        }
        let decoded = self.inner.decode_incrementally(&mut text.decoder, id)?;
        Ok(text.release(&decoded, false))
    }

    /// The reply text still held back once generation ends.
    fn finish_text(&self, text: &mut ReplyText) -> Result<String> {
        let decoded = self.inner.finish_incremental_decode(&mut text.decoder)?;
        Ok(text.release(&decoded, true))
    }

    /// The stagnant-text rule: four consecutive tokens that cannot change
    /// the text end the reply. Returns whether `id` ends it. A token cannot
    /// change the text when it is a skipped special, an `<unusedN>`
    /// placeholder or past the vocabulary; byte-fallback pieces decode to
    /// U+FFFD on their own, so a character split across tokens never counts.
    /// This is the old whole-decode rule ("the decode did not change")
    /// counted per token.
    fn ends_stagnant_run(&self, run: &mut usize, id: u32) -> Result<bool> {
        let adds_no_text = id as usize >= self.vocab_size
            || strip_unused_placeholders(&self.inner.decode(&[id])?).is_empty();
        *run = if adds_no_text { *run + 1 } else { 0 };
        Ok(*run >= 4)
    }
}

/// Incrementally decoded reply text with `<unusedN>` placeholders stripped
/// as it streams. A trailing piece that could still become a placeholder is
/// held until later text completes or breaks it, so the released text
/// concatenates to `strip_unused_placeholders` of the whole decode.
#[derive(Debug, Clone)]
struct ReplyText {
    decoder: IncrementalDecoder,
    held: String,
}

impl ReplyText {
    fn new(skip_special_tokens: bool) -> Self {
        Self {
            decoder: IncrementalDecoder::new(skip_special_tokens),
            held: String::new(),
        }
    }

    fn release(&mut self, decoded: &str, finished: bool) -> String {
        self.held.push_str(decoded);
        let keep = if finished {
            self.held.len()
        } else {
            partial_placeholder_start(&self.held)
        };
        let ready = self.held.drain(..keep).collect::<String>();
        strip_unused_placeholders(&ready)
    }
}

/// Where `text`'s trailing, possibly unfinished `<unusedN>` placeholder
/// starts, or `text.len()` when nothing at its end can become one. Only the
/// last `<` can start one: a placeholder holds no other `<`.
fn partial_placeholder_start(text: &str) -> usize {
    const MARKER: &str = "<unused";
    let Some(start) = text.rfind('<') else {
        return text.len();
    };
    let tail = &text[start..];
    let partial = if tail.len() <= MARKER.len() {
        MARKER.starts_with(tail)
    } else {
        tail.starts_with(MARKER) && tail[MARKER.len()..].bytes().all(|b| b.is_ascii_digit())
    };
    if partial {
        start
    } else {
        text.len()
    }
}

struct GemmaDefaults {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

fn defaults_for_variant(variant: ModelVariant) -> GemmaDefaults {
    match variant {
        ModelVariant::Gemma34BIt => GemmaDefaults {
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_hidden_layers: 34,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 131_072,
        },
        _ => GemmaDefaults {
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_hidden_layers: 34,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 131_072,
        },
    }
}

fn parse_gemma3_config(
    config_str: &str,
    variant: ModelVariant,
    tokenizer_vocab_size: usize,
    checkpoint_vocab_size: Option<usize>,
) -> Result<Gemma3Config> {
    let root_value: Value = serde_json::from_str(config_str)?;
    let has_text_config = root_value.get("text_config").is_some();
    let source = root_value
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| root_value.clone());

    let mut object: BTreeMap<String, Value> = source
        .as_object()
        .ok_or_else(|| Error::InvalidInput("Invalid Gemma config.json format".to_string()))?
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let defaults = defaults_for_variant(variant);

    let mut set_default = |key: &str, value: Value| {
        if !object.contains_key(key) {
            object.insert(key.to_string(), value);
        }
    };

    set_default("attention_bias", Value::Bool(false));
    set_default(
        "hidden_activation",
        Value::String("gelu_pytorch_tanh".to_string()),
    );
    set_default("hidden_size", Value::from(defaults.hidden_size as u64));
    set_default(
        "intermediate_size",
        Value::from(defaults.intermediate_size as u64),
    );
    // MLX gemma3.py applies these multimodal defaults for Gemma 3 4B configs
    // when they are omitted from text_config.
    if has_text_config {
        set_default("num_attention_heads", Value::from(8u64));
        set_default("num_key_value_heads", Value::from(4u64));
    }

    set_default(
        "num_attention_heads",
        Value::from(defaults.num_attention_heads as u64),
    );
    set_default(
        "num_hidden_layers",
        Value::from(defaults.num_hidden_layers as u64),
    );
    set_default(
        "num_key_value_heads",
        Value::from(defaults.num_key_value_heads as u64),
    );
    set_default("head_dim", Value::from(defaults.head_dim as u64));
    set_default("rms_norm_eps", Value::from(1e-6f64));
    set_default("rope_theta", Value::from(1_000_000f64));
    set_default("rope_local_base_freq", Value::from(10_000f64));
    set_default("query_pre_attn_scalar", Value::from(256u64));
    set_default("sliding_window", Value::from(512u64));
    set_default("sliding_window_pattern", Value::from(6u64));
    set_default(
        "max_position_embeddings",
        Value::from(defaults.max_position_embeddings as u64),
    );
    let resolved_vocab_size = checkpoint_vocab_size.unwrap_or({
        if has_text_config {
            262_208
        } else {
            tokenizer_vocab_size
        }
    });
    if let Some(config_vocab_size) = object.get("vocab_size").and_then(|value| value.as_u64()) {
        if config_vocab_size as usize != resolved_vocab_size {
            info!(
                "Overriding Gemma vocab_size from config {} to {}",
                config_vocab_size, resolved_vocab_size
            );
        }
    }
    object.insert(
        "vocab_size".to_string(),
        Value::from(resolved_vocab_size as u64),
    );

    let config =
        serde_json::from_value::<Gemma3Config>(Value::Object(object.into_iter().collect()))
            .map_err(Error::from)?;

    Ok(config)
}

fn should_use_language_model_prefix(config_str: &str) -> bool {
    let Ok(root) = serde_json::from_str::<Value>(config_str) else {
        return false;
    };

    if root
        .get("architectures")
        .and_then(|v| v.as_array())
        .is_some_and(|architectures| {
            architectures.iter().any(|entry| {
                entry
                    .as_str()
                    .is_some_and(|name| name == "Gemma3ForConditionalGeneration")
            })
        })
    {
        return true;
    }

    root.get("model_type")
        .and_then(|v| v.as_str())
        .is_some_and(|model_type| model_type == "gemma3")
}

const MAX_SAFE_TENSORS_HEADER_SIZE: usize = 100_000_000;

fn tensor_shape_from_safetensors_header(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<Vec<usize>>> {
    let mut file = fs::File::open(safetensors_path)?;

    let mut n_buf = [0u8; 8];
    file.read_exact(&mut n_buf)?;
    let header_len_u64 = u64::from_le_bytes(n_buf);
    let header_len: usize = header_len_u64
        .try_into()
        .map_err(|_| Error::InvalidInput("Invalid safetensors header length".to_string()))?;
    if header_len > MAX_SAFE_TENSORS_HEADER_SIZE {
        return Err(Error::InvalidInput(format!(
            "Safetensors header too large: {header_len}"
        )));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let metadata: Value = serde_json::from_slice(&header_buf)?;
    let tensor_entry = match metadata.get(tensor_name) {
        Some(entry) => entry,
        None => return Ok(None),
    };
    let shape = match tensor_entry.get("shape").and_then(|shape| shape.as_array()) {
        Some(shape) => shape,
        None => return Ok(None),
    };

    let dims = shape
        .iter()
        .map(|dim| dim.as_u64().map(|value| value as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Invalid shape metadata for tensor {tensor_name} in {}",
                safetensors_path.display()
            ))
        })?;

    Ok(Some(dims))
}

fn infer_embed_vocab_size_from_safetensors(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<usize>> {
    let shape = match tensor_shape_from_safetensors_header(safetensors_path, tensor_name)? {
        Some(shape) => shape,
        None => return Ok(None),
    };

    let Some(vocab_size) = shape.first().copied() else {
        return Ok(None);
    };

    Ok(Some(vocab_size))
}

fn select_gemma3_dense_dtype(device: &DeviceProfile, checkpoint_dtype: Option<DType>) -> DType {
    if device.kind.is_metal() {
        // Prefer F16 for Gemma on Metal to keep memory use in check on larger checkpoints.
        DType::F16
    } else {
        device.select_model_dtype_with_checkpoint(ModelFamily::Gemma3Chat, checkpoint_dtype)
    }
}

pub struct Gemma3ChatModel {
    variant: ModelVariant,
    device: DeviceProfile,
    compute_dtype: DType,
    /// DS1.6 catalog verdict for committed prefix reuse on the load backend.
    prefix_reuse: bool,
    tokenizer: GemmaTokenizer,
    text_model: Gemma3PhysicalModel,
}

impl InferenceStateContractProvider for Gemma3ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
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

impl Gemma3ChatModel {
    pub fn max_context_tokens(&self) -> Result<usize> {
        let context = self.text_model.max_context_tokens();
        if context == 0 {
            return Err(Error::ModelLoadError(
                "Gemma 3 checkpoint has a zero context length".into(),
            ));
        }
        Ok(context)
    }

    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device: DeviceProfile,
        prefix_reuse: bool,
    ) -> Result<Self> {
        let tokenizer = GemmaTokenizer::load(model_dir)?;

        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let mut use_language_model_prefix = should_use_language_model_prefix(&config_str);
        let mut inferred_vocab_size: Option<usize> = None;
        let checkpoint_dtype = checkpoint_dtype_from_config_json(&config_str);
        let dtype = select_gemma3_dense_dtype(&device, checkpoint_dtype);

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb_base = if index_path.exists() {
            let index_data = fs::read_to_string(&index_path)?;
            let index: Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;
            if weight_map
                .keys()
                .any(|tensor_name| tensor_name.starts_with("language_model."))
            {
                use_language_model_prefix = true;
            }

            let embed_tensor_name = if use_language_model_prefix {
                "language_model.model.embed_tokens.weight"
            } else {
                "model.embed_tokens.weight"
            };
            if let Some(shard_name) = weight_map.get(embed_tensor_name).and_then(|v| v.as_str()) {
                let shard_path = model_dir.join(shard_name);
                inferred_vocab_size =
                    infer_embed_vocab_size_from_safetensors(&shard_path, embed_tensor_name)?;
            } else {
                let fallback_tensor_name =
                    if embed_tensor_name == "language_model.model.embed_tokens.weight" {
                        "model.embed_tokens.weight"
                    } else {
                        "language_model.model.embed_tokens.weight"
                    };
                if let Some(shard_name) = weight_map
                    .get(fallback_tensor_name)
                    .and_then(|v| v.as_str())
                {
                    let shard_path = model_dir.join(shard_name);
                    inferred_vocab_size =
                        infer_embed_vocab_size_from_safetensors(&shard_path, fallback_tensor_name)?;
                    if fallback_tensor_name.starts_with("language_model.") {
                        use_language_model_prefix = true;
                    }
                }
            }

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
            for tensor_name in [
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
            ] {
                if let Some(vocab_size) =
                    infer_embed_vocab_size_from_safetensors(&weights_path, tensor_name)?
                {
                    inferred_vocab_size = Some(vocab_size);
                    if tensor_name.starts_with("language_model.") {
                        use_language_model_prefix = true;
                    }
                    break;
                }
            }
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? }
        };

        if let Some(vocab_size) = inferred_vocab_size {
            info!(
                "Gemma {} using checkpoint embedding vocab size {}",
                variant.dir_name(),
                vocab_size
            );
        } else {
            warn!(
                "Could not infer Gemma {} embedding vocab size from checkpoint, falling back to tokenizer vocab {}",
                variant.dir_name(),
                tokenizer.vocab_size
            );
        }

        let config = parse_gemma3_config(
            &config_str,
            variant,
            tokenizer.vocab_size,
            inferred_vocab_size,
        )?;
        info!(
            "Gemma {} config resolved: layers={}, heads={}, kv_heads={}, head_dim={}, hidden_size={}, sliding_window={}, sliding_window_pattern={}, max_position_embeddings={}, dtype={:?}",
            variant.dir_name(),
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.hidden_size,
            config.sliding_window,
            config.sliding_window_pattern,
            config.max_position_embeddings,
            dtype
        );

        let vb = if use_language_model_prefix {
            vb_base.pp("language_model")
        } else {
            vb_base
        };

        let text_model = Gemma3PhysicalModel::load(config, vb)?;

        info!(
            "Loaded physical Gemma chat model {} on {:?}",
            variant.dir_name(),
            device.kind
        );

        Ok(Self {
            variant,
            device,
            compute_dtype: dtype,
            tokenizer,
            text_model,
            prefix_reuse,
        })
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
            "Gemma generation requires scheduler-owned physical state".into(),
        ))
    }

    /// Start a session in engine-owned physical KV, prefilling the
    /// request's sealed prompt (the tokens a capacity replay writes back).
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
                "Gemma resumable prefill requires at least one private prompt token".into(),
            ));
        }
        // DS1.5: a managed prefix attach starts the logical cursor at the
        // attached physical cursor.
        let position = cache.context_len();
        Ok(ChatDecodeState {
            cache,
            unconsumed_logits: None,
            position,
            pending_token: None,
            prefill_progress: position,
            generated_ids: Vec::new(),
            sampler: ChatSampler::new(config.clone(), prompt_ids).with_json_object_constraint(
                std::sync::Arc::new(self.tokenizer.inner.clone()),
                vec![
                    Some(self.tokenizer.specials.end_of_turn),
                    Some(self.tokenizer.specials.eos),
                ]
                .into_iter()
                .flatten()
                .collect(),
            ),
            text: ReplyText::new(config.skip_special_tokens),
            stagnant_steps: 0,
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
                "Gemma replay requires an empty reservation and a non-empty journal".into(),
            ));
        }
        Ok(ChatDecodeState {
            cache,
            unconsumed_logits: None,
            position: 0,
            pending_token: Some(saved.pending_token),
            prefill_progress: saved.prefill_progress,
            generated_ids: saved.generated_ids.clone(),
            sampler: saved.sampler.clone(),
            text: saved.text.clone(),
            stagnant_steps: saved.stagnant_steps,
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
            .ok_or_else(|| Error::InvalidInput("Gemma state has no pending replay".into()))?;
        if span_start != state.position
            || span_start != state.cache.context_len()
            || span_start >= span_end
            || span_end > replay_end
            || prompt_ids.len() != state.prefill_progress
        {
            return Err(Error::InvalidInput(format!(
                "Gemma replay span [{span_start},{span_end}) does not continue cursor {} of {replay_end}",
                state.position
            )));
        }
        let ids = replay_journal_span(prompt_ids, &state.generated_ids, span_start, span_end)?;
        let input = Tensor::from_vec(ids, (1, span_end - span_start), &self.device.device)?;
        self.text_model
            .prefill_physical(&input, span_start, &mut state.cache)?;
        state.position = span_end;
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
            || state.unconsumed_logits.is_some()
            || state.pending_token.is_some()
            || !state.generated_ids.is_empty()
        {
            return Err(Error::InvalidInput(format!(
                "Gemma resumable prefill span [{span_start},{span_end}) is incompatible with cursor {} and prompt length {}",
                state.prefill_progress,
                prompt_ids.len()
            )));
        }
        let physical_start = state.cache.context_len();
        let first_span = span_start == 0;
        if state.position != physical_start
            || (!first_span && physical_start != span_start)
            || (first_span && physical_start >= span_end)
        {
            return Err(Error::InferenceError(format!(
                "Gemma resumable prefill physical cursor {physical_start} is incompatible with logical span [{span_start},{span_end})"
            )));
        }
        let input = Tensor::from_slice(
            &prompt_ids[physical_start..span_end],
            (1, span_end - physical_start),
            &self.device.device,
        )?;
        let logits = self
            .text_model
            .forward_physical(&input, physical_start, &mut state.cache)?;
        if state.cache.context_len() != span_end {
            return Err(Error::InferenceError(format!(
                "Gemma resumable prefill committed physical cursor {} instead of {span_end}",
                state.cache.context_len()
            )));
        }
        state.position = span_end;
        state.prefill_progress = span_end;
        let complete = span_end == prompt_ids.len();
        if complete {
            state.unconsumed_logits = Some(logits);
        }
        Ok(complete)
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.replay_end.is_some() {
            return Err(Error::InferenceError(
                "Gemma decode cannot run before its replay completes".into(),
            ));
        }
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            return self.finish_decode(state, String::new());
        }
        if let Some(token) = state.pending_token.take() {
            let input = Tensor::from_vec(vec![token], (1, 1), &self.device.device)?;
            state.unconsumed_logits = Some(self.text_model.forward_physical(
                &input,
                state.position,
                &mut state.cache,
            )?);
            state.position += 1;
        }
        let logits = state.unconsumed_logits.take().ok_or_else(|| {
            Error::InferenceError("Gemma decode quantum has no unconsumed logits".into())
        })?;
        let (next, entry) = self.sample_token(state, &logits)?;
        self.apply_sample(state, next, entry)
    }

    /// DS9.3: sample one token, resolving raw logprobs into the public
    /// payload when the request asked for them. The entry is recorded by
    /// `apply_sample` only when the token is actually emitted.
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

    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        if states.iter().any(|state| {
            state.finished
                || state.generated_ids.len() >= state.max_new_tokens
                || state.unconsumed_logits.is_some()
                || state.replay_end.is_some()
        }) {
            return Err(Error::InvalidInput(
                "Gemma continuous batch contains an unready decode state".into(),
            ));
        }
        let tokens = states
            .iter_mut()
            .map(|state| {
                state.pending_token.take().ok_or_else(|| {
                    Error::InferenceError(
                        "Gemma continuous decode state has no scheduled token".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let positions = states
            .iter()
            .map(|state| state.position)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(tokens, (states.len(), 1), &self.device.device)?;
        let mut caches = states
            .iter_mut()
            .map(|state| &mut state.cache)
            .collect::<Vec<_>>();
        let logits =
            self.text_model
                .forward_physical_decode_batch(&input, &positions, &mut caches)?;
        for state in states.iter_mut() {
            state.position += 1;
            state.pending_logprobs.clear();
        }
        // Every row samples from one readback of the whole step.
        let logits = logits.reshape((states.len(), ()))?;
        let mut samplers = states
            .iter_mut()
            .map(|state| &mut state.sampler)
            .collect::<Vec<_>>();
        let sampled = sample_chat_rows(&mut samplers, &logits, self.tokenizer.vocab_size)?;
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
                self.apply_sample(state, next, entry)
            })
            .collect()
    }

    fn apply_sample(
        &self,
        state: &mut ChatDecodeState,
        next: u32,
        logprob_entry: Option<crate::engine::TokenLogprob>,
    ) -> Result<ChatDecodeStep> {
        if next == self.tokenizer.specials.end_of_turn
            || next == self.tokenizer.specials.eos
            || next == self.tokenizer.specials.start_of_turn
            || self.tokenizer.specials.bos.is_some_and(|bos| next == bos)
            || state.sampler.is_configured_stop(next)
        {
            return self.finish_decode(state, String::new());
        }
        if let Some(entry) = logprob_entry {
            state.pending_logprobs.push(entry);
        }
        state.generated_ids.push(next);
        state.pending_token = Some(next);
        let delta = self.tokenizer.decode_next(&mut state.text, next)?;
        let stagnant = self
            .tokenizer
            .ends_stagnant_run(&mut state.stagnant_steps, next)?;
        if stagnant || state.generated_ids.len() >= state.max_new_tokens {
            return self.finish_decode(state, delta);
        }
        Ok(ChatDecodeStep {
            delta,
            text: String::new(),
            tokens_generated: state.generated_ids.len(),
            finished: false,
        })
    }

    /// Finish the row: flush the text still held back and return the whole
    /// text, which only a terminal step carries.
    fn finish_decode(
        &self,
        state: &mut ChatDecodeState,
        mut delta: String,
    ) -> Result<ChatDecodeStep> {
        state.finished = true;
        let suffix = self.tokenizer.finish_text(&mut state.text)?;
        delta.push_str(&suffix);
        Ok(ChatDecodeStep {
            delta,
            text: strip_unused_placeholders(state.text.decoder.decoded_text())
                .trim()
                .to_string(),
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

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        u64::try_from(self.text_model.hidden_size())
            .ok()
            .and_then(|hidden| {
                hidden.checked_mul(u64::try_from(self.compute_dtype.size_in_bytes()).ok()?)
            })
            .ok_or_else(|| Error::Overloaded("Gemma decode workspace estimate overflow".into()))
    }

    pub fn runtime_device_kind(&self) -> String {
        format!("{:?}", self.device.kind).to_ascii_lowercase()
    }

    pub fn runtime_compute_dtype(&self) -> Option<String> {
        Some(format!("{:?}", self.compute_dtype).to_ascii_lowercase())
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

        let mut system_parts = Vec::new();
        let mut turns: Vec<(ChatRole, String)> = Vec::new();

        for message in messages {
            let content = if matches!(message.role, ChatRole::Assistant) {
                strip_think_blocks(message.content.trim())
            } else {
                message.content.trim().to_string()
            };
            if content.is_empty() {
                continue;
            }

            if matches!(message.role, ChatRole::System) {
                system_parts.push(content);
            } else {
                turns.push((message.role.clone(), content));
            }
        }

        let system = if system_parts.is_empty() {
            "You are a helpful assistant.".to_string()
        } else {
            system_parts.join("\n\n")
        };

        if let Some((role, first_content)) = turns.first_mut() {
            if matches!(role, ChatRole::User) {
                *first_content = format!("{system}\n\n{first_content}");
            } else {
                turns.insert(0, (ChatRole::User, system));
            }
        } else {
            turns.push((ChatRole::User, system));
        }

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        for (role, content) in &turns {
            let role_name = match role {
                ChatRole::Assistant => "model",
                ChatRole::User | ChatRole::System | ChatRole::Tool => "user",
            };

            ids.push(self.tokenizer.specials.start_of_turn);
            ids.extend(self.tokenizer.encode_text(role_name)?);
            ids.extend(self.tokenizer.encode_text("\n")?);
            ids.extend(self.tokenizer.encode_text(content)?);
            ids.push(self.tokenizer.specials.end_of_turn);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(self.tokenizer.specials.start_of_turn);
        ids.extend(self.tokenizer.encode_text("model")?);
        ids.extend(self.tokenizer.encode_text("\n")?);

        Ok(ids)
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }
}

fn strip_think_blocks(input: &str) -> String {
    let mut output = input.to_string();
    let open = "<think>";
    let close = "</think>";

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

    output.trim().to_string()
}

fn strip_unused_placeholders(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while cursor < input.len() {
        let tail = &input[cursor..];
        if let Some(after_prefix) = tail.strip_prefix("<unused") {
            let mut digit_len = 0usize;
            for ch in after_prefix.chars() {
                if ch.is_ascii_digit() {
                    digit_len += ch.len_utf8();
                } else {
                    break;
                }
            }

            if digit_len > 0 {
                let rest = &after_prefix[digit_len..];
                if let Some(after_marker) = rest.strip_prefix('>') {
                    let consumed = input.len() - after_marker.len() - cursor;
                    cursor += consumed;
                    continue;
                }
            }
        }

        let ch = tail.chars().next().unwrap_or_default();
        out.push(ch);
        cursor += ch.len_utf8();
    }

    out
}

#[cfg(test)]
fn argmax(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    let capped = vocab_limit.min(logits.dim(0)?);
    if capped == 0 {
        return Err(Error::InferenceError(
            "No valid logits in constrained vocabulary".to_string(),
        ));
    }
    let logits = if capped < logits.dim(0)? {
        logits.narrow(0, 0, capped)?
    } else {
        logits.clone()
    };
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    crate::models::shared::telemetry::record_dtype_cast();
    crate::models::shared::telemetry::record_host_read(DType::U32, 1);
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

#[cfg(test)]
fn select_next_token(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    match logits.rank() {
        // [vocab]
        1 => argmax(logits, vocab_limit),
        // [seq, vocab]
        2 => {
            let seq_len = logits.dim(0)?;
            argmax(&logits.i(seq_len.saturating_sub(1))?, vocab_limit)
        }
        // [batch, seq, vocab]
        3 => {
            let seq_len = logits.dim(1)?;
            argmax(&logits.i((0, seq_len.saturating_sub(1)))?, vocab_limit)
        }
        _ => Err(Error::InferenceError(format!(
            "Unexpected Gemma logits rank: {} with dims {:?}",
            logits.rank(),
            logits.dims()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        argmax, parse_gemma3_config, select_gemma3_dense_dtype, select_next_token,
        strip_unused_placeholders,
    };
    use crate::backends::{DeviceCapabilities, DeviceKind, DeviceProfile};
    use crate::model::ModelVariant;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn strip_unused_placeholders_removes_marker_tokens() {
        let input = "<unused6421> hello <unused9>world<unused123>";
        let output = strip_unused_placeholders(input);
        assert_eq!(output, " hello world");
    }

    #[test]
    fn strip_unused_placeholders_keeps_normal_angle_bracket_text() {
        let input = "a <unused> b <unusedx12> c";
        let output = strip_unused_placeholders(input);
        assert_eq!(output, input);
    }

    #[test]
    fn missing_context_uses_variant_native_limit() {
        let config = r#"{"vocab_size": 262208}"#;
        let four_b = parse_gemma3_config(config, ModelVariant::Gemma34BIt, 262_208, None)
            .expect("4B defaults");

        assert_eq!(four_b.max_position_embeddings, 131_072);
    }

    #[test]
    fn explicit_context_overrides_variant_default() {
        let config = r#"{"vocab_size":262208,"max_position_embeddings":777}"#;
        let parsed = parse_gemma3_config(config, ModelVariant::Gemma34BIt, 262_208, None)
            .expect("explicit context");

        assert_eq!(parsed.max_position_embeddings, 777);
    }

    #[test]
    fn gemma3_dense_cuda_uses_checkpoint_dtype() {
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

        assert_eq!(
            select_gemma3_dense_dtype(&profile, Some(DType::F32)),
            DType::F32
        );
    }

    #[test]
    fn gemma3_argmax_stays_inside_the_constrained_vocabulary() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.8, 0.4, 10.0], (4,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        assert_eq!(argmax(&logits, 3).unwrap(), 1);
    }

    #[test]
    fn gemma3_selects_from_the_last_sequence_position() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.9f32, 0.1, 0.0, -0.2, 0.4, 0.7], (2, 3), &device).unwrap();

        assert_eq!(select_next_token(&logits, 3).unwrap(), 2);
    }

    #[test]
    fn gemma3_argmax_rejects_an_empty_vocabulary() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.2], (2,), &device).unwrap();
        let err = argmax(&logits, 0).expect_err("zero vocabulary should be rejected");

        assert!(format!("{err}").contains("No valid logits"));
    }
}

/// The tiny Gemma 3 chat model over sixteen whole-character pieces.
#[cfg(test)]
pub(crate) fn tiny_chat_model_for_test(device: &candle_core::Device) -> Gemma3ChatModel {
    stream_tests::tiny_chat_model(device, &stream_tests::PLAIN_PIECES)
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use crate::backends::{DeviceCapabilities, DeviceKind};
    use crate::models::architectures::gemma3::core::tiny_gemma3_model_for_test;
    use crate::models::shared::attention::physical::test_physical_caches_on;
    use crate::models::shared::telemetry::thread_host_reads;
    use candle_core::Device;

    const PROMPT: [u32; 3] = [0, 1, 2];
    const MAX_NEW_TOKENS: usize = 2_048;

    /// Sixteen pieces with "é" and "€" split into byte-fallback tokens and
    /// an `<unused7>` placeholder split across "<", "unused" and "7>".
    const SPLIT_PIECES: [&str; 16] = [
        "▁a", "b", "c", "▁", "<0xC3>", "<0xA9>", "<0xE2>", "<0x82>", "<0xAC>", "d", "<", "unused",
        "7>", "e", "f", "g",
    ];
    /// Sixteen whole-character pieces.
    pub(super) const PLAIN_PIECES: [&str; 16] = [
        "▁a", "b", "c", "▁", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    ];

    /// A WordLevel tokenizer over `pieces` with Gemma's decoder (`▁` to a
    /// space, byte fallback, fuse). Ids listed in `special` are special
    /// tokens a decode skips.
    fn gemma_tokenizer(pieces: &[&str], special: &[u32]) -> Tokenizer {
        let vocab = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| (piece.to_string(), serde_json::json!(id)))
            .collect::<serde_json::Map<_, _>>();
        let added = special
            .iter()
            .map(|&id| {
                serde_json::json!({
                    "id": id, "content": pieces[id as usize], "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect::<Vec<_>>();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"}
            ]},
            "model": {"type": "WordLevel", "unk_token": pieces[1], "vocab": vocab}
        });
        Tokenizer::from_hf_json_bytes(json.to_string().as_bytes()).unwrap()
    }

    /// The specials sit past the vocabulary, so only the token budget or
    /// the stagnant-text rule ends a row.
    fn gemma_chat_tokenizer(inner: Tokenizer) -> GemmaTokenizer {
        GemmaTokenizer {
            vocab_size: inner.vocab_size(),
            inner,
            specials: SpecialTokenIds {
                bos: None,
                eos: 100,
                start_of_turn: 101,
                end_of_turn: 102,
            },
        }
    }

    pub(super) fn tiny_chat_model(device: &Device, pieces: &[&str]) -> Gemma3ChatModel {
        Gemma3ChatModel {
            variant: ModelVariant::Gemma34BIt,
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
            prefix_reuse: false,
            tokenizer: gemma_chat_tokenizer(gemma_tokenizer(pieces, &[])),
            text_model: tiny_gemma3_model_for_test(
                device,
                pieces.len(),
                PROMPT.len() + MAX_NEW_TOKENS + 1,
            ),
        }
    }

    fn prefilled_rows(
        model: &Gemma3ChatModel,
        device: &Device,
        configs: &[ChatGenerationConfig],
        max_new_tokens: usize,
    ) -> Vec<ChatDecodeState> {
        let capacity = PROMPT.len() + max_new_tokens + 1;
        test_physical_caches_on(device, 2, 1, 2, capacity, configs.len())
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

    /// Each row's generated ids, streamed deltas, terminal text and decoder
    /// divergences, decoded through the continuous batch after one
    /// single-row step per row.
    fn generate(
        model: &Gemma3ChatModel,
        device: &Device,
        configs: &[ChatGenerationConfig],
    ) -> Vec<(Vec<u32>, Vec<String>, String, u64)> {
        let mut states = prefilled_rows(model, device, configs, MAX_NEW_TOKENS);
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
            .map(|((state, deltas), text)| {
                let divergences = state.text.decoder.divergences();
                (state.generated_ids, deltas, text.unwrap(), divergences)
            })
            .collect()
    }

    /// The old path: the placeholder-stripped decode of the whole sequence
    /// after every token, diffed against the previous one, with the old
    /// stagnant-text rule. Returns its deltas, the step that rule finished
    /// at, and whether every decode extended the previous one with
    /// well-formed text (only then was the old stream itself right).
    fn whole_sequence_stream(
        tokenizer: &GemmaTokenizer,
        ids: &[u32],
    ) -> (Vec<String>, Option<usize>, bool) {
        let mut assembled = String::new();
        let mut stagnant = 0;
        let mut well_formed = true;
        let mut deltas = Vec::new();
        for end in 1..=ids.len() {
            let filtered = ids[..end]
                .iter()
                .copied()
                .filter(|id| (*id as usize) < tokenizer.vocab_size)
                .collect::<Vec<_>>();
            let decoded = strip_unused_placeholders(&tokenizer.inner.decode(&filtered).unwrap());
            let delta = match decoded.strip_prefix(assembled.as_str()) {
                Some(delta) => delta.to_string(),
                None => {
                    well_formed = false;
                    let common = assembled
                        .chars()
                        .zip(decoded.chars())
                        .take_while(|(a, b)| a == b)
                        .count();
                    decoded.chars().skip(common).collect()
                }
            };
            well_formed &= !delta.contains('\u{FFFD}');
            deltas.push(delta);
            if decoded == assembled {
                stagnant += 1;
                if stagnant >= 4 {
                    return (deltas, Some(end), well_formed);
                }
            } else {
                stagnant = 0;
                assembled = decoded;
            }
        }
        (deltas, None, well_formed)
    }

    /// A greedy and two seeded 2,048-token runs through the continuous
    /// batch end with the old terminal text, and stream the whole-sequence
    /// decode exactly unless a stray byte made the decoder rewrite emitted
    /// text. With `step_for_step`, every step also streams what the old path
    /// streamed (whole characters, no placeholder to hold back).
    fn assert_stream_matches_whole_sequence_decode(
        device: &Device,
        pieces: &[&str],
        step_for_step: bool,
    ) {
        let model = tiny_chat_model(device, pieces);
        let sampled = |seed, top_k| ChatGenerationConfig {
            temperature: 1.5,
            top_k,
            seed,
            ..ChatGenerationConfig::default()
        };
        let configs = [
            ChatGenerationConfig::default(),
            sampled(21, 0),
            sampled(22, 9),
        ];
        for (row, (ids, deltas, text, divergences)) in
            generate(&model, device, &configs).into_iter().enumerate()
        {
            assert_eq!(ids.len(), MAX_NEW_TOKENS, "row {row} ended early");
            assert_eq!(deltas.len(), ids.len(), "row {row}: one step per token");
            let whole = strip_unused_placeholders(&model.tokenizer.inner.decode(&ids).unwrap());
            assert_eq!(text, whole.trim(), "row {row}: terminal text");
            if divergences == 0 {
                assert!(
                    deltas.concat() == whole,
                    "row {row}: stream != whole decode"
                );
            }
            let (old, stagnant_end, well_formed) = whole_sequence_stream(&model.tokenizer, &ids);
            assert_eq!(stagnant_end, None, "row {row}: no piece is silent");
            if step_for_step {
                assert!(well_formed, "row {row}: the old stream was well-formed");
                let first_difference = deltas.iter().zip(&old).position(|(new, old)| new != old);
                assert_eq!(first_difference, None, "row {row}: first differing step");
            }
        }
    }

    fn assert_streams_on(device: &Device) {
        assert_stream_matches_whole_sequence_decode(device, &PLAIN_PIECES, true);
        assert_stream_matches_whole_sequence_decode(device, &SPLIT_PIECES, false);
    }

    /// Run the pending token's forward as `decode_step` does and return
    /// the logits it leaves for sampling.
    fn pending_logits(model: &Gemma3ChatModel, state: &mut ChatDecodeState) -> Vec<f32> {
        let pending = state.pending_token.take().expect("pending token");
        let input = Tensor::from_vec(vec![pending], (1, 1), &model.device.device).unwrap();
        let logits = model
            .text_model
            .forward_physical(&input, state.position, &mut state.cache)
            .unwrap();
        state.position += 1;
        state.unconsumed_logits = Some(logits.clone());
        logits.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// D2: a session released for capacity after `SUSPEND_AT` tokens and
    /// rebuilt in a fresh cache by replaying its prompt and generated tokens
    /// over the original span boundaries resumes exactly: bit-identical
    /// next-step logits, and the same greedy and seeded streams (held
    /// placeholder text included). A one-span replay runs other op shapes
    /// and is held to a tolerance.
    fn assert_replay_resumes_exactly(device: &Device) {
        const REPLAY_NEW_TOKENS: usize = 40;
        const SUSPEND_AT: usize = 9;
        let sampled = ChatGenerationConfig {
            temperature: 1.5,
            seed: 11,
            ..ChatGenerationConfig::default()
        };
        for pieces in [&PLAIN_PIECES, &SPLIT_PIECES] {
            let model = tiny_chat_model(device, pieces);
            for config in [ChatGenerationConfig::default(), sampled.clone()] {
                for one_span in [false, true] {
                    let mut caches = test_physical_caches_on(
                        device,
                        2,
                        1,
                        2,
                        PROMPT.len() + REPLAY_NEW_TOKENS + 1,
                        3,
                    );
                    let fresh = caches.pop().unwrap();
                    let mut rows = caches
                        .into_iter()
                        .map(|cache| {
                            let mut state = model
                                .begin_resumable_prefill_managed(
                                    &PROMPT,
                                    REPLAY_NEW_TOKENS,
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
                    for _ in 0..SUSPEND_AT {
                        let step = model.decode_step(&mut baseline).unwrap();
                        assert_eq!(model.decode_step(&mut suspended).unwrap().delta, step.delta);
                    }
                    assert!(
                        !baseline.finished,
                        "the fixture ended before the suspension"
                    );
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
                        let resumed = model.decode_step(&mut replayed).unwrap();
                        assert_eq!(resumed.delta, step.delta);
                        assert_eq!(resumed.text, step.text);
                    }
                    assert!(replayed.finished);
                    assert_eq!(replayed.generated_ids, baseline.generated_ids);
                }
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
            "gemma3 replayed_session_resumes_exactly_on_metal",
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
            "gemma3 incremental_stream_matches_whole_sequence_decode_on_metal",
        ) else {
            return;
        };
        assert_streams_on(&device);
    }

    /// The stagnant-text rule, now counted per token, ends a reply at the
    /// same token as the old rule over the whole decode, and the streamed
    /// text up to there equals the old text.
    #[test]
    fn stagnant_text_rule_matches_the_whole_sequence_rule() {
        // "<pad>" is a skipped special and "<unused3>" a whole placeholder:
        // both silent. Byte pieces are not: alone they decode to U+FFFD.
        let pieces = [
            "<pad>",
            "▁a",
            "b",
            "<0xC3>",
            "<0xA9>",
            "<0xE2>",
            "<0x82>",
            "<0xAC>",
            "<unused3>",
            "<",
            "unused",
            "4>",
            "▁",
        ];
        let tokenizer = gemma_chat_tokenizer(gemma_tokenizer(&pieces, &[0]));
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut stagnant_finishes, mut budget_finishes) = (0, 0);
        for _ in 0..400 {
            let ids = (0..96)
                .map(|_| {
                    let draw = next();
                    // Half the tokens silent; ids 13 and up are out of
                    // vocabulary and silent too.
                    if draw % 2 == 0 {
                        [0u32, 8, 13][(draw >> 8) as usize % 3]
                    } else {
                        1 + (draw >> 8) as u32 % 12
                    }
                })
                .collect::<Vec<_>>();
            let (_, old_end, _) = whole_sequence_stream(&tokenizer, &ids);

            let mut text = ReplyText::new(true);
            let mut streamed = String::new();
            let mut stagnant = 0;
            let mut new_end = None;
            for (index, &id) in ids.iter().enumerate() {
                streamed.push_str(&tokenizer.decode_next(&mut text, id).unwrap());
                if tokenizer.ends_stagnant_run(&mut stagnant, id).unwrap() {
                    new_end = Some(index + 1);
                    break;
                }
            }
            streamed.push_str(&tokenizer.finish_text(&mut text).unwrap());
            assert_eq!(new_end, old_end, "ids {ids:?}");
            let end = new_end.unwrap_or(ids.len());
            let in_vocab = ids[..end]
                .iter()
                .copied()
                .filter(|id| (*id as usize) < tokenizer.vocab_size)
                .collect::<Vec<_>>();
            let whole = tokenizer.inner.decode(&in_vocab).unwrap();
            assert_eq!(text.decoder.decoded_text(), whole, "ids {ids:?}");
            if text.decoder.divergences() == 0 {
                assert_eq!(streamed, strip_unused_placeholders(&whole), "ids {ids:?}");
            }
            match new_end {
                Some(_) => stagnant_finishes += 1,
                None => budget_finishes += 1,
            }
        }
        assert!(stagnant_finishes > 0 && budget_finishes > 0);
    }

    #[test]
    fn reply_text_streams_split_characters_and_placeholders_exactly() {
        let tokenizer = gemma_chat_tokenizer(gemma_tokenizer(&SPLIT_PIECES, &[]));
        // " a", "é" in two byte tokens, "b", "€" in three, "<unused7>"
        // across three pieces, "c", then "<" "unused" kept as text.
        let ids = [0, 4, 5, 1, 6, 7, 8, 10, 11, 12, 2, 10, 11];
        let mut text = ReplyText::new(true);
        let mut deltas = ids
            .iter()
            .map(|&id| tokenizer.decode_next(&mut text, id).unwrap())
            .collect::<Vec<_>>();
        deltas.push(tokenizer.finish_text(&mut text).unwrap());
        assert_eq!(
            deltas,
            [" a", "", "é", "b", "", "", "€", "", "", "", "c", "", "", "<unused"]
        );
        assert_eq!(text.decoder.divergences(), 0);
        assert_eq!(
            deltas.concat(),
            strip_unused_placeholders(&tokenizer.inner.decode(&ids).unwrap())
        );
    }

    #[test]
    fn reply_text_survives_a_stray_byte_after_an_emitted_character() {
        let tokenizer = gemma_chat_tokenizer(gemma_tokenizer(&SPLIT_PIECES, &[]));
        // "é" then a stray continuation byte: ByteFallback decodes the
        // whole byte run as U+FFFD, rewriting the emitted "é".
        let ids = [4, 5, 5, 1];
        let mut text = ReplyText::new(true);
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&tokenizer.decode_next(&mut text, id).unwrap());
        }
        streamed.push_str(&tokenizer.finish_text(&mut text).unwrap());
        let whole = tokenizer.inner.decode(&ids).unwrap();
        assert_eq!(whole, "\u{FFFD}\u{FFFD}\u{FFFD}b");
        assert_eq!(streamed, "é\u{FFFD}\u{FFFD}\u{FFFD}b");
        assert_eq!(text.decoder.decoded_text(), whole);
        assert_eq!(text.decoder.divergences(), 1);
    }

    #[test]
    fn partial_placeholders_are_held_until_decided() {
        let mut text = ReplyText::new(true);
        assert_eq!(text.release("ab<unu", false), "ab");
        assert_eq!(text.release("sed12", false), "");
        assert_eq!(text.release(">c<", false), "c");
        assert_eq!(text.release("x", false), "<x");
        assert_eq!(text.release("<unused", true), "<unused");
    }

    /// A step of a multi-row batch reads the device back once, whatever mix
    /// of greedy, sampled and logprob rows it holds.
    fn assert_one_readback_per_batch_step(device: &Device) {
        let model = tiny_chat_model(device, &PLAIN_PIECES);
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
            "gemma3 continuous_batch_reads_back_once_per_step_on_metal",
        ) else {
            return;
        };
        assert_one_readback_per_batch_step(&device);
    }
}
