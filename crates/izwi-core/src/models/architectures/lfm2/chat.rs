//! LFM2/LFM2.5 GGUF text-chat model loader and generation.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::info;

use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::BackendKind;
use crate::backends::DeviceProfile;
use crate::engine::{InvocationTensorLease, StageDescriptor};
use crate::error::{Error, Result};
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRole};
use crate::models::shared::telemetry::record_prefill_sequence_span;
use crate::models::shared::weights::gguf::GgufLoader;
use crate::tokenizer::Tokenizer;

use super::backbone::{Lfm2ShortConvRuntimeState, QuantizedLfm2Backbone};
use super::config::{parse_lfm2_backbone_config, Lfm2BackboneConfig};
use super::physical::{
    lfm2_managed_cache_contract, lfm2_physical_state_spec, Lfm2PhysicalStateSpec,
};

pub struct ChatDecodeState {
    shortconv: Lfm2ShortConvRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
    position: usize,
    prefill_progress: usize,
}

pub(crate) struct Lfm2ChatDecodeCheckpoint {
    shortconv: Lfm2ShortConvRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    assembled: String,
    finished: bool,
    position: usize,
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
            assembled: self.assembled.clone(),
            finished: self.finished,
            position: self.position,
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
        self.assembled = checkpoint.assembled;
        self.finished = checkpoint.finished;
        self.position = checkpoint.position;
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
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
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
}

struct ChatTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    decode_piece_cache: Vec<OnceLock<String>>,
}

struct PromptScaffoldTokens {
    system_header: Vec<u32>,
    user_header: Vec<u32>,
    assistant_header: Vec<u32>,
    newline: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lfm2PrefillMode {
    Auto,
    Full,
    Token,
}

impl Lfm2PrefillMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Full => "full",
            Self::Token => "token",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lfm2PrefillExecution {
    Full,
}

impl Lfm2PrefillExecution {
    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Lfm2PrefillConfig {
    mode: Lfm2PrefillMode,
    token_prompt_threshold: usize,
}

impl Lfm2PrefillConfig {
    const DEFAULT_TOKEN_THRESHOLD: usize = 64;

    fn resolve(self, _prompt_tokens: usize) -> Lfm2PrefillExecution {
        // Candle's quantized LFM2 ShortConv cache ignores `index_pos` for
        // one-token calls, so token prefill can reuse state from a previous
        // request. A full prompt pass replaces both attention and ShortConv
        // state deterministically at the request boundary.
        Lfm2PrefillExecution::Full
    }
}

fn parse_lfm2_prefill_mode(raw: Option<&str>) -> Lfm2PrefillMode {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("full" | "sequence") => Lfm2PrefillMode::Full,
        Some("token" | "token_mode") => Lfm2PrefillMode::Token,
        _ => Lfm2PrefillMode::Auto,
    }
}

fn parse_lfm2_prefill_threshold(raw: Option<&str>) -> usize {
    raw.map(str::trim)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(Lfm2PrefillConfig::DEFAULT_TOKEN_THRESHOLD)
}

fn lfm2_prefill_config() -> &'static Lfm2PrefillConfig {
    static CONFIG: OnceLock<Lfm2PrefillConfig> = OnceLock::new();
    CONFIG.get_or_init(|| Lfm2PrefillConfig {
        mode: parse_lfm2_prefill_mode(std::env::var("IZWI_LFM2_PREFILL_MODE").ok().as_deref()),
        token_prompt_threshold: parse_lfm2_prefill_threshold(
            std::env::var("IZWI_LFM2_PREFILL_TOKEN_THRESHOLD")
                .ok()
                .as_deref(),
        ),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lfm2DefaultSystemPolicy {
    Auto,
    Always,
    Never,
}

impl Lfm2DefaultSystemPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Always => "always",
            Self::Never => "never",
        }
    }
}

fn parse_lfm2_default_system_policy(raw: Option<&str>) -> Lfm2DefaultSystemPolicy {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("always" | "on" | "true" | "1") => Lfm2DefaultSystemPolicy::Always,
        Some("never" | "off" | "false" | "0") => Lfm2DefaultSystemPolicy::Never,
        _ => Lfm2DefaultSystemPolicy::Auto,
    }
}

fn lfm2_default_system_policy() -> &'static Lfm2DefaultSystemPolicy {
    static POLICY: OnceLock<Lfm2DefaultSystemPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        parse_lfm2_default_system_policy(
            std::env::var("IZWI_LFM2_DEFAULT_SYSTEM_POLICY")
                .ok()
                .as_deref(),
        )
    })
}

fn should_prepend_default_system(
    messages: &[ChatMessage],
    policy: Lfm2DefaultSystemPolicy,
) -> bool {
    if matches!(
        messages.first().map(|message| &message.role),
        Some(ChatRole::System)
    ) {
        return false;
    }
    match policy {
        Lfm2DefaultSystemPolicy::Always => true,
        Lfm2DefaultSystemPolicy::Never => false,
        Lfm2DefaultSystemPolicy::Auto => {
            !(messages.len() == 1
                && matches!(messages.first().map(|m| &m.role), Some(ChatRole::User)))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lfm2PromptStylePolicy {
    Standard,
    Aggressive,
}

impl Lfm2PromptStylePolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Aggressive => "aggressive",
        }
    }
}

fn parse_lfm2_prompt_style_policy(raw: Option<&str>) -> Lfm2PromptStylePolicy {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("aggressive" | "compact" | "lean") => Lfm2PromptStylePolicy::Aggressive,
        _ => Lfm2PromptStylePolicy::Standard,
    }
}

fn lfm2_prompt_style_policy() -> &'static Lfm2PromptStylePolicy {
    static POLICY: OnceLock<Lfm2PromptStylePolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        parse_lfm2_prompt_style_policy(
            std::env::var("IZWI_LFM2_PROMPT_STYLE_POLICY")
                .ok()
                .as_deref(),
        )
    })
}

fn should_use_aggressive_single_turn_prompt(
    messages: &[ChatMessage],
    prepend_default_system: bool,
    style_policy: Lfm2PromptStylePolicy,
) -> bool {
    if prepend_default_system {
        return false;
    }
    let single_user_turn = messages.len() == 1
        && matches!(
            messages.first().map(|message| &message.role),
            Some(ChatRole::User)
        );
    if !single_user_turn {
        return false;
    }
    matches!(style_policy, Lfm2PromptStylePolicy::Aggressive)
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
            decode_piece_cache: (0..vocab_size).map(|_| OnceLock::new()).collect(),
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }

    fn decode_token_piece(&self, token_id: u32) -> Result<&str> {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return Ok("");
        }
        if let Some(piece) = self.decode_piece_cache[idx].get() {
            return Ok(piece.as_str());
        }
        let decoded = self.inner.decode(&[token_id])?;
        let _ = self.decode_piece_cache[idx].set(decoded);
        Ok(self.decode_piece_cache[idx]
            .get()
            .map(String::as_str)
            .unwrap_or(""))
    }
}

pub struct Lfm2ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
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
        let prompt_scaffold = PromptScaffoldTokens::load(&tokenizer)?;

        info!(
            "Loaded LFM2 GGUF chat model on {:?} from {}",
            device.kind,
            gguf_path.display()
        );
        let prefill = lfm2_prefill_config();
        info!(
            "LFM2 prefill policy: mode={}, token_prompt_threshold={}",
            prefill.mode.as_str(),
            prefill.token_prompt_threshold
        );
        info!(
            "LFM2 default system policy: {}",
            lfm2_default_system_policy().as_str()
        );
        info!(
            "LFM2 prompt style policy: {}",
            lfm2_prompt_style_policy().as_str()
        );

        Ok(Self {
            device,
            tokenizer,
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
        _config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        if prompt_ids.is_empty() || cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "LFM2 managed prefill requires a non-empty prompt and empty cache".into(),
            ));
        }
        Ok(ChatDecodeState {
            shortconv: self.text_model.new_shortconv_state(),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: None,
            generated_ids: Vec::with_capacity(max_new_tokens.max(1)),
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            position: 0,
            prefill_progress: 0,
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
            return Ok(ChatDecodeStep {
                delta: String::new(),
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
        let next = argmax(&logits)?;
        let is_stop = next == self.tokenizer.specials.im_end
            || next == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(next);
        let delta = if is_stop {
            state.finished = true;
            String::new()
        } else {
            let delta = self.tokenizer.decode_token_piece(next)?.to_string();
            state.generated_ids.push(next);
            state.assembled.push_str(&delta);
            if (should_check_repetition_loop(state.generated_ids.len())
                && has_token_repetition_loop(&state.generated_ids))
                || state.generated_ids.len() >= state.max_new_tokens
            {
                state.finished = true;
            } else {
                state.pending_token = Some(next);
            }
            delta
        };
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
        let mut next_tokens = Vec::with_capacity(states.len());
        for row in 0..states.len() {
            next_tokens.push(argmax(&logits.i(row)?)?);
        }
        let mut steps = Vec::with_capacity(states.len());
        for (state, next) in states.iter_mut().zip(next_tokens) {
            state.position = state.position.saturating_add(1);
            let is_stop = next == self.tokenizer.specials.im_end
                || next == self.tokenizer.specials.eos
                || self.tokenizer.specials.eos_alt == Some(next);
            let delta = if is_stop {
                state.finished = true;
                String::new()
            } else {
                let delta = self.tokenizer.decode_token_piece(next)?.to_string();
                state.generated_ids.push(next);
                state.assembled.push_str(&delta);
                if should_check_repetition_loop(state.generated_ids.len())
                    && has_token_repetition_loop(&state.generated_ids)
                    || state.generated_ids.len() >= state.max_new_tokens
                {
                    state.finished = true;
                } else {
                    state.pending_token = Some(next);
                }
                delta
            };
            steps.push(ChatDecodeStep {
                delta,
                text: if state.finished {
                    state.assembled.trim().to_string()
                } else {
                    String::new()
                },
                tokens_generated: state.generated_ids.len(),
                input_tokens_committed: 1,
                finished: state.finished,
            });
        }
        Ok(steps)
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Lfm2PhysicalStateSpec> {
        lfm2_physical_state_spec(&self.config, stage_graphs)
    }

    pub(crate) fn generate_with_callback_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        self.generate_with_callback_state(messages, max_new_tokens, on_delta, cache, shortconv)
    }

    fn generate_with_callback_state(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
    ) -> Result<ChatGenerationOutput> {
        let total_started = Instant::now();
        let prompt_build_started = Instant::now();
        let prompt_ids = self.build_prompt(messages)?;
        let prompt_build_ms = prompt_build_started.elapsed().as_secs_f64() * 1000.0;
        let model = &self.text_model;

        let prompt_len = prompt_ids.len();
        let prefill_started = Instant::now();
        let prefill_cfg = *lfm2_prefill_config();
        let prefill_exec = prefill_cfg.resolve(prompt_len);
        let (mut logits, mut position, prefill_steps) =
            self.prefill_prompt(model, prompt_ids.as_slice(), prefill_exec, cache, shortconv)?;
        let prefill_forward_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;

        let max_new_tokens = max_new_tokens.max(1);
        let mut generated_ids = Vec::with_capacity(max_new_tokens);
        let mut assembled = String::with_capacity(max_new_tokens.saturating_mul(4));
        let decode_started = Instant::now();
        let mut first_delta_ms: Option<f64> = None;

        while generated_ids.len() < max_new_tokens {
            let next = argmax(&logits)?;
            if next == self.tokenizer.specials.im_end
                || next == self.tokenizer.specials.eos
                || self.tokenizer.specials.eos_alt == Some(next)
            {
                break;
            }

            let delta = self.tokenizer.decode_token_piece(next)?;
            generated_ids.push(next);
            if !delta.is_empty() {
                if first_delta_ms.is_none() {
                    first_delta_ms = Some(total_started.elapsed().as_secs_f64() * 1000.0);
                }
                on_delta(delta);
            }
            assembled.push_str(delta);

            if should_check_repetition_loop(generated_ids.len())
                && has_token_repetition_loop(&generated_ids)
            {
                break;
            }

            let next_tensor = Tensor::from_slice(&[next], (1, 1), &self.device.device)?;
            logits = model
                .forward_tokens_physical(&next_tensor, position, cache, shortconv)
                .map_err(|e| Error::InferenceError(format!("LFM2 GGUF decode failed: {e}")))?;
            position += 1;
        }

        if tracing::enabled!(tracing::Level::DEBUG) {
            let total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
            let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
            tracing::debug!(
                target: "izwi::lfm2::timing",
                prompt_tokens = prompt_len,
                prefill_policy = prefill_cfg.mode.as_str(),
                prefill_execution = prefill_exec.as_str(),
                prefill_steps,
                prompt_build_ms,
                prefill_forward_ms,
                first_delta_ms = first_delta_ms.unwrap_or(total_ms),
                decode_ms,
                total_ms,
                generated_tokens = generated_ids.len(),
                "LFM2 chat timing breakdown"
            );
        }

        Ok(ChatGenerationOutput {
            text: assembled.trim().to_string(),
            tokens_generated: generated_ids.len(),
        })
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

        let prepend_default_system =
            should_prepend_default_system(messages, *lfm2_default_system_policy());
        let prompt_style = *lfm2_prompt_style_policy();
        if should_use_aggressive_single_turn_prompt(messages, prepend_default_system, prompt_style)
        {
            return self.build_aggressive_single_turn_prompt(messages[0].content.as_str());
        }

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        let last_assistant_index = messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant))
            .map(|index| index + usize::from(prepend_default_system));

        let mut prompt_index = 0usize;
        if prepend_default_system {
            self.append_prompt_message(
                &mut ids,
                prompt_index,
                &ChatRole::System,
                "You are a helpful assistant.",
                last_assistant_index,
            )?;
            prompt_index += 1;
        }

        for message in messages {
            self.append_prompt_message(
                &mut ids,
                prompt_index,
                &message.role,
                message.content.as_str(),
                last_assistant_index,
            )?;
            prompt_index += 1;
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

    fn build_aggressive_single_turn_prompt(&self, content: &str) -> Result<Vec<u32>> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one tokenizable message".to_string(),
            ));
        }

        // Aggressive TTFT mode intentionally uses a leaner single-turn prompt to
        // reduce prefill token count for common benchmark traffic.
        let mut ids = self.tokenizer.encode_text(trimmed)?;
        ids.extend_from_slice(&self.prompt_scaffold.newline);
        ids.push(self.tokenizer.specials.im_start);
        ids.extend_from_slice(&self.prompt_scaffold.assistant_header);
        Ok(ids)
    }

    fn prefill_prompt(
        &self,
        model: &QuantizedLfm2Backbone,
        prompt_ids: &[u32],
        exec: Lfm2PrefillExecution,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
    ) -> Result<(Tensor, usize, usize)> {
        if prompt_ids.is_empty() {
            return Err(Error::InvalidInput(
                "LFM2 chat prompt must include at least one token".to_string(),
            ));
        }

        match exec {
            Lfm2PrefillExecution::Full => {
                record_prefill_sequence_span(prompt_ids.len());
                let input_ids =
                    Tensor::from_slice(prompt_ids, (1, prompt_ids.len()), &self.device.device)?;
                let logits = model
                    .forward_tokens_physical(&input_ids, 0, cache, shortconv)
                    .map_err(|e| Error::InferenceError(format!("LFM2 GGUF forward failed: {e}")))?;
                Ok((logits, prompt_ids.len(), 1))
            }
        }
    }
}

impl InferenceStateContractProvider for Lfm2ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        Ok(InferenceStateCapability::Managed(
            lfm2_managed_cache_contract(&self.config)?,
        ))
    }
}

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
            )));
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

fn strip_past_assistant_thinking(input: &str) -> Cow<'_, str> {
    if let Some((_reasoning, tail)) = input.rsplit_once("</think>") {
        Cow::Owned(tail.trim().to_string())
    } else {
        Cow::Borrowed(input.trim())
    }
}

fn has_suffix_repeat(ids: &[u32], span: usize, repeats: usize) -> bool {
    if span == 0 || repeats < 2 || ids.len() < span * repeats {
        return false;
    }
    let tail_start = ids.len() - span;
    let tail = &ids[tail_start..];
    (2..=repeats).all(|rep| {
        let start = ids.len() - (span * rep);
        &ids[start..start + span] == tail
    })
}

fn should_check_repetition_loop(len: usize) -> bool {
    len >= 48 && len.is_multiple_of(4)
}

fn has_token_repetition_loop(ids: &[u32]) -> bool {
    // Catch common degenerate loops from greedy decode where the same token span
    // is emitted repeatedly (frequent in tiny reasoning models).
    if ids.len() < 48 {
        return false;
    }
    const PATTERNS: &[(usize, usize)] = &[(24, 3), (16, 3), (12, 3), (8, 4), (6, 5)];
    PATTERNS
        .iter()
        .any(|(span, repeats)| has_suffix_repeat(ids, *span, *repeats))
}

#[cfg(test)]
mod tests {
    use super::{
        has_token_repetition_loop, parse_lfm2_default_system_policy, parse_lfm2_prefill_mode,
        parse_lfm2_prefill_threshold, parse_lfm2_prompt_style_policy, should_check_repetition_loop,
        should_prepend_default_system, should_use_aggressive_single_turn_prompt,
        strip_past_assistant_thinking, Lfm2DefaultSystemPolicy, Lfm2PrefillConfig,
        Lfm2PrefillExecution, Lfm2PrefillMode, Lfm2PromptStylePolicy,
    };
    use crate::models::shared::chat::{ChatMessage, ChatRole};

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
    fn detects_token_repetition_loop() {
        let mut ids = Vec::new();
        let phrase = [1, 2, 3, 4, 5, 6, 7, 8];
        for _ in 0..5 {
            ids.extend(phrase.iter().copied());
        }
        ids.splice(0..0, vec![42; 16]);
        assert!(has_token_repetition_loop(&ids));
    }

    #[test]
    fn does_not_flag_short_sequences_as_loop() {
        let ids: Vec<u32> = (1..30).collect();
        assert!(!has_token_repetition_loop(&ids));
    }

    #[test]
    fn repetition_loop_check_interval_skips_between_boundaries() {
        assert!(!should_check_repetition_loop(47));
        assert!(should_check_repetition_loop(48));
        assert!(!should_check_repetition_loop(49));
        assert!(should_check_repetition_loop(52));
    }

    #[test]
    fn parse_prefill_mode_defaults_to_auto_for_unknown_values() {
        assert_eq!(parse_lfm2_prefill_mode(None), Lfm2PrefillMode::Auto);
        assert_eq!(
            parse_lfm2_prefill_mode(Some("unsupported")),
            Lfm2PrefillMode::Auto
        );
        assert_eq!(parse_lfm2_prefill_mode(Some("FULL")), Lfm2PrefillMode::Full);
        assert_eq!(
            parse_lfm2_prefill_mode(Some("token_mode")),
            Lfm2PrefillMode::Token
        );
    }

    #[test]
    fn parse_prefill_threshold_defaults_when_missing_or_invalid() {
        assert_eq!(
            parse_lfm2_prefill_threshold(None),
            Lfm2PrefillConfig::DEFAULT_TOKEN_THRESHOLD
        );
        assert_eq!(
            parse_lfm2_prefill_threshold(Some("0")),
            Lfm2PrefillConfig::DEFAULT_TOKEN_THRESHOLD
        );
        assert_eq!(parse_lfm2_prefill_threshold(Some("96")), 96);
    }

    #[test]
    fn prefill_policy_forces_request_isolated_full_passes() {
        let config = Lfm2PrefillConfig {
            mode: Lfm2PrefillMode::Auto,
            token_prompt_threshold: 64,
        };
        assert_eq!(config.resolve(16), Lfm2PrefillExecution::Full);
        assert_eq!(config.resolve(64), Lfm2PrefillExecution::Full);
        assert_eq!(config.resolve(65), Lfm2PrefillExecution::Full);

        let explicitly_unsafe = Lfm2PrefillConfig {
            mode: Lfm2PrefillMode::Token,
            token_prompt_threshold: usize::MAX,
        };
        assert_eq!(explicitly_unsafe.resolve(1), Lfm2PrefillExecution::Full);
    }

    #[test]
    fn parse_default_system_policy_defaults_to_auto_for_unknown_values() {
        assert_eq!(
            parse_lfm2_default_system_policy(None),
            Lfm2DefaultSystemPolicy::Auto
        );
        assert_eq!(
            parse_lfm2_default_system_policy(Some("unsupported")),
            Lfm2DefaultSystemPolicy::Auto
        );
        assert_eq!(
            parse_lfm2_default_system_policy(Some("always")),
            Lfm2DefaultSystemPolicy::Always
        );
        assert_eq!(
            parse_lfm2_default_system_policy(Some("never")),
            Lfm2DefaultSystemPolicy::Never
        );
    }

    #[test]
    fn auto_default_system_policy_skips_single_turn_user_prompt() {
        let single_turn = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }];
        assert!(!should_prepend_default_system(
            &single_turn,
            Lfm2DefaultSystemPolicy::Auto
        ));

        let multi_turn = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "hi".to_string(),
            },
        ];
        assert!(should_prepend_default_system(
            &multi_turn,
            Lfm2DefaultSystemPolicy::Auto
        ));
    }

    #[test]
    fn parse_prompt_style_policy_defaults_to_standard_for_unknown_values() {
        assert_eq!(
            parse_lfm2_prompt_style_policy(None),
            Lfm2PromptStylePolicy::Standard
        );
        assert_eq!(
            parse_lfm2_prompt_style_policy(Some("unsupported")),
            Lfm2PromptStylePolicy::Standard
        );
        assert_eq!(
            parse_lfm2_prompt_style_policy(Some("standard")),
            Lfm2PromptStylePolicy::Standard
        );
        assert_eq!(
            parse_lfm2_prompt_style_policy(Some("aggressive")),
            Lfm2PromptStylePolicy::Aggressive
        );
    }

    #[test]
    fn aggressive_single_turn_prompt_only_applies_to_single_user_turns() {
        let single_user_turn = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }];
        assert!(!should_use_aggressive_single_turn_prompt(
            &single_user_turn,
            false,
            Lfm2PromptStylePolicy::Standard
        ));
        assert!(should_use_aggressive_single_turn_prompt(
            &single_user_turn,
            false,
            Lfm2PromptStylePolicy::Aggressive
        ));
        assert!(!should_use_aggressive_single_turn_prompt(
            &single_user_turn,
            true,
            Lfm2PromptStylePolicy::Aggressive
        ));

        let multi_turn = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "hi".to_string(),
            },
        ];
        assert!(!should_use_aggressive_single_turn_prompt(
            &multi_turn,
            false,
            Lfm2PromptStylePolicy::Aggressive
        ));
    }
}
