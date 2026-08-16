//! Native Qwen3.8 chat model loader and text generation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::info;

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
use crate::models::shared::chat::{
    ChatGenerationConfig, ChatMessage, ChatReasoningEffort, ChatRole,
};
use crate::models::shared::sampling::{
    bounded_cuda_sampling_candidates, device_candidates_cover_top_p, sample_device_candidates,
};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::cache::qwen38_composite_cache_contract;
use super::native::{Qwen38NativeCheckpoint, QWEN38_27B_FP8_REVISION};
use super::telemetry::{
    record_cuda_kv_provider, record_sampling_bounded_cuda, record_sampling_device_argmax,
    record_sampling_host, snapshot as qwen38_optimization_telemetry_snapshot,
};
use super::text::{Qwen38ProjectionRepresentation, Qwen38TextModel, Qwen38TextRuntimeState};

const IMAGE_PAD_PLACEHOLDER: &str = "<|image_pad|>";
const VIDEO_PAD_PLACEHOLDER: &str = "<|video_pad|>";
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;
const MAX_PREFILL_CHUNK_SIZE: usize = 2048;
const CUDA_BF16_KV_ENV: &str = "IZWI_QWEN38_CUDA_BF16_KV";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen38KvStorageProvider {
    CpuF32,
    MetalF16,
    CudaF16Fallback,
    CudaBf16Candidate,
}

impl Qwen38KvStorageProvider {
    fn select(backend: BackendKind, cuda_bf16_override: Option<&str>) -> Self {
        match backend {
            BackendKind::Cpu => Self::CpuF32,
            BackendKind::Metal => Self::MetalF16,
            BackendKind::Cuda if qwen38_candidate_enabled(cuda_bf16_override) => {
                Self::CudaBf16Candidate
            }
            BackendKind::Cuda => Self::CudaF16Fallback,
        }
    }

    const fn dtype(self) -> DType {
        match self {
            Self::CpuF32 => DType::F32,
            Self::MetalF16 | Self::CudaF16Fallback => DType::F16,
            Self::CudaBf16Candidate => DType::BF16,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::CpuF32 => "portable_f32",
            Self::MetalF16 => "metal_f16",
            Self::CudaF16Fallback => "cuda_f16_fallback",
            Self::CudaBf16Candidate => "cuda_bf16_candidate",
        }
    }

    const fn fallback_reason(self) -> Option<&'static str> {
        match self {
            Self::CudaF16Fallback => Some(
                "CUDA BF16 KV is an unvalidated candidate; set IZWI_QWEN38_CUDA_BF16_KV=1 to test it",
            ),
            _ => None,
        }
    }
}

fn qwen38_candidate_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn qwen38_kv_storage_provider(backend: BackendKind) -> Qwen38KvStorageProvider {
    let requested = std::env::var(CUDA_BF16_KV_ENV).ok();
    Qwen38KvStorageProvider::select(backend, requested.as_deref())
}

fn qwen38_prefill_chunk_size() -> usize {
    std::env::var("IZWI_QWEN38_PREFILL_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE)
        .min(MAX_PREFILL_CHUNK_SIZE)
}

/// Fully prepared Qwen3.8 text prefill input. The runtime carries this exact
/// artifact into the executor so tokenization and position construction happen
/// once.
#[derive(Debug, Clone)]
pub struct Qwen38PreparedPrompt {
    prompt_ids: Vec<u32>,
    prompt_positions: Vec<[usize; 3]>,
    next_text_position: usize,
}

impl Qwen38PreparedPrompt {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    pub(crate) fn prompt_positions(&self) -> &[[usize; 3]] {
        &self.prompt_positions
    }
}

fn resolve_prepared_prompt<F>(
    prepared: Option<&Qwen38PreparedPrompt>,
    prepare: F,
) -> Result<Qwen38PreparedPrompt>
where
    F: FnOnce() -> Result<Qwen38PreparedPrompt>,
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

pub struct ChatDecodeState {
    text_state: Qwen38TextRuntimeState,
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
    config: ChatGenerationConfig,
    rng: SimpleRng,
}

impl ChatDecodeState {
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
                "Qwen3.8 physical KV reservation does not continue the session".into(),
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

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .physical_tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(Error::InferenceError(
                "Qwen3.8 tensor-state sequence identity changed".into(),
            ));
        }
        self.physical_tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self.physical_tensor_sequence.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 physical state has no tensor sequence".into())
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
    pub finished: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen38TextConfig {
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
    im_end: u32,
    eos: u32,
    eos_alt: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct Qwen38Tokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    literal_special_tokens: Vec<(String, u32)>,
    chat_template: String,
    default_enable_thinking: bool,
}

impl Qwen38Tokenizer {
    fn load_hf(model_dir: &Path) -> Result<Self> {
        let config = load_tokenizer_config_file(model_dir)?.ok_or_else(|| {
            Error::TokenizationError("Qwen3.8 tokenizer_config.json is missing".into())
        })?;
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, Some(248_320))?;
        let mut token_to_id = HashMap::new();
        for (id, entry) in &config.added_tokens_decoder {
            if let Ok(id) = id.parse::<u32>() {
                token_to_id.insert(entry.content.clone(), id);
            }
        }
        let id_for = |token: &str| {
            token_to_id
                .get(token)
                .copied()
                .or_else(|| inner.token_to_id(token))
        };
        let required = |token: &str| {
            id_for(token).ok_or_else(|| {
                Error::TokenizationError(format!("Missing required Qwen3.8 token {token}"))
            })
        };
        required("<|im_start|>")?;
        let im_end = required("<|im_end|>")?;
        required(IMAGE_PAD_PLACEHOLDER)?;
        required(VIDEO_PAD_PLACEHOLDER)?;
        let eos_alt = id_for("<|endoftext|>");
        let eos = config
            .eos_token
            .as_deref()
            .and_then(id_for)
            .or(eos_alt)
            .unwrap_or(im_end);
        let chat_template = config.chat_template.clone().ok_or_else(|| {
            Error::TokenizationError("Qwen3.8 tokenizer config has no chat_template".into())
        })?;
        let mut literal_special_tokens = token_to_id.into_iter().collect::<Vec<_>>();
        literal_special_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });
        Ok(Self {
            vocab_size: inner.vocab_size(),
            inner,
            specials: SpecialTokenIds {
                im_end,
                eos,
                eos_alt,
            },
            literal_special_tokens,
            default_enable_thinking: true,
            chat_template,
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

pub struct Qwen38ChatModel {
    device_kind: BackendKind,
    kv_storage_provider: Qwen38KvStorageProvider,
    variant: ModelVariant,
    tokenizer: Qwen38Tokenizer,
    text_config: Qwen38TextConfig,
    text_model: Qwen38TextModel,
}

fn qwen38_fp8_execution_mode(
    projection_representation: Qwen38ProjectionRepresentation,
) -> &'static str {
    match projection_representation {
        Qwen38ProjectionRepresentation::PackedQ8WithDenseBf16 => "q8_0_compressed_fallback",
        Qwen38ProjectionRepresentation::ExpandedF32
        | Qwen38ProjectionRepresentation::ExpandedF16
        | Qwen38ProjectionRepresentation::ExpandedBf16 => "expanded_fallback",
    }
}

fn qwen38_fp8_fallback_reason(
    projection_representation: Qwen38ProjectionRepresentation,
) -> &'static str {
    match projection_representation {
        Qwen38ProjectionRepresentation::PackedQ8WithDenseBf16 => {
            "CUDA applies weight_scale_inv during scale-aware FP8 dequantization and then requantizes projections to Q8_0 for Candle execution; native FP8 execution is not runtime-certified"
        }
        Qwen38ProjectionRepresentation::ExpandedF32
        | Qwen38ProjectionRepresentation::ExpandedF16
        | Qwen38ProjectionRepresentation::ExpandedBf16 => {
            "native block-FP8 GEMM is not runtime-certified; using the scale-exact expanded path"
        }
    }
}

impl InferenceStateContractProvider for Qwen38ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        Ok(InferenceStateCapability::Managed(
            self.managed_composite_cache_contract(
                self.kv_storage_provider.dtype(),
                default_kv_page_size(),
            )?,
        ))
    }
}

impl Qwen38ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::Qwen3827BFp8 {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Qwen3.8 chat variant: {variant}"
            )));
        }
        let checkpoint = Qwen38NativeCheckpoint::open(model_dir)?;
        let tokenizer = Qwen38Tokenizer::load_hf(model_dir)?;
        let text_config = checkpoint.config.text.clone();
        let text_model =
            Qwen38TextModel::load_native(&checkpoint.tensors, &checkpoint.config, &device.device)?;
        let projection_representation = text_model.projection_representation();
        let device_kind = BackendKind::from(device.kind);
        let kv_storage_provider = qwen38_kv_storage_provider(device_kind);
        if device_kind == BackendKind::Cuda {
            record_cuda_kv_provider(
                kv_storage_provider == Qwen38KvStorageProvider::CudaBf16Candidate,
            );
        }
        info!(
            variant = %variant,
            backend = ?device.kind,
            kv_storage_provider = kv_storage_provider.as_str(),
            revision = QWEN38_27B_FP8_REVISION,
            tensors = checkpoint.tensors.tensor_count(),
            resident_representation = projection_representation.as_str(),
            fp8_execution_mode = qwen38_fp8_execution_mode(projection_representation),
            "Loaded native Qwen3.8 text checkpoint"
        );
        Ok(Self {
            device_kind,
            kv_storage_provider,
            variant,
            tokenizer,
            text_config,
            text_model,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn text_config(&self) -> &Qwen38TextConfig {
        &self.text_config
    }

    pub fn max_context_tokens(&self) -> Result<usize> {
        if self.text_config.context_length == 0 {
            return Err(Error::ModelLoadError(
                "Qwen3.8 checkpoint has a zero context length".into(),
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
        qwen38_composite_cache_contract(&self.text_config, attention_dtype, preferred_page_tokens)
    }

    pub fn chat_template(&self) -> &str {
        &self.tokenizer.chat_template
    }

    pub fn default_enable_thinking(&self) -> bool {
        self.tokenizer.default_enable_thinking
    }

    pub fn checkpoint_revision(&self) -> Option<&str> {
        Some(QWEN38_27B_FP8_REVISION)
    }

    pub fn checkpoint_format(&self) -> &'static str {
        "safetensors_block_fp8"
    }

    pub fn runtime_compute_dtype(&self) -> Option<&'static str> {
        Some(match self.device_kind {
            BackendKind::Cpu => "f32",
            BackendKind::Metal => "f16",
            BackendKind::Cuda => "bf16",
        })
    }

    pub fn runtime_diagnostics(&self) -> serde_json::Value {
        let projection_representation = self.text_model.projection_representation();
        serde_json::json!({
            "checkpoint_revision": QWEN38_27B_FP8_REVISION,
            "checkpoint_format": "safetensors_block_fp8",
            "resident_representation": projection_representation.as_str(),
            "fp8_execution_mode": qwen38_fp8_execution_mode(projection_representation),
            "fallback_reason": qwen38_fp8_fallback_reason(projection_representation),
            "optimization_evidence": {
                "scope": "qwen38_process_lifetime",
                "cuda_runtime_validated": false,
                "counters": qwen38_optimization_telemetry_snapshot(),
                "managed_kv_counters_source": "runtime_metrics.kv_cache.models[].arenas[].operations",
                "managed_kv_counter_coverage": [
                    "allocation",
                    "workspace",
                    "host_synchronization",
                    "attention_provider",
                    "cuda_graph"
                ],
                "cuda_kv_storage": {
                    "candidate_switch": CUDA_BF16_KV_ENV,
                    "selected_provider": self.kv_storage_provider.as_str(),
                    "storage_dtype": format!("{:?}", self.kv_storage_provider.dtype()).to_ascii_lowercase(),
                    "fallback_reason": self.kv_storage_provider.fallback_reason(),
                    "runtime_validated": false,
                },
            },
            "vision_enabled": false,
        })
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
    ) -> Result<Qwen38PreparedPrompt> {
        self.prepare_prompt(messages, config)
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    pub fn device_kind(&self) -> BackendKind {
        self.device_kind
    }

    pub(crate) fn start_decode_state_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen38PreparedPrompt>,
        mut cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        let prepared = resolve_prepared_prompt(prepared, || self.prepare_prompt(messages, config))?;
        if prepared.prompt_ids.is_empty() || cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3.8 physical prefill requires a non-empty prompt and an empty reservation"
                    .into(),
            ));
        }
        let mut text_state = self.text_model.new_state();
        let logits = self
            .prefill_text_range_physical(
                &prepared,
                &mut text_state,
                &mut cache,
                0,
                prepared.prompt_ids.len(),
                true,
            )?
            .ok_or_else(|| {
                Error::InferenceError("Qwen3.8 physical prefill produced no logits".into())
            })?;
        let track_history =
            config.repetition_penalty > 1.0 || config.presence_penalty.abs() > f32::EPSILON;
        Ok(ChatDecodeState {
            text_state,
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_output: Some(logits),
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
            config: config.clone(),
            rng: SimpleRng::new(config.seed),
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.finished || state.tokens_generated >= state.max_new_tokens {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.clone(),
                tokens_generated: state.tokens_generated,
                finished: true,
            });
        }

        if let Some(pending) = state.pending_token.take() {
            state.unconsumed_output = Some(self.text_model.forward_token_id_at_physical(
                pending,
                [state.next_text_position; 3],
                &mut state.text_state,
                &mut state.physical_kv,
            )?);
            state.next_text_position += 1;
        }

        let history: &[u32] = if state.track_history {
            &state.history_ids
        } else {
            &[]
        };
        let next = take_quantum_sample(
            &mut state.unconsumed_output,
            self.tokenizer.vocab_size,
            &state.config,
            history,
            &mut state.rng,
        )?;
        if self.is_stop_token(next, &state.config) {
            state.finished = true;
            let delta = self.tokenizer.finish_decode(&mut state.decoder)?;
            state.assembled.push_str(&delta);
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.clone(),
                tokens_generated: state.tokens_generated,
                finished: true,
            });
        }

        let mut delta = self
            .tokenizer
            .decode_token_delta(&mut state.decoder, next)?;
        if state.track_history {
            state.history_ids.push(next);
        }
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
            finished: state.finished,
        })
    }

    fn is_stop_token(&self, token_id: u32, config: &ChatGenerationConfig) -> bool {
        token_id == self.tokenizer.specials.im_end
            || token_id == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(token_id)
            || config.stop_token_ids.contains(&token_id)
    }

    fn prefill_text_range_physical(
        &self,
        prepared: &Qwen38PreparedPrompt,
        text_state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        start: usize,
        end: usize,
        compute_final_logits: bool,
    ) -> Result<Option<Tensor>> {
        let mut logits = None;
        let mut chunk_start = start;
        let chunk_size = qwen38_prefill_chunk_size();
        while chunk_start < end {
            let chunk_end = (chunk_start + chunk_size).min(end);
            let compute_logits = compute_final_logits && chunk_end == end;
            if let Some(chunk_logits) = self.text_model.prefill_token_ids_physical(
                &prepared.prompt_ids[chunk_start..chunk_end],
                &prepared.prompt_positions[chunk_start..chunk_end],
                text_state,
                cache,
                compute_logits,
            )? {
                logits = Some(chunk_logits);
            }
            chunk_start = chunk_end;
        }
        Ok(logits)
    }

    fn prepare_prompt(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Qwen38PreparedPrompt> {
        let prompt = render_prompt(messages, config, self.default_enable_thinking())?;
        let image_placeholders = prompt.matches(IMAGE_PAD_PLACEHOLDER).count();
        let video_placeholders = prompt.matches(VIDEO_PAD_PLACEHOLDER).count();
        if !config.request.media_inputs.is_empty()
            || image_placeholders > 0
            || video_placeholders > 0
        {
            return Err(Error::InvalidInput(
                "Qwen3.8-27B-FP8 is text-only; image and video inputs are not enabled".into(),
            ));
        }
        let prompt_ids = self.tokenizer.encode_text(&prompt)?;
        let prompt_positions = build_text_positions(prompt_ids.len());
        Ok(Qwen38PreparedPrompt {
            next_text_position: prompt_positions.len(),
            prompt_ids,
            prompt_positions,
        })
    }
}

fn build_text_positions(token_count: usize) -> Vec<[usize; 3]> {
    (0..token_count).map(|idx| [idx; 3]).collect()
}

fn render_prompt(
    messages: &[ChatMessage],
    config: &ChatGenerationConfig,
    default_enable_thinking: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.8 chat prompt requires at least one message".to_string(),
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
    let enable_thinking = config
        .request
        .enable_thinking
        .unwrap_or(default_enable_thinking);
    let reasoning_instructions = if enable_thinking {
        match config.request.reasoning_effort.unwrap_or_default() {
            ChatReasoningEffort::Xhigh => Some(QWEN38_XHIGH_REASONING_INSTRUCTIONS),
            ChatReasoningEffort::Medium => None,
            ChatReasoningEffort::Low => Some(QWEN38_LOW_REASONING_INSTRUCTIONS),
        }
    } else {
        None
    };

    if !config.request.tools.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        if let Some(instructions) = reasoning_instructions {
            prompt.push_str(instructions);
            prompt.push_str("\n\n");
        }
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
    } else if leading_system && (!system_content.is_empty() || reasoning_instructions.is_some()) {
        prompt.push_str("<|im_start|>system\n");
        if let Some(instructions) = reasoning_instructions {
            prompt.push_str(instructions);
            if !system_content.is_empty() {
                prompt.push_str("\n\n");
            }
        }
        prompt.push_str(system_content);
        prompt.push_str("<|im_end|>\n");
    } else if let Some(instructions) = reasoning_instructions {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(instructions);
        prompt.push_str("<|im_end|>\n");
    }

    let last_query_index = last_query_index(messages)?;
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System {
            if index != 0 {
                return Err(Error::InvalidInput(
                    "Qwen3.8 system message must be the first message".to_string(),
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
                let preserve_thinking = config.request.preserve_thinking.unwrap_or(true);
                if preserve_thinking || index > last_query_index {
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
    if enable_thinking {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    Ok(prompt)
}

const QWEN38_XHIGH_REASONING_INSTRUCTIONS: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const QWEN38_LOW_REASONING_INSTRUCTIONS: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

fn last_query_index(messages: &[ChatMessage]) -> Result<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            (message.role == ChatRole::User && !is_tool_response(&message.content)).then_some(index)
        })
        .ok_or_else(|| {
            Error::InvalidInput("Qwen3.8 prompt requires at least one user query".to_string())
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

fn load_tokenizer_config_file(model_dir: &Path) -> Result<Option<TokenizerConfigFile>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_str = fs::read_to_string(config_path)?;
    let config: TokenizerConfigFile = serde_json::from_str(&config_str)?;
    Ok(Some(config))
}

fn take_quantum_sample(
    output: &mut Option<Tensor>,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let output = output.take().ok_or_else(|| {
        Error::InferenceError("Qwen3.8 decode quantum has no unconsumed model output".to_string())
    })?;
    sample_next_token(&output, vocab_size, config, history, rng)
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
            "Qwen3.8 sampler received vocab_size=0".to_string(),
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
        if logits.device().is_cuda() {
            record_sampling_device_argmax();
        } else {
            record_sampling_host();
        }
        return argmax_clamped(logits, vocab_size);
    }

    let cuda_sampling_attempted = logits.device().is_cuda();
    if let Some(candidates) = bounded_cuda_sampling_candidates(
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
                record_sampling_bounded_cuda(true);
                return Ok(sampled);
            }
        }
    }
    if cuda_sampling_attempted {
        record_sampling_bounded_cuda(false);
    }
    record_sampling_host();

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
        .ok_or_else(|| Error::InferenceError("Failed to sample Qwen3.8 token".to_string()))
}

fn logits_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.8 logits shape for sampling: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.8 logits rank for sampling: {rank}"
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
        "No valid Qwen3.8 logits to sample: 0 finite, {nan} NaN, \
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

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.8 logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.8 logits rank for argmax: {rank}"
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
            "Qwen3.8 argmax received vocab_size=0".to_string(),
        ));
    }

    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected Qwen3.8 logits shape for argmax: {:?}",
                    logits.shape().dims()
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Qwen3.8 logits rank for argmax: {rank}"
            )))
        }
    };

    let cols = logits.dim(0)?;
    let clamped = if vocab_size < cols {
        logits.narrow(0, 0, vocab_size)?
    } else {
        logits
    };
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
    if clamped.device().is_cuda() {
        record_sampling_host();
    }
    argmax_values(&values)
}

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

#[cfg(test)]
mod tests {
    use crate::models::shared::chat::ChatRequestConfig;

    use super::*;

    fn history_messages() -> Vec<ChatMessage> {
        vec![
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
        ]
    }

    #[test]
    fn prepared_prompt_exposes_ids_and_reuses_the_prepared_value() {
        let prepared = Qwen38PreparedPrompt {
            prompt_ids: vec![1, 2, 3],
            prompt_positions: build_text_positions(3),
            next_text_position: 3,
        };
        assert_eq!(prepared.prompt_ids(), &[1, 2, 3]);
        assert_eq!(
            prepared.prompt_positions(),
            &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        );
        let reused = resolve_prepared_prompt(Some(&prepared), || {
            Err(Error::InferenceError(
                "prepared prompt should skip reconstruction".into(),
            ))
        })
        .unwrap();
        assert_eq!(reused.prompt_ids(), prepared.prompt_ids());
    }

    #[test]
    fn cuda_bf16_kv_candidate_is_explicit_and_fail_closed() {
        for disabled in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("no"),
            Some("off"),
        ] {
            let provider = Qwen38KvStorageProvider::select(BackendKind::Cuda, disabled);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaF16Fallback);
            assert_eq!(provider.dtype(), DType::F16);
            assert!(provider.fallback_reason().is_some());
        }
        for enabled in [Some("1"), Some("true"), Some(" YES "), Some("on")] {
            let provider = Qwen38KvStorageProvider::select(BackendKind::Cuda, enabled);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaBf16Candidate);
            assert_eq!(provider.dtype(), DType::BF16);
            assert!(provider.fallback_reason().is_none());
        }
        assert_eq!(
            Qwen38KvStorageProvider::select(BackendKind::Cuda, Some("invalid")),
            Qwen38KvStorageProvider::CudaF16Fallback
        );
    }

    #[test]
    fn cuda_bf16_kv_switch_does_not_change_portable_storage_policy() {
        for candidate in [None, Some("0"), Some("1")] {
            let cpu = Qwen38KvStorageProvider::select(BackendKind::Cpu, candidate);
            let metal = Qwen38KvStorageProvider::select(BackendKind::Metal, candidate);
            assert_eq!(cpu, Qwen38KvStorageProvider::CpuF32);
            assert_eq!(cpu.dtype(), DType::F32);
            assert_eq!(metal, Qwen38KvStorageProvider::MetalF16);
            assert_eq!(metal.dtype(), DType::F16);
        }
    }

    #[test]
    fn cuda_diagnostics_identify_q8_0_fallback_without_changing_portable_modes() {
        let cuda_representation = Qwen38ProjectionRepresentation::PackedQ8WithDenseBf16;
        assert_eq!(
            cuda_representation.as_str(),
            "q8_0_requantized_projections_with_dense_bf16"
        );
        assert_eq!(
            qwen38_fp8_execution_mode(cuda_representation),
            "q8_0_compressed_fallback"
        );
        assert!(qwen38_fp8_fallback_reason(cuda_representation).contains("weight_scale_inv"));
        assert!(qwen38_fp8_fallback_reason(cuda_representation)
            .contains("native FP8 execution is not runtime-certified"));

        assert_eq!(
            Qwen38ProjectionRepresentation::ExpandedF32.as_str(),
            "expanded_f32"
        );
        assert_eq!(
            Qwen38ProjectionRepresentation::ExpandedF16.as_str(),
            "expanded_f16"
        );
        assert_eq!(
            qwen38_fp8_execution_mode(Qwen38ProjectionRepresentation::ExpandedF32),
            "expanded_fallback"
        );
        assert_eq!(
            qwen38_fp8_execution_mode(Qwen38ProjectionRepresentation::ExpandedF16),
            "expanded_fallback"
        );
        assert_eq!(
            qwen38_fp8_execution_mode(Qwen38ProjectionRepresentation::ExpandedBf16),
            "expanded_fallback"
        );
        let expanded_reason =
            "native block-FP8 GEMM is not runtime-certified; using the scale-exact expanded path";
        assert_eq!(
            qwen38_fp8_fallback_reason(Qwen38ProjectionRepresentation::ExpandedF32),
            expanded_reason
        );
        assert_eq!(
            qwen38_fp8_fallback_reason(Qwen38ProjectionRepresentation::ExpandedF16),
            expanded_reason
        );
    }

    #[test]
    fn defaults_to_xhigh_thinking_and_preserved_history() {
        let prompt = render_prompt(&history_messages(), &ChatGenerationConfig::default(), true)
            .expect("render Qwen3.8 prompt");

        assert!(prompt.starts_with(&format!(
            "<|im_start|>system\n{QWEN38_XHIGH_REASONING_INSTRUCTIONS}<|im_end|>\n"
        )));
        assert!(prompt.contains(
            "<|im_start|>assistant\n<think>\nreasoning first\n</think>\n\nFinal answer<|im_end|>\n"
        ));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn low_effort_with_tools_uses_qwen_coder_xml_contract() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                reasoning_effort: Some(ChatReasoningEffort::Low),
                tools: vec![serde_json::json!({
                    "type": "function",
                    "function": {"name": "lookup"}
                })],
                ..Default::default()
            },
            ..Default::default()
        };
        let prompt = render_prompt(
            &[ChatMessage {
                role: ChatRole::User,
                content: "Hi".to_string(),
            }],
            &config,
            true,
        )
        .expect("render Qwen3.8 tool prompt");

        assert!(prompt.starts_with(&format!(
            "<|im_start|>system\n{QWEN38_LOW_REASONING_INSTRUCTIONS}\n\n# Tools"
        )));
        assert!(prompt.contains("<tool_call>\n<function=example_function_name>"));
        assert!(prompt.contains("<tools>"));
    }

    #[test]
    fn disabled_thinking_emits_empty_block_without_reasoning_instruction() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                enable_thinking: Some(false),
                reasoning_effort: Some(ChatReasoningEffort::Low),
                ..Default::default()
            },
            ..Default::default()
        };
        let prompt = render_prompt(
            &[ChatMessage {
                role: ChatRole::User,
                content: "Hi".to_string(),
            }],
            &config,
            true,
        )
        .expect("render non-thinking Qwen3.8 prompt");

        assert!(!prompt.contains(QWEN38_LOW_REASONING_INSTRUCTIONS));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn preserve_thinking_can_be_disabled() {
        let config = ChatGenerationConfig {
            request: ChatRequestConfig {
                preserve_thinking: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        let prompt = render_prompt(&history_messages(), &config, true).unwrap();
        assert!(!prompt.contains("reasoning first"));
        assert!(prompt.contains("Final answer<|im_end|>"));
    }
}
