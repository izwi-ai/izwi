//! Native Qwen3.8 chat model loader and text generation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::info;

use crate::backends::device::cuda_compute_capability_supports_bf16;
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
use crate::models::shared::speculative_sampling::verify_speculative_prefix;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::cache::qwen38_composite_cache_contract_with_mtp;
use super::mtp::{Qwen38MtpDepth, Qwen38MtpHead, Qwen38MtpPairBatch};
use super::native::{ProjectionMaterialization, Qwen38NativeCheckpoint, QWEN38_27B_FP8_REVISION};
use super::telemetry::{
    record_cuda_kv_provider, record_mtp_policy, record_mtp_round, record_mtp_scalar_target_token,
    record_sampling_bounded_cuda, record_sampling_device_argmax, record_sampling_host,
    snapshot as qwen38_optimization_telemetry_snapshot,
};
use super::text::{Qwen38ProjectionRepresentation, Qwen38TextModel, Qwen38TextRuntimeState};

const IMAGE_PAD_PLACEHOLDER: &str = "<|image_pad|>";
const VIDEO_PAD_PLACEHOLDER: &str = "<|video_pad|>";
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;
const MAX_PREFILL_CHUNK_SIZE: usize = 2048;
const CUDA_BF16_KV_ENV: &str = "IZWI_QWEN38_CUDA_BF16_KV";
const MTP_ENABLED_ENV: &str = "IZWI_QWEN38_MTP";
const MTP_DRAFT_TOKENS_ENV: &str = "IZWI_QWEN38_MTP_DRAFT_TOKENS";
// Qwen's production serving guidance starts native MTP at depth one. Deeper
// recurrence increases proposal and rejected-prefix replay cost and remains an
// explicit, evidence-driven override.
const DEFAULT_MTP_DRAFT_TOKENS: usize = 1;
const MAX_MTP_DRAFT_TOKENS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen38MtpPolicy {
    Disabled,
    Enabled { draft_tokens: usize },
}

impl Qwen38MtpPolicy {
    fn from_process_environment() -> Result<Self> {
        Self::resolve(
            std::env::var(MTP_ENABLED_ENV).ok().as_deref(),
            std::env::var(MTP_DRAFT_TOKENS_ENV).ok().as_deref(),
        )
    }

    fn resolve(enabled: Option<&str>, draft_tokens: Option<&str>) -> Result<Self> {
        let enabled = match enabled.map(str::trim).map(str::to_ascii_lowercase) {
            None => true,
            Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on") => true,
            Some(value) if matches!(value.as_str(), "0" | "false" | "no" | "off") => false,
            Some(value) => {
                return Err(Error::ConfigError(format!(
                    "{MTP_ENABLED_ENV} must be a boolean, got `{value}`"
                )))
            }
        };
        if !enabled {
            return Ok(Self::Disabled);
        }
        let draft_tokens = match draft_tokens.map(str::trim) {
            None | Some("") => DEFAULT_MTP_DRAFT_TOKENS,
            Some(value) => value.parse::<usize>().map_err(|_| {
                Error::ConfigError(format!(
                    "{MTP_DRAFT_TOKENS_ENV} must be an integer in 1..={MAX_MTP_DRAFT_TOKENS}"
                ))
            })?,
        };
        if !(1..=MAX_MTP_DRAFT_TOKENS).contains(&draft_tokens) {
            return Err(Error::ConfigError(format!(
                "{MTP_DRAFT_TOKENS_ENV} must be in 1..={MAX_MTP_DRAFT_TOKENS}, got {draft_tokens}"
            )));
        }
        Ok(Self::Enabled { draft_tokens })
    }

    const fn draft_tokens(self) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::Enabled { draft_tokens } => Some(draft_tokens),
        }
    }

    const fn enabled(self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen38KvStorageProvider {
    CpuF32,
    MetalF16,
    CudaF16Fallback,
    CudaF16CapabilityFallback,
    CudaBf16Candidate,
}

impl Qwen38KvStorageProvider {
    fn select(
        backend: BackendKind,
        cuda_compute_capability: Option<(u32, u32)>,
        cuda_bf16_override: Option<&str>,
    ) -> Self {
        match backend {
            BackendKind::Cpu => Self::CpuF32,
            BackendKind::Metal => Self::MetalF16,
            BackendKind::Cuda
                if qwen38_candidate_enabled(cuda_bf16_override)
                    && qwen38_cuda_supports_bf16(cuda_compute_capability) =>
            {
                Self::CudaBf16Candidate
            }
            BackendKind::Cuda if qwen38_candidate_enabled(cuda_bf16_override) => {
                Self::CudaF16CapabilityFallback
            }
            BackendKind::Cuda => Self::CudaF16Fallback,
        }
    }

    const fn dtype(self) -> DType {
        match self {
            Self::CpuF32 => DType::F32,
            Self::MetalF16 | Self::CudaF16Fallback | Self::CudaF16CapabilityFallback => DType::F16,
            Self::CudaBf16Candidate => DType::BF16,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::CpuF32 => "portable_f32",
            Self::MetalF16 => "metal_f16",
            Self::CudaF16Fallback => "cuda_f16_fallback",
            Self::CudaF16CapabilityFallback => "cuda_f16_capability_fallback",
            Self::CudaBf16Candidate => "cuda_bf16_candidate",
        }
    }

    const fn fallback_reason(self) -> Option<&'static str> {
        match self {
            Self::CudaF16Fallback => Some(
                "CUDA BF16 KV is an unvalidated candidate; set IZWI_QWEN38_CUDA_BF16_KV=1 to test it",
            ),
            Self::CudaF16CapabilityFallback => Some(
                "CUDA BF16 KV requires an observed compute capability 8.0 or newer; using F16",
            ),
            _ => None,
        }
    }
}

fn qwen38_cuda_supports_bf16(compute_capability: Option<(u32, u32)>) -> bool {
    compute_capability.is_some_and(cuda_compute_capability_supports_bf16)
}

fn qwen38_candidate_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn qwen38_kv_storage_provider(
    backend: BackendKind,
    cuda_compute_capability: Option<(u32, u32)>,
) -> Qwen38KvStorageProvider {
    let requested = std::env::var(CUDA_BF16_KV_ENV).ok();
    Qwen38KvStorageProvider::select(backend, cuda_compute_capability, requested.as_deref())
}

fn qwen38_projection_materialization(device: &DeviceProfile) -> Result<ProjectionMaterialization> {
    qwen38_projection_materialization_policy(
        BackendKind::from(device.kind),
        device.capabilities.cuda_compute_capability,
        device.capabilities.supports_f16,
    )
}

fn qwen38_projection_materialization_policy(
    backend: BackendKind,
    cuda_compute_capability: Option<(u32, u32)>,
    supports_f16: bool,
) -> Result<ProjectionMaterialization> {
    match backend {
        BackendKind::Cpu => Ok(ProjectionMaterialization::F32),
        BackendKind::Metal => Ok(ProjectionMaterialization::F16),
        BackendKind::Cuda if qwen38_cuda_supports_bf16(cuda_compute_capability) => {
            Ok(ProjectionMaterialization::BF16)
        }
        BackendKind::Cuda if supports_f16 => Ok(ProjectionMaterialization::F16),
        BackendKind::Cuda => Err(Error::ModelLoadError(
            "Qwen3.8 CUDA requires F16 support when BF16 capability is unavailable".into(),
        )),
    }
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
    mtp_physical_kv: Option<PhysicalPagedKvCache>,
    mtp_anchor_hidden: Option<Tensor>,
    bootstrap_token: Option<u32>,
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

    pub(crate) fn uses_mtp_physical_kv(&self) -> bool {
        self.mtp_physical_kv.is_some()
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

    pub(crate) fn install_mtp_physical_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        let current = self.mtp_physical_kv.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 scalar state received an MTP cache reservation".into())
        })?;
        if current.arena().id() != cache.arena().id()
            || current.context_len() != cache.context_len()
        {
            return Err(Error::InferenceError(
                "Qwen3.8 MTP KV reservation does not continue the session".into(),
            ));
        }
        self.mtp_physical_kv = Some(cache);
        Ok(())
    }

    pub(crate) fn take_physical_write_completions(
        &mut self,
    ) -> Vec<std::sync::Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let mut completions = self.physical_kv.take_completed_writes();
        if let Some(mtp) = self.mtp_physical_kv.as_mut() {
            completions.extend(mtp.take_completed_writes());
        }
        completions
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
    pub input_tokens_committed: usize,
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
    cuda_compute_capability: Option<(u32, u32)>,
    kv_storage_provider: Qwen38KvStorageProvider,
    variant: ModelVariant,
    tokenizer: Qwen38Tokenizer,
    text_config: Qwen38TextConfig,
    text_model: Qwen38TextModel,
    mtp_policy: Qwen38MtpPolicy,
    mtp_head: Option<Qwen38MtpHead>,
}

fn qwen38_fp8_execution_mode(
    projection_representation: Qwen38ProjectionRepresentation,
) -> &'static str {
    match projection_representation {
        Qwen38ProjectionRepresentation::PackedQ8WithDenseF16
        | Qwen38ProjectionRepresentation::PackedQ8WithDenseBf16 => "q8_0_compressed_fallback",
        Qwen38ProjectionRepresentation::ExpandedF32
        | Qwen38ProjectionRepresentation::ExpandedF16
        | Qwen38ProjectionRepresentation::ExpandedBf16 => "expanded_fallback",
    }
}

fn qwen38_fp8_fallback_reason(
    projection_representation: Qwen38ProjectionRepresentation,
) -> &'static str {
    match projection_representation {
        Qwen38ProjectionRepresentation::PackedQ8WithDenseF16
        | Qwen38ProjectionRepresentation::PackedQ8WithDenseBf16 => {
            "CUDA applies weight_scale_inv during scale-aware FP8 dequantization and then requantizes projections to Q8_0 for Candle execution; native FP8 execution is not runtime-certified"
        }
        Qwen38ProjectionRepresentation::ExpandedF32
        | Qwen38ProjectionRepresentation::ExpandedF16
        | Qwen38ProjectionRepresentation::ExpandedBf16 => {
            "native block-FP8 GEMM is not runtime-certified; using the scale-exact expanded path"
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen38RepresentationDiagnostics {
    resident_representation: &'static str,
    fp8_execution_mode: &'static str,
    fallback_reason: &'static str,
    runtime_compute_dtype: &'static str,
}

fn qwen38_representation_diagnostics(
    projection_representation: Qwen38ProjectionRepresentation,
) -> Qwen38RepresentationDiagnostics {
    Qwen38RepresentationDiagnostics {
        resident_representation: projection_representation.as_str(),
        fp8_execution_mode: qwen38_fp8_execution_mode(projection_representation),
        fallback_reason: qwen38_fp8_fallback_reason(projection_representation),
        runtime_compute_dtype: projection_representation.compute_dtype(),
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
        let mtp_policy = Qwen38MtpPolicy::from_process_environment()?;
        let tokenizer = Qwen38Tokenizer::load_hf(model_dir)?;
        let text_config = checkpoint.config.text.clone();
        let projection_materialization = qwen38_projection_materialization(&device)?;
        let text_model = Qwen38TextModel::load_native(
            &checkpoint.tensors,
            &checkpoint.config,
            &device.device,
            projection_materialization,
        )?;
        let mtp_head = match mtp_policy {
            Qwen38MtpPolicy::Disabled => None,
            Qwen38MtpPolicy::Enabled { .. } => Some(Qwen38MtpHead::load_native(
                &checkpoint.tensors,
                &checkpoint.config,
                &checkpoint.mtp,
                &device.device,
                projection_materialization,
            )?),
        };
        record_mtp_policy(mtp_policy.enabled());
        let projection_representation = text_model.projection_representation();
        let device_kind = BackendKind::from(device.kind);
        let cuda_compute_capability = device.capabilities.cuda_compute_capability;
        let kv_storage_provider = qwen38_kv_storage_provider(device_kind, cuda_compute_capability);
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
            mtp_enabled = mtp_policy.enabled(),
            mtp_draft_tokens = mtp_policy.draft_tokens(),
            "Loaded native Qwen3.8 text checkpoint"
        );
        Ok(Self {
            device_kind,
            cuda_compute_capability,
            kv_storage_provider,
            variant,
            tokenizer,
            text_config,
            text_model,
            mtp_policy,
            mtp_head,
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
        qwen38_composite_cache_contract_with_mtp(
            &self.text_config,
            attention_dtype,
            preferred_page_tokens,
            self.mtp_policy.enabled(),
        )
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
        Some(
            qwen38_representation_diagnostics(self.text_model.projection_representation())
                .runtime_compute_dtype,
        )
    }

    pub fn runtime_diagnostics(&self) -> serde_json::Value {
        let projection_representation = self.text_model.projection_representation();
        let representation = qwen38_representation_diagnostics(projection_representation);
        serde_json::json!({
            "checkpoint_revision": QWEN38_27B_FP8_REVISION,
            "checkpoint_format": "safetensors_block_fp8",
            "resident_representation": representation.resident_representation,
            "runtime_compute_dtype": representation.runtime_compute_dtype,
            "fp8_execution_mode": representation.fp8_execution_mode,
            "fallback_reason": representation.fallback_reason,
            "optimization_evidence": {
                "scope": "qwen38_process_lifetime",
                "cuda_runtime_validated": false,
                "cuda_compute_capability": self.cuda_compute_capability.map(|(major, minor)| format!("{major}.{minor}")),
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
                    "quantized": false,
                    "physical_format": "dense",
                    "quantized_candidate_status": "unavailable_scale_contract_incomplete",
                },
                "mtp": {
                    "enabled": self.mtp_policy.enabled(),
                    "draft_tokens": self.mtp_policy.draft_tokens(),
                    "default_enabled": true,
                    "enabled_switch": MTP_ENABLED_ENV,
                    "depth_switch": MTP_DRAFT_TOKENS_ENV,
                    "implementation_status": "implemented_unvalidated",
                    "runtime_validated": false,
                    "performance_certified": false,
                    "scheduler_policy": "speculate_only_without_queue_pressure_or_concurrent_decode",
                    "execution_evidence": {
                        "speculative_round_counter": "mtp_rounds_total",
                        "scalar_target_counter": "mtp_scalar_target_tokens_total",
                        "accepted_draft_counter": "mtp_accepted_draft_tokens_total",
                        "replayed_target_counter": "mtp_target_replay_tokens_total",
                    },
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

    pub(crate) fn preferred_decode_tokens(&self) -> usize {
        self.mtp_policy
            .draft_tokens()
            .map_or(1, |draft_tokens| draft_tokens.saturating_add(1))
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
        mut mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<ChatDecodeState> {
        let prepared = resolve_prepared_prompt(prepared, || self.prepare_prompt(messages, config))?;
        if prepared.prompt_ids.is_empty()
            || cache.context_len() != 0
            || mtp_cache.as_ref().is_some_and(|mtp| mtp.context_len() != 0)
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 physical prefill requires a non-empty prompt and an empty reservation"
                    .into(),
            ));
        }
        let mut text_state = self.text_model.new_state();
        let (logits, target_hidden_states) = self
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
        let mut history_ids =
            initial_penalty_history(&prepared.prompt_ids, max_new_tokens, track_history);
        let mut rng = SimpleRng::new(config.seed);
        let (unconsumed_output, pending_token, bootstrap_token, mtp_anchor_hidden) =
            match (&self.mtp_head, mtp_cache.as_mut()) {
                (Some(head), Some(mtp_cache)) => {
                    let history: &[u32] = if track_history { &history_ids } else { &[] };
                    let anchor = sample_next_token(
                        &logits,
                        self.tokenizer.vocab_size,
                        config,
                        history,
                        &mut rng,
                    )?;
                    if track_history {
                        history_ids.push(anchor);
                    }
                    let prompt_embeddings =
                        self.text_model.embed_token_ids(&prepared.prompt_ids)?;
                    let anchor_embedding = self.text_model.embed_token_ids(&[anchor])?;
                    let pairs = Qwen38MtpPairBatch::shifted_prompt(
                        &prompt_embeddings,
                        &target_hidden_states,
                        &anchor_embedding,
                        &prepared.prompt_positions,
                    )?;
                    let hidden = head.forward_pairs(&pairs, mtp_cache)?;
                    let final_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
                    (None, Some(anchor), Some(anchor), Some(final_hidden))
                }
                (None, None) => (Some(logits), None, None, None),
                (Some(_), None) => {
                    return Err(Error::InferenceError(
                        "default-enabled Qwen3.8 MTP state has no managed MTP cache".into(),
                    ))
                }
                (None, Some(_)) => {
                    return Err(Error::InferenceError(
                        "disabled Qwen3.8 MTP received an unexpected managed MTP cache".into(),
                    ))
                }
            };
        Ok(ChatDecodeState {
            text_state,
            physical_kv: cache,
            mtp_physical_kv: mtp_cache,
            mtp_anchor_hidden,
            bootstrap_token,
            physical_tensor_sequence: None,
            unconsumed_output,
            pending_token,
            history_ids,
            decoder: IncrementalDecoder::new(true),
            tokens_generated: 0,
            track_history,
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            next_text_position: prepared.next_text_position,
            config: config.clone(),
            rng,
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        self.decode_quantum(state, 1)
    }

    pub(crate) fn decode_quantum(
        &self,
        state: &mut ChatDecodeState,
        input_budget: usize,
    ) -> Result<ChatDecodeStep> {
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

        if let Some(anchor) = state.bootstrap_token.take() {
            let delta = self.publish_token(state, anchor)?;
            return Ok(self.decode_step_result(state, delta, 0));
        }

        if self.mtp_head.is_some() {
            return self.decode_mtp_quantum(state, input_budget.max(1));
        }

        let mut input_tokens_committed = 0;
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
        let next = take_quantum_sample(
            &mut state.unconsumed_output,
            self.tokenizer.vocab_size,
            &state.config,
            history,
            &mut state.rng,
        )?;
        if state.track_history {
            state.history_ids.push(next);
        }
        state.pending_token = Some(next);
        let delta = self.publish_token(state, next)?;
        Ok(self.decode_step_result(state, delta, input_tokens_committed))
    }

    fn decode_step_result(
        &self,
        state: &ChatDecodeState,
        delta: String,
        input_tokens_committed: usize,
    ) -> ChatDecodeStep {
        ChatDecodeStep {
            delta,
            text: if state.finished {
                state.assembled.clone()
            } else {
                String::new()
            },
            tokens_generated: state.tokens_generated,
            input_tokens_committed,
            finished: state.finished,
        }
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

    fn decode_mtp_quantum(
        &self,
        state: &mut ChatDecodeState,
        input_budget: usize,
    ) -> Result<ChatDecodeStep> {
        let head = self
            .mtp_head
            .as_ref()
            .ok_or_else(|| Error::InferenceError("Qwen3.8 MTP decode has no loaded head".into()))?;
        let configured_depth = self.mtp_policy.draft_tokens().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 MTP decode has a disabled policy".into())
        })?;
        let mut delta = String::new();
        let mut committed = 0usize;

        while committed < input_budget && !state.finished {
            let remaining = input_budget - committed;
            if remaining == 1 {
                record_mtp_scalar_target_token();
                let pending = state.pending_token.ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 MTP scalar tail has no pending token".into())
                })?;
                let hidden = self.text_model.forward_token_id_hidden_at_physical(
                    pending,
                    [state.next_text_position; 3],
                    &mut state.text_state,
                    &mut state.physical_kv,
                )?;
                let logits = self
                    .text_model
                    .project_target_hidden_span(&hidden)?
                    .i((0, 0))?;
                let history = if state.track_history {
                    state.history_ids.as_slice()
                } else {
                    &[]
                };
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
                let mtp = state.mtp_physical_kv.as_mut().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 MTP scalar tail lost its cache".into())
                })?;
                let next_hidden = head.forward_step(
                    self.text_model.embed_token_ids(&[next])?,
                    hidden,
                    [state.next_text_position; 3],
                    mtp,
                )?;
                state.mtp_anchor_hidden = Some(next_hidden);
                state.pending_token = Some(next);
                state.next_text_position += 1;
                committed += 1;
                delta.push_str(&self.publish_token(state, next)?);
                continue;
            }

            let depth = configured_depth.min(remaining - 1);
            let depth = Qwen38MtpDepth::new(depth)?;
            let mtp = state
                .mtp_physical_kv
                .as_mut()
                .ok_or_else(|| Error::InferenceError("Qwen3.8 MTP decode lost its cache".into()))?;
            let mtp_checkpoint = mtp.logical_checkpoint();
            let anchor_hidden = state.mtp_anchor_hidden.as_ref().ok_or_else(|| {
                Error::InferenceError("Qwen3.8 MTP decode has no recurrent anchor".into())
            })?;
            let continuation_positions = (0..depth.get().saturating_sub(1))
                .map(|offset| [state.next_text_position + offset; 3])
                .collect::<Vec<_>>();
            let drafted = head.draft_recurrently_with_text(
                &self.text_model,
                anchor_hidden,
                depth,
                &continuation_positions,
                mtp,
                |_, logits| argmax_clamped(&logits.i((0, 0))?, self.tokenizer.vocab_size),
            );
            mtp.restore_logical_checkpoint(mtp_checkpoint)?;
            let drafted = drafted?;

            let pending = state.pending_token.ok_or_else(|| {
                Error::InferenceError("Qwen3.8 MTP verification has no pending token".into())
            })?;
            let mut target_inputs = Vec::with_capacity(drafted.token_ids.len() + 1);
            target_inputs.push(pending);
            target_inputs.extend_from_slice(&drafted.token_ids);
            let positions = (0..target_inputs.len())
                .map(|offset| [state.next_text_position + offset; 3])
                .collect::<Vec<_>>();
            let target_checkpoint = state.physical_kv.logical_checkpoint();
            let text_checkpoint = state.text_state.clone();
            let target_output = self
                .text_model
                .prefill_token_ids_with_hidden_physical(
                    &target_inputs,
                    &positions,
                    &mut state.text_state,
                    &mut state.physical_kv,
                    false,
                )?
                .ok_or_else(|| Error::InferenceError("empty MTP verification span".into()))?;
            let target_logits = self
                .text_model
                .project_target_hidden_span(&target_output.hidden_states)?;
            let mut host_rows = Vec::with_capacity(target_inputs.len());
            for row in 0..target_inputs.len() {
                let mut values = logits_to_vec(&target_logits.i((0, row))?)?;
                truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                host_rows.push(values);
            }
            let mut verification_history = if state.track_history {
                state.history_ids.clone()
            } else {
                Vec::new()
            };
            let verification = verify_speculative_prefix(
                &drafted.token_ids,
                &host_rows,
                &state.config,
                &mut verification_history,
                &mut state.rng,
            )?;

            let remaining_outputs = state.max_new_tokens.saturating_sub(state.tokens_generated);
            let mut kept = Vec::new();
            let mut non_stop = 0usize;
            for token in verification.emitted_tokens.iter().copied() {
                kept.push(token);
                if self.is_stop_token(token, &state.config) {
                    break;
                }
                non_stop += 1;
                if non_stop >= remaining_outputs {
                    break;
                }
            }
            let canonical_count = kept.len();
            let replay =
                canonical_count != target_inputs.len() || !verification.all_drafts_accepted();
            let canonical_hidden = if replay {
                state
                    .physical_kv
                    .restore_logical_checkpoint(target_checkpoint)?;
                state.text_state = text_checkpoint;
                self.text_model
                    .prefill_token_ids_with_hidden_physical(
                        &target_inputs[..canonical_count],
                        &positions[..canonical_count],
                        &mut state.text_state,
                        &mut state.physical_kv,
                        false,
                    )?
                    .ok_or_else(|| Error::InferenceError("empty MTP replay span".into()))?
                    .hidden_states
            } else {
                target_output.hidden_states
            };
            if state.track_history {
                state.history_ids = verification_history;
            }
            let canonical_pairs = Qwen38MtpPairBatch::new(
                self.text_model.embed_token_ids(&kept)?,
                canonical_hidden,
                positions[..canonical_count].to_vec(),
            )?;
            let canonical_mtp_hidden = head.forward_pairs(&canonical_pairs, mtp)?;
            state.mtp_anchor_hidden =
                Some(canonical_mtp_hidden.narrow(1, canonical_count - 1, 1)?);
            state.pending_token = kept.last().copied();
            state.next_text_position += canonical_count;
            committed += canonical_count;
            record_mtp_round(
                drafted.token_ids.len(),
                verification.accepted_draft_tokens,
                verification.emitted_bonus_token(),
                target_inputs.len(),
                if replay { canonical_count } else { 0 },
            );
            for token in kept {
                delta.push_str(&self.publish_token(state, token)?);
                if state.finished {
                    break;
                }
            }
        }

        Ok(self.decode_step_result(state, delta, committed))
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
    ) -> Result<Option<(Tensor, Tensor)>> {
        let mut logits = None;
        let mut hidden_chunks = Vec::new();
        let mut chunk_start = start;
        let chunk_size = qwen38_prefill_chunk_size();
        while chunk_start < end {
            let chunk_end = (chunk_start + chunk_size).min(end);
            let compute_logits = compute_final_logits && chunk_end == end;
            if let Some(output) = self.text_model.prefill_token_ids_with_hidden_physical(
                &prepared.prompt_ids[chunk_start..chunk_end],
                &prepared.prompt_positions[chunk_start..chunk_end],
                text_state,
                cache,
                compute_logits,
            )? {
                hidden_chunks.push(output.hidden_states);
                if let Some(chunk_logits) = output.logits {
                    logits = Some(chunk_logits);
                }
            }
            chunk_start = chunk_end;
        }
        match logits {
            Some(logits) => Ok(Some((logits, Tensor::cat(&hidden_chunks, 1)?))),
            None => Ok(None),
        }
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

#[derive(Clone)]
struct SimpleRng {
    state: u64,
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
    fn mtp_policy_is_enabled_at_depth_one_by_default() {
        let policy = Qwen38MtpPolicy::resolve(None, None).unwrap();
        assert_eq!(policy, Qwen38MtpPolicy::Enabled { draft_tokens: 1 });
        assert!(policy.enabled());
        assert_eq!(policy.draft_tokens(), Some(1));
    }

    #[test]
    fn mtp_policy_has_an_explicit_disable_and_bounded_depth_override() {
        for disabled in ["0", "false", "NO", " off "] {
            assert_eq!(
                Qwen38MtpPolicy::resolve(Some(disabled), Some("invalid")).unwrap(),
                Qwen38MtpPolicy::Disabled
            );
        }
        for depth in 1..=3 {
            assert_eq!(
                Qwen38MtpPolicy::resolve(Some("on"), Some(&depth.to_string())).unwrap(),
                Qwen38MtpPolicy::Enabled {
                    draft_tokens: depth
                }
            );
        }
        assert!(Qwen38MtpPolicy::resolve(Some("maybe"), None).is_err());
        assert!(Qwen38MtpPolicy::resolve(None, Some("0")).is_err());
        assert!(Qwen38MtpPolicy::resolve(None, Some("4")).is_err());
        assert!(Qwen38MtpPolicy::resolve(None, Some("many")).is_err());
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
            let provider =
                Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((8, 0)), disabled);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaF16Fallback);
            assert_eq!(provider.dtype(), DType::F16);
            assert!(provider.fallback_reason().is_some());
        }
        for enabled in [Some("1"), Some("true"), Some(" YES "), Some("on")] {
            let provider =
                Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((8, 0)), enabled);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaBf16Candidate);
            assert_eq!(provider.dtype(), DType::BF16);
            assert!(provider.fallback_reason().is_none());
        }
        assert_eq!(
            Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((8, 0)), Some("invalid")),
            Qwen38KvStorageProvider::CudaF16Fallback
        );
    }

    #[test]
    fn cuda_bf16_kv_candidate_requires_observed_ampere_or_newer_capability() {
        for compute_capability in [None, Some((7, 5))] {
            let provider =
                Qwen38KvStorageProvider::select(BackendKind::Cuda, compute_capability, Some("1"));
            assert_eq!(provider, Qwen38KvStorageProvider::CudaF16CapabilityFallback);
            assert_eq!(provider.dtype(), DType::F16);
            assert!(provider
                .fallback_reason()
                .expect("capability fallback reason")
                .contains("8.0 or newer"));
        }
        assert_eq!(
            Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((9, 0)), Some("1")),
            Qwen38KvStorageProvider::CudaBf16Candidate
        );
    }

    #[test]
    fn cuda_bf16_kv_switch_does_not_change_portable_storage_policy() {
        for candidate in [None, Some("0"), Some("1")] {
            let cpu = Qwen38KvStorageProvider::select(BackendKind::Cpu, None, candidate);
            let metal = Qwen38KvStorageProvider::select(BackendKind::Metal, None, candidate);
            assert_eq!(cpu, Qwen38KvStorageProvider::CpuF32);
            assert_eq!(cpu.dtype(), DType::F32);
            assert_eq!(metal, Qwen38KvStorageProvider::MetalF16);
            assert_eq!(metal.dtype(), DType::F16);
        }
    }

    #[test]
    fn qwen38_compute_materialization_is_capability_derived_and_portable() {
        assert_eq!(
            qwen38_projection_materialization_policy(BackendKind::Cpu, None, false).unwrap(),
            ProjectionMaterialization::F32
        );
        assert_eq!(
            qwen38_projection_materialization_policy(BackendKind::Metal, None, false).unwrap(),
            ProjectionMaterialization::F16
        );
        assert_eq!(
            qwen38_projection_materialization_policy(BackendKind::Cuda, Some((8, 0)), true)
                .unwrap(),
            ProjectionMaterialization::BF16
        );
        assert_eq!(
            qwen38_projection_materialization_policy(BackendKind::Cuda, Some((7, 5)), true)
                .unwrap(),
            ProjectionMaterialization::F16
        );
        assert_eq!(
            qwen38_projection_materialization_policy(BackendKind::Cuda, None, true).unwrap(),
            ProjectionMaterialization::F16
        );
        assert!(qwen38_projection_materialization_policy(BackendKind::Cuda, None, false).is_err());
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
        let f16_cuda_representation = Qwen38ProjectionRepresentation::PackedQ8WithDenseF16;
        assert_eq!(
            f16_cuda_representation.as_str(),
            "q8_0_requantized_projections_with_dense_f16"
        );
        assert_eq!(f16_cuda_representation.compute_dtype(), "f16");
        assert_eq!(
            qwen38_fp8_execution_mode(f16_cuda_representation),
            "q8_0_compressed_fallback"
        );
        let f16_diagnostics = qwen38_representation_diagnostics(f16_cuda_representation);
        assert_eq!(
            f16_diagnostics.resident_representation,
            "q8_0_requantized_projections_with_dense_f16"
        );
        assert_eq!(f16_diagnostics.runtime_compute_dtype, "f16");
        let bf16_diagnostics = qwen38_representation_diagnostics(cuda_representation);
        assert_eq!(
            bf16_diagnostics.resident_representation,
            "q8_0_requantized_projections_with_dense_bf16"
        );
        assert_eq!(bf16_diagnostics.runtime_compute_dtype, "bf16");

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
