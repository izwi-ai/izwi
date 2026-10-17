//! Native Qwen3.8 chat model loader and text generation.

#[cfg(test)]
mod attach_tests;
mod device_sampling;
#[cfg(test)]
mod history_attach_tests;
mod pipeline;
#[cfg(test)]
mod pipeline_tests;
#[cfg(test)]
pub(crate) mod recovery_tests;
mod timing;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_core::{DType, IndexOp, Tensor, D};
use serde::Deserialize;
use tracing::{info, warn};

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
    bounded_device_sampling_candidates, device_candidates_cover_top_p, sample_device_candidates,
};
use crate::models::shared::speculative_sampling::{
    propose_speculative_draft, verify_greedy_token_prefix, verify_speculative_prefix,
    verify_speculative_proposals,
};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::cache::qwen38_composite_cache_contract_with_mtp;
use super::mtp::{AdaptiveMtp, Qwen38MtpDepth, Qwen38MtpHead, Qwen38MtpPairBatch};
use super::native::{ProjectionMaterialization, Qwen38NativeCheckpoint, QWEN38_27B_FP8_REVISION};
use super::residency::Qwen38ResidencyPolicy;
use super::telemetry::{
    record_cuda_kv_provider, record_mtp_nonfinite_draft_fallback, record_mtp_policy,
    record_mtp_round, record_mtp_round_timing, record_mtp_scalar_target_token,
    record_sampling_bounded_cuda, record_sampling_device_argmax, record_sampling_host,
    snapshot as qwen38_optimization_telemetry_snapshot,
};
use super::text::{Qwen38TextModel, Qwen38TextRuntimeState};

const IMAGE_PAD_PLACEHOLDER: &str = "<|image_pad|>";
const VIDEO_PAD_PLACEHOLDER: &str = "<|video_pad|>";
const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;
/// Prefill chunk on CUDA once the dequantized prefill GEMM runs. Its weight
/// dequantization (~40 ms per pass over the 27B trunk, estimated) is paid
/// once per chunk, so 1,024 tokens amortize it over four times the span of
/// 256 while one chunk stays ~130 ms (estimate). Wider spans reach the model
/// only for an isolated request: the scheduler grants the preferred span
/// when nothing else is decoding or waiting (`prefill_chunk_for`), so
/// concurrent decode keeps the engine's 96-192-token spans. SGLang caps
/// hybrid GDN prefill at 2,048 because 8k chunks stall decode by ~600 ms.
const CUDA_GEMM_PREFILL_CHUNK_SIZE: usize = 1024;
const MAX_PREFILL_CHUNK_SIZE: usize = 2048;
const PREFILL_CHUNK_ENV: &str = "IZWI_QWEN38_PREFILL_CHUNK_SIZE";
const CUDA_BF16_KV_ENV: &str = "IZWI_QWEN38_CUDA_BF16_KV";
const MTP_ENABLED_ENV: &str = "IZWI_QWEN38_MTP";
const MTP_DRAFT_TOKENS_ENV: &str = "IZWI_QWEN38_MTP_DRAFT_TOKENS";
// Qwen's production serving guidance starts native MTP at depth one. Deeper
// recurrence increases proposal and rejected-prefix replay cost and remains an
// explicit, evidence-driven override.
const DEFAULT_MTP_ENABLED: bool = true;
const DEFAULT_MTP_DRAFT_TOKENS: usize = 1;
const MAX_MTP_DRAFT_TOKENS: usize = 3;
/// Decode tokens per engine step under CUDA graph decode without MTP
/// (`IZWI_QWEN38_DECODE_QUANTUM`, 1-64; 1 is one token per step).
pub(crate) const DECODE_QUANTUM_ENV: &str = "IZWI_QWEN38_DECODE_QUANTUM";
const DEFAULT_DECODE_QUANTUM: usize = 8;
/// Kill switch for pipelined graph decode (`legacy`, `off`, `0`, `false`):
/// a graph-decode quantum then runs its steps one at a time.
pub(crate) const DECODE_PIPELINE_ENV: &str = "IZWI_QWEN38_DECODE_PIPELINE";
/// Qwen3.8-only kill switch for cross-turn prefix reuse (`0`, `false`, `no`,
/// `off`). The global switches (`IZWI_ENABLE_PREFIX_CACHING=0`,
/// `IZWI_PREFIX_REUSE_AUTO=0`) turn it off for every model.
pub(crate) const PREFIX_REUSE_ENV: &str = "IZWI_QWEN38_PREFIX_REUSE";

/// Strict boolean for [`PREFIX_REUSE_ENV`]; unset keeps the registry's
/// decision.
fn qwen38_prefix_reuse_switch(raw: Option<&str>) -> Result<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase) {
        None => Ok(true),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on") => Ok(true),
        Some(value) if matches!(value.as_str(), "0" | "false" | "no" | "off") => Ok(false),
        Some(value) => Err(Error::ConfigError(format!(
            "{PREFIX_REUSE_ENV} must be a boolean, got `{value}`"
        ))),
    }
}

/// How the load decided cross-turn prefix reuse, and the prefills since:
/// `prefix_reuse` in the runtime diagnostics. A prefill attached at a shared
/// history prefix starts at the attached cursor and computes only the rest.
#[derive(Debug, Default)]
pub(crate) struct Qwen38PrefixReuse {
    /// The registry engaged prefix reuse for this model on its backend (the
    /// catalog cell in catalog-auto mode, or the operator's explicit switch).
    requested: bool,
    /// [`PREFIX_REUSE_ENV`] turned it off for Qwen3.8.
    switched_off: bool,
    attached_prefills: AtomicU64,
    fresh_prefills: AtomicU64,
    reused_prompt_tokens: AtomicU64,
    prefilled_prompt_tokens: AtomicU64,
    last_reused_tokens: AtomicU64,
    last_prompt_tokens: AtomicU64,
}

impl Qwen38PrefixReuse {
    fn new(requested: bool, switched_off: bool) -> Self {
        Self {
            requested,
            switched_off,
            ..Self::default()
        }
    }

    /// A request's prefill starts with `reused` of its `prompt` tokens
    /// already resident (zero unless it attached a shared prefix).
    fn record_prefill(&self, reused: usize, prompt: usize) {
        let counter = if reused > 0 {
            &self.attached_prefills
        } else {
            &self.fresh_prefills
        };
        counter.fetch_add(1, AtomicOrdering::Relaxed);
        let (reused, prompt) = (reused as u64, prompt as u64);
        self.reused_prompt_tokens
            .fetch_add(reused, AtomicOrdering::Relaxed);
        self.prefilled_prompt_tokens
            .fetch_add(prompt.saturating_sub(reused), AtomicOrdering::Relaxed);
        self.last_reused_tokens
            .store(reused, AtomicOrdering::Relaxed);
        self.last_prompt_tokens
            .store(prompt, AtomicOrdering::Relaxed);
    }
}

fn decode_quantum_tokens() -> usize {
    decode_quantum_from(std::env::var(DECODE_QUANTUM_ENV).ok().as_deref())
}

fn decode_quantum_from(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|tokens| (1..=64).contains(tokens))
        .unwrap_or(DEFAULT_DECODE_QUANTUM)
}

#[cfg(test)]
thread_local! {
    static DEVICE_SAMPLING_FOR_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: sample through the device sampler's portable references off CUDA
/// in this thread (step by step and pipelined alike) until the guard drops.
#[cfg(test)]
pub(crate) fn force_device_sampling_for_tests() -> ForcedDeviceSampling {
    DEVICE_SAMPLING_FOR_TESTS.with(|forced| forced.set(true));
    ForcedDeviceSampling
}

#[cfg(test)]
pub(crate) struct ForcedDeviceSampling;

#[cfg(test)]
impl Drop for ForcedDeviceSampling {
    fn drop(&mut self) {
        DEVICE_SAMPLING_FOR_TESTS.with(|forced| forced.set(false));
    }
}

/// Graph-decode quanta the scheduler granted, for `cuda_graphs.decode_quantum`
/// in the diagnostics: whether the multi-token quantum actually reached the
/// model, and how wide.
#[derive(Debug, Default)]
pub(crate) struct GraphQuantumCounters {
    quanta: AtomicU64,
    tokens: AtomicU64,
    pipelined_quanta: AtomicU64,
    last_granted: AtomicU64,
    max_granted: AtomicU64,
}

impl GraphQuantumCounters {
    fn record(&self, granted: usize, committed: usize, pipelined: bool) {
        let granted = granted as u64;
        self.quanta.fetch_add(1, AtomicOrdering::Relaxed);
        self.tokens
            .fetch_add(committed as u64, AtomicOrdering::Relaxed);
        if pipelined {
            self.pipelined_quanta.fetch_add(1, AtomicOrdering::Relaxed);
        }
        self.last_granted.store(granted, AtomicOrdering::Relaxed);
        self.max_granted.fetch_max(granted, AtomicOrdering::Relaxed);
    }

    fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "quanta": self.quanta.load(AtomicOrdering::Relaxed),
            "input_tokens": self.tokens.load(AtomicOrdering::Relaxed),
            "pipelined_quanta": self.pipelined_quanta.load(AtomicOrdering::Relaxed),
            "granted_last": self.last_granted.load(AtomicOrdering::Relaxed),
            "granted_max": self.max_granted.load(AtomicOrdering::Relaxed),
        })
    }
}

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
            None => DEFAULT_MTP_ENABLED,
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
    CudaBf16,
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
                if qwen38_bf16_kv_enabled(cuda_bf16_override)
                    && qwen38_cuda_supports_bf16(cuda_compute_capability) =>
            {
                Self::CudaBf16
            }
            BackendKind::Cuda if qwen38_bf16_kv_enabled(cuda_bf16_override) => {
                Self::CudaF16CapabilityFallback
            }
            BackendKind::Cuda => Self::CudaF16Fallback,
        }
    }

    const fn dtype(self) -> DType {
        match self {
            Self::CpuF32 => DType::F32,
            Self::MetalF16 | Self::CudaF16Fallback | Self::CudaF16CapabilityFallback => DType::F16,
            Self::CudaBf16 => DType::BF16,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::CpuF32 => "portable_f32",
            Self::MetalF16 => "metal_f16",
            Self::CudaF16Fallback => "cuda_f16_fallback",
            Self::CudaF16CapabilityFallback => "cuda_f16_capability_fallback",
            Self::CudaBf16 => "cuda_bf16",
        }
    }

    const fn fallback_reason(self) -> Option<&'static str> {
        match self {
            Self::CudaF16Fallback => {
                Some("CUDA BF16 KV disabled by IZWI_QWEN38_CUDA_BF16_KV; using F16")
            }
            Self::CudaF16CapabilityFallback => {
                Some("CUDA BF16 KV requires an observed compute capability 8.0 or newer; using F16")
            }
            _ => None,
        }
    }
}

fn qwen38_cuda_supports_bf16(compute_capability: Option<(u32, u32)>) -> bool {
    compute_capability.is_some_and(cuda_compute_capability_supports_bf16)
}

fn qwen38_bf16_kv_enabled(raw: Option<&str>) -> bool {
    // BF16 activations must retain their exponent range in persistent KV.
    // Narrowing finite values above 65504 to F16 inserts infinities into the
    // cache; later attention (including masked P*V) can then produce NaNs.
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        None | Some("1" | "true" | "yes" | "on")
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

/// Tokens per prefill forward for a load on `backend`: 1,024 on CUDA when
/// the dequantized prefill GEMM runs (`prefill_gemm`), 256 otherwise.
/// `IZWI_QWEN38_PREFILL_CHUNK_SIZE` (1-2,048) overrides both. Admission
/// prices the same chunk ([`prefill_working_set_bytes`]).
pub(crate) fn qwen38_prefill_chunk_tokens(backend: BackendKind, prefill_gemm: bool) -> usize {
    let default = if backend == BackendKind::Cuda && prefill_gemm {
        CUDA_GEMM_PREFILL_CHUNK_SIZE
    } else {
        DEFAULT_PREFILL_CHUNK_SIZE
    };
    std::env::var(PREFILL_CHUNK_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
        .min(MAX_PREFILL_CHUNK_SIZE)
}

/// Transient bytes of one prefill forward of `chunk_tokens` tokens: per
/// token the activation bound of one continuous-decode row (every
/// activation of the widest layer, counted in F32), which covers the F32
/// DeltaNet intermediates, the packed gate/up output and the norm
/// temporaries a prefill row holds at once. About 392 MiB at 1,024 tokens
/// on the 27B trunk.
pub(crate) fn prefill_working_set_bytes(
    cfg: &Qwen38TextConfig,
    chunk_tokens: usize,
) -> Result<u64> {
    continuous_decode_workspace_per_row_bytes(cfg, 0, None)?
        .checked_mul(chunk_tokens as u64)
        .ok_or_else(|| Error::ModelLoadError("Qwen3.8 prefill working set overflow".into()))
}

/// Fully prepared Qwen3.8 text prefill input. The runtime carries this exact
/// artifact into the executor so tokenization and position construction happen
/// once.
#[derive(Debug, Clone)]
pub struct Qwen38PreparedPrompt {
    prompt_ids: Vec<u32>,
    prompt_positions: Vec<[usize; 3]>,
    next_text_position: usize,
    /// Tokens of the stable history: everything before the final assistant
    /// generation header, which the conversation's next turn renders again.
    history_boundary: Option<usize>,
}

impl Qwen38PreparedPrompt {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    /// The prompt's stable-history boundary in tokens (see
    /// [`history_boundary`]); the engine snapshots hybrid state at its
    /// interval floor so the next turn attaches there.
    pub(crate) fn history_boundary(&self) -> Option<usize> {
        self.history_boundary
    }

    pub(crate) fn prompt_positions(&self) -> &[[usize; 3]] {
        &self.prompt_positions
    }
}

/// Tokens of a rendered prompt's stable history: everything before its final
/// `<|im_start|>`, the assistant generation header every Qwen3.8 render ends
/// with. The conversation's next turn renders exactly these tokens again: an
/// answer re-renders differently from the tokens that were generated (the
/// template rebuilds its `<think>` block, and drops it for earlier turns when
/// `preserve_thinking` is off), so nothing after the header is shared.
/// `<|im_start|>` is a special token, so the encoding splits there and the
/// prefix is a token boundary. A render that diverges earlier (a changed
/// system prompt or tool list) only shortens the token-exact prefix match.
fn history_boundary(prompt_ids: &[u32], im_start: u32) -> Option<usize> {
    prompt_ids
        .iter()
        .rposition(|id| *id == im_start)
        .filter(|boundary| *boundary > 0)
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

/// Durable CPU-only continuation record. It deliberately owns no tensors, cache
/// views, device events, or physical sequence identities.
#[derive(Clone)]
pub(crate) struct Qwen38ReplayCheckpoint {
    prepared: Qwen38PreparedPrompt,
    generated_ids: Vec<u32>,
    appended_tokens: usize,
    prefill_progress: usize,
    pending_token: Option<u32>,
    bootstrap_token: Option<u32>,
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
    adaptive_mtp: AdaptiveMtp,
}

impl Qwen38ReplayCheckpoint {
    pub(crate) fn replay_tokens(&self) -> usize {
        self.appended_tokens
    }
}

pub struct ChatDecodeState {
    replay: Option<std::sync::Arc<Qwen38ReplayCheckpoint>>,
    prepared: Qwen38PreparedPrompt,
    /// Append-only CPU journal; bounded by max_new_tokens. Step rollback
    /// stores only its length, avoiding quadratic history copies. Suspension
    /// temporarily clones at most 4 * max_new_tokens bytes (plus prompt IDs).
    generated_ids: Vec<u32>,
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
    /// Prompt tokens committed by scheduler-level chunked prefill so far.
    /// Equals the prompt length once prefill completed.
    prefill_progress: usize,
    config: ChatGenerationConfig,
    rng: SimpleRng,
    draft_rng: SimpleRng,
    adaptive_mtp: AdaptiveMtp,
    mtp_timings: Vec<timing::PendingRound>,
    /// DS9.3: logprob entries produced by the current decode step, drained
    /// by the registry right after the step. Cleared at each sample.
    pub(crate) pending_logprobs: Vec<crate::engine::TokenLogprob>,
    /// DS9.3: resolved logprob entry for the prefill-seeded anchor token,
    /// flushed by the decode quantum that publishes it.
    bootstrap_logprob: Option<crate::engine::TokenLogprob>,
}

impl ChatDecodeState {
    pub(crate) fn replay_tokens(&self) -> Option<usize> {
        self.replay.as_ref().map(|saved| saved.appended_tokens)
    }
    /// Caller must fence the completed step before releasing its physical state.
    pub(crate) fn replay_checkpoint(&self) -> Result<Qwen38ReplayCheckpoint> {
        if let Some(saved) = &self.replay {
            return Ok((**saved).clone());
        }
        if self.finished {
            return Err(Error::InvalidInput(
                "cannot suspend a finished Qwen3.8 sequence".into(),
            ));
        }
        let appended_tokens = self.physical_kv.context_len();
        let known = self
            .prepared
            .prompt_ids
            .len()
            .saturating_add(self.generated_ids.len());
        if appended_tokens > known || appended_tokens < self.prefill_progress {
            return Err(Error::InferenceError(
                "Qwen3.8 replay journal does not cover cache cursor".into(),
            ));
        }
        Ok(Qwen38ReplayCheckpoint {
            prepared: self.prepared.clone(),
            generated_ids: self.generated_ids.clone(),
            appended_tokens,
            prefill_progress: self.prefill_progress,
            pending_token: self.pending_token,
            bootstrap_token: self.bootstrap_token,
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

    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

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

    /// Swap in the scheduler-owned reservations for one shared-step quantum
    /// and snapshot the pre-quantum state so a failed batch step can restore
    /// every row exactly.
    pub(crate) fn begin_shared_step_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<Qwen38SharedStepCheckpoint> {
        let current = &self.physical_kv;
        if current.arena().id() != cache.arena().id()
            || current.context_len() != cache.context_len()
        {
            return Err(Error::InferenceError(
                "Qwen3.8 shared-step KV reservation does not continue the session".into(),
            ));
        }
        if self.uses_mtp_physical_kv() != mtp_cache.is_some() {
            return Err(Error::InferenceError(
                "Qwen3.8 shared-step MTP reservation does not match the decode state policy".into(),
            ));
        }
        if let (Some(current_mtp), Some(new_mtp)) =
            (self.mtp_physical_kv.as_ref(), mtp_cache.as_ref())
        {
            if current_mtp.arena().id() != new_mtp.arena().id()
                || current_mtp.context_len() != new_mtp.context_len()
            {
                return Err(Error::InferenceError(
                    "Qwen3.8 shared-step MTP KV reservation does not continue the session".into(),
                ));
            }
        }
        Ok(Qwen38SharedStepCheckpoint {
            replay: self.replay.clone(),
            text_state: self.text_state.clone(),
            mtp_anchor_hidden: self.mtp_anchor_hidden.clone(),
            unconsumed_output: self.unconsumed_output.clone(),
            pending_token: self.pending_token,
            bootstrap_token: self.bootstrap_token,
            next_text_position: self.next_text_position,
            history_ids: self.history_ids.clone(),
            generated_ids_len: self.generated_ids.len(),
            tokens_generated: self.tokens_generated,
            decoder: self.decoder.clone(),
            assembled: self.assembled.clone(),
            finished: self.finished,
            rng: self.rng.clone(),
            draft_rng: self.draft_rng.clone(),
            adaptive_mtp: self.adaptive_mtp.clone(),
            mtp_timings: self.mtp_timings.clone(),
            physical_kv: std::mem::replace(&mut self.physical_kv, cache),
            mtp_physical_kv: match mtp_cache {
                Some(new_mtp) => self.mtp_physical_kv.replace(new_mtp),
                None => None,
            },
        })
    }

    pub(crate) fn rollback_shared_step_quantum(&mut self, checkpoint: Qwen38SharedStepCheckpoint) {
        let Qwen38SharedStepCheckpoint {
            replay,
            text_state,
            physical_kv,
            mtp_physical_kv,
            mtp_anchor_hidden,
            unconsumed_output,
            pending_token,
            bootstrap_token,
            next_text_position,
            history_ids,
            generated_ids_len,
            tokens_generated,
            decoder,
            assembled,
            finished,
            rng,
            draft_rng,
            adaptive_mtp,
            mtp_timings,
        } = checkpoint;
        self.replay = replay;
        self.text_state = text_state;
        self.physical_kv = physical_kv;
        self.mtp_physical_kv = mtp_physical_kv;
        self.mtp_anchor_hidden = mtp_anchor_hidden;
        self.unconsumed_output = unconsumed_output;
        self.pending_token = pending_token;
        self.bootstrap_token = bootstrap_token;
        self.next_text_position = next_text_position;
        self.history_ids = history_ids;
        self.generated_ids.truncate(generated_ids_len);
        self.tokens_generated = tokens_generated;
        self.decoder = decoder;
        self.assembled = assembled;
        self.finished = finished;
        self.rng = rng;
        self.draft_rng = draft_rng;
        self.adaptive_mtp.restore_from_checkpoint(adaptive_mtp);
        self.mtp_timings = mtp_timings;
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

/// Pre-quantum snapshot for shared-step continuous decode. Mirrors the Qwen3
/// managed checkpoint: reservations are swapped for fresh views so staged KV
/// writes under the new views are abandoned when the previous views are
/// restored on rollback.
pub(crate) struct Qwen38SharedStepCheckpoint {
    replay: Option<std::sync::Arc<Qwen38ReplayCheckpoint>>,
    text_state: Qwen38TextRuntimeState,
    physical_kv: PhysicalPagedKvCache,
    mtp_physical_kv: Option<PhysicalPagedKvCache>,
    mtp_anchor_hidden: Option<Tensor>,
    unconsumed_output: Option<Tensor>,
    pending_token: Option<u32>,
    bootstrap_token: Option<u32>,
    next_text_position: usize,
    history_ids: Vec<u32>,
    generated_ids_len: usize,
    tokens_generated: usize,
    decoder: IncrementalDecoder,
    assembled: String,
    finished: bool,
    rng: SimpleRng,
    draft_rng: SimpleRng,
    adaptive_mtp: AdaptiveMtp,
    mtp_timings: Vec<timing::PendingRound>,
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
    im_start: u32,
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
        let im_start = required("<|im_start|>")?;
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
                im_start,
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
    performance: crate::performance::PerformanceConfig,
    load_timing: serde_json::Value,
    prefill_chunk_size: usize,
    cuda_compute_capability: Option<(u32, u32)>,
    kv_storage_provider: Qwen38KvStorageProvider,
    variant: ModelVariant,
    tokenizer: Qwen38Tokenizer,
    text_config: Qwen38TextConfig,
    text_model: Qwen38TextModel,
    mtp_policy: Qwen38MtpPolicy,
    mtp_head: Option<Qwen38MtpHead>,
    /// The cache contract shares prefixes (the registry engaged reuse and
    /// [`PREFIX_REUSE_ENV`] did not turn it off); an MTP head still keeps
    /// every domain private.
    prefix_caching: bool,
    prefix_reuse: Qwen38PrefixReuse,
    graph_quantum: GraphQuantumCounters,
    /// Device timings of the hot kernels measured once at load (CUDA;
    /// `text/kernel_timings.rs`), reported under `kernel_timings`.
    kernel_timings: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Qwen38RepresentationDiagnostics {
    resident_representation: &'static str,
    fp8_execution_mode: String,
    fallback_reason: String,
    runtime_compute_dtype: &'static str,
}

/// Diagnostics of the residency policy a model was loaded under.
fn qwen38_representation_diagnostics(
    policy: &Qwen38ResidencyPolicy,
) -> Qwen38RepresentationDiagnostics {
    let representation = policy.representation();
    Qwen38RepresentationDiagnostics {
        resident_representation: representation.as_str(),
        fp8_execution_mode: policy.execution_mode(),
        fallback_reason: policy.fallback_reason(),
        runtime_compute_dtype: representation.compute_dtype(),
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
        let performance = crate::performance::PerformanceConfig::default().resolve_env()?;
        Self::load_with_performance(model_dir, variant, device, &performance, false)
    }

    pub fn load_with_performance(
        model_dir: &Path,
        variant: ModelVariant,
        device: DeviceProfile,
        performance: &crate::performance::PerformanceConfig,
        prefix_reuse: bool,
    ) -> Result<Self> {
        performance.validate()?;
        if variant != ModelVariant::Qwen3827BFp8 {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Qwen3.8 chat variant: {variant}"
            )));
        }
        let prefix_reuse_switch =
            qwen38_prefix_reuse_switch(std::env::var(PREFIX_REUSE_ENV).ok().as_deref())?;
        let checkpoint =
            Qwen38NativeCheckpoint::open_with_options(model_dir, &performance.loading)?;
        let tokenizer = Qwen38Tokenizer::load_hf(model_dir)?;
        let text_config = checkpoint.config.text.clone();
        let projection_materialization = qwen38_projection_materialization(&device)?;
        // The same resolution load admission priced (cached device gates).
        let residency = Qwen38ResidencyPolicy::resolve(
            &device.device,
            projection_materialization,
            performance,
            &checkpoint.tensors,
            &checkpoint.config,
        );
        let mtp_policy = if residency.mtp_loaded() {
            Qwen38MtpPolicy::Enabled {
                draft_tokens: performance.cuda.mtp_draft_tokens,
            }
        } else {
            Qwen38MtpPolicy::Disabled
        };
        let text_model = Qwen38TextModel::load_native_with_policy(
            &checkpoint.tensors,
            &checkpoint.config,
            &device.device,
            projection_materialization,
            &performance.cuda,
            &residency,
        )?;
        let mtp_head = match mtp_policy {
            Qwen38MtpPolicy::Disabled => None,
            Qwen38MtpPolicy::Enabled { .. } => Some(Qwen38MtpHead::load_native_with_policy(
                &checkpoint.tensors,
                &checkpoint.config,
                &checkpoint.mtp,
                &device.device,
                projection_materialization,
                &performance.cuda,
                &residency,
            )?),
        };
        record_mtp_policy(mtp_policy.enabled());
        // After every load-time check: the timings run the resolved paths.
        let kernel_timings = text_model.kernel_timings();
        if kernel_timings["status"] == "measured" {
            info!(timings = %kernel_timings, "Qwen3.8 load-time kernel timings");
        }
        let projection_representation = text_model.projection_representation();
        let device_kind = BackendKind::from(device.kind);
        let cuda_compute_capability = device.capabilities.cuda_compute_capability;
        let kv_storage_provider = qwen38_kv_storage_provider(device_kind, cuda_compute_capability);
        if device_kind == BackendKind::Cuda {
            record_cuda_kv_provider(kv_storage_provider == Qwen38KvStorageProvider::CudaBf16);
        }
        info!(
            variant = %variant,
            backend = ?device.kind,
            kv_storage_provider = kv_storage_provider.as_str(),
            revision = QWEN38_27B_FP8_REVISION,
            tensors = checkpoint.tensors.tensor_count(),
            resident_representation = projection_representation.as_str(),
            fp8_execution_mode = %residency.execution_mode(),
            mtp_enabled = mtp_policy.enabled(),
            mtp_decision = residency.mtp_decision(),
            mtp_draft_tokens = mtp_policy.draft_tokens(),
            prefix_reuse = prefix_reuse && prefix_reuse_switch && !mtp_policy.enabled(),
            cuda_graphs = %text_model.cuda_graphs_summary()["phase"],
            "Loaded native Qwen3.8 text checkpoint"
        );
        Ok(Self {
            device_kind,
            performance: performance.clone(),
            load_timing: checkpoint.tensors.loading_diagnostics(),
            prefill_chunk_size: qwen38_prefill_chunk_tokens(
                device_kind,
                text_model.prefill_gemm_active(),
            ),
            cuda_compute_capability,
            kv_storage_provider,
            variant,
            tokenizer,
            text_config,
            text_model,
            mtp_policy,
            mtp_head,
            prefix_caching: prefix_reuse && prefix_reuse_switch,
            prefix_reuse: Qwen38PrefixReuse::new(prefix_reuse, !prefix_reuse_switch),
            graph_quantum: GraphQuantumCounters::default(),
            kernel_timings,
        })
    }

    /// Prefill span this model prefers for an isolated request: its chunk
    /// on CUDA once the dequantized prefill GEMM runs (the GEMM's weight
    /// expansion amortizes over the span), else the engine's threshold (0).
    pub fn preferred_prefill_tokens(&self) -> usize {
        if self.device_kind == BackendKind::Cuda && self.text_model.prefill_gemm_active() {
            self.prefill_chunk_size
        } else {
            0
        }
    }

    pub(crate) fn sustained_cuda_mtp_quantum(&self) -> bool {
        self.device_kind == BackendKind::Cuda
            && self.performance.cuda.enabled()
            && self.performance.cuda.mtp_quantum.enabled()
            && self.mtp_policy.enabled()
    }

    /// Whether this model's multi-token quantum outlives the request's soft
    /// scheduling SLA: the CUDA MTP quantum, or the graph-decode quantum
    /// (no MTP head). An interactive or streaming request passes that SLA
    /// (0.65-0.75 s at normal priority) early in almost every stream, after
    /// which the scheduler would grant one token per step for the rest of
    /// it. The quantum still yields to waiting or co-batched work and to hard
    /// deadlines, and the scheduler grants it only to a loaded CUDA model.
    pub(crate) fn sustained_decode_quantum(&self) -> bool {
        self.sustained_cuda_mtp_quantum()
            || (self.mtp_head.is_none()
                && self.text_model.graph_decode()
                && self.preferred_decode_tokens() > 1)
    }

    /// Device bytes the decode graphs may hold: the figure admission
    /// charged (`Qwen38ResidencyPolicy::graph_capacity_bytes`).
    pub(crate) fn graph_cache_capacity_bytes(&self) -> u64 {
        self.text_model
            .residency()
            .graph_capacity_bytes(&self.text_config, &self.performance)
            .unwrap_or(0)
    }

    fn device_sampling_enabled(&self) -> bool {
        #[cfg(test)]
        if DEVICE_SAMPLING_FOR_TESTS.with(std::cell::Cell::get) {
            return true;
        }
        self.device_kind == BackendKind::Cuda
            && self.performance.cuda.enabled()
            && self.performance.cuda.device_sampling.enabled()
    }

    /// Why a graph-decode quantum may not pipeline its steps, if it may not:
    /// pipelining chooses tokens with the device sampler, so it needs the
    /// device sampler the step-by-step decode uses.
    fn pipelining_refusal(&self) -> Option<String> {
        if crate::models::architectures::qwen38::fast_path::legacy_requested(DECODE_PIPELINE_ENV) {
            Some(format!("{DECODE_PIPELINE_ENV}=legacy"))
        } else if !self.device_sampling_enabled() {
            Some(
                "pipelined decode chooses tokens with the CUDA device sampler, which is off".into(),
            )
        } else {
            None
        }
    }

    fn sample_next_token(
        &self,
        logits: &Tensor,
        vocab_size: usize,
        config: &ChatGenerationConfig,
        history: &[u32],
        rng: &mut SimpleRng,
    ) -> Result<u32> {
        let (token, _) =
            self.sample_next_token_with_logprobs(logits, vocab_size, config, history, rng)?;
        Ok(token)
    }

    /// DS9.3: sample a token and, when the request asked for logprobs,
    /// route through the host sampler: the CUDA kernels path returns only
    /// token ids, so raw-distribution stats need the host route.
    fn sample_next_token_with_logprobs(
        &self,
        logits: &Tensor,
        vocab_size: usize,
        config: &ChatGenerationConfig,
        history: &[u32],
        rng: &mut SimpleRng,
    ) -> Result<(
        u32,
        Option<crate::models::shared::sampling::RawTokenLogprobs>,
    )> {
        if self.device_sampling_enabled() && !config.logprobs {
            let token = device_sampling::sample(logits, vocab_size, config, history, rng)?;
            record_sampling_bounded_cuda(true);
            return Ok((token, None));
        }
        // Logprob requests and explicit CUDA opt-outs use the host sampler;
        // CPU and Metal retain their established sampling routes.
        let logits = if logits.device().is_cuda() {
            logits.to_device(&candle_core::Device::Cpu)?
        } else {
            logits.clone()
        };
        sample_next_token_with_logprobs(&logits, vocab_size, config, history, rng)
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
            self.prefix_caching,
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
        Some(self.text_model.projection_representation().compute_dtype())
    }

    pub fn runtime_diagnostics(&self) -> serde_json::Value {
        let representation = qwen38_representation_diagnostics(self.text_model.residency());
        let residency_paths = self.text_model.residency().diagnostics();
        // Per projection kind once the policy planned the prefill GEMM;
        // otherwise the policy's own reason.
        let prefill_gemm_checks: Vec<_> = self
            .text_model
            .fp8_prefill_gemm_checks()
            .iter()
            .chain(
                self.mtp_head
                    .iter()
                    .flat_map(|head| head.fp8_prefill_gemm_checks().iter()),
            )
            .collect();
        let fp8_prefill_gemm = if prefill_gemm_checks.is_empty() {
            residency_paths["fp8_prefill_gemm"].clone()
        } else {
            super::fast_path::summarize(prefill_gemm_checks)
        };
        let optimization_counters = qwen38_optimization_telemetry_snapshot();
        let draft_acceptance_rate = ratio_or_none(
            optimization_counters.mtp_accepted_draft_tokens_total,
            optimization_counters.mtp_draft_tokens_total,
        );
        let bonus_rate = ratio_or_none(
            optimization_counters.mtp_bonus_tokens_total,
            optimization_counters.mtp_rounds_total,
        );
        let replay_amplification = ratio_or_none(
            optimization_counters.mtp_target_replay_tokens_total,
            optimization_counters.mtp_target_verified_tokens_total,
        );
        let observed_execution = match (
            optimization_counters.mtp_rounds_total > 0,
            optimization_counters.mtp_scalar_target_tokens_total > 0,
        ) {
            (true, true) => "mixed_speculative_and_scalar",
            (true, false) => "speculative",
            (false, true) => "scalar_only",
            (false, false) => "not_observed",
        };
        let optimization_evidence = serde_json::json!({
            "scope": "qwen38_process_lifetime",
            "cuda_runtime_validated": false,
            "performance": self.performance,
            "load_timing": self.load_timing,
            "cuda_compute_capability": self.cuda_compute_capability.map(|(major, minor)| format!("{major}.{minor}")),
            "counters": optimization_counters,
            "managed_kv_counters_source": "runtime_metrics.kv_cache.models[].arenas[].operations",
            "managed_kv_counter_coverage": [
                "allocation",
                "workspace",
                "workspace_budget_and_high_water",
                "full_request_page_claims",
                "host_synchronization",
                "attention_provider",
                "cuda_graph"
            ],
            "cuda_kv_storage": {
                "candidate_switch": CUDA_BF16_KV_ENV,
                "default_on_supported_cuda": true,
                "selected_provider": self.kv_storage_provider.as_str(),
                "storage_dtype": format!("{:?}", self.kv_storage_provider.dtype()).to_ascii_lowercase(),
                "fallback_reason": self.kv_storage_provider.fallback_reason(),
                "runtime_validated": false,
                "quantized": false,
                "physical_format": "dense",
                "quantized_candidate_status": "unavailable_scale_contract_incomplete",
            },
            "prefill": {
                "chunk_tokens": self.prefill_chunk_size,
                "target_hidden_retention": "final_row_only",
                "mtp_bootstrap": if self.mtp_policy.enabled() {
                    "streamed_shifted_chunks"
                } else {
                    "disabled"
                },
                "transient_memory_bound": "chunk_shaped",
                "fp8_projection_path": "rows >= 16: dequantized weight tiles (<= 128 MiB) + cuBLAS once checked per projection kind (fast_paths.fp8_prefill_gemm); otherwise the compact FP8 mm kernel",
                "fp8_projection_switch": super::residency::FP8_PREFILL_GEMM_ENV,
                "fp8_prefill_scratch_bytes": self.text_model.residency().prefill_scratch_bytes(),
                "chunk_override": PREFILL_CHUNK_ENV,
                "maximum_chunk_tokens": MAX_PREFILL_CHUNK_SIZE,
                "cuda_prefill_gemm_chunk_tokens": CUDA_GEMM_PREFILL_CHUNK_SIZE,
                "preferred_span_tokens": self.preferred_prefill_tokens(),
                "preferred_span_policy": "an isolated request (nothing decoding or waiting) prefills the whole chunk in one engine step; otherwise the engine's chunk threshold applies",
            },
            "mtp": {
                "enabled": self.mtp_policy.enabled(),
                "draft_tokens": self.mtp_policy.draft_tokens(),
                "adaptive": self.device_kind == BackendKind::Cuda && self.performance.cuda.enabled() && self.performance.cuda.mtp_adaptive,
                "depth_semantics": "starting_depth_when_adaptive_otherwise_fixed",
                "adaptive_objective": "completed_cuda_event_seconds_per_committed_token",
                "adaptive_timing": "delayed_nonblocking_event_query_includes_final_mtp_commit",
                "adaptive_depth_bounds": [0, 3],
                "prefix_commit": "compact_recurrence_reconstruction_without_target_weight_replay",
                "default_enabled": DEFAULT_MTP_ENABLED,
                "default_policy": "auto: on, except off on CUDA when graph decode is planned (IZWI_QWEN38_MTP=1 keeps it)",
                "decision": self.text_model.residency().mtp_decision(),
                "default_draft_tokens": DEFAULT_MTP_DRAFT_TOKENS,
                "enabled_switch": MTP_ENABLED_ENV,
                "depth_switch": MTP_DRAFT_TOKENS_ENV,
                "implementation_status": "implemented_unvalidated",
                "runtime_validated": false,
                "performance_certified": false,
                "scheduler_policy": "speculate_only_without_queue_pressure_or_concurrent_decode",
                "nonfinite_draft_policy": "discard_round_and_use_target_only_sampling_for_request",
                "execution_evidence": {
                    "observed_execution": observed_execution,
                    "draft_acceptance_rate": draft_acceptance_rate,
                    "bonus_rate": bonus_rate,
                    "target_replay_to_verified_ratio": replay_amplification,
                    "speculative_round_counter": "mtp_rounds_total",
                    "scalar_target_counter": "mtp_scalar_target_tokens_total",
                    "accepted_draft_counter": "mtp_accepted_draft_tokens_total",
                    "replayed_target_counter": "mtp_target_replay_tokens_total",
                },
            },
        });
        serde_json::json!({
            "checkpoint_revision": QWEN38_27B_FP8_REVISION,
            "checkpoint_format": "safetensors_block_fp8",
            "resident_representation": representation.resident_representation,
            "runtime_compute_dtype": representation.runtime_compute_dtype,
            "fp8_execution_mode": representation.fp8_execution_mode,
            "fallback_reason": representation.fallback_reason,
            "performance": self.performance,
            "load_timing": self.load_timing,
            "decode_graphs": {
                "requested": self.performance.cuda.enabled() && self.performance.cuda.decode_graphs.enabled(),
                "regions": ["residual_add_then_rms_norm", "explicit_native_fp8_gate_up_silu_down"],
                "max_verification_rows": 4,
                "target_layer_limit": 8,
                "retired_by_cuda_graph_decode": self.text_model.graph_decode(),
                "target_counters": self.text_model.graph_diagnostics(),
                "mtp_counters": self.mtp_head.as_ref().map(Qwen38MtpHead::graph_diagnostics),
                "maximum_cache_bytes": self.graph_cache_capacity_bytes(),
                "ownership": "model_owned_serialized_graphs_with_stable_inputs_outputs_and_weights",
                "fallback": "shape_or_capture_failure_negative_cached_eager_region",
                "q8_mlp_capture": "excluded_unretainable_candle_global_scratch",
                "packed_fp8_mlp_capture": "deferred_to_phase3",
                "full_model_capture": false,
                "runtime_validated": false,
            },
            "cuda_graphs": self.cuda_graphs_diagnostics(),
            "prefix_reuse": self.prefix_reuse_diagnostics(),
            "kernel_timings": self.kernel_timings,
            "optimization_evidence": optimization_evidence,
            "fast_paths": {
                "target": self.text_model.fast_path_diagnostics(),
                "mtp": self.mtp_head.as_ref().map(Qwen38MtpHead::fast_path_diagnostics),
                "fp8_projections": residency_paths["fp8_projections"],
                "fp8_gemv": residency_paths["fp8_gemv"],
                "fp8_prefill_gemm": fp8_prefill_gemm,
                "fp8_lm_head": self.text_model.fp8_lm_head_diagnostics(),
                "metal_fp8_gemv": super::fast_path::summarize(
                    self.text_model.fp8_gemv_checks().iter().chain(
                        self.mtp_head
                            .iter()
                            .flat_map(|head| head.fp8_gemv_checks().iter()),
                    ),
                ),
                "fp8_packing": super::fast_path::summarize(
                    self.text_model.fp8_packing().iter().chain(
                        self.mtp_head
                            .iter()
                            .flat_map(|head| head.fp8_packing().iter()),
                    ),
                ),
                "self_check_failures_total": optimization_counters.fast_path_self_check_failures_total,
                "resolution": "load_time_self_check_at_checkpoint_geometry",
                "runtime_validated": false,
            },
            "metal": {
                "projection_backend": self.performance.metal.projection_backend,
                "gemv_max_rows": 8,
                "prefill": "dequantize_to_transient_f16_then_matmul",
                "prefill_scratch_bytes": self.text_model.residency().prefill_scratch_bytes(),
                "runtime_validated": false,
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

    pub fn supports_continuous_decode_batch(&self) -> bool {
        true
    }

    /// Conservative per-row workspace estimate for hybrid batch collation.
    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        continuous_decode_workspace_per_row_bytes(
            &self.text_config,
            self.tokenizer.vocab_size,
            self.mtp_head
                .as_ref()
                .map(|_| self.preferred_decode_tokens()),
        )
    }

    /// DS9.4: whether this loaded model may run shared speculative envelopes
    /// inside continuous batches (kill-switchable, default follows MTP).
    pub(crate) fn mtp_in_continuous_enabled(&self) -> bool {
        self.mtp_head.is_some()
            && self.device_kind == BackendKind::Cuda
            && self.performance.cuda.enabled()
            && self.performance.cuda.mtp_in_continuous.enabled()
    }

    /// Decode inputs this model would like per engine step: the MTP round
    /// width with an MTP head; the graph-decode quantum
    /// (`IZWI_QWEN38_DECODE_QUANTUM`, default 8) under CUDA graph decode;
    /// otherwise one.
    pub(crate) fn preferred_decode_tokens(&self) -> usize {
        match self.mtp_policy.draft_tokens() {
            Some(draft_tokens) => {
                if self.device_kind == BackendKind::Cuda
                    && self.performance.cuda.enabled()
                    && self.performance.cuda.mtp_adaptive
                {
                    4
                } else {
                    draft_tokens.saturating_add(1)
                }
            }
            None if self.text_model.graph_decode() => decode_quantum_tokens(),
            None => 1,
        }
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
        self.prefix_reuse
            .record_prefill(0, prepared.prompt_ids.len());
        let mut text_state = self.text_model.new_state();
        let (logits, final_target_hidden) = match (&self.mtp_head, mtp_cache.as_mut()) {
            (Some(head), Some(mtp_cache)) => self.prefill_text_and_mtp_physical(
                &prepared,
                &mut text_state,
                &mut cache,
                Some((head, mtp_cache)),
            )?,
            (None, None) => {
                self.prefill_text_and_mtp_physical(&prepared, &mut text_state, &mut cache, None)?
            }
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
        let track_history =
            config.repetition_penalty > 1.0 || config.presence_penalty.abs() > f32::EPSILON;
        let mut history_ids =
            initial_penalty_history(&prepared.prompt_ids, max_new_tokens, track_history);
        let mut rng = SimpleRng::new(config.seed);
        let (unconsumed_output, pending_token, bootstrap_token, mtp_anchor_hidden) =
            match (&self.mtp_head, mtp_cache.as_mut()) {
                (Some(head), Some(mtp_cache)) => {
                    let history: &[u32] = if track_history { &history_ids } else { &[] };
                    let anchor = self.sample_next_token(
                        &logits,
                        self.tokenizer.vocab_size,
                        config,
                        history,
                        &mut rng,
                    )?;
                    if track_history {
                        history_ids.push(anchor);
                    }
                    let anchor_embedding = self.text_model.embed_token_ids(&[anchor])?;
                    let pairs = Qwen38MtpPairBatch::single(
                        anchor_embedding,
                        final_target_hidden,
                        *prepared.prompt_positions.last().ok_or_else(|| {
                            Error::InferenceError(
                                "Qwen3.8 MTP prefill has no final prompt position".into(),
                            )
                        })?,
                    )?;
                    let hidden = head.forward_pairs(&pairs, mtp_cache)?;
                    (None, Some(anchor), Some(anchor), Some(hidden))
                }
                (None, None) => (Some(logits), None, None, None),
                // Cache/head consistency was validated before target prefill.
                _ => unreachable!("Qwen3.8 MTP cache/head consistency changed during prefill"),
            };
        let draft_rng = rng.fork();
        Ok(ChatDecodeState {
            replay: None,
            prepared: prepared.clone(),
            generated_ids: Vec::new(),
            text_state,
            physical_kv: cache,
            mtp_physical_kv: mtp_cache,
            mtp_anchor_hidden,
            bootstrap_token,
            physical_tensor_sequence: None,
            unconsumed_output,
            pending_token,
            history_ids,
            decoder: IncrementalDecoder::new(config.skip_special_tokens),
            tokens_generated: 0,
            track_history,
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            next_text_position: prepared.next_text_position,
            prefill_progress: prepared.prompt_ids.len(),
            config: config.clone(),
            rng,
            draft_rng,
            mtp_timings: Vec::new(),
            pending_logprobs: Vec::new(),
            bootstrap_logprob: None,
            adaptive_mtp: AdaptiveMtp::new(
                self.device_kind == BackendKind::Cuda
                    && self.performance.cuda.enabled()
                    && self.performance.cuda.mtp_adaptive,
                self.performance.cuda.mtp_draft_tokens,
            ),
        })
    }

    /// Build CPU continuation metadata with empty physical state. Replay spans
    /// are scheduled independently and must complete before decode resumes.
    pub(crate) fn begin_replay_state_physical(
        &self,
        saved: &Qwen38ReplayCheckpoint,
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
                "Qwen3.8 replay requires fresh matching cache reservations".into(),
            ));
        }
        Ok(ChatDecodeState {
            replay: (saved.appended_tokens > 0).then(|| std::sync::Arc::new(saved.clone())),
            prepared: saved.prepared.clone(),
            generated_ids: saved.generated_ids.clone(),
            text_state: self.text_model.new_state(),
            physical_kv: cache,
            mtp_physical_kv: mtp_cache,
            mtp_anchor_hidden: None,
            bootstrap_token: saved.bootstrap_token,
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
            adaptive_mtp: saved.adaptive_mtp.clone(),
            mtp_timings: Vec::new(),
            pending_logprobs: Vec::new(),
            bootstrap_logprob: None,
        })
    }

    /// Rebuild one scheduler quantum without sampling or emitting output.
    /// Target IDs stop at the append cursor; MTP additionally consumes the
    /// known successor, including a sampled token that target has not appended.
    pub(crate) fn continue_replay_physical(
        &self,
        state: &mut ChatDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        let saved = state
            .replay
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Qwen3.8 state has no pending replay".into()))?;
        if state.physical_kv.context_len() != span_start
            || span_end <= span_start
            || span_end > saved.appended_tokens
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 replay span does not continue its append cursor".into(),
            ));
        }
        let prompt_len = saved.prepared.prompt_ids.len();
        let known_len = prompt_len + saved.generated_ids.len();
        let token_at = |index: usize| {
            if index < prompt_len {
                saved.prepared.prompt_ids.get(index).copied()
            } else if index < known_len {
                saved.generated_ids.get(index - prompt_len).copied()
            } else if index == known_len && known_len == saved.appended_tokens {
                saved.pending_token
            } else {
                None
            }
        };
        let ids: Vec<_> = (span_start..=span_end).filter_map(token_at).collect();
        let positions: Vec<_> = (span_start..span_end)
            .map(|index| {
                if index < prompt_len {
                    saved.prepared.prompt_positions[index]
                } else {
                    [saved.prepared.next_text_position + index - prompt_len; 3]
                }
            })
            .collect();
        let prompt_complete = saved.prefill_progress == saved.prepared.prompt_ids.len();
        for start in (span_start..span_end).step_by(self.prefill_chunk_size) {
            let end = (start + self.prefill_chunk_size).min(span_end);
            let output = self
                .text_model
                .prefill_token_ids_with_hidden_physical(
                    &ids[start - span_start..end - span_start],
                    &positions[start - span_start..end - span_start],
                    &mut state.text_state,
                    &mut state.physical_kv,
                    end == saved.appended_tokens,
                )?
                .ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 replay produced no hidden state".into())
                })?;
            if let (Some(head), Some(mtp)) = (&self.mtp_head, state.mtp_physical_kv.as_mut()) {
                let count = (end - span_start)
                    .min(ids.len().saturating_sub(1))
                    .saturating_sub(start - span_start);
                if count > 0 {
                    let pairs = Qwen38MtpPairBatch::new(
                        self.text_model.embed_token_ids(
                            &ids[start - span_start + 1..start - span_start + count + 1],
                        )?,
                        output.hidden_states.narrow(1, 0, count)?,
                        positions[start - span_start..start - span_start + count].to_vec(),
                    )?;
                    let hidden = head.forward_pairs(&pairs, mtp)?;
                    if end == saved.appended_tokens && prompt_complete {
                        state.mtp_anchor_hidden = Some(hidden.narrow(1, count - 1, 1)?);
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

    /// Convenience restoration used by adapters without scheduler replay spans.
    pub(crate) fn restore_decode_state_physical(
        &self,
        saved: &Qwen38ReplayCheckpoint,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<ChatDecodeState> {
        let mut state = self.begin_replay_state_physical(saved, cache, mtp_cache)?;
        if saved.appended_tokens > 0 {
            self.continue_replay_physical(&mut state, 0, saved.appended_tokens)?;
        }
        Ok(state)
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        self.decode_quantum(state, 1)
    }

    /// Create a decode state with an empty physical reservation and no
    /// prefill progress. Prompt spans are appended by
    /// [`Self::continue_chunked_prefill_physical`] under scheduler control.
    pub(crate) fn begin_chunked_prefill_state_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen38PreparedPrompt>,
        cache: PhysicalPagedKvCache,
        mtp_cache: Option<PhysicalPagedKvCache>,
    ) -> Result<ChatDecodeState> {
        let prepared = resolve_prepared_prompt(prepared, || self.prepare_prompt(messages, config))?;
        // A managed prefix attach hands the state a reservation whose physical
        // cursor already sits above zero (shared pages plus a forked tensor
        // snapshot). The logical prefill cursor starts at the attached
        // cursor; the state must still compute at least one private token.
        let prefill_start = cache.context_len();
        if prepared.prompt_ids.is_empty()
            || prefill_start >= prepared.prompt_ids.len()
            || mtp_cache.as_ref().is_some_and(|mtp| mtp.context_len() != 0)
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 chunked prefill requires a non-empty prompt with a cache cursor below it"
                    .into(),
            ));
        }
        if self.mtp_head.is_some() != mtp_cache.is_some() {
            return Err(Error::InferenceError(
                "Qwen3.8 chunked prefill MTP policy does not match its cache reservation".into(),
            ));
        }
        let track_history =
            config.repetition_penalty > 1.0 || config.presence_penalty.abs() > f32::EPSILON;
        let history_ids =
            initial_penalty_history(&prepared.prompt_ids, max_new_tokens, track_history);
        let mut rng = SimpleRng::new(config.seed);
        let draft_rng = rng.fork();
        self.prefix_reuse
            .record_prefill(prefill_start, prepared.prompt_ids.len());
        Ok(ChatDecodeState {
            replay: None,
            prepared: prepared.clone(),
            generated_ids: Vec::new(),
            text_state: self.text_model.new_state(),
            physical_kv: cache,
            mtp_physical_kv: mtp_cache,
            mtp_anchor_hidden: None,
            bootstrap_token: None,
            physical_tensor_sequence: None,
            unconsumed_output: None,
            pending_token: None,
            history_ids,
            decoder: IncrementalDecoder::new(config.skip_special_tokens),
            tokens_generated: 0,
            track_history,
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            next_text_position: prepared.next_text_position,
            prefill_progress: prefill_start,
            config: config.clone(),
            rng,
            draft_rng,
            mtp_timings: Vec::new(),
            pending_logprobs: Vec::new(),
            bootstrap_logprob: None,
            adaptive_mtp: AdaptiveMtp::new(
                self.device_kind == BackendKind::Cuda
                    && self.performance.cuda.enabled()
                    && self.performance.cuda.mtp_adaptive,
                self.performance.cuda.mtp_draft_tokens,
            ),
        })
    }

    /// Prefill the next scheduler-owned prompt span
    /// `[span_start, span_end)` into the state's physical cache.
    /// Returns `true` when the prompt completed and the decode quantum is
    /// seeded (logits or MTP bootstrap) exactly like the monolithic path;
    /// returns `false` when more spans remain.
    pub(crate) fn continue_chunked_prefill_physical(
        &self,
        state: &mut ChatDecodeState,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen38PreparedPrompt>,
        span_start: usize,
        span_end: usize,
        prompt_tokens: usize,
    ) -> Result<bool> {
        let prepared = resolve_prepared_prompt(prepared, || self.prepare_prompt(messages, config))?;
        let total = prepared.prompt_ids.len();
        let start = state.prefill_progress;
        if total != prompt_tokens {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 resumable prefill expected {prompt_tokens} prompt tokens but prepared {total}"
            )));
        }
        if start != span_start {
            return Err(Error::InferenceError(format!(
                "Qwen3.8 resumable prefill state is at {start} but the scheduler requested {span_start}"
            )));
        }
        if start >= total {
            return Err(Error::InvalidInput(
                "Qwen3.8 chunked prefill already completed for this session".into(),
            ));
        }
        if span_end <= start || span_end > total {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 chunked prefill span [{start},{span_end}) is outside the remaining prompt of {total} tokens"
            )));
        }
        if state.physical_kv.context_len() != start {
            return Err(Error::InferenceError(format!(
                "Qwen3.8 chunked prefill cache holds {} tokens but the next span starts at {start}",
                state.physical_kv.context_len()
            )));
        }
        // Above token zero the hybrid state must already hold the history
        // (an earlier span's, or a restored prefix snapshot). Prefilling from
        // fresh zero state there would silently drop it.
        if start > 0 && !state.text_state.linear_state_initialized() {
            return Err(Error::InferenceError(format!(
                "Qwen3.8 prefill resumes at token {start} without hybrid state; an attached prefix must restore its snapshot first"
            )));
        }
        let compute_logits = span_end == total;
        let output = self.text_model.prefill_token_ids_with_hidden_physical(
            &prepared.prompt_ids[start..span_end],
            &prepared.prompt_positions[start..span_end],
            &mut state.text_state,
            &mut state.physical_kv,
            compute_logits,
        )?;
        let output = output.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 chunked prefill span produced no output".into())
        })?;
        // Feed known-successor MTP pairs for the span exactly like the
        // monolithic prefill loop; the final anchor pair is seeded below.
        if let Some((head, mtp_cache)) = match (&self.mtp_head, state.mtp_physical_kv.as_mut()) {
            (Some(head), Some(mtp_cache)) => Some((head, mtp_cache)),
            _ => None,
        } {
            let known_rows = known_mtp_rows(start, span_end, total);
            if known_rows > 0 {
                let embeddings = self
                    .text_model
                    .embed_token_ids(&prepared.prompt_ids[start + 1..start + 1 + known_rows])?;
                let predecessor_hidden = output.hidden_states.narrow(1, 0, known_rows)?;
                let pairs = Qwen38MtpPairBatch::new(
                    embeddings,
                    predecessor_hidden,
                    prepared.prompt_positions[start..start + known_rows].to_vec(),
                )?;
                head.forward_pairs(&pairs, mtp_cache)?;
            }
        }
        if !compute_logits {
            state.prefill_progress = span_end;
            return Ok(false);
        }

        let logits = output.logits.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 chunked prefill completion produced no logits".into())
        })?;
        let final_target_hidden =
            output
                .hidden_states
                .narrow(1, output.hidden_states.dim(1)?.saturating_sub(1), 1)?;
        match (&self.mtp_head, state.mtp_physical_kv.as_mut()) {
            (Some(head), Some(mtp_cache)) => {
                let history: &[u32] = if state.track_history {
                    &state.history_ids
                } else {
                    &[]
                };
                state.pending_logprobs.clear();
                let (anchor, anchor_logprob) = self.sample_next_token_with_logprobs(
                    &logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?;
                if state.track_history {
                    state.history_ids.push(anchor);
                }
                state.bootstrap_logprob = anchor_logprob
                    .filter(|_| !self.is_stop_token(anchor, &state.config))
                    .map(|raw| {
                        crate::models::shared::sampling::resolve_token_logprob(
                            &self.tokenizer.inner,
                            &raw,
                        )
                    })
                    .transpose()?;
                let anchor_embedding = self.text_model.embed_token_ids(&[anchor])?;
                let pairs = Qwen38MtpPairBatch::single(
                    anchor_embedding,
                    final_target_hidden,
                    *prepared.prompt_positions.last().ok_or_else(|| {
                        Error::InferenceError(
                            "Qwen3.8 chunked prefill has no final prompt position".into(),
                        )
                    })?,
                )?;
                let hidden = head.forward_pairs(&pairs, mtp_cache)?;
                state.mtp_anchor_hidden = Some(hidden);
                state.bootstrap_token = Some(anchor);
                state.pending_token = Some(anchor);
            }
            (None, None) => {
                state.unconsumed_output = Some(logits);
            }
            _ => unreachable!("Qwen3.8 chunked prefill cache/head consistency changed"),
        }
        state.prefill_progress = total;
        Ok(true)
    }

    /// Advance every scheduled row by exactly one token inside one engine
    /// step. A solo row retains MTP semantics. Shared rows batch
    /// target-model projections, MLPs, and ragged full attention while keeping
    /// every row's DeltaNet and convolution state transactionally independent.
    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.replay.is_some()
                || state.finished
                || state.tokens_generated >= state.max_new_tokens
                || state.bootstrap_token.is_some()
                || state.unconsumed_output.is_some()
                || state.pending_token.is_none()
                || state.physical_kv.context_len() != state.next_text_position
                || (self.mtp_head.is_some()
                    && (state
                        .mtp_physical_kv
                        .as_ref()
                        .is_none_or(|cache| cache.context_len() != state.next_text_position)
                        || state.mtp_anchor_hidden.is_none()))
                || (self.mtp_head.is_none()
                    && (state.mtp_physical_kv.is_some() || state.mtp_anchor_hidden.is_some()))
            {
                return Err(Error::InvalidInput(
                    "continuous chat batch contains a non-decodable hybrid state".into(),
                ));
            }
        }
        let state_count = states.len();
        let mut token_ids = Vec::with_capacity(states.len());
        let mut positions = Vec::with_capacity(states.len());
        let mut text_states = Vec::with_capacity(states.len());
        let mut caches = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let pending = state.pending_token.take().expect("pending token checked");
            token_ids.push(pending);
            positions.push([state.next_text_position; 3]);
            text_states.push(&mut state.text_state);
            caches.push(&mut state.physical_kv);
        }
        let target = self.text_model.forward_token_ids_batch_at_physical(
            &token_ids,
            &positions,
            &mut text_states,
            &mut caches,
        )?;
        drop(text_states);
        drop(caches);

        // The device argmax has no logprobs, so a logprob row keeps the
        // batch on the per-row sampler.
        let batch_greedy = self.device_sampling_enabled()
            && states.iter().all(|state| {
                state.config.temperature <= 1e-5 && !state.track_history && !state.config.logprobs
            });
        let mut sampled = if batch_greedy {
            device_sampling::greedy(&target.logits.squeeze(1)?, self.tokenizer.vocab_size)?
        } else {
            Vec::with_capacity(state_count)
        };
        let mut logprob_entries = vec![None; state_count];
        if !batch_greedy {
            for (row, state) in states.iter_mut().enumerate() {
                state.pending_logprobs.clear();
                let history: &[u32] = if state.track_history {
                    &state.history_ids
                } else {
                    &[]
                };
                let row_logits = target.logits.i((row, 0))?;
                let (token, logprob_entry) = self.sample_next_token_with_logprobs(
                    &row_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?;
                if let Some(raw) = logprob_entry {
                    logprob_entries[row] =
                        Some(crate::models::shared::sampling::resolve_token_logprob(
                            &self.tokenizer.inner,
                            &raw,
                        )?);
                }
                sampled.push(token);
            }
        }
        let terminal_rows = states
            .iter()
            .zip(&sampled)
            .map(|(state, token)| {
                sample_finishes_row(
                    self.is_stop_token(*token, &state.config),
                    state.tokens_generated,
                    state.max_new_tokens,
                )
            })
            .collect::<Vec<_>>();

        let mtp_hidden = if let Some(head) = self.mtp_head.as_ref() {
            // The MTP forward also writes the scheduler-owned MTP KV domain
            // for the input token consumed by the target above. Even terminal
            // rows must participate so the multi-domain transaction can
            // commit; only their next-anchor materialization is unnecessary.
            let embeddings = self.text_model.embed_decode_token_ids(&sampled)?;
            let mut mtp_caches = states
                .iter_mut()
                .map(|state| {
                    state.mtp_physical_kv.as_mut().ok_or_else(|| {
                        Error::InferenceError("Qwen3.8 shared row lost its MTP cache".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let target_hidden = target.hidden_states.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3.8 MTP rows need the target hidden states, which graph decode omits"
                        .into(),
                )
            })?;
            let hidden =
                head.forward_steps_batch(&embeddings, target_hidden, &positions, &mut mtp_caches)?;
            Some(hidden)
        } else {
            None
        };

        let mut steps = Vec::with_capacity(state_count);
        for (row, state) in states.iter_mut().enumerate() {
            let next = sampled[row];
            if state.track_history {
                state.history_ids.push(next);
            }
            if let Some(entry) = logprob_entries[row].take() {
                if !terminal_rows[row] {
                    state.pending_logprobs.push(entry);
                }
            }
            if !terminal_rows[row] {
                if let Some(hidden) = mtp_hidden.as_ref() {
                    state.mtp_anchor_hidden = Some(hidden.i(row)?.unsqueeze(0)?);
                }
            }
            if terminal_rows[row] {
                state.mtp_anchor_hidden = None;
                state.pending_token = None;
            } else {
                state.pending_token = Some(next);
            }
            state.next_text_position = state.next_text_position.saturating_add(1);
            let delta = self.publish_token(state, next)?;
            debug_assert!(!terminal_rows[row] || state.finished);
            steps.push(self.decode_step_result(state, delta, 1));
        }
        Ok(steps)
    }

    /// DS9.4: one shared speculative envelope for continuous rows. Every row
    /// must be MTP-resident (cache + anchor + pending token) and non-terminal.
    /// The round depth is homogeneous (the minimum over active rows) so drafts
    /// batch through the MTP layer while verification and commit stay per row
    /// on each row's own physical cache. Rows exit raggedly: a row that
    /// finishes (stop token, output cap) drops out at the round boundary.
    pub(crate) fn decode_speculative_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
        input_budget: usize,
    ) -> Result<Vec<ChatDecodeStep>> {
        let head = self.mtp_head.as_ref().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 speculative batch requires a loaded MTP head".into())
        })?;
        let state_count = states.len();
        if state_count == 0 {
            return Ok(Vec::new());
        }
        let mut deltas = vec![String::new(); state_count];
        let mut committed = vec![0usize; state_count];
        let mut finished = vec![false; state_count];
        // Rows entering their first envelope carry the prefill bootstrap
        // token, which the solo quantum publishes as its own decode step
        // before any MTP round. Mirror that here: the anchor is emitted with
        // no KV commit and the row sits out the rest of this call, so a
        // scalar round never sees a non-decodable state and the row's budget
        // accounting matches the solo bootstrap step.
        let mut sat_out = vec![false; state_count];
        for row in 0..state_count {
            let state = &mut *states[row];
            finished[row] = state.finished;
            if let Some(anchor) = state.bootstrap_token.take() {
                if let Some(entry) = state.bootstrap_logprob.take() {
                    if !self.is_stop_token(anchor, &state.config) {
                        state.pending_logprobs.push(entry);
                    }
                }
                deltas[row].push_str(&self.publish_token(state, anchor)?);
                finished[row] = state.finished;
                sat_out[row] = !state.finished;
            }
        }
        let mut active: Vec<usize> = (0..state_count).collect();
        active.retain(|&row| {
            !finished[row] && !sat_out[row] && committed[row] < input_budget
        });

        while !active.is_empty() {
            // DS9.3: logprob requests need the scalar sampler's raw logits,
            // so they collapse the whole envelope to scalar rounds (the
            // registry drains their entries from the family states).
            let wants_logprobs = active
                .iter()
                .any(|&row| states[row].config.logprobs);
            // DS9.4: depth is per row. A homogeneous minimum would couple the
            // rows' proposal counts (and with them their sampled streams and
            // acceptance rng draws) to whatever co-batched rows happen to
            // have smaller remaining budgets; per-row depths keep every row's
            // round sequence identical to its solo quantum's.
            let depths: Vec<usize> = if wants_logprobs {
                vec![0; active.len()]
            } else {
                active
                    .iter()
                    .map(|&row| {
                        let state = &*states[row];
                        let remaining_budget = input_budget - committed[row];
                        let remaining_output = state
                            .max_new_tokens
                            .saturating_sub(state.tokens_generated)
                            .max(1);
                        state
                            .adaptive_mtp
                            .depth(remaining_output.min(remaining_budget))
                            .min(remaining_budget)
                            .min(remaining_output)
                    })
                    .collect()
            };
            let max_depth = depths.iter().copied().max().unwrap_or(0);

            if max_depth == 0 {
                // Scalar round: the shared one-token-per-row batched path.
                let mut subset = take_state_subset(states, &active);
                let steps = self.decode_step_batch(&mut subset)?;
                for (position, &row) in active.iter().enumerate() {
                    deltas[row].push_str(&steps[position].delta);
                    committed[row] += 1;
                    finished[row] = steps[position].finished;
                }
                active.retain(|&row| !finished[row] && committed[row] < input_budget);
                continue;
            }

            let round_started = std::time::Instant::now();
            // Per-row MTP checkpoints: the draft's KV writes are provisional
            // and rolled back before verification rewrites the positions.
            let mut mtp_checkpoints = Vec::with_capacity(active.len());
            for &row in &active {
                let state = &mut *states[row];
                let mtp = state.mtp_physical_kv.as_mut().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 speculative row lost its MTP cache".into())
                })?;
                mtp_checkpoints.push(mtp.logical_checkpoint());
            }
            let mut currents: Vec<Tensor> = Vec::with_capacity(active.len());
            for &row in &active {
                let state = &*states[row];
                let anchor = state.mtp_anchor_hidden.as_ref().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 speculative row has no recurrent anchor".into())
                })?;
                currents.push(anchor.clone());
            }

            // Batched recurrent draft: one MTP-layer forward per draft step
            // across the rows still drafting at that depth, with per-row
            // sampling between. Rows whose depth is already exhausted wait
            // for the round's verification with their own (possibly empty)
            // proposal list.
            let mut draft_tokens = vec![Vec::new(); active.len()];
            let mut draft_proposals = vec![Vec::new(); active.len()];
            let mut draft_histories = vec![Vec::new(); active.len()];
            for (position, &row) in active.iter().enumerate() {
                let state = &*states[row];
                if state.track_history {
                    draft_histories[position] = state.history_ids.clone();
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
                    let state = &mut *states[row];
                    let logits = self
                        .text_model
                        .project_with_shared_lm_head(&currents[position])?;
                    let mut values = logits_to_vec(&logits.i((0, 0))?)?;
                    truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                    if values.iter().any(|value| !value.is_finite()) {
                        draft_error_row = Some(position);
                        warn!(
                            position = states[row].next_text_position,
                            draft_depth = max_depth,
                            draft_step = step_index,
                            "Qwen3.8 speculative batch produced non-finite draft logits; continuing the envelope with target-only sampling"
                        );
                        break 'draft;
                    }
                    // The draft selects through the same sampler the solo
                    // rounds use, so batched and solo proposals agree given
                    // equal seeds and history.
                    let token = if state.config.temperature > 1e-5 {
                        let proposal = propose_speculative_draft(
                            &values,
                            &state.config,
                            &mut draft_histories[position],
                            &mut state.draft_rng,
                        )?;
                        let token_id = proposal.token_id;
                        draft_proposals[position].push(proposal);
                        token_id
                    } else {
                        sample_next_token(
                            &logits.i((0, 0))?,
                            self.tokenizer.vocab_size,
                            &state.config,
                            &draft_histories[position],
                            &mut state.draft_rng,
                        )?
                    };
                    if state.track_history {
                        draft_histories[position].push(token);
                    }
                    tokens_step.push(token);
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
                    let embeddings = self.text_model.embed_decode_token_ids(&advance_tokens)?;
                    let hiddens = Tensor::cat(
                        &advancing
                            .iter()
                            .map(|&position| &currents[position])
                            .collect::<Vec<_>>(),
                        0,
                    )?;
                    let positions = advancing
                        .iter()
                        .map(|&position| {
                            let row = active[position];
                            [states[row].next_text_position + step_index; 3]
                        })
                        .collect::<Vec<_>>();
                    let mut subset = take_state_subset(states, &active);
                    let mut caches = Vec::with_capacity(subset.len());
                    for (position, state) in subset.iter_mut().enumerate() {
                        if !advancing.contains(&position) {
                            continue;
                        }
                        let mtp = state.mtp_physical_kv.as_mut().ok_or_else(|| {
                            Error::InferenceError(
                                "Qwen3.8 speculative row lost its MTP cache".into(),
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
                }
            }

            for (position, &row) in active.iter().enumerate() {
                let state = &mut *states[row];
                let mtp = state.mtp_physical_kv.as_mut().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 speculative row lost its MTP cache".into())
                })?;
                mtp.restore_logical_checkpoint(mtp_checkpoints[position].clone())?;
                if draft_error_row == Some(position) {
                    state.adaptive_mtp.disable_after_nonfinite_draft();
                    record_mtp_nonfinite_draft_fallback();
                }
            }
            if draft_error_row.is_some() {
                // The envelope falls back to a scalar round together; the
                // offending row's policy is disabled for the rest of its run.
                let mut subset = take_state_subset(states, &active);
                let steps = self.decode_step_batch(&mut subset)?;
                for (position, &row) in active.iter().enumerate() {
                    deltas[row].push_str(&steps[position].delta);
                    committed[row] += 1;
                    finished[row] = steps[position].finished;
                }
                active.retain(|&row| !finished[row] && committed[row] < input_budget);
                continue;
            }

            let mut round_committed = vec![0usize; active.len()];
            let mut round_finished = vec![false; active.len()];
            for (position, &row) in active.iter().enumerate() {
                let state = &mut *states[row];
                let pending = state.pending_token.ok_or_else(|| {
                    Error::InferenceError(
                        "Qwen3.8 speculative verification has no pending token".into(),
                    )
                })?;
                let drafted = &draft_tokens[position];
                if drafted.is_empty() {
                    // Depth-0 row inside a mixed round: the solo quantum's
                    // scalar tail. One target forward, one sample, one MTP
                    // anchor step, no proposals to verify.
                    record_mtp_scalar_target_token();
                    let pending = state.pending_token.ok_or_else(|| {
                        Error::InferenceError(
                            "Qwen3.8 speculative scalar tail has no pending token".into(),
                        )
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
                    let history: &[u32] = if state.track_history {
                        &state.history_ids
                    } else {
                        &[]
                    };
                    let next = self.sample_next_token(
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
                        Error::InferenceError("Qwen3.8 speculative row lost its MTP cache".into())
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
                    round_committed[position] = 1;
                    let remaining = (input_budget - committed[row]).min(
                        state
                            .max_new_tokens
                            .saturating_sub(state.tokens_generated)
                            .max(1),
                    );
                    if state.adaptive_mtp.can_train(remaining) {
                        state.adaptive_mtp.observe(0, 1, round_started.elapsed(), remaining);
                    }
                    let delta = self.publish_token(state, next)?;
                    deltas[row].push_str(&delta);
                    if state.finished {
                        round_finished[position] = true;
                    }
                    continue;
                }
                let mut target_inputs = Vec::with_capacity(drafted.len() + 1);
                target_inputs.push(pending);
                target_inputs.extend_from_slice(drafted);
                let positions = (0..target_inputs.len())
                    .map(|offset| [state.next_text_position + offset; 3])
                    .collect::<Vec<_>>();
                let target_output = self.text_model.verify_token_ids_physical(
                    &target_inputs,
                    &positions,
                    &mut state.text_state,
                    &mut state.physical_kv,
                )?;
                let target_logits = self
                    .text_model
                    .project_target_hidden_span(&target_output.hidden_states)?;
                let mut verification_history = if state.track_history {
                    state.history_ids.clone()
                } else {
                    Vec::new()
                };
                let verification = if state.config.temperature > 1e-5 {
                    let mut host_rows = Vec::with_capacity(target_inputs.len());
                    for row_index in 0..target_inputs.len() {
                        let mut values = logits_to_vec(&target_logits.i((0, row_index))?)?;
                        truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                        host_rows.push(values);
                    }
                    verify_speculative_proposals(
                        &draft_proposals[position],
                        &host_rows,
                        &state.config,
                        &mut verification_history,
                        &mut state.rng,
                    )?
                } else {
                    let target_tokens = (0..target_inputs.len())
                        .map(|row_index| {
                            let logits = target_logits.i((0, row_index))?;
                            self.sample_next_token(
                                &logits,
                                self.tokenizer.vocab_size,
                                &state.config,
                                &[],
                                &mut state.rng,
                            )
                        })
                        .collect::<std::result::Result<Vec<_>, Error>>()?;
                    verify_greedy_token_prefix(
                        drafted,
                        &target_tokens,
                        &mut verification_history,
                    )?
                };
                let remaining_outputs =
                    state.max_new_tokens.saturating_sub(state.tokens_generated);
                let kept = canonical_emitted_prefix(
                    &verification.emitted_tokens,
                    remaining_outputs,
                    |token| self.is_stop_token(token, &state.config),
                );
                let canonical_count = kept.len();
                if canonical_count == 0 {
                    return Err(Error::InferenceError(
                        "Qwen3.8 speculative round verified an empty prefix".into(),
                    ));
                }
                let canonical_hidden = target_output.commit_prefix(
                    canonical_count,
                    &mut state.text_state,
                    &mut state.physical_kv,
                )?;
                if state.track_history {
                    state.history_ids.extend_from_slice(&kept);
                }
                let canonical_pairs = Qwen38MtpPairBatch::new(
                    self.text_model.embed_token_ids(&kept)?,
                    canonical_hidden,
                    positions[..canonical_count].to_vec(),
                )?;
                let mtp = state.mtp_physical_kv.as_mut().ok_or_else(|| {
                    Error::InferenceError("Qwen3.8 speculative row lost its MTP cache".into())
                })?;
                let canonical_mtp_hidden = head.forward_pairs(&canonical_pairs, mtp)?;
                state.mtp_anchor_hidden =
                    Some(canonical_mtp_hidden.narrow(1, canonical_count - 1, 1)?);
                state.pending_token = kept.last().copied();
                state.next_text_position += canonical_count;
                round_committed[position] = canonical_count;
                // Scheduler-limited tails do not train the controller, matching
                // the solo quantum's RoundTimer gating: the observation budget
                // is the round's own remaining slice, not the output cap.
                let remaining = (input_budget - committed[row]).min(
                    state
                        .max_new_tokens
                        .saturating_sub(state.tokens_generated)
                        .max(1),
                );
                if state.adaptive_mtp.can_train(remaining) {
                    state.adaptive_mtp.observe(
                        depths[position],
                        canonical_count,
                        round_started.elapsed(),
                        remaining,
                    );
                }
                record_mtp_round(
                    drafted.len(),
                    verification.accepted_draft_tokens,
                    verification.emitted_bonus_token() && canonical_count == target_inputs.len(),
                    target_inputs.len(),
                    0,
                );
                for token in &kept {
                    deltas[row].push_str(&self.publish_token(state, *token)?);
                    if state.finished {
                        break;
                    }
                }
                if state.finished || state.pending_token.is_none() {
                    round_finished[position] = true;
                }
            }
            for (position, &row) in active.iter().enumerate() {
                committed[row] += round_committed[position];
                finished[row] = round_finished[position];
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

    pub(crate) fn decode_quantum(
        &self,
        state: &mut ChatDecodeState,
        input_budget: usize,
    ) -> Result<ChatDecodeStep> {
        if state.replay.is_some() {
            return Err(Error::InvalidInput(
                "Qwen3.8 decode cannot run before replay completes".into(),
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

        if let Some(anchor) = state.bootstrap_token.take() {
            if let Some(entry) = state.bootstrap_logprob.take() {
                if !self.is_stop_token(anchor, &state.config) {
                    state.pending_logprobs.push(entry);
                }
            }
            let delta = self.publish_token(state, anchor)?;
            return Ok(self.decode_step_result(state, delta, 0));
        }

        if self.mtp_head.is_some() {
            return self.decode_mtp_quantum(state, input_budget.max(1));
        }
        if self.text_model.graph_decode() {
            return self.decode_graph_quantum(state, input_budget.max(1));
        }
        // One step per call without graph decode, as before it.
        self.decode_scalar_step(state)
    }

    /// One scalar decode step: forward the pending token (through graph
    /// decode when it resolved, on the batched single-row path), sample the
    /// next token from the output, publish it.
    fn decode_scalar_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        let mut input_tokens_committed = 0;
        if let Some(pending) = state.pending_token.take() {
            let position = [state.next_text_position; 3];
            state.unconsumed_output = Some(if self.text_model.graph_decode() {
                // Graph decode lives on the batched single-row path.
                self.text_model
                    .forward_token_ids_batch_at_physical(
                        &[pending],
                        &[position],
                        &mut [&mut state.text_state],
                        &mut [&mut state.physical_kv],
                    )?
                    .logits
                    .i((0, 0))?
            } else {
                self.text_model.forward_token_id_at_physical(
                    pending,
                    position,
                    &mut state.text_state,
                    &mut state.physical_kv,
                )?
            });
            state.next_text_position += 1;
            input_tokens_committed = 1;
        }

        let history: &[u32] = if state.track_history {
            &state.history_ids
        } else {
            &[]
        };
        let output = state
            .unconsumed_output
            .take()
            .ok_or_else(|| Error::InferenceError("missing Qwen3.8 output".into()))?;
        state.pending_logprobs.clear();
        let (next, logprob_entry) = self.sample_next_token_with_logprobs(
            &output,
            self.tokenizer.vocab_size,
            &state.config,
            history,
            &mut state.rng,
        )?;
        if state.track_history {
            state.history_ids.push(next);
        }
        let resolved = logprob_entry
            .map(|raw| {
                crate::models::shared::sampling::resolve_token_logprob(&self.tokenizer.inner, &raw)
            })
            .transpose()?;
        if let Some(entry) = resolved {
            if !self.is_stop_token(next, &state.config) {
                state.pending_logprobs.push(entry);
            }
        }
        state.pending_token = Some(next);
        let delta = self.publish_token(state, next)?;
        Ok(self.decode_step_result(state, delta, input_tokens_committed))
    }

    /// A graph-decode quantum: up to `budget` decode steps in one engine call.
    /// Once the whole-step graph replays they are pipelined (step k+1 queued
    /// before token k is read back); until then, or when the request cannot
    /// pipeline, they run one at a time. Reports one logprob per emitted
    /// token.
    fn decode_graph_quantum(
        &self,
        state: &mut ChatDecodeState,
        budget: usize,
    ) -> Result<ChatDecodeStep> {
        let mut delta = String::new();
        let mut committed = 0usize;
        let mut pipelined = false;
        // Each step replaces the pending logprobs; the quantum reports them all.
        let mut logprobs = Vec::new();
        let mut iteration = 0usize;
        while iteration < budget && !state.finished {
            if let Some(step) = self.decode_quantum_pipelined(state, budget - iteration)? {
                delta.push_str(&step.delta);
                committed += step.input_tokens_committed;
                pipelined = true;
                break;
            }
            let step = self.decode_scalar_step(state)?;
            iteration += 1;
            logprobs.append(&mut state.pending_logprobs);
            delta.push_str(&step.delta);
            committed += step.input_tokens_committed;
            if step.finished {
                break;
            }
        }
        state.pending_logprobs = logprobs;
        self.graph_quantum.record(budget, committed, pipelined);
        Ok(self.decode_step_result(state, delta, committed))
    }

    /// The rest of a quantum with each next step queued before the current
    /// token is read back (the whole-step graph replaying). Every step
    /// forwards the pending token and chooses the next one on the device
    /// (`pipeline::DeviceSelector`, the step-by-step device sampler's
    /// choice), which feeds the next step's input slot. The host reads token
    /// k while step k+1 runs, then commits step k's KV write and publishes
    /// the token as a scalar step does. A token that finishes the stream
    /// discards the queued step: its DeltaNet outputs are dropped (they are
    /// fresh buffers, and the row's state goes back to the one after the
    /// committed step) and its KV slot stays uncommitted (the step waits for
    /// its writes when dropped). `Ok(None)`: the request or the graph cannot
    /// pipeline (fewer than two steps left, logprobs, replay, an MTP head, no
    /// pending token, the device sampler off, or the graph not replaying);
    /// the caller decodes step by step.
    fn decode_quantum_pipelined(
        &self,
        state: &mut ChatDecodeState,
        budget: usize,
    ) -> Result<Option<ChatDecodeStep>> {
        struct Queued {
            step: super::text::EnqueuedGraphStep,
            token: pipeline::PendingToken,
            state_after: Qwen38TextRuntimeState,
        }
        let model = &self.text_model;
        let Some(pending) = state.pending_token else {
            return Ok(None);
        };
        // The output cap ends the stream exactly; nothing is queued past it.
        let budget = budget.min(state.max_new_tokens.saturating_sub(state.tokens_generated));
        if budget < 2
            || state.finished
            || state.replay.is_some()
            || state.config.logprobs
            || self.mtp_head.is_some()
            || state.unconsumed_output.is_some()
            || state.bootstrap_token.is_some()
            || self.pipelining_refusal().is_some()
            || !model.graph_step_ready()
        {
            return Ok(None);
        }
        let history: &[u32] = if state.track_history {
            &state.history_ids
        } else {
            &[]
        };
        let mut selector = pipeline::DeviceSelector::new(
            model.device(),
            self.tokenizer.vocab_size,
            &state.config,
            state.track_history,
            history,
        )?;
        let base = state.next_text_position;
        let Some(first) = model.enqueue_graph_step(
            Some(pending),
            base,
            &mut state.text_state,
            &state.physical_kv,
        )?
        else {
            return Ok(None);
        };
        state.pending_token = None;
        state.pending_logprobs.clear();
        let token_slot = model.graph_step_token().ok_or_else(|| {
            Error::InferenceError("the Qwen3.8 whole-step graph has no step buffer".into())
        })??;
        let draw = |rng: &mut SimpleRng, draws: bool| if draws { rng.next_f32() } else { 0.0 };
        let uniform = draw(&mut state.rng, selector.draws());
        let mut current = Queued {
            token: selector.select(&first.logits, &token_slot, uniform)?,
            step: first,
            state_after: state.text_state.clone(),
        };
        let mut launched = 1usize;
        let mut delta = String::new();
        let mut committed = 0usize;
        loop {
            let next = if launched < budget {
                match model.enqueue_graph_step(
                    None,
                    base + launched,
                    &mut state.text_state,
                    &state.physical_kv,
                )? {
                    Some(step) => {
                        launched += 1;
                        let uniform = draw(&mut state.rng, selector.draws());
                        Some(Queued {
                            token: selector.select(&step.logits, &token_slot, uniform)?,
                            step,
                            state_after: state.text_state.clone(),
                        })
                    }
                    None => None,
                }
            } else {
                None
            };
            let token = selector.wait(current.token)?;
            model.commit_enqueued_step(current.step, &mut state.physical_kv)?;
            state.next_text_position += 1;
            committed += 1;
            // Publish as the scalar step does.
            if state.track_history {
                state.history_ids.push(token);
            }
            state.pending_token = Some(token);
            delta.push_str(&self.publish_token(state, token)?);
            match next {
                // The queued step already forwards this token.
                Some(next) if !state.finished => {
                    state.pending_token = None;
                    current = next;
                }
                // The stream ended: discard the queued step.
                Some(discarded) => {
                    state.text_state = current.state_after;
                    drop(discarded);
                    break;
                }
                None => break,
            }
        }
        Ok(Some(self.decode_step_result(state, delta, committed)))
    }

    /// `cuda_graphs` in the diagnostics: graph decode's resolution, phase and
    /// counters, and the decode quantum it asks the scheduler for.
    /// Cross-turn prefix reuse: whether this load shares prefixes and why,
    /// where a chat turn's shared history ends, and the prompt tokens every
    /// prefill since reused or computed. The managed cache's own counters
    /// (`kv_cache.counters` and `kv_cache.models[]` in `/v1/metrics`) show
    /// hits, misses, snapshot publications and the snapshot budget.
    fn prefix_reuse_diagnostics(&self) -> serde_json::Value {
        let shared = self.prefix_caching && !self.mtp_policy.enabled();
        let reason = if shared {
            "engaged: committed KV pages and hybrid-state snapshots are shared at each turn's history boundary"
        } else if self.prefix_reuse.switched_off {
            "off: IZWI_QWEN38_PREFIX_REUSE turned it off for Qwen3.8"
        } else if !self.prefix_caching {
            "off: not engaged for this backend (the catalog cell is not supported here, or IZWI_ENABLE_PREFIX_CACHING=0 / IZWI_PREFIX_REUSE_AUTO=0)"
        } else {
            "off: the MTP head is loaded; its draft attention state has no prefix snapshot, so every domain stays private (CUDA leaves the head out by default; IZWI_QWEN38_MTP=1 loads it)"
        };
        let counters = &self.prefix_reuse;
        let load = |counter: &AtomicU64| counter.load(AtomicOrdering::Relaxed);
        serde_json::json!({
            "engaged": shared,
            "reason": reason,
            "requested": counters.requested,
            "switch": PREFIX_REUSE_ENV,
            "history_boundary": "before the prompt's final <|im_start|> (the assistant generation header)",
            "snapshot_interval_tokens": default_kv_page_size(),
            "attached_prefills": load(&counters.attached_prefills),
            "fresh_prefills": load(&counters.fresh_prefills),
            "reused_prompt_tokens": load(&counters.reused_prompt_tokens),
            "prefilled_prompt_tokens": load(&counters.prefilled_prompt_tokens),
            "last_prefill": {
                "reused_tokens": load(&counters.last_reused_tokens),
                "prompt_tokens": load(&counters.last_prompt_tokens),
            },
            "managed_counters": "runtime_metrics.kv_cache: counters.prefix_hits, prefix_misses, reused_tokens, tensor_snapshot_publishes, tensor_snapshot_attaches; models[].prefix_snapshots, prefix_snapshot_bytes, prefix_snapshot_budget_bytes",
        })
    }

    fn cuda_graphs_diagnostics(&self) -> serde_json::Value {
        let mut summary = self.text_model.cuda_graphs_summary();
        let mut quantum = self.graph_quantum.summary();
        quantum["preferred"] = serde_json::json!(self.preferred_decode_tokens());
        quantum["sustained"] = serde_json::json!(self.sustained_decode_quantum());
        quantum["switch"] = serde_json::json!(DECODE_QUANTUM_ENV);
        quantum["pipelining"] = match self.pipelining_refusal() {
            None => serde_json::json!({ "backend": "fused" }),
            Some(reason) => serde_json::json!({ "backend": "legacy", "reason": reason }),
        };
        summary["decode_quantum"] = quantum;
        summary
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

    fn observe_completed_mtp_timings(&self, state: &mut ChatDecodeState) {
        // Bounded event owners also participate in the quantum checkpoint. A
        // cancelled quantum restores the old observations and policy together.
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
                super::telemetry::record_mtp_completed_timing(elapsed);
            } else {
                index += 1;
            }
        }
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
        let _configured_depth = self.mtp_policy.draft_tokens().ok_or_else(|| {
            Error::InferenceError("Qwen3.8 MTP decode has a disabled policy".into())
        })?;
        let mut delta = String::new();
        let mut committed = 0usize;

        while committed < input_budget && !state.finished {
            let remaining = (input_budget - committed).min(
                state
                    .max_new_tokens
                    .saturating_sub(state.tokens_generated)
                    .max(1),
            );
            self.observe_completed_mtp_timings(state);
            let round_start = Instant::now();
            let timer = if state.adaptive_mtp.can_train(remaining) {
                timing::RoundTimer::start(self.text_model.device())
            } else {
                None
            };
            // DS9.3: speculative rounds commit drafted tokens whose raw
            // logits the scalar sampler never observed, so logprob requests
            // force depth 0 (the MTP domain still advances each round).
            let selected_depth = if state.config.logprobs {
                0
            } else {
                state.adaptive_mtp.depth(remaining)
            };
            if selected_depth == 0 {
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
                state.pending_logprobs.clear();
                let (next, logprob_entry) = self.sample_next_token_with_logprobs(
                    &logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    history,
                    &mut state.rng,
                )?;
                if state.track_history {
                    state.history_ids.push(next);
                }
                let resolved = logprob_entry
                    .map(|raw| {
                        crate::models::shared::sampling::resolve_token_logprob(
                            &self.tokenizer.inner,
                            &raw,
                        )
                    })
                    .transpose()?;
                if let Some(entry) = resolved {
                    if !self.is_stop_token(next, &state.config) {
                        state.pending_logprobs.push(entry);
                    }
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
                record_mtp_round_timing(0, 1, round_start.elapsed(), 0, remaining);
                self.observe_completed_mtp_timings(state);
                if let Some(pending) = timer.and_then(|timer| timer.finish(0, 1, remaining)) {
                    if state.mtp_timings.len() == 4 {
                        state.mtp_timings.remove(0);
                    }
                    state.mtp_timings.push(pending);
                }
                continue;
            }

            let depth = selected_depth;
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
            let stochastic_drafting = state.config.temperature > 1e-5;
            let mut draft_history = if state.track_history {
                state.history_ids.clone()
            } else {
                Vec::new()
            };
            let mut draft_proposals = Vec::with_capacity(depth.get());
            let device_sampling = self.device_sampling_enabled();
            let mut device_proposals = Vec::with_capacity(depth.get());
            let draft_rng_checkpoint = state.draft_rng.clone();
            let drafted = head.draft_recurrently_with_text(
                &self.text_model,
                anchor_hidden,
                depth,
                &continuation_positions,
                mtp,
                |_, logits| {
                    let logits = logits.i((0, 0))?;
                    if device_sampling {
                        if !stochastic_drafting {
                            let token = match device_sampling::sample_or_abort(
                                &logits,
                                self.tokenizer.vocab_size,
                                &state.config,
                                &draft_history,
                                &mut state.draft_rng,
                                "draft",
                            )? {
                                ControlFlow::Continue(token) => token,
                                ControlFlow::Break(error) => return Ok(ControlFlow::Break(error)),
                            };
                            record_sampling_bounded_cuda(true);
                            if state.track_history {
                                draft_history.push(token);
                            }
                            return Ok(ControlFlow::Continue(token));
                        }
                        let (token, q) = match device_sampling::propose_or_abort(
                            &logits.unsqueeze(0)?,
                            self.tokenizer.vocab_size,
                            &state.config,
                            &mut draft_history,
                            &mut state.draft_rng,
                        )? {
                            ControlFlow::Continue(proposal) => proposal,
                            ControlFlow::Break(error) => return Ok(ControlFlow::Break(error)),
                        };
                        device_proposals.push(q);
                        return Ok(ControlFlow::Continue(token));
                    }
                    let mut values = logits_to_vec(&logits)?;
                    if self.tokenizer.vocab_size == 0 || values.len() < self.tokenizer.vocab_size {
                        return Err(Error::InvalidInput(
                            "invalid Qwen3.8 draft sampling vocabulary".into(),
                        ));
                    }
                    truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                    if !values.iter().any(|value| value.is_finite()) {
                        return Ok(ControlFlow::Break(Error::InferenceError(
                            "No finite Qwen3.8 draft logits".into(),
                        )));
                    }
                    if !stochastic_drafting {
                        // The compatibility sampler already needs host logits
                        // for numerical classification; reuse this same row.
                        let logits = Tensor::from_vec(
                            values,
                            self.tokenizer.vocab_size,
                            &candle_core::Device::Cpu,
                        )?;
                        let token = sample_next_token(
                            &logits,
                            self.tokenizer.vocab_size,
                            &state.config,
                            &draft_history,
                            &mut state.draft_rng,
                        )?;
                        if state.track_history {
                            draft_history.push(token);
                        }
                        return Ok(ControlFlow::Continue(token));
                    }
                    let proposal = propose_speculative_draft(
                        &values,
                        &state.config,
                        &mut draft_history,
                        &mut state.draft_rng,
                    )?;
                    let token = proposal.token_id;
                    draft_proposals.push(proposal);
                    Ok(ControlFlow::Continue(token))
                },
            );
            mtp.restore_logical_checkpoint(mtp_checkpoint)?;
            let drafted = match drafted? {
                ControlFlow::Continue(drafted) => drafted,
                ControlFlow::Break(error) => {
                    // No target forward, target RNG, canonical history or
                    // output has changed. Discard even earlier valid proposals
                    // from this round and never sample this MTP state again.
                    state.draft_rng = draft_rng_checkpoint;
                    state.adaptive_mtp.disable_after_nonfinite_draft();
                    record_mtp_nonfinite_draft_fallback();
                    warn!(
                        position = state.next_text_position,
                        draft_depth = depth.get(),
                        error = %error,
                        "Qwen3.8 MTP produced no finite draft logits; continuing this request with target-only sampling"
                    );
                    continue;
                }
            };

            let pending = state.pending_token.ok_or_else(|| {
                Error::InferenceError("Qwen3.8 MTP verification has no pending token".into())
            })?;
            let mut target_inputs = Vec::with_capacity(drafted.token_ids.len() + 1);
            target_inputs.push(pending);
            target_inputs.extend_from_slice(&drafted.token_ids);
            let positions = (0..target_inputs.len())
                .map(|offset| [state.next_text_position + offset; 3])
                .collect::<Vec<_>>();
            let target_output = self.text_model.verify_token_ids_physical(
                &target_inputs,
                &positions,
                &mut state.text_state,
                &mut state.physical_kv,
            )?;
            let target_logits = self
                .text_model
                .project_target_hidden_span(&target_output.hidden_states)?;
            let mut verification_history = if state.track_history {
                state.history_ids.clone()
            } else {
                Vec::new()
            };
            let verification = if stochastic_drafting && device_sampling {
                device_sampling::verify(
                    &drafted.token_ids,
                    &device_proposals,
                    &target_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    &mut verification_history,
                    &mut state.rng,
                )?
            } else if !stochastic_drafting && device_sampling && state.track_history {
                device_sampling::verify_greedy(
                    &drafted.token_ids,
                    &target_logits,
                    self.tokenizer.vocab_size,
                    &state.config,
                    &mut verification_history,
                )?
            } else if !stochastic_drafting && !state.track_history {
                let target_tokens = if device_sampling {
                    device_sampling::greedy(&target_logits.squeeze(0)?, self.tokenizer.vocab_size)?
                } else {
                    (0..target_inputs.len())
                        .map(|row| {
                            let logits = target_logits.i((0, row))?;
                            self.sample_next_token(
                                &logits,
                                self.tokenizer.vocab_size,
                                &state.config,
                                &[],
                                &mut state.rng,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?
                };
                verify_greedy_token_prefix(
                    &drafted.token_ids,
                    &target_tokens,
                    &mut verification_history,
                )?
            } else {
                let mut host_rows = Vec::with_capacity(target_inputs.len());
                for row in 0..target_inputs.len() {
                    let mut values = logits_to_vec(&target_logits.i((0, row))?)?;
                    truncate_logits_to_vocab(&mut values, self.tokenizer.vocab_size);
                    host_rows.push(values);
                }
                if stochastic_drafting {
                    verify_speculative_proposals(
                        &draft_proposals,
                        &host_rows,
                        &state.config,
                        &mut verification_history,
                        &mut state.rng,
                    )?
                } else {
                    verify_speculative_prefix(
                        &drafted.token_ids,
                        &host_rows,
                        &state.config,
                        &mut verification_history,
                        &mut state.rng,
                    )?
                }
            };

            let remaining_outputs = state.max_new_tokens.saturating_sub(state.tokens_generated);
            let kept = canonical_emitted_prefix(
                &verification.emitted_tokens,
                remaining_outputs,
                |token| self.is_stop_token(token, &state.config),
            );
            let canonical_count = kept.len();
            let canonical_hidden = target_output.commit_prefix(
                canonical_count,
                &mut state.text_state,
                &mut state.physical_kv,
            )?;
            if state.track_history {
                state.history_ids.extend_from_slice(&kept);
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
                verification.emitted_bonus_token() && canonical_count == target_inputs.len(),
                target_inputs.len(),
                0,
            );
            record_mtp_round_timing(
                selected_depth,
                canonical_count,
                round_start.elapsed(),
                if canonical_count < target_inputs.len() {
                    canonical_count
                } else {
                    0
                },
                remaining,
            );
            self.observe_completed_mtp_timings(state);
            if let Some(pending) =
                timer.and_then(|timer| timer.finish(selected_depth, canonical_count, remaining))
            {
                if state.mtp_timings.len() == 4 {
                    state.mtp_timings.remove(0);
                }
                state.mtp_timings.push(pending);
            }
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

    /// Prefill the target and, when enabled, stream the shifted MTP prompt in
    /// the same bounded chunks. Only the final target hidden row is retained.
    ///
    /// A prompt of N tokens contributes N-1 known shifted MTP pairs while the
    /// target is running. The final pair depends on the sampled target anchor
    /// and is appended by `start_decode_state_physical` after this returns.
    fn prefill_text_and_mtp_physical(
        &self,
        prepared: &Qwen38PreparedPrompt,
        text_state: &mut Qwen38TextRuntimeState,
        cache: &mut PhysicalPagedKvCache,
        mut mtp: Option<(&Qwen38MtpHead, &mut PhysicalPagedKvCache)>,
    ) -> Result<(Tensor, Tensor)> {
        let mut logits = None;
        let mut final_hidden = None;
        let end = prepared.prompt_ids.len();
        let mut chunk_start = 0;
        let chunk_size = self.prefill_chunk_size;
        while chunk_start < end {
            let chunk_end = (chunk_start + chunk_size).min(end);
            let compute_logits = chunk_end == end;
            if let Some(output) = self.text_model.prefill_token_ids_with_hidden_physical(
                &prepared.prompt_ids[chunk_start..chunk_end],
                &prepared.prompt_positions[chunk_start..chunk_end],
                text_state,
                cache,
                compute_logits,
            )? {
                let known_rows = known_mtp_rows(chunk_start, chunk_end, end);
                if known_rows > 0 {
                    if let Some((head, mtp_cache)) = mtp.as_mut() {
                        let embeddings = self.text_model.embed_token_ids(
                            &prepared.prompt_ids[chunk_start + 1..chunk_start + 1 + known_rows],
                        )?;
                        let predecessor_hidden = output.hidden_states.narrow(1, 0, known_rows)?;
                        let pairs = Qwen38MtpPairBatch::new(
                            embeddings,
                            predecessor_hidden,
                            prepared.prompt_positions[chunk_start..chunk_start + known_rows]
                                .to_vec(),
                        )?;
                        head.forward_pairs(&pairs, mtp_cache)?;
                    }
                }
                if chunk_end == end {
                    final_hidden = Some(output.hidden_states.narrow(
                        1,
                        output.hidden_states.dim(1)?.saturating_sub(1),
                        1,
                    )?);
                }
                if let Some(chunk_logits) = output.logits {
                    logits = Some(chunk_logits);
                }
            }
            chunk_start = chunk_end;
        }
        let logits = logits.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 physical prefill produced no logits".into())
        })?;
        let final_hidden = final_hidden.ok_or_else(|| {
            Error::InferenceError("Qwen3.8 physical prefill produced no hidden state".into())
        })?;
        Ok((logits, final_hidden))
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
        let history_boundary = history_boundary(&prompt_ids, self.tokenizer.specials.im_start);
        Ok(Qwen38PreparedPrompt {
            next_text_position: prompt_positions.len(),
            prompt_ids,
            prompt_positions,
            history_boundary,
        })
    }
}

/// Pure per-row geometry bound, excluding model-owned graph cache capacity.
/// `Some(rows)` includes MTP transient and conservative verification storage.
fn continuous_decode_workspace_per_row_bytes(
    cfg: &Qwen38TextConfig,
    vocab: usize,
    mtp_verification_rows: Option<usize>,
) -> Result<u64> {
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
    let elements = hidden
        .and_then(|hidden| hidden.checked_mul(8))
        .and_then(|base| base.checked_add(ff?.checked_mul(2)?))
        .and_then(|base| base.checked_add(q?))
        .and_then(|base| base.checked_add(kv?.checked_mul(2)?))
        .and_then(|base| base.checked_add(conv?))
        .and_then(|target| {
            if mtp_verification_rows.is_some() {
                target.checked_add(
                    hidden?
                        .checked_mul(8)?
                        .checked_add(ff?.checked_mul(2)?)?
                        .checked_add(q?)?
                        .checked_add(kv?.checked_mul(2)?)?,
                )
            } else {
                Some(target)
            }
        })
        .ok_or_else(|| {
            Error::Overloaded("continuous decode workspace estimate overflow".to_string())
        })?;
    // The recurrent/attention collation tensors use F32 in portable and
    // state-update paths even when projection activations use F16/BF16.
    let transient = elements.checked_mul(4).ok_or_else(|| {
        Error::Overloaded("continuous decode workspace byte estimate overflow".to_string())
    })?;
    let retained = match mtp_verification_rows {
        Some(rows) => verification_workspace_bytes(cfg, vocab, rows)?,
        None => 0,
    };
    // CUDA graph caches are model-owned resident/deferred reservations in
    // runtime/lifecycle/qwen38_memory.rs, shared across every request row.
    // Charging them here would reserve the same cache again for each row.
    transient
        .checked_add(retained)
        .ok_or_else(|| Error::Overloaded("Qwen3.8 verification workspace overflow".into()))
}

/// Conservative bound for all simultaneously retained verification state,
/// compact intermediates, recovery copies, probability rows and scratch.
/// Count every layer as linear so unusual layer schedules cannot underprice it.
fn verification_workspace_bytes(cfg: &Qwen38TextConfig, vocab: usize, rows: usize) -> Result<u64> {
    let checked = || -> Option<u64> {
        let rows = u64::try_from(rows.min(4)).ok()?;
        let layers = u64::try_from(cfg.block_count).ok()?;
        let inner = u64::try_from(cfg.ssm_inner_size).ok()?;
        let key = u64::try_from(cfg.ssm_state_size).ok()?;
        let heads = u64::try_from(cfg.ssm_group_count).ok()?;
        let value_heads = u64::try_from(cfg.ssm_time_step_rank).ok()?;
        let recurrent = key.checked_mul(inner)?;
        let conv = key.checked_mul(heads)?.checked_mul(2)?.checked_add(inner)?;
        let history = conv.checked_mul(cfg.ssm_conv_kernel.saturating_sub(1) as u64)?;
        let compact = conv
            .checked_mul(2)?
            .checked_add(value_heads.checked_mul(2)?)?
            .checked_mul(rows)?;
        // Initial checkpoint and reconstructed prefix may coexist with the
        // final verified state already counted in the session's base claim.
        let retained = recurrent
            .checked_mul(2)?
            .checked_add(history)?
            .checked_add(compact)?
            .checked_mul(layers)?;
        let recurrence_scratch = recurrent.checked_mul(6)?;
        let logits = (vocab as u64).checked_mul(rows)?.checked_mul(6)?;
        retained
            .checked_add(recurrence_scratch)?
            .checked_add(logits)?
            .checked_add((cfg.embedding_length as u64).checked_mul(rows)?)?
            .checked_mul(4)
    };
    checked()
        .ok_or_else(|| Error::Overloaded("Qwen3.8 verification storage estimate overflow".into()))
}

fn build_text_positions(token_count: usize) -> Vec<[usize; 3]> {
    (0..token_count).map(|idx| [idx; 3]).collect()
}

fn known_mtp_rows(chunk_start: usize, chunk_end: usize, prompt_len: usize) -> usize {
    chunk_end
        .min(prompt_len.saturating_sub(1))
        .saturating_sub(chunk_start)
}

fn canonical_emitted_prefix(
    tokens: &[u32],
    remaining_outputs: usize,
    is_stop: impl Fn(u32) -> bool,
) -> Vec<u32> {
    let mut kept = Vec::with_capacity(tokens.len().min(remaining_outputs));
    if remaining_outputs == 0 {
        return kept;
    }
    for &token in tokens {
        kept.push(token);
        if is_stop(token) || kept.len() >= remaining_outputs {
            break;
        }
    }
    kept
}

fn sample_finishes_row(
    is_stop_token: bool,
    tokens_generated: usize,
    max_new_tokens: usize,
) -> bool {
    is_stop_token || tokens_generated.saturating_add(1) >= max_new_tokens
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
            // The template's tool turn: a user turn wrapping the result.
            ChatRole::Tool => {
                prompt.push_str("<|im_start|>user\n<tool_response>\n");
                prompt.push_str(message.content.trim());
                prompt.push_str("\n</tool_response><|im_end|>\n");
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
    let host = if logits.device().is_cuda() {
        logits.to_device(&candle_core::Device::Cpu)?
    } else {
        logits.clone()
    };
    let raw_values = logits_to_vec(&host)?;
    let mut values = raw_values.clone();
    truncate_logits_to_vocab(&mut values, vocab_size);
    if values.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.8 sampler received no in-vocabulary logits".to_string(),
        ));
    }
    let (logsumexp, top) =
        crate::models::shared::sampling::raw_logprobs_stats(&values, config.top_logprobs)?;
    let token = sample_next_token(&host, vocab_size, config, history, rng)?;
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
                if cuda_sampling_attempted {
                    record_sampling_bounded_cuda(true);
                }
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

fn ratio_or_none(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
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
        ((self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32)
            .min(f32::from_bits(1.0f32.to_bits() - 1))
    }
}

/// DS9.4: reborrows the given rows out of the envelope for a sub-batch call.
/// Row ids must be strictly increasing. The walk advances one row per take,
/// so each split happens at `id - taken_count` rows into the shrinking rest.
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

#[cfg(test)]
mod tests {
    use crate::models::shared::chat::ChatRequestConfig;

    use super::*;

    // Dimensions enforced by native::validate_hf_config for the shipped 27B
    // checkpoint. No tokenizer, weights, device, or model load is needed.
    fn shipped_workspace_config() -> Qwen38TextConfig {
        Qwen38TextConfig {
            architecture: "qwen3_5".into(),
            block_count: 64,
            context_length: 262_144,
            embedding_length: 5_120,
            feed_forward_length: 17_408,
            attention_head_count: 24,
            attention_head_count_kv: 4,
            attention_key_length: 256,
            attention_value_length: 256,
            rope_dimension_sections: vec![11, 11, 10],
            rope_dimension_count: 64,
            rope_freq_base: 10_000_000.0,
            attention_layer_norm_rms_epsilon: 1e-6,
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6_144,
            full_attention_interval: 4,
        }
    }

    #[test]
    fn shipped_decode_workspace_prices_adaptive_fixed_and_disabled_mtp() {
        let cfg = shipped_workspace_config();
        // Adaptive CUDA reserves four target positions even when its initial
        // configured draft depth is one; fixed depth one needs two positions.
        for (rows, verification, total) in
            [(4, 474_382_336, 475_144_192), (2, 451_887_104, 452_648_960)]
        {
            assert_eq!(
                verification_workspace_bytes(&cfg, 248_320, rows).unwrap(),
                verification,
            );
            assert_eq!(
                continuous_decode_workspace_per_row_bytes(&cfg, 248_320, Some(rows)).unwrap(),
                total,
            );
            // Only target/MTP transient geometry is added to verification;
            // the two model-owned 8 MiB graph caches must not reappear here.
            assert_eq!(total - verification, 761_856);
        }
        assert_eq!(
            continuous_decode_workspace_per_row_bytes(&cfg, 248_320, None).unwrap(),
            401_408,
        );
        // Disabled MTP must not evaluate unused verification geometry.
        assert_eq!(
            continuous_decode_workspace_per_row_bytes(&cfg, usize::MAX, None).unwrap(),
            401_408,
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn decode_workspace_rejects_geometry_and_verification_overflow() {
        let cfg = shipped_workspace_config();
        let mut malformed = Vec::new();
        let mut hidden = cfg.clone();
        hidden.embedding_length = usize::MAX;
        malformed.push(hidden);
        let mut attention = cfg.clone();
        attention.attention_head_count = usize::MAX;
        malformed.push(attention);
        let mut convolution = cfg.clone();
        convolution.ssm_group_count = usize::MAX;
        malformed.push(convolution);
        let mut layers = cfg.clone();
        layers.block_count = usize::MAX;
        malformed.push(layers);
        for cfg in malformed {
            assert!(matches!(
                continuous_decode_workspace_per_row_bytes(&cfg, 248_320, Some(4)),
                Err(Error::Overloaded(_)),
            ));
        }
        assert!(matches!(
            continuous_decode_workspace_per_row_bytes(&cfg, usize::MAX, Some(4)),
            Err(Error::Overloaded(_)),
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn decode_workspace_rejects_f32_byte_conversion_overflow() {
        let mut cfg = shipped_workspace_config();
        // The element sum fits u64; converting that sum to F32 bytes does not.
        cfg.embedding_length = usize::MAX / 32;
        let error = continuous_decode_workspace_per_row_bytes(&cfg, 248_320, None)
            .expect_err("F32 byte count must not wrap");
        assert!(matches!(
            error,
            Error::Overloaded(message)
                if message == "continuous decode workspace byte estimate overflow",
        ));
    }

    #[test]
    fn canonical_prefix_stops_at_each_eos_and_output_budget() {
        let tokens = [10, 11, 12, 13];
        for budget in 0..=4 {
            assert_eq!(
                canonical_emitted_prefix(&tokens, budget, |_| false),
                tokens[..budget]
            );
            for eos in tokens {
                let count = tokens.iter().position(|token| *token == eos).unwrap() + 1;
                assert_eq!(
                    canonical_emitted_prefix(&tokens, budget, |token| token == eos),
                    tokens[..budget.min(count)]
                );
            }
        }
    }

    #[test]
    fn unit_float_upper_edge_remains_strictly_below_one() {
        let largest = f32::from_bits(1.0f32.to_bits() - 1);
        for value in [0, 1, u32::MAX - 128, u32::MAX - 1, u32::MAX] {
            let converted = ((value as f64 / (u32::MAX as f64 + 1.0)) as f32).min(largest);
            assert!((0.0..1.0).contains(&converted));
        }
    }

    #[test]
    fn streamed_mtp_prefill_covers_every_shifted_prompt_row_once() {
        let prompt_len = 1_025;
        let chunk_size = 256;
        let mut covered = 0;
        let mut chunk_start = 0;
        while chunk_start < prompt_len {
            let chunk_end = (chunk_start + chunk_size).min(prompt_len);
            covered += known_mtp_rows(chunk_start, chunk_end, prompt_len);
            chunk_start = chunk_end;
        }

        assert_eq!(covered, prompt_len - 1);
        assert_eq!(known_mtp_rows(0, 1, 1), 0);
        assert_eq!(known_mtp_rows(0, 1, 2), 1);
        assert_eq!(known_mtp_rows(768, 1_024, prompt_len), 256);
        assert_eq!(known_mtp_rows(1_024, 1_025, prompt_len), 0);

        for boundaries in [
            vec![0, 1, prompt_len],
            vec![0, 17, 511, 512, 1_024, prompt_len],
            (0..=prompt_len).collect::<Vec<_>>(),
        ] {
            let covered = boundaries
                .windows(2)
                .map(|span| known_mtp_rows(span[0], span[1], prompt_len))
                .sum::<usize>();
            assert_eq!(covered, prompt_len - 1, "boundaries={boundaries:?}");
        }
    }

    #[test]
    fn prefill_chunk_follows_the_backend_and_the_prefill_gemm_with_an_override() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var(PREFILL_CHUNK_ENV);
        assert_eq!(qwen38_prefill_chunk_tokens(BackendKind::Cuda, true), 1024);
        assert_eq!(qwen38_prefill_chunk_tokens(BackendKind::Cuda, false), 256);
        assert_eq!(qwen38_prefill_chunk_tokens(BackendKind::Metal, true), 256);
        assert_eq!(qwen38_prefill_chunk_tokens(BackendKind::Cpu, true), 256);
        for (value, expected) in [("512", 512), ("0", 1024), ("junk", 1024), ("999999", 2048)] {
            std::env::set_var(PREFILL_CHUNK_ENV, value);
            assert_eq!(
                qwen38_prefill_chunk_tokens(BackendKind::Cuda, true),
                expected,
                "{value}"
            );
        }
        std::env::set_var(PREFILL_CHUNK_ENV, "64");
        assert_eq!(qwen38_prefill_chunk_tokens(BackendKind::Cpu, false), 64);
        std::env::remove_var(PREFILL_CHUNK_ENV);
        // Admission prices one decode row's activation bound per prefill
        // token: 392 MiB for a 1,024-token chunk of the 27B trunk.
        let cfg = shipped_workspace_config();
        assert_eq!(
            prefill_working_set_bytes(&cfg, 1024).unwrap(),
            401_408 * 1024
        );
        assert_eq!(prefill_working_set_bytes(&cfg, 256).unwrap(), 401_408 * 256);
    }

    #[test]
    fn terminal_mtp_rows_are_identified_before_anchor_retention() {
        assert!(sample_finishes_row(true, 0, 32));
        assert!(sample_finishes_row(false, 31, 32));
        assert!(sample_finishes_row(false, usize::MAX, usize::MAX));
        assert!(!sample_finishes_row(false, 30, 32));
    }

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

    /// The first-turn boundary is where requests sharing only a system
    /// prompt diverge: the second `<|im_start|>`, before the history boundary.
    #[test]
    fn preamble_boundary_ends_the_first_turn() {
        let tokenizer = byte_level_tokenizer();
        let im_start = tokenizer.specials.im_start;
        let prepared = |ids: Vec<u32>| {
            let positions = build_text_positions(ids.len());
            let boundary = history_boundary(&ids, im_start);
            crate::models::registry::NativeChatPreparedPrompt::Qwen38(Qwen38PreparedPrompt {
                next_text_position: ids.len(),
                prompt_positions: positions,
                prompt_ids: ids,
                history_boundary: boundary,
            })
        };
        // System turn, user turn, assistant header.
        let shared = prepared(vec![im_start, 1, 2, im_start, 3, 4, im_start, 5]);
        assert_eq!(shared.managed_snapshot_boundary(), Some(6));
        assert_eq!(shared.managed_preamble_boundary(), Some(3));
        // A single user turn has nothing before its history boundary.
        let single = prepared(vec![im_start, 3, 4, im_start, 5]);
        assert_eq!(single.managed_snapshot_boundary(), Some(3));
        assert_eq!(single.managed_preamble_boundary(), None);
        // No declared history: no first-turn boundary either.
        let bare = prepared(vec![1, 2, 3]);
        assert_eq!(bare.managed_preamble_boundary(), None);
    }

    #[test]
    fn prepared_prompt_exposes_ids_and_reuses_the_prepared_value() {
        let prepared = Qwen38PreparedPrompt {
            prompt_ids: vec![1, 2, 3],
            prompt_positions: build_text_positions(3),
            next_text_position: 3,
            history_boundary: None,
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
    fn prefix_reuse_switch_is_a_strict_boolean_defaulting_on() {
        assert!(qwen38_prefix_reuse_switch(None).unwrap());
        for on in ["1", "true", " YES ", "on"] {
            assert!(qwen38_prefix_reuse_switch(Some(on)).unwrap(), "{on}");
        }
        for off in ["0", "false", "No", "off"] {
            assert!(!qwen38_prefix_reuse_switch(Some(off)).unwrap(), "{off}");
        }
        let error = qwen38_prefix_reuse_switch(Some("maybe")).unwrap_err();
        assert!(error.to_string().contains(PREFIX_REUSE_ENV), "{error}");
    }

    #[test]
    fn prefix_reuse_counts_attached_and_fresh_prefill_tokens() {
        let reuse = Qwen38PrefixReuse::new(true, false);
        reuse.record_prefill(0, 100);
        reuse.record_prefill(64, 180);
        let load = |counter: &AtomicU64| counter.load(AtomicOrdering::Relaxed);
        assert_eq!(load(&reuse.fresh_prefills), 1);
        assert_eq!(load(&reuse.attached_prefills), 1);
        assert_eq!(load(&reuse.reused_prompt_tokens), 64);
        assert_eq!(load(&reuse.prefilled_prompt_tokens), 100 + 116);
        assert_eq!(load(&reuse.last_reused_tokens), 64);
        assert_eq!(load(&reuse.last_prompt_tokens), 180);
    }

    #[test]
    fn mtp_policy_is_enabled_at_depth_one_by_default() {
        let policy = Qwen38MtpPolicy::resolve(None, None).unwrap();
        const { assert!(DEFAULT_MTP_ENABLED) };
        assert_eq!(
            policy,
            Qwen38MtpPolicy::Enabled {
                draft_tokens: DEFAULT_MTP_DRAFT_TOKENS
            }
        );
        assert!(policy.enabled());
        assert_eq!(policy.draft_tokens(), Some(DEFAULT_MTP_DRAFT_TOKENS));
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
    fn mtp_diagnostic_ratios_do_not_report_nan_before_execution() {
        assert_eq!(ratio_or_none(0, 0), None);
        assert_eq!(ratio_or_none(3, 4), Some(0.75));
    }

    #[test]
    fn cuda_bf16_kv_defaults_on_with_an_explicit_opt_out() {
        for disabled in [Some(""), Some("0"), Some("false"), Some("no"), Some("off")] {
            let provider =
                Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((8, 0)), disabled);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaF16Fallback);
            assert_eq!(provider.dtype(), DType::F16);
            assert!(provider.fallback_reason().is_some());
        }
        for enabled in [None, Some("1"), Some("true"), Some(" YES "), Some("on")] {
            let provider =
                Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((8, 0)), enabled);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaBf16);
            assert_eq!(provider.dtype(), DType::BF16);
            assert!(provider.fallback_reason().is_none());
        }
        assert_eq!(
            Qwen38KvStorageProvider::select(BackendKind::Cuda, Some((8, 0)), Some("invalid")),
            Qwen38KvStorageProvider::CudaF16Fallback
        );
    }

    #[test]
    fn cuda_bf16_kv_requires_observed_ampere_or_newer_capability() {
        for compute_capability in [None, Some((7, 5))] {
            for requested in [None, Some("1")] {
                let provider = Qwen38KvStorageProvider::select(
                    BackendKind::Cuda,
                    compute_capability,
                    requested,
                );
                assert_eq!(provider, Qwen38KvStorageProvider::CudaF16CapabilityFallback);
                assert_eq!(provider.dtype(), DType::F16);
                assert!(provider
                    .fallback_reason()
                    .expect("capability fallback reason")
                    .contains("8.0 or newer"));
            }
        }
        for capability in [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0)] {
            let provider =
                Qwen38KvStorageProvider::select(BackendKind::Cuda, Some(capability), None);
            assert_eq!(provider, Qwen38KvStorageProvider::CudaBf16);
            assert_eq!(provider.dtype(), DType::BF16);
            assert_eq!(provider.dtype().size_in_bytes(), DType::F16.size_in_bytes());
        }
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
    fn diagnostics_follow_the_residency_policy_on_every_backend() {
        use super::super::residency::Qwen38ResidencyPolicy;
        use super::super::text::Qwen38ProjectionRepresentation;
        let performance = crate::performance::PerformanceConfig::default();
        // CUDA without its device gates (the static estimate) keeps Q8_0.
        let cuda = qwen38_representation_diagnostics(&Qwen38ResidencyPolicy::resolve_static(
            BackendKind::Cuda,
            &performance,
        ));
        assert_eq!(
            cuda.resident_representation,
            "q8_0_requantized_projections_with_dense_bf16"
        );
        assert_eq!(cuda.runtime_compute_dtype, "bf16");
        assert!(
            cuda.fp8_execution_mode.starts_with("q8_0_fallback:"),
            "{}",
            cuda.fp8_execution_mode
        );
        assert!(cuda.fallback_reason.contains("weight_scale_inv"));
        assert_eq!(
            Qwen38ProjectionRepresentation::PackedQ8WithDenseF16.as_str(),
            "q8_0_requantized_projections_with_dense_f16"
        );
        assert_eq!(
            Qwen38ProjectionRepresentation::PackedQ8WithDenseF16.compute_dtype(),
            "f16"
        );

        // Native block FP8 reports its self-checked mode.
        let native = qwen38_representation_diagnostics(
            &Qwen38ResidencyPolicy::forced_native_fp8_for_tests(ProjectionMaterialization::BF16),
        );
        assert_eq!(
            native.resident_representation,
            "native_block_fp8_with_q8_fallback_dense_bf16"
        );
        assert_eq!(
            native.fp8_execution_mode,
            "native_block_fp8_auto_self_checked"
        );
        assert!(native.fallback_reason.contains("not runtime-certified"));

        // The CPU keeps Q8_0 projections, or its F32 expansion when told to.
        let cpu = qwen38_representation_diagnostics(&Qwen38ResidencyPolicy::resolve_static(
            BackendKind::Cpu,
            &performance,
        ));
        assert_eq!(
            cpu.resident_representation,
            "q8_0_projections_with_dense_f32"
        );
        assert_eq!(cpu.runtime_compute_dtype, "f32");
        assert_eq!(cpu.fp8_execution_mode, "q8_0_cpu_requantized");
        assert!(cpu
            .fallback_reason
            .contains("IZWI_CPU_PROJECTION_BACKEND=expanded"));
        let mut cpu_expanded = crate::performance::PerformanceConfig::default();
        cpu_expanded.cpu.projection_backend = crate::performance::CpuProjectionBackend::Expanded;
        let cpu = qwen38_representation_diagnostics(&Qwen38ResidencyPolicy::resolve_static(
            BackendKind::Cpu,
            &cpu_expanded,
        ));
        assert_eq!(cpu.resident_representation, "expanded_f32");
        assert_eq!(cpu.fp8_execution_mode, "expanded_fallback");
        assert_eq!(
            cpu.fallback_reason,
            "cpu.projection_backend=expanded: native block-FP8 GEMM is not runtime-certified; using the scale-exact expanded path"
        );
        // Metal keeps raw block FP8 in a Metal build, and expands to F16
        // when told to (or without the Metal kernels).
        let metal = qwen38_representation_diagnostics(&Qwen38ResidencyPolicy::resolve_static(
            BackendKind::Metal,
            &performance,
        ));
        if cfg!(feature = "metal") {
            assert_eq!(
                metal.resident_representation,
                "native_block_fp8_with_dense_f16"
            );
            assert_eq!(
                metal.fp8_execution_mode,
                "native_block_fp8_auto_self_checked"
            );
            assert!(metal.fallback_reason.contains("transient F16"));
        } else {
            assert_eq!(metal.resident_representation, "expanded_f16");
            assert!(metal.fallback_reason.contains("need a Metal build"));
        }
        assert_eq!(metal.runtime_compute_dtype, "f16");
        let mut expanded = crate::performance::PerformanceConfig::default();
        expanded.metal.projection_backend = crate::performance::MetalProjectionBackend::Expanded;
        let metal = qwen38_representation_diagnostics(&Qwen38ResidencyPolicy::resolve_static(
            BackendKind::Metal,
            &expanded,
        ));
        assert_eq!(metal.resident_representation, "expanded_f16");
        assert_eq!(metal.fp8_execution_mode, "expanded_fallback");
        assert_eq!(metal.fallback_reason, "metal.projection_backend=expanded");
        assert_eq!(
            Qwen38ProjectionRepresentation::ExpandedBf16.as_str(),
            "expanded_bf16"
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

    fn byte_level_char(byte: u8) -> char {
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
            .expect("the byte-level alphabet covers every byte");
        char::from_u32(codepoints[index]).expect("valid byte-level codepoint")
    }

    /// Byte-level BPE whose Qwen3.8 special tokens (the chat markers, the
    /// think tags, the media pads) encode atomically, as in the published
    /// tokenizer.
    fn byte_level_tokenizer() -> Qwen38Tokenizer {
        let mut tokens: Vec<String> = (0..=u8::MAX)
            .map(|byte| byte_level_char(byte).to_string())
            .collect();
        let mut specials = Vec::new();
        for literal in [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            IMAGE_PAD_PLACEHOLDER,
            VIDEO_PAD_PLACEHOLDER,
            "<think>",
            "</think>",
        ] {
            specials.push((literal.to_string(), tokens.len() as u32));
            tokens.push(literal.to_string());
        }
        let mut token_types = vec![1; 256];
        token_types.extend([3; 7]);
        let mut inner =
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).expect("tokenizer");
        inner
            .register_gguf_token_types(&tokens, &token_types)
            .expect("atomic special tokens");
        let id = |literal: &str| {
            specials
                .iter()
                .find(|(token, _)| token == literal)
                .map(|(_, id)| *id)
                .unwrap()
        };
        let special_ids = SpecialTokenIds {
            im_start: id("<|im_start|>"),
            im_end: id("<|im_end|>"),
            eos: id("<|im_end|>"),
            eos_alt: Some(id("<|endoftext|>")),
        };
        specials.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });
        Qwen38Tokenizer {
            vocab_size: inner.vocab_size(),
            specials: special_ids,
            inner,
            literal_special_tokens: specials,
            chat_template: String::new(),
            default_enable_thinking: true,
        }
    }

    /// The declared history boundary is exactly what the next turn shares:
    /// across three turns, with thinking on and off, every reasoning effort,
    /// preserved and dropped reasoning, tools, Unicode and reasoning inside
    /// the answers, each render reproduces the previous render's tokens up to
    /// its boundary, and the boundary opens the assistant generation header.
    #[test]
    fn next_turn_shares_the_declared_history_boundary() {
        let tokenizer = byte_level_tokenizer();
        let im_start = tokenizer.specials.im_start;
        let tool = serde_json::json!({"type": "function", "function": {"name": "lookup"}});
        for (thinking, effort, preserve, tools) in [
            (None, None, None, Vec::new()),
            (Some(false), None, None, Vec::new()),
            (
                Some(true),
                Some(ChatReasoningEffort::Low),
                Some(false),
                Vec::new(),
            ),
            (
                Some(true),
                Some(ChatReasoningEffort::Medium),
                None,
                vec![tool.clone()],
            ),
            (
                Some(false),
                Some(ChatReasoningEffort::Xhigh),
                Some(false),
                vec![tool],
            ),
        ] {
            let config = ChatGenerationConfig {
                request: ChatRequestConfig {
                    enable_thinking: thinking,
                    reasoning_effort: effort,
                    preserve_thinking: preserve,
                    tools,
                    ..Default::default()
                },
                ..Default::default()
            };
            let message = |role, content: &str| ChatMessage {
                role,
                content: content.to_string(),
            };
            let mut conversation = vec![
                message(ChatRole::System, "Be precise."),
                message(ChatRole::User, "First question — café 中文 🌍"),
            ];
            let answers = [
                "<think>\nreasoning one\n</think>\n\nFirst answer.",
                "reasoning two</think>\nSecond answer ✓",
                "Third answer, no reasoning",
            ];
            let mut previous: Option<(Vec<u32>, usize)> = None;
            for (turn, answer) in answers.iter().enumerate() {
                let prompt = render_prompt(&conversation, &config, true).unwrap();
                let ids = tokenizer.encode_text(&prompt).unwrap();
                let boundary = history_boundary(&ids, im_start).expect("declared boundary");
                assert_eq!(ids[boundary], im_start);
                assert!(tokenizer
                    .inner
                    .decode_with_special_tokens(&ids[boundary..])
                    .unwrap()
                    .starts_with("<|im_start|>assistant\n"));
                if let Some((previous_ids, previous_boundary)) = &previous {
                    assert!(boundary > *previous_boundary);
                    assert_eq!(
                        &ids[..*previous_boundary],
                        &previous_ids[..*previous_boundary],
                        "turn {turn} ({config:?}) diverges inside the previous history"
                    );
                }
                previous = Some((ids, boundary));
                conversation.push(message(ChatRole::Assistant, answer));
                conversation.push(message(ChatRole::User, "Follow-up ✓"));
            }
        }
        assert_eq!(history_boundary(&[im_start, 7, 8], im_start), None);
        assert_eq!(history_boundary(&[1, 2, 3], im_start), None);
        assert_eq!(history_boundary(&[1, im_start, 3], im_start), Some(1));
    }

    /// The boundary reaches the engine request through the registry's
    /// prepared prompt, which is how the scheduler and the managed cache see it.
    #[test]
    fn the_registry_prepared_prompt_carries_the_boundary() {
        let prepared = Qwen38PreparedPrompt {
            prompt_ids: vec![1, 2, 3],
            prompt_positions: build_text_positions(3),
            next_text_position: 3,
            history_boundary: Some(2),
        };
        let native = crate::models::registry::NativeChatPreparedPrompt::Qwen38(prepared);
        assert_eq!(native.managed_snapshot_boundary(), Some(2));
    }
}
