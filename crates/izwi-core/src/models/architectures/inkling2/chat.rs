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
use super::drafter::Inkling2Drafter;
use super::prompt::{
    forced_call, json_reply, render_prompt, CONTENT_TEXT, IMAGE_SLOT, JSON_REPLY_GRAMMAR,
};
use super::reply::{Inkling2Reply, Inkling2ReplyTokens};
use super::state::{
    kv_dtype, managed_contract, DrafterLayers, Inkling2Caches, Inkling2ConvRuntimeState,
    Inkling2DrafterCacheContext, Inkling2Prepared, Inkling2RowsPagedContext,
};
use super::trunk::{Inkling2ConvState, Inkling2PassRecord, Inkling2Trunk};
use super::vision::{image_tokens, preprocess_image, Inkling2Vision};
use crate::backends::kv::KvWriteBatchCompletion;
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::kernels::cuda::stretch_graphs::StretchGraphs;
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{
    replay_journal_span, ChatGenerationConfig, ChatMediaInput, ChatMessage,
};
use crate::models::shared::media::{
    decode_image, fetch_media_bytes, image_media_host_bytes, MediaResourceEstimate,
};
use crate::models::shared::sampling::{resolve_token_logprob, ChatSampler};
use crate::tokenizer::Tokenizer;

/// Context a sequence may reach when the checkpoint allows more. Every held
/// token costs keys and values in all 48 layer applications, so the
/// published 1M-token limit is not served.
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 32_768;

/// Selects where the experts run: `auto` (on the device on CUDA, on the host
/// elsewhere), `host`, or `device`.
pub const EXPERT_BACKEND_ENV: &str = "IZWI_INKLING2_MOE_BACKEND";
/// Selects the attention core of decode steps and short spans
/// (`state::FUSED_ATTENTION_MAX_SPAN`): `auto` (the fused paged kernel on
/// Metal and CUDA once it passes its load-time check, the composed form on
/// the CPU), `on` (the fused kernel wherever it passes) or `off` (always
/// composed).
pub const FUSED_ATTENTION_ENV: &str = "IZWI_INKLING2_FUSED_ATTENTION";
/// Serves the block drafter (`mtp.safetensors`) for speculative decoding:
/// `on` loads it when the file is present; `off` (the default) does not.
pub const DRAFTER_ENV: &str = "IZWI_INKLING2_DRAFTER";
/// Whether [`DRAFTER_ENV`] asks for the drafter.
pub fn drafter_asked() -> bool {
    crate::env::value(DRAFTER_ENV)
        .map(|value| value.trim().to_ascii_lowercase())
        .is_some_and(|value| matches!(value.as_str(), "on" | "1" | "true"))
}

/// Kill switch for the decode step's CUDA graphs.
pub const DECODE_GRAPHS_ENV: &str = "IZWI_INKLING2_DECODE_GRAPHS";
/// Device memory the decode graphs hold once captured: stable copies of one
/// token's stage inputs and outputs for every layer application, and the
/// expert kernels' scratch.
const DECODE_GRAPH_DEVICE_BYTES: u64 = 64 << 20;

/// Bytes a load on `backend` should set aside for the decode graphs.
pub fn decode_graph_device_bytes(backend: BackendKind) -> u64 {
    let on = backend == BackendKind::Cuda
        && crate::env::flag_or(DECODE_GRAPHS_ENV, true)
        && !crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off()
        && ExpertBackendRequest::from_env().wants_device(backend);
    if on {
        DECODE_GRAPH_DEVICE_BYTES
    } else {
        0
    }
}

/// The decode step runs one row at a time.
fn one_row(rows: usize) -> bool {
    rows == 1
}

/// Largest relative error the device experts may show against the host path
/// at load before the model falls back to the host.
const DEVICE_EXPERT_TOLERANCE: f32 = 1e-3;

/// Where the experts were asked to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertBackendRequest {
    Auto,
    Host,
    Device,
}

impl ExpertBackendRequest {
    pub fn from_env() -> Self {
        Self::parse(crate::env::value(EXPERT_BACKEND_ENV).as_deref())
    }

    pub(crate) fn parse(value: Option<&str>) -> Self {
        match value
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("host") => Self::Host,
            Some("device") => Self::Device,
            _ => Self::Auto,
        }
    }

    /// Whether this request puts the experts on a device of `backend`.
    pub fn wants_device(self, backend: BackendKind) -> bool {
        match self {
            Self::Host => false,
            Self::Device => true,
            Self::Auto => backend == BackendKind::Cuda,
        }
    }
}

/// Where the experts run and why.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertPlacement {
    pub on_device: bool,
    pub reason: String,
    /// Worst relative error of the load-time check, when it ran.
    pub check_error: Option<f32>,
}

fn chat_error(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("Inkling 2 chat: {message}"))
}

/// Images a request may carry.
pub const MAX_IMAGES: usize = 4;
/// Image tokens one image may take. The reference never resizes an image;
/// the vision tower's attention is quadratic in four times this, so a larger
/// image is scaled down to fit (1,600 tokens is 1,600 x 1,600 pixels).
pub const MAX_IMAGE_TOKENS: usize = 1_600;

/// Device bytes the vision tower works in for one image: a chunk of
/// attention scores, its softmax and product, and the widest activations.
const VISION_WORKSPACE_BYTES: u64 = 1 << 30;

/// Host and device bytes a request's images need while its prompt is
/// prepared. Rejects more than [`MAX_IMAGES`], anything that is not an
/// image, and an empty source.
pub fn media_resource_estimate(media_inputs: &[ChatMediaInput]) -> Result<MediaResourceEstimate> {
    let host_bytes = image_media_host_bytes(media_inputs, "Inkling 2")?;
    Ok(MediaResourceEstimate {
        host_bytes,
        backend_tensor_bytes: if media_inputs.is_empty() {
            0
        } else {
            VISION_WORKSPACE_BYTES
        },
    })
}

/// A decoded image: `[height, width, channels]` U8 with 3 (RGB) or 4 (RGBA)
/// channels.
#[derive(Debug, Clone)]
pub struct Inkling2ImageInput {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
}

/// The size an image is used at: its own unless it would take more than
/// `max_tokens` image tokens, then the largest size of the same shape that
/// fits.
pub fn fitted_image_size(
    height: usize,
    width: usize,
    cell: usize,
    max_tokens: usize,
) -> (usize, usize) {
    if image_tokens(height, width, cell) <= max_tokens {
        return (height, width);
    }
    let scale = ((max_tokens * cell * cell) as f64 / (height * width) as f64).sqrt();
    let (mut h, mut w) = (
        ((height as f64 * scale).floor() as usize).max(1),
        ((width as f64 * scale).floor() as usize).max(1),
    );
    // Rounding each side up to whole cells can still overshoot.
    while image_tokens(h, w, cell) > max_tokens {
        if h >= w && h > 1 {
            h -= 1;
        } else if w > 1 {
            w -= 1;
        } else {
            break;
        }
    }
    (h, w)
}

/// A prompt that carries images: its tokens, with one image slot per image
/// token, and the features that take the slots' place. The features are on
/// the host (`[slots, hidden]`, F32) so a suspended session holds no device
/// memory for them.
#[derive(Debug, Clone)]
pub struct Inkling2PreparedPrompt {
    prompt_ids: Vec<u32>,
    image_features: Tensor,
}

impl Inkling2PreparedPrompt {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    /// Image tokens in the prompt.
    pub fn image_tokens(&self) -> usize {
        self.image_features.dim(0).unwrap_or(0)
    }
}

/// What sampling one token did to the reply.
enum Taken {
    /// A configured stop: not emitted.
    Stop,
    /// The reply ended on it; the text it completed.
    Ended(String),
    /// A token kept, now pending; the text it added.
    Next(u32, String),
}

/// A span run through the trunk, its writes prepared and not committed.
struct SpanRun {
    hidden: Tensor,
    prepared: Inkling2Prepared,
    /// The convolution state after the whole span.
    conv: Inkling2ConvState,
    record: Inkling2PassRecord,
    /// The drafter head's fused state of each position, when served.
    drafter_heads: Option<Tensor>,
}

/// The image features a span needs: `used` of them belong to tokens before
/// the span, and only the span's first `prompt` tokens are prompt tokens (a
/// generated token is never an image slot, whatever its id).
struct SpanImages<'a> {
    features: &'a Tensor,
    used: usize,
    prompt: usize,
}

pub struct ChatDecodeState {
    conv: Inkling2ConvRuntimeState,
    physical_kv: Inkling2Caches,
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
    /// Features of the prompt's image tokens, on the host.
    image_features: Option<Tensor>,
    /// The drafter head's fused state of the last committed position, when
    /// the drafter is served.
    drafter_head: Option<Tensor>,
}

pub(crate) struct ChatDecodeCheckpoint {
    drafter_head: Option<Tensor>,
    conv: Inkling2ConvRuntimeState,
    physical_kv: Inkling2Caches,
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
/// device tensors (a prompt's image features are host memory); the released
/// state is rebuilt by replaying the prompt and every generated token but the
/// pending one.
#[derive(Clone)]
pub(crate) struct ChatReplayCheckpoint {
    replay_tokens: usize,
    pending_token: u32,
    prefill_progress: usize,
    generated_ids: Vec<u32>,
    sampler: ChatSampler,
    reply: Inkling2Reply,
    max_new_tokens: usize,
    image_features: Option<Tensor>,
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
            image_features: self.image_features.clone(),
        })
    }

    fn continues(&self, cache: &Inkling2Caches) -> Result<()> {
        if !self.physical_kv.continued_by(cache) {
            return Err(chat_error(
                "a managed reservation does not continue the session",
            ));
        }
        Ok(())
    }

    /// Takes the next quantum's reservation: the main cache and, when the
    /// sliding-window applications have their own domain, theirs.
    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
        windowed: Option<PhysicalPagedKvCache>,
    ) -> Result<ChatDecodeCheckpoint> {
        let cache = Inkling2Caches::new(cache, windowed)?;
        self.continues(&cache)?;
        Ok(ChatDecodeCheckpoint {
            drafter_head: self.drafter_head.clone(),
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
        windowed: Option<PhysicalPagedKvCache>,
    ) -> Result<()> {
        let cache = Inkling2Caches::new(cache, windowed)?;
        self.continues(&cache)?;
        self.physical_kv = cache;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(&mut self, checkpoint: ChatDecodeCheckpoint) {
        self.drafter_head = checkpoint.drafter_head;
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

    /// The sliding-window domain's cache, when the state has one.
    #[cfg(test)]
    pub(crate) fn windowed_cache_for_test(&self) -> Option<&PhysicalPagedKvCache> {
        self.physical_kv.windowed()
    }

    /// The convolution ring as the engine would publish it at a snapshot.
    #[cfg(test)]
    pub(crate) fn conv_for_test(&self) -> Inkling2ConvRuntimeState {
        self.conv.clone()
    }

    /// What the engine does before an attached prefix's first span: puts the
    /// ring's snapshot at that cursor in place.
    #[cfg(test)]
    pub(crate) fn restore_conv_for_test(&mut self, conv: Inkling2ConvRuntimeState) {
        self.conv = conv;
    }

    /// The reservation the engine would hand back for the next quantum.
    #[cfg(test)]
    pub(crate) fn cache_view_for_test(&self) -> PhysicalPagedKvCache {
        let main = self.physical_kv.main();
        PhysicalPagedKvCache::new(
            main.arena().clone(),
            main.arena()
                .config()
                .layers
                .iter()
                .map(|layer| layer.binding)
                .collect(),
            main.blocks.clone(),
            main.context_len(),
        )
        .unwrap()
    }
}

pub struct Inkling2ChatModel {
    variant: ModelVariant,
    device: DeviceProfile,
    config: Inkling2Config,
    tokenizer: Tokenizer,
    /// The tokenizer as the sampler's grammar takes it, shared by requests.
    grammar_tokenizer: Arc<Tokenizer>,
    reply_tokens: Inkling2ReplyTokens,
    trunk: Inkling2Trunk,
    max_context_tokens: usize,
    experts: ExpertPlacement,
    /// CUDA graphs of a decode step's pure stages, or why there are none.
    graphs: std::result::Result<StretchGraphs, String>,
    /// The fused attention kernel's load-time check (its worst error
    /// relative to the host reference), or why attention is composed.
    fused_attention: std::result::Result<f64, String>,
    /// The vision tower, when the checkpoint has one.
    vision: Option<Inkling2Vision>,
    /// Whether prompts share committed prefixes (set at load from the
    /// catalog's prefix-reuse cell and the operator's setting).
    prefix_caching: bool,
    /// The block drafter, when served ([`DRAFTER_ENV`]).
    drafter: Option<Inkling2Drafter>,
    /// Speculative rounds run and their proposals kept.
    speculation: SpeculationCounters,
}

/// Counts of speculative rounds, for the diagnostics.
#[derive(Debug, Default)]
struct SpeculationCounters {
    rounds: std::sync::atomic::AtomicU64,
    proposed: std::sync::atomic::AtomicU64,
    kept: std::sync::atomic::AtomicU64,
}

impl SpeculationCounters {
    fn record(&self, proposed: usize, kept: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.rounds.fetch_add(1, Relaxed);
        self.proposed.fetch_add(proposed as u64, Relaxed);
        self.kept.fetch_add(kept as u64, Relaxed);
    }

    fn diagnostics(&self) -> serde_json::Value {
        use std::sync::atomic::Ordering::Relaxed;
        let (rounds, proposed, kept) = (
            self.rounds.load(Relaxed),
            self.proposed.load(Relaxed),
            self.kept.load(Relaxed),
        );
        serde_json::json!({
            "rounds": rounds,
            "proposed": proposed,
            "kept": kept,
            "kept_per_round": if rounds == 0 { 0.0 } else { kept as f64 / rounds as f64 },
        })
    }
}

impl InferenceStateContractProvider for Inkling2ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        Ok(InferenceStateCapability::Managed(managed_contract(
            &self.config,
            kv_dtype(self.device_kind()),
            self.prefix_caching,
            self.drafter_layers(),
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
        Self::from_parts(
            variant,
            device,
            config,
            tokenizer,
            &checkpoint,
            ExpertBackendRequest::from_env(),
        )
    }

    /// The drafter, when [`DRAFTER_ENV`] asks for it and the model directory
    /// has its weights. A drafter that does not fit the model fails the load
    /// rather than serving wrong drafts.
    fn load_drafter(
        dir: &Path,
        config: &Inkling2Config,
        device: &candle_core::Device,
    ) -> Result<Option<Inkling2Drafter>> {
        if !drafter_asked() {
            return Ok(None);
        }
        if !dir.join(super::checkpoint::MTP_FILE).is_file() {
            tracing::warn!(
                "Inkling 2: {DRAFTER_ENV}=on, but {} has no {}; decoding without drafts",
                dir.display(),
                super::checkpoint::MTP_FILE
            );
            return Ok(None);
        }
        Self::drafter_from(Inkling2Drafter::load(dir, device)?, config).map(Some)
    }

    /// Checks that `drafter` drafts for this model.
    fn drafter_from(drafter: Inkling2Drafter, config: &Inkling2Config) -> Result<Inkling2Drafter> {
        let c = drafter.config();
        let applications = config.application_count();
        if c.hidden != config.hidden
            || c.vocab != config.unpadded_vocab
            || c.taps.last() != Some(&applications)
            || c.taps.iter().any(|tap| *tap > applications)
        {
            return Err(Error::ModelLoadError(
                "Inkling 2: the drafter does not fit the model's width, vocabulary or layers"
                    .into(),
            ));
        }
        Ok(drafter)
    }

    /// Serves `drafter` with this model, for a test.
    #[cfg(test)]
    pub(crate) fn with_drafter_for_test(mut self, drafter: Inkling2Drafter) -> Self {
        self.drafter = Some(Self::drafter_from(drafter, &self.config).unwrap());
        self
    }

    /// The drafter's layers as the caches hold them.
    pub(crate) fn drafter_layers(&self) -> Option<DrafterLayers> {
        self.drafter.as_ref().map(|drafter| {
            let c = drafter.config();
            DrafterLayers {
                layers: c.layers,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim,
            }
        })
    }

    /// Tokens a decode quantum may take: the anchor and the drafted tokens
    /// when the drafter is served, else one.
    pub fn preferred_decode_tokens(&self) -> usize {
        self.drafter
            .as_ref()
            .map_or(1, |drafter| drafter.config().drafted() + 1)
    }

    /// Whether attention runs in the arena's fused kernel
    /// (`kernels::paged_bias_sink`): where it is asked for, the head width
    /// fits, and the kernel agrees with its host reference on this device.
    fn resolve_fused_attention(
        config: &Inkling2Config,
        device: &candle_core::Device,
        backend: BackendKind,
    ) -> std::result::Result<f64, String> {
        use crate::kernels::paged_bias_sink;
        let request =
            crate::env::value(FUSED_ATTENTION_ENV).map(|value| value.trim().to_ascii_lowercase());
        match request.as_deref() {
            Some("off" | "0" | "false") => {
                return Err(format!("{FUSED_ATTENTION_ENV}=off"));
            }
            Some("on" | "1" | "true") => {}
            _ if backend == BackendKind::Cpu => {
                return Err(format!(
                    "the composed form is the default on the CPU; {FUSED_ATTENTION_ENV}=on selects the fused kernel"
                ));
            }
            _ => {}
        }
        for heads in [config.full, config.sliding] {
            if paged_bias_sink::lanes(heads.head_dim).is_none() {
                return Err(format!(
                    "the fused kernel does not take a head width of {}",
                    heads.head_dim
                ));
            }
        }
        let storage = match backend {
            BackendKind::Cpu => candle_core::DType::F32,
            BackendKind::Metal => candle_core::DType::F16,
            BackendKind::Cuda => candle_core::DType::BF16,
        };
        match paged_bias_sink::self_check(device, storage) {
            Ok(check) => match check.failure {
                None => Ok(check.worst_ratio),
                Some(failure) => Err(format!("the fused kernel failed its check: {failure}")),
            },
            Err(error) => Err(format!("the fused kernel's check did not run: {error}")),
        }
    }

    /// Whether attention runs in the fused kernel.
    pub fn fused_attention(&self) -> &std::result::Result<f64, String> {
        &self.fused_attention
    }

    /// Runs attention in the fused kernel whatever the load chose.
    #[cfg(test)]
    pub(crate) fn with_fused_attention_for_test(mut self) -> Self {
        self.fused_attention = Ok(0.0);
        self
    }

    /// Composes attention in the trunk whatever the load chose, for a test
    /// that compares the two forms.
    #[cfg(test)]
    pub(crate) fn with_composed_attention_for_test(mut self) -> Self {
        self.fused_attention = Err("composed for a test".to_string());
        self
    }

    /// Puts the experts on the device when asked to, and keeps them there
    /// only if they agree with the host path.
    fn place_experts(
        trunk: &mut Inkling2Trunk,
        request: ExpertBackendRequest,
        backend: BackendKind,
    ) -> ExpertPlacement {
        let host = |reason: String, check_error| ExpertPlacement {
            on_device: false,
            reason,
            check_error,
        };
        if !request.wants_device(backend) {
            return host(
                match request {
                    ExpertBackendRequest::Host => format!("{EXPERT_BACKEND_ENV}=host"),
                    _ => format!(
                        "the host path is the default here; {EXPERT_BACKEND_ENV}=device runs the experts on the device"
                    ),
                },
                None,
            );
        }
        if let Err(error) = trunk.experts_to_device() {
            trunk.experts_to_host();
            return host(format!("upload failed: {error}"), None);
        }
        match trunk.check_device_experts() {
            Ok(error) if error <= DEVICE_EXPERT_TOLERANCE => ExpertPlacement {
                on_device: true,
                reason: "checked against the host path at load".to_string(),
                check_error: Some(error),
            },
            Ok(error) => {
                trunk.experts_to_host();
                host(
                    format!("the device experts differ from the host path by {error:e}"),
                    Some(error),
                )
            }
            Err(error) => {
                trunk.experts_to_host();
                host(format!("the device check failed: {error}"), None)
            }
        }
    }

    pub(crate) fn from_parts(
        variant: ModelVariant,
        device: DeviceProfile,
        config: Inkling2Config,
        tokenizer: Tokenizer,
        checkpoint: &Inkling2Checkpoint,
        expert_backend: ExpertBackendRequest,
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
        let mut trunk = Inkling2Trunk::load(checkpoint, &config, &device.device)?;
        let experts =
            Self::place_experts(&mut trunk, expert_backend, BackendKind::from(device.kind));
        if !experts.on_device && expert_backend.wants_device(BackendKind::from(device.kind)) {
            tracing::warn!("Inkling 2 experts stay on the host: {}", experts.reason);
        }
        // The stages are pure only with the experts on the device: the host
        // expert path reads activations back.
        let graphs = if experts.on_device {
            StretchGraphs::resolve(
                "Inkling 2",
                DECODE_GRAPHS_ENV,
                &device.device,
                trunk.decode_stretches(),
            )
            .map(|graphs| graphs.with_widths(one_row))
        } else {
            Err("the experts run on the host".to_string())
        };
        let drafter = Self::load_drafter(checkpoint.dir(), &config, &device.device)?;
        let fused_attention =
            Self::resolve_fused_attention(&config, &device.device, BackendKind::from(device.kind));
        if let Err(reason) = &fused_attention {
            tracing::debug!("Inkling 2 attention is composed: {reason}");
        }
        let vision = config
            .vision
            .is_some()
            .then(|| Inkling2Vision::load(checkpoint, &config, &device.device))
            .transpose()?;
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
            grammar_tokenizer: Arc::new(tokenizer.clone()),
            tokenizer,
            reply_tokens,
            trunk,
            max_context_tokens,
            experts,
            graphs,
            fused_attention,
            vision,
            prefix_caching: false,
            drafter,
            speculation: SpeculationCounters::default(),
        })
    }

    /// Share prompt prefixes across requests.
    pub(crate) fn set_prefix_caching(&mut self, enabled: bool) {
        self.prefix_caching = enabled;
    }

    pub(crate) fn prefix_caching(&self) -> bool {
        self.prefix_caching
    }

    /// Tokens of a rendered prompt that the conversation's next turn renders
    /// again: everything before its last message. The next turn's prompt
    /// repeats this one and adds the reply and a new message, so keys,
    /// values and the convolution ring committed up to here are what it
    /// attaches. The last message is left out because a forced tool call or
    /// a JSON reply ends the prompt in a message the next turn will not
    /// have.
    pub(crate) fn history_boundary(&self, prompt_ids: &[u32]) -> Option<usize> {
        let tokens = &self.reply_tokens;
        prompt_ids
            .iter()
            .rposition(|id| *id == tokens.message_model || tokens.other_roles.contains(id))
            .filter(|boundary| *boundary > 0)
    }

    /// Where the experts run.
    pub fn expert_placement(&self) -> &ExpertPlacement {
        &self.experts
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
            "experts": {
                "backend": if self.experts.on_device { "device" } else { "host" },
                "reason": self.experts.reason,
                "check_error": self.experts.check_error,
                "device_bytes": self.trunk.expert_device_bytes(),
            },
            "layer_applications": self.config.application_count(),
            "expert_resident_bytes": self.trunk.expert_resident_bytes(),
            "prefix_sharing": self.prefix_caching,
            "drafter": match &self.drafter {
                Some(drafter) => serde_json::json!({
                    "drafted_per_round": drafter.config().drafted(),
                    "speculation": self.speculation.diagnostics(),
                }),
                None => serde_json::json!(null),
            },
            "attention": match &self.fused_attention {
                Ok(error) => serde_json::json!({ "core": "fused", "check_error": error }),
                Err(reason) => serde_json::json!({ "core": "composed", "reason": reason }),
            },
            "dense": {
                "projection_dtype": format!("{:?}", self.trunk.projection_dtype()),
                "widened_projections": self.trunk.widened_projections(),
                "vision_projection_dtype": self
                    .vision
                    .as_ref()
                    .map(|vision| format!("{:?}", vision.projection_dtype())),
            },
            "max_context_tokens": self.max_context_tokens,
            "decode_graphs": match &self.graphs {
                Ok(graphs) => graphs.diagnostics(),
                Err(reason) => serde_json::json!({ "off": reason }),
            },
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    /// A continuous batch's rows run in one pass where the model batches
    /// rows, else one after another inside the step.
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
                "an Inkling 2 prompt with images is prepared with its images".to_string(),
            ));
        }
        render_prompt(messages, &config.request, &[], json_reply(config))?.encode(&self.tokenizer)
    }

    /// Whether this checkpoint takes images.
    pub fn accepts_images(&self) -> bool {
        self.vision.is_some()
    }

    /// Image tokens `image` takes in a prompt, after any scaling down.
    pub fn image_tokens_for(&self, height: usize, width: usize) -> Result<usize> {
        let cell = self.vision()?.config().cell();
        let (height, width) = fitted_image_size(height, width, cell, MAX_IMAGE_TOKENS);
        Ok(image_tokens(height, width, cell))
    }

    fn vision(&self) -> Result<&Inkling2Vision> {
        self.vision.as_ref().ok_or_else(|| {
            Error::InvalidInput("this Inkling 2 checkpoint has no vision tower".to_string())
        })
    }

    /// Renders a conversation whose messages carry `images` in order (one
    /// per image placeholder), encodes the images and seals both. Each image
    /// must already fit [`MAX_IMAGE_TOKENS`] (see [`fitted_image_size`]).
    pub fn prepare_prompt_with_images(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
        images: &[Inkling2ImageInput],
    ) -> Result<Inkling2PreparedPrompt> {
        let vision = self.vision()?;
        if images.is_empty() || images.len() > MAX_IMAGES {
            return Err(Error::InvalidInput(format!(
                "Inkling 2 takes 1 to {MAX_IMAGES} images per request, not {}",
                images.len()
            )));
        }
        let cell = vision.config().cell();
        let mut slots = Vec::with_capacity(images.len());
        let mut features = Vec::with_capacity(images.len());
        for image in images {
            let cells = preprocess_image(
                &image.pixels,
                image.height,
                image.width,
                image.channels,
                cell,
            )?;
            if cells.tokens() > MAX_IMAGE_TOKENS {
                return Err(Error::InvalidInput(format!(
                    "an image of {}x{} pixels takes {} image tokens; the limit is {MAX_IMAGE_TOKENS}",
                    image.width,
                    image.height,
                    cells.tokens()
                )));
            }
            slots.push(cells.tokens());
            features.push(
                vision
                    .encode(&cells)?
                    .to_device(&candle_core::Device::Cpu)?,
            );
        }
        let prompt_ids = render_prompt(messages, &config.request, &slots, json_reply(config))?
            .encode(&self.tokenizer)?;
        let image_features = Tensor::cat(&features, 0)?;
        let slot_id = self.config.image_token_id;
        if super::prompt::special_id(&self.tokenizer, IMAGE_SLOT)? != slot_id {
            return Err(chat_error(
                "the tokenizer's image slot is not the config's image token",
            ));
        }
        let in_prompt = prompt_ids.iter().filter(|id| **id == slot_id).count();
        if in_prompt != image_features.dim(0)? {
            return Err(Error::InvalidInput(format!(
                "the prompt has {in_prompt} image tokens for {} image features; text may not contain the image token",
                image_features.dim(0)?
            )));
        }
        Ok(Inkling2PreparedPrompt {
            prompt_ids,
            image_features,
        })
    }

    /// The prompt of a request with images: fetches and decodes each image,
    /// scales down one that would take more than [`MAX_IMAGE_TOKENS`], and
    /// seals the tokens with the image features.
    pub fn prepare_prompt(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Inkling2PreparedPrompt> {
        let cell = self.vision()?.config().cell();
        media_resource_estimate(&config.request.media_inputs)?;
        let images = config
            .request
            .media_inputs
            .iter()
            .map(|media| {
                let image = decode_image(&fetch_media_bytes(&media.source)?)?;
                let (width, height) = (image.width() as usize, image.height() as usize);
                let (fit_height, fit_width) =
                    fitted_image_size(height, width, cell, MAX_IMAGE_TOKENS);
                let image = if (fit_height, fit_width) == (height, width) {
                    image
                } else {
                    image.resize_exact(
                        fit_width as u32,
                        fit_height as u32,
                        image::imageops::FilterType::CatmullRom,
                    )
                };
                let (channels, pixels) = if image.color().has_alpha() {
                    (4, image.to_rgba8().into_raw())
                } else {
                    (3, image.to_rgb8().into_raw())
                };
                Ok(Inkling2ImageInput {
                    height: fit_height,
                    width: fit_width,
                    channels,
                    pixels,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.prepare_prompt_with_images(messages, config, &images)
    }

    /// What a span of the journal (`prompt_ids`, then generated tokens)
    /// needs of the prompt's image features.
    fn span_images<'a>(
        &self,
        features: Option<&'a Tensor>,
        prompt_ids: &[u32],
        span_start: usize,
        span_end: usize,
    ) -> Option<SpanImages<'a>> {
        let features = features?;
        let slot = self.config.image_token_id;
        let before = span_start.min(prompt_ids.len());
        Some(SpanImages {
            features,
            used: prompt_ids[..before]
                .iter()
                .filter(|id| **id == slot)
                .count(),
            prompt: span_end.min(prompt_ids.len()).saturating_sub(span_start),
        })
    }

    /// Runs `tokens` at positions `start..` and commits their keys, values
    /// and convolution state. Returns the last token's logits (`[vocab]`)
    /// when asked.
    /// Runs `tokens` at positions `start..` through the trunk and writes
    /// their keys and values (and the drafter's) into the prepared append it
    /// returns, uncommitted, with the convolution state after the span. On
    /// an error nothing is left prepared.
    fn run_span(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut Inkling2Caches,
        conv: &Inkling2ConvRuntimeState,
        images: Option<SpanImages<'_>>,
        keep_conv_inputs: bool,
    ) -> Result<SpanRun> {
        if tokens.is_empty() || cache.context_len() != start || conv.cursor() != start as u64 {
            return Err(chat_error(format!(
                "a span at {start} does not continue the state (cache {}, convolution {})",
                cache.context_len(),
                conv.cursor()
            )));
        }
        cache.validate(&self.config, self.drafter_layers())?;
        let embeddings = match images {
            Some(images) if images.prompt > 0 => {
                let prompt = images.prompt.min(tokens.len());
                let head = self.trunk.embed_with_images(
                    &tokens[..prompt],
                    self.config.image_token_id,
                    images.features,
                    images.used,
                )?;
                if prompt < tokens.len() {
                    Tensor::cat(&[&head, &self.trunk.embed(&tokens[prompt..])?], 0)?
                } else {
                    head
                }
            }
            _ => self.trunk.embed(tokens)?,
        };
        let mut working = conv.conv()?.clone();
        let mut prepared = cache.prepare_append(start, tokens.len())?;
        let mut record = Inkling2PassRecord::with_taps(
            self.drafter
                .as_ref()
                .map(|drafter| drafter.config().taps.clone())
                .unwrap_or_default(),
        );
        record.keep_conv_inputs = keep_conv_inputs;
        let run = (|| {
            let hidden = {
                let mut context =
                    cache.context(&self.config, &mut prepared, self.fused_attention.is_ok());
                self.trunk.forward_span_recorded(
                    &embeddings,
                    start,
                    &mut working,
                    &mut context,
                    self.graphs.as_ref().ok(),
                    &mut record,
                )?
            };
            let drafter_heads = self.write_drafter_span(cache, &mut prepared, &record, start)?;
            Ok((hidden, drafter_heads))
        })();
        match run {
            Ok((hidden, drafter_heads)) => Ok(SpanRun {
                hidden,
                prepared,
                conv: working,
                record,
                drafter_heads,
            }),
            Err(error) => {
                // The span's error is the one to report; an abort that
                // also fails must not replace it.
                if let Err(abort) = cache.abort(prepared) {
                    tracing::warn!("Inkling 2: aborting a failed span also failed: {abort}");
                }
                Err(error)
            }
        }
    }

    /// The drafter's keys and values of a span's positions, from the taps
    /// the pass recorded, written into the prepared append; returns the
    /// head's fused state of each position (`[seq, hidden]`).
    fn write_drafter_span(
        &self,
        cache: &Inkling2Caches,
        prepared: &mut Inkling2Prepared,
        record: &Inkling2PassRecord,
        start: usize,
    ) -> Result<Option<Tensor>> {
        let Some(drafter) = &self.drafter else {
            return Ok(None);
        };
        let taps = record
            .taps
            .as_ref()
            .ok_or_else(|| chat_error("the pass recorded no drafter taps"))?;
        let fused = drafter.fuse(taps)?;
        for layer in 0..drafter.config().layers {
            let (keys, values) = drafter.context_kv(layer, &fused.layer(layer)?, start)?;
            cache.write_drafter_span(&self.config, prepared, layer, &keys, &values)?;
        }
        Ok(Some(fused.head()?))
    }

    /// Runs `tokens` at positions `start..` and commits their keys, values
    /// and convolution state. Returns the last token's logits (`[vocab]`)
    /// when asked.
    #[allow(clippy::too_many_arguments)]
    fn forward_span(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut Inkling2Caches,
        conv: &mut Inkling2ConvRuntimeState,
        drafter_head: &mut Option<Tensor>,
        want_logits: bool,
        images: Option<SpanImages<'_>>,
    ) -> Result<Option<Tensor>> {
        let run = self.run_span(tokens, start, cache, conv, images, false)?;
        cache.commit(run.prepared)?;
        conv.advance(run.conv, tokens.len());
        if let Some(heads) = run.drafter_heads {
            *drafter_head = Some(heads.get(tokens.len() - 1)?);
        }
        if !want_logits {
            return Ok(None);
        }
        let last = run.hidden.narrow(0, tokens.len() - 1, 1)?;
        Ok(Some(self.trunk.logits(&last)?.squeeze(0)?))
    }

    fn new_state(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: Inkling2Caches,
    ) -> Result<ChatDecodeState> {
        let attached = cache.context_len();
        let forced = forced_call(&config.request)?;
        let json_text = json_reply(config);
        // The sampler holds tool calls to the request's grammar, and a JSON
        // reply to one object: the same runtime, started inside the text
        // message the prompt opened. The shared JSON machine is not used: it
        // would hold the whole reply, message headers included.
        let mut sampling = config.clone();
        sampling.constrain_json_object = false;
        if json_text {
            sampling.request.tool_constraint = Some(Box::new(crate::chat_tools::ToolConstraint {
                opener: CONTENT_TEXT.to_string(),
                call: JSON_REPLY_GRAMMAR.to_string(),
                forced: Some(JSON_REPLY_GRAMMAR.to_string()),
                max_calls: Some(1),
                strict: true,
            }));
        }
        // The end of the turn first (the grammar's end of sequence), then the
        // end of a message, which closes what a finished constraint leaves
        // open.
        let turn_ends = vec![
            self.reply_tokens.end_sampling,
            self.reply_tokens.end_message,
        ];
        Ok(ChatDecodeState {
            conv: Inkling2ConvRuntimeState::new(self.trunk.new_conv_state()?),
            physical_kv: cache,
            physical_tensor_sequence: None,
            unconsumed_logits: None,
            pending_token: None,
            generated_ids: Vec::new(),
            sampler: ChatSampler::new(sampling, prompt_ids)
                .with_json_object_constraint(self.grammar_tokenizer.clone(), turn_ends),
            reply: Inkling2Reply::new(self.reply_tokens.clone(), forced.as_ref(), json_text),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
            position: attached,
            prefill_progress: attached,
            replay_end: None,
            pending_logprobs: Vec::new(),
            image_features: None,
            drafter_head: None,
        })
    }

    /// [`Self::begin_resumable_prefill_managed`] for a prompt with images.
    pub(crate) fn begin_prepared_prefill_managed(
        &self,
        prepared: &Inkling2PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: impl Into<Inkling2Caches>,
    ) -> Result<ChatDecodeState> {
        let cache = cache.into();
        // The slots of two images are the same tokens: a prompt with images
        // shares no prefix.
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "an Inkling 2 prompt with images does not attach a shared prefix".into(),
            ));
        }
        let mut state = self.begin_resumable_prefill_managed(
            &prepared.prompt_ids,
            max_new_tokens,
            config,
            cache,
        )?;
        state.image_features = Some(prepared.image_features.clone());
        Ok(state)
    }

    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: impl Into<Inkling2Caches>,
    ) -> Result<ChatDecodeState> {
        let cache = cache.into();
        // An attached prefix starts the cache above zero: its pages hold the
        // prompt's first `attached` tokens, and the engine restores the
        // convolution ring's snapshot at that cursor before the first span.
        let attached = cache.context_len();
        if prompt_ids.is_empty() || attached >= prompt_ids.len() {
            return Err(Error::InvalidInput(
                "Inkling 2 managed prefill requires a non-empty prompt beyond any attached prefix"
                    .into(),
            ));
        }
        if attached > 0 && !self.prefix_caching {
            return Err(Error::InvalidInput(
                "Inkling 2 managed prefill received an attached prefix without prefix caching"
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
        let images = self.span_images(
            state.image_features.as_ref(),
            prompt_ids,
            span_start,
            span_end,
        );
        let logits = self.forward_span(
            &prompt_ids[span_start..span_end],
            span_start,
            &mut state.physical_kv,
            &mut state.conv,
            &mut state.drafter_head,
            complete,
            images,
        )?;
        state.prefill_progress = span_end;
        state.position = span_end;
        if complete {
            state.unconsumed_logits = logits;
        }
        Ok(complete)
    }

    /// Starts a session and prefills its whole prompt.
    pub(crate) fn start_decode_managed(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: impl Into<Inkling2Caches>,
    ) -> Result<ChatDecodeState> {
        let mut state =
            self.begin_resumable_prefill_managed(prompt_ids, max_new_tokens, config, cache)?;
        self.continue_resumable_prefill(&mut state, prompt_ids, 0, prompt_ids.len())?;
        Ok(state)
    }

    /// [`Self::start_decode_managed`] for a prompt with images.
    pub(crate) fn start_prepared_decode_managed(
        &self,
        prepared: &Inkling2PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: impl Into<Inkling2Caches>,
    ) -> Result<ChatDecodeState> {
        let mut state =
            self.begin_prepared_prefill_managed(prepared, max_new_tokens, config, cache)?;
        let prompt_ids = &prepared.prompt_ids;
        self.continue_resumable_prefill(&mut state, prompt_ids, 0, prompt_ids.len())?;
        Ok(state)
    }

    /// Starts rebuilding a suspended session in an empty reservation.
    pub(crate) fn begin_replay_managed(
        &self,
        saved: &ChatReplayCheckpoint,
        cache: impl Into<Inkling2Caches>,
    ) -> Result<ChatDecodeState> {
        let cache = cache.into();
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
            image_features: saved.image_features.clone(),
            drafter_head: None,
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
        let images = self.span_images(
            state.image_features.as_ref(),
            prompt_ids,
            span_start,
            span_end,
        );
        self.forward_span(
            &ids,
            span_start,
            &mut state.physical_kv,
            &mut state.conv,
            &mut state.drafter_head,
            false,
            images,
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
                &mut state.drafter_head,
                true,
                None,
            )?;
            state.position += 1;
            committed = 1;
        }
        self.sample_step(state, committed)
    }

    /// Samples the next token from the state's unconsumed logits and
    /// advances the reply; `committed` input tokens were run to get them.
    fn sample_step(&self, state: &mut ChatDecodeState, committed: usize) -> Result<ChatDecodeStep> {
        let logits = state
            .unconsumed_logits
            .take()
            .ok_or_else(|| chat_error("the decode state has no logits to sample"))?;
        state.pending_logprobs.clear();
        match self.take_token(state, &logits)? {
            Taken::Stop => self.finish(state, String::new(), committed),
            Taken::Ended(delta) => self.finish(state, delta, committed),
            Taken::Next(_, delta) if state.generated_ids.len() >= state.max_new_tokens => {
                self.finish(state, delta, committed)
            }
            Taken::Next(_, delta) => Ok(ChatDecodeStep {
                delta,
                text: String::new(),
                tokens_generated: state.generated_ids.len(),
                input_tokens_committed: committed,
                finished: false,
            }),
        }
    }

    /// Samples one token from `logits` with the state's sampler and gives
    /// it to the reply. A token the reply keeps is generated and becomes
    /// the pending token (the next one the model runs); one that stops or
    /// ends the reply is not. The token cap is the caller's.
    fn take_token(&self, state: &mut ChatDecodeState, logits: &Tensor) -> Result<Taken> {
        let vocab = self.config.unpadded_vocab;
        let (next, entry) = if state.sampler.wants_logprobs() {
            let (token, raw) = state.sampler.sample_with_logprobs(logits, vocab)?;
            let entry = raw
                .map(|raw| resolve_token_logprob(&self.tokenizer, &raw))
                .transpose()?;
            (token, entry)
        } else {
            (state.sampler.sample(logits, vocab)?, None)
        };
        if next as usize >= vocab {
            return Err(chat_error(format!(
                "sampled token {next} is outside the vocabulary"
            )));
        }
        if state.sampler.is_configured_stop(next) {
            return Ok(Taken::Stop);
        }
        // Every sampled token advances the reply and, unless it ends the
        // turn, the model's own state on the next step, shown or not.
        let delta = state.reply.push(&self.tokenizer, next)?;
        if state.reply.finished() {
            return Ok(Taken::Ended(delta));
        }
        state.generated_ids.push(next);
        state.pending_token = Some(next);
        if let Some(entry) = entry {
            state.pending_logprobs.push(entry);
        }
        Ok(Taken::Next(next, delta))
    }

    /// Whether a decode quantum of `budget` tokens can run as a speculative
    /// round for `state`.
    fn speculates(&self, state: &ChatDecodeState, budget: usize) -> bool {
        self.drafter.is_some()
            && budget >= 2
            && state.replay_end.is_none()
            && !state.finished
            && state.pending_token.is_some()
            && state.unconsumed_logits.is_none()
            && state.drafter_head.is_some()
            && state
                .max_new_tokens
                .saturating_sub(state.generated_ids.len())
                >= 2
    }

    /// One speculative round: the drafter proposes the tokens after the
    /// pending one (the anchor), the trunk runs the anchor and the proposals
    /// in one span, and the state's own sampler then picks each next token
    /// from those logits in turn. While its pick equals the proposal at that
    /// position the round goes on; the first pick that differs (or the
    /// token after the last proposal) becomes the pending token. Every
    /// emitted token is the sampler's own pick from the model's logits given
    /// the tokens before it, so the reply is what decoding one token at a
    /// time gives, up to the summation order of a wider span; the drafts
    /// only decide how many of those logits one pass yields.
    fn speculative_round(
        &self,
        state: &mut ChatDecodeState,
        budget: usize,
    ) -> Result<ChatDecodeStep> {
        let proposals = self.propose(state, budget)?;
        self.verify_proposals(state, proposals)
    }

    /// The drafter's proposals after the pending token: as many as the
    /// quantum and the token cap leave room for.
    fn propose(&self, state: &ChatDecodeState, budget: usize) -> Result<Vec<u32>> {
        let drafter = self
            .drafter
            .as_ref()
            .ok_or_else(|| chat_error("speculation without a drafter"))?;
        let anchor = state
            .pending_token
            .ok_or_else(|| chat_error("speculation without an anchor"))?;
        let f_rnn = state
            .drafter_head
            .clone()
            .ok_or_else(|| chat_error("speculation before any committed position"))?;
        let position = state.position;
        // Every proposal kept costs a generated token: leave room for the
        // token after the last one.
        let room = state.max_new_tokens - state.generated_ids.len() - 1;
        let drafted = drafter.config().drafted().min(budget - 1).min(room);

        let anchor_embedding = self.trunk.embed(&[anchor])?.squeeze(0)?;
        let block = {
            let context = Inkling2DrafterCacheContext {
                caches: &state.physical_kv,
                config: &self.config,
            };
            drafter.draft_block(&anchor_embedding, position, &context)?
        };
        let rows = drafter.mask_rows(&block)?;
        let project = |hidden: &Tensor| self.trunk.unembed(hidden);
        let mut chain = drafter.markov_chain(&f_rnn, anchor)?;
        let mut proposals = Vec::with_capacity(drafted);
        for row in 0..drafted {
            let logits = chain.step_projected(row, &rows.get(row)?, &project)?;
            let token = logits.argmax(0)?.to_scalar::<u32>()?;
            chain.advance(token)?;
            proposals.push(token);
        }
        Ok(proposals)
    }

    /// Runs the pending token and `proposals` in one span and keeps what
    /// the sampler's own picks agree with (see [`Self::speculative_round`]).
    fn verify_proposals(
        &self,
        state: &mut ChatDecodeState,
        proposals: Vec<u32>,
    ) -> Result<ChatDecodeStep> {
        let anchor = state
            .pending_token
            .ok_or_else(|| chat_error("speculation without an anchor"))?;
        let position = state.position;
        if state.generated_ids.len() + proposals.len() >= state.max_new_tokens {
            return Err(chat_error(
                "more proposals than the token cap leaves room for",
            ));
        }
        let mut tokens = Vec::with_capacity(proposals.len() + 1);
        tokens.push(anchor);
        tokens.extend(&proposals);
        let run = self.run_span(
            &tokens,
            position,
            &mut state.physical_kv,
            &state.conv,
            None,
            true,
        )?;
        let logits = match self.trunk.logits(&run.hidden) {
            Ok(logits) => logits,
            Err(error) => {
                let _ = state.physical_kv.abort(run.prepared);
                return Err(error);
            }
        };

        // Pick from each position's logits in turn.
        state.pending_token = None;
        state.pending_logprobs.clear();
        let mut delta = String::new();
        let mut consumed = 0usize;
        let mut finished = false;
        for (index, proposal) in proposals.iter().map(Some).chain([None]).enumerate() {
            consumed = index + 1;
            match self.take_token(state, &logits.get(index)?)? {
                Taken::Stop => {
                    finished = true;
                    break;
                }
                Taken::Ended(text) => {
                    delta.push_str(&text);
                    finished = true;
                    break;
                }
                Taken::Next(next, text) => {
                    delta.push_str(&text);
                    if state.generated_ids.len() >= state.max_new_tokens {
                        finished = true;
                        break;
                    }
                    if proposal != Some(&next) {
                        break;
                    }
                    // The proposal at the next position is the token just
                    // picked: its keys and values are in the span already.
                    state.pending_token = None;
                }
            }
        }

        // Keep the anchor and the proposals the picks agreed with.
        let conv = if consumed == tokens.len() {
            run.conv
        } else {
            run.record.conv_after(consumed, self.config.conv_kernel)?
        };
        state.physical_kv.commit_prefix(run.prepared, consumed)?;
        state.conv.advance(conv, consumed);
        state.position += consumed;
        if let Some(heads) = run.drafter_heads {
            state.drafter_head = Some(heads.get(consumed - 1)?);
        }
        self.speculation.record(proposals.len(), consumed - 1);
        if finished {
            return self.finish(state, delta, consumed);
        }
        Ok(ChatDecodeStep {
            delta,
            text: String::new(),
            tokens_generated: state.generated_ids.len(),
            input_tokens_committed: consumed,
            finished: false,
        })
    }

    /// [`Self::verify_proposals`] with proposals a test chooses.
    #[cfg(test)]
    pub(crate) fn verify_proposals_for_test(
        &self,
        state: &mut ChatDecodeState,
        proposals: Vec<u32>,
    ) -> Result<ChatDecodeStep> {
        self.verify_proposals(state, proposals)
    }

    /// A decode quantum of up to `budget` tokens: a speculative round when
    /// the drafter is served, else one decode step.
    pub fn decode_quantum(
        &self,
        state: &mut ChatDecodeState,
        budget: usize,
    ) -> Result<ChatDecodeStep> {
        if self.speculates(state, budget) {
            return self.speculative_round(state, budget);
        }
        self.decode_step(state)
    }

    /// Whether a continuous batch runs its rows in one pass. That needs the
    /// fused attention kernel (the composed form attends one sequence at a
    /// time); where the decode step's CUDA graphs run, rows run one after
    /// another so that each replays them.
    pub(crate) fn batches_rows(&self) -> bool {
        self.fused_attention.is_ok() && self.graphs.is_err()
    }

    /// Runs every row's pending token through the trunk in one pass and
    /// leaves its logits unconsumed. Either every row advances or none does.
    fn forward_rows(&self, states: &mut [&mut ChatDecodeState]) -> Result<()> {
        let mut tokens = Vec::with_capacity(states.len());
        let mut positions = Vec::with_capacity(states.len());
        for state in states.iter() {
            let token = state
                .pending_token
                .ok_or_else(|| chat_error("a batched row has no pending token"))?;
            if state.physical_kv.context_len() != state.position
                || state.conv.cursor() != state.position as u64
            {
                return Err(chat_error(
                    "a batched row's caches do not continue its position",
                ));
            }
            state
                .physical_kv
                .validate(&self.config, self.drafter_layers())?;
            tokens.push(token);
            positions.push(state.position);
        }
        let embeddings = self.trunk.embed(&tokens)?;
        let mut working = states
            .iter()
            .map(|state| Ok(state.conv.conv()?.clone()))
            .collect::<Result<Vec<_>>>()?;
        let mut prepared = states
            .iter()
            .map(|state| state.physical_kv.prepare_append(state.position, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut record = Inkling2PassRecord::with_taps(
            self.drafter
                .as_ref()
                .map(|drafter| drafter.config().taps.clone())
                .unwrap_or_default(),
        );
        let hidden = (|| {
            let hidden = {
                let rows = states
                    .iter()
                    .zip(prepared.iter_mut())
                    .map(|(state, prepared)| (&state.physical_kv, prepared))
                    .collect();
                let mut context = Inkling2RowsPagedContext::new(&self.config, rows)?;
                let mut convs = working.iter_mut().collect::<Vec<_>>();
                self.trunk.forward_rows(
                    &embeddings,
                    &positions,
                    &mut convs,
                    &mut context,
                    Some(&mut record),
                )?
            };
            // Each row's drafter keys and values, at its own position.
            let mut heads = Vec::new();
            if let Some(taps) = &record.taps {
                for (row, (state, prepared)) in states.iter().zip(prepared.iter_mut()).enumerate() {
                    let mut row_record = Inkling2PassRecord::default();
                    row_record.taps = Some(taps.narrow(0, row, 1)?);
                    if let Some(head) = self.write_drafter_span(
                        &state.physical_kv,
                        prepared,
                        &row_record,
                        state.position,
                    )? {
                        heads.push(head.get(0)?);
                    }
                }
            }
            Ok((hidden, heads))
        })();
        let (hidden, heads) = match hidden {
            Ok(done) => done,
            Err(error) => {
                for (state, prepared) in states.iter_mut().zip(prepared) {
                    if let Err(abort) = state.physical_kv.abort(prepared) {
                        tracing::warn!("Inkling 2: aborting a failed batch also failed: {abort}");
                    }
                }
                return Err(error);
            }
        };
        let logits = self.trunk.logits(&hidden)?;
        for (row, ((state, prepared), conv)) in
            states.iter_mut().zip(prepared).zip(working).enumerate()
        {
            state.physical_kv.commit(prepared)?;
            state.conv.advance(conv, 1);
            if let Some(head) = heads.get(row) {
                state.drafter_head = Some(head.clone());
            }
            state.unconsumed_logits = Some(logits.get(row)?);
            state.pending_token = None;
            state.position += 1;
        }
        Ok(())
    }

    /// One decode step of every row: in one pass where the model batches
    /// rows ([`Self::batches_rows`]), else one after another. A row's
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
        if states.len() > 1 && self.batches_rows() {
            self.forward_rows(states)?;
            return states
                .iter_mut()
                .map(|state| self.sample_step(state, 1))
                .collect();
        }
        states
            .iter_mut()
            .map(|state| self.decode_step(state))
            .collect()
    }
}
