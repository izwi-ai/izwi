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
use super::prompt::{
    forced_call, json_reply, render_prompt, CONTENT_TEXT, IMAGE_SLOT, JSON_REPLY_GRAMMAR,
};
use super::reply::{Inkling2Reply, Inkling2ReplyTokens};
use super::state::{
    kv_dtype, managed_contract, paged_geometry, Inkling2ConvRuntimeState, Inkling2PagedContext,
};
use super::trunk::Inkling2Trunk;
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
    /// Features of the prompt's image tokens, on the host.
    image_features: Option<Tensor>,
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
    /// The tokenizer as the sampler's grammar takes it, shared by requests.
    grammar_tokenizer: Arc<Tokenizer>,
    reply_tokens: Inkling2ReplyTokens,
    trunk: Inkling2Trunk,
    max_context_tokens: usize,
    experts: ExpertPlacement,
    /// CUDA graphs of a decode step's pure stages, or why there are none.
    graphs: std::result::Result<StretchGraphs, String>,
    /// The vision tower, when the checkpoint has one.
    vision: Option<Inkling2Vision>,
    /// Whether prompts share committed prefixes (set at load from the
    /// catalog's prefix-reuse cell and the operator's setting).
    prefix_caching: bool,
}

impl InferenceStateContractProvider for Inkling2ChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        Ok(InferenceStateCapability::Managed(managed_contract(
            &self.config,
            kv_dtype(self.device_kind()),
            self.prefix_caching,
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
                    _ => "the device kernels are CUDA only".to_string(),
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
            vision,
            prefix_caching: false,
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
    fn forward_span(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut PhysicalPagedKvCache,
        conv: &mut Inkling2ConvRuntimeState,
        want_logits: bool,
        images: Option<SpanImages<'_>>,
    ) -> Result<Option<Tensor>> {
        if tokens.is_empty() || cache.context_len() != start || conv.cursor() != start as u64 {
            return Err(chat_error(format!(
                "a span at {start} does not continue the state (cache {}, convolution {})",
                cache.context_len(),
                conv.cursor()
            )));
        }
        cache.validate_sparse_model_layers(&paged_geometry(&self.config))?;
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
        let hidden = {
            let mut context = Inkling2PagedContext {
                cache: &*cache,
                prepared: &mut prepared,
            };
            self.trunk.forward_span(
                &embeddings,
                start,
                &mut working,
                &mut context,
                self.graphs.as_ref().ok(),
                None,
            )
        };
        let hidden = match hidden {
            Ok(hidden) => hidden,
            Err(error) => {
                // The trunk's error is the one to report; an abort that
                // also fails must not replace it.
                if let Err(abort) = cache.abort_prepared(prepared) {
                    tracing::warn!("Inkling 2: aborting a failed span also failed: {abort}");
                }
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
        })
    }

    /// [`Self::begin_resumable_prefill_managed`] for a prompt with images.
    pub(crate) fn begin_prepared_prefill_managed(
        &self,
        prepared: &Inkling2PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
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
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
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

    /// [`Self::start_decode_managed`] for a prompt with images.
    pub fn start_prepared_decode_managed(
        &self,
        prepared: &Inkling2PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
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
            image_features: saved.image_features.clone(),
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
                true,
                None,
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
