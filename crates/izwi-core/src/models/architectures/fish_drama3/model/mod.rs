//! The family's model loop (§3.5, §7.1–7.5): the embedding, trunk, tower,
//! compact head, sampler and codec composed into prefill spans, decode
//! quanta and finished generations, on the CPU and on Metal. A Metal load
//! runs its self-checks (`metal::self_check`) before it serves, and its
//! decode samples on the device with two readbacks a frame
//! (`metal::sampling`).
//!
//! A request ([`FishDrama3RequestState`]) owns its retained state: the
//! trunk's DeltaNet state, its two paged KV views ("trunk" and "tower",
//! §7.2), its sampler, the prompt rows, the trunk's last final-norm hidden
//! `h_t`, the frames generated so far, its budget and its stop flag. The
//! model is shared and stateless between calls, so the engine (Phase 2) can
//! run many requests through one load.
//!
//! **Quanta.** Every step runs inside a quantum ([`FishDrama3Model::begin_quantum`],
//! `commit_quantum`, `abort_quantum`): an aborted quantum restores every
//! piece bit for bit, including the KV lengths, the sampler's noise and RAS
//! window, `h_t` and the frame list, so a failed batch or a cancelled
//! request never leaves a half-written row. Commits check the §7.2
//! invariant between the two KV lengths. Cancellation is checked between
//! quanta, never inside one.
//!
//! **Prefill** ([`FishDrama3Model::prefill_span`]) runs the next cell of a
//! grid fixed per load, rows `[k·S, min((k + 1)·S, P))` with `S` the
//! load's `prefill_span_rows`: the trunk over them, then the tower
//! teacher-forced over the same rows except the last prompt row, whose label
//! is the first sampled frame. A span's bits depend on how the rows split (a
//! one-row span takes another GEMM path, and the DeltaNet chunks each span
//! on its own), as upstream's do, so the family owns the grid: no caller or
//! scheduler can cut a span short, and a request's bits never depend on load
//! or on other requests. Spans are never batched across requests either.
//!
//! **Decode** ([`FishDrama3Model::decode_quantum`]) makes one frame per row
//! for any number of requests: the compact head at `h_t`, the main draw
//! (eos masked on the first frame unless the request turns that off), a
//! stop on eos, else the tower at `t` producing `F_{t+1}`, its embedding
//! and one trunk step at `t + 1`. Every kernel runs row by row, so a row's
//! bits never depend on its batch.
//!
//! The audio side (committed frames through the codec stream) is in
//! `audio.rs`, budgets in `budget.rs`, long text in `session.rs`.

pub mod audio;
pub mod budget;
pub mod session;

#[cfg(test)]
pub(crate) mod tests;

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use serde_json::json;

use super::checkpoint::FishDrama3Checkpoint;
use super::codec::{
    FishDrama3CodecArtifact, FishDrama3DacConfig, FishDrama3DacDecoder, FishDrama3DacStreamState,
};
use super::config::{FishDrama3Config, FishDrama3ConfigPolicy};
use super::embed::{prompt_rows, FishDrama3Embedding};
use super::head::FishDrama3CompactHead;
use super::metal::sampling::{
    count_fallback, device_cap, race as device_race, rank, read_back, DEVICE_RANKS,
};
use super::precision::{activation_dtype, check_activations, moe_backend, FishDrama3MetalChecks};
use super::prompt::{FishDrama3Codes, FishDrama3Prompt, FishDrama3Reference};
use super::reference_cache::{FishDrama3ReferenceAudio, FishDrama3ReferenceCache, MAX_REFERENCES};
use super::residency::FishDrama3Residency;
use super::sampling::{
    race as host_race, FishDrama3DrawPolicy, FishDrama3MainDraw, FishDrama3RowSampler,
    FishDrama3SamplerCheckpoint, FishDrama3SamplingParams,
};
use super::tokenizer::FishDrama3Tokenizer;
use super::tower::{
    check_lengths, FishDrama3Tower, FishDrama3TowerCheckpoint, FishDrama3TowerOptions,
    FishDrama3TowerStepRow,
};
use super::trunk::{
    FishDrama3StepRow, FishDrama3Trunk, FishDrama3TrunkCheckpoint, FishDrama3TrunkOptions,
    FishDrama3TrunkState,
};
use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub use audio::{
    FishDrama3AudioCandidate, FishDrama3AudioStream, FISH_DRAMA3_AUDIO_CHUNK_FRAMES,
    FISH_DRAMA3_FIRST_AUDIO_FRAMES,
};
pub use budget::{
    auto_frame_budget, frame_budget, FishDrama3StopReason, MAX_EXPLICIT_FRAMES,
    MAX_REQUEST_POSITIONS,
};
pub use session::{
    FishDrama3ChunkPlan, FishDrama3ChunkResult, FishDrama3ChunkSession,
    FishDrama3ChunkSessionOptions,
};

/// Prefill rows per span on the CPU and Metal (CUDA takes 1,024 in Phase 3).
pub const DEFAULT_PREFILL_SPAN_ROWS: usize = 256;

/// Overrides the prefill grid's cell: rows per span, fixed per load (the
/// scheduler's grid reads it back from the model).
pub const PREFILL_SPAN_ENV: &str = "IZWI_FISH_DRAMA3_PREFILL_SPAN";

/// The prefill grid's cell from `raw` (the [`PREFILL_SPAN_ENV`] value),
/// else `default`: 1 to 8,192 rows (one cell never exceeds a prompt).
pub fn prefill_span_rows(raw: Option<&str>, default: usize) -> Result<usize> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(default);
    };
    match raw.parse::<usize>() {
        Ok(rows) if (1..=super::prompt::MAX_PROMPT_ROWS).contains(&rows) => Ok(rows),
        _ => Err(Error::ConfigError(format!(
            "{PREFILL_SPAN_ENV}={raw} must be a row count from 1 to {}",
            super::prompt::MAX_PROMPT_ROWS
        ))),
    }
}

/// Polled between quanta: an `Err` (normally [`Error::Cancelled`]) stops a
/// request at its last committed boundary.
pub type FishDrama3CancelCheck<'a> = &'a dyn Fn() -> Result<()>;

/// What the loop needs from a codec: the family's DAC, or in tests a
/// stand-in with the same frame contract (a stub of 2,048 samples a frame,
/// §10.1), so the loop is tested without codec weights.
pub trait FishDrama3FrameCodec: Send + Sync {
    fn sample_rate(&self) -> u32;
    fn samples_per_frame(&self) -> Result<usize>;
    /// Codes per frame.
    fn codebooks(&self) -> usize;
    /// Bytes one stream's retained history may reach (one copy; a staged
    /// quantum holds two).
    fn stream_history_bound_bytes(&self) -> Result<u64>;
    /// Decodes `codebooks` (`[codebooks][frames]`) after `state`'s frames;
    /// an error leaves `state` unchanged.
    fn stream_frames(
        &self,
        state: &mut FishDrama3DacStreamState,
        codebooks: &[Vec<u32>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<f32>>;
    /// [`Self::stream_frames`] for several streams at once; batched output
    /// must equal scalar output bit for bit.
    fn stream_frames_batch(
        &self,
        states: &mut [FishDrama3DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>>;
    /// Codes of mono `samples` at `sample_rate`.
    fn encode_reference(
        &self,
        samples: &[f32],
        sample_rate: u32,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3Codes>;
}

impl FishDrama3FrameCodec for FishDrama3DacDecoder {
    fn sample_rate(&self) -> u32 {
        self.config().sample_rate
    }

    fn samples_per_frame(&self) -> Result<usize> {
        self.config().samples_per_frame()
    }

    fn codebooks(&self) -> usize {
        self.config().num_codebooks()
    }

    fn stream_history_bound_bytes(&self) -> Result<u64> {
        self.config().streaming_history_bound_bytes()
    }

    fn stream_frames(
        &self,
        state: &mut FishDrama3DacStreamState,
        codebooks: &[Vec<u32>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        self.push_frames(state, codebooks, check)
    }

    fn stream_frames_batch(
        &self,
        states: &mut [FishDrama3DacStreamState],
        codebooks: &[Vec<Vec<u32>>],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        self.push_frames_batch(states, codebooks, check)
    }

    fn encode_reference(
        &self,
        samples: &[f32],
        sample_rate: u32,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3Codes> {
        self.encode_reference_audio_with_cancel(samples, sample_rate, check)
    }
}

/// How a model load stores its weights and runs prefill.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3ModelOptions {
    pub trunk: FishDrama3TrunkOptions,
    pub tower: FishDrama3TowerOptions,
    /// The prefill grid's cell ([`DEFAULT_PREFILL_SPAN_ROWS`] unless a
    /// backend prefers another), fixed per load: span bits depend on the split.
    pub prefill_span_rows: usize,
    /// [`FishDrama3ConfigPolicy::Published`] except for test checkpoints.
    pub config_policy: FishDrama3ConfigPolicy,
    /// A Metal load's activation dtype: `None` for the backend's (F16),
    /// `Some(F32)` for F32 throughout (`IZWI_FISH_DRAMA3_DTYPE`).
    pub activations: Option<DType>,
}

impl Default for FishDrama3ModelOptions {
    fn default() -> Self {
        Self {
            trunk: FishDrama3TrunkOptions::default(),
            tower: FishDrama3TowerOptions::default(),
            prefill_span_rows: DEFAULT_PREFILL_SPAN_ROWS,
            config_policy: FishDrama3ConfigPolicy::Published,
            activations: None,
        }
    }
}

impl FishDrama3ModelOptions {
    /// The defaults with the load-time environment over them for a load on
    /// `residency`: `IZWI_FISH_DRAMA3_DTYPE`, `IZWI_FISH_DRAMA3_MOE_BACKEND`
    /// and `IZWI_FISH_DRAMA3_PREFILL_SPAN` (over `prefill_span_rows`, the
    /// backend's default cell), each validated.
    pub fn from_env(residency: FishDrama3Residency, prefill_span_rows: usize) -> Result<Self> {
        let read = |name: &str| std::env::var(name).ok();
        Ok(Self {
            trunk: FishDrama3TrunkOptions {
                moe_backend: moe_backend(read(super::precision::MOE_BACKEND_ENV).as_deref())?,
                ..Default::default()
            },
            prefill_span_rows: self::prefill_span_rows(
                read(PREFILL_SPAN_ENV).as_deref(),
                prefill_span_rows,
            )?,
            activations: Some(activation_dtype(
                residency,
                read(super::precision::DTYPE_ENV).as_deref(),
            )?),
            ..Self::default()
        })
    }
}

/// One paged-attention domain of the state contract (§7.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3PagedDomain {
    /// The model layers it binds, in slot order.
    pub layers: Vec<u32>,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl FishDrama3PagedDomain {
    /// Key and value bytes of one position across every layer.
    pub fn bytes_per_position(&self, dtype: DType) -> usize {
        2 * self.layers.len() * self.kv_heads * self.head_dim * dtype.size_in_bytes()
    }
}

/// What one request's retained state costs, from the config alone, so
/// admission prices it before any weight is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3StateGeometry {
    /// Domain 1: the trunk's full-attention layers.
    pub trunk: FishDrama3PagedDomain,
    /// Domain 2: the tower's 13 layers, numbered after the trunk's.
    pub tower: FishDrama3PagedDomain,
    /// Domain 3: every DeltaNet layer's F32 recurrent state.
    pub recurrent_bytes: usize,
    /// Domain 4: every DeltaNet layer's F32 conv history.
    pub conv_bytes: usize,
}

impl FishDrama3StateGeometry {
    pub fn of(config: &FishDrama3Config) -> Self {
        let t = &config.trunk;
        let w = &config.tower;
        let linear = t.linear_attention_layers().len();
        let first_tower = t.layers.len() as u32;
        let tower_layers = (w.shared_layers + w.residual_layers()) as u32;
        Self {
            trunk: FishDrama3PagedDomain {
                layers: t
                    .full_attention_layers()
                    .into_iter()
                    .map(|layer| layer as u32)
                    .collect(),
                kv_heads: t.attention.kv_heads,
                head_dim: t.attention.head_dim,
            },
            tower: FishDrama3PagedDomain {
                layers: (first_tower..first_tower + tower_layers).collect(),
                kv_heads: w.attention.kv_heads,
                head_dim: w.attention.head_dim,
            },
            recurrent_bytes: linear
                * t.linear.value_heads
                * t.linear.key_head_dim
                * t.linear.value_head_dim
                * 4,
            conv_bytes: linear * (t.linear.conv_kernel - 1) * t.linear.conv_channels() * 4,
        }
    }

    /// The tensor domains' bytes, the same for every request.
    pub fn tensor_bytes(&self) -> usize {
        self.recurrent_bytes + self.conv_bytes
    }

    /// One request holding `positions` trunk rows (prompt and frames), its
    /// KV in `kv`: the tower holds one row fewer.
    pub fn request_bytes(&self, positions: usize, kv: DType) -> usize {
        self.tensor_bytes()
            + positions * self.trunk.bytes_per_position(kv)
            + positions.saturating_sub(1) * self.tower.bytes_per_position(kv)
    }
}

/// A request's generation settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishDrama3GenerationParams {
    pub sampling: FishDrama3SamplingParams,
    /// Frames the request may make ([`frame_budget`]); the draw after the
    /// last must be `<|im_end|>`, else `FrameLimit` (unlike Fish S2, which
    /// fails at its N-th frame without that draw).
    pub max_frames: usize,
    /// Keep eos out of the first frame's draws (§7.5); parity replays of
    /// upstream, which has no such mask, turn it off.
    pub mask_first_eos: bool,
}

impl FishDrama3GenerationParams {
    pub fn new(sampling: FishDrama3SamplingParams, max_frames: usize) -> Self {
        Self {
            sampling,
            max_frames,
            mask_first_eos: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if self.max_frames == 0 || self.max_frames > MAX_EXPLICIT_FRAMES {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 frame budget must be 1 to {MAX_EXPLICIT_FRAMES}; got {}",
                self.max_frames
            )));
        }
        Ok(())
    }
}

/// What a step did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3Step {
    Prefill {
        consumed: usize,
        /// Prompt rows in so far.
        position: usize,
        complete: bool,
    },
    /// A frame was made; `frames` counts them all.
    Frame { frames: usize },
    Finished {
        frames: usize,
        reason: FishDrama3StopReason,
    },
}

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// One request's retained state.
pub struct FishDrama3RequestState {
    id: u64,
    model: u64,
    prompt: Arc<FishDrama3Prompt>,
    params: FishDrama3GenerationParams,
    trunk: FishDrama3TrunkState,
    trunk_cache: PhysicalPagedKvCache,
    tower_cache: PhysicalPagedKvCache,
    sampler: FishDrama3RowSampler,
    /// Prompt rows the trunk has consumed.
    prefilled: usize,
    /// `h_t`: the trunk's final-norm hidden at its last row, `[1, hidden]`.
    last_hidden: Option<Tensor>,
    /// `F_t`: the last row the trunk ran, `[main, c0, …]`, the next tower
    /// step's input frame.
    last_row: Vec<u32>,
    /// The frames made so far, `[codebooks][frames]`.
    codes: Vec<Vec<u32>>,
    stop: Option<FishDrama3StopReason>,
    quantum: Option<u64>,
    next_quantum: u64,
}

impl std::fmt::Debug for FishDrama3RequestState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3RequestState")
            .field("id", &self.id)
            .field("prompt_rows", &self.prompt.rows)
            .field("prefilled", &self.prefilled)
            .field("frames", &self.frames())
            .field("stop", &self.stop)
            .field("quantum", &self.quantum)
            .finish_non_exhaustive()
    }
}

impl FishDrama3RequestState {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn prompt(&self) -> &FishDrama3Prompt {
        &self.prompt
    }

    pub fn params(&self) -> &FishDrama3GenerationParams {
        &self.params
    }

    pub fn prompt_rows(&self) -> usize {
        self.prompt.rows
    }

    pub fn prefilled(&self) -> usize {
        self.prefilled
    }

    pub fn is_prefilled(&self) -> bool {
        self.prefilled == self.prompt.rows
    }

    pub fn frames(&self) -> usize {
        self.codes.first().map_or(0, Vec::len)
    }

    /// The frames made so far, `[codebooks][frames]`.
    pub fn codes(&self) -> &[Vec<u32>] {
        &self.codes
    }

    pub fn generated_codes(&self) -> FishDrama3Codes {
        FishDrama3Codes::new(self.codes.clone())
    }

    pub fn stop_reason(&self) -> Option<FishDrama3StopReason> {
        self.stop
    }

    pub fn is_finished(&self) -> bool {
        self.stop.is_some()
    }

    pub fn in_quantum(&self) -> bool {
        self.quantum.is_some()
    }

    pub fn trunk_kv_len(&self) -> usize {
        self.trunk_cache.context_len()
    }

    pub fn tower_kv_len(&self) -> usize {
        self.tower_cache.context_len()
    }

    /// The trunk's DeltaNet state (the engine stages it into its tensor
    /// arena between quanta).
    pub fn trunk_state(&self) -> &FishDrama3TrunkState {
        &self.trunk
    }

    /// The trunk's DeltaNet state, to bind, stage and restore against the
    /// engine's tensor arena.
    pub(crate) fn trunk_state_mut(&mut self) -> &mut FishDrama3TrunkState {
        &mut self.trunk
    }

    /// Both views' sealed write receipts since the last call (the engine
    /// acknowledges them; the private helpers drop them).
    pub(crate) fn take_write_completions(
        &mut self,
    ) -> (
        Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
        Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
    ) {
        (
            self.trunk_cache.take_completed_writes(),
            self.tower_cache.take_completed_writes(),
        )
    }

    /// §7.2: level while the prompt is prefilled, the tower one row behind
    /// once it is in.
    fn check_lengths(&self) -> Result<()> {
        check_lengths(
            self.trunk_cache.context_len(),
            self.tower_cache.context_len(),
            self.prompt.rows,
        )?;
        if self.trunk.position() != self.trunk_cache.context_len() {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 trunk state is at {}; its KV holds {}",
                self.trunk.position(),
                self.trunk_cache.context_len()
            )));
        }
        Ok(())
    }

    fn require_quantum(&self, what: &str) -> Result<()> {
        if self.quantum.is_none() {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 {what} needs an open quantum"
            )));
        }
        Ok(())
    }

    /// `h_t` on the host, for exact comparisons in tests.
    #[cfg(test)]
    pub(crate) fn last_hidden_values(&self) -> Vec<f32> {
        host_values(self.last_hidden().unwrap()).unwrap()
    }

    fn last_hidden(&self) -> Result<&Tensor> {
        self.last_hidden.as_ref().ok_or_else(|| {
            Error::InferenceError("Fish Audio Drama 3 request has no trunk hidden yet".into())
        })
    }
}

/// A rollback point of one request: every piece of its retained state as
/// it was when the quantum began.
pub struct FishDrama3QuantumCheckpoint {
    request: u64,
    quantum: u64,
    trunk: FishDrama3TrunkCheckpoint,
    tower: FishDrama3TowerCheckpoint,
    sampler: FishDrama3SamplerCheckpoint,
    prefilled: usize,
    frames: usize,
    last_hidden: Option<Tensor>,
    last_row: Vec<u32>,
    stop: Option<FishDrama3StopReason>,
}

impl std::fmt::Debug for FishDrama3QuantumCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3QuantumCheckpoint")
            .field("request", &self.request)
            .field("quantum", &self.quantum)
            .field("prefilled", &self.prefilled)
            .field("frames", &self.frames)
            .finish_non_exhaustive()
    }
}

/// A request's preparation (§7.1 stage 0): its prompt, built from its
/// encoded references and its text, sealed to the model load whose codec
/// encoded the references.
#[derive(Debug)]
pub struct FishDrama3PreparedArtifact {
    prompt: Arc<FishDrama3Prompt>,
    model: u64,
    references: usize,
    reference_cache_hits: usize,
    preparation_ms: f64,
}

impl FishDrama3PreparedArtifact {
    pub fn prompt(&self) -> &Arc<FishDrama3Prompt> {
        &self.prompt
    }

    pub fn prompt_rows(&self) -> usize {
        self.prompt.rows
    }

    pub fn references(&self) -> usize {
        self.references
    }

    pub fn reference_cache_hits(&self) -> usize {
        self.reference_cache_hits
    }

    pub fn preparation_ms(&self) -> f64 {
        self.preparation_ms
    }

    /// Host bytes the artifact keeps until the request ends: its `[11][T]`
    /// prompt values.
    pub fn retained_bytes(&self) -> Result<u64> {
        self.prompt
            .values
            .iter()
            .try_fold(0u64, |total, values| {
                total.checked_add(u64::try_from(values.len()).ok()?.checked_mul(4)?)
            })
            .ok_or_else(|| {
                Error::Overloaded("Fish Audio Drama 3 prepared prompt bytes overflow".into())
            })
    }
}

/// A whole streamed generation.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3Synthesis {
    pub codes: FishDrama3Codes,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Frames each codec quantum decoded, in order.
    pub audio_quanta: Vec<usize>,
}

pub struct FishDrama3Model {
    identity: u64,
    config: FishDrama3Config,
    tokenizer: FishDrama3Tokenizer,
    embedding: FishDrama3Embedding,
    trunk: FishDrama3Trunk,
    tower: FishDrama3Tower,
    head: FishDrama3CompactHead,
    codec: Arc<dyn FishDrama3FrameCodec>,
    reference_cache: FishDrama3ReferenceCache,
    device: Device,
    prefill_span_rows: usize,
    /// Both paged domains' dtype: F32 on the CPU, F16 on Metal unless a
    /// load-time check put them in F32.
    kv_dtype: DType,
    /// What a Metal load's checks found (`None` on the CPU).
    metal_checks: Option<FishDrama3MetalChecks>,
    /// The content identity a durable speech checkpoint records
    /// ([`super::artifacts::pinned_fingerprint`]), set by the loader.
    artifact_fingerprint: Option<String>,
}

impl std::fmt::Debug for FishDrama3Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Model")
            .field("identity", &self.identity)
            .field("trunk", &self.trunk)
            .field("tower", &self.tower)
            .field("prefill_span_rows", &self.prefill_span_rows)
            .finish_non_exhaustive()
    }
}

fn private_cache_error(message: impl Into<String>) -> Error {
    Error::InvalidInput(format!("Fish Audio Drama 3 private KV: {}", message.into()))
}

/// Page size of the helpers' private arenas.
const PRIVATE_PAGE_TOKENS: usize = 16;

/// A private, request-sized cache over `domain` for `tokens` positions on
/// `device` in `dtype`, outside any lifecycle-owned arena: the
/// whole-generation helpers use it; the engine passes lease-backed views
/// instead. Distinct `generation`s keep the two domains' arena ids apart,
/// so a checkpoint of one never restores into the other.
pub(crate) fn private_cache(
    domain: &FishDrama3PagedDomain,
    tokens: usize,
    generation: u32,
    device: &Device,
    dtype: DType,
) -> Result<PhysicalPagedKvCache> {
    let u32_of = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| private_cache_error(format!("{what} exceeds u32")))
    };
    let pages = tokens.div_ceil(PRIVATE_PAGE_TOKENS).max(1);
    let backend = crate::backends::backend_kind_for_device(device);
    let id = KvArenaId {
        // Never registered with the engine, like the load-time probes.
        model_instance: crate::engine::ModelInstanceId::new(u64::MAX),
        backend,
        // Metal's location is a registry id the arena does not compare.
        device_ordinal: (backend != BackendKind::Cpu).then_some(0),
        generation,
    };
    let group = KvGroupId::new(0);
    let bindings = domain
        .layers
        .iter()
        .enumerate()
        .map(|(physical, &model_layer)| {
            Ok(KvLayerBinding {
                model_layer,
                physical_layer: u32_of(physical, "layer")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let config = KvArenaConfig {
        id,
        group,
        page_tokens: PRIVATE_PAGE_TOKENS as u32,
        capacity_pages: u32_of(pages, "page count")?,
        growth: None,
        dtype,
        layers: bindings
            .iter()
            .map(|&binding| {
                Ok(KvLayerConfig {
                    binding,
                    num_kv_heads: u32_of(domain.kv_heads, "KV heads")?,
                    key_head_dim: u32_of(domain.head_dim, "head dimension")?,
                    value_head_dim: u32_of(domain.head_dim, "head dimension")?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let arena: Arc<dyn KvArena> = match backend {
        BackendKind::Cpu => Arc::new(CpuKvArena::new(config)?),
        #[cfg(any(feature = "cuda", feature = "metal"))]
        BackendKind::Metal | BackendKind::Cuda => Arc::new(
            crate::backends::kv::CandleAcceleratorKvArena::new_mutation_only(
                config,
                device.clone(),
            )?,
        ),
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        BackendKind::Metal | BackendKind::Cuda => {
            return Err(private_cache_error(
                "an accelerator cache needs the metal or cuda feature",
            ))
        }
    };
    let blocks = (0..pages)
        .map(|index| {
            Ok(CacheBlockRef {
                arena: id,
                group,
                index: u32_of(index, "page index")?,
                slot_generation: 1,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PhysicalPagedKvCache::new(arena, bindings, blocks, 0)
}

fn host_values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?)
}

impl FishDrama3Model {
    /// Loads the model and its DAC from `model_dir` on `device` (the CPU or
    /// Metal; CUDA is refused until its residency lands).
    pub fn load(
        model_dir: &Path,
        device: &Device,
        options: FishDrama3ModelOptions,
    ) -> Result<Self> {
        let artifact = FishDrama3CodecArtifact::load(model_dir)?;
        let weights = artifact.load_weights(device, DType::F32)?;
        let codec =
            FishDrama3DacDecoder::load(FishDrama3DacConfig::current(), weights.var_builder())?;
        Self::load_with_codec(model_dir, device, options, Arc::new(codec))
    }

    /// [`Self::load`] with a codec the caller supplies.
    pub fn load_with_codec(
        model_dir: &Path,
        device: &Device,
        options: FishDrama3ModelOptions,
        codec: Arc<dyn FishDrama3FrameCodec>,
    ) -> Result<Self> {
        if options.prefill_span_rows == 0 {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 prefill spans need at least one row".into(),
            ));
        }
        let config = FishDrama3Config::load_with_policy(model_dir, options.config_policy)?;
        let samples_per_frame = codec.samples_per_frame()?;
        if codec.codebooks() != config.tower.codebooks || samples_per_frame == 0 {
            return Err(Error::ModelLoadError(format!(
                "Fish Audio Drama 3 codec decodes {} codebooks at {samples_per_frame} samples a \
                 frame; the tower makes {} codebooks",
                codec.codebooks(),
                config.tower.codebooks
            )));
        }
        let tokenizer = FishDrama3Tokenizer::load(model_dir, &config)?;
        let checkpoint = FishDrama3Checkpoint::open(model_dir)?;
        checkpoint.validate_census(&config)?;
        let residency = FishDrama3Residency::of(device);
        residency.require_implemented()?;
        let activations = check_activations(residency, options.activations)?;
        let embedding = FishDrama3Embedding::load_on(&checkpoint, &config, device)?;
        let mut trunk = FishDrama3Trunk::load(&checkpoint, &config, device, options.trunk)?;
        let mut tower = FishDrama3Tower::load(&checkpoint, &config, device, options.tower)?;
        let mut head = FishDrama3CompactHead::load(&checkpoint, &config, device)?;
        let (kv_dtype, metal_checks) = if residency == FishDrama3Residency::Metal {
            let (checks, kv) = super::metal::self_check::run(
                &checkpoint,
                &config,
                &embedding,
                &mut trunk,
                &mut tower,
                &mut head,
                activations,
            )?;
            (kv, Some(checks))
        } else {
            (DType::F32, None)
        };
        let geometry = FishDrama3StateGeometry::of(&config);
        if trunk.attention_layers() != geometry.trunk.layers
            || tower.attention_layers() != geometry.tower.layers
        {
            return Err(Error::ModelLoadError(
                "Fish Audio Drama 3 paged domains differ from the state geometry".into(),
            ));
        }
        Ok(Self {
            identity: NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            config,
            tokenizer,
            embedding,
            trunk,
            tower,
            head,
            codec,
            reference_cache: FishDrama3ReferenceCache::default(),
            device: device.clone(),
            prefill_span_rows: options.prefill_span_rows,
            kv_dtype,
            metal_checks,
            artifact_fingerprint: None,
        })
    }

    /// The load with the content identity a durable speech checkpoint
    /// records.
    pub fn with_artifact_fingerprint(mut self, fingerprint: String) -> Self {
        self.artifact_fingerprint = Some(fingerprint);
        self
    }

    pub fn artifact_fingerprint(&self) -> Option<&str> {
        self.artifact_fingerprint.as_deref()
    }

    pub fn config(&self) -> &FishDrama3Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &FishDrama3Tokenizer {
        &self.tokenizer
    }

    pub fn embedding(&self) -> &FishDrama3Embedding {
        &self.embedding
    }

    pub fn trunk(&self) -> &FishDrama3Trunk {
        &self.trunk
    }

    pub fn tower(&self) -> &FishDrama3Tower {
        &self.tower
    }

    pub fn head(&self) -> &FishDrama3CompactHead {
        &self.head
    }

    pub fn codec(&self) -> &dyn FishDrama3FrameCodec {
        self.codec.as_ref()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn prefill_span_rows(&self) -> usize {
        self.prefill_span_rows
    }

    /// Both paged domains' dtype: what the engine's state contract must
    /// store (F16 on Metal unless a load-time check moved it to F32).
    pub fn kv_dtype(&self) -> DType {
        self.kv_dtype
    }

    /// What a Metal load's checks found (`None` on the CPU).
    pub fn metal_checks(&self) -> Option<&FishDrama3MetalChecks> {
        self.metal_checks.as_ref()
    }

    /// Positions one request may use: the trunk's and §7.5's.
    pub fn max_positions(&self) -> usize {
        self.config.trunk.max_positions.min(MAX_REQUEST_POSITIONS)
    }

    pub fn state_geometry(&self) -> FishDrama3StateGeometry {
        FishDrama3StateGeometry::of(&self.config)
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        let mut summary = json!({
            "trunk": self.trunk.diagnostics_summary(),
            "tower": self.tower.diagnostics_summary(),
            "head_bytes": self.head.resident_bytes(),
            "embedding_host_bytes": self.embedding.host_bytes(),
            "sample_rate": self.codec.sample_rate(),
            "prefill_span_rows": self.prefill_span_rows,
            "kv_dtype": format!("{:?}", self.kv_dtype),
        });
        if let Some(checks) = &self.metal_checks {
            summary["embedding_device_bytes"] = json!(self.embedding.device_bytes());
            summary["metal_checks"] = checks.diagnostics();
            summary["resident_sampling"] = json!({
                "frames_total": super::metal::sampling::fish_drama3_resident_frames(),
                "readbacks_total": super::metal::sampling::fish_drama3_resident_readbacks(),
                "host_fallbacks_total":
                    super::metal::sampling::fish_drama3_resident_host_fallbacks(),
            });
        }
        summary
    }

    /// The codes of one reference clip, through the model's cache: mono
    /// mean, 44.1 kHz, clamped, then the codec. Returns whether it hit.
    pub fn encode_reference(
        &self,
        audio: &FishDrama3ReferenceAudio<'_>,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<(FishDrama3Codes, bool)> {
        let rate = self.codec.sample_rate();
        self.reference_cache
            .get_or_encode(audio, rate, check, |samples| {
                self.codec.encode_reference(samples, rate, check)
            })
    }

    /// Every reference's codes, cancellable between references (§7.1
    /// stage 0); at most [`MAX_REFERENCES`].
    pub fn encode_references(
        &self,
        references: &[(&str, FishDrama3ReferenceAudio<'_>)],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<Vec<FishDrama3Reference>> {
        if references.len() > MAX_REFERENCES {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 takes at most {MAX_REFERENCES} references; got {}",
                references.len()
            )));
        }
        references
            .iter()
            .map(|(transcript, audio)| {
                check()?;
                Ok(FishDrama3Reference {
                    transcript: (*transcript).to_string(),
                    codes: self.encode_reference(audio, check)?.0,
                })
            })
            .collect()
    }

    /// A request's preparation (§7.1 stage 0): every reference encoded
    /// through the cache, cancellable between references, then the prompt
    /// built and checked against `max_frames` and the load's positions.
    pub fn prepare_request(
        &self,
        references: &[(&str, FishDrama3ReferenceAudio<'_>)],
        text: &str,
        max_frames: usize,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3PreparedArtifact> {
        let started = std::time::Instant::now();
        if references.len() > MAX_REFERENCES {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 takes at most {MAX_REFERENCES} references; got {}",
                references.len()
            )));
        }
        let mut encoded = Vec::with_capacity(references.len());
        let mut hits = 0;
        for (transcript, audio) in references {
            check()?;
            let (codes, hit) = self.encode_reference(audio, check)?;
            hits += usize::from(hit);
            encoded.push(FishDrama3Reference {
                transcript: (*transcript).to_string(),
                codes,
            });
        }
        check()?;
        let prompt = super::prompt::build_prompt(&self.tokenizer, &encoded, [], text)?;
        budget::check_positions(prompt.rows, max_frames, self.max_positions())?;
        Ok(FishDrama3PreparedArtifact {
            prompt: Arc::new(prompt),
            model: self.identity,
            references: encoded.len(),
            reference_cache_hits: hits,
            preparation_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// A new request over a preparation of this load, with the engine's
    /// empty first-quantum KV views and the `positions` its retained row
    /// holds.
    pub fn new_prepared_request(
        &self,
        artifact: &FishDrama3PreparedArtifact,
        params: FishDrama3GenerationParams,
        trunk_cache: PhysicalPagedKvCache,
        tower_cache: PhysicalPagedKvCache,
        positions: usize,
    ) -> Result<FishDrama3RequestState> {
        if artifact.model != self.identity {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 preparation belongs to another model load".into(),
            ));
        }
        self.new_request_within(
            artifact.prompt.clone(),
            params,
            trunk_cache,
            tower_cache,
            positions,
        )
    }

    /// Installs the engine's views of the two paged domains for the next
    /// quantum: the managed cache hands out a fresh lease-backed view each
    /// quantum, whose length is the committed one. `tower` is `None` when
    /// the quantum appends nothing to the tower (a final one-row prompt
    /// cell), which keeps the current view. Refused inside a quantum, or
    /// for a view of another geometry or length.
    pub fn rebind_caches(
        &self,
        state: &mut FishDrama3RequestState,
        trunk: PhysicalPagedKvCache,
        tower: Option<PhysicalPagedKvCache>,
    ) -> Result<()> {
        self.check_state(state)?;
        if state.quantum.is_some() {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 caches cannot change inside a quantum".into(),
            ));
        }
        self.trunk.validate_cache(&trunk)?;
        if trunk.context_len() != state.trunk_cache.context_len() {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 trunk view holds {} rows; the request holds {}",
                trunk.context_len(),
                state.trunk_cache.context_len()
            )));
        }
        if let Some(tower) = &tower {
            self.tower.validate_cache(tower)?;
            if tower.context_len() != state.tower_cache.context_len() {
                return Err(Error::InvalidInput(format!(
                    "Fish Audio Drama 3 tower view holds {} rows; the request holds {}",
                    tower.context_len(),
                    state.tower_cache.context_len()
                )));
            }
        }
        state.trunk_cache = trunk;
        if let Some(tower) = tower {
            state.tower_cache = tower;
        }
        Ok(())
    }

    fn check_state(&self, state: &FishDrama3RequestState) -> Result<()> {
        if state.model != self.identity {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 request belongs to another model load".into(),
            ));
        }
        Ok(())
    }

    /// A new request over `prompt` with the caller's empty KV views, which
    /// hold every position it may reach.
    pub fn new_request(
        &self,
        prompt: impl Into<Arc<FishDrama3Prompt>>,
        params: FishDrama3GenerationParams,
        trunk_cache: PhysicalPagedKvCache,
        tower_cache: PhysicalPagedKvCache,
    ) -> Result<FishDrama3RequestState> {
        let positions = trunk_cache
            .capacity_tokens()
            .min(tower_cache.capacity_tokens() + 1);
        self.new_request_within(prompt, params, trunk_cache, tower_cache, positions)
    }

    /// [`Self::new_request`] whose views grow quantum by quantum (the
    /// engine's managed ones), so the caller names the positions the
    /// request may reach.
    fn new_request_within(
        &self,
        prompt: impl Into<Arc<FishDrama3Prompt>>,
        params: FishDrama3GenerationParams,
        trunk_cache: PhysicalPagedKvCache,
        tower_cache: PhysicalPagedKvCache,
        positions: usize,
    ) -> Result<FishDrama3RequestState> {
        let prompt = prompt.into();
        params.validate()?;
        let width = self.config.tower.frame_width();
        if prompt.values.len() != width || prompt.values.iter().any(|v| v.len() != prompt.rows) {
            return Err(Error::InvalidInput(format!(
                "a Fish Audio Drama 3 prompt is [{width}][rows] values of {} rows",
                prompt.rows
            )));
        }
        self.trunk.validate_cache(&trunk_cache)?;
        self.tower.validate_cache(&tower_cache)?;
        if trunk_cache.context_len() != 0 || tower_cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 request caches must begin empty".into(),
            ));
        }
        budget::check_positions(
            prompt.rows,
            params.max_frames,
            self.max_positions().min(positions),
        )?;
        let sampler =
            FishDrama3RowSampler::new(params.sampling, self.config.ids, prompt.main_ids())?;
        Ok(FishDrama3RequestState {
            id: NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
            model: self.identity,
            prompt,
            params,
            trunk: self.trunk.new_state()?,
            trunk_cache,
            tower_cache,
            sampler,
            prefilled: 0,
            last_hidden: None,
            last_row: Vec::new(),
            codes: vec![Vec::new(); self.config.tower.codebooks],
            stop: None,
            quantum: None,
            next_quantum: 1,
        })
    }

    /// [`Self::new_request`] over private caches sized for the prompt and
    /// the budget (tests and offline tools).
    pub fn new_private_request(
        &self,
        prompt: impl Into<Arc<FishDrama3Prompt>>,
        params: FishDrama3GenerationParams,
    ) -> Result<FishDrama3RequestState> {
        let prompt = prompt.into();
        params.validate()?;
        budget::check_positions(prompt.rows, params.max_frames, self.max_positions())?;
        let geometry = self.state_geometry();
        let positions = prompt.rows + params.max_frames;
        let trunk = private_cache(&geometry.trunk, positions, 1, &self.device, self.kv_dtype)?;
        let tower = private_cache(&geometry.tower, positions, 2, &self.device, self.kv_dtype)?;
        self.new_request(prompt, params, trunk, tower)
    }

    /// [`Self::new_private_request`] whose sampler draws the recorded
    /// exponentials `noise`, in order (parity replays of an upstream
    /// rollout).
    #[cfg(test)]
    pub(crate) fn new_private_replay(
        &self,
        prompt: impl Into<Arc<FishDrama3Prompt>>,
        params: FishDrama3GenerationParams,
        noise: Vec<f32>,
    ) -> Result<FishDrama3RequestState> {
        let mut state = self.new_private_request(prompt, params)?;
        state.sampler = FishDrama3RowSampler::with_recorded_noise(
            params.sampling,
            self.config.ids,
            state.prompt.main_ids(),
            noise,
        )?;
        Ok(state)
    }

    /// Opens a quantum on `state`: every later step may be undone by
    /// [`Self::abort_quantum`] until [`Self::commit_quantum`].
    pub fn begin_quantum(
        &self,
        state: &mut FishDrama3RequestState,
    ) -> Result<FishDrama3QuantumCheckpoint> {
        self.check_state(state)?;
        if state.quantum.is_some() {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 request already has an open quantum".into(),
            ));
        }
        state.check_lengths()?;
        let trunk = state.trunk.begin_quantum(&state.trunk_cache)?;
        let tower = self.tower.begin_quantum(&state.tower_cache)?;
        let quantum = state.next_quantum;
        state.next_quantum += 1;
        state.quantum = Some(quantum);
        Ok(FishDrama3QuantumCheckpoint {
            request: state.id,
            quantum,
            trunk,
            tower,
            sampler: state.sampler.checkpoint(),
            prefilled: state.prefilled,
            frames: state.frames(),
            last_hidden: state.last_hidden.clone(),
            last_row: state.last_row.clone(),
            stop: state.stop,
        })
    }

    fn check_checkpoint(
        &self,
        state: &FishDrama3RequestState,
        checkpoint: &FishDrama3QuantumCheckpoint,
    ) -> Result<()> {
        self.check_state(state)?;
        if checkpoint.request != state.id || state.quantum != Some(checkpoint.quantum) {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 quantum checkpoint is foreign, stale or out of order".into(),
            ));
        }
        Ok(())
    }

    /// Accepts the open quantum once the KV lengths agree (§7.2). On an
    /// error the quantum stays open for [`Self::abort_quantum`].
    pub fn commit_quantum(
        &self,
        state: &mut FishDrama3RequestState,
        checkpoint: &FishDrama3QuantumCheckpoint,
    ) -> Result<()> {
        self.check_checkpoint(state, checkpoint)?;
        state.check_lengths()?;
        state
            .trunk
            .commit_quantum(&state.trunk_cache, &checkpoint.trunk)?;
        self.tower
            .commit_quantum(&state.tower_cache, &checkpoint.tower)?;
        state.quantum = None;
        Ok(())
    }

    /// Restores `state` to `checkpoint`, bit for bit.
    pub fn abort_quantum(
        &self,
        state: &mut FishDrama3RequestState,
        checkpoint: FishDrama3QuantumCheckpoint,
    ) -> Result<()> {
        self.check_checkpoint(state, &checkpoint)?;
        state
            .trunk
            .abort_quantum(&mut state.trunk_cache, checkpoint.trunk)?;
        self.tower
            .abort_quantum(&mut state.tower_cache, checkpoint.tower)?;
        state.sampler.restore(checkpoint.sampler);
        state.prefilled = checkpoint.prefilled;
        for codebook in &mut state.codes {
            codebook.truncate(checkpoint.frames);
        }
        state.last_hidden = checkpoint.last_hidden;
        state.last_row = checkpoint.last_row;
        state.stop = checkpoint.stop;
        state.quantum = None;
        Ok(())
    }

    /// The prompt rows the next [`Self::prefill_span`] of `state` runs, or
    /// `None` once the prompt is in: the scheduler counts cells, not rows.
    pub fn prefill_cell(&self, state: &FishDrama3RequestState) -> Option<Range<usize>> {
        let (a, rows) = (state.prefilled, state.prompt.rows);
        (a < rows).then(|| {
            let cell = self.prefill_span_rows;
            a..((a / cell + 1) * cell).min(rows)
        })
    }

    /// Prefills the next grid cell of `state` ([`Self::prefill_cell`]) inside
    /// its open quantum: the trunk over rows `[a, b)`, then the tower
    /// teacher-forced over [`FishDrama3Tower::teacher_forced_rows`] of them,
    /// each labelled with the next prompt row. On an error abort the quantum.
    pub fn prefill_span(&self, state: &mut FishDrama3RequestState) -> Result<FishDrama3Step> {
        self.check_state(state)?;
        state.require_quantum("prefill")?;
        let Some(cell) = self.prefill_cell(state) else {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 prompt is already prefilled".into(),
            ));
        };
        if cell.start % self.prefill_span_rows != 0 {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 prefill stands at row {}, off its {}-row grid",
                cell.start, self.prefill_span_rows
            )));
        }
        let rows = state.prompt.rows;
        let (a, b) = (cell.start, cell.end);
        let span = prompt_rows(&state.prompt, a..b)?;
        let embeds = self.embedding.embed_tensor(&span, &self.device)?;
        let hidden =
            self.trunk
                .forward_span(&embeds, a, &mut state.trunk, &mut state.trunk_cache)?;
        let forced = FishDrama3Tower::teacher_forced_rows(a..b, rows);
        if !forced.is_empty() {
            let frames = prompt_rows(&state.prompt, forced.start..forced.end + 1)?;
            self.tower.forward_span(
                &self.embedding,
                &hidden.narrow(0, 0, forced.len())?,
                &frames,
                forced.start,
                &mut state.tower_cache,
            )?;
        }
        state.check_lengths()?;
        // `h_t` stays where it was made: no readback on an accelerator.
        state.last_hidden = Some(hidden.narrow(0, b - a - 1, 1)?.contiguous()?);
        let width = self.config.tower.frame_width();
        state.last_row = span[span.len() - width..].to_vec();
        state.prefilled = b;
        Ok(FishDrama3Step::Prefill {
            consumed: b - a,
            position: b,
            complete: b == rows,
        })
    }

    /// One frame for each of `rows` inside their open quanta (§3.5): the
    /// compact head at `h_t`, the main draw, a stop on eos (or on a spent
    /// budget), else the tower at `t` for `F_{t+1}` and a trunk step at
    /// `t + 1`. Rows run one at a time through every kernel. A finished row
    /// reports `Finished` and does nothing. On an error abort every row's
    /// quantum.
    pub fn decode_quantum(
        &self,
        rows: &mut [&mut FishDrama3RequestState],
    ) -> Result<Vec<FishDrama3Step>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput("a decode quantum needs a row".into()));
        }
        let ids = self.config.ids;
        let mut steps = Vec::with_capacity(rows.len());
        // The sampled main token of every row that goes on.
        let mut mains = Vec::with_capacity(rows.len());
        let mut drawing = Vec::with_capacity(rows.len());
        for (index, state) in rows.iter_mut().enumerate() {
            self.check_state(state)?;
            state.require_quantum("decode")?;
            if !state.is_prefilled() {
                return Err(Error::InferenceError(
                    "Fish Audio Drama 3 decode cannot run before prefill completes".into(),
                ));
            }
            if let Some(reason) = state.stop {
                steps.push(Some(FishDrama3Step::Finished {
                    frames: state.frames(),
                    reason,
                }));
            } else {
                drawing.push(index);
                steps.push(None);
            }
            mains.push(None);
        }
        let draws = if self.device.is_cpu() {
            drawing
                .iter()
                .map(|&index| {
                    let state = &mut rows[index];
                    let logits = host_values(&self.head.logits(state.last_hidden()?)?)?;
                    let allow_eos = !(state.params.mask_first_eos && state.frames() == 0);
                    state.sampler.sample_main(&logits, allow_eos)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            self.draw_main_on_device(rows, &drawing)?
        };
        for (&index, draw) in drawing.iter().zip(draws) {
            let state = &mut rows[index];
            let frames = state.frames();
            let reason = if draw.token == ids.eos {
                Some(FishDrama3StopReason::EndOfSpeech)
            } else if frames >= state.params.max_frames {
                Some(FishDrama3StopReason::FrameLimit)
            } else {
                None
            };
            if let Some(reason) = reason {
                state.stop = Some(reason);
                steps[index] = Some(FishDrama3Step::Finished { frames, reason });
            } else {
                mains[index] = Some(draw.token);
            }
        }
        if mains.iter().any(Option::is_some) {
            self.advance(rows, &mains)?;
        }
        Ok(rows
            .iter()
            .zip(steps)
            .map(|(state, step)| {
                step.unwrap_or(FishDrama3Step::Frame {
                    frames: state.frames(),
                })
            })
            .collect())
    }

    /// The main draws of `rows[drawing]` on the device: each row's compact
    /// head, ranked once, raced at the request's policy and at RAS's from
    /// exponentials its sampler draws in the host's order, and every row's
    /// picks read back at once (readback 1 of the frame). A draw whose kept
    /// prefix runs past the device's ranks re-runs on the host from the
    /// same exponentials.
    fn draw_main_on_device(
        &self,
        rows: &mut [&mut FishDrama3RequestState],
        drawing: &[usize],
    ) -> Result<Vec<FishDrama3MainDraw>> {
        if drawing.is_empty() {
            return Ok(Vec::new());
        }
        let candidates = self.head.len();
        let mut picks = Vec::with_capacity(drawing.len());
        let mut kept = Vec::with_capacity(drawing.len());
        for &index in drawing {
            let state = &mut *rows[index];
            let allow_eos = !(state.params.mask_first_eos && state.frames() == 0);
            let first_candidate = usize::from(!allow_eos);
            let logits = self.head.logits(state.last_hidden()?)?.squeeze(0)?;
            let (first_noise, retry_noise) = state.sampler.main_noise()?;
            let policy = state.params.sampling.main;
            let ranks = device_cap(&policy)
                .unwrap_or(DEVICE_RANKS)
                .max(FishDrama3DrawPolicy::RAS.top_k.unwrap_or(1))
                .min(DEVICE_RANKS);
            let ranked = rank(&logits, first_candidate, ranks)?;
            let noise = |values: &[f32]| Tensor::from_slice(values, candidates, &self.device);
            let first = device_race(&ranked, &policy, &noise(&first_noise)?)?;
            let retry = device_race(&ranked, &FishDrama3DrawPolicy::RAS, &noise(&retry_noise)?)?;
            picks.push(Tensor::cat(
                &[
                    &first.pick,
                    &retry.pick,
                    &first.non_finite,
                    &first.overflow,
                    &retry.overflow,
                ],
                0,
            )?);
            kept.push((logits, first_noise, retry_noise, first_candidate, policy));
        }
        let values = read_back(&Tensor::cat(&picks, 0)?)?;
        let mut draws = Vec::with_capacity(drawing.len());
        let (rows_read, _) = values.as_chunks::<5>();
        for ((&index, row), (logits, first_noise, retry_noise, first_candidate, policy)) in
            drawing.iter().zip(rows_read).zip(kept)
        {
            let [first, retry, non_finite, first_overflow, retry_overflow] = row;
            // The row's logits on the host, read once if a fallback needs them.
            let mut host: Option<Vec<f32>> = None;
            let mut on_host = |logits: &Tensor| -> Result<Vec<f32>> {
                if let Some(values) = &host {
                    return Ok(values.clone());
                }
                let values = host_values(logits)?;
                host = Some(values.clone());
                Ok(values)
            };
            if *non_finite > 0 {
                // The host sampler's error, with its counts.
                let state = &mut *rows[index];
                return Err(state
                    .sampler
                    .clone()
                    .sample_main(&on_host(&logits)?, first_candidate == 0)
                    .err()
                    .unwrap_or_else(|| {
                        Error::InferenceError(format!(
                            "Fish Audio Drama 3 main logits hold {non_finite} non-finite values"
                        ))
                    }));
            }
            let mut decide = |pick: u32,
                              overflow: u32,
                              policy: &FishDrama3DrawPolicy,
                              noise: &[f32]|
             -> Result<usize> {
                if overflow == 0 {
                    return Ok(pick as usize);
                }
                count_fallback();
                host_race(&on_host(&logits)?, policy, first_candidate, noise)
            };
            let first = decide(*first, *first_overflow, &policy, &first_noise)?;
            let retry = decide(
                *retry,
                *retry_overflow,
                &FishDrama3DrawPolicy::RAS,
                &retry_noise,
            )?;
            draws.push(rows[index].sampler.finish_main(first, retry));
        }
        Ok(draws)
    }

    /// The tower and trunk steps of the rows with a main token.
    fn advance(
        &self,
        rows: &mut [&mut FishDrama3RequestState],
        mains: &[Option<u32>],
    ) -> Result<()> {
        let width = self.config.tower.frame_width();
        let mut hidden = Vec::new();
        let mut tower_rows = Vec::new();
        for (state, main) in rows.iter_mut().zip(mains) {
            let Some(main) = *main else { continue };
            let state: &mut FishDrama3RequestState = state;
            hidden.push(state.last_hidden()?.clone());
            tower_rows.push(FishDrama3TowerStepRow {
                cache: &mut state.tower_cache,
                input: &state.last_row,
                main,
                sampler: &mut state.sampler,
            });
        }
        let hidden = Tensor::cat(&hidden, 0)?;
        let frames = self
            .tower
            .forward_step(&self.embedding, &hidden, &mut tower_rows)?;
        drop(tower_rows);
        let embeds = self.embedding.embed_tensor(&frames, &self.device)?;
        let mut trunk_rows = rows
            .iter_mut()
            .zip(mains)
            .filter(|(_, main)| main.is_some())
            .map(|(state, _)| {
                let state: &mut FishDrama3RequestState = state;
                FishDrama3StepRow {
                    state: &mut state.trunk,
                    cache: &mut state.trunk_cache,
                }
            })
            .collect::<Vec<_>>();
        let next = self.trunk.forward_step(&embeds, &mut trunk_rows)?;
        drop(trunk_rows);
        let going = rows
            .iter_mut()
            .zip(mains)
            .filter(|(_, main)| main.is_some())
            .map(|(state, _)| state);
        for (r, state) in going.enumerate() {
            let frame = &frames[r * width..(r + 1) * width];
            for (codebook, &code) in state.codes.iter_mut().zip(&frame[1..]) {
                codebook.push(code);
            }
            state.last_row = frame.to_vec();
            // On the device: no readback between a frame and the next.
            state.last_hidden = Some(next.narrow(0, r, 1)?.contiguous()?);
            state.check_lengths()?;
        }
        Ok(())
    }

    /// Commits `checkpoint`, or aborts it when the commit refuses.
    fn commit_or_abort(
        &self,
        state: &mut FishDrama3RequestState,
        checkpoint: FishDrama3QuantumCheckpoint,
    ) -> Result<()> {
        if let Err(error) = self.commit_quantum(state, &checkpoint) {
            self.abort_quantum(state, checkpoint)?;
            return Err(error);
        }
        // Nobody acknowledges a private helper's receipts.
        state.take_write_completions();
        Ok(())
    }

    /// Prefills `state` cell by cell, one quantum each, polling `check`
    /// before every cell. On an error the state stays at its last committed
    /// cell.
    pub fn run_prefill(
        &self,
        state: &mut FishDrama3RequestState,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<()> {
        while !state.is_prefilled() {
            check()?;
            let checkpoint = self.begin_quantum(state)?;
            if let Err(error) = self.prefill_span(state) {
                self.abort_quantum(state, checkpoint)?;
                return Err(error);
            }
            self.commit_or_abort(state, checkpoint)?;
        }
        Ok(())
    }

    /// One decode quantum over `rows`, each in its own quantum: all abort
    /// when the step fails; each commits (or aborts alone) otherwise.
    pub fn decode_round(
        &self,
        rows: &mut [&mut FishDrama3RequestState],
    ) -> Result<Vec<FishDrama3Step>> {
        let mut checkpoints = Vec::with_capacity(rows.len());
        for state in rows.iter_mut() {
            match self.begin_quantum(state) {
                Ok(checkpoint) => checkpoints.push(checkpoint),
                Err(error) => {
                    for (state, checkpoint) in rows.iter_mut().zip(checkpoints) {
                        self.abort_quantum(state, checkpoint)?;
                    }
                    return Err(error);
                }
            }
        }
        match self.decode_quantum(rows) {
            Ok(steps) => {
                let mut failure = None;
                for (state, checkpoint) in rows.iter_mut().zip(checkpoints) {
                    if let Err(error) = self.commit_or_abort(state, checkpoint) {
                        failure.get_or_insert(error);
                    }
                }
                failure.map_or(Ok(steps), Err)
            }
            Err(error) => {
                for (state, checkpoint) in rows.iter_mut().zip(checkpoints) {
                    self.abort_quantum(state, checkpoint)?;
                }
                Err(error)
            }
        }
    }

    /// Decodes every row to its stop, polling `check` before every quantum.
    /// On an error every row stays at its last committed frame.
    pub fn run_decode(
        &self,
        rows: &mut [&mut FishDrama3RequestState],
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<()> {
        loop {
            let mut going = rows
                .iter_mut()
                .filter(|state| !state.is_finished())
                .map(|state| &mut **state)
                .collect::<Vec<_>>();
            if going.is_empty() {
                return Ok(());
            }
            check()?;
            self.decode_round(&mut going)?;
        }
    }

    /// A whole generation over private caches: the frames up to eos, or
    /// the frame-limit error.
    pub fn generate_frames(
        &self,
        prompt: impl Into<Arc<FishDrama3Prompt>>,
        params: FishDrama3GenerationParams,
        check: FishDrama3CancelCheck<'_>,
    ) -> Result<FishDrama3Codes> {
        let mut state = self.new_private_request(prompt, params)?;
        self.run_prefill(&mut state, check)?;
        self.run_decode(&mut [&mut state], check)?;
        state
            .stop
            .ok_or_else(|| Error::InferenceError("Fish Audio Drama 3 decode did not stop".into()))?
            .require_complete()?;
        Ok(state.generated_codes())
    }

    /// A whole generation streamed through the codec as the engine runs it
    /// (§7.1): codec quanta on committed frames, the first after 4 frames,
    /// then every 16, the rest after eos; `on_audio` sees each chunk. The
    /// PCM is `frames × samples_per_frame` long.
    pub fn synthesize(
        &self,
        prompt: impl Into<Arc<FishDrama3Prompt>>,
        params: FishDrama3GenerationParams,
        check: FishDrama3CancelCheck<'_>,
        on_audio: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<FishDrama3Synthesis> {
        let mut state = self.new_private_request(prompt, params)?;
        self.run_prefill(&mut state, check)?;
        let mut stream = FishDrama3AudioStream::new();
        let mut samples = Vec::new();
        let mut audio_quanta = Vec::new();
        loop {
            if !state.is_finished() {
                check()?;
                self.decode_round(&mut [&mut state])?;
            }
            if let Some(reason) = state.stop {
                // A cut-off take is an error before its tail is decoded.
                reason.require_complete()?;
            }
            while let Some(frames) = self.audio_quantum_frames(&state, &stream) {
                check()?;
                let candidate = self.stage_audio(&state, &stream, frames, check)?;
                let chunk = stream.commit(candidate)?;
                on_audio(&chunk)?;
                samples.extend(chunk);
                audio_quanta.push(frames);
            }
            if state.is_finished() {
                break;
            }
        }
        let tail = self.finalize_audio(&state, &mut stream, check)?;
        if !tail.is_empty() {
            on_audio(&tail)?;
            samples.extend(tail);
        }
        Ok(FishDrama3Synthesis {
            codes: state.generated_codes(),
            samples,
            sample_rate: self.codec.sample_rate(),
            audio_quanta,
        })
    }
}
