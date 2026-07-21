//! Model executor - handles forward pass execution.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[path = "executor/audio.rs"]
mod audio;
#[path = "executor/dispatch.rs"]
mod dispatch;
#[path = "executor/handler_asr.rs"]
mod handler_asr;
#[path = "executor/handler_audio_chat.rs"]
mod handler_audio_chat;
#[path = "executor/handler_chat.rs"]
mod handler_chat;
#[path = "executor/handler_tts.rs"]
mod handler_tts;
#[path = "executor/prefix_cache.rs"]
mod prefix_cache;
#[path = "executor/state.rs"]
mod state;
#[path = "executor/streaming.rs"]
mod streaming;

pub(crate) use streaming::{
    deliver_committed_streams, CommittedStreamDelivery, IncrementalStreamDeliveryWorkers,
    StreamDeliveryFailure, StreamDeliveryFailureKind,
};

use super::config::EngineCoreConfig;
use super::execution::{
    BatchDispatch, CacheMode, CancellationGranularity, ConcurrencyClass, DispatchState,
    ExecutionCapabilities, ExecutionDisposition, ExecutionFailure, ExecutionMode, ExecutionProfile,
    FailureKind, FailureOrigin, FailureScope, FinishReason, HealthImpact, NativeBatchMode,
    OutcomeProvenance, PhysicalBatch, PlanId, PrefillMode, RetryDisposition, SessionKey,
    YieldReason,
};
use super::output::StreamingOutput;
use super::request::EngineCoreRequest;
use super::resources::{
    BatchWorkspaceLease, ReservationClass, ReservationOwner, ResourceAmount, ResourceAuthority,
    ResourceLease, ResourceVector,
};
use super::scheduler::ScheduledRequest;
use super::types::AudioOutput;
use crate::backends::{
    can_parallelize_requests, BackendContext, BackendKind, BackendPreference, BackendRouter,
    BackendSelectionSource,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::Qwen3ManagedCache;
use crate::models::architectures::qwen3::tts::{Qwen3TtsModel, TtsSessionCacheRequest};
use crate::models::architectures::qwen35::chat::Qwen35PrefixSnapshot;
use crate::models::registry::{AsrModelLease, NativeAsrModel, NativeChatModel, QwenTtsModelLease};
use crate::models::ModelRegistry;
use prefix_cache::{configured_qwen35_prefix_cache_bytes, ExactPrefixCache};
use state::{ActiveAsrDecode, ActiveChatDecode, ActiveQwenTtsDecode};

fn qwen3_managed_cache_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<Qwen3ManagedCache> {
    if reservation.txn_id != scheduled.plan_id || reservation.session != scheduled.session_key() {
        return Err(Error::InferenceError(
            "managed Qwen3 reservation crossed its scheduled row fence".to_string(),
        ));
    }
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed Qwen3 row has no model runtime".to_string())
    })?;
    if reservation.domains.len() != 1 {
        return Err(Error::InvalidInput(
            "native Qwen3 chat requires exactly one managed KV domain".to_string(),
        ));
    }
    let domain = &reservation.domains[0];
    if runtime.plan().model_instance != domain.arena.model_instance
        || runtime.plan().backend != BackendKind::Cpu
    {
        return Err(Error::InferenceError(
            "managed Qwen3 reservation does not match its loaded CPU runtime".to_string(),
        ));
    }
    let group = runtime
        .plan()
        .groups
        .iter()
        .find(|group| group.arena == domain.arena && group.domain == domain.domain)
        .ok_or_else(|| {
            Error::InferenceError(
                "managed Qwen3 reservation references an unresolved arena domain".to_string(),
            )
        })?;
    let crate::kv::ResolvedKvGroupKind::PagedAttention { layers } = &group.kind else {
        return Err(Error::InvalidInput(
            "native Qwen3 chat cannot consume a model-state KV group".to_string(),
        ));
    };
    let table = domain
        .provisional_groups
        .iter()
        .find(|table| table.group == group.id)
        .ok_or_else(|| {
            Error::InferenceError(
                "managed Qwen3 reservation omitted its resolved block table".to_string(),
            )
        })?;
    let arena = runtime.arena(group.arena).ok_or_else(|| {
        Error::InferenceError("managed Qwen3 arena is no longer live".to_string())
    })?;
    Qwen3ManagedCache::new(
        arena.clone(),
        layers.clone(),
        table.blocks.clone(),
        domain.expected_committed_tokens as usize,
    )
}

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

/// Configuration for the model executor.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Path to models directory
    pub models_dir: PathBuf,
    /// Backend to use (cpu, metal, cuda)
    pub backend: BackendKind,
    /// Resolved backend/device context for this worker.
    pub backend_context: BackendContext,
    /// Data type (float32, float16, bfloat16)
    pub dtype: String,
    /// KV cache storage dtype hint (e.g. float16, int8).
    pub kv_cache_dtype: String,
    /// Number of threads
    pub num_threads: usize,
    /// Maximum number of requests to execute in parallel.
    pub request_parallelism: usize,
    /// Decode-time KV cache page size.
    pub kv_page_size: usize,
    /// Optional shared model registry for loaded runtime models.
    pub model_registry: Option<Arc<ModelRegistry>>,
    /// Shared physical resource authority used for model-owned cache lifetime.
    pub resource_authority: Option<Arc<ResourceAuthority>>,
    /// Bytes represented by one scheduler logical KV block for logical
    /// scheduling metrics only. Model-owned physical cache authorization must
    /// come from the loaded model adapter.
    pub logical_kv_block_bytes: u64,
    /// Maximum width of a model-native tensor batch on this backend.
    pub max_tensor_batch_size: usize,
    /// Exact model variants enabled for static tensor execution on this worker.
    pub static_tensor_batch_variants: Arc<HashSet<ModelVariant>>,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("models_dir", &self.models_dir)
            .field("backend", &self.backend)
            .field("backend_context", &self.backend_context)
            .field("dtype", &self.dtype)
            .field("kv_cache_dtype", &self.kv_cache_dtype)
            .field("num_threads", &self.num_threads)
            .field("request_parallelism", &self.request_parallelism)
            .field("kv_page_size", &self.kv_page_size)
            .field(
                "model_registry",
                &self.model_registry.as_ref().map(|_| "<shared>"),
            )
            .field(
                "resource_authority",
                &self.resource_authority.as_ref().map(|_| "<shared>"),
            )
            .field("logical_kv_block_bytes", &self.logical_kv_block_bytes)
            .field("max_tensor_batch_size", &self.max_tensor_batch_size)
            .field(
                "static_tensor_batch_variants",
                &self.static_tensor_batch_variants.len(),
            )
            .finish()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        let backend_context = BackendRouter::resolve_context(
            BackendPreference::Auto,
            BackendSelectionSource::Default,
        );
        let backend_kind = backend_context.backend_kind;
        let num_threads = 4;
        Self {
            models_dir: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models"),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: "float16".to_string(),
            num_threads,
            request_parallelism: Self::request_parallelism_for(backend_kind, num_threads),
            kv_page_size: 64,
            model_registry: None,
            resource_authority: None,
            logical_kv_block_bytes: 0,
            max_tensor_batch_size: 1,
            static_tensor_batch_variants: Arc::new(HashSet::new()),
        }
    }
}

impl From<&EngineCoreConfig> for WorkerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        let backend_context =
            BackendRouter::resolve_context_for_kind(config.backend, BackendSelectionSource::Config);
        let backend_kind = backend_context.backend_kind;
        let num_threads = config.num_threads.max(1);
        Self {
            models_dir: config.models_dir.clone(),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            num_threads,
            request_parallelism: Self::request_parallelism_for(backend_kind, num_threads),
            kv_page_size: config.block_size.max(1),
            model_registry: None,
            resource_authority: None,
            logical_kv_block_bytes: (config.kv_cache_memory_bytes() / config.max_blocks.max(1))
                as u64,
            max_tensor_batch_size: config
                .max_batch_size
                .min(Self::tensor_batch_cap(backend_kind))
                .max(1),
            static_tensor_batch_variants: Arc::new(HashSet::new()),
        }
    }
}

impl WorkerConfig {
    fn tensor_batch_cap(backend: BackendKind) -> usize {
        match backend {
            BackendKind::Cpu | BackendKind::Metal => 2,
            BackendKind::Cuda => 8,
        }
    }

    fn request_parallelism_override() -> Option<usize> {
        std::env::var("IZWI_REQUEST_PARALLELISM")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn resolve_request_parallelism(
        backend: BackendKind,
        num_threads: usize,
        override_value: Option<usize>,
    ) -> usize {
        // Candle's Metal path is intentionally serialized in dispatch. Do not
        // let an environment override inflate coordinator capacity beyond what
        // the executor can actually run concurrently.
        if backend == BackendKind::Metal {
            return 1;
        }
        let default_parallelism = match backend {
            // CPU workloads already use `num_threads` for BLAS/Rayon/intra-op work, so
            // keep inter-request fan-out conservative unless explicitly overridden.
            BackendKind::Cpu => 1,
            BackendKind::Metal => unreachable!("Metal is clamped above"),
            BackendKind::Cuda => num_threads.max(1),
        };

        override_value.unwrap_or(default_parallelism).max(1)
    }

    fn request_parallelism_for(backend: BackendKind, num_threads: usize) -> usize {
        Self::resolve_request_parallelism(
            backend,
            num_threads,
            Self::request_parallelism_override(),
        )
    }
}

/// Output from the executor after a forward pass.
pub const REQUEST_DEADLINE_EXCEEDED: &str = "request deadline exceeded";

#[derive(Debug, Clone)]
pub struct ExecutorOutput {
    /// Request ID
    pub request_id: String,
    /// Generated audio samples
    pub audio: Option<AudioOutput>,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Optional input transcription for speech-to-speech requests.
    pub input_transcription: Option<String>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Optional per-request phase timing override from model-specific execution paths.
    pub phase_timing_override: Option<ExecutorPhaseTiming>,
    /// Optional ASR diagnostics payload surfaced by model-specific paths.
    pub asr_diagnostics: Option<serde_json::Value>,
    /// Error if any
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutorPhaseTiming {
    /// Audio/media decode duration in milliseconds.
    pub media_decode_ms: Option<f64>,
    /// Input normalization duration in milliseconds.
    pub normalization_ms: Option<f64>,
    /// Prefill phase duration in milliseconds.
    pub prefill_ms: Option<f64>,
    /// Decode phase duration in milliseconds.
    pub decode_ms: Option<f64>,
    /// Sampling duration in milliseconds.
    pub sampling_ms: Option<f64>,
    /// Codec encode/decode duration in milliseconds.
    pub codec_ms: Option<f64>,
    /// Postprocess duration in milliseconds.
    pub postprocess_ms: Option<f64>,
    /// Time to first user-visible output in milliseconds since model execution start.
    pub first_output_ms_since_start: Option<f64>,
    /// Number of prefill steps attributed to this request.
    pub prefill_steps: Option<u32>,
    /// Number of decode steps attributed to this request.
    pub decode_steps: Option<u32>,
}

impl ExecutorPhaseTiming {
    pub fn with_media_decode_ms(media_decode_ms: f64) -> Self {
        Self {
            media_decode_ms: Some(media_decode_ms.max(0.0)),
            ..Self::default()
        }
    }
}

impl ExecutorOutput {
    pub fn error(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: Some(error.into()),
        }
    }

    pub fn cancelled(request_id: String) -> Self {
        Self::terminal(request_id)
    }

    /// Construct a terminal payload whose precise outcome is carried by the
    /// authoritative execution disposition.
    pub fn terminal(request_id: String) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        }
    }
}

/// Backend-neutral result produced by one model-owned session safe point.
/// Native handlers must choose sequence, yield, or atomic semantics explicitly.
#[derive(Debug, Clone)]
pub struct ModelSessionResult {
    pub output: ExecutorOutput,
    pub disposition: ExecutionDisposition,
    pub safe_point: bool,
    pub provenance: OutcomeProvenance,
    pub staged_stream_outputs: Vec<StreamingOutput>,
}

impl ModelSessionResult {
    fn executor_failure(message: String) -> ExecutionDisposition {
        ExecutionDisposition::Failed(ExecutionFailure {
            kind: FailureKind::Executor,
            scope: FailureScope::Row,
            retry: RetryDisposition::Never,
            health: HealthImpact::None,
            message,
        })
    }

    pub fn sequence(output: ExecutorOutput) -> Self {
        let disposition = if let Some(message) = output.error.as_ref() {
            Self::executor_failure(message.clone())
        } else if output.finished {
            ExecutionDisposition::Finished(FinishReason::Completed)
        } else {
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        };
        let provenance = if matches!(disposition, ExecutionDisposition::Failed(_)) {
            OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
        } else {
            OutcomeProvenance::produced_output()
        };
        Self {
            output,
            disposition,
            safe_point: true,
            provenance,
            staged_stream_outputs: Vec::new(),
        }
    }

    pub fn yielded(output: ExecutorOutput, reason: YieldReason) -> Self {
        Self {
            output,
            disposition: ExecutionDisposition::Yielded(reason),
            safe_point: true,
            provenance: OutcomeProvenance::produced_output(),
            staged_stream_outputs: Vec::new(),
        }
    }

    pub fn cancelled(mut output: ExecutorOutput) -> Self {
        output.finished = true;
        Self {
            output,
            disposition: ExecutionDisposition::Finished(FinishReason::Cancelled),
            safe_point: true,
            provenance: OutcomeProvenance::started(),
            staged_stream_outputs: Vec::new(),
        }
    }

    pub fn cancelled_before_dispatch(mut output: ExecutorOutput) -> Self {
        output.finished = true;
        Self {
            output,
            disposition: ExecutionDisposition::Finished(FinishReason::Cancelled),
            safe_point: true,
            provenance: OutcomeProvenance::not_started(),
            staged_stream_outputs: Vec::new(),
        }
    }

    pub fn atomic(mut output: ExecutorOutput) -> Self {
        let disposition = if let Some(message) = output.error.as_ref() {
            Self::executor_failure(message.clone())
        } else if output.finished {
            ExecutionDisposition::Finished(FinishReason::Completed)
        } else {
            let message = "atomic model session returned before reaching a terminal state";
            output.error = Some(message.to_string());
            output.finished = true;
            ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message))
        };
        let provenance = if matches!(disposition, ExecutionDisposition::Failed(_)) {
            OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
        } else {
            OutcomeProvenance::produced_output()
        };
        Self {
            output,
            disposition,
            safe_point: true,
            provenance,
            staged_stream_outputs: Vec::new(),
        }
    }

    fn with_staged_stream_outputs(mut self, outputs: Vec<StreamingOutput>) -> Self {
        self.staged_stream_outputs = outputs;
        self
    }
}

/// Executor payload fenced to the exact scheduler transaction that produced it.
#[derive(Debug, Clone)]
pub struct ExecutorStepResult {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub disposition: ExecutionDisposition,
    pub safe_point: bool,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
    /// Physical model-owned cache retained after this safe point. Unknown is
    /// reported explicitly when a backend/model cannot observe all storage.
    pub observed_resources: ResourceVector,
    pub output: ExecutorOutput,
    pub staged_stream_outputs: Vec<StreamingOutput>,
    /// Optional physical KV write acknowledgement for this exact row.
    pub managed_cache: Option<super::ManagedCacheReceipt>,
}

impl ExecutorStepResult {
    pub fn new(scheduled: &ScheduledRequest, output: ExecutorOutput) -> Self {
        let session_result = if output.finished || output.error.is_some() {
            ModelSessionResult::atomic(output)
        } else {
            // Compatibility for third-party/test executors. Native production
            // handlers use `from_session` with an explicit session result.
            ModelSessionResult::sequence(output)
        };
        Self::from_session(scheduled, session_result)
    }

    pub fn from_session(scheduled: &ScheduledRequest, session_result: ModelSessionResult) -> Self {
        Self {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            disposition: session_result.disposition,
            safe_point: session_result.safe_point,
            dispatch: BatchDispatch::serial(),
            provenance: session_result.provenance,
            observed_resources: ResourceVector::zero(),
            output: session_result.output,
            staged_stream_outputs: session_result.staged_stream_outputs,
            managed_cache: None,
        }
    }

    pub fn with_dispatch(mut self, dispatch: BatchDispatch) -> Self {
        self.dispatch = dispatch;
        self
    }

    pub fn with_provenance(mut self, provenance: OutcomeProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_observed_resources(mut self, resources: ResourceVector) -> Self {
        self.observed_resources = resources;
        self
    }

    pub fn with_managed_cache_receipt(mut self, receipt: super::ManagedCacheReceipt) -> Self {
        self.managed_cache = Some(receipt);
        self
    }
}

/// Model executor trait - abstracts the model inference backend.
pub struct PhysicalBatchExecution<'a> {
    pub batch: &'a PhysicalBatch,
    pub requests: &'a [&'a EngineCoreRequest],
    pub scheduled: &'a [ScheduledRequest],
}

#[derive(Debug)]
pub struct PhysicalDispatchError {
    pub error: Error,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
}

impl PhysicalDispatchError {
    pub(crate) fn not_started(error: Error, width: usize, origin: FailureOrigin) -> Self {
        Self {
            error,
            dispatch: BatchDispatch::not_dispatched(width),
            provenance: OutcomeProvenance::failure(origin, DispatchState::NotStarted),
        }
    }

    pub(crate) fn started(error: Error, dispatch: BatchDispatch, origin: FailureOrigin) -> Self {
        Self {
            error,
            dispatch,
            provenance: OutcomeProvenance::failure(origin, DispatchState::Started),
        }
    }
}

pub type PhysicalDispatchResult =
    std::result::Result<Vec<ExecutorStepResult>, PhysicalDispatchError>;

impl PhysicalBatchExecution<'_> {
    pub fn expected_dispatch(&self) -> BatchDispatch {
        self.batch.expected_dispatch()
    }

    pub fn validate(&self) -> Result<()> {
        self.batch.validate()?;
        if self.batch.rows.len() != self.scheduled.len()
            || self.scheduled.len() != self.requests.len()
        {
            return Err(Error::InferenceError(
                "physical executor inputs do not match the batch width".to_string(),
            ));
        }

        let expected = self
            .batch
            .rows
            .iter()
            .map(|row| ((row.plan_id, row.session.clone()), &row.work))
            .collect::<HashMap<_, _>>();
        let mut scheduled_ids = HashSet::with_capacity(self.scheduled.len());
        for scheduled in self.scheduled {
            let key = (scheduled.plan_id, scheduled.session_key());
            let work = expected.get(&key).ok_or_else(|| {
                Error::InferenceError(
                    "scheduled work is not present in the physical batch envelope".to_string(),
                )
            })?;
            if **work != scheduled.work {
                return Err(Error::InferenceError(
                    "scheduled work differs from the physical batch quantum".to_string(),
                ));
            }
            if !scheduled_ids.insert(scheduled.request_id.as_str()) {
                return Err(Error::InferenceError(
                    "physical executor inputs contain a duplicate request".to_string(),
                ));
            }
        }

        let request_ids = self
            .requests
            .iter()
            .map(|request| request.id.as_str())
            .collect::<HashSet<_>>();
        if request_ids.len() != self.requests.len() || request_ids != scheduled_ids {
            return Err(Error::InferenceError(
                "physical executor request snapshots do not match scheduled rows".to_string(),
            ));
        }

        let is_prefill = self.scheduled[0].is_prefill;
        if self
            .scheduled
            .iter()
            .any(|scheduled| scheduled.is_prefill != is_prefill)
        {
            return Err(Error::InferenceError(
                "one physical batch cannot mix prefill and decode dispatch".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_prefill(&self) -> bool {
        self.scheduled
            .first()
            .is_some_and(|scheduled| scheduled.is_prefill)
    }
}

pub trait ModelExecutor: Send + Sync {
    /// Effective loaded-model/request/backend execution profile. Executors
    /// that cannot prove their behavior return `None` and therefore remain on
    /// the conservative compatibility path.
    fn execution_profile(&self, _request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        None
    }

    /// Effective capabilities. The default is deliberately conservative so an
    /// executor must opt in before the scheduler relies on incremental or batch behavior.
    fn execution_capabilities(&self, request: &EngineCoreRequest) -> ExecutionCapabilities {
        self.execution_profile(request)
            .map(|profile| profile.capabilities())
            .unwrap_or_default()
    }

    /// Execute one already-validated physical batch transaction. Native
    /// tensor adapters override this boundary; compatibility executors retain
    /// their existing phase methods at width one.
    fn execute_physical_batch(
        &self,
        execution: PhysicalBatchExecution<'_>,
    ) -> PhysicalDispatchResult {
        let width = execution.scheduled.len().max(1);
        execution.validate().map_err(|error| {
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        let dispatch = execution.expected_dispatch();
        let result = if execution.is_prefill() {
            self.execute_prefill(execution.requests, execution.scheduled)
        } else {
            self.execute_decode(execution.requests, execution.scheduled)
        };
        result
            .map(|mut outputs| {
                let actual_dispatch = if !outputs.is_empty()
                    && outputs
                        .iter()
                        .all(|output| output.provenance.dispatch_state == DispatchState::NotStarted)
                {
                    BatchDispatch::not_dispatched(width)
                } else {
                    dispatch
                };
                for output in &mut outputs {
                    output.dispatch = actual_dispatch;
                }
                outputs
            })
            .map_err(|error| PhysicalDispatchError::started(error, dispatch, FailureOrigin::Model))
    }

    /// Execute prefill pass for newly admitted or in-progress prefill requests.
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>>;

    /// Execute decode pass for running requests.
    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>>;

    /// Check if the executor is ready.
    fn is_ready(&self) -> bool;

    /// Initialize the executor (load models, etc.)
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown the executor.
    fn shutdown(&mut self) -> Result<()>;

    /// Cleanup transient per-request state held by the executor backend.
    fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
        CacheReleaseReport::unconfirmed()
    }

    /// Cleanup state for one exact request incarnation. Legacy executors may
    /// conservatively clear all state for the public request ID.
    fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        self.cleanup_request(&session.request_id)
    }

    /// Purge model-owned reusable cache state before one model is unloaded.
    fn purge_model_cache(&self, _variant: ModelVariant) -> CacheReleaseReport {
        CacheReleaseReport::unconfirmed()
    }
}

/// Proof returned after an executor cache cleanup request. Preemption may only
/// recompute when the executor confirms that the exact session no longer owns
/// tensor cache state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheReleaseReport {
    pub confirmed: bool,
    pub released_sessions: usize,
}

impl CacheReleaseReport {
    pub const fn confirmed(released_sessions: usize) -> Self {
        Self {
            confirmed: true,
            released_sessions,
        }
    }

    pub const fn unconfirmed() -> Self {
        Self {
            confirmed: false,
            released_sessions: 0,
        }
    }
}

#[derive(Debug, Default)]
struct CacheResourceReservation {
    reserved_bytes: u64,
    observed_blocks: usize,
    lease: Option<ResourceLease>,
}

pub struct NativeExecutor {
    config: WorkerConfig,
    initialized: bool,
    loaded_tts_model: Option<Arc<Qwen3TtsModel>>,
    chat_decode_states: Mutex<HashMap<SessionKey, ActiveChatDecode>>,
    qwen35_prefix_cache: ExactPrefixCache<NativeChatModel, Qwen35PrefixSnapshot>,
    asr_decode_states: Mutex<HashMap<SessionKey, ActiveAsrDecode>>,
    qwen_tts_decode_states: Mutex<HashMap<SessionKey, ActiveQwenTtsDecode>>,
    cache_resource_leases: Mutex<HashMap<SessionKey, CacheResourceReservation>>,
}

impl NativeExecutor {
    /// Create a new native executor.
    pub fn new(config: WorkerConfig) -> Self {
        let qwen35_prefix_cache = ExactPrefixCache::new(configured_qwen35_prefix_cache_bytes());
        Self {
            config,
            initialized: false,
            loaded_tts_model: None,
            chat_decode_states: Mutex::new(HashMap::new()),
            qwen35_prefix_cache,
            asr_decode_states: Mutex::new(HashMap::new()),
            qwen_tts_decode_states: Mutex::new(HashMap::new()),
            cache_resource_leases: Mutex::new(HashMap::new()),
        }
    }

    fn qwen_model_for_request(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<(Arc<Qwen3TtsModel>, Option<QwenTtsModelLease>)> {
        if let Some(lease) = request.prepared_qwen_tts_model_lease_for_executor()? {
            return Ok((lease.model_arc(), Some(lease)));
        }
        if let Some(registry) = &self.config.model_registry {
            let variant = request.model_variant.ok_or_else(|| {
                Error::InferenceError("Qwen TTS request is missing model variant".to_string())
            })?;
            let lease = registry.try_get_qwen_tts_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("Qwen TTS model {variant} is not loaded"))
            })?;
            return Ok((lease.model_arc(), Some(lease)));
        }
        self.loaded_tts_model
            .clone()
            .map(|model| (model, None))
            .ok_or_else(|| Error::InferenceError("Executor model not initialized".to_string()))
    }

    fn asr_model_for_request(
        &self,
        request: &EngineCoreRequest,
        variant: ModelVariant,
    ) -> Result<(Arc<NativeAsrModel>, AsrModelLease)> {
        if let Some(lease) = request.prepared_asr_model_lease_for_executor()? {
            return Ok((lease.model_arc(), lease));
        }
        self.with_registry(|registry| {
            let lease = registry.try_get_asr_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
            Ok((lease.model_arc(), lease))
        })
    }

    fn with_registry<T>(&self, f: impl FnOnce(&ModelRegistry) -> Result<T>) -> Result<T> {
        let registry =
            self.config.model_registry.as_ref().ok_or_else(|| {
                Error::InferenceError("Model registry is not configured".to_string())
            })?;
        f(registry)
    }

    fn run_blocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        let run_catching_panic = || {
            let unwind_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            match unwind_result {
                Ok(result) => result,
                Err(payload) => {
                    let message = panic_payload_to_string(payload.as_ref());
                    error!("Model execution panicked: {message}");
                    Err(Error::InferenceError(format!(
                        "Model execution panicked: {message}"
                    )))
                }
            }
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                // Long-running CPU inference should not monopolize Tokio workers; this allows
                // async tasks (including SSE stream forwarding) to continue making progress.
                tokio::task::block_in_place(run_catching_panic)
            }
            _ => run_catching_panic(),
        }
    }

    fn reserve_scheduled_cache(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<()> {
        let Some(authority) = self.config.resource_authority.as_ref() else {
            return Ok(());
        };

        for item in scheduled {
            let Some(request) = requests
                .iter()
                .copied()
                .find(|request| request.id == item.request_id)
            else {
                continue;
            };
            let Some(profile) = self.execution_profile(request) else {
                continue;
            };
            if profile.cache_mode != CacheMode::OpaqueModelOwned
                || !profile.resolved_from_loaded_model
            {
                continue;
            }
            let session = item.session_key();
            self.reserve_exact_session_cache(authority, &session, || {
                self.authorized_session_cache_bytes(request)
            })?;
        }
        Ok(())
    }

    /// Restore every scheduled model-owned cache lease to a pending claim
    /// before model code can replace, release, or fail while mutating its
    /// physical decode state. Successful non-terminal execution reconciles the
    /// new positive observation afterwards.
    fn prepare_scheduled_cache(&self, scheduled: &[ScheduledRequest]) -> Result<()> {
        if self.config.resource_authority.is_none() {
            return Ok(());
        }
        let reservations = self.cache_resource_leases.lock().map_err(|_| {
            Error::InferenceError("cache resource reservation mutex poisoned".to_string())
        })?;
        let zero = cache_resource_vector(self.config.backend, 0);
        for item in scheduled {
            let session = item.session_key();
            let Some(reservation) = reservations.get(&session) else {
                continue;
            };
            let lease = reservation.lease.as_ref().ok_or_else(|| {
                Error::InferenceError("cache allocation has no physical resource lease".to_string())
            })?;
            lease.prepare_materialized_release(zero)?;
        }
        Ok(())
    }

    fn reserve_exact_session_cache(
        &self,
        authority: &Arc<ResourceAuthority>,
        session: &SessionKey,
        authorize: impl FnOnce() -> Result<u64>,
    ) -> Result<()> {
        let mut reservations = self.cache_resource_leases.lock().map_err(|_| {
            Error::InferenceError("cache resource reservation mutex poisoned".to_string())
        })?;
        if reservations.contains_key(session) {
            return Ok(());
        }

        // Authorization is pure, but it remains under the exact-session map
        // lock so concurrent executor entry cannot repeat it or double-reserve.
        // Later decode steps reuse this lease until exact-session cleanup.
        let authorized_bytes = authorize()?;
        let owner = ReservationOwner::new(
            ReservationClass::Cache,
            format!("{}:{}", session.request_id, session.epoch),
        );
        let resources = cache_resource_vector(self.config.backend, authorized_bytes);
        let lease = match self.config.backend {
            BackendKind::Cpu | BackendKind::Metal => authority.track_advisory(owner, resources)?,
            BackendKind::Cuda => authority.reserve(owner, resources)?,
        };
        reservations.insert(
            session.clone(),
            CacheResourceReservation {
                reserved_bytes: authorized_bytes,
                observed_blocks: 0,
                lease: Some(lease),
            },
        );
        Ok(())
    }

    fn authorized_session_cache_bytes(&self, request: &EngineCoreRequest) -> Result<u64> {
        let variant = request.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Request {} is missing model variant for cache authorization",
                request.id
            ))
        })?;
        match request.task_type {
            super::types::TaskType::Chat => {
                let prompt_tokens = request.prompt_tokens.len();
                request
                    .prepared_chat_model_for_executor()?
                    .session_cache_reservation_bytes(
                        prompt_tokens,
                        request.params.max_tokens.max(1),
                    )
            }
            super::types::TaskType::ASR => {
                let (model, _model_lease) = self.asr_model_for_request(request, variant)?;
                model.session_cache_reservation_bytes(
                    request.asr_language_for_execution(),
                    request.asr_prompt_for_execution(),
                    request.params.max_tokens.max(1),
                )
            }
            super::types::TaskType::TTS => {
                let params = Self::to_tts_params(request);
                let text = request
                    .text
                    .as_deref()
                    .ok_or_else(|| Error::InvalidInput("TTS request missing text".to_string()))?;
                let reference = Self::reference_from_request(request)?;
                let (model, _model_lease) = self.qwen_model_for_request(request)?;
                model.session_cache_reservation_bytes(TtsSessionCacheRequest {
                    text,
                    reference: reference.as_ref(),
                    language: request.language.as_deref(),
                    instruct: request.voice_description.as_deref(),
                    uses_preset_speaker: !model.available_speakers().is_empty(),
                    max_frames: params.max_frames,
                })
            }
            super::types::TaskType::SpeechToSpeech => Err(Error::InvalidInput(
                "Speech-to-speech does not expose model-owned session cache authorization"
                    .to_string(),
            )),
        }
    }

    fn observed_session_cache_bytes(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        output: &ExecutorOutput,
    ) -> Option<u64> {
        if output.finished || output.error.is_some() {
            return Some(0);
        }
        let session = scheduled.session_key();
        match request.task_type {
            super::types::TaskType::Chat => self
                .chat_decode_states
                .lock()
                .ok()?
                .get(&session)?
                .state
                .session_cache_bytes(),
            super::types::TaskType::ASR => self
                .asr_decode_states
                .lock()
                .ok()?
                .get(&session)?
                .state
                .session_cache_bytes(),
            super::types::TaskType::TTS => self
                .qwen_tts_decode_states
                .lock()
                .ok()?
                .get(&session)?
                .state
                .session_cache_bytes(),
            super::types::TaskType::SpeechToSpeech => None,
        }
    }

    fn reconcile_scheduled_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        output: &ExecutorOutput,
    ) -> Result<ResourceVector> {
        // Every scheduled cache lease was restored to pending before dispatch.
        // A terminal or failed operation must therefore remain pending until
        // cleanup drops the exact physical state and its lease; never turn its
        // zero observation into a post-operation materialization transition.
        if output.finished || output.error.is_some() {
            return Ok(cache_observation(0));
        }
        let Some(profile) = self.execution_profile(request) else {
            return Ok(ResourceVector::zero());
        };
        if profile.cache_mode != CacheMode::OpaqueModelOwned || !profile.resolved_from_loaded_model
        {
            return Ok(ResourceVector::zero());
        }
        let observed_bytes = require_known_cache_bytes(
            self.observed_session_cache_bytes(request, scheduled, output),
            scheduled,
        )?;
        let observation = cache_observation(observed_bytes);
        if self.config.resource_authority.is_none() {
            return Ok(observation);
        }

        let session = scheduled.session_key();
        let mut reservations = self.cache_resource_leases.lock().map_err(|_| {
            Error::InferenceError("cache resource reservation mutex poisoned".to_string())
        })?;
        let reservation = reservations.get_mut(&session).ok_or_else(|| {
            Error::InferenceError("cache allocation has no exact-session reservation".to_string())
        })?;
        if observed_bytes > 0 {
            if observed_bytes > reservation.reserved_bytes {
                if matches!(self.config.backend, BackendKind::Cpu | BackendKind::Metal) {
                    // CPU and unified-memory Metal use advisory capacity. Candle
                    // can reuse a larger pooled allocation than the requested
                    // power-of-two bucket, so grow the tracked lease to the
                    // exact observed backing rather than failing inference
                    // after the allocation already exists.
                    reservation
                        .lease
                        .as_mut()
                        .ok_or_else(|| {
                            Error::InferenceError(
                                "cache allocation has no physical resource lease".to_string(),
                            )
                        })?
                        .resize(cache_resource_vector(self.config.backend, observed_bytes))?;
                    reservation.reserved_bytes = observed_bytes;
                } else {
                    return Err(Error::InferenceError(format!(
                        "materialized session cache uses {observed_bytes} bytes, exceeding its {}-byte authorization for request {} epoch {}",
                        reservation.reserved_bytes, session.request_id, session.epoch
                    )));
                }
            }
            reservation
                .lease
                .as_ref()
                .ok_or_else(|| {
                    Error::InferenceError(
                        "cache allocation has no physical resource lease".to_string(),
                    )
                })?
                .record_materialized_usage(cache_resource_vector(
                    self.config.backend,
                    observed_bytes,
                ))?;
        }
        reservation.observed_blocks = scheduled.block_ids.len();
        Ok(observation)
    }
}

fn cache_resource_vector(backend: BackendKind, bytes: u64) -> ResourceVector {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(bytes),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(bytes),
        BackendKind::Cuda => resources.device_bytes = ResourceAmount::Known(bytes),
    }
    resources
}

fn cache_observation(bytes: u64) -> ResourceVector {
    ResourceVector {
        kv_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    }
}

fn cache_observation_after_release(report: CacheReleaseReport) -> ResourceVector {
    if report.confirmed {
        cache_observation(0)
    } else {
        unknown_cache_observation()
    }
}

fn require_known_cache_bytes(observed: Option<u64>, scheduled: &ScheduledRequest) -> Result<u64> {
    observed.ok_or_else(|| {
        Error::InferenceError(format!(
            "loaded model did not report cache bytes for session {}:{}",
            scheduled.request_id, scheduled.sequence_id
        ))
    })
}

fn unknown_cache_observation() -> ResourceVector {
    ResourceVector {
        kv_bytes: ResourceAmount::Unknown,
        ..ResourceVector::zero()
    }
}

fn static_qwen_tts_batch_eligible(
    request: &EngineCoreRequest,
    loaded_has_speakers: bool,
    rollout_enabled: bool,
) -> bool {
    matches!(request.task_type, super::types::TaskType::TTS)
        && !request.streaming
        && !request.has_tts_reference_for_execution()
        && request
            .model_variant
            .and_then(|variant| variant.speech_capabilities())
            .is_some_and(|capabilities| capabilities.supports_builtin_voices)
        && loaded_has_speakers
        && rollout_enabled
        && request.execution_adapter_binding().is_some_and(|binding| {
            binding.stages.iter().any(|stage| {
                stage.batch_mode == NativeBatchMode::Static
                    && request.prepared_stage_cost(stage.id).is_some()
            })
        })
}

impl ModelExecutor for NativeExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let variant = request.model_variant?;
        let mut profile = ExecutionProfile::fail_closed(
            self.config.backend,
            Some(variant),
            ExecutionMode::Atomic,
        );
        profile.compute_dtype = self.config.dtype.clone();
        profile.kv_dtype = self.config.kv_cache_dtype.clone();
        profile.cache_namespace = Some(format!(
            "{}:{}:{}:{}",
            variant,
            self.config.backend.as_str(),
            self.config.dtype,
            self.config.kv_cache_dtype
        ));

        let loaded_incremental = match request.task_type {
            super::types::TaskType::Chat => {
                request
                    .prepared_chat_model_for_executor()
                    .ok()
                    .map(|model| match model.as_ref() {
                        NativeChatModel::Qwen3(model) => model.supports_incremental_decode(),
                        NativeChatModel::Qwen35(model) => model.supports_incremental_decode(),
                        NativeChatModel::Gemma3(_) | NativeChatModel::Lfm2(_) => false,
                    })
            }
            super::types::TaskType::ASR => request
                .prepared_asr_model_for_executor()
                .ok()
                .flatten()
                .or_else(|| {
                    self.config
                        .model_registry
                        .as_ref()
                        .and_then(|registry| registry.try_get_asr(variant))
                })
                .map(|model| model.supports_incremental_decode()),
            super::types::TaskType::TTS => {
                let loaded = request
                    .prepared_qwen_tts_model_for_executor()
                    .ok()
                    .flatten()
                    .or_else(|| {
                        self.config
                            .model_registry
                            .as_ref()
                            .and_then(|registry| registry.try_get_qwen_tts(variant))
                    })
                    .is_some()
                    || (self.config.model_registry.is_none() && self.loaded_tts_model.is_some());
                loaded.then_some(variant.family() == crate::catalog::ModelFamily::Qwen3Tts)
            }
            super::types::TaskType::SpeechToSpeech => self
                .config
                .model_registry
                .as_ref()
                .and_then(|registry| registry.try_get_audio_chat(variant))
                .map(|_| false),
        };
        let loaded_has_speakers = request
            .prepared_qwen_tts_model_for_executor()
            .ok()
            .flatten()
            .or_else(|| {
                self.config
                    .model_registry
                    .as_ref()
                    .and_then(|registry| registry.try_get_qwen_tts(variant))
            })
            .or_else(|| self.loaded_tts_model.clone())
            .is_some_and(|model| !model.available_speakers().is_empty());
        let static_tts_batch = static_qwen_tts_batch_eligible(
            request,
            loaded_has_speakers,
            self.config.static_tensor_batch_variants.contains(&variant),
        );
        let continuous_chat_batch = matches!(request.task_type, super::types::TaskType::Chat)
            && request
                .prepared_chat_model_for_executor()
                .ok()
                .is_some_and(|model| model.supports_continuous_decode_batch())
            && request.execution_adapter_binding().is_some_and(|binding| {
                binding
                    .stages
                    .iter()
                    .any(|stage| stage.batch_mode == NativeBatchMode::Continuous)
            });
        profile.resolved_from_loaded_model = loaded_incremental.is_some();
        let implementation_incremental =
            loaded_incremental.unwrap_or_else(|| match request.task_type {
                super::types::TaskType::Chat => {
                    matches!(variant.family(), crate::catalog::ModelFamily::Qwen35Chat)
                        || matches!(
                            variant,
                            ModelVariant::Qwen306B
                                | ModelVariant::Qwen306B4Bit
                                | ModelVariant::Qwen317B
                                | ModelVariant::Qwen317B4Bit
                        )
                }
                super::types::TaskType::ASR => {
                    variant.family() == crate::catalog::ModelFamily::Qwen3Asr
                }
                super::types::TaskType::TTS => {
                    variant.family() == crate::catalog::ModelFamily::Qwen3Tts
                }
                super::types::TaskType::SpeechToSpeech => false,
            });

        if implementation_incremental
            && (!matches!(request.task_type, super::types::TaskType::ASR) || request.streaming)
        {
            profile.mode = ExecutionMode::Sequence;
            profile.prefill = PrefillMode::Full;
            profile.incremental_decode = true;
            profile.cache_mode = CacheMode::OpaqueModelOwned;
            // These adapters keep all mutable decode state inside the exact
            // SessionKey maps below. Removing the entry drops every tensor
            // reference and a fresh prefill can reconstruct it from input.
            profile.recompute_safe = profile.resolved_from_loaded_model;
            profile.cache_release_safe = profile.resolved_from_loaded_model;
        }
        if matches!(request.task_type, super::types::TaskType::ASR) {
            // Long audio can switch to a full chunk-plan operation after media
            // decode, so cancellation is conservatively operation-boundary.
            profile.cancellation = CancellationGranularity::OperationBoundary;
        }

        if static_tts_batch {
            // Preset-speaker Qwen TTS owns a real model tensor-batch API. It is
            // an atomic full-generation operation, not a continuous sequence.
            profile.mode = ExecutionMode::Atomic;
            profile.prefill = PrefillMode::None;
            profile.incremental_decode = false;
            profile.cache_mode = CacheMode::None;
            profile.recompute_safe = false;
            profile.cache_release_safe = false;
            profile.prefill_batch = NativeBatchMode::Static;
            profile.decode_batch = NativeBatchMode::None;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = self.config.max_tensor_batch_size.max(1);
        } else if continuous_chat_batch {
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::Continuous;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = self.config.max_tensor_batch_size.max(1);
        } else {
            let request_parallel_width = if can_parallelize_requests(self.config.backend) {
                self.config.request_parallelism.max(1)
            } else {
                1
            };
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.concurrency = if request_parallel_width > 1 {
                ConcurrencyClass::Batchable
            } else {
                ConcurrencyClass::Exclusive
            };
            profile.max_batch_size = request_parallel_width;
        }
        if request.managed_cache_runtime().is_some() {
            profile.cache_mode = CacheMode::ExternalPaged;
        }
        Some(profile)
    }

    fn execute_physical_batch(
        &self,
        execution: PhysicalBatchExecution<'_>,
    ) -> PhysicalDispatchResult {
        let width = execution.scheduled.len().max(1);
        let expected_dispatch = execution.expected_dispatch();
        execution.validate().map_err(|error| {
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        if !self.initialized {
            return Err(PhysicalDispatchError::not_started(
                Error::InferenceError("Executor not initialized".into()),
                width,
                FailureOrigin::ExecutorValidation,
            ));
        }
        if execution.batch.mode == NativeBatchMode::Static {
            if !execution.is_prefill()
                || execution.batch.lane.capability_id != "tts"
                || execution
                    .requests
                    .iter()
                    .any(|request| request.task_type != super::types::TaskType::TTS)
            {
                return Err(PhysicalDispatchError::not_started(
                    Error::InferenceError(
                        "static tensor batch was routed to an incompatible native stage"
                            .to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            if execution.scheduled.len() > self.config.max_tensor_batch_size.max(1) {
                return Err(PhysicalDispatchError::not_started(
                    Error::Overloaded(
                        "static tensor batch exceeds the backend width cap".to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            return self
                .execute_static_tts_requests(execution.requests, execution.scheduled)
                .map_err(|error| {
                    PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
                });
        }
        if execution.batch.mode == NativeBatchMode::Continuous {
            if execution.is_prefill()
                || execution.batch.lane.capability_id != "chat"
                || execution
                    .requests
                    .iter()
                    .any(|request| request.task_type != super::types::TaskType::Chat)
            {
                return Err(PhysicalDispatchError::not_started(
                    Error::InferenceError(
                        "continuous tensor batch was routed to an incompatible native stage"
                            .to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            if execution.scheduled.len() > self.config.max_tensor_batch_size.max(1) {
                return Err(PhysicalDispatchError::not_started(
                    Error::Overloaded(
                        "continuous tensor batch exceeds the backend width cap".to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            return self
                .execute_continuous_chat_requests_with_rows(
                    execution.requests,
                    execution.scheduled,
                    Some(&execution.batch.rows),
                )
                .map_err(|error| {
                    PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
                });
        }
        let result = if execution.is_prefill() {
            self.execute_requests_with_rows(
                execution.requests,
                execution.scheduled,
                Some(&execution.batch.rows),
            )
        } else {
            self.execute_requests_with_rows(
                execution.requests,
                execution.scheduled,
                Some(&execution.batch.rows),
            )
        };
        result.map_err(|error| {
            PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
        })
    }

    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing native executor");
        if self.config.model_registry.is_none() {
            let device = self.config.backend_context.device.clone();
            let model = Qwen3TtsModel::load(
                &self.config.models_dir,
                device,
                self.config.kv_page_size.max(1),
                &self.config.kv_cache_dtype,
            )?;
            self.loaded_tts_model = Some(Arc::new(model));
            debug!(
                "Native executor loaded TTS model from {:?}",
                self.config.models_dir
            );
        } else {
            debug!("Native executor will use shared model registry");
        }
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down native executor");
        let mut chat = self
            .chat_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("chat decode state mutex poisoned".to_string()))?;
        let mut asr = self
            .asr_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("ASR decode state mutex poisoned".to_string()))?;
        let mut tts = self.qwen_tts_decode_states.lock().map_err(|_| {
            Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
        })?;
        let mut reservations = self.cache_resource_leases.lock().map_err(|_| {
            Error::InferenceError("cache resource reservation mutex poisoned".to_string())
        })?;
        let zero = cache_resource_vector(self.config.backend, 0);
        for reservation in reservations.values() {
            let lease = reservation.lease.as_ref().ok_or_else(|| {
                Error::InferenceError("cache allocation has no physical resource lease".to_string())
            })?;
            lease.prepare_materialized_release(zero)?;
        }
        chat.clear();
        self.qwen35_prefix_cache.clear();
        asr.clear();
        tts.clear();
        reservations.clear();
        drop((chat, asr, tts, reservations));
        self.initialized = false;
        self.loaded_tts_model = None;
        Ok(())
    }

    fn cleanup_request(&self, request_id: &str) -> CacheReleaseReport {
        let (Ok(mut chat), Ok(mut asr), Ok(mut tts), Ok(mut reservations)) = (
            self.chat_decode_states.lock(),
            self.asr_decode_states.lock(),
            self.qwen_tts_decode_states.lock(),
            self.cache_resource_leases.lock(),
        ) else {
            return CacheReleaseReport::unconfirmed();
        };
        let zero = cache_resource_vector(self.config.backend, 0);
        for (session, reservation) in reservations.iter() {
            if session.request_id != request_id {
                continue;
            }
            let Some(lease) = reservation.lease.as_ref() else {
                return CacheReleaseReport::unconfirmed();
            };
            if lease.prepare_materialized_release(zero).is_err() {
                return CacheReleaseReport::unconfirmed();
            }
        }

        let mut released = 0usize;
        released = released.saturating_add(retain_other_sessions_locked(&mut chat, request_id));
        released = released.saturating_add(retain_other_sessions_locked(&mut asr, request_id));
        released = released.saturating_add(retain_other_sessions_locked(&mut tts, request_id));
        released =
            released.saturating_add(retain_other_sessions_locked(&mut reservations, request_id));
        CacheReleaseReport::confirmed(released)
    }

    fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        let (Ok(mut chat), Ok(mut asr), Ok(mut tts), Ok(mut reservations)) = (
            self.chat_decode_states.lock(),
            self.asr_decode_states.lock(),
            self.qwen_tts_decode_states.lock(),
            self.cache_resource_leases.lock(),
        ) else {
            return CacheReleaseReport::unconfirmed();
        };
        if let Some(reservation) = reservations.get(session) {
            let Some(lease) = reservation.lease.as_ref() else {
                return CacheReleaseReport::unconfirmed();
            };
            if lease
                .prepare_materialized_release(cache_resource_vector(self.config.backend, 0))
                .is_err()
            {
                return CacheReleaseReport::unconfirmed();
            }
        }

        let released = usize::from(chat.remove(session).is_some())
            .saturating_add(usize::from(asr.remove(session).is_some()))
            .saturating_add(usize::from(tts.remove(session).is_some()))
            .saturating_add(usize::from(reservations.remove(session).is_some()));
        CacheReleaseReport::confirmed(released)
    }

    fn purge_model_cache(&self, variant: ModelVariant) -> CacheReleaseReport {
        CacheReleaseReport::confirmed(self.qwen35_prefix_cache.purge_variant(variant))
    }
}

/// Unified executor that wraps a model executor implementation.
#[derive(Clone)]
struct BatchWorkspaceContext {
    backend: BackendKind,
    authority: Arc<ResourceAuthority>,
}

#[derive(Clone)]
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
    batch_workspace: Option<BatchWorkspaceContext>,
}

impl UnifiedExecutor {
    /// Create a new unified executor with native backend.
    pub fn new_native(config: WorkerConfig) -> Self {
        let batch_workspace =
            config
                .resource_authority
                .as_ref()
                .map(|authority| BatchWorkspaceContext {
                    backend: config.backend,
                    authority: authority.clone(),
                });
        Self {
            inner: Arc::new(RwLock::new(Box::new(NativeExecutor::new(config)))),
            batch_workspace,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(executor: Box<dyn ModelExecutor>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(executor)),
            batch_workspace: None,
        }
    }

    pub(super) fn reserve_batch_workspace(
        &self,
        batch: &PhysicalBatch,
    ) -> Result<Option<BatchWorkspaceLease>> {
        if batch.workspace.workspace_bytes()? == 0 {
            return Ok(None);
        }
        let context = self.batch_workspace.as_ref().ok_or_else(|| {
            Error::Overloaded(
                "physical batch requires workspace but no resource authority is installed"
                    .to_string(),
            )
        })?;
        if batch.lane.backend != context.backend {
            return Err(Error::InvalidInput(
                "physical batch workspace backend does not match its executor".to_string(),
            ));
        }
        context
            .authority
            .reserve_batch_workspace(batch.lane.execution_group, batch.batch_id, batch.workspace)
            .map(Some)
    }

    /// Execute one exact physical batch envelope.
    pub async fn execute_physical_batch(
        &self,
        batch: &PhysicalBatch,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> PhysicalDispatchResult {
        let executor = self.inner.read().await;
        executor.execute_physical_batch(PhysicalBatchExecution {
            batch,
            requests,
            scheduled,
        })
    }

    pub async fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let executor = self.inner.read().await;
        executor.execution_profile(request)
    }

    /// Check if ready.
    pub async fn is_ready(&self) -> bool {
        let executor = self.inner.read().await;
        executor.is_ready()
    }

    /// Initialize.
    pub async fn initialize(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.initialize()
    }

    /// Shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.shutdown()
    }

    /// Cleanup transient backend state for a completed/aborted request.
    pub async fn cleanup_request(&self, request_id: &str) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.cleanup_request(request_id)
    }

    pub async fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.cleanup_session(session)
    }

    pub async fn purge_model_cache(&self, variant: ModelVariant) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.purge_model_cache(variant)
    }
}

fn retain_other_sessions_locked<T>(states: &mut HashMap<SessionKey, T>, request_id: &str) -> usize {
    let before = states.len();
    states.retain(|session, _| session.request_id != request_id);
    before.saturating_sub(states.len())
}

/// Decode base64-encoded audio to samples.
pub fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    let (samples, _) = decode_audio_base64_with_rate(audio_b64)?;
    Ok(samples)
}

fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    audio::decode_audio_base64_with_rate(audio_b64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::request::StreamStagingBuffer;
    use crate::engine::{CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot};
    use crate::model::ModelVariant;
    use base64::Engine;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct FixedCapacityProvider {
        capacity: ResourceVector,
    }

    impl PhysicalCapacityProvider for FixedCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: self.capacity,
                source: CapacitySource::Test,
            }
        }
    }

    #[derive(Debug)]
    struct MutableCacheCapacityProvider {
        capacity: ResourceVector,
        available: std::sync::Mutex<ResourceVector>,
    }

    impl MutableCacheCapacityProvider {
        fn new(capacity: ResourceVector, available: ResourceVector) -> Self {
            Self {
                capacity,
                available: std::sync::Mutex::new(available),
            }
        }

        fn set_available(&self, available: ResourceVector) {
            *self
                .available
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = available;
        }
    }

    impl PhysicalCapacityProvider for MutableCacheCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: *self
                    .available
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()),
                source: CapacitySource::Test,
            }
        }
    }

    fn materialized_cache_fixture(
        backend: BackendKind,
        request_id: &str,
    ) -> (
        NativeExecutor,
        Arc<ResourceAuthority>,
        Arc<MutableCacheCapacityProvider>,
        EngineCoreRequest,
        ScheduledRequest,
    ) {
        let capacity = cache_resource_vector(backend, 200);
        let live_headroom = cache_resource_vector(backend, 100);
        let provider = Arc::new(MutableCacheCapacityProvider::new(capacity, live_headroom));
        let authority = Arc::new(ResourceAuthority::new(provider.clone()));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, format!("{request_id}:7")),
                cache_resource_vector(backend, 100),
            )
            .unwrap();
        provider.set_available(cache_resource_vector(backend, 0));
        lease
            .record_materialized_usage(cache_resource_vector(backend, 100))
            .unwrap();

        let mut config = WorkerConfig::default();
        config.backend = backend;
        config.resource_authority = Some(authority.clone());
        let executor = NativeExecutor::new(config);
        let session = SessionKey::new(request_id.to_string(), 7);
        executor.cache_resource_leases.lock().unwrap().insert(
            session,
            CacheResourceReservation {
                reserved_bytes: 100,
                observed_blocks: 1,
                lease: Some(lease),
            },
        );
        let mut request = EngineCoreRequest::tts("cache transition");
        request.id = request_id.to_string();
        let scheduled = ScheduledRequest {
            plan_id: 7,
            request_id: request_id.to_string(),
            sequence_id: 7,
            num_tokens: 1,
            is_prefill: false,
            block_ids: vec![1],
            num_computed_tokens: 1,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Decode,
                input: crate::engine::InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        };
        (executor, authority, provider, request, scheduled)
    }

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.backend, config.backend_context.backend_kind);
    }

    #[test]
    fn test_worker_config_from_engine_config_uses_backend_context() {
        let mut engine = EngineCoreConfig::default();
        engine.backend = BackendKind::Cpu;

        let config = WorkerConfig::from(&engine);
        assert_eq!(config.backend, config.backend_context.backend_kind);
        assert_eq!(config.request_parallelism, 1);
        assert_eq!(
            config.backend_context.source,
            BackendSelectionSource::Config
        );
    }

    #[test]
    fn test_request_parallelism_defaults_are_backend_aware() {
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Metal, 8, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cuda, 8, None),
            8
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, Some(3)),
            3
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Metal, 8, Some(3)),
            1
        );
    }

    #[test]
    fn tensor_batch_caps_are_backend_conservative() {
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Cpu), 2);
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Metal), 2);
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Cuda), 8);
    }

    #[test]
    fn physical_batch_workspace_uses_the_backend_resource_domain_and_releases() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut capacity = ResourceVector::zero();
            match backend {
                BackendKind::Cpu => capacity.host_bytes = ResourceAmount::Known(64),
                BackendKind::Metal => capacity.unified_bytes = ResourceAmount::Known(64),
                BackendKind::Cuda => {
                    capacity.host_bytes = ResourceAmount::Known(64);
                    capacity.device_bytes = ResourceAmount::Known(64);
                }
            }
            let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
                capacity,
            })));
            let mut executor = UnifiedExecutor::new_for_test(Box::new(NativeExecutor::new(
                WorkerConfig::default(),
            )));
            executor.batch_workspace = Some(BatchWorkspaceContext {
                backend,
                authority: authority.clone(),
            });
            let lane = super::super::BatchLaneKey {
                execution_group: super::super::ExecutionGroupId::new(7),
                model_instance: super::super::ModelInstanceId::new(8),
                adapter_instance: super::super::AdapterInstanceId::new(9),
                adapter_abi: super::super::AdapterAbiRevision::new(1),
                capability_id: "test".to_string(),
                stage_id: super::super::StageId::new(1),
                backend,
                device_ordinal: None,
                compute_dtype: "f32".to_string(),
                state_dtype: "f32".to_string(),
                tensor_layout: "exact".to_string(),
                quantization: "none".to_string(),
                state_schema: "none".to_string(),
                kernel_mode: "test".to_string(),
                semantic_mode: "test".to_string(),
                shape_bucket: "exact.1".to_string(),
            };
            let expected_workspace = match backend {
                BackendKind::Cpu => ResourceVector {
                    host_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
                BackendKind::Metal => ResourceVector {
                    unified_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
                BackendKind::Cuda => ResourceVector {
                    host_bytes: ResourceAmount::Known(3),
                    device_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
            };
            let batch = PhysicalBatch {
                batch_id: super::super::BatchId::new(10),
                lane: lane.clone(),
                mode: NativeBatchMode::None,
                budget: super::super::BatchBudget::width_one(),
                rows: vec![super::super::ReadyQuantum {
                    plan_id: 1,
                    session: SessionKey::new("workspace".to_string(), 1),
                    lane,
                    work: super::super::WorkUnit::AtomicJob {
                        kind: "test".to_string(),
                    },
                    cost: super::super::WorkCost::new(1, 1, 8),
                    managed_cache: None,
                }],
                materialized_tensor_elements: 1,
                workspace: expected_workspace,
            };

            let workspace = executor
                .reserve_batch_workspace(&batch)
                .unwrap()
                .expect("workspace lease");
            assert_eq!(workspace.resources(), expected_workspace);
            assert_eq!(authority.snapshot().reservations, 1);
            drop(workspace);
            assert_eq!(authority.snapshot().reservations, 0);
        }
    }

    #[test]
    fn static_tts_batch_eligibility_is_fail_closed() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let mut request = EngineCoreRequest::tts("hello").with_model_variant(variant);
        assert!(!static_qwen_tts_batch_eligible(&request, true, true));

        let model_instance = super::super::ModelInstanceId::new(1);
        request.bind_model_instance(model_instance).unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.max_batch_size = 2;
        let stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(1),
            "tts.generate",
            &profile,
            NativeBatchMode::Static,
        );
        let stage_id = stage.id;
        request
            .bind_execution_adapter(super::super::ExecutionAdapterBinding {
                execution_group_id: super::super::ExecutionGroupId::new(1),
                model_instance_id: model_instance,
                adapter_instance_id: super::super::AdapterInstanceId::new(1),
                adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
                model_variant: variant,
                capability_id: "tts".to_string(),
                stages: Arc::from([stage]),
            })
            .unwrap();
        assert!(!static_qwen_tts_batch_eligible(&request, true, true));
        request
            .install_prepared_stage_cost(stage_id, super::super::WorkCost::new(1, 1, 0))
            .unwrap();
        assert!(static_qwen_tts_batch_eligible(&request, true, true));
        assert!(!static_qwen_tts_batch_eligible(&request, true, false));
        assert!(!static_qwen_tts_batch_eligible(&request, false, true));

        request.streaming = true;
        assert!(!static_qwen_tts_batch_eligible(&request, true, true));
        request.streaming = false;
        request.reference_audio = Some("audio".to_string());
        request.reference_text = Some("reference".to_string());
        assert!(!static_qwen_tts_batch_eligible(&request, true, true));

        let voice_design = EngineCoreRequest::tts("hello")
            .with_model_variant(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
        assert!(!static_qwen_tts_batch_eligible(&voice_design, true, true));
    }

    #[test]
    fn exact_session_cleanup_releases_backend_cache_lease_once() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let capacity = cache_resource_vector(backend, 4096);
            let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
                capacity,
            })));
            let lease = authority
                .reserve(
                    ReservationOwner::new(ReservationClass::Cache, "session:7"),
                    cache_resource_vector(backend, 1024),
                )
                .expect("cache lease");
            let executor = NativeExecutor::new(WorkerConfig::default());
            let session = SessionKey::new("session".to_string(), 7);
            executor
                .cache_resource_leases
                .lock()
                .expect("cache lease map")
                .insert(
                    session.clone(),
                    CacheResourceReservation {
                        reserved_bytes: 1024,
                        observed_blocks: 1,
                        lease: Some(lease),
                    },
                );
            assert_eq!(authority.snapshot().reservations, 1);

            let stale = SessionKey::new("session".to_string(), 6);
            let stale_report = executor.cleanup_session(&stale);
            assert!(stale_report.confirmed);
            assert_eq!(stale_report.released_sessions, 0);
            assert_eq!(authority.snapshot().reservations, 1);

            let report = executor.cleanup_session(&session);
            assert!(report.confirmed);
            assert_eq!(report.released_sessions, 1);
            assert_eq!(authority.snapshot().reservations, 0);
            assert_eq!(executor.cleanup_session(&session).released_sessions, 0);
        }
    }

    #[test]
    fn terminal_incremental_cache_stays_pending_until_exact_cleanup() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let (executor, authority, provider, request, scheduled) =
                materialized_cache_fixture(backend, "terminal-cache");
            executor
                .prepare_scheduled_cache(std::slice::from_ref(&scheduled))
                .unwrap();

            let observed = executor
                .reconcile_scheduled_cache(
                    &request,
                    &scheduled,
                    &ExecutorOutput::terminal(request.id.clone()),
                )
                .unwrap();
            assert_eq!(observed, cache_observation(0));

            // The model's terminal path has released its physical cache. The
            // old lease still owns the same bytes as pending authorization, so
            // another admission cannot spend that newly visible headroom.
            provider.set_available(cache_resource_vector(backend, 100));
            assert!(matches!(
                authority.reserve(
                    ReservationOwner::new(ReservationClass::Request, "terminal-racer"),
                    cache_resource_vector(backend, 50),
                ),
                Err(Error::Overloaded(_))
            ));

            let report = executor.cleanup_session(&scheduled.session_key());
            assert!(report.confirmed);
            assert_eq!(authority.snapshot().reservations, 0);
            let replacement = authority
                .reserve(
                    ReservationOwner::new(ReservationClass::Request, "terminal-replacement"),
                    cache_resource_vector(backend, 50),
                )
                .unwrap();
            drop(replacement);
        }
    }

    #[test]
    fn model_error_cache_cleanup_cannot_double_spend_materialized_release() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let (executor, authority, provider, request, scheduled) =
                materialized_cache_fixture(backend, "error-cache");
            executor
                .prepare_scheduled_cache(std::slice::from_ref(&scheduled))
                .unwrap();

            let observed = executor
                .reconcile_scheduled_cache(
                    &request,
                    &scheduled,
                    &ExecutorOutput::error(request.id.clone(), "model failed"),
                )
                .unwrap();
            assert_eq!(observed, cache_observation(0));

            provider.set_available(cache_resource_vector(backend, 100));
            assert!(matches!(
                authority.reserve(
                    ReservationOwner::new(ReservationClass::Request, "error-racer"),
                    cache_resource_vector(backend, 50),
                ),
                Err(Error::Overloaded(_))
            ));

            let report = executor.cleanup_request(&request.id);
            assert!(report.confirmed);
            assert_eq!(authority.snapshot().reservations, 0);
        }
    }

    #[test]
    fn cache_authorization_runs_once_per_exact_session_on_every_backend() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let capacity = cache_resource_vector(backend, 4096);
            let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
                capacity,
            })));
            let mut config = WorkerConfig::default();
            config.backend = backend;
            let executor = NativeExecutor::new(config);
            let session = SessionKey::new("session".to_string(), 7);
            let authorizations = AtomicUsize::new(0);

            executor
                .reserve_exact_session_cache(&authority, &session, || {
                    authorizations.fetch_add(1, Ordering::Relaxed);
                    Ok(1024)
                })
                .unwrap();
            executor
                .reserve_exact_session_cache(&authority, &session, || {
                    authorizations.fetch_add(1, Ordering::Relaxed);
                    Err(Error::InferenceError(
                        "existing exact session must not be reauthorized".to_string(),
                    ))
                })
                .unwrap();

            assert_eq!(authorizations.load(Ordering::Relaxed), 1);
            assert_eq!(
                authority.snapshot().reserved,
                cache_resource_vector(backend, 1024)
            );
            assert_eq!(
                executor
                    .cache_resource_leases
                    .lock()
                    .unwrap()
                    .get(&session)
                    .unwrap()
                    .reserved_bytes,
                1024
            );

            let next_epoch = SessionKey::new("session".to_string(), 8);
            executor
                .reserve_exact_session_cache(&authority, &next_epoch, || {
                    authorizations.fetch_add(1, Ordering::Relaxed);
                    Ok(1024)
                })
                .unwrap();
            assert_eq!(authorizations.load(Ordering::Relaxed), 2);
        }
    }

    #[test]
    fn cpu_and_metal_cache_authorization_is_advisory_while_cuda_is_guarded() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            let capacity = cache_resource_vector(backend, 8);
            let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
                capacity,
            })));
            let mut config = WorkerConfig::default();
            config.backend = backend;
            let executor = NativeExecutor::new(config);
            let session = SessionKey::new(format!("{backend:?}"), 1);

            executor
                .reserve_exact_session_cache(&authority, &session, || Ok(16))
                .expect("advisory unified-memory cache claim");
            assert_eq!(
                authority.snapshot().reserved,
                cache_resource_vector(backend, 16)
            );
        }

        let backend = BackendKind::Cuda;
        let capacity = cache_resource_vector(backend, 8);
        let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
            capacity,
        })));
        let mut config = WorkerConfig::default();
        config.backend = backend;
        let executor = NativeExecutor::new(config);
        let session = SessionKey::new("cuda".to_string(), 1);
        assert!(matches!(
            executor.reserve_exact_session_cache(&authority, &session, || Ok(16)),
            Err(Error::Overloaded(_))
        ));
    }

    #[test]
    fn chat_cache_authorization_requires_private_exact_preparation() {
        let executor = NativeExecutor::new(WorkerConfig::default());
        let request =
            EngineCoreRequest::chat(Vec::new()).with_model_variant(ModelVariant::Qwen306B);

        let error = executor
            .authorized_session_cache_bytes(&request)
            .expect_err("public prompt fields must not authorize physical cache");
        assert!(error
            .to_string()
            .contains("missing exact model prompt preparation"));
    }

    #[test]
    fn model_owned_cache_observation_is_required() {
        let scheduled = ScheduledRequest {
            plan_id: 9,
            request_id: "cache-contract".to_string(),
            sequence_id: 42,
            num_tokens: 1,
            is_prefill: false,
            block_ids: Vec::new(),
            num_computed_tokens: 1,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Decode,
                input: crate::engine::InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        };

        assert_eq!(
            require_known_cache_bytes(Some(4_096), &scheduled).unwrap(),
            4_096
        );
        let error = require_known_cache_bytes(None, &scheduled).unwrap_err();
        assert!(error.to_string().contains("cache-contract:42"));

        assert_eq!(
            cache_observation_after_release(CacheReleaseReport::confirmed(1)).kv_bytes,
            ResourceAmount::Known(0)
        );
        assert_eq!(
            cache_observation_after_release(CacheReleaseReport::unconfirmed()).kv_bytes,
            ResourceAmount::Unknown
        );
    }

    #[test]
    fn test_run_blocking_converts_panic_to_error() {
        let result = NativeExecutor::run_blocking(|| -> Result<()> {
            panic!("executor panic sentinel");
        });

        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error from panic");
        };
        assert!(message.contains("executor panic sentinel"));
    }

    #[test]
    fn test_run_blocking_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result =
            runtime.block_on(async { NativeExecutor::run_blocking(|| Ok::<_, Error>(())) });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_stages_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result = runtime.block_on(async {
            let tx = StreamStagingBuffer::default();
            let mut sequence = 0usize;
            NativeExecutor::stream_audio(
                &tx,
                "req-1",
                &mut sequence,
                vec![0.1, -0.1],
                24_000,
                false,
            )?;
            let chunk = tx
                .take()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::InferenceError("missing staged chunk".to_string()))?;
            if chunk.request_id != "req-1" || chunk.sequence != 0 || chunk.samples.len() != 2 {
                return Err(Error::InferenceError(
                    "unexpected streamed chunk payload".to_string(),
                ));
            }
            Ok::<(), Error>(())
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_tts_params_uses_model_native_auto_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
        request.params.max_tokens = 0;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn test_to_tts_params_clamps_to_model_native_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        request.params.max_tokens = 50_000;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn unloaded_models_cannot_claim_native_batch_capability() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut config = WorkerConfig::default();
            config.backend = backend;
            config.request_parallelism = 4;
            let executor = NativeExecutor::new(config);
            let mut request = EngineCoreRequest::tts("batch me");
            request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);

            let profile = executor.execution_profile(&request).unwrap();
            assert_eq!(profile.backend, backend);
            assert_eq!(profile.mode, ExecutionMode::Sequence);
            assert_eq!(profile.prefill, PrefillMode::Full);
            assert!(!profile.capabilities().native_batch);
            assert_eq!(profile.decode_batch, NativeBatchMode::None);
            let expected_parallelism = if backend == BackendKind::Metal { 1 } else { 4 };
            assert_eq!(profile.max_batch_size, expected_parallelism);
            assert_eq!(
                profile.concurrency,
                if expected_parallelism > 1 {
                    ConcurrencyClass::Batchable
                } else {
                    ConcurrencyClass::Exclusive
                }
            );
            request.streaming = true;
            assert!(!executor.execution_capabilities(&request).native_batch);
            request.streaming = false;
            request.reference_audio = Some("reference".to_string());
            assert!(!executor.execution_capabilities(&request).native_batch);
        }
    }

    #[test]
    fn model_session_results_declare_safe_points_and_terminal_semantics() {
        let sequence = ModelSessionResult::sequence(ExecutorOutput {
            request_id: "sequence".to_string(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 1,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        });
        assert_eq!(
            sequence.disposition,
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        );
        assert!(sequence.safe_point);

        let atomic = ModelSessionResult::atomic(ExecutorOutput {
            request_id: "atomic".to_string(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        });
        assert!(matches!(
            atomic.disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                kind: FailureKind::InvalidOutput,
                ..
            })
        ));
        assert!(atomic.output.finished);

        let cancelled =
            ModelSessionResult::cancelled(ExecutorOutput::cancelled("cancelled".to_string()));
        assert_eq!(
            cancelled.disposition,
            ExecutionDisposition::Finished(FinishReason::Cancelled)
        );
        assert!(cancelled.output.error.is_none());
    }

    #[test]
    fn decode_audio_base64_with_rate_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            // 2 stereo frames: [L,R]=[0.25,0.75] then [0.5,-0.5]
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let (samples, sample_rate) =
            decode_audio_base64_with_rate(&b64).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        // After downmixing, expected mono values are averages: 0.5 and 0.0.
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample was {}", samples[1]);
    }

    #[test]
    fn decode_request_audio_with_rate_accepts_raw_audio_bytes() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.25f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let request = EngineCoreRequest::asr_bytes(wav_bytes);
        let (samples, sample_rate) =
            audio::decode_request_audio_with_rate(&request).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!(
            (samples[0] - 0.25).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(
            (samples[1] + 0.25).abs() < 0.02,
            "second sample was {}",
            samples[1]
        );
    }
}
