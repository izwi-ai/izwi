//! Engine core - the central orchestrator for inference.
//!
//! The engine core coordinates:
//! - Request scheduling
//! - Model execution
//! - KV cache management
//! - Output processing

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use super::cache::composite::{
    allocate_composite_retained_state, project_composite_paged_contract,
    unload_composite_retained_state, CompositeRetainedStateRuntimeV2,
};
use super::cache::managed::{
    ManagedClaimPolicy, ManagedKvCacheManager, ManagedStateCapacityRequest, PortableFitHeadroom,
    StateFitReserve,
};
use super::cache::physical::{InvocationPhysicalKey, PhysicalStateManager};
use super::config::EngineCoreConfig;
#[cfg(test)]
use super::execution::CacheMode;
use super::execution::{
    AdapterAbiRevision, AdapterInstanceId, BatchBudget, BatchDispatch, BatchId, BatchKey,
    BatchLaneKey, ClockedStateSpan, ConcurrencyClass, DeadlinePhase, DispatchState,
    ExecutionDisposition, ExecutionFailure, ExecutionGroupId, ExecutionMode, ExecutionPlan,
    ExecutionProfile, ExecutionReport, ExecutionState, ExecutionTracker, FailureKind,
    FailureOrigin, FailureScope, FinishReason as ExecutionFinishReason, HealthImpact,
    ModelInstanceId, NativeBatchMode, OutcomeProvenance, OutputVisibility, PhysicalBatch,
    PhysicalLaunchPolicy, PrefillMode, ReadyQuantum, RealtimeStageOutcome, RealtimeSubphase,
    RetryDisposition, SequencePhase, StageId, StageShapePolicy, WorkCost, WorkUnit,
};
use super::execution_group::{
    ExecutedEngineStep, ExecutionGroupRunner, ExecutionPhase, PreparedEngineStep,
    PreparedPhysicalDispatch, PreparedStepRecovery,
};
use super::executor::REQUEST_DEADLINE_EXCEEDED;
use super::executor::{
    deliver_committed_streams, CacheReleaseReport, CommittedStreamDelivery, ExecutorOutput,
    ExecutorStepResult, IncrementalStreamDeliveryWorkers, PendingQuantumDecision,
    PendingQuantumFinalizeStatus, StreamDeliveryFailure, StreamDeliveryFailureKind,
    UnifiedExecutor, WorkerConfig,
};
use super::metrics::{
    model_calls_for_dispatch, record_engine_execution_outcome, record_engine_mixed_batch,
    record_engine_physical_batch, record_engine_step_model_calls,
    record_engine_stream_checkpoint_committed, record_engine_stream_checkpoint_rejection,
    record_engine_workspace_wait, EngineWorkspaceWaitOutcome,
};
use super::output::OutputProcessor;
use super::request::{
    EngineCoreRequest, FencedStreamProgress, RealtimeAsrOperationPayload,
    RealtimeAsrOperationWaiter, RealtimeAsrTerminalOutcome, RequestStatus, StreamProgressBudget,
    STREAM_PROGRESS_MAX_BUFFERED_BYTES, STREAM_PROGRESS_QUEUE_CAPACITY,
};
use super::scheduler::{
    BeginTerminalRelease, ExecutionLane, LaneScope, LaneSelection, Scheduler, SchedulerConfig,
    TerminalReleaseCause,
};
use super::types::{EngineOutput, FinishReason as OutputFinishReason, LatencyBreakdown, RequestId};
use super::{ResourceAmount, ResourceVector};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

#[cfg(test)]
use super::types::AudioOutput;

mod commit;
mod delivery;
mod formation;
mod managed_state;
mod plan;
mod release;
mod step;
use managed_state::reprobe_managed_prefix;
#[cfg(test)]
use managed_state::resolve_managed_token_reach;

#[derive(Debug, Clone, Default)]
struct RequestPhaseTiming {
    pre_engine_ms: Option<f64>,
    first_scheduled_at: Option<Instant>,
    queue_wait_ms: f64,
    media_decode_ms: Option<f64>,
    normalization_ms: Option<f64>,
    prefill_ms: f64,
    decode_ms: f64,
    decode_started_at: Option<Instant>,
    last_decode_commit_at: Option<Instant>,
    decode_tokens: usize,
    first_token_commit_at: Option<Instant>,
    last_token_commit_at: Option<Instant>,
    sampling_ms: Option<f64>,
    codec_ms: Option<f64>,
    postprocess_ms: Option<f64>,
    first_output_ms: Option<f64>,
    prefill_steps: u32,
    decode_steps: u32,
}

impl RequestPhaseTiming {
    fn admitted(request: &EngineCoreRequest, inserted_at: Instant) -> Self {
        Self {
            pre_engine_ms: request.runtime_admitted_at.map(|admitted| {
                inserted_at
                    .saturating_duration_since(admitted)
                    .as_secs_f64()
                    * 1000.0
            }),
            ..Self::default()
        }
    }

    /// Record `tokens` committed at `at`; returns the gap since the previous
    /// token commit, when there was one.
    fn record_token_commit(
        &mut self,
        decode: bool,
        tokens: usize,
        at: Instant,
    ) -> Option<Duration> {
        if tokens == 0 {
            return None;
        }
        let gap = self
            .last_token_commit_at
            .map(|previous| at.saturating_duration_since(previous));
        self.first_token_commit_at.get_or_insert(at);
        self.last_token_commit_at = Some(at);
        if decode {
            self.decode_tokens = self.decode_tokens.saturating_add(tokens);
            self.last_decode_commit_at = Some(at);
        }
        gap
    }

    fn decode_wall_ms(&self) -> Option<f64> {
        Some(
            self.last_decode_commit_at?
                .saturating_duration_since(self.decode_started_at?)
                .as_secs_f64()
                * 1000.0,
        )
    }

    fn post_first_token_ms(&self) -> Option<f64> {
        Some(
            self.last_token_commit_at?
                .saturating_duration_since(self.first_token_commit_at?)
                .as_secs_f64()
                * 1000.0,
        )
    }
}

/// Steps on which one session's execution plan may fail to begin before the
/// request fails. The run loop retries a failed step every 50 ms at most, so
/// a plan that can never begin used to retry forever (TPIPE-15, LFA-10); a
/// transient failure clears well within this bound.
const MAX_PLAN_BEGIN_ATTEMPTS: u32 = 64;

/// Count one more failed plan begin for `session`: whether its attempts are
/// now exhausted.
fn plan_begin_attempts_exhausted(
    attempts: &mut HashMap<super::SessionKey, u32>,
    session: &super::SessionKey,
) -> bool {
    let count = attempts.entry(session.clone()).or_insert(0);
    *count = count.saturating_add(1);
    *count >= MAX_PLAN_BEGIN_ATTEMPTS
}

/// One scheduled row whose managed-state preparation failed with an error
/// other than capacity backpressure.
#[derive(Debug)]
struct ManagedPrepareFailure {
    scheduled: super::scheduler::ScheduledRequest,
    session: super::SessionKey,
    error: Error,
}

/// Why [`EngineCore::begin_execution_plan`] did not register a plan.
#[derive(Debug)]
enum PlanBeginError {
    /// The request's bound adapter cannot run the work its executor profile
    /// schedules: no single stage matches it, or the stage's batching
    /// disagrees with the executor. The binding, the profile and the work
    /// derived from them are fixed for the session, and the scheduler has
    /// already fallen back to scalar execution, so a retry selects the same
    /// work and fails the same way. The request fails terminally.
    AdapterContract(Error),
    /// Any other failure rolls the schedule back for the run loop to retry.
    Engine(Error),
}

impl From<Error> for PlanBeginError {
    fn from(error: Error) -> Self {
        Self::Engine(error)
    }
}

#[derive(Debug)]
struct CommittedExecutorOutput {
    session: super::SessionKey,
    output: ExecutorOutput,
    disposition: ExecutionDisposition,
    provenance: OutcomeProvenance,
    staged_stream_outputs: Vec<super::output::StreamingOutput>,
}

pub(super) struct CommittedEngineStep {
    pub(super) outputs: Vec<EngineOutput>,
    pub(super) stream_deliveries: Vec<CommittedStreamDelivery>,
}

#[derive(Debug)]
pub(super) struct StreamProgressRejection {
    pub(super) kind: StreamDeliveryFailureKind,
    message: String,
}

impl StreamProgressRejection {
    fn invalid(message: impl Into<String>) -> Self {
        record_engine_stream_checkpoint_rejection();
        Self {
            kind: StreamDeliveryFailureKind::InvalidProgress,
            message: message.into(),
        }
    }

    fn cancelled() -> Self {
        record_engine_stream_checkpoint_rejection();
        Self {
            kind: StreamDeliveryFailureKind::Cancelled,
            message: "stream progress request was cancelled".to_string(),
        }
    }

    fn deadline() -> Self {
        record_engine_stream_checkpoint_rejection();
        Self {
            kind: StreamDeliveryFailureKind::RequestDeadline,
            message: "stream progress request deadline elapsed".to_string(),
        }
    }
}

impl std::fmt::Display for StreamProgressRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

struct PhysicalBatchAssembly {
    physical_batch: PhysicalBatch,
    requests: Vec<Arc<EngineCoreRequest>>,
    scheduled: Vec<super::scheduler::ScheduledRequest>,
    output_visibility: super::OutputVisibility,
    shape_policy: StageShapePolicy,
    workspace_base_bytes: u64,
    launch_policy: PhysicalLaunchPolicy,
    /// Session plans and summed work of the validated rows, so a push
    /// validates only its own row (SCH-F9).
    row_keys: HashSet<(super::SessionKey, u64)>,
    row_cost: WorkCost,
}

/// A row [`PhysicalBatchAssembly::try_push`] did not accept, handed back.
type RefusedBatchRow = (
    Arc<EngineCoreRequest>,
    super::scheduler::ScheduledRequest,
    ReadyQuantum,
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveStreamBatch {
    lane: BatchLaneKey,
    output_visibility: OutputVisibility,
    rows: HashMap<u64, super::SessionKey>,
}

/// Core-owned identity snapshot for one physical ticket that has left the
/// core lock but has not yet committed or rolled back.
#[derive(Debug, Clone)]
struct InFlightPhysicalDispatch {
    registered_at: Instant,
    phase: ExecutionPhase,
    physical_batch: PhysicalBatch,
    scheduled: Vec<super::scheduler::ScheduledRequest>,
    output_visibility: OutputVisibility,
    managed_cache_reservations: Vec<super::ManagedCacheReservation>,
    launch_policy: PhysicalLaunchPolicy,
    /// Execution lanes of the rows, fixed at registration.
    lanes: Vec<ExecutionLane>,
}

impl InFlightPhysicalDispatch {
    fn from_prepared(dispatch: &PreparedPhysicalDispatch, lanes: Vec<ExecutionLane>) -> Self {
        Self {
            registered_at: Instant::now(),
            phase: dispatch.phase(),
            physical_batch: dispatch.physical_batch().clone(),
            scheduled: dispatch.scheduled().to_vec(),
            output_visibility: dispatch.output_visibility(),
            managed_cache_reservations: dispatch.managed_cache_reservations().to_vec(),
            launch_policy: dispatch.launch_policy(),
            lanes,
        }
    }

    fn contains_session(&self, session: &super::SessionKey) -> bool {
        self.physical_batch
            .rows
            .iter()
            .any(|row| &row.session == session)
    }

    fn contains_plan(&self, plan_id: u64) -> bool {
        self.physical_batch
            .rows
            .iter()
            .any(|row| row.plan_id == plan_id)
    }

    fn stream_fence(&self) -> Option<ActiveStreamBatch> {
        (self.output_visibility == OutputVisibility::IncrementalCommitted).then(|| {
            ActiveStreamBatch {
                lane: self.physical_batch.lane.clone(),
                output_visibility: self.output_visibility,
                rows: self
                    .physical_batch
                    .rows
                    .iter()
                    .map(|row| (row.plan_id, row.session.clone()))
                    .collect(),
            }
        })
    }

    fn validate_completion(
        &self,
        completed: &super::execution_group::ExecutedPhysicalBatch,
    ) -> Result<()> {
        if self.phase != completed.phase
            || self.physical_batch != completed.physical_batch
            || self.managed_cache_reservations != completed.managed_cache_reservations
        {
            return Err(Error::InferenceError(
                "physical completion does not match its registered dispatch ticket".to_string(),
            ));
        }
        Ok(())
    }
}

/// Chunked-claim holders of one model seen blocked on managed capacity since
/// its holder set last changed. Claims only grow until a holder releases, so
/// a holder blocked under an unchanged set is still blocked.
#[derive(Debug, Default)]
struct ClaimWaiters {
    holders: Vec<super::SessionKey>,
    blocked: HashSet<super::SessionKey>,
}

/// One session's run of refused batch workspaces.
#[derive(Debug, Clone, Copy)]
struct WorkspaceWait {
    started: Instant,
    /// Attempts since the last shape reduction; drives the backoff.
    attempts: u32,
}

impl WorkspaceWait {
    fn start() -> Self {
        Self {
            started: Instant::now(),
            attempts: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LifecycleRetryPolicy {
    max_execution_retries: u32,
    execution_backoff_base: Duration,
    execution_backoff_max: Duration,
    /// How long a session may keep retrying a refused batch workspace while
    /// no other session makes progress. A refusal happens before device
    /// entry, so retrying is exact; this replaces the attempt budget.
    workspace_wait_budget: Duration,
    cleanup_backoff_base: Duration,
    cleanup_backoff_max: Duration,
    cleanup_budget_per_step: usize,
}

impl Default for LifecycleRetryPolicy {
    fn default() -> Self {
        Self {
            max_execution_retries: 3,
            execution_backoff_base: Duration::from_millis(5),
            execution_backoff_max: Duration::from_millis(200),
            // Several CUDA capacity samples (fresh for 250 ms, stale for at
            // most 1 s): three attempts 35 ms apart all read the same sample.
            workspace_wait_budget: Duration::from_secs(5),
            cleanup_backoff_base: Duration::from_millis(25),
            cleanup_backoff_max: Duration::from_secs(1),
            cleanup_budget_per_step: 16,
        }
    }
}

impl LifecycleRetryPolicy {
    fn exponential_delay(base: Duration, max: Duration, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1).min(20);
        base.saturating_mul(1u32 << exponent).min(max)
    }

    fn execution_delay(self, attempt: u32) -> Duration {
        Self::exponential_delay(
            self.execution_backoff_base,
            self.execution_backoff_max,
            attempt,
        )
    }

    fn cleanup_delay(self, attempt: u32) -> Duration {
        Self::exponential_delay(self.cleanup_backoff_base, self.cleanup_backoff_max, attempt)
    }
}

/// The context a chat model needs to be useful on CPU and Metal: below it a
/// load is refused (503) instead of publishing, say, a 64-token context that
/// rejects every real prompt.
const CHAT_MINIMUM_CONTEXT_TOKENS: u64 = 4_096;

/// How a chat model's CUDA KV pool is sized at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaResidentPool {
    /// No resident fit: the arena grows on demand up to the model context.
    Disabled,
    /// A materialized arena of at most one model context, fitted to device
    /// headroom after the CUDA context reserve and decode workspace. For
    /// small models that share the GPU.
    OneContext,
    /// Like [`Self::OneContext`], but with incremental CUDA chat the pool may
    /// cover every retained row's context, up to device headroom.
    RetainedRows,
}

fn merge_optional_phase_ms(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        let value = value.max(0.0);
        *target = Some(target.unwrap_or(0.0) + value);
    }
}

/// A completed chat row reports `MaxTokens` when it spent its whole budget.
///
/// The output processor maps every completed disposition to `StopToken`, so
/// a reply cut off at `max_tokens` would read as a natural stop. Every chat
/// family counts only emitted tokens (stop tokens are never counted), and the
/// request's `max_tokens` is already bounded by the remaining context, so a
/// row whose count reached the budget was cut off.
fn chat_budget_finish_reason(
    reason: Option<OutputFinishReason>,
    task_type: super::types::TaskType,
    max_tokens: usize,
    generated: usize,
) -> Option<OutputFinishReason> {
    if reason == Some(OutputFinishReason::StopToken)
        && task_type == super::types::TaskType::Chat
        && max_tokens > 0
        && generated >= max_tokens
    {
        Some(OutputFinishReason::MaxTokens)
    } else {
        reason
    }
}

/// Execution plans keyed by plan id, with a per-session index so that the
/// checks run for every scheduled row are lookups instead of scans of every
/// active plan (SCH-F9). Reads go through the map; every mutation goes
/// through a method that keeps the index exact.
#[derive(Debug, Clone, Default)]
struct ActivePlans {
    by_plan: HashMap<u64, ExecutionPlan>,
    sessions: HashMap<super::SessionKey, usize>,
}

impl std::ops::Deref for ActivePlans {
    type Target = HashMap<u64, ExecutionPlan>;

    fn deref(&self) -> &Self::Target {
        &self.by_plan
    }
}

impl ActivePlans {
    fn contains_session(&self, session: &super::SessionKey) -> bool {
        self.sessions.contains_key(session)
    }

    fn index(sessions: &mut HashMap<super::SessionKey, usize>, session: &super::SessionKey) {
        *sessions.entry(session.clone()).or_default() += 1;
    }

    fn unindex(sessions: &mut HashMap<super::SessionKey, usize>, session: &super::SessionKey) {
        if let Some(count) = sessions.get_mut(session) {
            *count -= 1;
            if *count == 0 {
                sessions.remove(session);
            }
        }
    }

    fn insert(&mut self, plan_id: u64, plan: ExecutionPlan) -> Option<ExecutionPlan> {
        Self::index(&mut self.sessions, &plan.session);
        let replaced = self.by_plan.insert(plan_id, plan);
        if let Some(replaced) = replaced.as_ref() {
            Self::unindex(&mut self.sessions, &replaced.session);
        }
        replaced
    }

    fn remove(&mut self, plan_id: &u64) -> Option<ExecutionPlan> {
        let removed = self.by_plan.remove(plan_id);
        if let Some(removed) = removed.as_ref() {
            Self::unindex(&mut self.sessions, &removed.session);
        }
        removed
    }

    fn retain(&mut self, mut keep: impl FnMut(&u64, &ExecutionPlan) -> bool) {
        let sessions = &mut self.sessions;
        self.by_plan.retain(|plan_id, plan| {
            let kept = keep(plan_id, plan);
            if !kept {
                Self::unindex(sessions, &plan.session);
            }
            kept
        });
    }

    fn clear(&mut self) {
        self.by_plan.clear();
        self.sessions.clear();
    }

    #[cfg(test)]
    fn extend(&mut self, plans: impl IntoIterator<Item = (u64, ExecutionPlan)>) {
        for (plan_id, plan) in plans {
            self.insert(plan_id, plan);
        }
    }

    /// Mutable access for tests that reshape a plan; the session is the index
    /// key and must not change.
    #[cfg(test)]
    fn get_mut(&mut self, plan_id: &u64) -> Option<&mut ExecutionPlan> {
        self.by_plan.get_mut(plan_id)
    }
}

/// A session's execution profile, accepted by its adapter contract. A
/// profile resolved from the loaded model depends only on the request and
/// that model, so it is resolved once per session instead of on every step.
struct CachedExecutionProfile {
    epoch: u64,
    profile: Arc<ExecutionProfile>,
}

/// The engine core - manages the inference loop.
pub struct EngineCore {
    /// Configuration
    config: EngineCoreConfig,
    /// Request scheduler
    scheduler: Scheduler,
    /// Physical managed-cache arenas and transactional block tables.
    managed_kv_cache: ManagedKvCacheManager,
    /// Lifecycle owner for invocation-scoped physical state arenas.
    physical_state: PhysicalStateManager,
    /// Strong load-generation identity for composite retained backings.
    composite_retained_state: HashMap<ModelInstanceId, Arc<CompositeRetainedStateRuntimeV2>>,
    /// Model executor
    executor: UnifiedExecutor,
    /// Output processor
    output_processor: OutputProcessor,
    /// Active requests (by ID)
    requests: HashMap<RequestId, Arc<EngineCoreRequest>>,
    /// Execution profiles resolved from a loaded model, by request and
    /// keyed to the session epoch they were resolved for (SCH-F9).
    execution_profiles: HashMap<RequestId, CachedExecutionProfile>,
    /// Request start times (for timing)
    request_start_times: HashMap<RequestId, Instant>,
    /// Per-request phase timing accumulated by scheduler steps.
    request_phase_timings: HashMap<RequestId, RequestPhaseTiming>,
    /// Per-session lifecycle and active-plan fence.
    execution_trackers: HashMap<RequestId, ExecutionTracker>,
    /// Plans prepared under the core lock and awaiting one validated result.
    active_plans: ActivePlans,
    /// Exact physical tickets currently owned outside the core lock.
    in_flight_dispatches: HashMap<BatchId, InFlightPhysicalDispatch>,
    /// Recovery left armed if a borrowed low-level step future is cancelled.
    pending_runner_recovery: Option<PreparedStepRecovery>,
    /// Prepared physical KV transactions keyed by their exact execution plan.
    active_managed_cache: HashMap<u64, super::ManagedCacheReservation>,
    /// Terminal sessions whose table release waits for an in-flight row.
    pending_managed_releases: HashSet<super::SessionKey>,
    /// Exact cleanup confirmations registered by retained Runtime owners.
    /// Senders remain armed across executor/managed cleanup retries.
    cleanup_confirmation_waiters: HashMap<super::SessionKey, Vec<oneshot::Sender<Result<()>>>>,
    /// Exact physical envelopes allowed to publish pre-quantum progress.
    active_stream_batches: HashMap<BatchId, ActiveStreamBatch>,
    /// Next sequence number accepted for each exact streaming session.
    stream_sequence_cursors: HashMap<super::SessionKey, usize>,
    /// Sessions that have committed progress before their physical quantum.
    incremental_stream_sessions: HashSet<super::SessionKey>,
    /// Durable typed terminal events created outside a committed executor
    /// result (for example cancellation, deadline expiry, or failed
    /// preemption) and delivered by the next successful engine step.
    pending_terminal_outputs: VecDeque<CommittedExecutorOutput>,
    /// Consecutive retryable executor failures for each exact session.
    execution_retry_attempts: HashMap<super::SessionKey, u32>,
    /// Consecutive steps on which a session's execution plan failed to
    /// begin; bounded by [`MAX_PLAN_BEGIN_ATTEMPTS`] (TPIPE-15).
    plan_begin_attempts: HashMap<super::SessionKey, u32>,
    /// Consecutive steps whose managed-state preparation failed for a session
    /// with a non-backpressure error; bounded by [`MAX_PLAN_BEGIN_ATTEMPTS`]
    /// so one failing row can never stall every other row (H2).
    managed_prepare_attempts: HashMap<super::SessionKey, u32>,
    workspace_progress_epoch: u64,
    workspace_retry_progress: HashMap<super::SessionKey, u64>,
    /// Each session's current run of refused batch workspaces.
    workspace_waits: HashMap<super::SessionKey, WorkspaceWait>,
    workspace_row_limits: HashMap<super::SessionKey, usize>,
    workspace_pressure: HashMap<super::SessionKey, super::scheduler::ScheduledRequest>,
    /// Claim holders seen blocked, per model, for the claim deadlock breaker.
    claim_waiters: HashMap<ModelInstanceId, ClaimWaiters>,
    retry_policy: LifecycleRetryPolicy,
    #[cfg(test)]
    restart_after_reset_hook: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Model calls of each committed step that dispatched work.
    #[cfg(test)]
    step_model_calls: Vec<usize>,
    /// Phase preferred when decode and prefill compete for a single physical
    /// candidate slot. It advances only after the preferred phase actually
    /// owns a ticket, so blocked work does not waste the device opportunity.
    next_single_candidate_phase: HashMap<LaneScope, SequencePhase>,
    /// Whether the engine has been initialized
    initialized: bool,
    /// Step counter for periodic cache housekeeping.
    /// Monotonic identity for physical dispatch envelopes.
    next_batch_id: u64,
}

impl EngineCore {
    /// The engine is about to wait for work: end the device's busy period.
    pub(crate) fn end_device_busy_period(&self) {
        self.executor.end_device_busy_period();
    }

    /// The thread that runs this engine's device work (model calls, KV page
    /// operations, state allocation and frees), if its executor has a device.
    pub(crate) fn device_thread(&self) -> Option<Arc<crate::backends::DeviceThread>> {
        self.executor.device_thread().cloned()
    }

    pub(crate) fn synchronize_worker_device(&self) -> Result<()> {
        self.managed_kv_cache.synchronize_worker()
    }
    fn force_scalar_execution(mut profile: ExecutionProfile) -> ExecutionProfile {
        profile.prefill_batch = NativeBatchMode::None;
        profile.decode_batch = NativeBatchMode::None;
        profile.max_batch_size = 1;
        profile.concurrency = ConcurrencyClass::Exclusive;
        profile
    }

    /// Create a new engine core.
    pub fn new(config: EngineCoreConfig) -> Result<Self> {
        let worker_config = WorkerConfig::from(&config);
        Self::new_with_worker(config, worker_config)
    }

    /// Create a new engine core with an explicit worker configuration.
    pub fn new_with_worker(config: EngineCoreConfig, worker_config: WorkerConfig) -> Result<Self> {
        info!("Creating engine core");

        let managed_resource_authority = worker_config.resource_authority.clone();
        let managed_worker_backend = worker_config.backend_context.backend_kind;
        let managed_worker_device = worker_config.backend_context.device.device.clone();
        let executor = UnifiedExecutor::new_native(worker_config);
        Self::new_with_executor_and_managed_authority(
            config,
            executor,
            managed_resource_authority,
            Some((managed_worker_backend, managed_worker_device)),
        )
    }

    #[cfg(test)]
    fn new_with_executor(config: EngineCoreConfig, executor: UnifiedExecutor) -> Result<Self> {
        Self::new_with_executor_and_managed_authority(config, executor, None, None)
    }

    fn new_with_executor_and_managed_authority(
        config: EngineCoreConfig,
        executor: UnifiedExecutor,
        managed_resource_authority: Option<Arc<super::ResourceAuthority>>,
        managed_worker: Option<(BackendKind, candle_core::Device)>,
    ) -> Result<Self> {
        let mut config = config;
        // Direct EngineCore users must cross the same fail-closed cache-policy
        // boundary as RuntimeService users before workers or arenas start.
        // Catalog-auto prefix reuse resolves to one namespace here so the
        // policy and the managed cache salt cannot diverge.
        config.apply_prefix_engagement();
        let cache_policy = config.resolved_kv_cache_policy()?;
        // Create scheduler
        let scheduler_config = SchedulerConfig::from(&config);
        let scheduler = Scheduler::new(scheduler_config);

        // Create output processor
        let output_processor = OutputProcessor::new(config.sample_rate);
        let managed_prefix_salt = config
            .enable_prefix_caching
            .then_some(config.managed_prefix_cache_salt.as_deref())
            .flatten()
            .map(|salt| Sha256::digest(salt.as_bytes()).into());
        let max_prefix_cache_pages = match cache_policy.effective.prefix {
            crate::config::PrefixCachePolicy::Disabled => 0,
            crate::config::PrefixCachePolicy::Namespaced { max_pages, .. } => max_pages,
        };

        let (mut managed_kv_cache, physical_state) = match managed_worker {
            Some((backend, device)) => (
                ManagedKvCacheManager::for_worker_with_prefix_cache_policy(
                    managed_resource_authority.clone(),
                    managed_prefix_salt,
                    max_prefix_cache_pages,
                    backend,
                    device.clone(),
                ),
                PhysicalStateManager::for_worker(managed_resource_authority, backend, device),
            ),
            None => (
                ManagedKvCacheManager::with_prefix_cache_policy(
                    managed_resource_authority.clone(),
                    managed_prefix_salt,
                    max_prefix_cache_pages,
                ),
                PhysicalStateManager::cpu(managed_resource_authority),
            ),
        };
        // DS4 hierarchical offload is explicit opt-in: a resolved policy only
        // exists when the operator configured a host pool budget.
        managed_kv_cache.set_host_offload_policy(
            crate::engine::cache::offload::HostOffloadPolicy::resolve(&config)?,
        );
        managed_kv_cache.set_prefix_snapshot_rows(config.max_prefix_snapshots);
        managed_kv_cache.set_automatic_prefix_cache_pages(config.max_prefix_cache_pages.is_none());

        Ok(Self {
            config,
            scheduler,
            managed_kv_cache,
            physical_state,
            composite_retained_state: HashMap::new(),
            executor,
            output_processor,
            requests: HashMap::new(),
            execution_profiles: HashMap::new(),
            request_start_times: HashMap::new(),
            request_phase_timings: HashMap::new(),
            execution_trackers: HashMap::new(),
            active_plans: ActivePlans::default(),
            in_flight_dispatches: HashMap::new(),
            pending_runner_recovery: None,
            active_managed_cache: HashMap::new(),
            pending_managed_releases: HashSet::new(),
            cleanup_confirmation_waiters: HashMap::new(),
            active_stream_batches: HashMap::new(),
            stream_sequence_cursors: HashMap::new(),
            incremental_stream_sessions: HashSet::new(),
            pending_terminal_outputs: VecDeque::new(),
            execution_retry_attempts: HashMap::new(),
            plan_begin_attempts: HashMap::new(),
            managed_prepare_attempts: HashMap::new(),
            workspace_progress_epoch: 0,
            workspace_retry_progress: HashMap::new(),
            workspace_waits: HashMap::new(),
            workspace_row_limits: HashMap::new(),
            workspace_pressure: HashMap::new(),
            claim_waiters: HashMap::new(),
            retry_policy: LifecycleRetryPolicy::default(),
            #[cfg(test)]
            restart_after_reset_hook: None,
            #[cfg(test)]
            step_model_calls: Vec::new(),
            next_single_candidate_phase: HashMap::new(),
            initialized: false,
            next_batch_id: 1,
        })
    }

    /// An engine core whose managed state fits against `authority`.
    #[cfg(test)]
    pub(crate) fn new_with_managed_authority_for_test(
        config: EngineCoreConfig,
        authority: Arc<super::ResourceAuthority>,
    ) -> Result<Self> {
        let mut worker_config = WorkerConfig::from(&config);
        worker_config.resource_authority = Some(authority);
        Self::new_with_worker(config, worker_config)
    }

    #[cfg(test)]
    pub(crate) fn new_with_unified_executor(
        config: EngineCoreConfig,
        executor: UnifiedExecutor,
    ) -> Result<Self> {
        info!("Creating engine core");
        Self::new_with_executor(config, executor)
    }

    /// Initialize the engine core.
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing engine core");

        // Initialize executor backend
        self.executor.initialize().await?;

        self.initialized = true;
        info!("Engine core initialized");

        Ok(())
    }

    /// Add a request to the engine.
    pub fn add_request(&mut self, mut request: EngineCoreRequest) -> Result<()> {
        let request_id = request.id.clone();

        if self.requests.contains_key(&request_id) {
            return Err(Error::InvalidInput(format!(
                "Request {} already exists",
                request_id
            )));
        }

        // Chat prompt tokens drive scheduler/KV accounting and must come from
        // exact model preparation, never from the public mutable request fields.
        request.seal_execution_preparation()?;
        request.enforce_chat_context_window(self.config.max_seq_len)?;

        if let Some(model_instance) = request.model_instance_id() {
            if request.v2_state_descriptor().is_some() {
                let runtime = request.v2_state_runtime().ok_or_else(|| {
                    Error::InferenceError(format!(
                        "model instance {} published state ABI v2 before a resolved v2 runtime was installed",
                        model_instance.get()
                    ))
                })?;
                let execution = request.execution_adapter_binding().ok_or_else(|| {
                    Error::InferenceError(
                        "state ABI v2 runtime has no bound execution adapter".to_string(),
                    )
                })?;
                runtime.validate_against(self.managed_kv_cache.worker_backend(), execution)?;
                if let Some(physical) = runtime.managed_kv_runtime() {
                    request.install_managed_cache_runtime(physical.clone())?;
                    // DS1.5: probe the managed prefix index at admission so
                    // the first prefill span can start above an already-
                    // resident shared prefix. The probe self-gates on the
                    // prefix namespace; a probe error plans a zero-start
                    // prefill and leaves the row to surface the underlying
                    // index failure at prepare time.
                    match self
                        .managed_kv_cache
                        .probe_managed_prefix(&physical, &request)
                    {
                        Ok(cursor) => request.set_managed_prefix_cursor(cursor),
                        Err(error) => warn!(
                            request_id = %request_id,
                            error = %error,
                            "managed prefix probe failed; planning a zero-start prefill"
                        ),
                    }
                }
            }
        }

        // Add to scheduler. A public ID remains unavailable while an expired
        // incarnation awaits capability-authoritative executor cleanup.
        if !self.scheduler.add_request(&request) {
            return Err(Error::InvalidInput(format!(
                "Request {} already exists or is awaiting cache cleanup",
                request_id
            )));
        }

        // Track request
        let inserted_at = Instant::now();
        self.request_phase_timings.insert(
            request_id.clone(),
            RequestPhaseTiming::admitted(&request, inserted_at),
        );
        self.requests.insert(request_id.clone(), Arc::new(request));
        self.request_start_times
            .insert(request_id.clone(), inserted_at);

        debug!(
            request_id = %request_id,
            correlation_id = ?self.requests.get(&request_id).and_then(|req| req.correlation_id.as_deref()),
            "Added request to engine core"
        );

        Ok(())
    }

    /// Atomically admit an already-authenticated realtime ASR request into its
    /// dedicated FIFO scheduler state. This route deliberately bypasses normal
    /// one-shot audio preprocessing; audio enters only through operation rows.
    pub(crate) fn add_realtime_asr_session(
        &mut self,
        mut request: EngineCoreRequest,
    ) -> Result<super::SessionKey> {
        let request_id = request.id.clone();
        if request.task_type != super::TaskType::ASR
            || !request.is_realtime_asr_session()
            || request.model_variant.is_none_or(|variant| {
                !matches!(
                    variant.family(),
                    crate::catalog::ModelFamily::Voxtral | crate::catalog::ModelFamily::NemotronAsr
                )
            })
        {
            return Err(Error::InvalidInput(
                "realtime ASR admission requires an ingress-enabled authenticated ASR request"
                    .into(),
            ));
        }
        if self.requests.contains_key(&request_id) {
            return Err(Error::InvalidInput(format!(
                "Request {request_id} already exists"
            )));
        }
        request.seal_execution_preparation()?;
        let model_instance = request.model_instance_id().ok_or_else(|| {
            Error::InvalidInput(
                "realtime ASR admission requires an exact loaded model instance".into(),
            )
        })?;
        let execution = request.execution_adapter_binding().ok_or_else(|| {
            Error::InvalidInput(
                "realtime ASR admission requires a loaded execution adapter binding".into(),
            )
        })?;
        match request.model_variant.expect("validated realtime variant").family() {
            crate::catalog::ModelFamily::Voxtral => crate::models::architectures::voxtral::authenticate_voxtral_realtime_execution_binding(execution)?,
            crate::catalog::ModelFamily::NemotronAsr => crate::models::architectures::nemotron::asr::authenticate_realtime_execution_binding(execution)?,
            _ => unreachable!("validated realtime family"),
        }
        if execution.model_instance_id != model_instance {
            return Err(Error::InvalidInput(
                "realtime ASR execution binding crossed its loaded model instance".into(),
            ));
        }
        let descriptor = request.v2_state_descriptor().ok_or_else(|| {
            Error::InvalidInput("realtime ASR admission requires a state ABI v2 descriptor".into())
        })?;
        let runtime = request.v2_state_runtime().ok_or_else(|| {
            Error::InvalidInput(
                "realtime ASR admission requires a resolved state ABI v2 runtime".into(),
            )
        })?;
        if descriptor != &runtime.descriptor {
            return Err(Error::InvalidInput(
                "realtime ASR state descriptor differs from its resolved runtime".into(),
            ));
        }
        runtime.validate_against(self.managed_kv_cache.worker_backend(), execution)?;
        let physical = runtime.managed_kv_runtime().ok_or_else(|| {
            Error::InvalidInput(
                "realtime ASR admission requires a physical managed state runtime".into(),
            )
        })?;
        if physical.plan().model_instance != model_instance {
            return Err(Error::InvalidInput(
                "realtime ASR managed runtime crossed its loaded model instance".into(),
            ));
        }
        request.install_managed_cache_runtime(physical)?;
        if !self.scheduler.add_realtime_session(&request) {
            return Err(Error::InvalidInput(format!(
                "Request {request_id} already exists or is awaiting cache cleanup"
            )));
        }
        let session = self.get_session_key(&request_id).ok_or_else(|| {
            Error::InferenceError(format!(
                "realtime request {request_id} lost its scheduler session"
            ))
        })?;
        let inserted_at = Instant::now();
        self.request_phase_timings.insert(
            request_id.clone(),
            RequestPhaseTiming::admitted(&request, inserted_at),
        );
        self.requests.insert(request_id.clone(), Arc::new(request));
        self.request_start_times.insert(request_id, inserted_at);
        Ok(session)
    }

    #[cfg(test)]
    pub(crate) async fn enqueue_realtime_asr_push(
        &mut self,
        session: &super::SessionKey,
        samples: Arc<[f32]>,
        sample_rate: u32,
        max_output_steps: usize,
        max_cache_append: usize,
    ) -> Result<(super::RealtimeOperationId, RealtimeAsrOperationWaiter)> {
        let logical_units = u64::try_from(samples.len()).unwrap_or(u64::MAX).max(1);
        self.enqueue_realtime_asr_push_with_cost(
            session,
            samples,
            sample_rate,
            max_output_steps,
            max_cache_append,
            WorkCost::new(logical_units, logical_units, 0),
        )
        .await
    }

    pub(crate) async fn enqueue_realtime_asr_push_with_cost(
        &mut self,
        session: &super::SessionKey,
        samples: Arc<[f32]>,
        sample_rate: u32,
        max_output_steps: usize,
        max_cache_append: usize,
        preparation_cost: WorkCost,
    ) -> Result<(super::RealtimeOperationId, RealtimeAsrOperationWaiter)> {
        if samples.is_empty() || sample_rate == 0 || max_output_steps == 0 {
            return Err(Error::InvalidInput(
                "realtime ASR push requires samples, a sample rate, and an output-step bound"
                    .into(),
            ));
        }
        let request = self
            .requests
            .get(&session.request_id)
            .filter(|_| self.get_session_key(&session.request_id).as_ref() == Some(session))
            .cloned()
            .ok_or_else(|| {
                Error::InvalidInput("realtime ASR session is stale or missing".into())
            })?;
        let (operation_id, input) = self.scheduler.enqueue_realtime_push(
            session,
            samples.len(),
            max_output_steps,
            max_cache_append,
        )?;
        debug_assert_eq!(input.len(), samples.len());
        match request.install_realtime_asr_operation_with_cost(
            operation_id,
            RealtimeAsrOperationPayload::Push {
                samples,
                sample_rate,
            },
            preparation_cost,
        ) {
            Ok(waiter) => Ok((operation_id, waiter)),
            Err(error) => {
                let message = error.to_string();
                let _ = self
                    .terminate_request_session(
                        session,
                        ExecutorOutput::error(session.request_id.clone(), message.clone()),
                        ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                            message.clone(),
                        )),
                        OutcomeProvenance::not_started(),
                    )
                    .await;
                Err(Error::InferenceError(format!(
                    "realtime ASR payload installation failed; session rolled back: {message}"
                )))
            }
        }
    }

    pub(crate) async fn enqueue_realtime_asr_finish_with_cost(
        &mut self,
        session: &super::SessionKey,
        max_output_steps: usize,
        max_cache_append: usize,
        preparation_cost: WorkCost,
    ) -> Result<(super::RealtimeOperationId, RealtimeAsrOperationWaiter)> {
        if max_output_steps == 0 {
            return Err(Error::InvalidInput(
                "realtime ASR finish requires an output-step bound".into(),
            ));
        }
        let request = self
            .requests
            .get(&session.request_id)
            .filter(|_| self.get_session_key(&session.request_id).as_ref() == Some(session))
            .cloned()
            .ok_or_else(|| {
                Error::InvalidInput("realtime ASR session is stale or missing".into())
            })?;
        let operation_id =
            self.scheduler
                .enqueue_realtime_finish(session, max_output_steps, max_cache_append)?;
        match request.install_realtime_asr_operation_with_cost(
            operation_id,
            RealtimeAsrOperationPayload::Finish,
            preparation_cost,
        ) {
            Ok(waiter) => Ok((operation_id, waiter)),
            Err(error) => {
                let message = error.to_string();
                let _ = self
                    .terminate_request_session(
                        session,
                        ExecutorOutput::error(session.request_id.clone(), message.clone()),
                        ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                            message.clone(),
                        )),
                        OutcomeProvenance::not_started(),
                    )
                    .await;
                Err(Error::InferenceError(format!(
                    "realtime ASR payload installation failed; session rolled back: {message}"
                )))
            }
        }
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        !self.pending_terminal_outputs.is_empty()
            || self.scheduler.has_pending_work()
            || self.scheduler.has_due_cleanup(Instant::now())
    }

    /// Whether the session has a plan begun or a dispatch registered that
    /// has not committed or rolled back.
    pub(crate) fn has_in_flight_work(&self, session: &super::SessionKey) -> bool {
        self.active_plans.contains_session(session)
            || self
                .in_flight_dispatches
                .values()
                .any(|dispatch| dispatch.contains_session(session))
    }

    /// See [`super::scheduler::Scheduler::next_wakeup`].
    pub(crate) fn next_wakeup(&self, now: Instant) -> Option<Instant> {
        self.scheduler.next_wakeup(now)
    }

    /// Check if a request exists.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    pub(crate) fn has_pending_terminal_output(&self, request_id: &RequestId) -> bool {
        self.pending_terminal_outputs
            .iter()
            .any(|committed| committed.session.request_id == *request_id)
    }

    /// Get the set of model variants currently referenced by active engine requests.
    pub fn active_model_variants(&self) -> HashSet<ModelVariant> {
        self.requests
            .values()
            .filter_map(|request| request.model_variant)
            .collect()
    }

    /// Get request status.
    pub fn get_request_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        self.scheduler.get_status(request_id)
    }

    /// Return the current scheduler incarnation for a public request ID.
    pub fn get_session_key(&self, request_id: &RequestId) -> Option<super::SessionKey> {
        self.scheduler
            .get_sequence_id(request_id)
            .map(|epoch| super::SessionKey::new(request_id.clone(), epoch))
    }

    #[cfg(test)]
    pub(crate) fn defer_execution_retry_for_test(
        &mut self,
        session: &super::SessionKey,
        not_before: Instant,
    ) -> bool {
        self.scheduler.defer_execution_retry(session, not_before)
    }

    #[cfg(test)]
    pub(crate) fn set_hard_deadline_for_test(
        &mut self,
        request_id: &RequestId,
        deadline: Instant,
    ) -> bool {
        self.scheduler
            .set_hard_deadline_for_test(request_id, deadline)
    }

    /// Get number of pending (waiting) requests.
    pub fn pending_request_count(&self) -> usize {
        self.scheduler.waiting_count()
    }

    /// Get number of running requests.
    pub fn running_request_count(&self) -> usize {
        self.scheduler.running_count()
    }

    /// Exact physical managed-arena state.
    pub fn managed_kv_runtime_snapshot(&self) -> super::ManagedKvRuntimeSnapshot {
        self.managed_kv_cache.runtime_snapshot()
    }

    /// Get configuration.
    pub fn config(&self) -> &EngineCoreConfig {
        &self.config
    }
}

impl Drop for EngineCore {
    fn drop(&mut self) {
        // Note: We can't do async cleanup in drop, so we just log
        if self.initialized {
            debug!("EngineCore dropped while still initialized");
        }
    }
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod tests;

#[cfg(test)]
mod decode_timing_tests {
    use super::RequestPhaseTiming;
    use std::time::{Duration, Instant};

    #[test]
    fn decode_wall_counts_commits_and_includes_between_quantum_waits() {
        let start = Instant::now();
        let mut timing = RequestPhaseTiming::default();
        // A token sampled during prefill belongs to completion usage, not decode.
        timing.record_token_commit(false, 1, start);
        assert_eq!(timing.decode_wall_ms(), None);
        timing.decode_started_at = Some(start + Duration::from_millis(10));
        timing.record_token_commit(true, 2, start + Duration::from_millis(30));
        // Drafts/failed quanta never call record_token_commit; an empty commit
        // must not move the endpoint. Service time excludes a 50ms scheduler wait.
        timing.decode_ms = 40.0;
        timing.record_token_commit(true, 1, start + Duration::from_millis(100));
        timing.record_token_commit(true, 0, start + Duration::from_millis(150));
        assert_eq!(timing.decode_tokens, 3);
        assert_eq!(timing.decode_wall_ms(), Some(90.0));
        assert_eq!(timing.post_first_token_ms(), Some(100.0));
        assert!(timing.decode_wall_ms().unwrap() > timing.decode_ms);
    }

    #[test]
    fn token_commits_report_the_gap_since_the_previous_commit() {
        let start = Instant::now();
        let mut timing = RequestPhaseTiming::default();
        assert_eq!(timing.record_token_commit(false, 1, start), None);
        assert_eq!(
            timing.record_token_commit(true, 4, start + Duration::from_millis(40)),
            Some(Duration::from_millis(40))
        );
        // An empty commit is not a token and leaves the reference point.
        assert_eq!(
            timing.record_token_commit(true, 0, start + Duration::from_millis(60)),
            None
        );
        assert_eq!(
            timing.record_token_commit(true, 1, start + Duration::from_millis(70)),
            Some(Duration::from_millis(30))
        );
    }

    #[test]
    fn entered_decode_without_committed_tokens_has_no_rate_denominator() {
        let timing = RequestPhaseTiming {
            decode_started_at: Some(Instant::now()),
            decode_ms: 12.0,
            ..Default::default()
        };
        assert_eq!(timing.decode_wall_ms(), None);
        assert_eq!(timing.post_first_token_ms(), None);
    }
}

#[cfg(test)]
#[path = "fish_s2_finalization_tests.rs"]
mod fish_s2_finalization_tests;
