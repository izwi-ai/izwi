//! Request scheduler with support for FCFS and priority-based scheduling.
//!
//! The scheduler manages request queues and decides which requests to process
//! in each engine step. It handles:
//! - Waiting queue (new requests awaiting processing)
//! - Running queue (requests currently being processed)
//! - Token budget management
//! - Capability-aware execution admission

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use tracing::debug;

use super::config::EngineCoreConfig;
use super::execution::{
    CacheMode, ExecutionProfile, NativeBatchMode, PrefillMode, RealtimePreparationMode,
    RealtimeStageOutcome, RealtimeSubphase,
};
use super::request::{EngineCoreRequest, RequestStatus, WorkloadClass};
use super::types::{Priority, RequestId, SequenceId, TaskType};
use super::{InputRange, PlanId, RealtimeOperationId, SequencePhase, SessionKey, WorkUnit};
use crate::model::ModelVariant;

pub(super) const MAX_DECODE_ONLY_STEPS_WITH_WAITING_FULL_PREFILL: usize = 8;
pub(super) const MAX_DECODE_ONLY_STEPS_WITH_WAITING_INCREMENTAL_PREFILL: usize = 8;
pub(super) const MAX_REALTIME_ONLY_STEPS_WITH_READY_DECODE: usize = 8;

/// Scheduling policy for the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    FCFS,
    /// Priority-based scheduling (higher priority first)
    Priority,
    /// Weighted workload-class service with priority ordering inside each class.
    #[default]
    WeightedFair,
}

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum tokens per step (token budget)
    pub max_tokens_per_step: usize,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,
    /// Threshold for chunked prefill
    pub chunked_prefill_threshold: usize,
    /// Defer lower-priority decode while a higher-priority request is waiting.
    pub enable_preemption: bool,
    /// Enable VAD-triggered preemption (for audio interruption handling)
    pub enable_vad_preemption: bool,
    /// Enable adaptive, latency-aware batching heuristics.
    pub enable_adaptive_batching: bool,
    /// Minimum token budget for adaptive scheduling.
    pub min_tokens_per_step: usize,
    /// Target time-to-first-token.
    pub target_ttft_ms: f64,
    /// Target decode time per output token.
    pub target_decode_tpot_ms: f64,
    /// Wait time interval used for priority aging.
    pub priority_aging_ms: u64,
    /// Enable deadline-aware scheduling boosts.
    pub enable_deadline_scheduling: bool,
    /// Soft SLA budget for critical requests.
    pub critical_sla_ms: u64,
    /// Soft SLA budget for high-priority requests.
    pub high_sla_ms: u64,
    /// Soft SLA budget for normal-priority requests.
    pub normal_sla_ms: u64,
    /// Soft SLA budget for low-priority requests.
    pub low_sla_ms: u64,
    /// Enable thermal/power-aware adaptive throttling.
    pub enable_power_adaptive: bool,
    /// External thermal pressure hint in [0, 1].
    pub thermal_pressure_hint: f64,
    /// Power-save mode for low-power local edge devices.
    pub power_save_mode: bool,
    /// Enable multi-token decode quanta when latency pressure is low.
    pub enable_decode_quanta: bool,
    /// Maximum decode tokens per request in one scheduler step.
    pub max_decode_tokens_per_request: usize,
}

/// VAD preemption event - signals that user started speaking.
#[derive(Debug, Clone)]
pub struct VadPreemptionEvent {
    /// Timestamp of the VAD detection
    pub timestamp: Instant,
    /// Speech probability from VAD
    pub speech_probability: f32,
    /// Request IDs that should be preempted (currently generating requests)
    pub requests_to_preempt: Vec<RequestId>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_step: 384,
            policy: SchedulingPolicy::WeightedFair,
            enable_chunked_prefill: false,
            chunked_prefill_threshold: 192,
            enable_preemption: false,
            enable_vad_preemption: true,
            enable_adaptive_batching: false,
            min_tokens_per_step: 96,
            target_ttft_ms: 250.0,
            target_decode_tpot_ms: 40.0,
            priority_aging_ms: 1_000,
            enable_deadline_scheduling: true,
            critical_sla_ms: 200,
            high_sla_ms: 400,
            normal_sla_ms: 1_000,
            low_sla_ms: 2_500,
            enable_power_adaptive: false,
            thermal_pressure_hint: 0.0,
            power_save_mode: false,
            enable_decode_quanta: false,
            max_decode_tokens_per_request: 2,
        }
    }
}

impl From<&EngineCoreConfig> for SchedulerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        Self {
            max_batch_size: config.max_batch_size,
            max_tokens_per_step: config.max_tokens_per_step,
            policy: config.scheduling_policy,
            enable_chunked_prefill: config.enable_chunked_prefill,
            chunked_prefill_threshold: config.chunked_prefill_threshold,
            enable_preemption: config.enable_preemption,
            enable_vad_preemption: true, // Default to enabled for audio apps
            enable_adaptive_batching: config.enable_adaptive_batching,
            min_tokens_per_step: config.min_tokens_per_step,
            target_ttft_ms: config.target_ttft_ms,
            target_decode_tpot_ms: config.target_decode_tpot_ms,
            priority_aging_ms: config.priority_aging_ms,
            enable_deadline_scheduling: config.enable_deadline_scheduling,
            critical_sla_ms: config.critical_sla_ms,
            high_sla_ms: config.high_sla_ms,
            normal_sla_ms: config.normal_sla_ms,
            low_sla_ms: config.low_sla_ms,
            enable_power_adaptive: config.enable_power_adaptive,
            thermal_pressure_hint: config.thermal_pressure_hint,
            power_save_mode: config.power_save_mode,
            enable_decode_quanta: config.enable_decode_quanta,
            max_decode_tokens_per_request: config.max_decode_tokens_per_request,
        }
    }
}

/// A request wrapper for priority queue ordering.
#[derive(Debug, Clone)]
struct PriorityRequest {
    request_id: RequestId,
    priority: Priority,
    workload_class: WorkloadClass,
    arrival_time: Instant,
}

impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier arrival time
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => self
                .workload_class
                .adaptive_score_boost()
                .partial_cmp(&other.workload_class.adaptive_score_boost())
                .unwrap_or(Ordering::Equal)
                .then_with(|| other.arrival_time.cmp(&self.arrival_time)), // Earlier is greater
            ord => ord,
        }
    }
}

/// Result of scheduling a step.
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Requests scheduled for decode (already running)
    pub decode_requests: Vec<ScheduledRequest>,
    /// Requests scheduled for prefill (new requests)
    pub prefill_requests: Vec<ScheduledRequest>,
    /// Requests rejected before execution because their caller deadline elapsed.
    pub expired_requests: Vec<ExpiredRequest>,
    /// Total tokens to process this step
    pub total_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpiredRequest {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalReleaseCause {
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BeginTerminalRelease {
    Started { confirmation_required: bool },
    AlreadyPending { confirmation_required: bool },
    StaleOrMissing,
}

#[derive(Debug, Clone)]
struct PendingRelease {
    session: SessionKey,
    cause: TerminalReleaseCause,
    confirmation_required: bool,
    cleanup_confirmed: bool,
    terminal_delivered: bool,
    cleanup_attempts: u32,
    retry_at: Instant,
}

impl ExpiredRequest {
    pub fn session_key(&self) -> SessionKey {
        SessionKey::new(self.request_id.clone(), self.sequence_id)
    }
}

impl ScheduleResult {
    pub fn empty() -> Self {
        Self {
            decode_requests: Vec::new(),
            prefill_requests: Vec::new(),
            expired_requests: Vec::new(),
            total_tokens: 0,
        }
    }

    /// Check if there's any work to do
    pub fn has_work(&self) -> bool {
        self.has_execution_work() || !self.expired_requests.is_empty()
    }

    pub fn has_execution_work(&self) -> bool {
        !self.decode_requests.is_empty() || !self.prefill_requests.is_empty()
    }

    /// Get all scheduled request IDs
    pub fn all_request_ids(&self) -> Vec<RequestId> {
        let mut ids: Vec<_> = self
            .decode_requests
            .iter()
            .chain(self.prefill_requests.iter())
            .map(|r| r.request_id.clone())
            .collect();
        ids.dedup();
        ids
    }
}

/// A request that has been scheduled for processing.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Monotonic identity for this exact scheduling decision.
    pub plan_id: PlanId,
    /// Request ID
    pub request_id: RequestId,
    /// Sequence ID
    pub sequence_id: SequenceId,
    /// Number of tokens to process this step
    pub num_tokens: usize,
    /// Whether this is a prefill (first pass) or decode (continuation)
    pub is_prefill: bool,
    /// Number of tokens already computed (for chunked prefill)
    pub num_computed_tokens: usize,
    /// Authoritative bounded unit of work for the executor.
    pub work: WorkUnit,
}

impl ScheduledRequest {
    /// Stable identity for the public request ID's current scheduler incarnation.
    pub fn session_key(&self) -> SessionKey {
        SessionKey::new(self.request_id.clone(), self.sequence_id)
    }
}

/// Runtime telemetry used by adaptive scheduling.
#[derive(Debug, Clone)]
pub struct SchedulerTelemetry {
    /// Exponential moving average of time-to-first-token.
    pub avg_ttft_ms: f64,
    /// Exponential moving average of decode time per generated token.
    pub avg_decode_tpot_ms: f64,
    /// Exponential moving average of waiting queue age.
    pub avg_queue_age_ms: f64,
    /// Current adaptive token budget.
    pub dynamic_tokens_per_step: usize,
    /// Current adaptive prefill chunk threshold.
    pub dynamic_prefill_chunk_threshold: usize,
    /// Exponential moving average of prefill chunk backoff pressure.
    pub prefill_backoff_ewma: f64,
}

impl SchedulerTelemetry {
    fn new(default_budget: usize) -> Self {
        let default_chunk = default_budget.max(32);
        Self {
            avg_ttft_ms: 0.0,
            avg_decode_tpot_ms: 0.0,
            avg_queue_age_ms: 0.0,
            dynamic_tokens_per_step: default_budget.max(1),
            dynamic_prefill_chunk_threshold: default_chunk,
            prefill_backoff_ewma: 0.0,
        }
    }

    fn update_ewma(current: &mut f64, sample: f64, alpha: f64) {
        if sample <= 0.0 {
            return;
        }
        if *current <= 0.0 {
            *current = sample;
        } else {
            *current = (*current * (1.0 - alpha)) + (sample * alpha);
        }
    }
}

/// Request scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    /// Waiting queue (FCFS mode)
    waiting_fcfs: VecDeque<RequestId>,
    /// Waiting queue (Priority mode)
    waiting_priority: BinaryHeap<PriorityRequest>,
    /// Membership index for waiting requests (enables O(1) removals and lazy queue cleanup).
    waiting_members: HashSet<RequestId>,
    /// Running requests (by request ID)
    running: HashMap<RequestId, RunningRequest>,
    /// Request metadata
    requests: HashMap<RequestId, RequestMetadata>,
    /// FIFO ingress for engine-owned realtime sessions. These sessions never
    /// enter the ordinary prompt/decode queues.
    realtime_sessions: HashMap<RequestId, RealtimeSchedulerState>,
    /// Terminal sessions whose executor-owned cache has not yet been proven
    /// released, or whose terminal event has not yet been delivered. Logical
    /// blocks and the public request ID remain fenced until both conditions are
    /// satisfied.
    pending_releases: HashMap<RequestId, PendingRelease>,
    /// Next sequence ID
    next_sequence_id: SequenceId,
    /// Next execution plan identity.
    next_plan_id: PlanId,
    /// Monotonic fair-service clock for realtime session rotation.
    next_realtime_service_clock: u64,
    /// Monotonic scheduling cycle used for one-cycle preemption resume fences.
    /// Adaptive scheduling telemetry.
    telemetry: SchedulerTelemetry,
    /// Completed scheduling quanta by workload class for weighted service.
    class_service: HashMap<WorkloadClass, u64>,
    /// Decode-only transactions observed while an indivisible full prefill was
    /// waiting. This bounds starvation without pretending that a Full adapter
    /// can safely resume a scheduler-authored chunk.
    decode_only_steps_with_waiting_full_prefill: usize,
    /// Decode-bearing transactions observed while a resumable prefill had no
    /// scheduler slot. Unlike a full prefill, one incremental span can share
    /// the transaction once this bounded wait expires.
    decode_only_steps_with_waiting_incremental_prefill: usize,
    /// Transactions in which realtime rows consumed every decode slot while an
    /// ordinary decode row was ready. One ordinary slot is reserved once this
    /// bounded debt is reached.
    realtime_only_steps_with_ready_decode: usize,
}

#[derive(Debug, Clone)]
struct RealtimePendingOperation {
    id: RealtimeOperationId,
    external: RealtimeExternalOperation,
    phase: RealtimeSubphase,
    remaining_output_steps: usize,
    committed_cache_append: usize,
}

#[derive(Debug, Clone)]
enum RealtimeExternalOperation {
    Push {
        input: InputRange,
        max_cache_append: usize,
    },
    Finish {
        max_cache_append: usize,
    },
}

impl RealtimePendingOperation {
    fn max_cache_append(&self) -> usize {
        match self.external {
            RealtimeExternalOperation::Push {
                max_cache_append, ..
            }
            | RealtimeExternalOperation::Finish { max_cache_append } => max_cache_append,
        }
    }

    fn work(&self, committed_input_samples: usize, committed_decode_steps: usize) -> WorkUnit {
        match self.phase {
            RealtimeSubphase::Preparation => {
                let (mode, input) = match self.external {
                    RealtimeExternalOperation::Push { input, .. } => {
                        (RealtimePreparationMode::Push, input)
                    }
                    RealtimeExternalOperation::Finish { .. } => (
                        RealtimePreparationMode::Finish,
                        InputRange {
                            start: committed_input_samples,
                            end: committed_input_samples,
                        },
                    ),
                };
                WorkUnit::RealtimePreparation {
                    operation_id: self.id,
                    mode,
                    input,
                    max_output_steps: self.remaining_output_steps,
                    max_cache_append: self.max_cache_append(),
                    retained_state_input: InputRange {
                        start: self.id.get().saturating_sub(1) as usize,
                        end: self.id.get() as usize,
                    },
                    auxiliary_state: None,
                }
            }
            RealtimeSubphase::PromptPrefill { cache_append } => WorkUnit::RealtimePromptPrefill {
                operation_id: self.id,
                max_output_steps: self.remaining_output_steps,
                cache_append,
            },
            RealtimeSubphase::DecodeContinuation => WorkUnit::RealtimeDecodeContinuation {
                operation_id: self.id,
                max_output_steps: self.remaining_output_steps,
                max_cache_append: 1,
                retained_state_input: InputRange {
                    start: committed_decode_steps,
                    end: committed_decode_steps.saturating_add(1),
                },
                auxiliary_state: None,
            },
            RealtimeSubphase::Completion => WorkUnit::RealtimeCompletion {
                operation_id: self.id,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct RealtimeSchedulerState {
    sequence_id: SequenceId,
    committed_input_samples: usize,
    committed_decode_steps: usize,
    enqueued_input_samples: usize,
    next_operation_id: u64,
    pending: VecDeque<RealtimePendingOperation>,
    in_flight: Option<RealtimeInFlight>,
    last_service_clock: u64,
    finish_enqueued: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealtimeInFlight {
    plan_id: PlanId,
    operation_id: RealtimeOperationId,
    subphase: RealtimeSubphase,
}

/// A scheduler-authenticated realtime transition whose publication contains no
/// remaining fallible work. The scheduler must not be mutated between prepare
/// and publish; the engine core owns that serialized boundary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedRealtimeStageOutcome {
    session_epoch: SequenceId,
    outcome: RealtimeStageOutcome,
    remaining_output_steps: usize,
    committed_cache_append: usize,
    committed_input_samples: usize,
    committed_decode_steps: usize,
    final_stage: bool,
}

/// Metadata for a request in the scheduler.
#[derive(Debug, Clone)]
struct RequestMetadata {
    request_id: RequestId,
    sequence_id: SequenceId,
    task_type: TaskType,
    model_variant: Option<ModelVariant>,
    priority: Priority,
    workload_class: WorkloadClass,
    arrival_time: Instant,
    deadline_at: Instant,
    hard_deadline: Option<Instant>,
    total_prompt_tokens: usize,
    max_tokens: usize,
    cache_policy: RequestCachePolicy,
    retry_not_before: Option<Instant>,
}

#[derive(Debug, Clone)]
struct RequestCachePolicy {
    mode: Option<CacheMode>,
    prefill: PrefillMode,
    decode_batch: NativeBatchMode,
    recompute_safe: bool,
    cache_release_safe: bool,
    preferred_decode_tokens: usize,
}

impl Default for RequestCachePolicy {
    fn default() -> Self {
        Self {
            mode: None,
            // Fail closed until the loaded adapter explicitly proves resumable
            // incremental-prefill safe points.
            prefill: PrefillMode::Full,
            decode_batch: NativeBatchMode::None,
            recompute_safe: false,
            cache_release_safe: false,
            preferred_decode_tokens: 1,
        }
    }
}

impl RequestCachePolicy {}

/// State for a running request.
#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: RequestId,
    sequence_id: SequenceId,
    /// Number of tokens processed so far (prompt + generated)
    num_tokens_processed: usize,
    /// Number of tokens generated so far
    num_tokens_generated: usize,
    /// Whether prefill is complete
    prefill_complete: bool,
    /// Whether a prefill quantum has been scheduled but not yet committed.
    prefill_in_flight: bool,
    /// A committed decoder quantum requested a distinct terminal model stage.
    finalize_pending: bool,
    /// The exact finalization quantum has been planned but not yet committed.
    finalize_in_flight: bool,
    /// Scheduler-visible incremental-prefill quanta committed for this request.
    incremental_prefill_quanta_committed: usize,
    /// Priority of this request
    priority: Priority,
    /// Coarse latency/throughput class for this request.
    workload_class: WorkloadClass,
    /// Whether this request has produced its first output token.
    first_token_emitted: bool,
    /// Whether this request is temporarily paused due to preemption.
    paused: bool,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        let mut telemetry = SchedulerTelemetry::new(config.max_tokens_per_step);
        telemetry.dynamic_prefill_chunk_threshold = config.chunked_prefill_threshold.max(32);
        Self {
            config,
            waiting_fcfs: VecDeque::new(),
            waiting_priority: BinaryHeap::new(),
            waiting_members: HashSet::new(),
            running: HashMap::new(),
            requests: HashMap::new(),
            realtime_sessions: HashMap::new(),
            pending_releases: HashMap::new(),
            next_sequence_id: 0,
            next_plan_id: 1,
            next_realtime_service_clock: 1,
            telemetry,
            class_service: HashMap::new(),
            decode_only_steps_with_waiting_full_prefill: 0,
            decode_only_steps_with_waiting_incremental_prefill: 0,
            realtime_only_steps_with_ready_decode: 0,
        }
    }

    fn ensure_realtime_service_clock_capacity(&mut self, quanta: usize) {
        let required = u64::try_from(quanta).unwrap_or(u64::MAX);
        if self
            .next_realtime_service_clock
            .checked_add(required)
            .is_some()
        {
            return;
        }
        let mut order = self
            .realtime_sessions
            .iter()
            .map(|(request_id, state)| {
                (
                    state.last_service_clock,
                    state.sequence_id,
                    request_id.clone(),
                )
            })
            .collect::<Vec<_>>();
        order.sort_unstable();
        for (rank, (_, _, request_id)) in order.into_iter().enumerate() {
            if let Some(state) = self.realtime_sessions.get_mut(&request_id) {
                state.last_service_clock = u64::try_from(rank).unwrap_or(u64::MAX - 1);
            }
        }
        self.next_realtime_service_clock = u64::try_from(self.realtime_sessions.len())
            .unwrap_or(u64::MAX - 1)
            .saturating_add(1);
    }

    /// Add a request to the waiting queue.
    pub fn add_request(&mut self, request: &EngineCoreRequest) -> bool {
        if self.requests.contains_key(&request.id)
            || self.pending_releases.contains_key(&request.id)
        {
            return false;
        }

        let sequence_id = self.next_sequence_id;
        self.next_sequence_id += 1;
        let arrival_time = request.arrival_time;

        let max_tokens = match (request.task_type, request.params.max_tokens) {
            (TaskType::TTS, 0) => usize::MAX,
            (_, 0) => 2048,
            (_, value) => value,
        };
        let deadline_at = request.deadline.unwrap_or_else(|| {
            arrival_time + self.deadline_for_request(request.priority, request.workload_class)
        });

        let metadata = RequestMetadata {
            request_id: request.id.clone(),
            sequence_id,
            task_type: request.task_type,
            model_variant: request.model_variant,
            priority: request.priority,
            workload_class: request.workload_class,
            arrival_time,
            deadline_at,
            hard_deadline: request.deadline,
            total_prompt_tokens: request.num_prompt_tokens(),
            // TTS uses max_tokens=0 to indicate "auto". Keep scheduler decode budget
            // effectively unbounded so model-level stop criteria can terminate naturally.
            // For other task types, guard against zero-budget stalls if upstream
            // validation is ever bypassed.
            max_tokens,
            cache_policy: RequestCachePolicy::default(),
            retry_not_before: None,
        };

        self.requests.insert(request.id.clone(), metadata);

        self.enqueue_waiting_request(request.id.clone());

        debug!(
            "Added request {} to waiting queue (sequence_id={}, prompt_tokens={})",
            request.id,
            sequence_id,
            request.num_prompt_tokens()
        );
        true
    }

    pub(crate) fn add_realtime_session(&mut self, request: &EngineCoreRequest) -> bool {
        if !request.is_realtime_asr_session()
            || self.requests.contains_key(&request.id)
            || self.pending_releases.contains_key(&request.id)
        {
            return false;
        }
        let sequence_id = self.next_sequence_id;
        self.next_sequence_id = self.next_sequence_id.saturating_add(1);
        let arrival_time = request.arrival_time;
        let deadline_at = request.deadline.unwrap_or_else(|| {
            arrival_time + self.deadline_for_request(request.priority, request.workload_class)
        });
        self.requests.insert(
            request.id.clone(),
            RequestMetadata {
                request_id: request.id.clone(),
                sequence_id,
                task_type: request.task_type,
                model_variant: request.model_variant,
                priority: request.priority,
                workload_class: request.workload_class,
                arrival_time,
                deadline_at,
                hard_deadline: request.deadline,
                total_prompt_tokens: 0,
                max_tokens: usize::MAX,
                cache_policy: RequestCachePolicy::default(),
                retry_not_before: None,
            },
        );
        self.running.insert(
            request.id.clone(),
            RunningRequest {
                request_id: request.id.clone(),
                sequence_id,
                num_tokens_processed: 0,
                num_tokens_generated: 0,
                prefill_complete: true,
                prefill_in_flight: false,
                finalize_pending: false,
                finalize_in_flight: false,
                incremental_prefill_quanta_committed: 0,
                priority: request.priority,
                workload_class: request.workload_class,
                first_token_emitted: false,
                paused: false,
            },
        );
        let initial_service_clock = self.next_realtime_service_clock.saturating_sub(1);
        self.realtime_sessions.insert(
            request.id.clone(),
            RealtimeSchedulerState {
                sequence_id,
                committed_input_samples: 0,
                committed_decode_steps: 0,
                enqueued_input_samples: 0,
                next_operation_id: 1,
                pending: VecDeque::new(),
                in_flight: None,
                last_service_clock: initial_service_clock,
                finish_enqueued: false,
            },
        );
        true
    }

    pub(crate) fn enqueue_realtime_push(
        &mut self,
        session: &SessionKey,
        sample_count: usize,
        max_output_steps: usize,
        max_cache_append: usize,
    ) -> crate::error::Result<(RealtimeOperationId, InputRange)> {
        if sample_count == 0 || max_cache_append == 0 {
            return Err(crate::error::Error::InvalidInput(
                "realtime push requires non-empty samples and a cache append ceiling".into(),
            ));
        }
        let state = self
            .realtime_sessions
            .get_mut(&session.request_id)
            .ok_or_else(|| {
                crate::error::Error::InvalidInput("realtime session is not active".into())
            })?;
        if state.sequence_id != session.epoch || state.finish_enqueued {
            return Err(crate::error::Error::InvalidInput(
                "realtime push crossed its session fence or followed finish".into(),
            ));
        }
        let end = state
            .enqueued_input_samples
            .checked_add(sample_count)
            .ok_or_else(|| {
                crate::error::Error::InvalidInput("realtime sample cursor overflow".into())
            })?;
        let input = InputRange::new(state.enqueued_input_samples, end)?;
        let id = RealtimeOperationId::new(state.next_operation_id);
        state.next_operation_id = state.next_operation_id.checked_add(1).ok_or_else(|| {
            crate::error::Error::InferenceError("realtime operation identity overflow".into())
        })?;
        state.enqueued_input_samples = end;
        state.pending.push_back(RealtimePendingOperation {
            id,
            external: RealtimeExternalOperation::Push {
                input,
                max_cache_append,
            },
            phase: RealtimeSubphase::Preparation,
            remaining_output_steps: max_output_steps,
            committed_cache_append: 0,
        });
        Ok((id, input))
    }

    pub(crate) fn enqueue_realtime_finish(
        &mut self,
        session: &SessionKey,
        max_output_steps: usize,
        max_cache_append: usize,
    ) -> crate::error::Result<RealtimeOperationId> {
        if max_cache_append == 0 {
            return Err(crate::error::Error::InvalidInput(
                "realtime finish requires a cache append ceiling".into(),
            ));
        }
        let state = self
            .realtime_sessions
            .get_mut(&session.request_id)
            .ok_or_else(|| {
                crate::error::Error::InvalidInput("realtime session is not active".into())
            })?;
        if state.sequence_id != session.epoch || state.finish_enqueued {
            return Err(crate::error::Error::InvalidInput(
                "realtime finish crossed its session fence or was already queued".into(),
            ));
        }
        let id = RealtimeOperationId::new(state.next_operation_id);
        state.next_operation_id = state.next_operation_id.checked_add(1).ok_or_else(|| {
            crate::error::Error::InferenceError("realtime operation identity overflow".into())
        })?;
        state.finish_enqueued = true;
        state.pending.push_back(RealtimePendingOperation {
            id,
            external: RealtimeExternalOperation::Finish { max_cache_append },
            phase: RealtimeSubphase::Preparation,
            remaining_output_steps: max_output_steps,
            committed_cache_append: 0,
        });
        Ok(id)
    }

    /// Install the loaded executor's cache contract for one exact scheduler
    /// incarnation. Stale profiles cannot mutate a reused public request ID.
    pub fn update_execution_profile(
        &mut self,
        session: &SessionKey,
        profile: &ExecutionProfile,
    ) -> bool {
        let Some(metadata) = self.requests.get_mut(&session.request_id) else {
            return false;
        };
        if metadata.sequence_id != session.epoch {
            return false;
        }
        metadata.cache_policy = RequestCachePolicy {
            mode: Some(profile.cache_mode),
            // Only an explicit incremental contract is scheduler-resumable.
            // `None` is not a weaker form of incremental prefill; for a
            // sequence with prompt tokens it must fail closed as indivisible.
            prefill: if profile.prefill == PrefillMode::Incremental {
                PrefillMode::Incremental
            } else {
                PrefillMode::Full
            },
            decode_batch: profile.decode_batch,
            recompute_safe: profile.recompute_safe,
            cache_release_safe: profile.cache_release_safe,
            preferred_decode_tokens: profile.preferred_decode_tokens.max(1),
        };
        true
    }

    /// Schedule requests for the next step.
    pub fn schedule(&mut self) -> ScheduleResult {
        let mut result = ScheduleResult::empty();
        result.expired_requests = self.expire_deadlines();
        let scheduling_now = Instant::now();
        let mut remaining_batch = self.config.max_batch_size;
        self.refresh_queue_age_sample();
        self.update_dynamic_budget();

        let total_budget = self.current_token_budget();
        let waiting_full_prefill = self.has_eligible_full_prefill(scheduling_now);
        let force_full_prefill_service = waiting_full_prefill
            && self.decode_only_steps_with_waiting_full_prefill
                >= MAX_DECODE_ONLY_STEPS_WITH_WAITING_FULL_PREFILL;
        let waiting_incremental_prefill = self.has_eligible_incremental_prefill(scheduling_now);
        let force_incremental_prefill_service = !force_full_prefill_service
            && waiting_incremental_prefill
            && self.decode_only_steps_with_waiting_incremental_prefill
                >= MAX_DECODE_ONLY_STEPS_WITH_WAITING_INCREMENTAL_PREFILL;
        let latency_sensitive_waiting = self.has_latency_sensitive_waiting();
        let throughput_waiting_only =
            !latency_sensitive_waiting && self.has_throughput_or_background_waiting();
        let mut decode_budget = total_budget;
        let mut reserved_prefill_budget = 0;
        if self.config.enable_adaptive_batching && total_budget > 0 {
            let target_ttft_ms = self.config.target_ttft_ms;
            let mut prefill_share: f64 = if self.telemetry.avg_ttft_ms > target_ttft_ms {
                0.55
            } else if self.telemetry.avg_ttft_ms > target_ttft_ms * 0.8 {
                0.40
            } else {
                0.25
            };
            if latency_sensitive_waiting {
                prefill_share = prefill_share.max(0.55);
            } else if throughput_waiting_only {
                prefill_share = if self.running.is_empty() {
                    prefill_share.max(0.50)
                } else {
                    prefill_share.min(0.30)
                };
            }
            reserved_prefill_budget = ((total_budget as f64) * prefill_share) as usize;
            reserved_prefill_budget = reserved_prefill_budget.clamp(1, total_budget);
            decode_budget = total_budget.saturating_sub(reserved_prefill_budget);
        } else if self.config.policy == SchedulingPolicy::WeightedFair
            && !self.waiting_members.is_empty()
            && total_budget > 0
        {
            reserved_prefill_budget = total_budget.clamp(1, 32);
            decode_budget = total_budget.saturating_sub(reserved_prefill_budget);
        }
        if force_full_prefill_service {
            // Full-prefill adapters expose no resumable scheduler safe point.
            // Give one such prompt an isolated transaction after a bounded
            // number of decode-only commits instead of mixing it behind live
            // decode output or pretending it can honor a token chunk.
            decode_budget = 0;
            reserved_prefill_budget = total_budget;
        } else if force_incremental_prefill_service {
            // Preserve both a token budget and a physical row for one
            // resumable span. The remaining rows/tokens can continue decode.
            let forced_prefill_budget = if total_budget > 1 {
                self.config
                    .chunked_prefill_threshold
                    .max(1)
                    .min(total_budget - 1)
            } else {
                total_budget
            };
            reserved_prefill_budget = reserved_prefill_budget.max(forced_prefill_budget);
            decode_budget = total_budget.saturating_sub(reserved_prefill_budget);
        }
        let mut remaining_decode_budget = decode_budget;

        let ordinary_decode_ready = self.running.iter().any(|(request_id, running)| {
            if !running.prefill_complete || self.realtime_sessions.contains_key(request_id) {
                return false;
            }
            self.requests.get(request_id).is_some_and(|metadata| {
                metadata
                    .retry_not_before
                    .is_none_or(|not_before| not_before <= scheduling_now)
                    && metadata.max_tokens > running.num_tokens_generated
            })
        });
        let reserve_ordinary_decode = ordinary_decode_ready
            && self.realtime_only_steps_with_ready_decode
                >= MAX_REALTIME_ONLY_STEPS_WITH_READY_DECODE;

        // Realtime operations are already admitted, session-fenced quanta.
        // Schedule at most one FIFO head per session and keep their source
        // sample clock independent from token-prefill/decode accounting.
        self.ensure_realtime_service_clock_capacity(self.realtime_sessions.len());
        let mut realtime_candidates = self
            .realtime_sessions
            .iter()
            .filter_map(|(request_id, state)| {
                (state.in_flight.is_none() && !state.pending.is_empty()).then_some((
                    state.last_service_clock,
                    state.sequence_id,
                    request_id.clone(),
                ))
            })
            .collect::<Vec<_>>();
        realtime_candidates.sort_unstable();
        for (_, _, request_id) in realtime_candidates {
            if remaining_batch == 0 || remaining_decode_budget == 0 {
                break;
            }
            if reserve_ordinary_decode && (remaining_batch <= 1 || remaining_decode_budget <= 1) {
                break;
            }
            let Some(state) = self.realtime_sessions.get_mut(&request_id) else {
                continue;
            };
            let Some(operation) = state.pending.front().cloned() else {
                continue;
            };
            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            state.in_flight = Some(RealtimeInFlight {
                plan_id,
                operation_id: operation.id,
                subphase: operation.phase,
            });
            state.last_service_clock = self.next_realtime_service_clock;
            self.next_realtime_service_clock = self
                .next_realtime_service_clock
                .checked_add(1)
                .expect("realtime service clock capacity was reserved");
            let num_computed_tokens = state.committed_input_samples;
            let work = operation.work(state.committed_input_samples, state.committed_decode_steps);
            result.decode_requests.push(ScheduledRequest {
                plan_id,
                request_id,
                sequence_id: state.sequence_id,
                num_tokens: 1,
                is_prefill: false,
                num_computed_tokens,
                work,
            });
            remaining_decode_budget = remaining_decode_budget.saturating_sub(1);
            remaining_batch -= 1;
            result.total_tokens = result.total_tokens.saturating_add(1);
        }

        // Finalization is a distinct, cache-free model stage. Schedule it
        // ahead of ordinary decode so a completed acoustic row cannot consume
        // another decoder quantum or be stranded by its output-token ceiling.
        let mut finalize_candidates = self
            .running
            .iter()
            .filter(|(id, running)| {
                running.finalize_pending
                    && !running.finalize_in_flight
                    && !self.realtime_sessions.contains_key(*id)
            })
            .filter_map(|(id, running)| {
                let metadata = self.requests.get(id)?;
                if metadata
                    .retry_not_before
                    .is_some_and(|not_before| not_before > scheduling_now)
                {
                    return None;
                }
                Some((
                    id.clone(),
                    running.sequence_id,
                    running.priority,
                    running.num_tokens_processed,
                ))
            })
            .collect::<Vec<_>>();
        finalize_candidates
            .sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.1.cmp(&right.1)));
        for (request_id, sequence_id, _priority, num_computed_tokens) in finalize_candidates {
            if remaining_batch == 0 || remaining_decode_budget == 0 {
                break;
            }
            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            if let Some(running) = self.running.get_mut(&request_id) {
                running.finalize_in_flight = true;
            }
            result.decode_requests.push(ScheduledRequest {
                plan_id,
                request_id,
                sequence_id,
                num_tokens: 1,
                is_prefill: false,
                num_computed_tokens,
                work: WorkUnit::SequenceFinalize {
                    max_output_steps: 1,
                },
            });
            remaining_decode_budget = remaining_decode_budget.saturating_sub(1);
            remaining_batch -= 1;
            result.total_tokens = result.total_tokens.saturating_add(1);
        }

        // Phase 1: schedule decode requests (already running prefill-complete requests).
        let mut decode_candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| r.prefill_complete && !r.finalize_pending)
            .filter(|(id, _)| !self.realtime_sessions.contains_key(*id))
            .filter_map(|(id, r)| {
                let metadata = self.requests.get(id)?;
                if metadata
                    .retry_not_before
                    .is_some_and(|not_before| not_before > scheduling_now)
                {
                    return None;
                }
                let remaining_decode_tokens =
                    metadata.max_tokens.saturating_sub(r.num_tokens_generated);
                if remaining_decode_tokens == 0 {
                    return None;
                }
                let overdue_ms = self.request_overdue_ms(metadata);
                Some((
                    id.clone(),
                    r.sequence_id,
                    r.priority,
                    r.num_tokens_processed,
                    remaining_decode_tokens,
                    r.num_tokens_generated,
                    r.paused,
                    metadata.workload_class,
                    overdue_ms,
                    metadata.cache_policy.decode_batch == NativeBatchMode::Continuous,
                    metadata.cache_policy.preferred_decode_tokens,
                ))
            })
            .collect();

        if self.config.enable_adaptive_batching
            && self.config.policy != SchedulingPolicy::WeightedFair
        {
            // Favor overdue requests first, then requests close to completion.
            decode_candidates.sort_by(|a, b| {
                b.8.partial_cmp(&a.8)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.6.cmp(&a.6))
                    .then_with(|| {
                        b.7.adaptive_score_boost()
                            .partial_cmp(&a.7.adaptive_score_boost())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| a.4.cmp(&b.4))
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.5.cmp(&b.5))
            });
        } else if self.config.policy == SchedulingPolicy::WeightedFair {
            let mut simulated_service = self.class_service.clone();
            let mut fair_order = Vec::with_capacity(decode_candidates.len());
            while !decode_candidates.is_empty() {
                let next_index = (0..decode_candidates.len())
                    .min_by(|left, right| {
                        let a = &decode_candidates[*left];
                        let b = &decode_candidates[*right];
                        Self::compare_class_service_with(&simulated_service, a.7, b.7)
                            .then_with(|| b.2.cmp(&a.2))
                            .then_with(|| {
                                b.8.partial_cmp(&a.8).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .then_with(|| a.1.cmp(&b.1))
                    })
                    .unwrap_or(0);
                let candidate = decode_candidates.remove(next_index);
                let next_service = simulated_service
                    .get(&candidate.7)
                    .copied()
                    .unwrap_or_default()
                    .saturating_add(1);
                simulated_service.insert(candidate.7, next_service);
                fair_order.push(candidate);
            }
            decode_candidates = fair_order;
        } else {
            decode_candidates.sort_by(|a, b| match self.config.policy {
                SchedulingPolicy::FCFS => a.1.cmp(&b.1),
                SchedulingPolicy::Priority => b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)),
                SchedulingPolicy::WeightedFair => Ordering::Equal,
            });
        }
        let has_decode_demand = !decode_candidates.is_empty();
        let highest_waiting_priority = self
            .waiting_members
            .iter()
            .filter_map(|request_id| self.requests.get(request_id).map(|m| m.priority))
            .max();
        let effective_prefill_chunk_threshold =
            self.effective_prefill_chunk_threshold(has_decode_demand);

        for (
            request_id,
            sequence_id,
            priority,
            num_computed,
            remaining_decode_tokens,
            _generated_tokens,
            _paused,
            workload_class,
            overdue_ms,
            continuous_decode,
            preferred_decode_tokens,
        ) in decode_candidates
        {
            if self.config.enable_preemption
                && highest_waiting_priority.is_some_and(|waiting| waiting > priority)
            {
                continue;
            }
            if remaining_batch == 0 || remaining_decode_budget == 0 {
                break;
            }
            if force_incremental_prefill_service && remaining_batch <= 1 {
                break;
            }
            if self.config.policy == SchedulingPolicy::WeightedFair
                && !self.waiting_members.is_empty()
                && remaining_batch <= 1
            {
                break;
            }

            let num_tokens = self.decode_token_quanta(
                remaining_decode_budget,
                remaining_decode_tokens,
                self.waiting_count() > 0,
                overdue_ms,
                workload_class,
                preferred_decode_tokens,
                continuous_decode,
            );
            if num_tokens == 0 {
                continue;
            }

            if let Some(running) = self.running.get_mut(&request_id) {
                running.paused = false;
            }

            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            result.decode_requests.push(ScheduledRequest {
                plan_id,
                request_id: request_id.clone(),
                sequence_id,
                num_tokens,
                is_prefill: false,
                num_computed_tokens: num_computed,
                work: WorkUnit::SequenceStep {
                    phase: SequencePhase::Decode,
                    input: InputRange {
                        start: num_computed,
                        end: num_computed.saturating_add(num_tokens),
                    },
                    max_output_steps: num_tokens,
                    auxiliary_state: None,
                },
            });

            remaining_decode_budget = remaining_decode_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            result.total_tokens += num_tokens;
            self.record_class_service(workload_class, num_tokens);
        }

        let served_realtime = result.decode_requests.iter().any(|row| {
            matches!(
                &row.work,
                WorkUnit::RealtimePush { .. }
                    | WorkUnit::RealtimeFinish { .. }
                    | WorkUnit::RealtimePreparation { .. }
                    | WorkUnit::RealtimePromptPrefill { .. }
                    | WorkUnit::RealtimeDecodeContinuation { .. }
                    | WorkUnit::RealtimeCompletion { .. }
            )
        });
        let served_ordinary_decode = result.decode_requests.iter().any(|row| {
            matches!(
                &row.work,
                WorkUnit::SequenceStep {
                    phase: SequencePhase::Decode,
                    ..
                }
            )
        });
        if !ordinary_decode_ready || served_ordinary_decode {
            self.realtime_only_steps_with_ready_decode = 0;
        } else if served_realtime {
            self.realtime_only_steps_with_ready_decode =
                self.realtime_only_steps_with_ready_decode.saturating_add(1);
        }

        // Phase 2: schedule prefill requests.
        let mut remaining_prefill_budget =
            if force_full_prefill_service || force_incremental_prefill_service {
                reserved_prefill_budget
            } else if self.config.enable_adaptive_batching
                || self.config.policy == SchedulingPolicy::WeightedFair
            {
                reserved_prefill_budget.saturating_add(remaining_decode_budget)
            } else {
                remaining_decode_budget
            };
        let prefill_admission_cap = usize::MAX;
        let mut prefill_admissions = 0usize;
        let mut scheduling_full_prefill_batch = false;

        // Phase 2a: continue incomplete prefills before admitting new waiting requests.
        // An in-flight quantum is excluded until update_after_step commits its
        // progress, preventing duplicate plans when schedule is polled again.
        let mut incomplete_prefill_candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| !r.prefill_complete && !r.prefill_in_flight)
            .filter_map(|(id, r)| {
                let metadata = self.requests.get(id)?;
                if metadata
                    .retry_not_before
                    .is_some_and(|not_before| not_before > scheduling_now)
                {
                    return None;
                }
                Some((
                    id.clone(),
                    r.priority,
                    r.sequence_id,
                    r.num_tokens_processed,
                ))
            })
            .collect();
        incomplete_prefill_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.3.cmp(&b.3)));

        for (request_id, _priority, sequence_id, num_computed) in incomplete_prefill_candidates {
            if remaining_batch == 0 || remaining_prefill_budget == 0 {
                break;
            }

            let metadata = match self.requests.get(&request_id) {
                Some(m) => m.clone(),
                None => continue,
            };

            let remaining_prompt = metadata.total_prompt_tokens.saturating_sub(num_computed);
            if remaining_prompt == 0 {
                if let Some(running) = self.running.get_mut(&request_id) {
                    running.prefill_complete = true;
                    running.prefill_in_flight = false;
                    running.paused = false;
                }
                continue;
            }

            let full_prefill = metadata.cache_policy.prefill == PrefillMode::Full;
            if force_full_prefill_service && !full_prefill {
                continue;
            }
            if force_incremental_prefill_service && full_prefill {
                continue;
            }
            if !full_prefill && scheduling_full_prefill_batch {
                continue;
            }
            let mut target_tokens = remaining_prompt;
            if !full_prefill
                && self.config.enable_chunked_prefill
                && target_tokens > effective_prefill_chunk_threshold
            {
                target_tokens = effective_prefill_chunk_threshold;
            }
            if full_prefill {
                if !result.decode_requests.is_empty()
                    || (!result.prefill_requests.is_empty() && !scheduling_full_prefill_batch)
                {
                    continue;
                }
                scheduling_full_prefill_batch = true;
            } else {
                target_tokens = target_tokens.min(remaining_prefill_budget);
            }
            if target_tokens == 0 {
                continue;
            }

            let original_target_tokens = target_tokens;
            let num_tokens = target_tokens;
            self.record_prefill_backoff(original_target_tokens, num_tokens);

            if let Some(running) = self.running.get_mut(&request_id) {
                running.prefill_in_flight = true;
                running.paused = false;
            }

            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            result.prefill_requests.push(ScheduledRequest {
                plan_id,
                request_id: request_id.clone(),
                sequence_id,
                num_tokens,
                is_prefill: true,
                num_computed_tokens: num_computed,
                work: WorkUnit::SequenceStep {
                    phase: SequencePhase::Prefill,
                    input: InputRange {
                        start: num_computed,
                        end: num_computed.saturating_add(num_tokens),
                    },
                    max_output_steps: num_tokens.max(1),
                    auxiliary_state: None,
                },
            });

            if full_prefill {
                remaining_batch -= 1;
            } else {
                remaining_prefill_budget = remaining_prefill_budget.saturating_sub(num_tokens);
                remaining_batch -= 1;
            }
            result.total_tokens += num_tokens;
            self.record_class_service(metadata.workload_class, num_tokens);
        }

        let mut deferred_waiting = Vec::new();
        let max_waiting_attempts = self.waiting_count();
        let mut waiting_attempts = 0usize;

        while remaining_batch > 0
            && remaining_prefill_budget > 0
            && prefill_admissions < prefill_admission_cap
            && waiting_attempts < max_waiting_attempts
        {
            let request_id = match self.take_next_waiting_request() {
                Some(id) => id,
                None => break,
            };
            waiting_attempts = waiting_attempts.saturating_add(1);

            let metadata = match self.requests.get(&request_id) {
                Some(m) => m.clone(),
                None => continue,
            };

            // Check if already running (shouldn't happen, but safety check)
            if self.running.contains_key(&request_id) {
                continue;
            }

            // Calculate tokens for this prefill.
            let full_prefill = metadata.cache_policy.prefill == PrefillMode::Full;
            if force_full_prefill_service && !full_prefill {
                deferred_waiting.push(request_id);
                continue;
            }
            if force_incremental_prefill_service && full_prefill {
                deferred_waiting.push(request_id);
                continue;
            }
            if !full_prefill && scheduling_full_prefill_batch {
                deferred_waiting.push(request_id);
                continue;
            }
            let mut target_tokens = metadata.total_prompt_tokens;

            // Apply chunked prefill if enabled and prompt is long
            if !full_prefill
                && self.config.enable_chunked_prefill
                && target_tokens > effective_prefill_chunk_threshold
            {
                target_tokens = effective_prefill_chunk_threshold;
            }

            // Full-prefill executors cannot honor a scheduler chunk. Permit one
            // indivisible prompt only in an otherwise empty transaction.
            if full_prefill {
                if !result.decode_requests.is_empty()
                    || (!result.prefill_requests.is_empty() && !scheduling_full_prefill_batch)
                {
                    deferred_waiting.push(request_id);
                    continue;
                }
                scheduling_full_prefill_batch = true;
            } else {
                target_tokens = target_tokens.min(remaining_prefill_budget);
            }
            if target_tokens == 0 {
                break;
            }

            let original_target_tokens = target_tokens;
            let num_tokens = target_tokens;
            self.record_prefill_backoff(original_target_tokens, num_tokens);

            // Create running state
            let running = RunningRequest {
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens_processed: 0,
                num_tokens_generated: 0,
                // Scheduling is not a commit. A failed/retryable prefill must
                // remain a prefill until update_after_step confirms that the
                // complete prompt was actually consumed.
                prefill_complete: false,
                prefill_in_flight: true,
                finalize_pending: false,
                finalize_in_flight: false,
                incremental_prefill_quanta_committed: 0,
                priority: metadata.priority,
                workload_class: metadata.workload_class,
                first_token_emitted: false,
                paused: false,
            };

            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            result.prefill_requests.push(ScheduledRequest {
                plan_id,
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens,
                is_prefill: true,
                num_computed_tokens: 0,
                work: WorkUnit::SequenceStep {
                    phase: SequencePhase::Prefill,
                    input: InputRange {
                        start: 0,
                        end: num_tokens,
                    },
                    max_output_steps: num_tokens.max(1),
                    auxiliary_state: None,
                },
            });

            self.running.insert(request_id.clone(), running);

            if full_prefill {
                remaining_batch -= 1;
            } else {
                remaining_prefill_budget = remaining_prefill_budget.saturating_sub(num_tokens);
                remaining_batch -= 1;
            }
            prefill_admissions = prefill_admissions.saturating_add(1);
            result.total_tokens += num_tokens;
            self.record_class_service(metadata.workload_class, num_tokens);
        }

        for request_id in deferred_waiting {
            self.enqueue_waiting_request(request_id);
        }

        result
    }

    /// Update request state after a step.
    pub fn update_after_step(
        &mut self,
        request_id: &RequestId,
        tokens_processed: usize,
        tokens_generated: usize,
        step_time_ms: f64,
    ) {
        if tokens_processed > 0 || tokens_generated > 0 {
            if let Some(metadata) = self.requests.get_mut(request_id) {
                metadata.retry_not_before = None;
            }
        }
        let incremental_prefill = self
            .requests
            .get(request_id)
            .is_some_and(|metadata| metadata.cache_policy.prefill == PrefillMode::Incremental);
        if let Some(running) = self.running.get_mut(request_id) {
            let committed_incremental_prefill =
                incremental_prefill && !running.prefill_complete && tokens_processed > 0;
            running.prefill_in_flight = false;
            running.paused = false;
            running.num_tokens_processed += tokens_processed;
            running.num_tokens_generated += tokens_generated;
            if committed_incremental_prefill {
                running.incremental_prefill_quanta_committed = running
                    .incremental_prefill_quanta_committed
                    .saturating_add(1);
                crate::engine::metrics::record_engine_incremental_prefill_commit(
                    tokens_processed,
                    running.incremental_prefill_quanta_committed == 2,
                );
            }

            // Check if prefill is now complete
            if let Some(metadata) = self.requests.get(request_id) {
                if running.num_tokens_processed >= metadata.total_prompt_tokens {
                    running.prefill_complete = true;
                }

                if !running.first_token_emitted && tokens_generated > 0 {
                    running.first_token_emitted = true;
                    let ttft_ms = metadata.arrival_time.elapsed().as_secs_f64() * 1000.0;
                    SchedulerTelemetry::update_ewma(&mut self.telemetry.avg_ttft_ms, ttft_ms, 0.20);
                }
            }

            if tokens_generated > 0 && step_time_ms > 0.0 {
                let tpot_ms = step_time_ms / tokens_generated as f64;
                SchedulerTelemetry::update_ewma(
                    &mut self.telemetry.avg_decode_tpot_ms,
                    tpot_ms,
                    0.15,
                );
            }
        }
        self.update_dynamic_budget();
    }

    /// Move one committed retained sequence onto its load-sealed finalization
    /// stage. This is deliberately independent of generated-token capacity:
    /// the final decoder frame may consume the request's last output token.
    pub fn request_sequence_finalize(
        &mut self,
        request_id: &RequestId,
    ) -> crate::error::Result<()> {
        let running = self.running.get_mut(request_id).ok_or_else(|| {
            crate::error::Error::InferenceError(
                "cannot finalize a request that is not running".into(),
            )
        })?;
        if !running.prefill_complete || running.prefill_in_flight {
            return Err(crate::error::Error::InferenceError(
                "cannot finalize a sequence before committed prefill".into(),
            ));
        }
        running.finalize_pending = true;
        running.finalize_in_flight = false;
        running.paused = false;
        Ok(())
    }

    pub(crate) fn prepare_realtime_stage_outcome(
        &self,
        session: &SessionKey,
        outcome: RealtimeStageOutcome,
    ) -> crate::error::Result<PreparedRealtimeStageOutcome> {
        let state = self
            .realtime_sessions
            .get(&session.request_id)
            .ok_or_else(|| {
                crate::error::Error::InferenceError("committed realtime session is missing".into())
            })?;
        let head = state.pending.front().ok_or_else(|| {
            crate::error::Error::InferenceError(
                "committed realtime operation queue is empty".into(),
            )
        })?;
        if state.sequence_id != session.epoch
            || state.in_flight
                != Some(RealtimeInFlight {
                    plan_id: outcome.plan_id,
                    operation_id: outcome.operation_id,
                    subphase: outcome.completed,
                })
            || head.id != outcome.operation_id
            || head.phase != outcome.completed
        {
            return Err(crate::error::Error::InferenceError(
                "realtime operation commit crossed its FIFO session fence".into(),
            ));
        }
        let transition_valid = matches!(
            (head.phase, outcome.next),
            (
                RealtimeSubphase::Preparation,
                Some(
                    RealtimeSubphase::PromptPrefill { .. }
                        | RealtimeSubphase::DecodeContinuation
                        | RealtimeSubphase::Completion
                )
            ) | (
                RealtimeSubphase::PromptPrefill { .. },
                Some(RealtimeSubphase::DecodeContinuation | RealtimeSubphase::Completion)
            ) | (
                RealtimeSubphase::DecodeContinuation,
                Some(RealtimeSubphase::DecodeContinuation | RealtimeSubphase::Completion)
            ) | (RealtimeSubphase::Completion, None)
        ) || (outcome.next.is_none()
            && !matches!(head.phase, RealtimeSubphase::Completion));
        if !transition_valid {
            return Err(crate::error::Error::InferenceError(
                "realtime stage outcome requested an invalid subphase transition".into(),
            ));
        }
        if matches!(
            head.phase,
            RealtimeSubphase::Preparation | RealtimeSubphase::Completion
        ) && (outcome.output_steps != 0 || outcome.cache_append != 0)
        {
            return Err(crate::error::Error::InferenceError(
                "realtime preparation/completion cannot report decoder progress".into(),
            ));
        }
        match head.phase {
            RealtimeSubphase::PromptPrefill { cache_append }
                if cache_append == 0
                    || outcome.cache_append != cache_append
                    || outcome.output_steps != 1 =>
            {
                return Err(crate::error::Error::InferenceError(
                    "realtime prompt prefill did not append its exact KV span".into(),
                ));
            }
            RealtimeSubphase::DecodeContinuation
                if outcome.cache_append != 1 || outcome.output_steps != 1 =>
            {
                return Err(crate::error::Error::InferenceError(
                    "realtime decode continuation must append exactly one KV token".into(),
                ));
            }
            _ => {}
        }
        if outcome.output_steps > head.remaining_output_steps
            || outcome.output_steps
                > usize::from(matches!(
                    head.phase,
                    RealtimeSubphase::PromptPrefill { .. } | RealtimeSubphase::DecodeContinuation
                ))
        {
            return Err(crate::error::Error::InferenceError(
                "realtime stage exceeded its remaining output-step bound".into(),
            ));
        }
        let committed_cache_append = head
            .committed_cache_append
            .checked_add(outcome.cache_append)
            .ok_or_else(|| {
                crate::error::Error::InferenceError("realtime KV append clock overflow".into())
            })?;
        if committed_cache_append > head.max_cache_append() {
            return Err(crate::error::Error::InferenceError(
                "realtime stage exceeded its operation KV append ceiling".into(),
            ));
        }
        if let Some(RealtimeSubphase::PromptPrefill { cache_append }) = outcome.next {
            if cache_append == 0
                || committed_cache_append
                    .checked_add(cache_append)
                    .is_none_or(|total| total > head.max_cache_append())
            {
                return Err(crate::error::Error::InferenceError(
                    "realtime prompt prefill exceeds its operation KV append ceiling".into(),
                ));
            }
        }
        let remaining_output_steps = head.remaining_output_steps - outcome.output_steps;
        if outcome.next.is_some()
            && remaining_output_steps == 0
            && !matches!(outcome.next, Some(RealtimeSubphase::Completion))
        {
            return Err(crate::error::Error::InferenceError(
                "realtime stage cannot continue after exhausting its output-step bound".into(),
            ));
        }

        let final_stage = outcome.next.is_none();
        if !final_stage && outcome.input_consumed != 0 {
            return Err(crate::error::Error::InferenceError(
                "intermediate realtime stages cannot commit source input".into(),
            ));
        }
        if final_stage {
            match head.external {
                RealtimeExternalOperation::Push { input, .. } => {
                    if input.start != state.committed_input_samples
                        || outcome.input_consumed != input.len()
                    {
                        return Err(crate::error::Error::InferenceError(
                            "realtime push completion did not consume its exact sample interval"
                                .into(),
                        ));
                    }
                }
                RealtimeExternalOperation::Finish { .. } if outcome.input_consumed != 0 => {
                    return Err(crate::error::Error::InferenceError(
                        "realtime finish cannot consume source samples".into(),
                    ));
                }
                RealtimeExternalOperation::Finish { .. } => {}
            }
        }
        let committed_input_samples = if final_stage {
            match head.external {
                RealtimeExternalOperation::Push { input, .. } => input.end,
                RealtimeExternalOperation::Finish { .. } => state.committed_input_samples,
            }
        } else {
            state.committed_input_samples
        };
        let committed_decode_steps = state
            .committed_decode_steps
            .checked_add(outcome.output_steps)
            .ok_or_else(|| {
                crate::error::Error::InferenceError(
                    "realtime retained decode clock overflow".into(),
                )
            })?;
        Ok(PreparedRealtimeStageOutcome {
            session_epoch: session.epoch,
            outcome,
            remaining_output_steps,
            committed_cache_append,
            committed_input_samples,
            committed_decode_steps,
            final_stage,
        })
    }

    /// Publishes a previously prepared transition. All validation and checked
    /// arithmetic happened in `prepare_realtime_stage_outcome`; violating the
    /// serialized prepare/publish boundary is an engine invariant failure.
    pub(crate) fn publish_prepared_realtime_stage_outcome(
        &mut self,
        session: &SessionKey,
        prepared: PreparedRealtimeStageOutcome,
    ) {
        assert_eq!(session.epoch, prepared.session_epoch);
        let state = self
            .realtime_sessions
            .get_mut(&session.request_id)
            .expect("prepared realtime session remains present until publication");
        assert_eq!(state.sequence_id, prepared.session_epoch);
        assert_eq!(
            state.in_flight,
            Some(RealtimeInFlight {
                plan_id: prepared.outcome.plan_id,
                operation_id: prepared.outcome.operation_id,
                subphase: prepared.outcome.completed,
            })
        );
        let head = state
            .pending
            .front_mut()
            .expect("prepared realtime FIFO head remains present until publication");
        assert_eq!(head.id, prepared.outcome.operation_id);
        assert_eq!(head.phase, prepared.outcome.completed);
        head.remaining_output_steps = prepared.remaining_output_steps;
        head.committed_cache_append = prepared.committed_cache_append;
        state.committed_decode_steps = prepared.committed_decode_steps;
        if prepared.final_stage {
            state.committed_input_samples = prepared.committed_input_samples;
            state.pending.pop_front();
        } else {
            head.phase = prepared
                .outcome
                .next
                .expect("prepared non-final transition has a next phase");
        }
        state.in_flight = None;
    }

    pub(crate) fn commit_realtime_stage_outcome(
        &mut self,
        session: &SessionKey,
        outcome: RealtimeStageOutcome,
    ) -> crate::error::Result<()> {
        let prepared = self.prepare_realtime_stage_outcome(session, outcome)?;
        self.publish_prepared_realtime_stage_outcome(session, prepared);
        Ok(())
    }

    pub(crate) fn release_realtime_operation_for_retry(
        &mut self,
        session: &SessionKey,
        plan_id: PlanId,
        operation_id: RealtimeOperationId,
        subphase: RealtimeSubphase,
    ) -> bool {
        let Some(state) = self.realtime_sessions.get_mut(&session.request_id) else {
            return false;
        };
        if state.sequence_id != session.epoch
            || state.in_flight
                != Some(RealtimeInFlight {
                    plan_id,
                    operation_id,
                    subphase,
                })
            || state.pending.front().map(|operation| operation.id) != Some(operation_id)
            || state.pending.front().map(|operation| operation.phase) != Some(subphase)
        {
            return false;
        }
        state.in_flight = None;
        true
    }

    /// Advance starvation debt once per successfully committed physical
    /// transaction. Scheduler polling, retry deferral, and rolled-back model
    /// work must not mutate this service clock.
    pub(crate) fn record_committed_batch_service(
        &mut self,
        committed_requests: &[RequestId],
        decode_transaction: bool,
    ) {
        if committed_requests.is_empty() {
            return;
        }
        let served_full_prefill = !decode_transaction
            && committed_requests.iter().any(|request_id| {
                self.requests
                    .get(request_id)
                    .is_some_and(|metadata| metadata.cache_policy.prefill == PrefillMode::Full)
            });
        let served_incremental_prefill = !decode_transaction
            && committed_requests.iter().any(|request_id| {
                self.requests.get(request_id).is_some_and(|metadata| {
                    metadata.cache_policy.prefill == PrefillMode::Incremental
                })
            });
        if served_full_prefill || !self.has_pending_full_prefill() {
            self.decode_only_steps_with_waiting_full_prefill = 0;
        } else if decode_transaction {
            self.decode_only_steps_with_waiting_full_prefill = self
                .decode_only_steps_with_waiting_full_prefill
                .saturating_add(1);
        }
        if served_incremental_prefill || !self.has_pending_incremental_prefill() {
            self.decode_only_steps_with_waiting_incremental_prefill = 0;
        } else if decode_transaction {
            self.decode_only_steps_with_waiting_incremental_prefill = self
                .decode_only_steps_with_waiting_incremental_prefill
                .saturating_add(1);
        }
    }

    /// Release an uncommitted prefill quantum so the exact request session can
    /// retry it without changing committed scheduler progress.
    pub fn release_execution_quantum_for_retry(&mut self, session: &SessionKey) -> bool {
        let Some(metadata) = self.requests.get(&session.request_id) else {
            return false;
        };
        if metadata.sequence_id != session.epoch {
            return false;
        }

        let Some(running) = self.running.get_mut(&session.request_id) else {
            return false;
        };
        if running.sequence_id != session.epoch {
            return false;
        }

        running.prefill_in_flight = false;
        running.finalize_in_flight = false;
        running.prefill_complete = running.num_tokens_processed >= metadata.total_prompt_tokens;
        true
    }

    /// Remove service charged when a scheduled quantum never reached
    /// execution. The engine calls this exactly once while removing the
    /// corresponding active plan, so capacity retries cannot lose weighted
    /// fair-share merely by being selected.
    pub(crate) fn refund_unexecuted_service(
        &mut self,
        session: &SessionKey,
        scheduled_tokens: usize,
    ) -> bool {
        if self.config.policy != SchedulingPolicy::WeightedFair {
            return true;
        }
        let Some(metadata) = self.requests.get(&session.request_id) else {
            return false;
        };
        if metadata.sequence_id != session.epoch {
            return false;
        }
        let service = self
            .class_service
            .entry(metadata.workload_class)
            .or_default();
        *service = service.saturating_sub(scheduled_tokens.max(1) as u64);
        true
    }

    /// Defer the next execution quantum for an exact session. This clears an
    /// uncommitted prefill marker without changing committed progress.
    pub(crate) fn defer_execution_retry(
        &mut self,
        session: &SessionKey,
        not_before: Instant,
    ) -> bool {
        let Some(metadata) = self.requests.get_mut(&session.request_id) else {
            return false;
        };
        if metadata.sequence_id != session.epoch {
            return false;
        }
        let Some(running) = self.running.get_mut(&session.request_id) else {
            return false;
        };
        if running.sequence_id != session.epoch {
            return false;
        }
        running.prefill_in_flight = false;
        metadata.retry_not_before = Some(not_before);
        true
    }

    /// Restart an exact running request incarnation from prefill after an
    /// executor reports that its session must be recomputed.
    pub fn restart_request_for_recompute(&mut self, session: &SessionKey) -> bool {
        let Some(metadata) = self.requests.get(&session.request_id) else {
            return false;
        };
        if metadata.sequence_id != session.epoch {
            return false;
        }

        let Some(running) = self.running.get_mut(&session.request_id) else {
            return false;
        };
        if running.sequence_id != session.epoch {
            return false;
        }

        running.num_tokens_processed = 0;
        running.incremental_prefill_quanta_committed = 0;
        running.num_tokens_generated = 0;
        running.prefill_complete = false;
        running.prefill_in_flight = false;
        running.finalize_pending = false;
        running.finalize_in_flight = false;
        running.first_token_emitted = false;
        running.paused = true;
        true
    }

    /// Select the cheapest lower-priority capacity owner that can be replayed
    /// without changing client-visible output. Once a token has been emitted,
    /// generated-token history would be required for lossless recomputation;
    /// those sessions therefore remain protected.
    pub(crate) fn capacity_preemption_candidate(
        &self,
        candidates: impl IntoIterator<Item = SessionKey>,
        blocked_priority: Priority,
    ) -> Option<SessionKey> {
        if !self.config.enable_preemption {
            return None;
        }
        candidates
            .into_iter()
            .filter_map(|session| {
                let metadata = self.requests.get(&session.request_id)?;
                let running = self.running.get(&session.request_id)?;
                (metadata.sequence_id == session.epoch
                    && running.sequence_id == session.epoch
                    && metadata.priority < blocked_priority
                    && metadata.cache_policy.recompute_safe
                    && metadata.cache_policy.cache_release_safe
                    && !running.first_token_emitted)
                    .then_some((metadata.priority, running.num_tokens_processed, session))
            })
            .min_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| left.2.request_id.cmp(&right.2.request_id))
            })
            .map(|(_, _, session)| session)
    }

    /// Mark a request as finished and remove it.
    pub fn finish_request(&mut self, request_id: &RequestId) {
        self.remove_from_waiting(request_id);
        self.running.remove(request_id);
        self.requests.remove(request_id);
        self.realtime_sessions.remove(request_id);
    }

    /// Move an exact active session into terminal quarantine. The request ID
    /// remains fenced until capability-authoritative cleanup is confirmed and
    /// its terminal event has been delivered.
    pub(crate) fn begin_terminal_release(
        &mut self,
        session: &SessionKey,
        cause: TerminalReleaseCause,
    ) -> BeginTerminalRelease {
        if let Some(pending) = self.pending_releases.get(&session.request_id) {
            return if pending.session == *session {
                BeginTerminalRelease::AlreadyPending {
                    confirmation_required: pending.confirmation_required,
                }
            } else {
                BeginTerminalRelease::StaleOrMissing
            };
        }

        let Some(metadata) = self.requests.get(&session.request_id).cloned() else {
            return BeginTerminalRelease::StaleOrMissing;
        };
        if metadata.sequence_id != session.epoch {
            return BeginTerminalRelease::StaleOrMissing;
        }

        self.remove_from_waiting(&session.request_id);
        let running = self.running.remove(&session.request_id);
        if running
            .as_ref()
            .is_some_and(|running| running.sequence_id != session.epoch)
        {
            if let Some(running) = running {
                self.running.insert(session.request_id.clone(), running);
            }
            return BeginTerminalRelease::StaleOrMissing;
        }
        self.requests.remove(&session.request_id);
        self.realtime_sessions.remove(&session.request_id);

        let confirmation_required =
            running.is_some() && metadata.cache_policy.mode != Some(CacheMode::None);
        self.pending_releases.insert(
            session.request_id.clone(),
            PendingRelease {
                session: session.clone(),
                cause,
                confirmation_required,
                cleanup_confirmed: false,
                terminal_delivered: false,
                cleanup_attempts: 0,
                retry_at: Instant::now(),
            },
        );
        BeginTerminalRelease::Started {
            confirmation_required,
        }
    }

    /// Confirm capability-authoritative cleanup for an exact terminal session.
    pub(crate) fn confirm_session_release(&mut self, session: &SessionKey) -> bool {
        let remove = {
            let Some(pending) = self.pending_releases.get_mut(&session.request_id) else {
                return false;
            };
            if pending.session != *session {
                return false;
            }
            if !pending.cleanup_confirmed {
                pending.cleanup_confirmed = true;
                debug!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    cause = ?pending.cause,
                    "Confirmed exact-session terminal cache release"
                );
            }
            pending.terminal_delivered
        };
        if remove {
            self.pending_releases.remove(&session.request_id);
        }
        true
    }

    pub(crate) fn mark_terminal_delivered(&mut self, session: &SessionKey) -> bool {
        let remove = {
            let Some(pending) = self.pending_releases.get_mut(&session.request_id) else {
                return false;
            };
            if pending.session != *session {
                return false;
            }
            pending.terminal_delivered = true;
            pending.cleanup_confirmed
        };
        if remove {
            self.pending_releases.remove(&session.request_id);
        }
        true
    }

    pub(crate) fn pending_release_confirmation_required(
        &self,
        session: &SessionKey,
    ) -> Option<bool> {
        self.pending_releases
            .get(&session.request_id)
            .filter(|pending| pending.session == *session)
            .map(|pending| pending.confirmation_required)
    }

    pub(crate) fn due_cleanup_sessions(&self, now: Instant, limit: usize) -> Vec<SessionKey> {
        let mut pending: Vec<_> = self
            .pending_releases
            .values()
            .filter(|pending| {
                pending.confirmation_required
                    && !pending.cleanup_confirmed
                    && pending.retry_at <= now
            })
            .collect();
        pending.sort_by_key(|pending| (pending.retry_at, pending.session.epoch));
        pending
            .into_iter()
            .take(limit)
            .map(|pending| pending.session.clone())
            .collect()
    }

    pub(crate) fn record_cleanup_retry(
        &mut self,
        session: &SessionKey,
        retry_at: Instant,
    ) -> Option<u32> {
        let pending = self.pending_releases.get_mut(&session.request_id)?;
        if pending.session != *session || pending.cleanup_confirmed {
            return None;
        }
        pending.cleanup_attempts = pending.cleanup_attempts.saturating_add(1);
        pending.retry_at = retry_at;
        Some(pending.cleanup_attempts)
    }

    pub(crate) fn pending_cleanup_attempts(&self, session: &SessionKey) -> Option<u32> {
        self.pending_releases
            .get(&session.request_id)
            .filter(|pending| pending.session == *session)
            .map(|pending| pending.cleanup_attempts)
    }

    pub(crate) fn has_due_cleanup(&self, now: Instant) -> bool {
        self.pending_releases.values().any(|pending| {
            pending.confirmation_required && !pending.cleanup_confirmed && pending.retry_at <= now
        })
    }

    pub(crate) fn force_release_all_after_executor_shutdown(&mut self) {
        self.pending_releases.clear();
    }

    /// Compatibility helper retained for scheduler-level tests. Core runtime
    /// paths use the generalized exact-session release protocol above.
    pub fn confirm_expired_session_cleanup(&mut self, session: &SessionKey) -> bool {
        let confirmed = self.confirm_session_release(session);
        if confirmed {
            self.mark_terminal_delivered(session);
        }
        confirmed
    }

    /// Compatibility helper retained for existing deadline tests.
    pub(crate) fn pending_expired_cleanup_sessions(&self) -> Vec<SessionKey> {
        let mut sessions: Vec<_> = self
            .pending_releases
            .values()
            .filter(|pending| !pending.cleanup_confirmed)
            .map(|pending| pending.session.clone())
            .collect();
        sessions.sort_by_key(|session| session.epoch);
        sessions
    }

    /// Abort a request.
    pub fn abort_request(&mut self, request_id: &RequestId) -> bool {
        self.remove_from_waiting(request_id);

        // Remove from running
        if self.running.remove(request_id).is_some() {
            self.requests.remove(request_id);
            return true;
        }

        self.requests.remove(request_id);
        false
    }

    /// Check if a request exists in the scheduler.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Get request status.
    pub fn get_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        if self.running.contains_key(request_id) {
            Some(RequestStatus::Running)
        } else if self.requests.contains_key(request_id) {
            Some(RequestStatus::Waiting)
        } else {
            None
        }
    }

    /// Get number of waiting requests.
    pub fn waiting_count(&self) -> usize {
        self.waiting_members.len()
    }

    /// Get number of running requests.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        self.waiting_count() > 0
            || self
                .running
                .keys()
                .any(|request_id| !self.realtime_sessions.contains_key(request_id))
            || self
                .realtime_sessions
                .values()
                .any(|state| state.in_flight.is_some() || !state.pending.is_empty())
    }

    /// Get running request info.
    pub fn get_running_info(&self, request_id: &RequestId) -> Option<(usize, usize)> {
        self.running
            .get(request_id)
            .map(|r| (r.num_tokens_processed, r.num_tokens_generated))
    }

    /// Get sequence ID for a request.
    pub fn get_sequence_id(&self, request_id: &RequestId) -> Option<SequenceId> {
        self.requests.get(request_id).map(|m| m.sequence_id)
    }

    /// Adaptive scheduler telemetry.
    pub fn telemetry(&self) -> SchedulerTelemetry {
        self.telemetry.clone()
    }

    #[cfg(test)]
    pub(crate) fn set_hard_deadline_for_test(
        &mut self,
        request_id: &RequestId,
        deadline: Instant,
    ) -> bool {
        let Some(metadata) = self.requests.get_mut(request_id) else {
            return false;
        };
        metadata.hard_deadline = Some(deadline);
        true
    }

    // Helper methods

    fn select_next_waiting_request(&self) -> Option<RequestId> {
        if self.waiting_members.is_empty() {
            return None;
        }
        if self.config.policy == SchedulingPolicy::WeightedFair {
            return self.select_weighted_waiting_request();
        }
        if !self.config.enable_adaptive_batching {
            return match self.config.policy {
                SchedulingPolicy::FCFS => self.waiting_fcfs.front().cloned(),
                SchedulingPolicy::Priority => {
                    self.waiting_priority.peek().map(|r| r.request_id.clone())
                }
                SchedulingPolicy::WeightedFair => unreachable!("weighted policy handled above"),
            };
        }

        self.waiting_members.iter().cloned().max_by(|a, b| {
            let score_a = self.adaptive_waiting_score(a);
            let score_b = self.adaptive_waiting_score(b);
            score_a.total_cmp(&score_b).then_with(|| {
                let sequence_a = self
                    .requests
                    .get(a)
                    .map(|m| m.sequence_id)
                    .unwrap_or(u64::MAX);
                let sequence_b = self
                    .requests
                    .get(b)
                    .map(|m| m.sequence_id)
                    .unwrap_or(u64::MAX);
                sequence_b.cmp(&sequence_a)
            })
        })
    }

    fn take_next_waiting_request(&mut self) -> Option<RequestId> {
        if self.waiting_members.is_empty() {
            return None;
        }

        if self.config.policy == SchedulingPolicy::WeightedFair {
            let next = self.select_weighted_waiting_request()?;
            self.remove_from_waiting(&next);
            return Some(next);
        }

        if self.config.enable_adaptive_batching {
            let next = self.select_next_waiting_request()?;
            self.remove_from_waiting(&next);
            return Some(next);
        }

        match self.config.policy {
            SchedulingPolicy::FCFS => {
                while let Some(candidate) = self.waiting_fcfs.pop_front() {
                    if self.waiting_members.remove(&candidate) {
                        return Some(candidate);
                    }
                }
                None
            }
            SchedulingPolicy::Priority => {
                while let Some(candidate) = self.waiting_priority.pop() {
                    if self.waiting_members.remove(&candidate.request_id) {
                        return Some(candidate.request_id);
                    }
                }
                None
            }
            SchedulingPolicy::WeightedFair => unreachable!("weighted policy handled above"),
        }
    }

    fn remove_from_waiting(&mut self, request_id: &RequestId) {
        self.waiting_members.remove(request_id);
        self.waiting_fcfs
            .retain(|candidate| candidate != request_id);
        self.waiting_priority
            .retain(|candidate| &candidate.request_id != request_id);
    }

    fn enqueue_waiting_request(&mut self, request_id: RequestId) {
        let Some(metadata) = self.requests.get(&request_id) else {
            return;
        };
        if !self.waiting_members.insert(request_id.clone()) {
            return;
        }
        match self.config.policy {
            SchedulingPolicy::FCFS => self.waiting_fcfs.push_back(request_id),
            SchedulingPolicy::Priority => self.waiting_priority.push(PriorityRequest {
                request_id,
                priority: metadata.priority,
                workload_class: metadata.workload_class,
                arrival_time: metadata.arrival_time,
            }),
            SchedulingPolicy::WeightedFair => self.waiting_fcfs.push_back(request_id),
        }
    }

    fn select_weighted_waiting_request(&self) -> Option<RequestId> {
        let selected_class = self
            .waiting_members
            .iter()
            .filter_map(|id| {
                self.requests
                    .get(id)
                    .map(|metadata| metadata.workload_class)
            })
            .min_by(|a, b| self.compare_class_service(*a, *b))?;

        let candidates = self.waiting_members.iter().filter(|id| {
            self.requests
                .get(*id)
                .is_some_and(|metadata| metadata.workload_class == selected_class)
        });
        match self.config.policy {
            SchedulingPolicy::WeightedFair | SchedulingPolicy::Priority => {
                candidates.cloned().max_by(|a, b| {
                    let metadata_a = self.requests.get(a);
                    let metadata_b = self.requests.get(b);
                    metadata_a
                        .map(|m| m.priority)
                        .cmp(&metadata_b.map(|m| m.priority))
                        .then_with(|| {
                            metadata_b
                                .map(|m| m.deadline_at)
                                .cmp(&metadata_a.map(|m| m.deadline_at))
                        })
                        .then_with(|| self.compare_arrival_for_max(a, b))
                })
            }
            SchedulingPolicy::FCFS => candidates.cloned().min_by_key(|id| {
                self.requests
                    .get(id)
                    .map(|m| m.sequence_id)
                    .unwrap_or(u64::MAX)
            }),
        }
    }

    fn compare_arrival_for_max(&self, a: &RequestId, b: &RequestId) -> Ordering {
        let sequence_a = self
            .requests
            .get(a)
            .map(|metadata| metadata.sequence_id)
            .unwrap_or(u64::MAX);
        let sequence_b = self
            .requests
            .get(b)
            .map(|metadata| metadata.sequence_id)
            .unwrap_or(u64::MAX);
        sequence_b.cmp(&sequence_a)
    }

    fn compare_class_service(&self, a: WorkloadClass, b: WorkloadClass) -> Ordering {
        Self::compare_class_service_with(&self.class_service, a, b)
    }

    fn compare_class_service_with(
        service: &HashMap<WorkloadClass, u64>,
        a: WorkloadClass,
        b: WorkloadClass,
    ) -> Ordering {
        let service_a = service.get(&a).copied().unwrap_or_default();
        let service_b = service.get(&b).copied().unwrap_or_default();
        let weighted_a = service_a.saturating_mul(Self::workload_weight(b));
        let weighted_b = service_b.saturating_mul(Self::workload_weight(a));
        weighted_a
            .cmp(&weighted_b)
            .then_with(|| Self::workload_order(a).cmp(&Self::workload_order(b)))
    }

    fn workload_weight(workload_class: WorkloadClass) -> u64 {
        match workload_class {
            WorkloadClass::Realtime => 8,
            WorkloadClass::Interactive => 6,
            WorkloadClass::Streaming => 5,
            WorkloadClass::Online => 4,
            WorkloadClass::Batch => 2,
            WorkloadClass::Background => 1,
        }
    }

    fn workload_order(workload_class: WorkloadClass) -> u8 {
        match workload_class {
            WorkloadClass::Realtime => 0,
            WorkloadClass::Interactive => 1,
            WorkloadClass::Streaming => 2,
            WorkloadClass::Online => 3,
            WorkloadClass::Batch => 4,
            WorkloadClass::Background => 5,
        }
    }

    fn record_class_service(&mut self, workload_class: WorkloadClass, tokens: usize) {
        if self.config.policy != SchedulingPolicy::WeightedFair {
            return;
        }
        let service = self.class_service.entry(workload_class).or_default();
        *service = service.saturating_add(tokens.max(1) as u64);
    }

    fn expire_deadlines(&mut self) -> Vec<ExpiredRequest> {
        let now = Instant::now();
        let mut expired: Vec<_> = self
            .requests
            .values()
            .filter(|metadata| {
                metadata
                    .hard_deadline
                    .is_some_and(|deadline| deadline <= now)
            })
            .map(|metadata| ExpiredRequest {
                request_id: metadata.request_id.clone(),
                sequence_id: metadata.sequence_id,
            })
            .collect();
        expired.sort_by_key(|request| request.sequence_id);
        for request in &expired {
            self.begin_terminal_release(&request.session_key(), TerminalReleaseCause::TimedOut);
        }
        expired
    }

    fn deadline_for_request(&self, priority: Priority, workload_class: WorkloadClass) -> Duration {
        let ms = match priority {
            Priority::Critical => self.config.critical_sla_ms.max(1),
            Priority::High => self.config.high_sla_ms.max(1),
            Priority::Normal => self.config.normal_sla_ms.max(1),
            Priority::Low => self.config.low_sla_ms.max(1),
        };
        let scaled = ((ms as f64) * workload_class.deadline_scale()).round() as u64;
        Duration::from_millis(scaled.max(1))
    }

    fn request_overdue_ms(&self, metadata: &RequestMetadata) -> f64 {
        if !self.config.enable_deadline_scheduling {
            return 0.0;
        }
        let now = Instant::now();
        if now <= metadata.deadline_at {
            0.0
        } else {
            (now - metadata.deadline_at).as_secs_f64() * 1000.0
        }
    }

    fn decode_token_quanta(
        &self,
        remaining_decode_budget: usize,
        remaining_request_tokens: usize,
        has_waiting_work: bool,
        overdue_ms: f64,
        workload_class: WorkloadClass,
        preferred_decode_tokens: usize,
        continuous_decode: bool,
    ) -> usize {
        let base = remaining_decode_budget.min(remaining_request_tokens).max(1);
        let preferred_decode_tokens = preferred_decode_tokens.max(1);
        if continuous_decode {
            let exact_solo = self.running.len() == 1
                && self
                    .running
                    .values()
                    .filter(|request| request.prefill_complete)
                    .count()
                    == 1
                && !has_waiting_work
                && overdue_ms <= 0.0;
            if exact_solo && preferred_decode_tokens > 1 {
                return preferred_decode_tokens.min(base).max(1);
            }
            return 1.min(base);
        }
        if !self.config.enable_decode_quanta && preferred_decode_tokens == 1 {
            return 1.min(base);
        }
        if workload_class.prefers_single_token_decode() {
            return 1.min(base);
        }
        let active_decode_requests = self.running.values().filter(|r| r.prefill_complete).count();
        if has_waiting_work || overdue_ms > 0.0 {
            return 1.min(base);
        }
        if active_decode_requests > 1 {
            return 1.min(base);
        }

        let mut max_quanta = if preferred_decode_tokens > 1 {
            preferred_decode_tokens
        } else {
            self.config.max_decode_tokens_per_request.max(1)
        }
        .min(base);

        if self.config.enable_power_adaptive && self.config.power_save_mode {
            max_quanta = max_quanta.min(2);
        }
        if self.running.len() > 2 {
            max_quanta = max_quanta.min(2);
        }

        max_quanta.max(1)
    }

    fn adaptive_waiting_score(&self, request_id: &RequestId) -> f64 {
        let Some(metadata) = self.requests.get(request_id) else {
            return 0.0;
        };
        let base_priority = metadata.priority as i32 as f64;
        let age_ms = metadata.arrival_time.elapsed().as_millis() as f64;
        let age_boost = age_ms / self.config.priority_aging_ms.max(1) as f64;
        let overdue_ms = self.request_overdue_ms(metadata);
        let overdue_boost = if overdue_ms > 0.0 {
            2.0 + (overdue_ms / self.config.priority_aging_ms.max(1) as f64)
        } else {
            0.0
        };
        let prompt_bonus = 1.0
            / (1.0
                + (metadata.total_prompt_tokens as f64
                    / self.config.chunked_prefill_threshold.max(1) as f64));
        base_priority
            + metadata.workload_class.adaptive_score_boost()
            + age_boost
            + overdue_boost
            + (prompt_bonus * 0.2)
    }

    fn has_latency_sensitive_waiting(&self) -> bool {
        self.waiting_members.iter().any(|request_id| {
            self.requests
                .get(request_id)
                .map(|metadata| metadata.workload_class.is_latency_sensitive())
                .unwrap_or(false)
        })
    }

    fn has_throughput_or_background_waiting(&self) -> bool {
        self.waiting_members.iter().any(|request_id| {
            self.requests
                .get(request_id)
                .map(|metadata| {
                    matches!(
                        metadata.workload_class,
                        WorkloadClass::Batch | WorkloadClass::Background
                    )
                })
                .unwrap_or(false)
        })
    }

    fn has_eligible_full_prefill(&self, now: Instant) -> bool {
        let running_full = self.running.iter().any(|(request_id, running)| {
            !running.prefill_complete
                && !running.prefill_in_flight
                && self.requests.get(request_id).is_some_and(|metadata| {
                    metadata.cache_policy.prefill == PrefillMode::Full
                        && metadata.retry_not_before.is_none_or(|retry| retry <= now)
                })
        });
        running_full
            || self.waiting_members.iter().any(|request_id| {
                self.requests.get(request_id).is_some_and(|metadata| {
                    metadata.cache_policy.prefill == PrefillMode::Full
                        && metadata.retry_not_before.is_none_or(|retry| retry <= now)
                })
            })
    }

    fn has_pending_full_prefill(&self) -> bool {
        self.running.iter().any(|(request_id, running)| {
            !running.prefill_complete
                && self
                    .requests
                    .get(request_id)
                    .is_some_and(|metadata| metadata.cache_policy.prefill == PrefillMode::Full)
        }) || self.waiting_members.iter().any(|request_id| {
            self.requests
                .get(request_id)
                .is_some_and(|metadata| metadata.cache_policy.prefill == PrefillMode::Full)
        })
    }

    fn has_eligible_incremental_prefill(&self, now: Instant) -> bool {
        let running_incremental = self.running.iter().any(|(request_id, running)| {
            !running.prefill_complete
                && !running.prefill_in_flight
                && self.requests.get(request_id).is_some_and(|metadata| {
                    metadata.cache_policy.prefill == PrefillMode::Incremental
                        && metadata.retry_not_before.is_none_or(|retry| retry <= now)
                })
        });
        running_incremental
            || self.waiting_members.iter().any(|request_id| {
                self.requests.get(request_id).is_some_and(|metadata| {
                    metadata.cache_policy.prefill == PrefillMode::Incremental
                        && metadata.retry_not_before.is_none_or(|retry| retry <= now)
                })
            })
    }

    fn has_pending_incremental_prefill(&self) -> bool {
        self.running.iter().any(|(request_id, running)| {
            !running.prefill_complete
                && self.requests.get(request_id).is_some_and(|metadata| {
                    metadata.cache_policy.prefill == PrefillMode::Incremental
                })
        }) || self.waiting_members.iter().any(|request_id| {
            self.requests
                .get(request_id)
                .is_some_and(|metadata| metadata.cache_policy.prefill == PrefillMode::Incremental)
        })
    }

    fn refresh_queue_age_sample(&mut self) {
        let (sum_ms, count) = self
            .requests
            .values()
            .fold((0.0, 0usize), |(sum, n), metadata| {
                if self.running.contains_key(&metadata.request_id) {
                    (sum, n)
                } else {
                    (
                        sum + metadata.arrival_time.elapsed().as_secs_f64() * 1000.0,
                        n + 1,
                    )
                }
            });
        if count > 0 {
            let avg = sum_ms / count as f64;
            SchedulerTelemetry::update_ewma(&mut self.telemetry.avg_queue_age_ms, avg, 0.2);
        }
    }

    fn current_token_budget(&self) -> usize {
        let max_tokens = self.config.max_tokens_per_step.max(1);
        let min_tokens = self.config.min_tokens_per_step.min(max_tokens);
        let base = if self.config.enable_adaptive_batching {
            self.telemetry
                .dynamic_tokens_per_step
                .clamp(min_tokens, max_tokens)
        } else {
            max_tokens
        };

        if self.config.enable_power_adaptive {
            let throttled = ((base as f64) * self.thermal_budget_scale()) as usize;
            throttled.max(min_tokens).min(max_tokens)
        } else {
            base
        }
    }

    fn effective_prefill_chunk_threshold(&self, has_decode_demand: bool) -> usize {
        let base = if self.config.enable_adaptive_batching {
            self.telemetry.dynamic_prefill_chunk_threshold.max(32)
        } else {
            self.config.chunked_prefill_threshold.max(32)
        };
        let mut threshold = base;

        // If decode is already active, avoid over-investing in prefill this step.
        if has_decode_demand {
            threshold = threshold.min((base / 2).max(64));
        }

        // Favor throughput for single-request execution.
        if self.waiting_count() <= 1 && self.running.len() <= 1 {
            threshold = threshold.max(base).min(base.saturating_mul(2));
        }

        if self.config.enable_power_adaptive {
            threshold = ((threshold as f64) * self.thermal_budget_scale()) as usize;
        }

        threshold.max(32)
    }

    fn update_dynamic_budget(&mut self) {
        let max_tokens = self.config.max_tokens_per_step.max(1);
        let min_tokens = self.config.min_tokens_per_step.min(max_tokens);
        if !self.config.enable_adaptive_batching {
            self.telemetry.dynamic_tokens_per_step = max_tokens;
            self.telemetry.dynamic_prefill_chunk_threshold =
                self.config.chunked_prefill_threshold.max(32);
            return;
        }

        let current = self.telemetry.dynamic_tokens_per_step;
        let step = (max_tokens / 10).max(1);
        let mut target = current;

        if self.telemetry.avg_ttft_ms > self.config.target_ttft_ms * 1.15 {
            target = (current + step).min(max_tokens);
        } else if self.telemetry.avg_decode_tpot_ms > self.config.target_decode_tpot_ms * 1.20 {
            target = current.saturating_sub(step).max(min_tokens);
        } else if current < max_tokens {
            target = (current + (step / 2).max(1)).min(max_tokens);
        }

        if self.config.enable_power_adaptive {
            let scale = self.thermal_budget_scale();
            target = ((target as f64) * scale).round() as usize;
            target = target.clamp(min_tokens, max_tokens);
        }

        self.telemetry.dynamic_tokens_per_step = target;
        self.update_dynamic_prefill_chunk_threshold();
    }

    fn thermal_budget_scale(&self) -> f64 {
        let mut scale = 1.0;
        if self.config.enable_power_adaptive {
            let pressure = self.config.thermal_pressure_hint.clamp(0.0, 1.0);
            scale *= 1.0 - (pressure * 0.45);
            if self.config.power_save_mode {
                scale *= 0.75;
            }
        }
        scale.clamp(0.40, 1.0)
    }

    fn record_prefill_backoff(&mut self, original_tokens: usize, selected_tokens: usize) {
        if original_tokens == 0 {
            return;
        }
        let ratio = if selected_tokens >= original_tokens {
            0.0
        } else {
            1.0 - (selected_tokens as f64 / original_tokens as f64)
        };
        SchedulerTelemetry::update_ewma(&mut self.telemetry.prefill_backoff_ewma, ratio, 0.25);
    }

    fn update_dynamic_prefill_chunk_threshold(&mut self) {
        let base = self.config.chunked_prefill_threshold.max(32);
        let min_chunk = 32usize;
        let max_chunk = base.saturating_mul(2).max(64);
        let mut current = self
            .telemetry
            .dynamic_prefill_chunk_threshold
            .clamp(min_chunk, max_chunk);
        let step = (base / 8).max(8);

        if self.telemetry.prefill_backoff_ewma > 0.35 {
            current = current.saturating_sub(step).max(min_chunk);
        } else if self.telemetry.avg_ttft_ms > self.config.target_ttft_ms * 1.10 {
            current = current.saturating_add(step).min(max_chunk);
        } else if self.telemetry.prefill_backoff_ewma < 0.08 {
            current = current.saturating_add((step / 2).max(4)).min(max_chunk);
        }

        if self.config.enable_power_adaptive {
            current = ((current as f64) * self.thermal_budget_scale()) as usize;
        }

        self.telemetry.dynamic_prefill_chunk_threshold = current.max(min_chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::TaskType;
    use super::*;
    use crate::backends::BackendKind;
    use crate::engine::ExecutionMode;
    use crate::models::shared::chat::{ChatMessage, ChatRole};
    use std::time::Duration;

    fn small_scheduler() -> Scheduler {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::Priority,
            enable_chunked_prefill: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        Scheduler::new(config)
    }

    fn build_request(task_type: TaskType, id: &str, priority: Priority) -> EngineCoreRequest {
        let mut request = match task_type {
            TaskType::TTS => EngineCoreRequest::tts("hello world"),
            TaskType::ASR => EngineCoreRequest::asr("UklGRg=="),
            TaskType::Chat => EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "hello world".to_string(),
            }]),
            TaskType::SpeechToSpeech => EngineCoreRequest::speech_to_speech("UklGRg=="),
        }
        .with_priority(priority);

        request.id = id.to_string();
        request.prompt_tokens = vec![1];
        request
    }

    fn allow_recompute(scheduler: &mut Scheduler, request_id: &str) {
        let epoch = scheduler
            .get_sequence_id(&request_id.to_string())
            .expect("request epoch");
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.recompute_safe = true;
        profile.cache_release_safe = true;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.to_string(), epoch), &profile,));
    }

    fn allow_external_paged(scheduler: &mut Scheduler, request_id: &str) {
        let epoch = scheduler
            .get_sequence_id(&request_id.to_string())
            .expect("request epoch");
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.cache_mode = CacheMode::ExternalPaged;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.to_string(), epoch), &profile));
    }

    #[test]
    fn realtime_session_preserves_fifo_sample_clock_finish_fence_and_retry_head() {
        let mut scheduler = small_scheduler();
        let mut request = build_request(TaskType::ASR, "realtime", Priority::High);
        request
            .enable_realtime_asr_ingress()
            .expect("ASR realtime ingress");
        assert!(scheduler.add_realtime_session(&request));
        let session = SessionKey::new(
            request.id.clone(),
            scheduler
                .get_sequence_id(&request.id)
                .expect("session epoch"),
        );

        let (first_id, first_input) = scheduler
            .enqueue_realtime_push(&session, 160, 2, 4)
            .expect("first push");
        let (second_id, second_input) = scheduler
            .enqueue_realtime_push(&session, 80, 1, 3)
            .expect("second push");
        let finish_id = scheduler
            .enqueue_realtime_finish(&session, 4, 5)
            .expect("finish");
        assert_eq!(first_input, InputRange::new(0, 160).unwrap());
        assert_eq!(second_input, InputRange::new(160, 240).unwrap());
        assert!(scheduler.enqueue_realtime_push(&session, 1, 1, 1).is_err());
        assert!(scheduler.enqueue_realtime_finish(&session, 1, 1).is_err());

        let first = scheduler.schedule();
        assert_eq!(first.decode_requests.len(), 1);
        assert!(matches!(
            &first.decode_requests[0].work,
            WorkUnit::RealtimePreparation {
                operation_id,
                mode: RealtimePreparationMode::Push,
                input,
                retained_state_input,
                ..
            } if *operation_id == first_id
                && *input == first_input
                && *retained_state_input == InputRange::new(0, 1).unwrap()
        ));
        assert!(scheduler.release_realtime_operation_for_retry(
            &session,
            first.decode_requests[0].plan_id,
            first_id,
            RealtimeSubphase::Preparation,
        ));
        let retried = scheduler.schedule();
        assert!(matches!(
            &retried.decode_requests[0].work,
            WorkUnit::RealtimePreparation { operation_id, .. } if *operation_id == first_id
        ));
        assert!(!scheduler.release_realtime_operation_for_retry(
            &session,
            first.decode_requests[0].plan_id,
            first_id,
            RealtimeSubphase::Preparation,
        ));
        assert!(scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: first.decode_requests[0].plan_id,
                    operation_id: first_id,
                    completed: RealtimeSubphase::Preparation,
                    next: Some(RealtimeSubphase::PromptPrefill { cache_append: 2 }),
                    input_consumed: 0,
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .is_err());
        assert!(scheduler.schedule().decode_requests.is_empty());
        let prepared = scheduler
            .prepare_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: retried.decode_requests[0].plan_id,
                    operation_id: first_id,
                    completed: RealtimeSubphase::Preparation,
                    next: Some(RealtimeSubphase::PromptPrefill { cache_append: 2 }),
                    input_consumed: 0,
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .expect("prospective preparation transition");
        assert_eq!(
            scheduler.realtime_sessions[&session.request_id]
                .pending
                .front()
                .expect("FIFO head")
                .phase,
            RealtimeSubphase::Preparation
        );
        assert!(scheduler.schedule().decode_requests.is_empty());
        scheduler.publish_prepared_realtime_stage_outcome(&session, prepared);
        let prompt = scheduler.schedule();
        assert!(matches!(
            &prompt.decode_requests[0].work,
            WorkUnit::RealtimePromptPrefill {
                operation_id,
                cache_append: 2,
                ..
            } if *operation_id == first_id
        ));
        scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: prompt.decode_requests[0].plan_id,
                    operation_id: first_id,
                    completed: RealtimeSubphase::PromptPrefill { cache_append: 2 },
                    next: Some(RealtimeSubphase::DecodeContinuation),
                    input_consumed: 0,
                    output_steps: 1,
                    cache_append: 2,
                },
            )
            .expect("prompt transition");
        let decode = scheduler.schedule();
        assert!(matches!(
            &decode.decode_requests[0].work,
            WorkUnit::RealtimeDecodeContinuation {
                operation_id,
                retained_state_input,
                ..
            } if *operation_id == first_id
                && *retained_state_input == InputRange::new(1, 2).unwrap()
        ));
        scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: decode.decode_requests[0].plan_id,
                    operation_id: first_id,
                    completed: RealtimeSubphase::DecodeContinuation,
                    next: None,
                    input_consumed: first_input.len(),
                    output_steps: 1,
                    cache_append: 1,
                },
            )
            .expect("exact first completion");

        let second = scheduler.schedule();
        assert!(matches!(
            &second.decode_requests[0].work,
            WorkUnit::RealtimePreparation { operation_id, .. } if *operation_id == second_id
        ));
        assert!(scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: second.decode_requests[0].plan_id,
                    operation_id: second_id,
                    completed: RealtimeSubphase::Preparation,
                    next: None,
                    input_consumed: 79,
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .is_err());
        scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: second.decode_requests[0].plan_id,
                    operation_id: second_id,
                    completed: RealtimeSubphase::Preparation,
                    next: None,
                    input_consumed: second_input.len(),
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .expect("exact second commit");

        let finish = scheduler.schedule();
        assert!(matches!(
            &finish.decode_requests[0].work,
            WorkUnit::RealtimePreparation {
                operation_id,
                mode: RealtimePreparationMode::Finish,
                input,
                ..
            } if *operation_id == finish_id && input.is_empty()
        ));
        scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: finish.decode_requests[0].plan_id,
                    operation_id: finish_id,
                    completed: RealtimeSubphase::Preparation,
                    next: Some(RealtimeSubphase::Completion),
                    input_consumed: 0,
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .expect("finish preparation");
        let completion = scheduler.schedule();
        assert!(matches!(
            &completion.decode_requests[0].work,
            WorkUnit::RealtimeCompletion { operation_id } if *operation_id == finish_id
        ));
        scheduler
            .commit_realtime_stage_outcome(
                &session,
                RealtimeStageOutcome {
                    plan_id: completion.decode_requests[0].plan_id,
                    operation_id: finish_id,
                    completed: RealtimeSubphase::Completion,
                    next: None,
                    input_consumed: 0,
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .expect("finish commit");
        assert!(!scheduler.has_pending_work());
    }

    #[test]
    fn realtime_width_one_rotates_before_serving_an_older_session_continuation() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::Priority,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let mut older = build_request(TaskType::ASR, "realtime-older", Priority::High);
        let mut newer = build_request(TaskType::ASR, "realtime-newer", Priority::High);
        older
            .enable_realtime_asr_ingress()
            .expect("older realtime ingress");
        newer
            .enable_realtime_asr_ingress()
            .expect("newer realtime ingress");
        assert!(scheduler.add_realtime_session(&older));
        assert!(scheduler.add_realtime_session(&newer));
        let older_session = SessionKey::new(
            older.id.clone(),
            scheduler.get_sequence_id(&older.id).expect("older epoch"),
        );
        let newer_session = SessionKey::new(
            newer.id.clone(),
            scheduler.get_sequence_id(&newer.id).expect("newer epoch"),
        );
        let (older_operation, _) = scheduler
            .enqueue_realtime_push(&older_session, 80, 1, 2)
            .expect("older push");
        let (newer_operation, _) = scheduler
            .enqueue_realtime_push(&newer_session, 80, 1, 2)
            .expect("newer push");

        let first = scheduler.schedule();
        assert_eq!(first.decode_requests[0].request_id, older.id);
        scheduler
            .commit_realtime_stage_outcome(
                &older_session,
                RealtimeStageOutcome {
                    plan_id: first.decode_requests[0].plan_id,
                    operation_id: older_operation,
                    completed: RealtimeSubphase::Preparation,
                    next: Some(RealtimeSubphase::Completion),
                    input_consumed: 0,
                    output_steps: 0,
                    cache_append: 0,
                },
            )
            .expect("older preparation continuation");

        let second = scheduler.schedule();
        assert_eq!(second.decode_requests.len(), 1);
        assert_eq!(second.decode_requests[0].request_id, newer.id);
        assert!(matches!(
            &second.decode_requests[0].work,
            WorkUnit::RealtimePreparation { operation_id, .. }
                if *operation_id == newer_operation
        ));
    }

    #[test]
    fn ready_ordinary_decode_receives_bounded_service_amid_realtime_demand() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::Priority,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let ordinary = build_request(TaskType::Chat, "ordinary-decode", Priority::Normal);
        scheduler.add_request(&ordinary);
        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&ordinary.id, 1, 0, 1.0);

        let mut realtime = build_request(TaskType::ASR, "fair-realtime", Priority::High);
        realtime
            .enable_realtime_asr_ingress()
            .expect("realtime ingress");
        assert!(scheduler.add_realtime_session(&realtime));
        let session = SessionKey::new(
            realtime.id.clone(),
            scheduler.get_sequence_id(&realtime.id).expect("epoch"),
        );
        scheduler
            .enqueue_realtime_push(&session, 80, 1, 2)
            .expect("realtime push");
        scheduler.realtime_only_steps_with_ready_decode = MAX_REALTIME_ONLY_STEPS_WITH_READY_DECODE;

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.decode_requests.len(), 1);
        assert_eq!(scheduled.decode_requests[0].request_id, ordinary.id);
        assert!(matches!(
            scheduled.decode_requests[0].work,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                ..
            }
        ));
        assert_eq!(scheduler.realtime_only_steps_with_ready_decode, 0);
    }

    fn allow_incremental_prefill(scheduler: &mut Scheduler, request_id: &str) {
        let epoch = scheduler
            .get_sequence_id(&request_id.to_string())
            .expect("request epoch");
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Incremental;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.to_string(), epoch), &profile,));
    }

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = Scheduler::new(config);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn reused_request_id_receives_a_new_session_epoch() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            ..Default::default()
        });
        let request_id = "reused-id".to_string();

        scheduler.add_request(&build_request(
            TaskType::Chat,
            &request_id,
            Priority::Normal,
        ));
        let first = scheduler.schedule().prefill_requests.remove(0);
        let first_session = first.session_key();
        scheduler.finish_request(&request_id);

        scheduler.add_request(&build_request(
            TaskType::Chat,
            &request_id,
            Priority::Normal,
        ));
        let second = scheduler.schedule().prefill_requests.remove(0);
        let second_session = second.session_key();

        assert_eq!(first_session.request_id, second_session.request_id);
        assert_ne!(first_session.epoch, second_session.epoch);
        assert_eq!(second_session.epoch, second.sequence_id);
    }

    #[test]
    fn recompute_restart_preserves_session_metadata_and_restarts_prefill() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request_id = "recompute-current-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::High)
            .with_workload_class(WorkloadClass::Batch)
            .with_deadline(Some(Instant::now() + Duration::from_secs(30)));
        request.prompt_tokens = vec![1, 2, 3, 4];
        scheduler.add_request(&request);
        allow_recompute(&mut scheduler, &request_id);

        let scheduled = scheduler.schedule();
        let session = scheduled.prefill_requests[0].session_key();
        scheduler.update_after_step(&request_id, 4, 2, 1.0);
        scheduler
            .request_sequence_finalize(&request_id)
            .expect("committed decode can request finalization");
        let metadata_before = scheduler.requests[&request_id].clone();

        assert!(scheduler.restart_request_for_recompute(&session));

        let metadata_after = &scheduler.requests[&request_id];
        assert_eq!(metadata_after.sequence_id, metadata_before.sequence_id);
        assert_eq!(metadata_after.arrival_time, metadata_before.arrival_time);
        assert_eq!(metadata_after.deadline_at, metadata_before.deadline_at);
        assert_eq!(metadata_after.hard_deadline, metadata_before.hard_deadline);
        assert_eq!(metadata_after.priority, metadata_before.priority);
        assert_eq!(
            metadata_after.workload_class,
            metadata_before.workload_class
        );
        assert_eq!(
            metadata_after.cache_policy.mode,
            metadata_before.cache_policy.mode
        );
        assert_eq!(
            metadata_after.cache_policy.prefill,
            metadata_before.cache_policy.prefill
        );
        assert_eq!(
            metadata_after.cache_policy.recompute_safe,
            metadata_before.cache_policy.recompute_safe
        );
        assert_eq!(
            metadata_after.cache_policy.cache_release_safe,
            metadata_before.cache_policy.cache_release_safe
        );
        let running = &scheduler.running[&request_id];
        assert_eq!(running.sequence_id, session.epoch);
        assert_eq!(running.num_tokens_processed, 0);
        assert_eq!(running.num_tokens_generated, 0);
        assert!(!running.prefill_complete);
        assert!(!running.prefill_in_flight);
        assert!(!running.finalize_pending);
        assert!(!running.first_token_emitted);
        assert!(running.paused);

        let restarted = scheduler.schedule();
        assert_eq!(restarted.prefill_requests.len(), 1);
        assert_eq!(restarted.prefill_requests[0].session_key(), session);
        assert_eq!(restarted.prefill_requests[0].num_computed_tokens, 0);
    }

    #[test]
    fn capacity_preemption_protects_sessions_after_their_first_visible_token() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            enable_preemption: true,
            max_batch_size: 1,
            max_tokens_per_step: 4,
            ..Default::default()
        });
        let request_id = "capacity-preemption-owner".to_string();
        scheduler.add_request(&build_request(TaskType::Chat, &request_id, Priority::Low));
        allow_recompute(&mut scheduler, &request_id);
        let session = scheduler.schedule().prefill_requests[0].session_key();
        scheduler.update_after_step(&request_id, 1, 0, 1.0);

        assert_eq!(
            scheduler.capacity_preemption_candidate([session.clone()], Priority::High),
            Some(session.clone())
        );

        scheduler.update_after_step(&request_id, 1, 1, 1.0);
        assert_eq!(
            scheduler.capacity_preemption_candidate([session], Priority::High),
            None
        );
    }

    #[test]
    fn recompute_restart_rejects_stale_session_without_mutation() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request_id = "recompute-stale-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.prompt_tokens = vec![1, 2, 3, 4];
        scheduler.add_request(&request);

        let scheduled = scheduler.schedule();
        let current = scheduled.prefill_requests[0].session_key();
        let stale = SessionKey::new(request_id.clone(), current.epoch.saturating_add(1));
        assert!(!scheduler.restart_request_for_recompute(&SessionKey::new(
            "missing-recompute-session".to_string(),
            current.epoch
        ),));
        assert!(!scheduler.restart_request_for_recompute(&stale));

        let running = &scheduler.running[&request_id];
        assert_eq!(running.sequence_id, current.epoch);
        assert!(!running.prefill_complete);
        assert!(running.prefill_in_flight);
        assert!(!running.paused);
    }

    #[test]
    fn retry_release_makes_the_current_prefill_quantum_eligible_again() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request_id = "retry-current-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.prompt_tokens = vec![1; 8];
        scheduler.add_request(&request);
        let epoch = scheduler
            .get_sequence_id(&request_id)
            .expect("request session");
        let mut profile = ExecutionProfile::fail_closed(
            crate::backends::BackendKind::Cpu,
            request.model_variant,
            crate::engine::ExecutionMode::Sequence,
        );
        profile.prefill = PrefillMode::Full;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.clone(), epoch), &profile,));

        let first = scheduler.schedule();
        let session = first.prefill_requests[0].session_key();
        assert_eq!(first.prefill_requests[0].num_tokens, 8);
        assert!(scheduler.running[&request_id].prefill_in_flight);

        assert!(scheduler.release_execution_quantum_for_retry(&session));

        let running = &scheduler.running[&request_id];
        assert!(!running.prefill_in_flight);
        assert!(!running.prefill_complete);
        assert_eq!(running.num_tokens_processed, 0);
        assert_eq!(running.num_tokens_generated, 0);
        assert!(!running.paused);

        let retry = scheduler.schedule();
        assert_eq!(retry.prefill_requests.len(), 1);
        assert_eq!(retry.prefill_requests[0].session_key(), session);
        assert_eq!(retry.prefill_requests[0].num_computed_tokens, 0);
        assert_eq!(retry.prefill_requests[0].num_tokens, 8);
    }

    #[test]
    fn unexecuted_quantum_refunds_weighted_service_before_retry() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            policy: SchedulingPolicy::WeightedFair,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request_id = "capacity-service-refund".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.prompt_tokens = vec![1; 8];
        request.workload_class = WorkloadClass::Background;
        scheduler.add_request(&request);

        let scheduled = scheduler.schedule();
        let quantum = &scheduled.prefill_requests[0];
        let session = quantum.session_key();
        let tokens = quantum.num_tokens;
        assert_eq!(
            scheduler.class_service[&WorkloadClass::Background],
            tokens as u64
        );
        assert!(scheduler.refund_unexecuted_service(&session, tokens));
        assert_eq!(scheduler.class_service[&WorkloadClass::Background], 0);

        let stale = SessionKey::new(request_id, session.epoch.saturating_add(1));
        assert!(!scheduler.refund_unexecuted_service(&stale, tokens));
        assert_eq!(scheduler.class_service[&WorkloadClass::Background], 0);
    }

    #[test]
    fn retry_release_rejects_a_stale_session_without_mutation() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request_id = "retry-stale-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.prompt_tokens = vec![1; 8];
        scheduler.add_request(&request);

        let first = scheduler.schedule();
        let current = first.prefill_requests[0].session_key();
        let stale = SessionKey::new(request_id.clone(), current.epoch.saturating_add(1));

        assert!(!scheduler.release_execution_quantum_for_retry(&stale));

        let running = &scheduler.running[&request_id];
        assert!(running.prefill_in_flight);
        assert_eq!(running.num_tokens_processed, 0);
        assert_eq!(running.num_tokens_generated, 0);
        assert!(!running.paused);
    }

    #[test]
    fn deferred_retry_is_not_scheduled_before_its_deadline() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request = build_request(TaskType::Chat, "deferred-retry", Priority::Normal);
        assert!(scheduler.add_request(&request));
        let first = scheduler.schedule().prefill_requests.remove(0);
        let session = first.session_key();

        assert!(scheduler.defer_execution_retry(&session, Instant::now() + Duration::from_secs(1),));
        assert!(!scheduler.schedule().has_execution_work());

        assert!(scheduler.defer_execution_retry(&session, Instant::now()));
        let retry = scheduler.schedule();
        assert_eq!(retry.prefill_requests.len(), 1);
        assert_eq!(retry.prefill_requests[0].session_key(), session);
    }

    #[test]
    fn test_adaptive_aging_can_promote_old_request() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 32,
            policy: SchedulingPolicy::Priority,
            enable_adaptive_batching: true,
            priority_aging_ms: 100,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let old_id = "old-low".to_string();
        let fresh_id = "fresh-high".to_string();
        let old = EngineCoreRequest::tts("old request").with_priority(Priority::Low);
        let fresh = EngineCoreRequest::tts("new request").with_priority(Priority::High);

        let mut old = EngineCoreRequest {
            id: old_id.clone(),
            ..old
        };
        old.arrival_time = Instant::now() - Duration::from_secs(3);
        let fresh = EngineCoreRequest {
            id: fresh_id.clone(),
            ..fresh
        };

        scheduler.add_request(&old);
        scheduler.add_request(&fresh);
        if let Some(meta) = scheduler.requests.get_mut(&old_id) {
            meta.arrival_time = Instant::now() - Duration::from_secs(3);
        }

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].request_id, old_id);
    }

    #[test]
    fn adaptive_scheduler_promotes_latency_sensitive_workload() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 32,
            policy: SchedulingPolicy::Priority,
            enable_adaptive_batching: true,
            priority_aging_ms: 10_000,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let mut online = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "summarize this report".to_string(),
        }])
        .with_priority(Priority::High)
        .with_workload_class(WorkloadClass::Online);
        online.id = "online-high".to_string();
        online.prompt_tokens = vec![1, 2, 3, 4];

        let mut realtime = EngineCoreRequest::speech_to_speech("UklGRg==")
            .with_priority(Priority::Normal)
            .with_workload_class(WorkloadClass::Realtime);
        realtime.id = "realtime-normal".to_string();
        realtime.prompt_tokens = vec![5, 6];

        scheduler.add_request(&online);
        scheduler.add_request(&realtime);

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].request_id, "realtime-normal");
    }

    #[test]
    fn decode_quanta_respect_latency_sensitive_workload_class() {
        fn schedule_decode_for(workload_class: WorkloadClass) -> usize {
            let config = SchedulerConfig {
                max_batch_size: 1,
                max_tokens_per_step: 16,
                min_tokens_per_step: 1,
                policy: SchedulingPolicy::FCFS,
                enable_chunked_prefill: false,
                enable_preemption: false,
                enable_adaptive_batching: true,
                enable_decode_quanta: true,
                max_decode_tokens_per_request: 4,
                ..Default::default()
            };
            let mut scheduler = Scheduler::new(config);

            let mut request = EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "continue".to_string(),
            }])
            .with_workload_class(workload_class);
            request.id = format!("decode-{}", workload_class.as_str());
            request.prompt_tokens = vec![1];
            request.params.max_tokens = 16;

            scheduler.add_request(&request);
            let first = scheduler.schedule();
            assert_eq!(first.prefill_requests.len(), 1);
            scheduler.update_after_step(&request.id, 1, 1, 1.0);

            let second = scheduler.schedule();
            assert_eq!(second.decode_requests.len(), 1);
            second.decode_requests[0].num_tokens
        }

        assert_eq!(schedule_decode_for(WorkloadClass::Realtime), 1);
        assert_eq!(schedule_decode_for(WorkloadClass::Streaming), 1);
        assert_eq!(schedule_decode_for(WorkloadClass::Batch), 4);
    }

    #[test]
    fn test_abort_running_request_across_task_types() {
        let task_types = [
            TaskType::TTS,
            TaskType::ASR,
            TaskType::Chat,
            TaskType::SpeechToSpeech,
        ];

        for task_type in task_types {
            let mut scheduler = small_scheduler();
            let request_id = format!("abort-{task_type:?}");
            let request = build_request(task_type, &request_id, Priority::Normal);
            scheduler.add_request(&request);

            let scheduled = scheduler.schedule();
            assert_eq!(
                scheduled.prefill_requests.len(),
                1,
                "expected running request before abort for {task_type:?}"
            );
            assert_eq!(
                scheduler.get_status(&request_id),
                Some(RequestStatus::Running)
            );

            assert!(
                scheduler.abort_request(&request_id),
                "abort should report running request removal for {task_type:?}"
            );
            assert!(
                !scheduler.has_request(&request_id),
                "aborted {task_type:?} request should be removed from scheduler metadata"
            );
            assert_eq!(
                scheduler.get_status(&request_id),
                None,
                "aborted {task_type:?} request must not remain queued/running"
            );

            let after_abort = scheduler.schedule();
            assert!(
                !after_abort.has_work(),
                "no work should remain after aborting sole {task_type:?} request"
            );
        }
    }

    #[test]
    fn test_tts_auto_max_tokens_allows_decode_after_prefill() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let request_id = "tts-auto-max".to_string();
        let mut request = EngineCoreRequest::tts("hello world");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 0;
        scheduler.add_request(&request);

        let first = scheduler.schedule();
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].request_id, request_id);
        scheduler.update_after_step(&request_id, 1, 1, 1.0);

        let second = scheduler.schedule();
        assert_eq!(
            second.decode_requests.len(),
            1,
            "TTS auto max_tokens must still schedule decode"
        );
        assert_eq!(second.decode_requests[0].request_id, request_id);
    }

    #[test]
    fn test_non_tts_zero_max_tokens_gets_safe_default_budget() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let request_id = "chat-zero-max".to_string();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello world".to_string(),
        }]);
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 0;
        scheduler.add_request(&request);

        let first = scheduler.schedule();
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].request_id, request_id);
        scheduler.update_after_step(&request_id, 1, 1, 1.0);

        let second = scheduler.schedule();
        assert_eq!(
            second.decode_requests.len(),
            1,
            "Non-TTS zero max_tokens should be normalized to a safe decode budget"
        );
        assert_eq!(second.decode_requests[0].request_id, request_id);
    }

    #[test]
    fn configured_prefill_chunk_threshold_is_respected() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 16,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            chunked_prefill_threshold: 8,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let request_id = "backoff-prefill".to_string();
        let mut request = EngineCoreRequest::tts("long prompt");
        request.id = request_id.clone();
        request.prompt_tokens = vec![7; 8];
        scheduler.add_request(&request);
        allow_incremental_prefill(&mut scheduler, &request_id);

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].request_id, request_id);
        assert_eq!(
            scheduled.prefill_requests[0].num_tokens, 8,
            "scheduler should use its configured token chunk without fake KV pressure"
        );
    }

    #[test]
    fn incremental_prefill_continues_after_committed_partial_step() {
        let telemetry_before = crate::engine::metrics::engine_batch_metrics_snapshot();
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let request_id = "incremental-prefill".to_string();
        let mut request = EngineCoreRequest::tts("incremental prompt");
        request.id = request_id.clone();
        request.prompt_tokens = vec![7; 8];
        scheduler.add_request(&request);
        allow_incremental_prefill(&mut scheduler, &request_id);

        let first = scheduler.schedule();
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].num_tokens, 4);
        assert_eq!(first.prefill_requests[0].num_computed_tokens, 0);

        let duplicate = scheduler.schedule();
        assert!(
            !duplicate.has_work(),
            "an in-flight prefill quantum must not be scheduled twice"
        );

        scheduler.update_after_step(&request_id, 4, 0, 1.0);

        let second = scheduler.schedule();
        assert_eq!(second.prefill_requests.len(), 1);
        assert_eq!(second.prefill_requests[0].request_id, request_id);
        assert_eq!(second.prefill_requests[0].num_computed_tokens, 4);
        assert_eq!(second.prefill_requests[0].num_tokens, 4);
        assert_eq!(
            second.prefill_requests[0].work,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 4, end: 8 },
                max_output_steps: 4,
                auxiliary_state: None,
            }
        );
        scheduler.update_after_step(&request_id, 4, 0, 1.0);
        let telemetry_after = crate::engine::metrics::engine_batch_metrics_snapshot();
        assert!(
            telemetry_after.incremental_prefill_quanta_committed_total
                >= telemetry_before.incremental_prefill_quanta_committed_total + 2
        );
        assert!(
            telemetry_after.incremental_prefill_tokens_committed_total
                >= telemetry_before.incremental_prefill_tokens_committed_total + 8
        );
        assert!(
            telemetry_after.multispan_prefill_requests_total
                > telemetry_before.multispan_prefill_requests_total
        );
    }

    #[test]
    fn unproven_prefill_profile_fails_closed_to_an_isolated_full_prompt() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            chunked_prefill_threshold: 4,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let request_id = "full-prefill".to_string();
        let mut request = EngineCoreRequest::tts("full prompt");
        request.id = request_id.clone();
        request.prompt_tokens = vec![7; 8];
        scheduler.add_request(&request);

        let epoch = scheduler
            .get_sequence_id(&request_id)
            .expect("request epoch");
        let session = SessionKey::new(request_id.clone(), epoch);
        let scheduled = scheduler.schedule();

        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].session_key(), session);
        assert_eq!(scheduled.prefill_requests[0].num_tokens, 8);
        assert_eq!(scheduled.total_tokens, 8);
        assert_eq!(
            scheduled.prefill_requests[0].work,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 8 },
                max_output_steps: 8,
                auxiliary_state: None,
            }
        );
    }

    #[test]
    fn full_prefill_is_isolated_from_live_decode() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 16,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_adaptive_batching: false,
            ..Default::default()
        });

        let decode_id = "live-decode".to_string();
        let decode = build_request(TaskType::Chat, &decode_id, Priority::Normal);
        assert!(scheduler.add_request(&decode));
        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&decode_id, decode.num_prompt_tokens(), 1, 1.0);

        let full_id = "waiting-full-prefill".to_string();
        let mut full = build_request(TaskType::Chat, &full_id, Priority::Normal);
        full.prompt_tokens = vec![7; 4];
        assert!(scheduler.add_request(&full));

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.decode_requests.len(), 1);
        assert!(scheduled.prefill_requests.is_empty());
    }

    #[test]
    fn compatible_full_prefills_remain_logically_batchable() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 4,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        for request_id in ["full-a", "full-b"] {
            let mut request = build_request(TaskType::Chat, request_id, Priority::Normal);
            request.prompt_tokens = vec![1; 8];
            assert!(scheduler.add_request(&request));
        }

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.prefill_requests.len(), 2);
        assert!(scheduled.decode_requests.is_empty());
        assert_eq!(scheduled.total_tokens, 16);
    }

    #[test]
    fn waiting_full_prefill_gets_bounded_service_among_decode_steps() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 16,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_adaptive_batching: false,
            ..Default::default()
        });

        let decode_id = "persistent-decode".to_string();
        let mut decode = build_request(TaskType::Chat, &decode_id, Priority::Normal);
        decode.params.max_tokens = 64;
        assert!(scheduler.add_request(&decode));
        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&decode_id, decode.num_prompt_tokens(), 1, 1.0);

        let full_id = "bounded-full-prefill".to_string();
        let full = build_request(TaskType::Chat, &full_id, Priority::Normal);
        assert!(scheduler.add_request(&full));

        for step in 0..=MAX_DECODE_ONLY_STEPS_WITH_WAITING_FULL_PREFILL {
            let scheduled = scheduler.schedule();
            if !scheduled.prefill_requests.is_empty() {
                assert_eq!(scheduled.prefill_requests.len(), 1);
                assert_eq!(scheduled.prefill_requests[0].request_id, full_id);
                assert!(scheduled.decode_requests.is_empty());
                assert_eq!(step, MAX_DECODE_ONLY_STEPS_WITH_WAITING_FULL_PREFILL);
                return;
            }
            assert_eq!(
                scheduled.decode_requests.len(),
                1,
                "decode unexpectedly absent at bounded-service step {step}: {scheduled:?}"
            );
            scheduler.update_after_step(&decode_id, 1, 1, 1.0);
            scheduler.record_committed_batch_service(std::slice::from_ref(&decode_id), true);
        }
        panic!("full prefill was starved beyond the bounded decode-only window");
    }

    #[test]
    fn waiting_incremental_prefill_gets_a_bounded_shared_slot() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 16,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            chunked_prefill_threshold: 32,
            enable_adaptive_batching: false,
            ..Default::default()
        });

        let decode_ids = ["persistent-decode-a", "persistent-decode-b"];
        let mut decode_requests = Vec::new();
        for request_id in decode_ids {
            let mut request = build_request(TaskType::Chat, request_id, Priority::Normal);
            request.params.max_tokens = 64;
            assert!(scheduler.add_request(&request));
            decode_requests.push(request);
        }
        let initial = scheduler.schedule();
        assert_eq!(initial.prefill_requests.len(), 2);
        for request in &decode_requests {
            scheduler.update_after_step(&request.id, request.num_prompt_tokens(), 1, 1.0);
        }

        let incremental_id = "bounded-incremental-prefill".to_string();
        let mut incremental = build_request(TaskType::Chat, &incremental_id, Priority::Normal);
        incremental.prompt_tokens = vec![7; 64];
        assert!(scheduler.add_request(&incremental));
        let epoch = scheduler.get_sequence_id(&incremental_id).unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Incremental;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(incremental_id.clone(), epoch), &profile,));

        for step in 0..=MAX_DECODE_ONLY_STEPS_WITH_WAITING_INCREMENTAL_PREFILL {
            let debt_before = scheduler.decode_only_steps_with_waiting_incremental_prefill;
            let scheduled = scheduler.schedule();
            assert_eq!(
                scheduler.decode_only_steps_with_waiting_incremental_prefill, debt_before,
                "scheduler polling changed committed-service debt"
            );
            if let Some(prefill) = scheduled.prefill_requests.first() {
                assert_eq!(prefill.request_id, incremental_id);
                assert_eq!(scheduled.decode_requests.len(), 1);
                assert_eq!(step, MAX_DECODE_ONLY_STEPS_WITH_WAITING_INCREMENTAL_PREFILL);
                return;
            }
            assert_eq!(
                scheduled.decode_requests.len(),
                2,
                "decode rows disappeared before the bounded service step {step}: {scheduled:?}"
            );
            for request in &decode_requests {
                scheduler.update_after_step(&request.id, 1, 1, 1.0);
            }
            scheduler.record_committed_batch_service(
                &decode_requests
                    .iter()
                    .map(|request| request.id.clone())
                    .collect::<Vec<_>>(),
                true,
            );
        }
        panic!("incremental prefill was starved beyond the bounded decode-only window");
    }

    #[test]
    fn prefill_none_contract_cannot_be_misclassified_as_resumable() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let request_id = "none-is-not-resumable".to_string();
        let request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).unwrap();
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        assert_eq!(profile.prefill, PrefillMode::None);
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.clone(), epoch), &profile,));

        assert_eq!(
            scheduler.requests[&request_id].cache_policy.prefill,
            PrefillMode::Full
        );
    }

    #[test]
    fn retry_backoff_prefill_is_not_force_service_eligible() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let request_id = "incremental-retry-backoff".to_string();
        let request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        assert!(scheduler.add_request(&request));
        allow_incremental_prefill(&mut scheduler, &request_id);
        scheduler
            .requests
            .get_mut(&request_id)
            .unwrap()
            .retry_not_before = Some(Instant::now() + Duration::from_secs(60));
        scheduler.decode_only_steps_with_waiting_incremental_prefill =
            MAX_DECODE_ONLY_STEPS_WITH_WAITING_INCREMENTAL_PREFILL - 1;
        scheduler.record_committed_batch_service(&["decode-row".to_string()], true);

        assert!(!scheduler.has_eligible_incremental_prefill(Instant::now()));
        assert_eq!(
            scheduler.decode_only_steps_with_waiting_incremental_prefill,
            MAX_DECODE_ONLY_STEPS_WITH_WAITING_INCREMENTAL_PREFILL
        );
    }

    #[test]
    fn continuous_decode_profiles_schedule_one_token_membership_quanta() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let request_id = "continuous-decode".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.workload_class = WorkloadClass::Streaming;
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.decode_batch = NativeBatchMode::Continuous;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.clone(), epoch), &profile,));

        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, request.num_prompt_tokens(), 1, 1.0);
        let decode = scheduler.schedule();

        assert_eq!(decode.decode_requests.len(), 1);
        assert_eq!(decode.decode_requests[0].num_tokens, 1);
        assert_eq!(
            decode.decode_requests[0].work,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange {
                    start: request.num_prompt_tokens(),
                    end: request.num_prompt_tokens() + 1,
                },
                max_output_steps: 1,
                auxiliary_state: None,
            }
        );
    }

    #[test]
    fn continuous_decode_preserves_an_isolated_model_preferred_quantum() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            enable_decode_quanta: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let request_id = "continuous-solo-quantum".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.workload_class = WorkloadClass::Streaming;
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.decode_batch = NativeBatchMode::Continuous;
        profile.preferred_decode_tokens = 4;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.clone(), epoch), &profile));

        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, request.num_prompt_tokens(), 1, 1.0);

        let decode = scheduler.schedule();
        assert_eq!(decode.decode_requests.len(), 1);
        assert_eq!(decode.decode_requests[0].num_tokens, 4);
    }

    #[test]
    fn fcfs_prefill_preserves_head_order_without_fake_capacity_pressure() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 32,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let big_id = "big-head".to_string();
        let mut big = EngineCoreRequest::tts("big prompt");
        big.id = big_id.clone();
        big.prompt_tokens = vec![1; 12];
        scheduler.add_request(&big);

        let small_id = "small-tail".to_string();
        let mut small = EngineCoreRequest::tts("small prompt");
        small.id = small_id.clone();
        small.prompt_tokens = vec![2; 4];
        scheduler.add_request(&small);

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(
            scheduled.prefill_requests[0].request_id, big_id,
            "FCFS must not skip the queue head based on disconnected KV capacity"
        );
        assert_eq!(
            scheduler.get_status(&small_id),
            Some(RequestStatus::Waiting)
        );
    }

    #[test]
    fn external_paged_decode_uses_capability_authoritative_state() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request = build_request(TaskType::Chat, "managed-decode", Priority::Normal);
        let request_id = request.id.clone();
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).expect("epoch");
        let session = SessionKey::new(request_id.clone(), epoch);
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.cache_mode = CacheMode::ExternalPaged;
        assert!(scheduler.update_execution_profile(&session, &profile));

        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, 1, 0, 1.0);

        let decode = scheduler.schedule();
        assert_eq!(decode.decode_requests.len(), 1);
    }

    #[test]
    fn cacheless_sequences_schedule_without_block_projection() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let request_id = "cacheless-capability-cache".to_string();
        let request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).expect("epoch");
        let session = SessionKey::new(request_id.clone(), epoch);
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        assert!(scheduler.update_execution_profile(&session, &profile));

        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, 1, 0, 1.0);

        let decode = scheduler.schedule();
        assert_eq!(decode.decode_requests.len(), 1);
    }

    #[test]
    fn external_paged_prefill_admission_is_scheduler_metadata_only() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 32,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let req1_id = "prefix-source".to_string();
        let mut req1 = EngineCoreRequest::tts("prefix source");
        req1.id = req1_id.clone();
        req1.prompt_tokens = vec![10, 11, 12, 13];
        scheduler.add_request(&req1);
        allow_external_paged(&mut scheduler, &req1_id);

        let first = scheduler.schedule();
        assert_eq!(first.prefill_requests.len(), 1);
        scheduler.update_after_step(&req1_id, 4, 0, 1.0);
        scheduler.finish_request(&req1_id);

        let req2_id = "prefix-reuser".to_string();
        let mut req2 = EngineCoreRequest::tts("prefix reuser");
        req2.id = req2_id.clone();
        req2.prompt_tokens = vec![10, 11, 12, 13];
        scheduler.add_request(&req2);
        allow_external_paged(&mut scheduler, &req2_id);

        let second = scheduler.schedule();
        assert!(
            second
                .prefill_requests
                .iter()
                .any(|entry| entry.request_id == req2_id),
            "managed prefill admission must not depend on a scheduler cache projection"
        );
    }

    #[test]
    fn test_decode_quanta_can_schedule_multiple_tokens_per_step() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            enable_decode_quanta: true,
            max_decode_tokens_per_request: 4,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        let request_id = "decode-quanta".to_string();
        let mut request = EngineCoreRequest::tts("hello");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        scheduler.add_request(&request);

        let first = scheduler.schedule();
        assert_eq!(first.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, 1, 1, 1.0);

        let second = scheduler.schedule();
        assert_eq!(second.decode_requests.len(), 1);
        assert_eq!(
            second.decode_requests[0].num_tokens, 4,
            "decode quanta should grant multi-token decode when queue pressure is low"
        );
    }

    #[test]
    fn model_preferred_decode_quantum_works_with_global_quanta_disabled() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            enable_adaptive_batching: false,
            enable_decode_quanta: false,
            ..Default::default()
        });
        let request_id = "model-preferred-decode-quantum".to_string();
        let mut request = EngineCoreRequest::tts("hello");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).expect("epoch");
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.preferred_decode_tokens = 4;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.clone(), epoch), &profile));

        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, 1, 1, 1.0);

        let decode = scheduler.schedule();
        assert_eq!(decode.decode_requests.len(), 1);
        assert_eq!(decode.decode_requests[0].num_tokens, 4);
    }

    #[test]
    fn test_decode_quanta_backs_off_when_multiple_decodes_are_active() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            enable_decode_quanta: true,
            max_decode_tokens_per_request: 4,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        for request_id in ["decode-a", "decode-b"] {
            let mut request = EngineCoreRequest::tts("hello");
            request.id = request_id.to_string();
            request.prompt_tokens = vec![1];
            scheduler.add_request(&request);
            allow_incremental_prefill(&mut scheduler, request_id);
        }

        let first = scheduler.schedule();
        assert_eq!(first.prefill_requests.len(), 2);
        for request_id in ["decode-a", "decode-b"] {
            scheduler.update_after_step(&request_id.to_string(), 1, 1, 1.0);
        }

        let second = scheduler.schedule();
        assert_eq!(second.decode_requests.len(), 2);
        assert!(second
            .decode_requests
            .iter()
            .all(|request| request.num_tokens == 1));
    }

    #[test]
    fn adaptive_queue_cleanup_is_deterministic_and_bounded() {
        let config = SchedulerConfig {
            enable_adaptive_batching: true,
            enable_preemption: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        for index in 0..256 {
            let mut request = EngineCoreRequest::tts("same workload");
            request.id = format!("request-{index:03}");
            scheduler.add_request(&request);
        }

        assert_eq!(
            scheduler.select_next_waiting_request().as_deref(),
            Some("request-000")
        );
        for index in 0..256 {
            scheduler.abort_request(&format!("request-{index:03}"));
        }
        assert!(scheduler.waiting_members.is_empty());
        assert!(scheduler.waiting_fcfs.is_empty());
        assert!(scheduler.waiting_priority.is_empty());
    }

    #[test]
    fn explicit_request_deadline_overrides_synthetic_sla() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let deadline = Instant::now() + Duration::from_secs(30);
        let request = EngineCoreRequest::tts("deadline").with_deadline(Some(deadline));
        let request_id = request.id.clone();
        scheduler.add_request(&request);

        assert_eq!(scheduler.requests[&request_id].deadline_at, deadline);
    }

    #[test]
    fn hard_deadlines_expire_in_sequence_order_without_execution() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        for id in ["expired-a", "expired-b"] {
            let mut request = EngineCoreRequest::tts(id)
                .with_deadline(Some(Instant::now() - Duration::from_millis(1)));
            request.id = id.to_string();
            scheduler.add_request(&request);
        }

        let result = scheduler.schedule();

        assert_eq!(
            result
                .expired_requests
                .iter()
                .map(|request| request.request_id.as_str())
                .collect::<Vec<_>>(),
            vec!["expired-a", "expired-b"]
        );
        assert!(!result.has_execution_work());
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn synthetic_sla_is_a_soft_priority_signal_not_a_hard_deadline() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut request = EngineCoreRequest::tts("soft-sla");
        request.id = "soft-sla".to_string();
        request.prompt_tokens = vec![1];
        scheduler.add_request(&request);
        scheduler.requests.get_mut(&request.id).unwrap().deadline_at =
            Instant::now() - Duration::from_secs(1);

        let result = scheduler.schedule();

        assert!(result.expired_requests.is_empty());
        assert_eq!(result.prefill_requests.len(), 1);
    }

    #[test]
    fn expiring_running_request_waits_for_exact_cleanup_before_id_reuse() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut request = EngineCoreRequest::tts("running-deadline");
        request.id = "running-deadline".to_string();
        request.prompt_tokens = vec![1];
        scheduler.add_request(&request);
        assert_eq!(scheduler.schedule().prefill_requests.len(), 1);
        scheduler
            .requests
            .get_mut(&request.id)
            .unwrap()
            .hard_deadline = Some(Instant::now() - Duration::from_millis(1));

        let result = scheduler.schedule();

        assert_eq!(result.expired_requests.len(), 1);
        assert_eq!(scheduler.running_count(), 0);
        let session = result.expired_requests[0].session_key();
        assert_eq!(
            scheduler.pending_expired_cleanup_sessions(),
            vec![session.clone()]
        );
        let stale = SessionKey::new(session.request_id.clone(), session.epoch + 1);
        assert!(!scheduler.confirm_expired_session_cleanup(&stale));
        assert!(!scheduler.add_request(&request));

        assert!(scheduler.confirm_expired_session_cleanup(&session));
        assert!(scheduler.pending_expired_cleanup_sessions().is_empty());
        assert!(scheduler.add_request(&request));
    }

    #[test]
    fn terminal_release_requires_exact_cleanup_and_delivery_before_id_reuse() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let request = build_request(TaskType::Chat, "terminal-fence", Priority::Normal);
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request.id).unwrap();
        let session = SessionKey::new(request.id.clone(), epoch);
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_release_safe = true;
        assert!(scheduler.update_execution_profile(&session, &profile));
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.prefill_requests.len(), 1);

        assert_eq!(
            scheduler.begin_terminal_release(&session, TerminalReleaseCause::Completed),
            BeginTerminalRelease::Started {
                confirmation_required: true,
            }
        );
        assert!(!scheduler.add_request(&request));
        let stale = SessionKey::new(request.id.clone(), session.epoch + 1);
        assert!(!scheduler.confirm_session_release(&stale));
        assert!(!scheduler.mark_terminal_delivered(&stale));

        assert!(scheduler.mark_terminal_delivered(&session));
        assert!(!scheduler.add_request(&request));
        assert!(scheduler.confirm_session_release(&session));
        assert!(scheduler.add_request(&request));
    }

    #[test]
    fn weighted_fair_policy_gives_background_bounded_service() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 1,
            policy: SchedulingPolicy::WeightedFair,
            ..Default::default()
        });
        for index in 0..8 {
            let mut request = EngineCoreRequest::tts("realtime");
            request.id = format!("realtime-{index}");
            request.prompt_tokens = vec![1];
            request.workload_class = WorkloadClass::Realtime;
            scheduler.add_request(&request);
        }
        let mut background = EngineCoreRequest::tts("background");
        background.id = "background".to_string();
        background.prompt_tokens = vec![1];
        background.workload_class = WorkloadClass::Background;
        scheduler.add_request(&background);

        let mut selected = Vec::new();
        for _ in 0..3 {
            let result = scheduler.schedule();
            let request_id = result.prefill_requests[0].request_id.clone();
            selected.push(request_id.clone());
            scheduler.finish_request(&request_id);
        }

        assert!(selected.iter().any(|id| id == "background"));
    }

    #[test]
    fn production_defaults_only_enable_physically_enforced_features() {
        let config = SchedulerConfig::default();
        assert_eq!(config.policy, SchedulingPolicy::WeightedFair);
        assert!(!config.enable_chunked_prefill);
        assert!(!config.enable_preemption);
        assert!(!config.enable_adaptive_batching);
        assert!(!config.enable_power_adaptive);
        assert!(!config.enable_decode_quanta);
    }

    #[test]
    fn committed_sequence_finalization_is_scheduled_as_a_distinct_cache_free_stage() {
        let mut scheduler = small_scheduler();
        let request = build_request(TaskType::TTS, "tts-finalize", Priority::Normal);
        assert!(scheduler.add_request(&request));
        let prefill = scheduler.schedule();
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request.id, request.num_prompt_tokens(), 0, 1.0);
        scheduler
            .request_sequence_finalize(&request.id)
            .expect("committed sequence can finalize");

        let scheduled = scheduler.schedule();
        assert_eq!(scheduled.decode_requests.len(), 1);
        let row = &scheduled.decode_requests[0];
        assert_eq!(row.request_id, request.id);
        assert_eq!(row.num_tokens, 1);
        assert!(matches!(
            row.work,
            WorkUnit::SequenceFinalize {
                max_output_steps: 1
            }
        ));
        assert!(scheduler.schedule().decode_requests.is_empty());

        let session = row.session_key();
        assert!(scheduler.release_execution_quantum_for_retry(&session));
        let retry = scheduler.schedule();
        assert_eq!(retry.decode_requests.len(), 1);
        assert!(matches!(
            retry.decode_requests[0].work,
            WorkUnit::SequenceFinalize { .. }
        ));
    }

    #[test]
    fn sequence_finalization_cannot_bypass_uncommitted_prefill() {
        let mut scheduler = small_scheduler();
        let request = build_request(TaskType::TTS, "tts-finalize-early", Priority::Normal);
        assert!(scheduler.add_request(&request));
        assert!(scheduler.request_sequence_finalize(&request.id).is_err());
    }
}
