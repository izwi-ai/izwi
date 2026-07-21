//! Request scheduler with support for FCFS and priority-based scheduling.
//!
//! The scheduler manages request queues and decides which requests to process
//! in each engine step. It handles:
//! - Waiting queue (new requests awaiting processing)
//! - Running queue (requests currently being processed)
//! - Token budget management
//! - KV cache allocation coordination

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use tracing::debug;

use super::config::EngineCoreConfig;
use super::execution::{CacheMode, ExecutionProfile, NativeBatchMode, PrefillMode};
use super::kv_cache::{CacheResidency, KVCacheManager};
use super::request::{EngineCoreRequest, RequestStatus, WorkloadClass};
use super::types::{BlockId, Priority, RequestId, SequenceId, TaskType};
use super::{InputRange, PlanId, SequencePhase, SessionKey, WorkUnit};
use crate::model::ModelVariant;

/// Scheduling policy for the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    /// First-come, first-served (default)
    #[default]
    FCFS,
    /// Priority-based scheduling (higher priority first)
    Priority,
    /// Weighted workload-class service with priority ordering inside each class.
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
    /// Enable prefix reuse backed by an executor-owned physical cache.
    pub enable_prefix_caching: bool,
    /// Threshold for chunked prefill
    pub chunked_prefill_threshold: usize,
    /// Enable preemption when KV cache is full
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
    /// Enable KV residency tiering hints (GPU <-> CPU residency).
    pub enable_kv_tiering: bool,
}

/// Preemption reason - why a request was preempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionReason {
    /// Memory pressure - KV cache is full
    MemoryPressure,
    /// VAD detected user speech during AI output (interruption)
    VadInterruption,
    /// Manual abort by user
    UserAbort,
    /// Timeout
    Timeout,
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
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_prefix_caching: true,
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
            enable_kv_tiering: false,
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
            enable_prefix_caching: config.enable_prefix_caching,
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
            enable_kv_tiering: config.enable_kv_tiering,
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
    /// Requests that were preempted to make room
    pub preempted_requests: Vec<SessionKey>,
    /// Requests rejected before execution because their caller deadline elapsed.
    pub expired_requests: Vec<ExpiredRequest>,
    /// Total tokens to process this step
    pub total_tokens: usize,
    /// Number of blocks allocated
    pub blocks_allocated: usize,
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
    PreemptionCleanupFailed,
    PreemptionCommitRejected,
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
            preempted_requests: Vec::new(),
            expired_requests: Vec::new(),
            total_tokens: 0,
            blocks_allocated: 0,
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
    /// KV cache blocks allocated to this request
    pub block_ids: Vec<BlockId>,
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
    /// Terminal sessions whose executor-owned cache has not yet been proven
    /// released, or whose terminal event has not yet been delivered. Logical
    /// blocks and the public request ID remain fenced until both conditions are
    /// satisfied.
    pending_releases: HashMap<RequestId, PendingRelease>,
    /// Next sequence ID
    next_sequence_id: SequenceId,
    /// Next execution plan identity.
    next_plan_id: PlanId,
    /// Monotonic scheduling cycle used for one-cycle preemption resume fences.
    schedule_generation: u64,
    /// Adaptive scheduling telemetry.
    telemetry: SchedulerTelemetry,
    /// Completed scheduling quanta by workload class for weighted service.
    class_service: HashMap<WorkloadClass, u64>,
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
    prompt_prefix_tokens: Vec<u32>,
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
    prefix_reuse_safe: bool,
}

impl Default for RequestCachePolicy {
    fn default() -> Self {
        Self {
            mode: None,
            prefill: PrefillMode::Incremental,
            decode_batch: NativeBatchMode::None,
            recompute_safe: false,
            cache_release_safe: false,
            prefix_reuse_safe: false,
        }
    }
}

impl RequestCachePolicy {
    fn allows_recompute_preemption(&self) -> bool {
        self.recompute_safe && self.cache_release_safe
    }

    fn allows_external_prefix_reuse(&self) -> bool {
        self.mode == Some(CacheMode::ExternalPaged) && self.prefix_reuse_safe
    }

    fn has_external_physical_cache(&self) -> bool {
        self.mode == Some(CacheMode::ExternalPaged)
    }

    /// Whether the loaded execution profile has established a real cache
    /// authority outside the scheduler's legacy logical block projection.
    ///
    /// Managed rows use backend arenas, opaque rows retain model-owned tensors,
    /// and cacheless rows retain no KV state. None of those modes may allocate
    /// fake scheduler BlockIds. `None` is kept as the legacy/unprofiled state so
    /// old isolated scheduler callers remain fail-compatible until the profile
    /// refresh performed by EngineCore.
    fn bypasses_legacy_logical_cache(&self) -> bool {
        self.mode.is_some()
    }
}

/// State for a running request.
#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: RequestId,
    sequence_id: SequenceId,
    /// Number of tokens processed so far (prompt + generated)
    num_tokens_processed: usize,
    /// Number of tokens generated so far
    num_tokens_generated: usize,
    /// KV cache blocks allocated
    block_ids: Vec<BlockId>,
    /// Whether prefill is complete
    prefill_complete: bool,
    /// Whether a prefill quantum has been scheduled but not yet committed.
    prefill_in_flight: bool,
    /// Priority of this request
    priority: Priority,
    /// Coarse latency/throughput class for this request.
    workload_class: WorkloadClass,
    /// Whether this request has produced its first output token.
    first_token_emitted: bool,
    /// Whether this request is temporarily paused due to preemption.
    paused: bool,
    /// Whether the scheduler has selected this session for two-phase
    /// preemption and is waiting for executor cache cleanup confirmation.
    preemption_pending: bool,
    /// Scheduling generation in which this preempted victim must remain paused
    /// so the request that caused preemption gets the first chance to allocate.
    preemption_defer_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct PrefillAllocationPlan {
    total_blocks_needed: usize,
    reusable_blocks: usize,
    additional_blocks: usize,
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
            pending_releases: HashMap::new(),
            next_sequence_id: 0,
            next_plan_id: 1,
            schedule_generation: 0,
            telemetry,
            class_service: HashMap::new(),
        }
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
            prompt_prefix_tokens: request.prompt_tokens.clone(),
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

    /// Install the loaded executor's cache contract for one exact scheduler
    /// incarnation. Stale profiles cannot mutate a reused public request ID.
    pub fn update_execution_profile(
        &mut self,
        session: &SessionKey,
        profile: &ExecutionProfile,
    ) -> bool {
        if profile.cache_mode == CacheMode::ExternalPaged
            && self
                .running
                .get(&session.request_id)
                .is_some_and(|running| !running.block_ids.is_empty())
        {
            // Cache authority is selected before first execution. Refuse a
            // mid-session promotion that would strand legacy logical blocks.
            return false;
        }
        let Some(metadata) = self.requests.get_mut(&session.request_id) else {
            return false;
        };
        if metadata.sequence_id != session.epoch {
            return false;
        }
        metadata.cache_policy = RequestCachePolicy {
            mode: Some(profile.cache_mode),
            prefill: profile.prefill,
            decode_batch: profile.decode_batch,
            recompute_safe: profile.recompute_safe,
            cache_release_safe: profile.cache_release_safe,
            prefix_reuse_safe: profile.prefix_reuse_safe,
        };
        true
    }

    /// Schedule requests for the next step.
    pub fn schedule(&mut self, kv_cache: &mut KVCacheManager) -> ScheduleResult {
        self.schedule_generation = self.schedule_generation.wrapping_add(1);
        let schedule_generation = self.schedule_generation;
        let mut result = ScheduleResult::empty();
        result.expired_requests = self.expire_deadlines();
        let scheduling_now = Instant::now();
        let mut remaining_batch = self.config.max_batch_size;
        self.refresh_queue_age_sample();
        self.update_dynamic_budget();

        let mut total_budget = self.current_token_budget();
        let latency_sensitive_waiting = self.has_latency_sensitive_waiting();
        let throughput_waiting_only =
            !latency_sensitive_waiting && self.has_throughput_or_background_waiting();
        let kv_stats = kv_cache.stats();
        let kv_utilization = if kv_stats.soft_max_blocks > 0 {
            kv_stats.allocated_blocks as f64 / kv_stats.soft_max_blocks as f64
        } else {
            0.0
        };
        if kv_utilization > 0.95 {
            total_budget = (total_budget.saturating_mul(45) / 100).max(1);
        } else if kv_utilization > 0.90 {
            total_budget = (total_budget.saturating_mul(65) / 100).max(1);
        } else if kv_utilization > 0.80 {
            total_budget = (total_budget.saturating_mul(80) / 100).max(1);
        }
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
            reserved_prefill_budget = total_budget.min(32).max(1);
            decode_budget = total_budget.saturating_sub(reserved_prefill_budget);
        }
        let mut remaining_decode_budget = decode_budget;

        // Phase 1: schedule decode requests (already running prefill-complete requests).
        let mut decode_candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| r.prefill_complete)
            .filter_map(|(id, r)| {
                if r.preemption_pending {
                    return None;
                }
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
                    r.block_ids.clone(),
                    r.num_tokens_processed,
                    remaining_decode_tokens,
                    r.num_tokens_generated,
                    r.paused,
                    metadata.workload_class,
                    overdue_ms,
                    metadata.cache_policy.decode_batch == NativeBatchMode::Continuous,
                    metadata.cache_policy.bypasses_legacy_logical_cache(),
                ))
            })
            .collect();

        if self.config.enable_adaptive_batching
            && self.config.policy != SchedulingPolicy::WeightedFair
        {
            // Favor overdue requests first, then requests close to completion.
            decode_candidates.sort_by(|a, b| {
                b.9.partial_cmp(&a.9)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.7.cmp(&a.7))
                    .then_with(|| {
                        b.8.adaptive_score_boost()
                            .partial_cmp(&a.8.adaptive_score_boost())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| a.5.cmp(&b.5))
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.6.cmp(&b.6))
            });
        } else if self.config.policy == SchedulingPolicy::WeightedFair {
            let mut simulated_service = self.class_service.clone();
            let mut fair_order = Vec::with_capacity(decode_candidates.len());
            while !decode_candidates.is_empty() {
                let next_index = (0..decode_candidates.len())
                    .min_by(|left, right| {
                        let a = &decode_candidates[*left];
                        let b = &decode_candidates[*right];
                        Self::compare_class_service_with(&simulated_service, a.8, b.8)
                            .then_with(|| b.2.cmp(&a.2))
                            .then_with(|| {
                                b.9.partial_cmp(&a.9).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .then_with(|| a.1.cmp(&b.1))
                    })
                    .unwrap_or(0);
                let candidate = decode_candidates.remove(next_index);
                let next_service = simulated_service
                    .get(&candidate.8)
                    .copied()
                    .unwrap_or_default()
                    .saturating_add(1);
                simulated_service.insert(candidate.8, next_service);
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
            self.effective_prefill_chunk_threshold(kv_utilization, has_decode_demand);

        for (
            request_id,
            sequence_id,
            priority,
            mut block_ids,
            num_computed,
            remaining_decode_tokens,
            _generated_tokens,
            _paused,
            workload_class,
            overdue_ms,
            continuous_decode,
            bypasses_legacy_logical_cache,
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
            if self.config.policy == SchedulingPolicy::WeightedFair
                && !self.waiting_members.is_empty()
                && remaining_batch <= 1
            {
                break;
            }

            let mut num_tokens = self.decode_token_quanta(
                remaining_decode_budget,
                remaining_decode_tokens,
                self.waiting_count() > 0,
                kv_utilization,
                overdue_ms,
                workload_class,
            );
            if continuous_decode {
                // Membership can change only between committed model quanta.
                // Keep each scheduler transaction to one decode token so new
                // rows may join and finished/cancelled rows may leave promptly.
                num_tokens = num_tokens.min(1);
            }
            if num_tokens == 0 {
                continue;
            }

            if bypasses_legacy_logical_cache {
                // Cache ownership belongs to the selected execution profile:
                // managed coordinators reserve physical pages later, opaque
                // adapters retain their own tensors, and cacheless rows retain
                // nothing. Legacy BlockIds cannot become a second authority.
                debug_assert!(block_ids.is_empty());
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
                    block_ids,
                    num_computed_tokens: num_computed,
                    work: WorkUnit::SequenceStep {
                        phase: SequencePhase::Decode,
                        input: InputRange {
                            start: num_computed,
                            end: num_computed.saturating_add(num_tokens),
                        },
                        max_output_steps: num_tokens,
                    },
                });
                remaining_decode_budget = remaining_decode_budget.saturating_sub(num_tokens);
                remaining_batch -= 1;
                result.total_tokens += num_tokens;
                self.record_class_service(workload_class, num_tokens);
                continue;
            }

            // Decode quanta are opportunistic: if KV pressure cannot satisfy the
            // selected chunk size, progressively back off before skipping.
            loop {
                let total_tokens = num_computed.saturating_add(num_tokens);
                let blocks_needed = kv_cache.blocks_for_tokens(total_tokens);
                let additional_blocks = blocks_needed.saturating_sub(block_ids.len());

                if additional_blocks > 0 && !kv_cache.can_allocate(additional_blocks) {
                    // Try preemption if enabled
                    if self.config.enable_preemption {
                        let protected = result.all_request_ids().into_iter().collect();
                        let preempted = self.try_preempt_for_blocks(
                            additional_blocks,
                            priority,
                            &protected,
                            kv_cache,
                        );
                        if !preempted.is_empty() {
                            result.preempted_requests.extend(preempted);
                        }
                    }
                }

                if additional_blocks == 0 || kv_cache.can_allocate(additional_blocks) {
                    break;
                }

                if num_tokens <= 1 {
                    num_tokens = 0;
                    break;
                }
                num_tokens = (num_tokens / 2).max(1);
            }
            if num_tokens == 0 {
                continue;
            }

            let total_tokens = num_computed.saturating_add(num_tokens);
            let blocks_needed = kv_cache.blocks_for_tokens(total_tokens);
            let additional_blocks = blocks_needed.saturating_sub(block_ids.len());

            // Check if we need to allocate more blocks
            if additional_blocks > 0 {
                if !kv_cache.can_allocate(additional_blocks) {
                    continue;
                }

                let extended_blocks = kv_cache.extend(&request_id, additional_blocks);
                if extended_blocks.len() < additional_blocks {
                    kv_cache.free(&request_id);
                    continue;
                }
                block_ids.extend(extended_blocks);
                result.blocks_allocated += additional_blocks;
            }

            // Shared-prefix blocks must be detached before appending decode tokens.
            if !block_ids.is_empty() && kv_cache.ensure_writable_last_block(&request_id).is_none() {
                if self.config.enable_preemption {
                    let protected = result.all_request_ids().into_iter().collect();
                    let preempted = self.try_preempt_for_blocks(1, priority, &protected, kv_cache);
                    if !preempted.is_empty() {
                        result.preempted_requests.extend(preempted);
                    }
                }
                if kv_cache.ensure_writable_last_block(&request_id).is_none() {
                    continue;
                }
            }

            if let Some(updated_blocks) = kv_cache.get_block_table(&request_id) {
                block_ids = updated_blocks.to_vec();
            }

            if let Some(running) = self.running.get_mut(&request_id) {
                running.paused = false;
                running.block_ids = block_ids.clone();
            }

            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            result.decode_requests.push(ScheduledRequest {
                plan_id,
                request_id: request_id.clone(),
                sequence_id,
                num_tokens,
                is_prefill: false,
                block_ids,
                num_computed_tokens: num_computed,
                work: WorkUnit::SequenceStep {
                    phase: SequencePhase::Decode,
                    input: InputRange {
                        start: num_computed,
                        end: num_computed.saturating_add(num_tokens),
                    },
                    max_output_steps: num_tokens,
                },
            });

            remaining_decode_budget = remaining_decode_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            result.total_tokens += num_tokens;
            self.record_class_service(workload_class, num_tokens);
        }

        // Phase 2: schedule prefill requests.
        let mut remaining_prefill_budget = if self.config.enable_adaptive_batching
            || self.config.policy == SchedulingPolicy::WeightedFair
        {
            reserved_prefill_budget.saturating_add(remaining_decode_budget)
        } else {
            remaining_decode_budget
        };
        let prefill_admission_cap = if has_decode_demand && kv_utilization > 0.90 {
            1
        } else if has_decode_demand && kv_utilization > 0.80 {
            2
        } else {
            usize::MAX
        };
        let mut prefill_admissions = 0usize;

        // Phase 2a: continue incomplete prefills before admitting new waiting requests.
        // An in-flight quantum is excluded until update_after_step commits its
        // progress, preventing duplicate plans when schedule is polled again.
        let mut incomplete_prefill_candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| !r.prefill_complete && !r.prefill_in_flight && !r.preemption_pending)
            .filter_map(|(id, r)| {
                if r.preemption_defer_generation == Some(schedule_generation) {
                    return None;
                }
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

        for (request_id, priority, sequence_id, num_computed) in incomplete_prefill_candidates {
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
            let mut target_tokens = remaining_prompt;
            if !full_prefill
                && self.config.enable_chunked_prefill
                && target_tokens > effective_prefill_chunk_threshold
            {
                target_tokens = effective_prefill_chunk_threshold;
            }
            if full_prefill {
                if target_tokens > remaining_prefill_budget && result.has_execution_work() {
                    continue;
                }
            } else {
                target_tokens = target_tokens.min(remaining_prefill_budget);
            }
            if target_tokens == 0 {
                continue;
            }

            let existing_blocks = self
                .running
                .get(&request_id)
                .map(|r| r.block_ids.len())
                .unwrap_or(0);
            let original_target_tokens = target_tokens;
            let mut num_tokens = target_tokens;
            let mut selected_blocks = None;
            let mut fresh_allocated_blocks = 0usize;

            if metadata.cache_policy.bypasses_legacy_logical_cache() {
                debug_assert_eq!(existing_blocks, 0);
                selected_blocks = Some(Vec::new());
            }

            while num_tokens > 0 && selected_blocks.is_none() {
                let total_tokens_after = num_computed.saturating_add(num_tokens);
                let plan = self.prefill_allocation_plan(
                    kv_cache,
                    &metadata.prompt_prefix_tokens,
                    total_tokens_after,
                    existing_blocks,
                    self.pending_releases.is_empty()
                        && metadata.cache_policy.allows_external_prefix_reuse(),
                );

                if plan.additional_blocks > 0 && !kv_cache.can_allocate(plan.additional_blocks) {
                    if self.config.enable_preemption {
                        let protected = result.all_request_ids().into_iter().collect();
                        let preempted = self.try_preempt_for_blocks(
                            plan.additional_blocks,
                            priority,
                            &protected,
                            kv_cache,
                        );
                        if !preempted.is_empty() {
                            result.preempted_requests.extend(preempted);
                        }
                    }
                }

                if plan.additional_blocks > 0 && !kv_cache.can_allocate(plan.additional_blocks) {
                    if !full_prefill && self.should_backoff_prefill_chunk(num_tokens) {
                        num_tokens = Self::halve_prefill_chunk(num_tokens);
                        continue;
                    }
                    break;
                }

                let (block_ids, fresh_blocks) = if existing_blocks == 0 {
                    let block_ids = if self.pending_releases.is_empty()
                        && self.config.enable_prefix_caching
                        && metadata.cache_policy.allows_external_prefix_reuse()
                    {
                        kv_cache.allocate_with_prefix_tokens(
                            &request_id,
                            plan.total_blocks_needed,
                            &metadata.prompt_prefix_tokens,
                        )
                    } else {
                        kv_cache.allocate(&request_id, plan.total_blocks_needed)
                    };
                    if block_ids.len() < plan.total_blocks_needed {
                        kv_cache.free(&request_id);
                        if !full_prefill && self.should_backoff_prefill_chunk(num_tokens) {
                            num_tokens = Self::halve_prefill_chunk(num_tokens);
                            continue;
                        }
                        break;
                    }
                    (
                        block_ids,
                        plan.total_blocks_needed
                            .saturating_sub(plan.reusable_blocks),
                    )
                } else {
                    if plan.additional_blocks > 0 {
                        let extended_blocks = kv_cache.extend(&request_id, plan.additional_blocks);
                        if extended_blocks.len() < plan.additional_blocks {
                            kv_cache.free(&request_id);
                            if !full_prefill && self.should_backoff_prefill_chunk(num_tokens) {
                                num_tokens = Self::halve_prefill_chunk(num_tokens);
                                continue;
                            }
                            break;
                        }
                    }
                    (
                        kv_cache
                            .get_block_table(&request_id)
                            .map(|ids| ids.to_vec())
                            .unwrap_or_default(),
                        plan.additional_blocks,
                    )
                };

                selected_blocks = Some(block_ids);
                fresh_allocated_blocks = fresh_blocks;
                break;
            }

            let Some(block_ids) = selected_blocks else {
                self.record_prefill_backoff(original_target_tokens, 0);
                continue;
            };
            self.record_prefill_backoff(original_target_tokens, num_tokens);

            if let Some(running) = self.running.get_mut(&request_id) {
                running.prefill_in_flight = true;
                running.paused = false;
                running.preemption_defer_generation = None;
                running.block_ids = block_ids.clone();
            }

            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            result.prefill_requests.push(ScheduledRequest {
                plan_id,
                request_id: request_id.clone(),
                sequence_id,
                num_tokens,
                is_prefill: true,
                block_ids,
                num_computed_tokens: num_computed,
                work: WorkUnit::SequenceStep {
                    phase: SequencePhase::Prefill,
                    input: InputRange {
                        start: num_computed,
                        end: num_computed.saturating_add(num_tokens),
                    },
                    max_output_steps: num_tokens.max(1),
                },
            });

            result.blocks_allocated += fresh_allocated_blocks;
            remaining_prefill_budget = remaining_prefill_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
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
            let mut target_tokens = metadata.total_prompt_tokens;

            // Apply chunked prefill if enabled and prompt is long
            if !full_prefill
                && self.config.enable_chunked_prefill
                && target_tokens > effective_prefill_chunk_threshold
            {
                target_tokens = effective_prefill_chunk_threshold;
            }

            // Full-prefill executors cannot honor a scheduler chunk. Permit one
            // indivisible over-budget prompt only in an otherwise empty step.
            if full_prefill {
                if target_tokens > remaining_prefill_budget && result.has_execution_work() {
                    deferred_waiting.push(request_id);
                    continue;
                }
            } else {
                target_tokens = target_tokens.min(remaining_prefill_budget);
            }
            if target_tokens == 0 {
                break;
            }

            let original_target_tokens = target_tokens;
            let mut num_tokens = target_tokens;
            let mut selected = None;
            let mut fresh_allocated_blocks = 0usize;

            if metadata.cache_policy.bypasses_legacy_logical_cache() {
                selected = Some(Vec::new());
            }

            while num_tokens > 0 && selected.is_none() {
                let plan = self.prefill_allocation_plan(
                    kv_cache,
                    &metadata.prompt_prefix_tokens,
                    num_tokens,
                    0,
                    self.pending_releases.is_empty()
                        && metadata.cache_policy.allows_external_prefix_reuse(),
                );

                if plan.additional_blocks > 0 && !kv_cache.can_allocate(plan.additional_blocks) {
                    if self.config.enable_preemption {
                        let preempted = self.try_preempt_for_blocks(
                            plan.additional_blocks,
                            metadata.priority,
                            &result.all_request_ids().into_iter().collect(),
                            kv_cache,
                        );
                        if !preempted.is_empty() {
                            result.preempted_requests.extend(preempted);
                        }
                    }
                }

                if plan.additional_blocks > 0 && !kv_cache.can_allocate(plan.additional_blocks) {
                    if !full_prefill && self.should_backoff_prefill_chunk(num_tokens) {
                        num_tokens = Self::halve_prefill_chunk(num_tokens);
                        continue;
                    }
                    break;
                }

                let block_ids = if self.pending_releases.is_empty()
                    && self.config.enable_prefix_caching
                    && metadata.cache_policy.allows_external_prefix_reuse()
                {
                    kv_cache.allocate_with_prefix_tokens(
                        &request_id,
                        plan.total_blocks_needed,
                        &metadata.prompt_prefix_tokens,
                    )
                } else {
                    kv_cache.allocate(&request_id, plan.total_blocks_needed)
                };
                if block_ids.len() < plan.total_blocks_needed {
                    debug!("Failed to allocate required blocks for {}", request_id);
                    kv_cache.free(&request_id);
                    if !full_prefill && self.should_backoff_prefill_chunk(num_tokens) {
                        num_tokens = Self::halve_prefill_chunk(num_tokens);
                        continue;
                    }
                    break;
                }

                fresh_allocated_blocks = plan
                    .total_blocks_needed
                    .saturating_sub(plan.reusable_blocks);
                selected = Some(block_ids);
                break;
            }

            let Some(block_ids) = selected else {
                self.record_prefill_backoff(original_target_tokens, 0);
                deferred_waiting.push(request_id);
                continue;
            };
            self.record_prefill_backoff(original_target_tokens, num_tokens);

            // Create running state
            let running = RunningRequest {
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens_processed: 0,
                num_tokens_generated: 0,
                block_ids: block_ids.clone(),
                // Scheduling is not a commit. A failed/retryable prefill must
                // remain a prefill until update_after_step confirms that the
                // complete prompt was actually consumed.
                prefill_complete: false,
                prefill_in_flight: true,
                priority: metadata.priority,
                workload_class: metadata.workload_class,
                first_token_emitted: false,
                paused: false,
                preemption_pending: false,
                preemption_defer_generation: None,
            };

            let plan_id = self.next_plan_id;
            self.next_plan_id = self.next_plan_id.saturating_add(1);
            result.prefill_requests.push(ScheduledRequest {
                plan_id,
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens,
                is_prefill: true,
                block_ids,
                num_computed_tokens: 0,
                work: WorkUnit::SequenceStep {
                    phase: SequencePhase::Prefill,
                    input: InputRange {
                        start: 0,
                        end: num_tokens,
                    },
                    max_output_steps: num_tokens.max(1),
                },
            });

            self.running.insert(request_id.clone(), running);

            result.blocks_allocated += fresh_allocated_blocks;
            remaining_prefill_budget = remaining_prefill_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            prefill_admissions = prefill_admissions.saturating_add(1);
            result.total_tokens += num_tokens;
            self.record_class_service(metadata.workload_class, num_tokens);
        }

        for request_id in deferred_waiting {
            self.enqueue_waiting_request(request_id);
        }

        if self.config.enable_kv_tiering {
            let mut hot = HashSet::new();
            for req in result
                .decode_requests
                .iter()
                .chain(result.prefill_requests.iter())
            {
                hot.insert(req.request_id.clone());
            }
            for request_id in self.running.keys() {
                let physical = self
                    .requests
                    .get(request_id)
                    .is_some_and(|metadata| metadata.cache_policy.has_external_physical_cache());
                if !physical {
                    continue;
                }
                if hot.contains(request_id) {
                    kv_cache.set_request_residency(request_id, CacheResidency::Gpu);
                } else {
                    kv_cache.set_request_residency(request_id, CacheResidency::Cpu);
                }
            }
        }

        result
    }

    /// Update request state after a step.
    pub fn update_after_step(
        &mut self,
        request_id: &RequestId,
        tokens_processed: usize,
        tokens_generated: usize,
        new_block_ids: Vec<BlockId>,
        step_time_ms: f64,
    ) {
        if tokens_processed > 0 || tokens_generated > 0 {
            if let Some(metadata) = self.requests.get_mut(request_id) {
                metadata.retry_not_before = None;
            }
        }
        if let Some(running) = self.running.get_mut(request_id) {
            running.prefill_in_flight = false;
            running.paused = false;
            running.num_tokens_processed += tokens_processed;
            running.num_tokens_generated += tokens_generated;
            running.block_ids.extend(new_block_ids);

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

    /// Release an uncommitted prefill quantum so the exact request session can
    /// retry it without changing committed scheduler or logical-cache state.
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
        running.prefill_complete = running.num_tokens_processed >= metadata.total_prompt_tokens;
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
        if running.sequence_id != session.epoch || running.preemption_pending {
            return false;
        }
        running.prefill_in_flight = false;
        metadata.retry_not_before = Some(not_before);
        true
    }

    /// Restart an exact running request incarnation from prefill after an
    /// executor reports that its session must be recomputed.
    pub fn restart_request_for_recompute(
        &mut self,
        session: &SessionKey,
        kv_cache: &mut KVCacheManager,
    ) -> bool {
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

        kv_cache.free(&session.request_id);
        running.block_ids.clear();
        running.num_tokens_processed = 0;
        running.num_tokens_generated = 0;
        running.prefill_complete = false;
        running.prefill_in_flight = false;
        running.first_token_emitted = false;
        running.paused = true;
        true
    }

    /// Mark a request as finished and remove it.
    pub fn finish_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) {
        self.remove_from_waiting(request_id);
        if let Some(running) = self.running.remove(request_id) {
            // Free KV cache blocks
            kv_cache.free(&running.request_id);
            debug!(
                "Finished request {}, freed {} blocks",
                request_id,
                running.block_ids.len()
            );
        }
        self.requests.remove(request_id);
    }

    /// Move an exact active session into terminal quarantine without releasing
    /// its logical cache. The request ID remains fenced until cleanup is
    /// confirmed and its terminal event has been delivered.
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

    /// Release logical cache for an exact terminal session only after the core
    /// has established that physical cleanup is complete or not required.
    pub(crate) fn confirm_session_release(
        &mut self,
        session: &SessionKey,
        kv_cache: &mut KVCacheManager,
    ) -> bool {
        let remove = {
            let Some(pending) = self.pending_releases.get_mut(&session.request_id) else {
                return false;
            };
            if pending.session != *session {
                return false;
            }
            if !pending.cleanup_confirmed {
                kv_cache.free(&session.request_id);
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

    pub(crate) fn force_release_all_after_executor_shutdown(
        &mut self,
        kv_cache: &mut KVCacheManager,
    ) {
        for request_id in self.pending_releases.keys() {
            kv_cache.free(request_id);
        }
        self.pending_releases.clear();
    }

    /// Compatibility helper retained for scheduler-level tests. Core runtime
    /// paths use the generalized exact-session release protocol above.
    pub fn confirm_expired_session_cleanup(
        &mut self,
        session: &SessionKey,
        kv_cache: &mut KVCacheManager,
    ) -> bool {
        let confirmed = self.confirm_session_release(session, kv_cache);
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
    pub fn abort_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) -> bool {
        self.remove_from_waiting(request_id);

        // Remove from running
        if let Some(running) = self.running.remove(request_id) {
            kv_cache.free(&running.request_id);
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
        self.waiting_count() > 0 || self.running_count() > 0
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

    #[cfg(test)]
    pub(crate) fn prepare_preemption_for_test(
        &mut self,
        request_id: &RequestId,
    ) -> Option<SessionKey> {
        let running = self.running.get_mut(request_id)?;
        running.prefill_in_flight = false;
        running.paused = true;
        running.preemption_pending = true;
        Some(SessionKey::new(request_id.clone(), running.sequence_id))
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

    fn should_backoff_prefill_chunk(&self, num_tokens: usize) -> bool {
        self.config.enable_chunked_prefill && num_tokens > 1
    }

    fn halve_prefill_chunk(num_tokens: usize) -> usize {
        num_tokens.saturating_sub(num_tokens / 2)
    }

    fn prefill_allocation_plan(
        &self,
        kv_cache: &KVCacheManager,
        prompt_tokens: &[u32],
        total_tokens_after: usize,
        existing_blocks: usize,
        allow_prefix_reuse: bool,
    ) -> PrefillAllocationPlan {
        let total_blocks_needed = kv_cache.blocks_for_tokens(total_tokens_after);
        let reusable_blocks =
            if self.config.enable_prefix_caching && allow_prefix_reuse && existing_blocks == 0 {
                kv_cache.estimate_prefix_reuse_blocks(prompt_tokens, total_blocks_needed)
            } else {
                0
            };
        let additional_blocks =
            total_blocks_needed.saturating_sub(existing_blocks.saturating_add(reusable_blocks));
        PrefillAllocationPlan {
            total_blocks_needed,
            reusable_blocks,
            additional_blocks,
        }
    }

    fn decode_token_quanta(
        &self,
        remaining_decode_budget: usize,
        remaining_request_tokens: usize,
        has_waiting_work: bool,
        kv_utilization: f64,
        overdue_ms: f64,
        workload_class: WorkloadClass,
    ) -> usize {
        let base = remaining_decode_budget.min(remaining_request_tokens).max(1);
        if !self.config.enable_decode_quanta {
            return 1.min(base);
        }
        if workload_class.prefers_single_token_decode() {
            return 1.min(base);
        }
        let active_decode_requests = self.running.values().filter(|r| r.prefill_complete).count();
        if has_waiting_work || overdue_ms > 0.0 || kv_utilization > 0.80 {
            return 1.min(base);
        }
        if active_decode_requests > 1 {
            return 1.min(base);
        }

        let mut max_quanta = self.config.max_decode_tokens_per_request.max(1).min(base);

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

    fn effective_prefill_chunk_threshold(
        &self,
        kv_utilization: f64,
        has_decode_demand: bool,
    ) -> usize {
        let base = if self.config.enable_adaptive_batching {
            self.telemetry.dynamic_prefill_chunk_threshold.max(32)
        } else {
            self.config.chunked_prefill_threshold.max(32)
        };
        let mut threshold = base;

        // Under memory pressure, shrink prefill chunks to avoid large transient spikes.
        if kv_utilization > 0.95 {
            threshold = threshold.min((base / 4).max(32));
        } else if kv_utilization > 0.85 {
            threshold = threshold.min((base / 2).max(64));
        }

        // If decode is already active, avoid over-investing in prefill this step.
        if has_decode_demand {
            threshold = threshold.min((base / 2).max(64));
        }

        // Favor throughput for single-request, low-pressure execution.
        if self.waiting_count() <= 1 && self.running.len() <= 1 && kv_utilization < 0.50 {
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

    /// Try to preempt running requests to free up the required number of blocks.
    /// Only preempts requests with lower priority than the requesting priority.
    /// Returns exact preempted scheduler incarnations.
    fn try_preempt_for_blocks(
        &mut self,
        blocks_needed: usize,
        requesting_priority: Priority,
        protected_requests: &HashSet<RequestId>,
        kv_cache: &mut KVCacheManager,
    ) -> Vec<SessionKey> {
        // A prepared victim must be reconciled with the executor before the
        // scheduler prepares another preemption plan or reuses its blocks.
        if self
            .running
            .values()
            .any(|running| running.preemption_pending)
        {
            return Vec::new();
        }
        // Collect candidates for preemption and score them by expected user impact.
        let mut candidates: Vec<_> =
            self.running
                .iter()
                .filter(|(_, r)| {
                    r.priority < requesting_priority
                        && !r.paused
                        && !r.preemption_pending
                        && !r.first_token_emitted
                        && !r.block_ids.is_empty()
                        && !protected_requests.contains(&r.request_id)
                        && self.requests.get(&r.request_id).is_some_and(|metadata| {
                            metadata.cache_policy.allows_recompute_preemption()
                        })
                })
                .map(|(id, r)| {
                    let (overdue_ms, age_ms, remaining_decode) =
                        if let Some(metadata) = self.requests.get(id) {
                            (
                                self.request_overdue_ms(metadata),
                                metadata.arrival_time.elapsed().as_secs_f64() * 1000.0,
                                metadata.max_tokens.saturating_sub(r.num_tokens_generated),
                            )
                        } else {
                            (0.0, 0.0, usize::MAX)
                        };
                    (
                        id.clone(),
                        r.priority,
                        kv_cache.reclaimable_blocks(id),
                        r.num_tokens_generated,
                        r.first_token_emitted,
                        overdue_ms,
                        age_ms,
                        remaining_decode,
                    )
                })
                .collect();
        candidates.retain(|candidate| candidate.2 > 0);

        // Order by:
        // 1) lowest priority first
        // 2) not overdue first (preserve deadline-sensitive requests)
        // 3) no user-visible token yet first
        // 4) least wasted work first
        // 5) youngest first (fairness guardrail)
        // 6) most remaining decode first
        // 7) biggest block footprint first (frees memory faster)
        candidates.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.5.partial_cmp(&b.5).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.4.cmp(&b.4))
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.6.partial_cmp(&b.6).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| b.7.cmp(&a.7))
                .then_with(|| b.2.cmp(&a.2))
        });

        let mut selected = Vec::new();
        let mut reclaimable = 0usize;
        for candidate in candidates {
            reclaimable = reclaimable.saturating_add(candidate.2);
            selected.push(candidate);
            if reclaimable >= blocks_needed {
                break;
            }
        }
        if reclaimable < blocks_needed {
            debug!(
                "Preemption plan rejected transactionally: reclaimable {} but needed {}",
                reclaimable, blocks_needed
            );
            return Vec::new();
        }

        let mut preempted = Vec::with_capacity(selected.len());
        for (request_id, _priority, _reclaimable, ..) in selected {
            let running_epoch = self
                .running
                .get(&request_id)
                .map(|running| running.sequence_id)
                .expect("selected preemption victim must still be running");
            if let Some(running) = self.running.get_mut(&request_id) {
                running.prefill_in_flight = false;
                running.paused = true;
                running.preemption_pending = true;
            }
            preempted.push(SessionKey::new(request_id, running_epoch));
        }
        preempted
    }

    /// Commit a prepared preemption only after executor cleanup confirms that
    /// the exact session no longer owns physical cache state.
    pub(crate) fn confirm_preemption(
        &mut self,
        session: &SessionKey,
        kv_cache: &mut KVCacheManager,
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
        if running.sequence_id != session.epoch || !running.preemption_pending {
            return false;
        }

        // Profiled requests have no scheduler BlockIds: managed pages or
        // model-owned tensors were already released by the executor before
        // this metadata commit. Unprofiled compatibility callers may still
        // contribute legacy blocks here.
        kv_cache.free_for_preemption(&session.request_id);
        running.block_ids.clear();
        running.num_tokens_processed = 0;
        running.num_tokens_generated = 0;
        running.prefill_complete = false;
        running.prefill_in_flight = false;
        running.first_token_emitted = false;
        running.paused = true;
        running.preemption_pending = false;
        running.preemption_defer_generation = Some(self.schedule_generation.wrapping_add(1));
        metadata.retry_not_before = None;
        true
    }

    /// Fail closed after the executor has already confirmed physical cleanup
    /// but the scheduler can no longer commit the prepared preemption.
    ///
    /// The exact session is terminally quarantined with logical cache released
    /// and its public ID fenced until the outer terminal event is delivered.
    /// This path intentionally does not resume the victim: its physical cache
    /// no longer exists.
    pub(crate) fn quarantine_rejected_confirmed_preemption(
        &mut self,
        session: &SessionKey,
        kv_cache: &mut KVCacheManager,
    ) -> bool {
        if let Some(pending) = self.pending_releases.get(&session.request_id) {
            return pending.session == *session;
        }
        if self
            .requests
            .get(&session.request_id)
            .is_some_and(|metadata| metadata.sequence_id != session.epoch)
        {
            return false;
        }
        let owns_pending_preemption =
            self.running
                .get(&session.request_id)
                .is_some_and(|running| {
                    running.sequence_id == session.epoch && running.preemption_pending
                });
        if !owns_pending_preemption {
            return false;
        }

        self.remove_from_waiting(&session.request_id);
        self.running.remove(&session.request_id);
        self.requests.remove(&session.request_id);
        kv_cache.free(&session.request_id);
        self.pending_releases.insert(
            session.request_id.clone(),
            PendingRelease {
                session: session.clone(),
                cause: TerminalReleaseCause::PreemptionCommitRejected,
                confirmation_required: false,
                cleanup_confirmed: true,
                terminal_delivered: false,
                cleanup_attempts: 0,
                retry_at: Instant::now(),
            },
        );
        true
    }

    pub(crate) fn quarantine_failed_preemption(&mut self, session: &SessionKey) -> bool {
        matches!(
            self.begin_terminal_release(session, TerminalReleaseCause::PreemptionCleanupFailed),
            BeginTerminalRelease::Started { .. } | BeginTerminalRelease::AlreadyPending { .. }
        )
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

    fn tiny_preemption_scheduler() -> (Scheduler, KVCacheManager) {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::Priority,
            enable_chunked_prefill: false,
            enable_preemption: true,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let scheduler = Scheduler::new(config);
        let kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 1,
            block_size: 2,
            ..Default::default()
        });
        (scheduler, kv_cache)
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

    fn prepare_exact_session_preemption(scheduler: &mut Scheduler, request_id: &str) -> SessionKey {
        scheduler
            .prepare_preemption_for_test(&request_id.to_string())
            .expect("running preemption fixture")
    }

    fn allow_external_prefix_reuse(scheduler: &mut Scheduler, request_id: &str) {
        let epoch = scheduler
            .get_sequence_id(&request_id.to_string())
            .expect("request epoch");
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.prefix_reuse_safe = true;
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 16,
            block_size: 16,
            ..Default::default()
        });
        let request_id = "reused-id".to_string();

        scheduler.add_request(&build_request(
            TaskType::Chat,
            &request_id,
            Priority::Normal,
        ));
        let first = scheduler.schedule(&mut kv_cache).prefill_requests.remove(0);
        let first_session = first.session_key();
        scheduler.finish_request(&request_id, &mut kv_cache);

        scheduler.add_request(&build_request(
            TaskType::Chat,
            &request_id,
            Priority::Normal,
        ));
        let second = scheduler.schedule(&mut kv_cache).prefill_requests.remove(0);
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 16,
            block_size: 4,
            ..Default::default()
        });
        let request_id = "recompute-current-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::High)
            .with_workload_class(WorkloadClass::Batch)
            .with_deadline(Some(Instant::now() + Duration::from_secs(30)));
        request.prompt_tokens = vec![1, 2, 3, 4];
        scheduler.add_request(&request);
        allow_recompute(&mut scheduler, &request_id);

        let scheduled = scheduler.schedule(&mut kv_cache);
        let session = scheduled.prefill_requests[0].session_key();
        assert!(scheduled.prefill_requests[0].block_ids.is_empty());
        scheduler.update_after_step(&request_id, 4, 2, Vec::new(), 1.0);
        let metadata_before = scheduler.requests[&request_id].clone();
        assert!(kv_cache.get_block_table(&request_id).is_none());

        assert!(scheduler.restart_request_for_recompute(&session, &mut kv_cache));

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
        assert_eq!(
            metadata_after.cache_policy.prefix_reuse_safe,
            metadata_before.cache_policy.prefix_reuse_safe
        );

        let running = &scheduler.running[&request_id];
        assert_eq!(running.sequence_id, session.epoch);
        assert_eq!(running.num_tokens_processed, 0);
        assert_eq!(running.num_tokens_generated, 0);
        assert!(running.block_ids.is_empty());
        assert!(!running.prefill_complete);
        assert!(!running.prefill_in_flight);
        assert!(!running.first_token_emitted);
        assert!(running.paused);
        assert!(kv_cache.get_block_table(&request_id).is_none());

        let restarted = scheduler.schedule(&mut kv_cache);
        assert_eq!(restarted.prefill_requests.len(), 1);
        assert_eq!(restarted.prefill_requests[0].session_key(), session);
        assert_eq!(restarted.prefill_requests[0].num_computed_tokens, 0);
        assert!(restarted.prefill_requests[0].block_ids.is_empty());
    }

    #[test]
    fn recompute_restart_rejects_stale_session_without_mutation() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 16,
            block_size: 4,
            ..Default::default()
        });
        let request_id = "recompute-stale-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.prompt_tokens = vec![1, 2, 3, 4];
        scheduler.add_request(&request);

        let scheduled = scheduler.schedule(&mut kv_cache);
        let current = scheduled.prefill_requests[0].session_key();
        let stale = SessionKey::new(request_id.clone(), current.epoch.saturating_add(1));
        let blocks_before = kv_cache
            .get_block_table(&request_id)
            .expect("scheduled request block table")
            .to_vec();

        assert!(!scheduler.restart_request_for_recompute(
            &SessionKey::new("missing-recompute-session".to_string(), current.epoch),
            &mut kv_cache,
        ));
        assert!(!scheduler.restart_request_for_recompute(&stale, &mut kv_cache));

        let running = &scheduler.running[&request_id];
        assert_eq!(running.sequence_id, current.epoch);
        assert_eq!(running.block_ids, blocks_before);
        assert!(!running.prefill_complete);
        assert!(running.prefill_in_flight);
        assert!(!running.paused);
        assert_eq!(
            kv_cache.get_block_table(&request_id),
            Some(blocks_before.as_slice())
        );
    }

    #[test]
    fn retry_release_makes_the_current_prefill_quantum_eligible_again() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 16,
            block_size: 4,
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

        let first = scheduler.schedule(&mut kv_cache);
        let session = first.prefill_requests[0].session_key();
        assert_eq!(first.prefill_requests[0].num_tokens, 8);
        let blocks_before = first.prefill_requests[0].block_ids.clone();
        assert!(scheduler.running[&request_id].prefill_in_flight);

        assert!(scheduler.release_execution_quantum_for_retry(&session));

        let running = &scheduler.running[&request_id];
        assert!(!running.prefill_in_flight);
        assert!(!running.prefill_complete);
        assert_eq!(running.num_tokens_processed, 0);
        assert_eq!(running.num_tokens_generated, 0);
        assert_eq!(running.block_ids, blocks_before);
        assert!(!running.paused);

        let retry = scheduler.schedule(&mut kv_cache);
        assert_eq!(retry.prefill_requests.len(), 1);
        assert_eq!(retry.prefill_requests[0].session_key(), session);
        assert_eq!(retry.prefill_requests[0].num_computed_tokens, 0);
        assert_eq!(retry.prefill_requests[0].num_tokens, 8);
    }

    #[test]
    fn retry_release_rejects_a_stale_session_without_mutation() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 4,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 16,
            block_size: 4,
            ..Default::default()
        });
        let request_id = "retry-stale-session".to_string();
        let mut request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        request.prompt_tokens = vec![1; 8];
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        let current = first.prefill_requests[0].session_key();
        let blocks_before = first.prefill_requests[0].block_ids.clone();
        let stale = SessionKey::new(request_id.clone(), current.epoch.saturating_add(1));

        assert!(!scheduler.release_execution_quantum_for_retry(&stale));

        let running = &scheduler.running[&request_id];
        assert!(running.prefill_in_flight);
        assert_eq!(running.num_tokens_processed, 0);
        assert_eq!(running.num_tokens_generated, 0);
        assert_eq!(running.block_ids, blocks_before);
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
        let mut kv_cache = KVCacheManager::new(Default::default());
        let request = build_request(TaskType::Chat, "deferred-retry", Priority::Normal);
        assert!(scheduler.add_request(&request));
        let first = scheduler.schedule(&mut kv_cache).prefill_requests.remove(0);
        let session = first.session_key();

        assert!(scheduler.defer_execution_retry(&session, Instant::now() + Duration::from_secs(1),));
        assert!(!scheduler.schedule(&mut kv_cache).has_execution_work());

        assert!(scheduler.defer_execution_retry(&session, Instant::now()));
        let retry = scheduler.schedule(&mut kv_cache);
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 128,
            block_size: 16,
            ..Default::default()
        });

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

        let scheduled = scheduler.schedule(&mut kv_cache);
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 128,
            block_size: 16,
            ..Default::default()
        });

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

        let scheduled = scheduler.schedule(&mut kv_cache);
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
            let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
                max_blocks: 128,
                block_size: 16,
                ..Default::default()
            });

            let mut request = EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "continue".to_string(),
            }])
            .with_workload_class(workload_class);
            request.id = format!("decode-{}", workload_class.as_str());
            request.prompt_tokens = vec![1];
            request.params.max_tokens = 16;

            scheduler.add_request(&request);
            let first = scheduler.schedule(&mut kv_cache);
            assert_eq!(first.prefill_requests.len(), 1);
            scheduler.update_after_step(&request.id, 1, 1, Vec::new(), 1.0);

            let second = scheduler.schedule(&mut kv_cache);
            assert_eq!(second.decode_requests.len(), 1);
            second.decode_requests[0].num_tokens
        }

        assert_eq!(schedule_decode_for(WorkloadClass::Realtime), 1);
        assert_eq!(schedule_decode_for(WorkloadClass::Streaming), 1);
        assert_eq!(schedule_decode_for(WorkloadClass::Batch), 4);
    }

    #[test]
    fn test_preemption_requeue_across_task_types() {
        let task_types = [
            TaskType::TTS,
            TaskType::ASR,
            TaskType::Chat,
            TaskType::SpeechToSpeech,
        ];

        for task_type in task_types {
            let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
            let low_id = format!("low-{task_type:?}");
            let high_id = format!("high-{task_type:?}");
            let low = build_request(task_type, &low_id, Priority::Low);
            scheduler.add_request(&low);
            allow_recompute(&mut scheduler, &low_id);

            let first = scheduler.schedule(&mut kv_cache);
            assert_eq!(
                first.prefill_requests.len(),
                1,
                "expected initial prefill for {task_type:?}"
            );
            assert_eq!(first.prefill_requests[0].request_id, low_id);
            assert!(first.prefill_requests[0].block_ids.is_empty());
            scheduler.update_after_step(&low_id, 1, 0, Vec::new(), 1.0);

            let high = build_request(task_type, &high_id, Priority::High);
            scheduler.add_request(&high);
            allow_recompute(&mut scheduler, &high_id);
            let preempted = prepare_exact_session_preemption(&mut scheduler, &low_id);
            assert_eq!(
                scheduler.get_status(&low_id),
                Some(RequestStatus::Running),
                "preempted {task_type:?} request should remain tracked for resume"
            );

            // With no fake logical capacity dependency, the higher-priority
            // request can be admitted while physical cleanup is confirmed.
            let second = scheduler.schedule(&mut kv_cache);
            assert_eq!(
                second.prefill_requests.len(),
                1,
                "high-priority {task_type:?} request should not wait on fake blocks"
            );
            assert_eq!(second.prefill_requests[0].request_id, high_id);
            assert!(second.prefill_requests[0].block_ids.is_empty());
            assert!(scheduler.confirm_preemption(&preempted, &mut kv_cache));
            scheduler.update_after_step(&high_id, 1, 0, Vec::new(), 1.0);
            scheduler.finish_request(&high_id, &mut kv_cache);

            // The confirmation installs a one-cycle resume fence so the work
            // that caused preemption retains its committed scheduling turn.
            assert!(!scheduler.schedule(&mut kv_cache).has_execution_work());
            let fourth = scheduler.schedule(&mut kv_cache);
            assert_eq!(
                fourth.prefill_requests.len(),
                1,
                "preempted {task_type:?} request should restart from prefill"
            );
            assert_eq!(fourth.prefill_requests[0].request_id, low_id);
            assert!(fourth.prefill_requests[0].is_prefill);
            assert_eq!(fourth.prefill_requests[0].num_computed_tokens, 0);
        }
    }

    #[test]
    fn preemption_requires_exact_cleanup_confirmation_without_logical_blocks() {
        let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
        let low = build_request(TaskType::Chat, "preempt-low", Priority::Low);
        assert!(scheduler.add_request(&low));
        allow_recompute(&mut scheduler, &low.id);
        let initial = scheduler.schedule(&mut kv_cache);
        assert_eq!(initial.prefill_requests.len(), 1);
        assert!(initial.prefill_requests[0].block_ids.is_empty());
        scheduler.update_after_step(&low.id, 1, 0, Vec::new(), 1.0);
        let victim = prepare_exact_session_preemption(&mut scheduler, &low.id);
        assert_eq!(kv_cache.stats().allocated_blocks, 0);

        let stale = SessionKey::new(victim.request_id.clone(), victim.epoch + 1);
        assert!(!scheduler.confirm_preemption(&stale, &mut kv_cache));
        assert!(scheduler.running[&low.id].preemption_pending);
        assert_eq!(scheduler.running[&low.id].num_tokens_processed, 1);

        assert!(scheduler.confirm_preemption(&victim, &mut kv_cache));
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        let running = &scheduler.running[&low.id];
        assert!(!running.preemption_pending);
        assert_eq!(running.num_tokens_processed, 0);
        assert!(running.block_ids.is_empty());
    }

    #[test]
    fn rejected_confirmed_preemption_terminally_quarantines_the_victim() {
        let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
        let low = build_request(TaskType::Chat, "rejected-preempt", Priority::Low);
        assert!(scheduler.add_request(&low));
        allow_recompute(&mut scheduler, &low.id);
        let initial = scheduler.schedule(&mut kv_cache);
        assert_eq!(initial.prefill_requests.len(), 1);
        assert!(initial.prefill_requests[0].block_ids.is_empty());
        scheduler.update_after_step(&low.id, 1, 0, Vec::new(), 1.0);
        let victim = prepare_exact_session_preemption(&mut scheduler, &low.id);
        assert!(scheduler
            .running
            .get(&victim.request_id)
            .is_some_and(|running| running.preemption_pending));

        // Inject the only state split that can reject a prepared exact-session
        // commit after the executor has already released physical cache.
        scheduler.requests.remove(&victim.request_id);
        assert!(!scheduler.confirm_preemption(&victim, &mut kv_cache));
        assert!(scheduler.quarantine_rejected_confirmed_preemption(&victim, &mut kv_cache));

        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        assert!(!scheduler.running.contains_key(&victim.request_id));
        assert_eq!(
            scheduler.pending_release_confirmation_required(&victim),
            Some(false)
        );
        assert!(!scheduler.add_request(&low));
        assert!(scheduler.mark_terminal_delivered(&victim));
        assert!(scheduler.add_request(&low));
    }

    #[test]
    fn preemption_is_transactional_when_victims_cannot_satisfy_demand() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            policy: SchedulingPolicy::Priority,
            enable_preemption: true,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 2,
            block_size: 1,
            ..Default::default()
        });
        for id in ["low-a", "low-b"] {
            let mut request = EngineCoreRequest::tts(id).with_priority(Priority::Low);
            request.id = id.to_string();
            request.prompt_tokens = vec![1];
            scheduler.add_request(&request);
        }
        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 2);
        for id in ["low-a", "low-b"] {
            scheduler.update_after_step(&id.to_string(), 1, 0, Vec::new(), 1.0);
        }
        let mut high = EngineCoreRequest::tts("high").with_priority(Priority::High);
        high.id = "high".to_string();
        high.prompt_tokens = vec![2, 2, 2];
        scheduler.add_request(&high);

        let second = scheduler.schedule(&mut kv_cache);
        assert!(second.preempted_requests.is_empty());
        assert_eq!(kv_cache.stats().allocated_blocks, 2);
        assert!(kv_cache.get_block_table(&"low-a".to_string()).is_some());
        assert!(kv_cache.get_block_table(&"low-b".to_string()).is_some());
    }

    #[test]
    fn preemption_never_discards_user_visible_output() {
        let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
        let low_id = "visible-low".to_string();
        let mut low = EngineCoreRequest::tts("low").with_priority(Priority::Low);
        low.id = low_id.clone();
        low.prompt_tokens = vec![1];
        scheduler.add_request(&low);
        assert_eq!(scheduler.schedule(&mut kv_cache).prefill_requests.len(), 1);
        scheduler.update_after_step(&low_id, 1, 1, Vec::new(), 1.0);

        let mut high = EngineCoreRequest::tts("high").with_priority(Priority::High);
        high.id = "visible-high".to_string();
        high.prompt_tokens = vec![2];
        scheduler.add_request(&high);
        let result = scheduler.schedule(&mut kv_cache);

        assert!(result.preempted_requests.is_empty());
        assert!(kv_cache.get_block_table(&low_id).is_some());
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
            let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
            let request_id = format!("abort-{task_type:?}");
            let request = build_request(task_type, &request_id, Priority::Normal);
            scheduler.add_request(&request);

            let scheduled = scheduler.schedule(&mut kv_cache);
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
                scheduler.abort_request(&request_id, &mut kv_cache),
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

            let after_abort = scheduler.schedule(&mut kv_cache);
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 32,
            block_size: 16,
            ..Default::default()
        });

        let request_id = "tts-auto-max".to_string();
        let mut request = EngineCoreRequest::tts("hello world");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 0;
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].request_id, request_id);
        scheduler.update_after_step(&request_id, 1, 1, Vec::new(), 1.0);

        let second = scheduler.schedule(&mut kv_cache);
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 32,
            block_size: 16,
            ..Default::default()
        });

        let request_id = "chat-zero-max".to_string();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello world".to_string(),
        }]);
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 0;
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].request_id, request_id);
        scheduler.update_after_step(&request_id, 1, 1, Vec::new(), 1.0);

        let second = scheduler.schedule(&mut kv_cache);
        assert_eq!(
            second.decode_requests.len(),
            1,
            "Non-TTS zero max_tokens should be normalized to a safe decode budget"
        );
        assert_eq!(second.decode_requests[0].request_id, request_id);
    }

    #[test]
    fn test_prefill_chunk_backoff_when_kv_is_tight() {
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 1,
            block_size: 4,
            ..Default::default()
        });

        let request_id = "backoff-prefill".to_string();
        let mut request = EngineCoreRequest::tts("long prompt");
        request.id = request_id.clone();
        request.prompt_tokens = vec![7; 8];
        scheduler.add_request(&request);

        let scheduled = scheduler.schedule(&mut kv_cache);
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].request_id, request_id);
        assert_eq!(
            scheduled.prefill_requests[0].num_tokens, 4,
            "Scheduler should back off prefill chunk to fit KV capacity"
        );
    }

    #[test]
    fn incremental_prefill_continues_after_committed_partial_step() {
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 8,
            block_size: 4,
            ..Default::default()
        });

        let request_id = "incremental-prefill".to_string();
        let mut request = EngineCoreRequest::tts("incremental prompt");
        request.id = request_id.clone();
        request.prompt_tokens = vec![7; 8];
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].num_tokens, 4);
        assert_eq!(first.prefill_requests[0].num_computed_tokens, 0);

        let duplicate = scheduler.schedule(&mut kv_cache);
        assert!(
            !duplicate.has_work(),
            "an in-flight prefill quantum must not be scheduled twice"
        );

        scheduler.update_after_step(&request_id, 4, 0, Vec::new(), 1.0);

        let second = scheduler.schedule(&mut kv_cache);
        assert!(second.preempted_requests.is_empty());
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
            }
        );
    }

    #[test]
    fn full_prefill_profile_schedules_the_entire_prompt_without_logical_cache() {
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 8,
            block_size: 4,
            ..Default::default()
        });

        let request_id = "full-prefill".to_string();
        let mut request = EngineCoreRequest::tts("full prompt");
        request.id = request_id.clone();
        request.prompt_tokens = vec![7; 8];
        scheduler.add_request(&request);

        let epoch = scheduler
            .get_sequence_id(&request_id)
            .expect("request epoch");
        let session = SessionKey::new(request_id.clone(), epoch);
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Full;
        assert!(scheduler.update_execution_profile(&session, &profile));

        let scheduled = scheduler.schedule(&mut kv_cache);

        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].session_key(), session);
        assert_eq!(scheduled.prefill_requests[0].num_tokens, 8);
        assert!(scheduled.prefill_requests[0].block_ids.is_empty());
        assert_eq!(scheduled.blocks_allocated, 0);
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        assert_eq!(scheduled.total_tokens, 8);
        assert_eq!(
            scheduled.prefill_requests[0].work,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 8 },
                max_output_steps: 8,
            }
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 16,
            block_size: 4,
            ..Default::default()
        });
        let request_id = "continuous-decode".to_string();
        let request = build_request(TaskType::Chat, &request_id, Priority::Normal);
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request_id).unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.decode_batch = NativeBatchMode::Continuous;
        assert!(scheduler
            .update_execution_profile(&SessionKey::new(request_id.clone(), epoch), &profile,));

        let prefill = scheduler.schedule(&mut kv_cache);
        assert_eq!(prefill.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, request.num_prompt_tokens(), 1, Vec::new(), 1.0);
        let decode = scheduler.schedule(&mut kv_cache);

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
            }
        );
    }

    #[test]
    fn test_prefill_defer_skips_oversized_head_request() {
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 2,
            block_size: 4,
            ..Default::default()
        });

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

        let scheduled = scheduler.schedule(&mut kv_cache);
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(
            scheduled.prefill_requests[0].request_id, small_id,
            "Oversized head request should be deferred instead of blocking all admissions"
        );
        assert_eq!(scheduler.get_status(&big_id), Some(RequestStatus::Waiting));
    }

    #[test]
    fn external_paged_decode_never_allocates_legacy_blocks() {
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            enable_adaptive_batching: false,
            ..Default::default()
        });
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 1,
            block_size: 1,
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

        let prefill = scheduler.schedule(&mut kv_cache);
        assert_eq!(prefill.prefill_requests.len(), 1);
        assert!(prefill.prefill_requests[0].block_ids.is_empty());
        scheduler.update_after_step(&request_id, 1, 0, Vec::new(), 1.0);

        let decode = scheduler.schedule(&mut kv_cache);
        assert_eq!(decode.decode_requests.len(), 1);
        assert!(decode.decode_requests[0].block_ids.is_empty());
        assert_eq!(decode.blocks_allocated, 0);
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        assert!(kv_cache.get_block_table(&request_id).is_none());
    }

    #[test]
    fn opaque_and_cacheless_sequences_never_allocate_legacy_blocks() {
        for (suffix, cache_mode) in [
            ("opaque", CacheMode::OpaqueModelOwned),
            ("cacheless", CacheMode::None),
        ] {
            let mut scheduler = Scheduler::new(SchedulerConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                enable_adaptive_batching: false,
                ..Default::default()
            });
            let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
                max_blocks: 1,
                block_size: 1,
                ..Default::default()
            });
            let request_id = format!("{suffix}-no-logical-kv");
            let request = build_request(TaskType::Chat, &request_id, Priority::Normal);
            assert!(scheduler.add_request(&request));
            let epoch = scheduler.get_sequence_id(&request_id).expect("epoch");
            let session = SessionKey::new(request_id.clone(), epoch);
            let mut profile =
                ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
            profile.cache_mode = cache_mode;
            assert!(scheduler.update_execution_profile(&session, &profile));

            let prefill = scheduler.schedule(&mut kv_cache);
            assert_eq!(prefill.prefill_requests.len(), 1);
            assert!(prefill.prefill_requests[0].block_ids.is_empty());
            scheduler.update_after_step(&request_id, 1, 0, Vec::new(), 1.0);

            let decode = scheduler.schedule(&mut kv_cache);
            assert_eq!(decode.decode_requests.len(), 1);
            assert!(decode.decode_requests[0].block_ids.is_empty());
            assert_eq!(kv_cache.stats().allocated_blocks, 0);
            assert!(kv_cache.get_block_table(&request_id).is_none());
        }
    }

    #[test]
    fn external_paged_prefill_does_not_consume_legacy_prefix_blocks() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 32,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_prefix_caching: true,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 2,
            block_size: 2,
            ..Default::default()
        });

        let req1_id = "prefix-source".to_string();
        let mut req1 = EngineCoreRequest::tts("prefix source");
        req1.id = req1_id.clone();
        req1.prompt_tokens = vec![10, 11, 12, 13];
        scheduler.add_request(&req1);
        allow_external_prefix_reuse(&mut scheduler, &req1_id);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        assert!(first.prefill_requests[0].block_ids.is_empty());
        assert_eq!(first.blocks_allocated, 0);
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        scheduler.update_after_step(&req1_id, 4, 0, Vec::new(), 1.0);

        let req2_id = "prefix-reuser".to_string();
        let mut req2 = EngineCoreRequest::tts("prefix reuser");
        req2.id = req2_id.clone();
        req2.prompt_tokens = vec![10, 11, 12, 13];
        scheduler.add_request(&req2);
        allow_external_prefix_reuse(&mut scheduler, &req2_id);

        let second = scheduler.schedule(&mut kv_cache);
        assert!(
            second
                .prefill_requests
                .iter()
                .any(|entry| entry.request_id == req2_id),
            "managed prefill admission must not depend on legacy KV capacity"
        );
        assert!(second
            .prefill_requests
            .iter()
            .find(|entry| entry.request_id == req2_id)
            .expect("external prefill")
            .block_ids
            .is_empty());
        assert_eq!(
            second.blocks_allocated, 0,
            "managed prefix policy must not allocate legacy KV blocks"
        );
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 32,
            block_size: 8,
            ..Default::default()
        });

        let request_id = "decode-quanta".to_string();
        let mut request = EngineCoreRequest::tts("hello");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        scheduler.update_after_step(&request_id, 1, 1, Vec::new(), 1.0);

        let second = scheduler.schedule(&mut kv_cache);
        assert_eq!(second.decode_requests.len(), 1);
        assert_eq!(
            second.decode_requests[0].num_tokens, 4,
            "decode quanta should grant multi-token decode when queue pressure is low"
        );
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
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 32,
            block_size: 8,
            ..Default::default()
        });

        for request_id in ["decode-a", "decode-b"] {
            let mut request = EngineCoreRequest::tts("hello");
            request.id = request_id.to_string();
            request.prompt_tokens = vec![1];
            scheduler.add_request(&request);
        }

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 2);
        for request_id in ["decode-a", "decode-b"] {
            scheduler.update_after_step(&request_id.to_string(), 1, 1, Vec::new(), 1.0);
        }

        let second = scheduler.schedule(&mut kv_cache);
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
        let mut kv_cache = KVCacheManager::new(Default::default());

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
            scheduler.abort_request(&format!("request-{index:03}"), &mut kv_cache);
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
        let mut kv_cache = KVCacheManager::new(Default::default());
        for id in ["expired-a", "expired-b"] {
            let mut request = EngineCoreRequest::tts(id)
                .with_deadline(Some(Instant::now() - Duration::from_millis(1)));
            request.id = id.to_string();
            scheduler.add_request(&request);
        }

        let result = scheduler.schedule(&mut kv_cache);

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
        let mut kv_cache = KVCacheManager::new(Default::default());
        let mut request = EngineCoreRequest::tts("soft-sla");
        request.id = "soft-sla".to_string();
        request.prompt_tokens = vec![1];
        scheduler.add_request(&request);
        scheduler.requests.get_mut(&request.id).unwrap().deadline_at =
            Instant::now() - Duration::from_secs(1);

        let result = scheduler.schedule(&mut kv_cache);

        assert!(result.expired_requests.is_empty());
        assert_eq!(result.prefill_requests.len(), 1);
    }

    #[test]
    fn expiring_running_request_waits_for_exact_cleanup_before_releasing_blocks() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut kv_cache = KVCacheManager::new(Default::default());
        let mut request = EngineCoreRequest::tts("running-deadline");
        request.id = "running-deadline".to_string();
        request.prompt_tokens = vec![1];
        scheduler.add_request(&request);
        assert_eq!(scheduler.schedule(&mut kv_cache).prefill_requests.len(), 1);
        assert!(kv_cache.stats().allocated_blocks > 0);
        scheduler
            .requests
            .get_mut(&request.id)
            .unwrap()
            .hard_deadline = Some(Instant::now() - Duration::from_millis(1));

        let result = scheduler.schedule(&mut kv_cache);

        assert_eq!(result.expired_requests.len(), 1);
        assert!(kv_cache.stats().allocated_blocks > 0);
        assert_eq!(scheduler.running_count(), 0);
        let session = result.expired_requests[0].session_key();
        assert_eq!(
            scheduler.pending_expired_cleanup_sessions(),
            vec![session.clone()]
        );
        let stale = SessionKey::new(session.request_id.clone(), session.epoch + 1);
        assert!(!scheduler.confirm_expired_session_cleanup(&stale, &mut kv_cache));
        assert!(kv_cache.stats().allocated_blocks > 0);
        assert!(!scheduler.add_request(&request));

        assert!(scheduler.confirm_expired_session_cleanup(&session, &mut kv_cache));
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        assert!(scheduler.pending_expired_cleanup_sessions().is_empty());
        assert!(scheduler.add_request(&request));
    }

    #[test]
    fn terminal_release_requires_exact_cleanup_and_delivery_before_id_reuse() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut kv_cache = KVCacheManager::new(Default::default());
        let request = build_request(TaskType::Chat, "terminal-fence", Priority::Normal);
        assert!(scheduler.add_request(&request));
        let epoch = scheduler.get_sequence_id(&request.id).unwrap();
        let session = SessionKey::new(request.id.clone(), epoch);
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_release_safe = true;
        assert!(scheduler.update_execution_profile(&session, &profile));
        let scheduled = scheduler.schedule(&mut kv_cache);
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert!(scheduled.prefill_requests[0].block_ids.is_empty());
        let allocated = kv_cache.stats().allocated_blocks;
        assert_eq!(allocated, 0);

        assert_eq!(
            scheduler.begin_terminal_release(&session, TerminalReleaseCause::Completed),
            BeginTerminalRelease::Started {
                confirmation_required: true,
            }
        );
        assert!(!scheduler.add_request(&request));
        let stale = SessionKey::new(request.id.clone(), session.epoch + 1);
        assert!(!scheduler.confirm_session_release(&stale, &mut kv_cache));
        assert!(!scheduler.mark_terminal_delivered(&stale));
        assert_eq!(kv_cache.stats().allocated_blocks, 0);

        assert!(scheduler.mark_terminal_delivered(&session));
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
        assert!(!scheduler.add_request(&request));
        assert!(scheduler.confirm_session_release(&session, &mut kv_cache));
        assert_eq!(kv_cache.stats().allocated_blocks, 0);
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
        let mut kv_cache = KVCacheManager::new(Default::default());
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
            let result = scheduler.schedule(&mut kv_cache);
            let request_id = result.prefill_requests[0].request_id.clone();
            selected.push(request_id.clone());
            scheduler.finish_request(&request_id, &mut kv_cache);
        }

        assert!(selected.iter().any(|id| id == "background"));
    }

    #[test]
    fn production_defaults_only_enable_physically_enforced_features() {
        let config = SchedulerConfig::default();
        assert_eq!(config.policy, SchedulingPolicy::FCFS);
        assert!(!config.enable_chunked_prefill);
        assert!(!config.enable_preemption);
        assert!(!config.enable_adaptive_batching);
        assert!(!config.enable_power_adaptive);
        assert!(!config.enable_decode_quanta);
        assert!(!config.enable_kv_tiering);
    }
}
