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
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::cache::managed::{ManagedKvCacheManager, ManagedStateCapacityRequest};
use super::cache::physical::{InvocationPhysicalKey, PhysicalStateManager};
use super::config::EngineCoreConfig;
#[cfg(test)]
use super::execution::CacheMode;
use super::execution::{
    AdapterAbiRevision, AdapterInstanceId, BatchBudget, BatchDispatch, BatchId, BatchKey,
    BatchLaneKey, ConcurrencyClass, DeadlinePhase, DispatchState, ExecutionDisposition,
    ExecutionFailure, ExecutionGroupId, ExecutionMode, ExecutionPlan, ExecutionProfile,
    ExecutionReport, ExecutionState, ExecutionTracker, FailureOrigin,
    FinishReason as ExecutionFinishReason, ModelInstanceId, NativeBatchMode, OutcomeProvenance,
    OutputVisibility, PhysicalBatch, PrefillMode, ReadyQuantum, RetryDisposition, StageId,
    StageShapePolicy, WorkCost, WorkUnit,
};
use super::execution_group::{
    ExecutedEngineStep, ExecutionGroupRunner, ExecutionPhase, PreparedEngineStep,
    PreparedExecutionBatch,
};
use super::executor::REQUEST_DEADLINE_EXCEEDED;
use super::executor::{
    deliver_committed_streams, CacheReleaseReport, CommittedStreamDelivery, ExecutorOutput,
    ExecutorStepResult, IncrementalStreamDeliveryWorkers, StreamDeliveryFailure,
    StreamDeliveryFailureKind, UnifiedExecutor, WorkerConfig,
};
use super::metrics::{
    record_engine_execution_outcome, record_engine_physical_batch,
    record_engine_stream_checkpoint_committed, record_engine_stream_checkpoint_rejection,
};
use super::output::OutputProcessor;
use super::request::{
    EngineCoreRequest, FencedStreamProgress, RequestStatus, StreamProgressBudget,
    STREAM_PROGRESS_MAX_BUFFERED_BYTES, STREAM_PROGRESS_QUEUE_CAPACITY,
};
use super::scheduler::{BeginTerminalRelease, Scheduler, SchedulerConfig, TerminalReleaseCause};
use super::types::{
    AudioOutput, EngineOutput, FinishReason as OutputFinishReason, LatencyBreakdown, RequestId,
};
use super::{ResourceAmount, ResourceVector};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

#[derive(Debug, Clone, Default)]
struct RequestPhaseTiming {
    first_scheduled_at: Option<Instant>,
    queue_wait_ms: f64,
    media_decode_ms: Option<f64>,
    normalization_ms: Option<f64>,
    prefill_ms: f64,
    decode_ms: f64,
    sampling_ms: Option<f64>,
    codec_ms: Option<f64>,
    postprocess_ms: Option<f64>,
    first_output_ms: Option<f64>,
    prefill_steps: u32,
    decode_steps: u32,
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
}

#[derive(Debug)]
struct ActiveStreamBatch {
    lane: BatchLaneKey,
    output_visibility: OutputVisibility,
    rows: HashMap<u64, super::SessionKey>,
}

fn resolve_managed_token_reach(
    backend: BackendKind,
    configured_context: usize,
    loaded_context: Option<usize>,
) -> Result<Option<u64>> {
    let Some(loaded_context) = loaded_context else {
        if backend == BackendKind::Cuda {
            return Err(Error::ModelLoadError(
                "CUDA managed state requires a positive loaded-model context limit".into(),
            ));
        }
        return Ok(None);
    };
    let effective = loaded_context.min(configured_context);
    if effective == 0 {
        let backend = if backend == BackendKind::Cuda {
            "CUDA"
        } else {
            "portable"
        };
        return Err(Error::ModelLoadError(format!(
            "{backend} managed state resolved a zero loaded-model context limit"
        )));
    }
    u64::try_from(effective).map(Some).map_err(|_| {
        let backend = if backend == BackendKind::Cuda {
            "CUDA"
        } else {
            "portable"
        };
        Error::ModelLoadError(format!("{backend} model context exceeds u64"))
    })
}

impl PhysicalBatchAssembly {
    fn materialized_tensor_elements(
        shape_policy: StageShapePolicy,
        rows: &[ReadyQuantum],
    ) -> Option<u64> {
        if shape_policy == StageShapePolicy::Padded {
            let maximum = rows.iter().map(|row| row.cost.tensor_elements).max()?;
            maximum.checked_mul(u64::try_from(rows.len()).ok()?)
        } else {
            rows.iter().try_fold(0u64, |total, row| {
                total.checked_add(row.cost.tensor_elements)
            })
        }
    }

    fn workspace_resources(
        backend: BackendKind,
        base_bytes: u64,
        rows: &[ReadyQuantum],
    ) -> Option<ResourceVector> {
        let generic = rows.iter().try_fold(
            ResourceVector {
                temporary_bytes: ResourceAmount::Known(base_bytes),
                ..ResourceVector::zero()
            },
            |total, row| total.checked_add(row.cost.workspace).ok(),
        )?;
        if generic.kv_bytes != ResourceAmount::Known(0)
            || generic.compute_slots != ResourceAmount::Known(0)
        {
            return None;
        }
        let known = |amount| match amount {
            ResourceAmount::Known(value) => Some(value),
            ResourceAmount::Unknown => None,
        };
        let host = known(generic.host_bytes)?;
        let accelerator = known(generic.device_bytes)?
            .checked_add(known(generic.unified_bytes)?)?
            .checked_add(known(generic.temporary_bytes)?)?;
        let mut workspace = ResourceVector::zero();
        match backend {
            BackendKind::Cpu => {
                workspace.host_bytes = ResourceAmount::Known(host.checked_add(accelerator)?)
            }
            BackendKind::Metal => {
                workspace.unified_bytes = ResourceAmount::Known(host.checked_add(accelerator)?)
            }
            BackendKind::Cuda => {
                workspace.host_bytes = ResourceAmount::Known(host);
                workspace.device_bytes = ResourceAmount::Known(accelerator);
            }
        }
        Some(workspace)
    }

    fn try_push(
        &mut self,
        request: Arc<EngineCoreRequest>,
        scheduled: super::scheduler::ScheduledRequest,
        row: ReadyQuantum,
    ) -> bool {
        let mut candidate = self.physical_batch.clone();
        candidate.rows.push(row);
        let Some(materialized) =
            Self::materialized_tensor_elements(self.shape_policy, &candidate.rows)
        else {
            return false;
        };
        let Some(workspace) = Self::workspace_resources(
            candidate.lane.backend,
            self.workspace_base_bytes,
            &candidate.rows,
        ) else {
            return false;
        };
        candidate.materialized_tensor_elements = materialized;
        candidate.workspace = workspace;
        if candidate.validate().is_err() {
            return false;
        }
        self.physical_batch = candidate;
        self.requests.push(request);
        self.scheduled.push(scheduled);
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct LifecycleRetryPolicy {
    max_execution_retries: u32,
    execution_backoff_base: Duration,
    execution_backoff_max: Duration,
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

fn merge_optional_phase_ms(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        let value = value.max(0.0);
        *target = Some(target.unwrap_or(0.0) + value);
    }
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
    /// Model executor
    executor: UnifiedExecutor,
    /// Output processor
    output_processor: OutputProcessor,
    /// Active requests (by ID)
    requests: HashMap<RequestId, Arc<EngineCoreRequest>>,
    /// Request start times (for timing)
    request_start_times: HashMap<RequestId, Instant>,
    /// Per-request phase timing accumulated by scheduler steps.
    request_phase_timings: HashMap<RequestId, RequestPhaseTiming>,
    /// Per-session lifecycle and active-plan fence.
    execution_trackers: HashMap<RequestId, ExecutionTracker>,
    /// Plans prepared under the core lock and awaiting one validated result.
    active_plans: HashMap<u64, ExecutionPlan>,
    /// Prepared physical KV transactions keyed by their exact execution plan.
    active_managed_cache: HashMap<u64, super::ManagedCacheReservation>,
    /// Terminal sessions whose table release waits for an in-flight row.
    pending_managed_releases: HashSet<super::SessionKey>,
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
    retry_policy: LifecycleRetryPolicy,
    /// Whether the engine has been initialized
    initialized: bool,
    /// Step counter for periodic cache housekeeping.
    /// Monotonic identity for physical dispatch envelopes.
    next_batch_id: u64,
}

impl EngineCore {
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

    fn apply_adapter_execution_contract(
        request: &EngineCoreRequest,
        mut profile: ExecutionProfile,
    ) -> Result<ExecutionProfile> {
        let Some(binding) = request.execution_adapter_binding() else {
            return Ok(Self::force_scalar_execution(profile));
        };

        let prefill_work = WorkUnit::SequenceStep {
            phase: super::SequencePhase::Prefill,
            input: super::InputRange { start: 0, end: 0 },
            max_output_steps: 1,
        };
        let decode_work = WorkUnit::SequenceStep {
            phase: super::SequencePhase::Decode,
            input: super::InputRange { start: 0, end: 0 },
            max_output_steps: 1,
        };
        let atomic_work = WorkUnit::AtomicJob {
            kind: format!("{:?}", request.task_type).to_ascii_lowercase(),
        };
        let pipeline_work = WorkUnit::PipelineStage {
            name: format!("{:?}", request.task_type).to_ascii_lowercase(),
            ordinal: 0,
        };

        let (prefill_stage, decode_stage) = match profile.mode {
            ExecutionMode::Sequence | ExecutionMode::Realtime => (
                Some(binding.stage_for_work(&prefill_work)?),
                Some(binding.stage_for_work(&decode_work)?),
            ),
            ExecutionMode::Atomic | ExecutionMode::Artifact => {
                (Some(binding.stage_for_work(&atomic_work)?), None)
            }
            ExecutionMode::Pipeline => (Some(binding.stage_for_work(&pipeline_work)?), None),
        };

        let verify_mode = |phase: &str,
                           declared: NativeBatchMode,
                           stage: Option<&super::StageDescriptor>|
         -> Result<NativeBatchMode> {
            let Some(stage) = stage else {
                return Ok(NativeBatchMode::None);
            };
            if stage.batch_mode == NativeBatchMode::None {
                return Ok(NativeBatchMode::None);
            }
            if stage.batch_mode != declared {
                return Err(Error::InferenceError(format!(
                    "loaded adapter stage {} advertises {:?} {phase} batching, but the executor declared {:?}",
                    stage.name, stage.batch_mode, declared
                )));
            }
            Ok(stage.batch_mode)
        };

        profile.prefill_batch = verify_mode("prefill", profile.prefill_batch, prefill_stage)?;
        profile.decode_batch = verify_mode("decode", profile.decode_batch, decode_stage)?;
        let max_batch_size = [prefill_stage, decode_stage]
            .into_iter()
            .flatten()
            .map(|stage| {
                if stage.concurrency == ConcurrencyClass::Batchable {
                    stage.max_batch_size
                } else {
                    1
                }
            })
            .max();
        profile.max_batch_size = max_batch_size
            .map(|maximum| profile.max_batch_size.min(maximum).max(1))
            .unwrap_or(1);
        profile.concurrency = if [prefill_stage, decode_stage]
            .into_iter()
            .flatten()
            .any(|stage| stage.concurrency == ConcurrencyClass::Batchable)
        {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        Ok(profile)
    }

    async fn refresh_scheduler_execution_profiles(&mut self) {
        let requests: Vec<_> = self.requests.values().cloned().collect();
        for request in requests {
            let Some(epoch) = self.scheduler.get_sequence_id(&request.id) else {
                continue;
            };
            let raw_profile = self
                .executor
                .execution_profile(&request)
                .await
                .unwrap_or_else(|| {
                    ExecutionProfile::fail_closed(
                        self.config.backend,
                        request.model_variant,
                        ExecutionMode::Atomic,
                    )
                });
            let profile = match Self::apply_adapter_execution_contract(
                &request,
                raw_profile.clone(),
            ) {
                Ok(profile) => profile,
                Err(error) => {
                    warn!(
                        request_id = %request.id,
                        error = %error,
                        "Loaded adapter contract disagrees with executor; forcing scalar scheduling"
                    );
                    Self::force_scalar_execution(raw_profile)
                }
            };
            self.scheduler.update_execution_profile(
                &super::SessionKey::new(request.id.clone(), epoch),
                &profile,
            );
        }
    }

    async fn begin_execution_plan(
        &mut self,
        scheduled: &super::scheduler::ScheduledRequest,
    ) -> Result<()> {
        let request = self
            .requests
            .get(&scheduled.request_id)
            .cloned()
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "scheduled request {} is missing from the engine",
                    scheduled.request_id
                ))
            })?;
        let raw_profile = self
            .executor
            .execution_profile(&request)
            .await
            .unwrap_or_else(|| {
                ExecutionProfile::fail_closed(
                    self.config.backend,
                    request.model_variant,
                    ExecutionMode::Atomic,
                )
            });
        let profile = Self::apply_adapter_execution_contract(&request, raw_profile)?;
        if scheduled.is_prefill
            && profile.prefill == PrefillMode::Full
            && (scheduled.num_computed_tokens != 0
                || scheduled.num_tokens < request.num_prompt_tokens())
        {
            return Err(Error::InferenceError(format!(
                "full-prefill request {} was scheduled as a partial prompt quantum",
                scheduled.request_id
            )));
        }
        let work = match profile.mode {
            ExecutionMode::Sequence | ExecutionMode::Realtime
                if scheduled.is_prefill && profile.prefill == PrefillMode::Full =>
            {
                WorkUnit::SequenceStep {
                    phase: super::SequencePhase::Prefill,
                    input: super::InputRange {
                        start: 0,
                        end: request.num_prompt_tokens(),
                    },
                    max_output_steps: 1,
                }
            }
            ExecutionMode::Sequence | ExecutionMode::Realtime => scheduled.work.clone(),
            ExecutionMode::Atomic | ExecutionMode::Artifact => WorkUnit::AtomicJob {
                kind: format!("{:?}", request.task_type).to_ascii_lowercase(),
            },
            ExecutionMode::Pipeline => WorkUnit::PipelineStage {
                name: format!("{:?}", request.task_type).to_ascii_lowercase(),
                ordinal: 0,
            },
        };
        let work_kind = match &work {
            WorkUnit::SequenceStep { phase, .. } => format!("{phase:?}").to_ascii_lowercase(),
            WorkUnit::AtomicJob { kind } => kind.clone(),
            WorkUnit::PipelineStage { name, ordinal } => format!("{name}:{ordinal}"),
        };
        // Managed arenas are registered once at model scope, while opaque
        // caches are authorized and reconciled from executor-owned tensors.
        // A row therefore never reserves either physical allocation again.
        let estimate = ResourceVector::zero();
        let bound_stage = request
            .execution_adapter_binding()
            .map(|binding| binding.stage_for_work(&work).cloned())
            .transpose()?;
        let bound_adapter = request
            .execution_adapter_binding()
            .zip(bound_stage.as_ref())
            .map(|(binding, stage)| binding.key_for_stage(stage.id))
            .transpose()?;
        let batch_mode = bound_stage
            .as_ref()
            .map_or(NativeBatchMode::None, |stage| stage.batch_mode);
        let max_batch_size = bound_stage.as_ref().map_or(1, |stage| {
            if stage.concurrency == ConcurrencyClass::Batchable {
                stage.max_batch_size.min(profile.max_batch_size).max(1)
            } else {
                1
            }
        });
        let plan = ExecutionPlan {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            work,
            batch_key: BatchKey {
                backend: profile.backend,
                model_variant: profile.model_variant,
                task_type: request.task_type,
                work_kind,
                compute_dtype: profile.compute_dtype,
                kv_dtype: profile.kv_dtype,
                cache_namespace: profile
                    .cache_namespace
                    .unwrap_or_else(|| "none".to_string()),
                adapter: bound_adapter,
            },
            batch_mode,
            max_batch_size,
            estimate,
            stage: bound_stage,
        };

        let tracker = self
            .execution_trackers
            .entry(scheduled.request_id.clone())
            .or_insert_with(|| ExecutionTracker::new(plan.session.clone()));
        if tracker.session() != &plan.session {
            return Err(Error::InferenceError(format!(
                "request {} already has a different active session",
                scheduled.request_id
            )));
        }
        if tracker.state() == ExecutionState::Queued {
            tracker.transition(ExecutionState::Admitted)?;
        }
        let running_state = match plan.work {
            WorkUnit::SequenceStep {
                phase: super::SequencePhase::Prefill,
                ..
            } => ExecutionState::Prefilling,
            WorkUnit::SequenceStep {
                phase: super::SequencePhase::Decode,
                ..
            } => ExecutionState::Decoding,
            WorkUnit::AtomicJob { .. } => ExecutionState::AtomicRunning,
            WorkUnit::PipelineStage { .. } => ExecutionState::PipelineRunning,
        };
        if tracker.state() != running_state {
            tracker.transition(running_state)?;
        }
        tracker.begin_plan(&plan)?;
        if self.active_plans.insert(plan.plan_id, plan).is_some() {
            return Err(Error::InferenceError(format!(
                "execution plan {} was prepared twice",
                scheduled.plan_id
            )));
        }
        Ok(())
    }

    fn rollback_unexecuted_schedule(&mut self, scheduled: &[super::scheduler::ScheduledRequest]) {
        let rolled_back_plans = scheduled
            .iter()
            .map(|scheduled| scheduled.plan_id)
            .collect::<HashSet<_>>();
        self.active_stream_batches.retain(|_, batch| {
            !batch
                .rows
                .keys()
                .any(|plan_id| rolled_back_plans.contains(plan_id))
        });
        for scheduled in scheduled {
            let session = scheduled.session_key();
            if let Some(reservation) = self.active_managed_cache.remove(&scheduled.plan_id) {
                if let Err(error) = self.managed_kv_cache.finalize(&reservation, None, false) {
                    warn!(
                        plan_id = scheduled.plan_id,
                        error = %error,
                        "Failed to abort an unexecuted managed KV transaction"
                    );
                }
            }
            if let Some(plan) = self.active_plans.remove(&scheduled.plan_id) {
                if plan.session != session {
                    warn!(
                        plan_id = scheduled.plan_id,
                        request_id = %scheduled.request_id,
                        "Unexecuted plan rollback found a mismatched session fence"
                    );
                }
                if !self
                    .scheduler
                    .refund_unexecuted_service(&session, scheduled.num_tokens)
                {
                    warn!(
                        plan_id = scheduled.plan_id,
                        request_id = %scheduled.request_id,
                        "Scheduler rejected unexecuted weighted-service refund"
                    );
                }
            }
            if let Some(tracker) = self.execution_trackers.get_mut(&scheduled.request_id) {
                if tracker.session() == &session {
                    tracker.rollback_unexecuted_plan(scheduled.plan_id);
                }
            }
            self.scheduler.release_execution_quantum_for_retry(&session);
            self.release_managed_session_if_ready(&session);
        }
    }

    fn defer_unexecuted_schedule_for_capacity(
        &mut self,
        scheduled: &[super::scheduler::ScheduledRequest],
    ) {
        self.rollback_unexecuted_schedule(scheduled);
        let retry_at = Instant::now() + self.retry_policy.execution_delay(1);
        for scheduled in scheduled {
            let session = scheduled.session_key();
            if !self.scheduler.defer_execution_retry(&session, retry_at) {
                warn!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    "Scheduler rejected managed KV capacity backpressure"
                );
            }
        }
    }

    fn report_from_result(result: &ExecutorStepResult) -> ExecutionReport {
        let output = &result.output;
        ExecutionReport {
            plan_id: result.plan_id,
            session: result.session.clone(),
            input_consumed: output.tokens_processed,
            output_produced: output.tokens_generated,
            observed_resources: result.observed_resources,
            dispatch: result.dispatch,
            provenance: result.provenance,
            elapsed: std::time::Duration::ZERO,
            safe_point: result.safe_point,
            disposition: result.disposition.clone(),
            output_finished: output.finished,
            output_has_error: output.error.is_some(),
        }
    }

    fn canonical_failure_dispatch(
        plan: &ExecutionPlan,
        result: &ExecutorStepResult,
    ) -> (BatchDispatch, DispatchState) {
        let width = result.dispatch.width.clamp(1, plan.max_batch_size.max(1));
        if result.dispatch.kind == super::BatchDispatchKind::NotDispatched {
            return (
                BatchDispatch::not_dispatched(width),
                DispatchState::NotStarted,
            );
        }
        let dispatch = match plan.batch_mode {
            NativeBatchMode::Static => {
                BatchDispatch::new(super::BatchDispatchKind::TensorStatic, width)
            }
            NativeBatchMode::Continuous => {
                BatchDispatch::new(super::BatchDispatchKind::TensorContinuous, width)
            }
            NativeBatchMode::None if width > 1 => {
                BatchDispatch::new(super::BatchDispatchKind::RequestParallel, width)
            }
            NativeBatchMode::None => BatchDispatch::serial(),
        };
        let dispatch_state = match result.provenance.dispatch_state {
            DispatchState::NotStarted | DispatchState::Started => DispatchState::Started,
            DispatchState::ProducedOutput => DispatchState::ProducedOutput,
        };
        (dispatch, dispatch_state)
    }

    async fn commit_executor_result(
        &mut self,
        mut result: ExecutorStepResult,
        step_time_ms: f64,
    ) -> Option<CommittedExecutorOutput> {
        let Some(plan) = self.active_plans.remove(&result.plan_id) else {
            if let Some(reservation) = self.active_managed_cache.remove(&result.plan_id) {
                let _ = self.managed_kv_cache.finalize(&reservation, None, false);
                self.release_managed_session_if_ready(&reservation.session);
            }
            warn!(
                plan_id = result.plan_id,
                request_id = %result.session.request_id,
                session_epoch = result.session.epoch,
                "Ignoring executor result for an inactive or already committed plan"
            );
            return None;
        };

        if self.incremental_stream_sessions.contains(&plan.session) {
            if let ExecutionDisposition::Failed(failure) = &mut result.disposition {
                if failure.retry != RetryDisposition::Never {
                    let message = format!(
                        "executor failed after committed stream progress; retry is unsafe: {}",
                        failure.message
                    );
                    failure.retry = RetryDisposition::Never;
                    failure.message = message.clone();
                    result.output.finished = true;
                    result.output.error = Some(message);
                    result.staged_stream_outputs.clear();
                }
            }
        }

        let staged_next_sequence = match self
            .validate_staged_stream_outputs(&plan.session, &result.staged_stream_outputs)
        {
            Ok(next) => Some(next),
            Err(error) => {
                let message = format!("invalid staged stream output: {error}");
                result.output =
                    ExecutorOutput::error(plan.session.request_id.clone(), message.clone());
                result.disposition =
                    ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
                result.safe_point = true;
                result.provenance = OutcomeProvenance::failure(
                    FailureOrigin::ExecutorValidation,
                    result.provenance.dispatch_state,
                );
                result.staged_stream_outputs.clear();
                None
            }
        };

        let retry_attempt = match &mut result.disposition {
            ExecutionDisposition::Failed(failure) if failure.retry != RetryDisposition::Never => {
                let attempts = self
                    .execution_retry_attempts
                    .entry(plan.session.clone())
                    .or_default();
                *attempts = attempts.saturating_add(1);
                if *attempts > self.retry_policy.max_execution_retries {
                    let message = format!(
                        "executor retry budget exhausted after {} attempts: {}",
                        self.retry_policy.max_execution_retries, failure.message
                    );
                    failure.retry = RetryDisposition::Never;
                    failure.message = message.clone();
                    result.output.finished = true;
                    result.output.error = Some(message);
                    None
                } else {
                    Some(*attempts)
                }
            }
            _ => {
                self.execution_retry_attempts.remove(&plan.session);
                None
            }
        };
        let managed_reservation = self.active_managed_cache.remove(&plan.plan_id);
        let report = Self::report_from_result(&result);
        // Validate the execution transition on a clone before publishing the
        // physical cache table. This keeps invalid/duplicate reports from
        // advancing KV state, while the serialized core lock guarantees that
        // installing the validated tracker cannot race another row commit.
        let prospective_tracker = self
            .execution_trackers
            .get(&plan.session.request_id)
            .cloned()
            .ok_or_else(|| {
                Error::InferenceError("execution tracker is missing for active plan".to_string())
            })
            .and_then(|mut tracker| {
                tracker.commit(&plan, &report)?;
                Ok(tracker)
            });

        let prospective_tracker = match prospective_tracker {
            Ok(tracker) => tracker,
            Err(err) => {
                if let Some(reservation) = managed_reservation.as_ref() {
                    let _ = self.managed_kv_cache.finalize(reservation, None, false);
                }
                self.release_managed_session_if_ready(&plan.session);
                let (failure_dispatch, failure_dispatch_state) =
                    Self::canonical_failure_dispatch(&plan, &result);
                let failure_report = ExecutionReport {
                    plan_id: plan.plan_id,
                    session: plan.session.clone(),
                    input_consumed: 0,
                    output_produced: 0,
                    observed_resources: ResourceVector::zero(),
                    dispatch: failure_dispatch,
                    provenance: OutcomeProvenance::failure(
                        FailureOrigin::StateCommit,
                        failure_dispatch_state,
                    ),
                    elapsed: std::time::Duration::ZERO,
                    safe_point: true,
                    disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        err.to_string(),
                    )),
                    output_finished: true,
                    output_has_error: true,
                };
                if let Some(tracker) = self.execution_trackers.get_mut(&plan.session.request_id) {
                    let _ = tracker.commit(&plan, &failure_report);
                }
                let message = format!("invalid executor result: {err}");
                return Some(CommittedExecutorOutput {
                    session: plan.session.clone(),
                    output: ExecutorOutput::error(plan.session.request_id, message.clone()),
                    disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        message,
                    )),
                    provenance: OutcomeProvenance::failure(
                        FailureOrigin::StateCommit,
                        failure_dispatch_state,
                    ),
                    staged_stream_outputs: Vec::new(),
                });
            }
        };

        let managed_commit = matches!(
            result.disposition,
            ExecutionDisposition::Progress | ExecutionDisposition::Yielded(_)
        );
        let managed_result = match managed_reservation.as_ref() {
            Some(reservation) => self.managed_kv_cache.finalize(
                reservation,
                result.managed_cache.as_ref(),
                managed_commit,
            ),
            None if result.managed_cache.is_some() => Err(Error::InferenceError(
                "executor returned an unplanned managed KV receipt".to_string(),
            )),
            None => Ok(()),
        };
        if let Err(error) = managed_result {
            let message = format!("managed KV commit failed: {error}");
            result.output = ExecutorOutput::error(plan.session.request_id.clone(), message.clone());
            result.disposition =
                ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
            result.safe_point = true;
            result.provenance = OutcomeProvenance::failure(
                FailureOrigin::StateCommit,
                result.provenance.dispatch_state,
            );
            result.staged_stream_outputs.clear();
            let failure_report = Self::report_from_result(&result);
            if let Some(tracker) = self.execution_trackers.get_mut(&plan.session.request_id) {
                let _ = tracker.commit(&plan, &failure_report);
            }
        } else {
            self.execution_trackers
                .insert(plan.session.request_id.clone(), prospective_tracker);
        }
        self.release_managed_session_if_ready(&plan.session);

        match &result.disposition {
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::RetrySameSession =>
            {
                let attempt = retry_attempt.unwrap_or(1);
                let retry_at = Instant::now() + self.retry_policy.execution_delay(attempt);
                if self
                    .scheduler
                    .defer_execution_retry(&plan.session, retry_at)
                {
                    return None;
                }
                let message = "scheduler rejected a same-session execution retry";
                return Some(CommittedExecutorOutput {
                    session: plan.session.clone(),
                    output: ExecutorOutput::error(
                        plan.session.request_id.clone(),
                        message.to_string(),
                    ),
                    disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        message,
                    )),
                    provenance: OutcomeProvenance::failure(
                        FailureOrigin::StateCommit,
                        result.provenance.dispatch_state,
                    ),
                    staged_stream_outputs: Vec::new(),
                });
            }
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::Recompute =>
            {
                let release = self.executor.cleanup_session(&plan.session).await;
                if release.confirmed {
                    self.request_managed_session_release(&plan.session);
                }
                if release.confirmed && self.scheduler.restart_request_for_recompute(&plan.session)
                {
                    let attempt = retry_attempt.unwrap_or(1);
                    let retry_at = Instant::now() + self.retry_policy.execution_delay(attempt);
                    if !self
                        .scheduler
                        .defer_execution_retry(&plan.session, retry_at)
                    {
                        let message = "scheduler rejected a deferred recompute retry";
                        return Some(CommittedExecutorOutput {
                            session: plan.session.clone(),
                            output: ExecutorOutput::error(plan.session.request_id.clone(), message),
                            disposition: ExecutionDisposition::Failed(
                                ExecutionFailure::invalid_output(message),
                            ),
                            provenance: OutcomeProvenance::failure(
                                FailureOrigin::StateCommit,
                                result.provenance.dispatch_state,
                            ),
                            staged_stream_outputs: Vec::new(),
                        });
                    }
                    self.execution_trackers.remove(&plan.session.request_id);
                    self.active_plans
                        .retain(|_, active| active.session != plan.session);
                    self.request_phase_timings
                        .entry(plan.session.request_id)
                        .and_modify(|timing| *timing = RequestPhaseTiming::default());
                    return None;
                }

                let message = if release.confirmed {
                    "scheduler rejected an execution recompute retry"
                } else {
                    "executor could not confirm physical cache release for recompute retry"
                };
                return Some(CommittedExecutorOutput {
                    session: plan.session.clone(),
                    output: ExecutorOutput::error(
                        plan.session.request_id.clone(),
                        message.to_string(),
                    ),
                    disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        message,
                    )),
                    provenance: OutcomeProvenance::failure(
                        FailureOrigin::Cleanup,
                        result.provenance.dispatch_state,
                    ),
                    staged_stream_outputs: Vec::new(),
                });
            }
            ExecutionDisposition::Progress
            | ExecutionDisposition::Yielded(_)
            | ExecutionDisposition::Finished(_) => {
                self.scheduler.update_after_step(
                    &plan.session.request_id,
                    result.output.tokens_processed,
                    result.output.tokens_generated,
                    step_time_ms,
                );
            }
            ExecutionDisposition::Failed(_) => {}
        }
        result.output.request_id = plan.session.request_id.clone();
        if !result.staged_stream_outputs.is_empty() {
            if let Some(next_sequence) = staged_next_sequence {
                self.stream_sequence_cursors
                    .insert(plan.session.clone(), next_sequence);
            }
        }
        Some(CommittedExecutorOutput {
            session: plan.session,
            output: result.output,
            disposition: result.disposition,
            provenance: result.provenance,
            staged_stream_outputs: result.staged_stream_outputs,
        })
    }

    fn work_cost(
        request: &EngineCoreRequest,
        work: &WorkUnit,
        stage: Option<&super::StageDescriptor>,
    ) -> Result<WorkCost> {
        if let Some(prepared) = stage.and_then(|stage| request.prepared_stage_cost(stage.id)) {
            return Ok(prepared);
        }
        let logical_units = match work {
            WorkUnit::SequenceStep {
                input,
                max_output_steps,
                ..
            } => {
                let input = u64::try_from(input.len()).map_err(|_| {
                    Error::Overloaded("execution input length exceeds work accounting".to_string())
                })?;
                let output = u64::try_from(*max_output_steps).map_err(|_| {
                    Error::Overloaded("execution output bound exceeds work accounting".to_string())
                })?;
                input.max(output).max(1)
            }
            WorkUnit::AtomicJob { .. } | WorkUnit::PipelineStage { .. } => 1,
        };
        let workspace_bytes = stage.map_or(Ok(0), |stage| {
            stage
                .workspace_per_work_unit_bytes
                .checked_mul(logical_units)
                .and_then(|bytes| bytes.checked_add(stage.workspace_per_row_bytes))
                .ok_or_else(|| Error::Overloaded("batch workspace estimate overflow".to_string()))
        })?;
        Ok(WorkCost::new(logical_units, logical_units, workspace_bytes))
    }

    fn batch_lane(plan: &ExecutionPlan, cost: WorkCost) -> BatchLaneKey {
        let adapter = plan.batch_key.adapter.as_ref();
        let stage = plan.stage.as_ref();
        let shape_policy = stage
            .map(|stage| stage.shape_policy)
            .unwrap_or(StageShapePolicy::Exact);
        let shape_bucket = match shape_policy {
            StageShapePolicy::Independent => "independent".to_string(),
            StageShapePolicy::Exact => format!("exact.{}", cost.tensor_elements),
            StageShapePolicy::Bucketed => format!(
                "bucket.{}",
                cost.tensor_elements
                    .checked_next_power_of_two()
                    .unwrap_or(u64::MAX)
            ),
            StageShapePolicy::Padded => "padded".to_string(),
            StageShapePolicy::Ragged => "ragged".to_string(),
        };
        BatchLaneKey {
            execution_group: adapter
                .map(|adapter| adapter.execution_group_id)
                .unwrap_or(ExecutionGroupId::new(0)),
            model_instance: adapter
                .map(|adapter| adapter.model_instance_id)
                .unwrap_or(ModelInstanceId::new(0)),
            adapter_instance: adapter
                .map(|adapter| adapter.adapter_instance_id)
                .unwrap_or(AdapterInstanceId::new(0)),
            adapter_abi: adapter
                .map(|adapter| adapter.adapter_abi_revision)
                .unwrap_or(AdapterAbiRevision::new(0)),
            capability_id: adapter
                .map(|adapter| adapter.capability_id.clone())
                .unwrap_or_else(|| "compatibility".to_string()),
            stage_id: adapter
                .map(|adapter| adapter.stage_id)
                .unwrap_or(StageId::new(0)),
            backend: plan.batch_key.backend,
            device_ordinal: None,
            compute_dtype: plan.batch_key.compute_dtype.clone(),
            state_dtype: plan.batch_key.kv_dtype.clone(),
            tensor_layout: format!("{shape_policy:?}").to_ascii_lowercase(),
            quantization: "adapter-owned".to_string(),
            state_schema: plan.batch_key.cache_namespace.clone(),
            kernel_mode: stage
                .map(|stage| stage.name.clone())
                .unwrap_or_else(|| "compatibility".to_string()),
            semantic_mode: format!(
                "{:?}.{}",
                plan.batch_key.task_type, plan.batch_key.work_kind
            )
            .to_ascii_lowercase(),
            shape_bucket,
        }
    }

    fn batch_budget(plan: &ExecutionPlan) -> Result<(BatchBudget, StageShapePolicy)> {
        let Some(stage) = plan.stage.as_ref() else {
            if plan.batch_mode == NativeBatchMode::None {
                return Ok((BatchBudget::width_one(), StageShapePolicy::Exact));
            }
            return Err(Error::InferenceError(
                "native tensor batch plan is missing its loaded stage contract".to_string(),
            ));
        };
        if stage.concurrency == ConcurrencyClass::Exclusive {
            return Ok((BatchBudget::width_one(), stage.shape_policy));
        }
        let budget = BatchBudget {
            max_rows: plan.max_batch_size.min(stage.max_batch_size).max(1),
            max_logical_units: stage.max_work_units,
            max_tensor_elements: u64::MAX,
            max_workspace_bytes: stage.max_workspace_bytes,
            max_padding_basis_points: stage.max_padding_basis_points,
            max_formation_delay: stage.max_formation_delay,
        };
        budget.validate()?;
        Ok((budget, stage.shape_policy))
    }

    fn allocate_batch_id(&mut self) -> Result<BatchId> {
        let batch_id = BatchId::new(self.next_batch_id);
        self.next_batch_id = self.next_batch_id.checked_add(1).ok_or_else(|| {
            Error::InferenceError("physical batch identity space was exhausted".to_string())
        })?;
        Ok(batch_id)
    }

    fn form_physical_batches(
        &mut self,
        requests: &[Arc<EngineCoreRequest>],
        scheduled: &[super::scheduler::ScheduledRequest],
        capacity_blocked: &mut Vec<super::scheduler::ScheduledRequest>,
    ) -> Result<Vec<PreparedExecutionBatch>> {
        let available = requests
            .iter()
            .map(|request| request.id.clone())
            .collect::<HashSet<_>>();
        let mut assemblies: Vec<PhysicalBatchAssembly> = Vec::new();

        for scheduled in scheduled {
            if !available.contains(&scheduled.request_id) {
                continue;
            }
            let request = self
                .requests
                .get(&scheduled.request_id)
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(format!(
                        "scheduled request {} disappeared during batch formation",
                        scheduled.request_id
                    ))
                })?;
            let plan = self
                .active_plans
                .get(&scheduled.plan_id)
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(format!(
                        "scheduled plan {} disappeared during batch formation",
                        scheduled.plan_id
                    ))
                })?;
            let cost = Self::work_cost(&request, &plan.work, plan.stage.as_ref())?;
            let lane = Self::batch_lane(&plan, cost);
            let (budget, shape_policy) = Self::batch_budget(&plan)?;
            let output_visibility = plan
                .stage
                .as_ref()
                .map_or(super::OutputVisibility::AfterQuantumCommit, |stage| {
                    stage.output_visibility
                });
            let planned_work = plan.work.clone();
            let managed_cache = match request.managed_cache_runtime().map(|runtime| {
                self.managed_kv_cache.prepare(
                    runtime,
                    plan.plan_id,
                    &plan.session,
                    &planned_work,
                    Some(&request),
                )
            }) {
                Some(Ok(reservation)) => reservation,
                None => None,
                Some(Err(Error::Backpressure(reason))) => {
                    capacity_blocked.push(scheduled.clone());
                    debug!(
                        request_id = %scheduled.request_id,
                        plan_id = scheduled.plan_id,
                        reason = %reason,
                        "Withholding only the row blocked by managed KV capacity"
                    );
                    continue;
                }
                Some(Err(error)) => return Err(error),
            };
            if let Some(reservation) = managed_cache.as_ref() {
                if self
                    .active_managed_cache
                    .insert(plan.plan_id, reservation.clone())
                    .is_some()
                {
                    let _ = self.managed_kv_cache.finalize(reservation, None, false);
                    return Err(Error::InferenceError(
                        "managed KV transaction was prepared twice".to_string(),
                    ));
                }
            }
            let row = ReadyQuantum {
                plan_id: plan.plan_id,
                session: plan.session.clone(),
                lane: lane.clone(),
                work: planned_work.clone(),
                cost,
                managed_cache,
            };
            let mut executable = scheduled.clone();
            executable.work = planned_work;

            let mut pending = Some((request, executable, row));
            if budget.max_rows > 1 {
                for assembly in &mut assemblies {
                    if assembly.physical_batch.lane != lane
                        || assembly.physical_batch.mode != plan.batch_mode
                        || assembly.physical_batch.budget != budget
                        || assembly.output_visibility != output_visibility
                        || assembly.shape_policy != shape_policy
                    {
                        continue;
                    }
                    let Some((request, scheduled, row)) = pending.take() else {
                        break;
                    };
                    if assembly.try_push(request.clone(), scheduled.clone(), row.clone()) {
                        break;
                    }
                    pending = Some((request, scheduled, row));
                }
            }

            if let Some((request, scheduled, row)) = pending {
                let materialized_tensor_elements = row.cost.tensor_elements;
                let workspace_base_bytes = plan
                    .stage
                    .as_ref()
                    .map(|stage| stage.workspace_base_bytes)
                    .unwrap_or(0);
                let workspace = PhysicalBatchAssembly::workspace_resources(
                    lane.backend,
                    workspace_base_bytes,
                    std::slice::from_ref(&row),
                )
                .ok_or_else(|| {
                    Error::Overloaded("batch workspace estimate overflow".to_string())
                })?;
                let physical_batch = PhysicalBatch {
                    batch_id: self.allocate_batch_id()?,
                    lane,
                    mode: plan.batch_mode,
                    budget,
                    rows: vec![row],
                    materialized_tensor_elements,
                    workspace,
                };
                physical_batch.validate()?;
                assemblies.push(PhysicalBatchAssembly {
                    physical_batch,
                    requests: vec![request],
                    scheduled: vec![scheduled],
                    output_visibility,
                    shape_policy,
                    workspace_base_bytes,
                });
            }
        }

        let mut prepared = Vec::with_capacity(assemblies.len());
        for assembly in assemblies {
            if assembly.output_visibility == OutputVisibility::IncrementalCommitted {
                let rows = assembly
                    .physical_batch
                    .rows
                    .iter()
                    .map(|row| (row.plan_id, row.session.clone()))
                    .collect();
                let fence = ActiveStreamBatch {
                    lane: assembly.physical_batch.lane.clone(),
                    output_visibility: assembly.output_visibility,
                    rows,
                };
                if self
                    .active_stream_batches
                    .insert(assembly.physical_batch.batch_id, fence)
                    .is_some()
                {
                    return Err(Error::InferenceError(
                        "physical stream batch identity was registered twice".to_string(),
                    ));
                }
            }
            let reservations = assembly
                .physical_batch
                .rows
                .iter()
                .filter_map(|row| row.managed_cache.clone())
                .collect();
            prepared.push(
                PreparedExecutionBatch::new(
                    assembly.physical_batch,
                    assembly.requests,
                    assembly.scheduled,
                    assembly.output_visibility,
                )
                .with_managed_cache_reservations(reservations)?,
            );
        }
        Ok(prepared)
    }

    fn merge_audio_output(
        existing: Option<AudioOutput>,
        current: Option<AudioOutput>,
    ) -> Option<AudioOutput> {
        match (existing, current) {
            (None, None) => None,
            (Some(existing), None) => Some(existing),
            (None, Some(current)) => Some(current),
            (Some(mut existing), Some(current)) => {
                if existing.sample_rate != current.sample_rate {
                    return Some(current);
                }
                if current.samples.is_empty() {
                    return Some(existing);
                }
                if existing.samples.is_empty() {
                    return Some(current);
                }

                let looks_cumulative = current.samples.len() >= existing.samples.len()
                    && current
                        .samples
                        .iter()
                        .zip(existing.samples.iter())
                        .all(|(cur, prev)| cur == prev);

                if looks_cumulative {
                    Some(current)
                } else {
                    existing.append(&current);
                    Some(existing)
                }
            }
        }
    }

    fn merge_executor_output(
        existing: Option<ExecutorOutput>,
        current: ExecutorOutput,
    ) -> ExecutorOutput {
        let Some(mut merged) = existing else {
            return current;
        };

        if merged.request_id != current.request_id {
            return current;
        }
        if merged.finished || merged.error.is_some() {
            return merged;
        }

        let ExecutorOutput {
            request_id: _,
            audio,
            text,
            input_transcription,
            tokens_processed,
            tokens_generated,
            finished,
            phase_timing_override,
            asr_diagnostics,
            error,
        } = current;

        merged.audio = Self::merge_audio_output(merged.audio.take(), audio);
        if text.is_some() {
            merged.text = text;
        }
        if input_transcription.is_some() {
            merged.input_transcription = input_transcription;
        }
        merged.tokens_processed = merged.tokens_processed.saturating_add(tokens_processed);
        merged.tokens_generated = merged.tokens_generated.saturating_add(tokens_generated);
        merged.finished |= finished;
        merged.phase_timing_override = phase_timing_override.or(merged.phase_timing_override);
        merged.asr_diagnostics = asr_diagnostics.or(merged.asr_diagnostics);
        if error.is_some() {
            merged.error = error;
        }

        merged
    }

    fn has_user_visible_output(exec_output: &ExecutorOutput) -> bool {
        exec_output.tokens_generated > 0
            || exec_output
                .text
                .as_ref()
                .is_some_and(|text| !text.is_empty())
            || exec_output
                .audio
                .as_ref()
                .is_some_and(|audio| !audio.samples.is_empty())
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

    fn new_with_executor(config: EngineCoreConfig, executor: UnifiedExecutor) -> Result<Self> {
        Self::new_with_executor_and_managed_authority(config, executor, None, None)
    }

    fn new_with_executor_and_managed_authority(
        config: EngineCoreConfig,
        executor: UnifiedExecutor,
        managed_resource_authority: Option<Arc<super::ResourceAuthority>>,
        managed_worker: Option<(BackendKind, candle_core::Device)>,
    ) -> Result<Self> {
        // Direct EngineCore users must cross the same fail-closed cache-policy
        // boundary as RuntimeService users before workers or arenas start.
        let cache_policy = config.resolved_kv_cache_policy()?;
        // Create scheduler
        let scheduler_config = SchedulerConfig::from(&config);
        let scheduler = Scheduler::new(scheduler_config);

        // Create output processor
        let output_processor =
            OutputProcessor::new(config.sample_rate).with_chunk_size(config.streaming_chunk_size);
        let managed_prefix_salt = config
            .enable_prefix_caching
            .then_some(config.managed_prefix_cache_salt.as_deref())
            .flatten()
            .map(|salt| Sha256::digest(salt.as_bytes()).into());
        let max_prefix_cache_pages = match cache_policy.effective.prefix {
            crate::config::PrefixCachePolicy::Disabled => 0,
            crate::config::PrefixCachePolicy::Namespaced { max_pages, .. } => max_pages,
        };

        let (managed_kv_cache, physical_state) = match managed_worker {
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

        Ok(Self {
            config,
            scheduler,
            managed_kv_cache,
            physical_state,
            executor,
            output_processor,
            requests: HashMap::new(),
            request_start_times: HashMap::new(),
            request_phase_timings: HashMap::new(),
            execution_trackers: HashMap::new(),
            active_plans: HashMap::new(),
            active_managed_cache: HashMap::new(),
            pending_managed_releases: HashSet::new(),
            active_stream_batches: HashMap::new(),
            stream_sequence_cursors: HashMap::new(),
            incremental_stream_sessions: HashSet::new(),
            pending_terminal_outputs: VecDeque::new(),
            execution_retry_attempts: HashMap::new(),
            retry_policy: LifecycleRetryPolicy::default(),
            initialized: false,
            next_batch_id: 1,
        })
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
                    request.install_managed_cache_runtime(physical)?;
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
        self.requests.insert(request_id.clone(), Arc::new(request));
        self.request_start_times
            .insert(request_id.clone(), Instant::now());
        self.request_phase_timings
            .insert(request_id.clone(), RequestPhaseTiming::default());

        debug!(
            request_id = %request_id,
            correlation_id = ?self.requests.get(&request_id).and_then(|req| req.correlation_id.as_deref()),
            "Added request to engine core"
        );

        Ok(())
    }

    fn terminal_release_cause(disposition: &ExecutionDisposition) -> Option<TerminalReleaseCause> {
        match disposition {
            ExecutionDisposition::Finished(ExecutionFinishReason::Completed) => {
                Some(TerminalReleaseCause::Completed)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled) => {
                Some(TerminalReleaseCause::Cancelled)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut) => {
                Some(TerminalReleaseCause::TimedOut)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Rejected)
            | ExecutionDisposition::Failed(ExecutionFailure {
                retry: RetryDisposition::Never,
                ..
            }) => Some(TerminalReleaseCause::Failed),
            _ => None,
        }
    }

    fn clear_exact_execution_state(&mut self, session: &super::SessionKey) {
        if self
            .execution_trackers
            .get(&session.request_id)
            .is_some_and(|tracker| tracker.session() == session)
        {
            self.execution_trackers.remove(&session.request_id);
        }
        self.active_plans.retain(|_, plan| plan.session != *session);
        self.active_stream_batches
            .retain(|_, batch| !batch.rows.values().any(|row| row == session));
        self.execution_retry_attempts.remove(session);
        self.stream_sequence_cursors.remove(session);
        self.incremental_stream_sessions.remove(session);
    }

    fn request_managed_session_release(&mut self, session: &super::SessionKey) {
        if self
            .active_managed_cache
            .values()
            .any(|reservation| &reservation.session == session)
        {
            self.pending_managed_releases.insert(session.clone());
            return;
        }
        if let Err(error) = self.managed_kv_cache.release_session(session) {
            warn!(
                request_id = %session.request_id,
                session_epoch = session.epoch,
                error = %error,
                "Failed to release a managed KV session table"
            );
        }
        self.pending_managed_releases.remove(session);
    }

    fn release_managed_session_if_ready(&mut self, session: &super::SessionKey) {
        if self.pending_managed_releases.contains(session)
            && !self
                .active_managed_cache
                .values()
                .any(|reservation| &reservation.session == session)
        {
            self.request_managed_session_release(session);
        }
    }

    fn validate_staged_stream_outputs(
        &self,
        session: &super::SessionKey,
        outputs: &[super::output::StreamingOutput],
    ) -> Result<usize> {
        let mut expected = self
            .stream_sequence_cursors
            .get(session)
            .copied()
            .unwrap_or(0);
        for (index, output) in outputs.iter().enumerate() {
            if output.request_id != session.request_id {
                return Err(Error::InferenceError(
                    "stream output request ID does not match its session".to_string(),
                ));
            }
            if output.sequence != expected {
                return Err(Error::InferenceError(format!(
                    "stream output sequence {} did not match expected {}",
                    output.sequence, expected
                )));
            }
            if output.is_final && index + 1 != outputs.len() {
                return Err(Error::InferenceError(
                    "final stream output must be the last committed event".to_string(),
                ));
            }
            expected = expected.checked_add(1).ok_or_else(|| {
                Error::InferenceError("stream output sequence space was exhausted".to_string())
            })?;
        }
        Ok(expected)
    }

    pub(super) fn commit_incremental_stream_progress(
        &mut self,
        progress: FencedStreamProgress,
    ) -> std::result::Result<CommittedStreamDelivery, StreamProgressRejection> {
        let plan = self.active_plans.get(&progress.plan_id).ok_or_else(|| {
            StreamProgressRejection::invalid("stream progress references an inactive plan")
        })?;
        if plan.session != progress.session {
            return Err(StreamProgressRejection::invalid(
                "stream progress session does not match its active plan",
            ));
        }
        if plan.stage.as_ref().map(|stage| stage.output_visibility)
            != Some(OutputVisibility::IncrementalCommitted)
        {
            return Err(StreamProgressRejection::invalid(
                "stream progress is not authorized by the active adapter stage",
            ));
        }

        let batch = self
            .active_stream_batches
            .get(&progress.batch_id)
            .ok_or_else(|| {
                StreamProgressRejection::invalid(
                    "stream progress references an inactive physical batch",
                )
            })?;
        if batch.output_visibility != OutputVisibility::IncrementalCommitted
            || batch.lane != progress.lane
            || batch.rows.get(&progress.plan_id) != Some(&progress.session)
        {
            return Err(StreamProgressRejection::invalid(
                "stream progress does not match its physical batch fence",
            ));
        }

        let tracker = self
            .execution_trackers
            .get(&progress.session.request_id)
            .ok_or_else(|| {
                StreamProgressRejection::invalid("stream progress has no execution tracker")
            })?;
        if tracker.session() != &progress.session
            || tracker.active_plan_id() != Some(progress.plan_id)
        {
            return Err(StreamProgressRejection::invalid(
                "stream progress does not match the active lifecycle transaction",
            ));
        }
        if self.scheduler.get_sequence_id(&progress.session.request_id)
            != Some(progress.session.epoch)
        {
            return Err(StreamProgressRejection::invalid(
                "stream progress belongs to a stale scheduler session",
            ));
        }
        if progress.output.request_id != progress.session.request_id || progress.output.is_final {
            return Err(StreamProgressRejection::invalid(
                "incremental progress must be non-final output for its exact request",
            ));
        }

        let request = self
            .requests
            .get(&progress.session.request_id)
            .ok_or_else(|| {
                StreamProgressRejection::invalid("stream progress request is no longer active")
            })?;
        if request.is_cancelled() {
            return Err(StreamProgressRejection::cancelled());
        }
        if request
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(StreamProgressRejection::deadline());
        }
        let tx = request.streaming_tx.clone().ok_or_else(|| {
            StreamProgressRejection::invalid("stream progress request has no delivery channel")
        })?;
        let policy = request.stream_policy;

        let expected = self
            .stream_sequence_cursors
            .get(&progress.session)
            .copied()
            .unwrap_or(0);
        if progress.output.sequence != expected {
            return Err(StreamProgressRejection::invalid(format!(
                "stream progress sequence {} did not match expected {}",
                progress.output.sequence, expected
            )));
        }
        let next = expected.checked_add(1).ok_or_else(|| {
            StreamProgressRejection::invalid("stream progress sequence space was exhausted")
        })?;
        self.stream_sequence_cursors
            .insert(progress.session.clone(), next);
        self.incremental_stream_sessions
            .insert(progress.session.clone());
        record_engine_stream_checkpoint_committed();

        Ok(CommittedStreamDelivery::from_progress(
            progress.session.clone(),
            tx,
            policy,
            progress,
        ))
    }

    fn record_unconfirmed_cleanup(&mut self, session: &super::SessionKey) {
        let attempt = self
            .scheduler
            .pending_cleanup_attempts(session)
            .unwrap_or_default()
            .saturating_add(1);
        let retry_at = Instant::now() + self.retry_policy.cleanup_delay(attempt);
        self.scheduler.record_cleanup_retry(session, retry_at);
        if attempt == 1 || attempt.is_power_of_two() {
            warn!(
                request_id = %session.request_id,
                session_epoch = session.epoch,
                cleanup_attempt = attempt,
                retry_in_ms = self.retry_policy.cleanup_delay(attempt).as_millis(),
                "Executor could not confirm exact-session cleanup; cache remains quarantined"
            );
        }
    }

    async fn attempt_pending_release_cleanup(&mut self, session: &super::SessionKey) {
        let Some(confirmation_required) = self
            .scheduler
            .pending_release_confirmation_required(session)
        else {
            return;
        };
        let release = self.executor.cleanup_session(session).await;
        if release.confirmed || !confirmation_required {
            self.scheduler.confirm_session_release(session);
        } else {
            self.record_unconfirmed_cleanup(session);
        }
    }

    async fn begin_terminal_release(
        &mut self,
        session: &super::SessionKey,
        cause: TerminalReleaseCause,
    ) {
        if matches!(
            self.scheduler.begin_terminal_release(session, cause),
            BeginTerminalRelease::Started { .. }
        ) {
            self.attempt_pending_release_cleanup(session).await;
        }
        self.request_managed_session_release(session);
        self.clear_exact_execution_state(session);
    }

    async fn reconcile_due_cleanup(&mut self) {
        let sessions = self
            .scheduler
            .due_cleanup_sessions(Instant::now(), self.retry_policy.cleanup_budget_per_step);
        for session in sessions {
            self.attempt_pending_release_cleanup(&session).await;
        }
    }

    /// Execute one step of the inference loop.
    ///
    /// The step consists of:
    /// 1. Schedule - select requests to process
    /// 2. Execute - run forward pass
    /// 3. Process - handle outputs, check stop conditions
    pub async fn step(&mut self) -> Result<Vec<EngineOutput>> {
        let Some(prepared) = self.prepare_step().await? else {
            return Ok(Vec::new());
        };
        let executed = self.execute_prepared_with_progress(prepared).await?;
        let committed = self.commit_step(executed).await?;
        let mut outputs = committed.outputs;
        let failed_streams = deliver_committed_streams(committed.stream_deliveries).await;
        self.reconcile_stream_delivery_failures(&mut outputs, failed_streams)
            .await;
        Ok(outputs)
    }

    async fn execute_prepared_with_progress(
        &mut self,
        prepared: PreparedEngineStep,
    ) -> Result<ExecutedEngineStep> {
        let (progress_tx, mut progress_rx) = mpsc::channel(STREAM_PROGRESS_QUEUE_CAPACITY);
        let progress_budget = StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES);
        let mut runner = tokio::spawn(ExecutionGroupRunner::execute(
            prepared,
            progress_tx,
            progress_budget,
        ));
        let (mut deliveries, mut delivery_failures) = IncrementalStreamDeliveryWorkers::new();
        let mut failures = HashMap::new();
        let mut progress_closed = false;
        let mut delivery_failures_closed = false;

        let mut executed = loop {
            tokio::select! {
                result = &mut runner => {
                    break match result {
                        Ok(executed) => executed,
                        Err(error) if error.is_panic() => {
                            std::panic::resume_unwind(error.into_panic())
                        }
                        Err(error) => {
                            return Err(Error::InferenceError(format!(
                                "execution group task was cancelled: {error}"
                            )));
                        }
                    };
                }
                progress = progress_rx.recv(), if !progress_closed => {
                    match progress {
                        Some(progress) => {
                            self.enqueue_incremental_progress(
                                progress,
                                &mut failures,
                                &mut deliveries,
                            );
                        }
                        None => progress_closed = true,
                    }
                }
                failure = delivery_failures.recv(), if !delivery_failures_closed => {
                    match failure {
                        Some(failure) => self.record_stream_failure(
                            failure,
                            &mut failures,
                            &mut deliveries,
                        ),
                        None => delivery_failures_closed = true,
                    }
                }
            }
        };

        while let Some(progress) = progress_rx.recv().await {
            self.enqueue_incremental_progress(progress, &mut failures, &mut deliveries);
        }
        let barrier_failures = deliveries.finish().await;
        while let Ok(failure) = delivery_failures.try_recv() {
            self.cancel_failed_stream(&failure);
            failures.entry(failure.session.clone()).or_insert(failure);
        }
        for failure in barrier_failures {
            self.cancel_failed_stream(&failure);
            failures.entry(failure.session.clone()).or_insert(failure);
        }
        let failures = failures.into_values().collect::<Vec<_>>();
        executed.apply_stream_delivery_failures(&failures);
        Ok(executed)
    }

    fn commit_progress_delivery(
        &mut self,
        progress: FencedStreamProgress,
    ) -> std::result::Result<CommittedStreamDelivery, StreamDeliveryFailure> {
        let session = progress.session.clone();
        match self.commit_incremental_stream_progress(progress) {
            Ok(delivery) => Ok(delivery),
            Err(error) => {
                warn!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    error = %error,
                    "Rejecting invalid incremental stream progress"
                );
                Err(StreamDeliveryFailure {
                    session,
                    kind: error.kind,
                })
            }
        }
    }

    fn cancel_failed_stream(&self, failure: &StreamDeliveryFailure) {
        if let Some(request) = self.requests.get(&failure.session.request_id) {
            if self.scheduler.get_sequence_id(&failure.session.request_id)
                == Some(failure.session.epoch)
            {
                if let Some(cancellation) = request.cancellation.as_ref() {
                    cancellation.store(true, std::sync::atomic::Ordering::Release);
                }
            }
        }
    }

    fn record_stream_failure(
        &self,
        failure: StreamDeliveryFailure,
        failures: &mut HashMap<super::SessionKey, StreamDeliveryFailure>,
        deliveries: &mut IncrementalStreamDeliveryWorkers,
    ) {
        self.cancel_failed_stream(&failure);
        deliveries.abandon_session(&failure.session);
        failures.entry(failure.session.clone()).or_insert(failure);
    }

    fn enqueue_incremental_progress(
        &mut self,
        progress: FencedStreamProgress,
        failures: &mut HashMap<super::SessionKey, StreamDeliveryFailure>,
        deliveries: &mut IncrementalStreamDeliveryWorkers,
    ) {
        if failures.contains_key(&progress.session) {
            return;
        }
        let result = match self.commit_progress_delivery(progress) {
            Ok(delivery) => deliveries.enqueue(delivery),
            Err(failure) => Err(failure),
        };
        if let Err(failure) = result {
            self.record_stream_failure(failure, failures, deliveries);
        }
    }

    /// Prepare an immutable execution transaction under the engine state lock.
    pub(super) async fn prepare_step(&mut self) -> Result<Option<PreparedEngineStep>> {
        // Ensure initialized
        if !self.initialized {
            self.initialize().await?;
        }

        // Phase 1: Schedule
        self.refresh_scheduler_execution_profiles().await;
        self.reconcile_due_cleanup().await;
        let schedule_result = self.scheduler.schedule();

        // Reconcile capability-authoritative state for expired sessions before
        // newly scheduled work executes.
        for expired in &schedule_result.expired_requests {
            let session = expired.session_key();
            self.attempt_pending_release_cleanup(&session).await;
        }

        for request in &schedule_result.expired_requests {
            self.pending_terminal_outputs
                .push_back(CommittedExecutorOutput {
                    session: request.session_key(),
                    output: ExecutorOutput::terminal(request.request_id.clone()),
                    disposition: ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut),
                    provenance: OutcomeProvenance::deadline(
                        DeadlinePhase::SchedulerQueue,
                        DispatchState::NotStarted,
                    ),
                    staged_stream_outputs: Vec::new(),
                });
        }

        if !schedule_result.has_execution_work() && self.pending_terminal_outputs.is_empty() {
            return Ok(None);
        }

        debug!(
            "Scheduled {} prefill, {} decode requests",
            schedule_result.prefill_requests.len(),
            schedule_result.decode_requests.len()
        );

        let prefill_scheduled = schedule_result.prefill_requests.clone();
        let decode_scheduled = schedule_result.decode_requests.clone();
        let all_scheduled = decode_scheduled
            .iter()
            .chain(prefill_scheduled.iter())
            .cloned()
            .collect::<Vec<_>>();
        for scheduled in &all_scheduled {
            if let Err(error) = self.begin_execution_plan(scheduled).await {
                self.rollback_unexecuted_schedule(&all_scheduled);
                return Err(error);
            }
        }
        let now = Instant::now();

        // Capture queue wait for first scheduling event.
        for scheduled in decode_scheduled.iter().chain(prefill_scheduled.iter()) {
            let request_id = &scheduled.request_id;
            let timing = self
                .request_phase_timings
                .entry(request_id.clone())
                .or_default();
            if timing.first_scheduled_at.is_none() {
                timing.first_scheduled_at = Some(now);
                if let Some(started) = self.request_start_times.get(request_id) {
                    timing.queue_wait_ms = started.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        let prefill_requests: Vec<Arc<EngineCoreRequest>> = prefill_scheduled
            .iter()
            .filter_map(|s| self.requests.get(&s.request_id).cloned())
            .collect();
        let decode_requests: Vec<Arc<EngineCoreRequest>> = decode_scheduled
            .iter()
            .filter_map(|s| self.requests.get(&s.request_id).cloned())
            .collect();

        if prefill_requests.is_empty()
            && decode_requests.is_empty()
            && self.pending_terminal_outputs.is_empty()
        {
            return Ok(None);
        }

        let mut capacity_blocked = Vec::new();
        let decode_batches = match self.form_physical_batches(
            &decode_requests,
            &decode_scheduled,
            &mut capacity_blocked,
        ) {
            Ok(batches) => batches,
            Err(Error::Backpressure(reason)) => {
                self.defer_unexecuted_schedule_for_capacity(&all_scheduled);
                debug!(reason = %reason, "Deferring scheduled work for managed KV capacity");
                return Ok(None);
            }
            Err(error) => {
                self.rollback_unexecuted_schedule(&all_scheduled);
                return Err(error);
            }
        };
        let prefill_batches = match self.form_physical_batches(
            &prefill_requests,
            &prefill_scheduled,
            &mut capacity_blocked,
        ) {
            Ok(batches) => batches,
            Err(Error::Backpressure(reason)) => {
                self.defer_unexecuted_schedule_for_capacity(&all_scheduled);
                debug!(reason = %reason, "Deferring scheduled work for managed KV capacity");
                return Ok(None);
            }
            Err(error) => {
                self.rollback_unexecuted_schedule(&all_scheduled);
                return Err(error);
            }
        };
        self.defer_unexecuted_schedule_for_capacity(&capacity_blocked);
        if decode_batches.is_empty()
            && prefill_batches.is_empty()
            && self.pending_terminal_outputs.is_empty()
        {
            return Ok(None);
        }

        Ok(Some(PreparedEngineStep::new(
            self.executor.clone(),
            decode_batches,
            prefill_batches,
        )))
    }

    /// Commit one completed execution transaction under the engine state lock.
    pub(super) async fn commit_step(
        &mut self,
        executed: ExecutedEngineStep,
    ) -> Result<CommittedEngineStep> {
        let ExecutedEngineStep { batches } = executed;

        let result_capacity = batches
            .iter()
            .map(|batch| batch.results.len())
            .sum::<usize>();
        let mut executor_outputs =
            Vec::with_capacity(result_capacity + self.pending_terminal_outputs.len());
        let mut stream_deliveries = Vec::new();
        for mut batch in batches {
            self.active_stream_batches
                .remove(&batch.physical_batch.batch_id);
            if let Err(error) = batch
                .report
                .validate_against(&batch.physical_batch, &self.active_plans)
            {
                warn!(
                    batch_id = batch.physical_batch.batch_id.get(),
                    error = %error,
                    "Rejecting an invalid physical batch report before state commit"
                );
                let message = format!("invalid physical batch report: {error}");
                let dispatch_state =
                    if batch.report.dispatch.kind == super::BatchDispatchKind::NotDispatched {
                        DispatchState::NotStarted
                    } else {
                        DispatchState::Started
                    };
                let failure_dispatch = if dispatch_state == DispatchState::NotStarted {
                    BatchDispatch::not_dispatched(batch.physical_batch.rows.len().max(1))
                } else {
                    batch.physical_batch.expected_dispatch()
                };
                for result in &mut batch.results {
                    result.output =
                        ExecutorOutput::error(result.session.request_id.clone(), message.clone());
                    result.disposition = ExecutionDisposition::Failed(
                        ExecutionFailure::invalid_output(message.clone()),
                    );
                    result.safe_point = true;
                    result.dispatch = failure_dispatch;
                    result.provenance =
                        OutcomeProvenance::failure(FailureOrigin::StateCommit, dispatch_state);
                    result.observed_resources = ResourceVector::zero();
                    result.staged_stream_outputs.clear();
                    result.managed_cache = None;
                }
            } else {
                record_engine_physical_batch(&batch.physical_batch, batch.report.dispatch);
            }

            for result in batch.results {
                let provenance = result.provenance;
                let entered_model = result.provenance.dispatch_state != DispatchState::NotStarted;
                let step_time_ms = if entered_model {
                    batch.report.elapsed.as_secs_f64() * 1000.0
                } else {
                    0.0
                };
                if entered_model {
                    if let Some(timing) = self
                        .request_phase_timings
                        .get_mut(&result.session.request_id)
                    {
                        match batch.phase {
                            ExecutionPhase::Decode => {
                                timing.decode_ms += step_time_ms;
                                timing.decode_steps = timing.decode_steps.saturating_add(1);
                            }
                            ExecutionPhase::Prefill => {
                                timing.prefill_ms += step_time_ms;
                                timing.prefill_steps = timing.prefill_steps.saturating_add(1);
                            }
                        }
                    }
                }
                match self.commit_executor_result(result, step_time_ms).await {
                    Some(committed) => executor_outputs.push(committed),
                    None => record_engine_execution_outcome(provenance),
                }
            }
        }
        // Terminal events are a durable outbox until all fallible work for the
        // step has completed. Draining earlier can lose an abort/deadline event
        // when maintenance or execution-plan preparation returns an error.
        executor_outputs.extend(self.pending_terminal_outputs.drain(..));

        // Phase 3: Process outputs
        let mut outputs = Vec::new();

        for committed in executor_outputs {
            record_engine_execution_outcome(committed.provenance);
            let CommittedExecutorOutput {
                session,
                output: exec_output,
                disposition,
                provenance,
                staged_stream_outputs,
            } = committed;
            let request_id = exec_output.request_id.clone();

            if !staged_stream_outputs.is_empty() {
                if let Some(request) = self.requests.get(&request_id) {
                    if let Some(tx) = request.streaming_tx.clone() {
                        stream_deliveries.push(CommittedStreamDelivery::new(
                            session.clone(),
                            tx,
                            request.stream_policy,
                            staged_stream_outputs,
                        ));
                    }
                }
            }

            // Get timing info
            let generation_time = self
                .request_start_times
                .get(&request_id)
                .map(|t| t.elapsed())
                .unwrap_or_default();
            let generation_time_ms = generation_time.as_secs_f64() * 1000.0;

            let sequence_id = session.epoch;

            if let Some(override_timing) = exec_output.phase_timing_override.as_ref() {
                let phase = self
                    .request_phase_timings
                    .entry(request_id.clone())
                    .or_default();
                merge_optional_phase_ms(
                    &mut phase.media_decode_ms,
                    override_timing.media_decode_ms,
                );
                merge_optional_phase_ms(
                    &mut phase.normalization_ms,
                    override_timing.normalization_ms,
                );
                if let Some(prefill_ms) = override_timing.prefill_ms {
                    phase.prefill_ms = prefill_ms.max(0.0);
                }
                if let Some(decode_ms) = override_timing.decode_ms {
                    phase.decode_ms = decode_ms.max(0.0);
                }
                merge_optional_phase_ms(&mut phase.sampling_ms, override_timing.sampling_ms);
                merge_optional_phase_ms(&mut phase.codec_ms, override_timing.codec_ms);
                merge_optional_phase_ms(&mut phase.postprocess_ms, override_timing.postprocess_ms);
                if let Some(prefill_steps) = override_timing.prefill_steps {
                    phase.prefill_steps = prefill_steps;
                }
                if let Some(decode_steps) = override_timing.decode_steps {
                    phase.decode_steps = decode_steps;
                }
                if let Some(first_output_ms_since_start) =
                    override_timing.first_output_ms_since_start
                {
                    let request_relative_first_output = (phase.queue_wait_ms
                        + first_output_ms_since_start.max(0.0))
                    .min(generation_time_ms);
                    phase
                        .first_output_ms
                        .get_or_insert(request_relative_first_output);
                }
            }

            // Process output
            let mut engine_output = self.output_processor.process_execution(
                exec_output.clone(),
                &disposition,
                sequence_id,
                generation_time,
            );
            engine_output.provenance = provenance;
            engine_output.token_stats.prompt_tokens = self
                .requests
                .get(&request_id)
                .map(|request| request.num_prompt_tokens())
                .unwrap_or(engine_output.token_stats.prompt_tokens);
            if exec_output.finished {
                if let Some((_, total_generated)) = self.scheduler.get_running_info(&request_id) {
                    let resolved_total = total_generated.max(engine_output.num_tokens);
                    engine_output.num_tokens = resolved_total;
                    engine_output.token_stats.generated_tokens = resolved_total;
                }
            }
            if Self::has_user_visible_output(&exec_output) {
                if let Some(phase) = self.request_phase_timings.get_mut(&request_id) {
                    phase.first_output_ms.get_or_insert(generation_time_ms);
                }
            }
            if let Some(phase) = self.request_phase_timings.get(&request_id).cloned() {
                engine_output.token_stats.prefill_time_ms = phase.prefill_ms as f32;
                engine_output.token_stats.decode_time_ms = phase.decode_ms as f32;
                if phase.decode_ms > 0.0 {
                    engine_output.token_stats.tokens_per_second =
                        (engine_output.token_stats.generated_tokens as f64 * 1000.0
                            / phase.decode_ms) as f32;
                }
                engine_output.latency_breakdown = Some(LatencyBreakdown {
                    queue_wait_ms: phase.queue_wait_ms,
                    media_decode_ms: phase.media_decode_ms,
                    normalization_ms: phase.normalization_ms,
                    prefill_ms: phase.prefill_ms,
                    decode_ms: phase.decode_ms,
                    sampling_ms: phase.sampling_ms,
                    codec_ms: phase.codec_ms,
                    postprocess_ms: phase.postprocess_ms,
                    ttft_ms: phase.first_output_ms,
                    total_ms: generation_time_ms,
                    prefill_steps: phase.prefill_steps,
                    decode_steps: phase.decode_steps,
                });
            }

            // Update scheduler state only from the authoritative disposition.
            if let Some(cause) = Self::terminal_release_cause(&disposition) {
                self.begin_terminal_release(&session, cause).await;
                self.requests.remove(&request_id);
                self.request_start_times.remove(&request_id);
                self.request_phase_timings.remove(&request_id);
                self.clear_exact_execution_state(&session);
                debug!("Finished request {}", request_id);
            }

            outputs.push(engine_output);
        }

        Ok(CommittedEngineStep {
            outputs,
            stream_deliveries,
        })
    }

    /// Confirm that a terminal output has been routed outside the core.
    ///
    /// The exact scheduler session remains fenced after [`Self::step`] returns
    /// its terminal output. Callers must acknowledge that same session only
    /// after placing the output in their delivery channel or return batch.
    pub fn acknowledge_terminal_output(&mut self, session: &super::SessionKey) -> bool {
        self.scheduler.mark_terminal_delivered(session)
    }

    /// Cancel a synchronous caller's exact session after its completion
    /// receiver is abandoned. There is no consumer for the cancellation event,
    /// so discard that exact terminal output and mark delivery complete while
    /// retaining the scheduler quarantine until executor cleanup is confirmed.
    pub(crate) async fn abandon_request_session(&mut self, session: &super::SessionKey) -> bool {
        let aborted = self.abort_request_session(session).await;
        let has_queued_terminal = self
            .pending_terminal_outputs
            .iter()
            .any(|pending| pending.session == *session);
        if !aborted && !has_queued_terminal {
            return false;
        }
        self.pending_terminal_outputs
            .retain(|pending| pending.session != *session);
        self.acknowledge_terminal_output(session);
        true
    }

    /// Retry cleanup for an abandoned exact session and return the bounded
    /// delay before another attempt. `None` means the quarantine is gone.
    pub(crate) async fn retry_abandoned_session_cleanup(
        &mut self,
        session: &super::SessionKey,
    ) -> Option<Duration> {
        self.attempt_pending_release_cleanup(session).await;
        self.scheduler
            .pending_cleanup_attempts(session)
            .map(|attempt| self.retry_policy.cleanup_delay(attempt.max(1)))
    }

    pub(crate) fn abandoned_session_cleanup_delay(
        &self,
        session: &super::SessionKey,
    ) -> Option<Duration> {
        self.scheduler
            .pending_cleanup_attempts(session)
            .map(|attempt| self.retry_policy.cleanup_delay(attempt.max(1)))
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        !self.pending_terminal_outputs.is_empty()
            || self.scheduler.has_pending_work()
            || self.scheduler.has_due_cleanup(Instant::now())
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
    pub(crate) fn set_hard_deadline_for_test(
        &mut self,
        request_id: &RequestId,
        deadline: Instant,
    ) -> bool {
        self.scheduler
            .set_hard_deadline_for_test(request_id, deadline)
    }

    async fn terminate_request_session(
        &mut self,
        session: &super::SessionKey,
        output: ExecutorOutput,
        disposition: ExecutionDisposition,
        provenance: OutcomeProvenance,
    ) -> bool {
        if self.scheduler.get_sequence_id(&session.request_id) != Some(session.epoch) {
            return false;
        }

        let Some(cause) = Self::terminal_release_cause(&disposition) else {
            return false;
        };
        self.begin_terminal_release(session, cause).await;
        self.requests.remove(&session.request_id);
        self.clear_exact_execution_state(session);
        self.pending_terminal_outputs
            .push_back(CommittedExecutorOutput {
                session: session.clone(),
                output,
                disposition,
                provenance,
                staged_stream_outputs: Vec::new(),
            });
        true
    }

    pub(crate) async fn handle_stream_delivery_failure(
        &mut self,
        failure: StreamDeliveryFailure,
    ) -> bool {
        let session = failure.session;
        let (output, disposition, provenance) = match failure.kind {
            StreamDeliveryFailureKind::Delivery => {
                let message = "committed stream delivery failed";
                (
                    ExecutorOutput::error(session.request_id.clone(), message),
                    ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message)),
                    OutcomeProvenance::failure(
                        FailureOrigin::StreamDelivery,
                        DispatchState::ProducedOutput,
                    ),
                )
            }
            StreamDeliveryFailureKind::Deadline => (
                ExecutorOutput::terminal(session.request_id.clone()),
                ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut),
                OutcomeProvenance::deadline(
                    DeadlinePhase::StreamDelivery,
                    DispatchState::ProducedOutput,
                ),
            ),
            StreamDeliveryFailureKind::Cancelled => (
                ExecutorOutput::cancelled(session.request_id.clone()),
                ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled),
                OutcomeProvenance::produced_output(),
            ),
            StreamDeliveryFailureKind::RequestDeadline => (
                ExecutorOutput::terminal(session.request_id.clone()),
                ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut),
                OutcomeProvenance::deadline(
                    DeadlinePhase::ModelExecution,
                    DispatchState::ProducedOutput,
                ),
            ),
            StreamDeliveryFailureKind::InvalidProgress => {
                let message = "executor emitted invalid incremental stream progress";
                (
                    ExecutorOutput::error(session.request_id.clone(), message),
                    ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message)),
                    OutcomeProvenance::failure(
                        FailureOrigin::ExecutorValidation,
                        DispatchState::ProducedOutput,
                    ),
                )
            }
        };
        self.terminate_request_session(&session, output, disposition, provenance)
            .await
    }

    fn overlay_terminal_stream_delivery_failure(
        output: &mut EngineOutput,
        kind: StreamDeliveryFailureKind,
    ) {
        let (message, finish_reason, provenance) = match kind {
            StreamDeliveryFailureKind::Delivery => (
                "committed stream delivery failed",
                OutputFinishReason::Error,
                OutcomeProvenance::failure(
                    FailureOrigin::StreamDelivery,
                    DispatchState::ProducedOutput,
                ),
            ),
            StreamDeliveryFailureKind::Deadline => (
                REQUEST_DEADLINE_EXCEEDED,
                OutputFinishReason::Error,
                OutcomeProvenance::deadline(
                    DeadlinePhase::StreamDelivery,
                    DispatchState::ProducedOutput,
                ),
            ),
            StreamDeliveryFailureKind::Cancelled => (
                "request cancelled",
                OutputFinishReason::Aborted,
                OutcomeProvenance::produced_output(),
            ),
            StreamDeliveryFailureKind::RequestDeadline => (
                REQUEST_DEADLINE_EXCEEDED,
                OutputFinishReason::Error,
                OutcomeProvenance::deadline(
                    DeadlinePhase::ModelExecution,
                    DispatchState::ProducedOutput,
                ),
            ),
            StreamDeliveryFailureKind::InvalidProgress => (
                "executor emitted invalid incremental stream progress",
                OutputFinishReason::Error,
                OutcomeProvenance::failure(
                    FailureOrigin::ExecutorValidation,
                    DispatchState::ProducedOutput,
                ),
            ),
        };
        output.is_finished = true;
        output.finish_reason = Some(finish_reason);
        output.error = Some(message.to_string());
        output.provenance = provenance;
    }

    /// Reconcile failures from the committed stream outbox with the exact
    /// public output for this step. A terminal result has already completed its
    /// scheduler lifecycle, so replace that result in place instead of queuing
    /// a second terminal event. Non-terminal rows still need normal exact-
    /// session termination.
    pub(crate) async fn reconcile_stream_delivery_failures(
        &mut self,
        outputs: &mut [EngineOutput],
        failures: Vec<StreamDeliveryFailure>,
    ) {
        for failure in failures {
            if let Some(output) = outputs.iter_mut().find(|output| {
                output.is_finished
                    && output.request_id == failure.session.request_id
                    && output.sequence_id == failure.session.epoch
            }) {
                Self::overlay_terminal_stream_delivery_failure(output, failure.kind);
            } else {
                self.handle_stream_delivery_failure(failure).await;
            }
        }
    }

    /// Abort only if the caller still owns the exact request incarnation.
    pub async fn abort_request_session(&mut self, session: &super::SessionKey) -> bool {
        let aborted = self
            .terminate_request_session(
                session,
                ExecutorOutput::terminal(session.request_id.clone()),
                ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled),
                OutcomeProvenance::not_started(),
            )
            .await;
        if !aborted {
            return false;
        }
        debug!(
            request_id = %session.request_id,
            session_epoch = session.epoch,
            "Queued exact-session cancellation"
        );
        true
    }

    /// Abort a request.
    pub async fn abort_request(&mut self, request_id: &RequestId) -> bool {
        let Some(session) = self.get_session_key(request_id) else {
            return false;
        };
        self.abort_request_session(&session).await
    }

    /// Abort all active requests that target a specific model variant.
    pub async fn abort_requests_for_variant(&mut self, variant: ModelVariant) -> Vec<RequestId> {
        let request_ids: Vec<RequestId> = self
            .requests
            .iter()
            .filter_map(|(request_id, request)| {
                if request.model_variant == Some(variant) {
                    Some(request_id.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut aborted = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            if self.abort_request(&request_id).await {
                aborted.push(request_id);
            }
        }
        aborted
    }

    /// Purge reusable executor cache state owned by one model variant.
    pub async fn purge_model_cache(&mut self, variant: ModelVariant) -> CacheReleaseReport {
        self.executor.purge_model_cache(variant).await
    }

    /// Allocate and admit physical KV backing while the authoritative model
    /// generation is still Loading. Ready publication happens only after this
    /// succeeds, and request admission performs lookup only.
    pub(crate) fn load_managed_model_cache(
        &mut self,
        model_instance: super::ModelInstanceId,
        capability: &crate::kv::InferenceStateCapability,
        logical_context_tokens: Option<usize>,
    ) -> Result<Option<Arc<super::ManagedKvModelRuntime>>> {
        self.load_managed_model_cache_with_capacity_policy(
            model_instance,
            capability,
            logical_context_tokens,
            None,
            false,
        )
    }

    /// Load managed state while allowing a model lifecycle to distinguish the
    /// scheduler's retained-session capacity from its simultaneously staged
    /// transaction width. `None` preserves the engine-wide legacy behavior.
    pub(crate) fn load_managed_model_cache_with_capacity_policy(
        &mut self,
        model_instance: super::ModelInstanceId,
        capability: &crate::kv::InferenceStateCapability,
        logical_context_tokens: Option<usize>,
        staged_transaction_rows: Option<u32>,
        fit_cuda_contiguous_context: bool,
    ) -> Result<Option<Arc<super::ManagedKvModelRuntime>>> {
        let Some(contract) = capability.managed_contract() else {
            return Ok(None);
        };
        let backend = self.managed_kv_cache.worker_backend();
        let retained_sequence_rows = u32::try_from(self.config.max_batch_size)
            .map_err(|_| Error::InvalidInput("managed state sequence limit exceeds u32".into()))?;
        let fit_cuda_contiguous_context =
            backend == BackendKind::Cuda && fit_cuda_contiguous_context;
        let staged_transaction_rows = staged_transaction_rows.unwrap_or(retained_sequence_rows);
        let logical_token_reach = if fit_cuda_contiguous_context {
            let maximum_tokens = logical_context_tokens.ok_or_else(|| {
                Error::ModelLoadError(
                    "CUDA managed state requires a positive loaded-model context limit".into(),
                )
            })?;
            Some(
                self.managed_kv_cache
                    .fit_cuda_contiguous_logical_token_reach(
                        model_instance,
                        contract,
                        u64::try_from(maximum_tokens).map_err(|_| {
                            Error::ModelLoadError("model context exceeds u64".into())
                        })?,
                        self.config.portable_context_reserve_bytes,
                        self.config.block_size,
                        retained_sequence_rows,
                        staged_transaction_rows,
                    )?,
            )
        } else {
            self.resolve_managed_token_reach_for_contract(
                backend,
                model_instance,
                contract,
                logical_context_tokens,
                1,
            )?
        };
        let runtime = self
            .managed_kv_cache
            .bind_request_with_capacity(
                model_instance,
                backend,
                ManagedStateCapacityRequest {
                    total_paged_pages: u32::try_from(self.config.max_blocks).map_err(|_| {
                        Error::InvalidInput("managed KV page budget exceeds u32".into())
                    })?,
                    logical_token_reach,
                    retained_sequence_rows,
                    staged_transaction_rows,
                },
                self.config.block_size,
                capability,
            )?
            .ok_or_else(|| {
                Error::InferenceError(
                    "managed KV load did not install a physical runtime".to_string(),
                )
            })?;
        runtime.synchronize_backing()?;
        Ok(Some(runtime))
    }

    pub(crate) fn load_managed_model_state(
        &mut self,
        model_instance: super::ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
    ) -> Result<Arc<super::ManagedKvModelRuntime>> {
        self.load_managed_model_state_with_portable_copies(
            model_instance,
            retained_state,
            logical_context_tokens,
            1,
        )
    }

    pub(crate) fn load_managed_model_state_with_portable_copies(
        &mut self,
        model_instance: super::ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
        portable_state_copies: u32,
    ) -> Result<Arc<super::ManagedKvModelRuntime>> {
        let backend = self.managed_kv_cache.worker_backend();
        let logical_token_reach = self.resolve_managed_token_reach_for_contract(
            backend,
            model_instance,
            retained_state,
            logical_context_tokens,
            portable_state_copies,
        )?;
        let runtime = self.managed_kv_cache.bind_model_state_with_capacity(
            model_instance,
            backend,
            ManagedStateCapacityRequest {
                total_paged_pages: u32::try_from(self.config.max_blocks).map_err(|_| {
                    Error::InvalidInput("managed KV page budget exceeds u32".into())
                })?,
                logical_token_reach,
                retained_sequence_rows: u32::try_from(self.config.max_batch_size).map_err(
                    |_| Error::InvalidInput("managed state sequence limit exceeds u32".into()),
                )?,
                staged_transaction_rows: u32::try_from(self.config.max_batch_size).map_err(
                    |_| Error::InvalidInput("managed state transaction limit exceeds u32".into()),
                )?,
            },
            self.config.block_size,
            retained_state,
        )?;
        runtime.synchronize_backing()?;
        Ok(runtime)
    }

    fn resolve_managed_token_reach_for_contract(
        &self,
        backend: BackendKind,
        model_instance: super::ModelInstanceId,
        contract: &crate::kv::v2::InferenceStateContract,
        loaded_context: Option<usize>,
        portable_state_copies: u32,
    ) -> Result<Option<u64>> {
        if !self.config.portable_context_auto
            && backend != BackendKind::Cuda
            && loaded_context.is_none()
        {
            return Ok(None);
        }
        let loaded_context = loaded_context.ok_or_else(|| {
            Error::ModelLoadError(format!(
                "{backend:?} managed state requires a positive loaded-model context limit"
            ))
        })?;
        if self.config.portable_context_auto && backend != BackendKind::Cuda {
            let maximum_tokens = u64::try_from(loaded_context)
                .map_err(|_| Error::ModelLoadError("model context exceeds u64".into()))?;
            return self
                .managed_kv_cache
                .fit_portable_logical_token_reach(
                    model_instance,
                    contract,
                    maximum_tokens,
                    self.config.portable_context_reserve_bytes,
                    self.config.block_size,
                    u32::try_from(self.config.max_batch_size).map_err(|_| {
                        Error::InvalidInput("managed state sequence limit exceeds u32".into())
                    })?,
                    u32::try_from(self.config.max_batch_size).map_err(|_| {
                        Error::InvalidInput("managed state transaction limit exceeds u32".into())
                    })?,
                    portable_state_copies,
                )
                .map(Some);
        }
        resolve_managed_token_reach(backend, self.config.max_seq_len, Some(loaded_context))
    }

    pub(crate) fn load_retained_tensor_state(
        &mut self,
        model_instance: super::ModelInstanceId,
        contract: &crate::kv::v2::InferenceStateContract,
        sequence_capacity: u32,
    ) -> Result<Arc<super::RetainedTensorStateRuntimeV2>> {
        self.physical_state
            .allocate_retained_tensor(model_instance, contract, sequence_capacity)
    }

    pub(crate) fn resolve_and_load_invocation_workspace(
        &mut self,
        model_instance: super::ModelInstanceId,
        adapter_instance: super::AdapterInstanceId,
        stage_graph: [u8; 32],
        stage: super::StageId,
        contract: &crate::kv::v2::InferenceStateContract,
        domain: &crate::kv::v2::InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<Arc<dyn crate::kv::v2::InvocationWorkspaceBackingV2>> {
        self.physical_state
            .resolve_and_allocate_invocation_workspace(
                model_instance,
                InvocationPhysicalKey {
                    adapter_instance,
                    stage_graph,
                    stage,
                    domain: domain.id(),
                },
                contract,
                domain,
                slot_count,
            )
    }

    /// Retire physical managed-KV state for one exact loaded-model instance.
    /// Variant-only purge is insufficient because an unload/reload may publish
    /// a different generation of the same variant.
    pub fn unload_managed_model_cache(
        &mut self,
        model_instance: super::ModelInstanceId,
    ) -> Result<bool> {
        let invocation = self.physical_state.unload_model(model_instance)?;
        let retained = self.managed_kv_cache.unload_model(model_instance)?;
        Ok(invocation || retained)
    }

    /// Abort every request tracked by the core and release executor state.
    pub async fn abort_all_requests(&mut self) -> Vec<RequestId> {
        let request_ids: Vec<_> = self.requests.keys().cloned().collect();
        let mut aborted = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            if self.abort_request(&request_id).await {
                aborted.push(request_id);
            }
        }
        aborted
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

    /// Shutdown the engine core.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down engine core");

        // Abort all pending requests
        let request_ids: Vec<_> = self.requests.keys().cloned().collect();
        for id in request_ids {
            self.abort_request(&id).await;
        }

        // Shutdown executor
        self.executor.shutdown().await?;

        // A successful executor shutdown is the final physical-release fence.
        self.scheduler.force_release_all_after_executor_shutdown();
        self.requests.clear();
        self.request_start_times.clear();
        self.request_phase_timings.clear();
        self.execution_trackers.clear();
        self.active_plans.clear();
        self.active_stream_batches.clear();
        self.stream_sequence_cursors.clear();
        self.incremental_stream_sessions.clear();
        self.pending_terminal_outputs.clear();
        self.execution_retry_attempts.clear();

        self.initialized = false;
        info!("Engine core shutdown complete");

        Ok(())
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
mod tests {
    use super::super::executor::{
        ExecutorOutput, ExecutorPhaseTiming, ExecutorStepResult, ModelExecutor,
    };
    use super::super::scheduler::ScheduledRequest;
    use super::super::types::{AudioOutput, TaskType};
    use super::*;
    use crate::model::ModelVariant;
    use crate::models::shared::chat::{ChatMessage, ChatRole};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn managed_token_reach_honors_portable_context_and_preserves_cuda_ceiling() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(
                resolve_managed_token_reach(backend, 4_096, Some(40_960)).unwrap(),
                Some(4_096)
            );
            assert_eq!(
                resolve_managed_token_reach(backend, 262_144, Some(40_960)).unwrap(),
                Some(40_960)
            );
        }
    }

    #[test]
    fn managed_token_reach_rejects_zero_and_cuda_requires_model_context() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert!(resolve_managed_token_reach(backend, 4_096, Some(0)).is_err());
        }
        assert_eq!(
            resolve_managed_token_reach(BackendKind::Cpu, 4_096, None).unwrap(),
            None
        );
        assert!(resolve_managed_token_reach(BackendKind::Cuda, 4_096, None).is_err());
    }

    fn scheduled_prefill(request_id: &str, sequence_id: u64) -> ScheduledRequest {
        ScheduledRequest {
            plan_id: sequence_id + 1,
            request_id: request_id.to_string(),
            sequence_id,
            num_tokens: 1,
            is_prefill: true,
            num_computed_tokens: 0,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Prefill,
                input: crate::engine::InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        }
    }

    fn scheduled_decode(request_id: &str, sequence_id: u64) -> ScheduledRequest {
        ScheduledRequest {
            plan_id: sequence_id + 101,
            request_id: request_id.to_string(),
            sequence_id,
            num_tokens: 1,
            is_prefill: false,
            num_computed_tokens: 1,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Decode,
                input: crate::engine::InputRange { start: 1, end: 2 },
                max_output_steps: 1,
            },
        }
    }

    fn text_progress(request_id: &str, sequence: usize) -> super::super::StreamingOutput {
        super::super::StreamingOutput {
            request_id: request_id.to_string(),
            sequence,
            samples: Vec::new(),
            sample_rate: 0,
            is_final: false,
            text: Some(format!("delta-{sequence}")),
            stats: None,
            asr_progress: None,
        }
    }

    fn wrap_outputs(
        scheduled: &[ScheduledRequest],
        outputs: Vec<ExecutorOutput>,
    ) -> Vec<ExecutorStepResult> {
        assert_eq!(scheduled.len(), outputs.len());
        scheduled
            .iter()
            .zip(outputs)
            .map(|(scheduled, output)| ExecutorStepResult::new(scheduled, output))
            .collect()
    }

    struct MockExecutor {
        initialized: bool,
        cleanup_calls: Arc<Mutex<Vec<String>>>,
    }

    impl MockExecutor {
        fn new(cleanup_calls: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                initialized: false,
                cleanup_calls,
            }
        }

        fn build_outputs(scheduled: &[ScheduledRequest]) -> Vec<ExecutorOutput> {
            scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: None,
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: usize::from(!entry.is_prefill),
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect()
        }
    }

    impl ModelExecutor for MockExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            profile.recompute_safe = true;
            profile.cache_release_safe = true;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(wrap_outputs(scheduled, Self::build_outputs(scheduled)))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(wrap_outputs(scheduled, Self::build_outputs(scheduled)))
        }

        fn is_ready(&self) -> bool {
            self.initialized
        }

        fn initialize(&mut self) -> Result<()> {
            self.initialized = true;
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            self.initialized = false;
            Ok(())
        }

        fn cleanup_request(&self, request_id: &str) -> super::super::executor::CacheReleaseReport {
            if let Ok(mut calls) = self.cleanup_calls.lock() {
                calls.push(request_id.to_string());
            }
            super::super::executor::CacheReleaseReport::confirmed(1)
        }
    }

    struct ManagedReceiptExecutor;

    impl ModelExecutor for ManagedReceiptExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            profile.cache_mode = CacheMode::ExternalPaged;
            profile.cache_release_safe = true;
            Some(profile)
        }

        fn execute_physical_batch(
            &self,
            execution: super::super::executor::PhysicalBatchExecution<'_>,
        ) -> super::super::executor::PhysicalDispatchResult {
            let dispatch = execution.expected_dispatch();
            Ok(execution
                .scheduled
                .iter()
                .zip(&execution.batch.rows)
                .map(|(scheduled, row)| {
                    let mut result = ExecutorStepResult::new(
                        scheduled,
                        ExecutorOutput {
                            request_id: scheduled.request_id.clone(),
                            audio: None,
                            text: None,
                            input_transcription: None,
                            tokens_processed: scheduled.num_tokens,
                            tokens_generated: 0,
                            finished: false,
                            phase_timing_override: None,
                            asr_diagnostics: None,
                            error: None,
                        },
                    );
                    result.dispatch = dispatch;
                    if let Some(reservation) = &row.managed_cache {
                        result.managed_cache = Some(reservation.completed_write_receipt_for_test());
                    }
                    result
                })
                .collect())
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            unreachable!("physical dispatch is overridden")
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            unreachable!("physical dispatch is overridden")
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
            CacheReleaseReport::confirmed(0)
        }
    }

    struct DeadlineCleanupExecutor {
        events: Arc<Mutex<Vec<String>>>,
        confirm_cleanup: bool,
    }

    impl DeadlineCleanupExecutor {
        fn new(events: Arc<Mutex<Vec<String>>>, confirm_cleanup: bool) -> Self {
            Self {
                events,
                confirm_cleanup,
            }
        }

        fn execute_phase(
            &self,
            phase: &str,
            scheduled: &[ScheduledRequest],
        ) -> Vec<ExecutorStepResult> {
            let mut outputs = Vec::with_capacity(scheduled.len());
            for entry in scheduled {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("execute-{phase}:{}", entry.request_id));
                outputs.push(ExecutorStepResult::new(
                    entry,
                    ExecutorOutput {
                        request_id: entry.request_id.clone(),
                        audio: None,
                        text: None,
                        input_transcription: None,
                        tokens_processed: entry.num_tokens,
                        tokens_generated: 0,
                        finished: false,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                    },
                ));
            }
            outputs
        }
    }

    impl ModelExecutor for DeadlineCleanupExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            // This synthetic executor stands in for an external physical
            // runtime whose exact-session cleanup must be confirmed.
            profile.cache_mode = super::super::execution::CacheMode::ExternalPaged;
            profile.cache_release_safe = true;
            profile.prefix_reuse_safe = true;
            profile.resolved_from_loaded_model = true;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.execute_phase("prefill", scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.execute_phase("decode", scheduled))
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn cleanup_session(
            &self,
            session: &super::super::SessionKey,
        ) -> super::super::executor::CacheReleaseReport {
            self.events
                .lock()
                .unwrap()
                .push(format!("cleanup:{}:{}", session.request_id, session.epoch));
            if self.confirm_cleanup {
                super::super::executor::CacheReleaseReport::confirmed(1)
            } else {
                super::super::executor::CacheReleaseReport::unconfirmed()
            }
        }
    }

    struct ImmediateFinishExecutor;

    impl ModelExecutor for ImmediateFinishExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: Some(format!("done-{}", entry.request_id)),
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: 1,
                    finished: true,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct CancelledExecutor;

    impl ModelExecutor for CancelledExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::from_session(
                        entry,
                        super::super::executor::ModelSessionResult::cancelled(
                            ExecutorOutput::cancelled(entry.request_id.clone()),
                        ),
                    )
                })
                .collect())
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct SelectiveFailureExecutor;

    impl ModelExecutor for SelectiveFailureExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            if scheduled.iter().any(|entry| entry.request_id == "bad") {
                return Err(Error::InferenceError("isolated failure".to_string()));
            }
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: Some("ok".to_string()),
                    input_transcription: None,
                    tokens_processed: entry.num_tokens,
                    tokens_generated: 1,
                    finished: true,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct PhaseTimingExecutor;

    impl ModelExecutor for PhaseTimingExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: Some("done".to_string()),
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: 1,
                    finished: true,
                    phase_timing_override: Some(ExecutorPhaseTiming {
                        media_decode_ms: Some(12.5),
                        sampling_ms: Some(1.25),
                        ..ExecutorPhaseTiming::default()
                    }),
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn bind_managed_test_state(
        core: &mut EngineCore,
        request: &mut EngineCoreRequest,
        model_instance: ModelInstanceId,
    ) {
        let variant = request.model_variant.expect("managed test model");
        let state_contract = crate::kv::test_contract();
        let capability = crate::kv::InferenceStateCapability::Managed(state_contract.clone());
        let physical = core
            .load_managed_model_cache(model_instance, &capability, None)
            .expect("load managed state")
            .expect("physical managed state");
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.cache_mode = CacheMode::ExternalPaged;
        profile.cache_namespace = Some("core-test-state-v2".to_string());
        profile.kv_dtype = "state_v2_resolved".to_string();
        profile.resolved_from_loaded_model = true;
        let stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(1),
            "chat.test.physical",
            &profile,
            NativeBatchMode::None,
        );
        let execution = super::super::ExecutionAdapterBinding {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: model_instance,
            adapter_instance_id: super::super::AdapterInstanceId::new(model_instance.get()),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "chat".to_string(),
            stages: Arc::from([stage]),
        };
        let descriptor = crate::kv::v2::CapabilityStateDescriptorV2::managed_for_stages_test(
            state_contract,
            &execution.stages,
        );
        let runtime = Arc::new(crate::kv::v2::CapabilityStateRuntimeV2::managed(
            crate::kv::v2::ManagedCapabilityRuntimeV2::seal(
                BackendKind::Cpu,
                &execution,
                descriptor,
                physical,
                crate::kv::v2::RetainedStateUseV2::ExternalPaged,
            )
            .expect("seal managed test runtime"),
        ));
        request
            .bind_execution_adapter(execution)
            .expect("bind test execution");
        request
            .bind_v2_state_runtime(runtime.clone(), runtime.state_fingerprint, BackendKind::Cpu)
            .expect("bind managed test state");
    }

    fn managed_test_request(
        core: &mut EngineCore,
        model_instance: ModelInstanceId,
        id: &str,
        tokens: Vec<u32>,
    ) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: id.to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.id = id.to_string();
        request.params.max_tokens = 1;
        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, tokens, None, 4096)
            .expect("prepare prompt");
        request
            .bind_model_instance(model_instance)
            .expect("model instance");
        bind_managed_test_state(core, &mut request, model_instance);
        request
    }

    #[test]
    fn test_engine_core_creation() {
        let config = EngineCoreConfig::default();
        let core = EngineCore::new(config);
        assert!(core.is_ok());
    }

    #[tokio::test]
    async fn test_add_request() {
        let config = EngineCoreConfig::default();
        let mut core = EngineCore::new(config).unwrap();

        let request = EngineCoreRequest::tts("Hello, world!");
        let result = core.add_request(request);
        assert!(result.is_ok());
        assert_eq!(core.pending_request_count(), 1);
    }

    #[test]
    fn v2_state_publication_fails_closed_until_runtime_is_resolved() {
        let mut core = EngineCore::new(EngineCoreConfig::default()).unwrap();
        let mut request = EngineCoreRequest::tts("v2");
        request
            .bind_model_instance(ModelInstanceId::new(101))
            .unwrap();
        request
            .bind_v2_state_descriptor(
                crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_test(),
                [7; 32],
            )
            .unwrap();

        let error = core
            .add_request(request)
            .expect_err("v2 must never fall back to model-owned execution");
        assert!(error.to_string().contains("resolved v2 runtime"));
        assert_eq!(core.pending_request_count(), 0);
    }

    #[test]
    fn load_sealed_stateless_v2_runtime_enters_the_engine() {
        let mut core = EngineCore::new(EngineCoreConfig {
            backend: BackendKind::Cpu,
            ..EngineCoreConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let instance = ModelInstanceId::new(102);
        let profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        let stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(0),
            "tts.compatibility",
            &profile,
            NativeBatchMode::None,
        );
        let execution = super::super::ExecutionAdapterBinding {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: instance,
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage]),
        };
        let descriptor = crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stages_test(
            &execution.stages,
        );
        let runtime = Arc::new(crate::kv::v2::CapabilityStateRuntimeV2::stateless(
            crate::kv::v2::StatelessCapabilityRuntimeV2::seal(
                BackendKind::Cpu,
                &execution,
                descriptor,
            )
            .unwrap(),
        ));
        let mut request = EngineCoreRequest::tts("v2 stateless").with_model_variant(variant);
        request.bind_execution_adapter(execution.clone()).unwrap();
        request
            .bind_v2_state_runtime(runtime.clone(), runtime.state_fingerprint, BackendKind::Cpu)
            .unwrap();

        core.add_request(request).unwrap();
        assert_eq!(core.pending_request_count(), 1);

        let cuda_descriptor = crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stages_test(
            &execution.stages,
        );
        let cuda_runtime = Arc::new(crate::kv::v2::CapabilityStateRuntimeV2::stateless(
            crate::kv::v2::StatelessCapabilityRuntimeV2::seal(
                BackendKind::Cuda,
                &execution,
                cuda_descriptor,
            )
            .unwrap(),
        ));
        let mut mismatched = EngineCoreRequest::tts("wrong backend").with_model_variant(variant);
        mismatched.bind_execution_adapter(execution).unwrap();
        mismatched
            .bind_v2_state_runtime(
                cuda_runtime.clone(),
                cuda_runtime.state_fingerprint,
                BackendKind::Cuda,
            )
            .unwrap();
        assert!(core.add_request(mismatched).is_err());
        assert_eq!(core.pending_request_count(), 1);
    }

    #[tokio::test]
    async fn test_abort_requests_for_variant_only_aborts_matching_requests() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let config = EngineCoreConfig::default();
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut req_a = EngineCoreRequest::tts("variant-a");
        req_a.id = "req-a".to_string();
        req_a.model_variant = Some(ModelVariant::Qwen34BGguf);

        let mut req_b = EngineCoreRequest::tts("variant-b");
        req_b.id = "req-b".to_string();
        req_b.model_variant = Some(ModelVariant::Qwen38BGguf);

        core.add_request(req_a).unwrap();
        core.add_request(req_b).unwrap();

        let aborted = core
            .abort_requests_for_variant(ModelVariant::Qwen34BGguf)
            .await;
        assert_eq!(aborted, vec!["req-a".to_string()]);
        assert!(!core.has_request(&"req-a".to_string()));
        assert!(core.has_request(&"req-b".to_string()));

        let calls = cleanup_calls.lock().unwrap().clone();
        assert!(calls.iter().any(|id| id == "req-a"));
        assert!(!calls.iter().any(|id| id == "req-b"));
    }

    #[tokio::test]
    async fn stream_backpressure_deadline_preserves_delivery_provenance() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::tts("stream deadline");
        request.id = "stream-deadline".to_string();
        core.add_request(request).unwrap();
        let session = core
            .get_session_key(&"stream-deadline".to_string())
            .expect("exact session");

        assert!(
            core.handle_stream_delivery_failure(StreamDeliveryFailure {
                session: session.clone(),
                kind: StreamDeliveryFailureKind::Deadline,
            })
            .await
        );

        let terminal = core
            .pending_terminal_outputs
            .back()
            .expect("stream terminal output");
        assert_eq!(terminal.session, session);
        assert_eq!(
            terminal.disposition,
            ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut)
        );
        assert_eq!(
            terminal.provenance,
            OutcomeProvenance::deadline(
                DeadlinePhase::StreamDelivery,
                DispatchState::ProducedOutput,
            )
        );
    }

    #[tokio::test]
    async fn terminal_stream_delivery_failure_replaces_success_without_duplicate_terminal() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let session = super::super::SessionKey::new("terminal-stream".to_string(), 17);
        let mut outputs = vec![core.output_processor.process_execution(
            ExecutorOutput::terminal(session.request_id.clone()),
            &ExecutionDisposition::Finished(ExecutionFinishReason::Completed),
            session.epoch,
            Duration::ZERO,
        )];

        core.reconcile_stream_delivery_failures(
            &mut outputs,
            vec![StreamDeliveryFailure {
                session,
                kind: StreamDeliveryFailureKind::Delivery,
            }],
        )
        .await;

        assert!(outputs[0].is_finished);
        assert_eq!(outputs[0].finish_reason, Some(OutputFinishReason::Error));
        assert_eq!(
            outputs[0].error.as_deref(),
            Some("committed stream delivery failed")
        );
        assert_eq!(
            outputs[0].provenance,
            OutcomeProvenance::failure(
                FailureOrigin::StreamDelivery,
                DispatchState::ProducedOutput,
            )
        );
        assert!(
            core.pending_terminal_outputs.is_empty(),
            "a completed scheduler session must not receive a second terminal event"
        );
    }

    #[tokio::test]
    async fn cancelled_execution_surfaces_aborted_terminal_output() {
        let executor = UnifiedExecutor::new_for_test(Box::new(CancelledExecutor));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::tts("cancelled");
        request.id = "cancelled".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();

        let outputs = core.step().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finished);
        assert_eq!(
            outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert_eq!(outputs[0].error.as_deref(), Some("request cancelled"));
        assert!(!core.has_request(&"cancelled".to_string()));
    }

    #[tokio::test]
    async fn waiting_and_running_aborts_emit_one_cancelled_output_each() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();

        let mut waiting = EngineCoreRequest::tts("waiting");
        waiting.id = "abort-waiting".to_string();
        waiting.prompt_tokens = vec![1];
        core.add_request(waiting).unwrap();
        let waiting_session = core.get_session_key(&"abort-waiting".to_string()).unwrap();
        assert!(core.abort_request_session(&waiting_session).await);
        let waiting_outputs = core.step().await.unwrap();
        assert_eq!(waiting_outputs.len(), 1);
        assert_eq!(waiting_outputs[0].sequence_id, waiting_session.epoch);
        assert_eq!(
            waiting_outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );

        let mut running = EngineCoreRequest::tts("running");
        running.id = "abort-running".to_string();
        running.prompt_tokens = vec![1];
        core.add_request(running).unwrap();
        let progress = core.step().await.unwrap();
        assert_eq!(progress.len(), 1);
        assert!(!progress[0].is_finished);
        let running_session = core.get_session_key(&"abort-running".to_string()).unwrap();
        assert!(core.abort_request_session(&running_session).await);
        let running_outputs = core.step().await.unwrap();
        assert_eq!(running_outputs.len(), 1);
        assert_eq!(running_outputs[0].sequence_id, running_session.epoch);
        assert_eq!(
            running_outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert!(core.step().await.unwrap().is_empty());

        let calls = cleanup_calls.lock().unwrap();
        assert_eq!(
            calls
                .iter()
                .filter(|id| id.as_str() == "abort-waiting")
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|id| id.as_str() == "abort-running")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn unconfirmed_cancel_cleanup_is_retried_and_keeps_the_id_fenced() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            events.clone(),
            false,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 1,
                enable_adaptive_batching: false,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.cleanup_backoff_base = Duration::ZERO;
        core.retry_policy.cleanup_backoff_max = Duration::ZERO;

        let mut request = EngineCoreRequest::tts("retry cleanup");
        request.id = "cleanup-fence".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request.clone()).unwrap();
        core.step().await.unwrap();
        let session = core.get_session_key(&request.id).unwrap();
        assert!(core.abort_request_session(&session).await);
        assert!(core.add_request(request.clone()).is_err());

        let outputs = core.step().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert!(core.step().await.unwrap().is_empty());
        let cleanup_attempts = events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with("cleanup:cleanup-fence:"))
            .count();
        assert!(cleanup_attempts >= 2, "cleanup was not retried");
        assert!(core.add_request(request).is_err());
    }

    #[tokio::test]
    async fn aborted_terminal_output_survives_plan_failure_until_successful_delivery() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(events, true)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                min_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 4,
                enable_chunked_prefill: false,
                enable_adaptive_batching: false,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut aborted = EngineCoreRequest::tts("durable terminal outbox");
        aborted.id = "durable-abort".to_string();
        aborted.prompt_tokens = vec![1];
        core.add_request(aborted.clone()).unwrap();
        core.step().await.unwrap();

        let aborted_session = core.get_session_key(&aborted.id).unwrap();
        assert!(core.abort_request_session(&aborted_session).await);
        assert!(core.has_pending_terminal_output(&aborted.id));
        assert!(core.add_request(aborted.clone()).is_err());

        let mut plan_fault = EngineCoreRequest::tts("force plan validation failure");
        plan_fault.id = "plan-fault".to_string();
        plan_fault.prompt_tokens = vec![2];
        core.add_request(plan_fault.clone()).unwrap();
        let plan_fault_session = core.get_session_key(&plan_fault.id).unwrap();
        core.execution_trackers.insert(
            plan_fault.id.clone(),
            ExecutionTracker::new(super::super::SessionKey::new(
                plan_fault.id.clone(),
                plan_fault_session.epoch + 1,
            )),
        );

        let error = core.step().await.expect_err("plan preparation must fail");
        assert!(error
            .to_string()
            .contains("already has a different active session"));
        assert!(core.has_pending_terminal_output(&aborted.id));
        assert_eq!(
            core.scheduler
                .pending_release_confirmation_required(&aborted_session),
            Some(true)
        );
        assert!(core.add_request(aborted.clone()).is_err());

        core.execution_trackers.remove(&plan_fault.id);
        assert!(core
            .active_plans
            .values()
            .all(|plan| plan.session != plan_fault_session));
        let retry_outputs = core.step().await.unwrap();
        let aborted_outputs: Vec<_> = retry_outputs
            .iter()
            .filter(|output| output.request_id == aborted.id)
            .collect();
        assert_eq!(aborted_outputs.len(), 1);
        assert_eq!(aborted_outputs[0].sequence_id, aborted_session.epoch);
        assert_eq!(
            aborted_outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert!(!core.has_pending_terminal_output(&aborted.id));
        assert_eq!(
            core.scheduler
                .pending_release_confirmation_required(&aborted_session),
            Some(true),
            "the delivered batch remains fenced until its outer consumer routes it"
        );
        assert!(core.add_request(aborted.clone()).is_err());
        assert!(core.acknowledge_terminal_output(&aborted_session));
        assert_eq!(
            core.scheduler
                .pending_release_confirmation_required(&aborted_session),
            None
        );

        let later_outputs = core.step().await.unwrap();
        assert!(later_outputs
            .iter()
            .all(|output| output.request_id != aborted.id));
        core.add_request(aborted).unwrap();
        assert_ne!(
            core.get_session_key(&"durable-abort".to_string())
                .unwrap()
                .epoch,
            aborted_session.epoch
        );
    }

    #[tokio::test]
    async fn stale_session_cannot_abort_reused_request_id() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut first = EngineCoreRequest::tts("first");
        first.id = "reused".to_string();
        core.add_request(first).unwrap();
        let first_session = core.get_session_key(&"reused".to_string()).unwrap();
        assert!(core.abort_request_session(&first_session).await);

        let mut second = EngineCoreRequest::tts("second");
        second.id = "reused".to_string();
        assert!(
            core.add_request(second.clone()).is_err(),
            "public ID must remain fenced until cancellation is delivered"
        );
        let outputs = core.step().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].sequence_id, first_session.epoch);
        assert_eq!(
            outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert_eq!(outputs[0].error.as_deref(), Some("request cancelled"));

        assert!(
            core.add_request(second.clone()).is_err(),
            "returning a terminal output must not acknowledge delivery inside the core"
        );
        assert!(core.acknowledge_terminal_output(&first_session));
        core.add_request(second).unwrap();
        let second_session = core.get_session_key(&"reused".to_string()).unwrap();
        assert_ne!(first_session.epoch, second_session.epoch);
        assert!(!core.abort_request_session(&first_session).await);
        assert!(core.has_request(&"reused".to_string()));
    }

    #[test]
    fn test_merge_executor_output_replaces_cumulative_audio_snapshots() {
        let first = ExecutorOutput {
            request_id: "req-a".to_string(),
            audio: Some(AudioOutput::new(vec![0.1, 0.2], 24_000)),
            text: Some("hello".to_string()),
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 1,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        };
        let second = ExecutorOutput {
            request_id: "req-a".to_string(),
            audio: Some(AudioOutput::new(vec![0.1, 0.2, 0.3], 24_000)),
            text: Some("hello world".to_string()),
            input_transcription: Some("hello there".to_string()),
            tokens_processed: 1,
            tokens_generated: 1,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        };

        let merged = EngineCore::merge_executor_output(Some(first), second);
        let audio = merged.audio.expect("merged audio");
        assert_eq!(audio.samples, vec![0.1, 0.2, 0.3]);
        assert_eq!(merged.input_transcription.as_deref(), Some("hello there"));
        assert_eq!(merged.text.as_deref(), Some("hello world"));
        assert_eq!(merged.tokens_processed, 2);
        assert_eq!(merged.tokens_generated, 2);
        assert!(merged.finished);
    }

    #[test]
    fn loaded_stage_contract_preserves_independent_request_parallelism() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let model_instance = super::super::ModelInstanceId::new(2);
        let mut request = EngineCoreRequest::tts("contract").with_model_variant(variant);
        request.bind_model_instance(model_instance).unwrap();
        let mut declared =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        declared.prefill_batch = NativeBatchMode::Static;
        declared.max_batch_size = 8;
        declared.concurrency = ConcurrencyClass::Batchable;
        let compatibility_stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(0),
            "tts.compatibility",
            &declared,
            NativeBatchMode::None,
        );
        request
            .bind_execution_adapter(super::super::ExecutionAdapterBinding {
                execution_group_id: super::super::ExecutionGroupId::new(1),
                model_instance_id: model_instance,
                adapter_instance_id: super::super::AdapterInstanceId::new(3),
                adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
                model_variant: variant,
                capability_id: "tts".to_string(),
                stages: Arc::from([compatibility_stage]),
            })
            .unwrap();

        let effective = EngineCore::apply_adapter_execution_contract(&request, declared).unwrap();
        assert_eq!(effective.prefill_batch, NativeBatchMode::None);
        assert_eq!(effective.decode_batch, NativeBatchMode::None);
        assert_eq!(effective.max_batch_size, 8);
        assert_eq!(effective.concurrency, ConcurrencyClass::Batchable);
    }

    #[test]
    // Explicitly dropping the closure releases its captured request Arc before
    // the test asserts unique ownership through Arc::get_mut.
    #[allow(clippy::drop_non_drop)]
    fn incremental_progress_requires_exact_batch_plan_session_lane_and_sequence() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
            .expect("core");
        let request_id = "fenced-progress".to_string();
        let mut request = EngineCoreRequest::tts("fenced progress");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.streaming = true;
        let cancellation = Arc::new(AtomicBool::new(false));
        request.cancellation = Some(cancellation.clone());
        let (delivery_tx, _delivery_rx) = mpsc::channel(8);
        request.streaming_tx = Some(delivery_tx);
        core.add_request(request).unwrap();
        let session = core.get_session_key(&request_id).unwrap();

        let profile = ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Atomic);
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            StageId::new(51),
            "test.atomic.incremental",
            &profile,
            NativeBatchMode::None,
        );
        stage.selector = super::super::StageWorkSelector::Atomic;
        stage.output_visibility = OutputVisibility::IncrementalCommitted;
        let plan_id = 7001;
        let plan = ExecutionPlan {
            plan_id,
            session: session.clone(),
            work: WorkUnit::AtomicJob {
                kind: "tts".to_string(),
            },
            batch_key: BatchKey {
                backend: BackendKind::Cpu,
                model_variant: None,
                task_type: TaskType::TTS,
                work_kind: "tts".to_string(),
                compute_dtype: "f32".to_string(),
                kv_dtype: "none".to_string(),
                cache_namespace: "none".to_string(),
                adapter: None,
            },
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            estimate: ResourceVector::zero(),
            stage: Some(stage),
        };
        let mut tracker = ExecutionTracker::new(session.clone());
        tracker.transition(ExecutionState::Admitted).unwrap();
        tracker.transition(ExecutionState::AtomicRunning).unwrap();
        tracker.begin_plan(&plan).unwrap();
        core.execution_trackers.insert(request_id.clone(), tracker);
        core.active_plans.insert(plan_id, plan.clone());

        let lane = EngineCore::batch_lane(&plan, WorkCost::new(1, 1, 0));
        let batch_id = BatchId::new(81);
        core.active_stream_batches.insert(
            batch_id,
            ActiveStreamBatch {
                lane: lane.clone(),
                output_visibility: OutputVisibility::IncrementalCommitted,
                rows: HashMap::from([(plan_id, session.clone())]),
            },
        );
        let request = core.requests.get(&request_id).unwrap().clone();
        let staging = request.stream_staging_buffer();
        let (progress_tx, mut progress_rx) = mpsc::channel(8);
        let binding = request
            .bind_stream_quantum(
                batch_id,
                lane.clone(),
                plan_id,
                session.clone(),
                OutputVisibility::IncrementalCommitted,
                progress_tx,
                StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES),
            )
            .unwrap();
        let mut next_progress = |sequence| {
            staging
                .push_with_policy(text_progress(&request_id, sequence), request.stream_policy)
                .unwrap();
            progress_rx.try_recv().expect("fenced progress")
        };

        let mut wrong_batch = next_progress(0);
        wrong_batch.batch_id = BatchId::new(82);
        assert!(core
            .commit_incremental_stream_progress(wrong_batch)
            .unwrap_err()
            .to_string()
            .contains("inactive physical batch"));

        let mut wrong_plan = next_progress(0);
        wrong_plan.plan_id += 1;
        assert!(core
            .commit_incremental_stream_progress(wrong_plan)
            .unwrap_err()
            .to_string()
            .contains("inactive plan"));

        let mut wrong_session = next_progress(0);
        wrong_session.session.epoch += 1;
        assert!(core
            .commit_incremental_stream_progress(wrong_session)
            .unwrap_err()
            .to_string()
            .contains("session does not match"));

        let mut wrong_lane = next_progress(0);
        wrong_lane.lane.shape_bucket.push_str(".stale");
        assert!(core
            .commit_incremental_stream_progress(wrong_lane)
            .unwrap_err()
            .to_string()
            .contains("physical batch fence"));

        let wrong_sequence = next_progress(4);
        assert!(core
            .commit_incremental_stream_progress(wrong_sequence)
            .unwrap_err()
            .to_string()
            .contains("did not match expected 0"));

        let delivery = core
            .commit_incremental_stream_progress(next_progress(0))
            .expect("exact progress must commit");
        drop(delivery);
        assert_eq!(core.stream_sequence_cursors.get(&session), Some(&1));
        assert!(core.incremental_stream_sessions.contains(&session));

        assert!(core
            .commit_incremental_stream_progress(next_progress(0))
            .unwrap_err()
            .to_string()
            .contains("did not match expected 1"));

        cancellation.store(true, Ordering::Release);
        assert_eq!(
            core.commit_incremental_stream_progress(next_progress(1))
                .unwrap_err()
                .kind,
            StreamDeliveryFailureKind::Cancelled
        );
        cancellation.store(false, Ordering::Release);

        drop(next_progress);
        drop(binding);
        drop(request);
        Arc::get_mut(core.requests.get_mut(&request_id).expect("active request"))
            .expect("core owns the last request reference")
            .deadline = Some(Instant::now());
        let request = core.requests.get(&request_id).unwrap().clone();
        let staging = request.stream_staging_buffer();
        let (progress_tx, mut progress_rx) = mpsc::channel(8);
        let _binding = request
            .bind_stream_quantum(
                batch_id,
                lane,
                plan_id,
                session,
                OutputVisibility::IncrementalCommitted,
                progress_tx,
                StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES),
            )
            .unwrap();
        staging
            .push_with_policy(text_progress(&request_id, 1), request.stream_policy)
            .unwrap();
        assert_eq!(
            core.commit_incremental_stream_progress(
                progress_rx.try_recv().expect("deadline-fenced progress")
            )
            .unwrap_err()
            .kind,
            StreamDeliveryFailureKind::RequestDeadline
        );
    }

    #[test]
    fn physical_batch_formation_uses_exact_loaded_lane_identity() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
            .expect("core");
        for id in ["physical-a", "physical-b"] {
            let mut request = EngineCoreRequest::tts("physical batch");
            request.id = id.to_string();
            core.add_request(request).unwrap();
        }
        let scheduled = ["physical-a", "physical-b"]
            .into_iter()
            .map(|id| {
                let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
                scheduled_prefill(id, epoch)
            })
            .collect::<Vec<_>>();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.max_batch_size = 2;
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(4),
            "tts.generate",
            &profile,
            NativeBatchMode::Static,
        );
        stage.workspace_base_bytes = 4;
        stage.workspace_per_row_bytes = 2;
        stage.workspace_per_work_unit_bytes = 1;
        stage.max_workspace_bytes = 10;
        let adapter_key = super::super::AdapterBindingKey {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: super::super::ModelInstanceId::new(2),
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            capability_id: "tts".to_string(),
            stage_id: stage.id,
        };
        for item in &scheduled {
            core.active_plans.insert(
                item.plan_id,
                ExecutionPlan {
                    plan_id: item.plan_id,
                    session: item.session_key(),
                    work: item.work.clone(),
                    batch_key: BatchKey {
                        backend: BackendKind::Cpu,
                        model_variant: None,
                        task_type: TaskType::TTS,
                        work_kind: "prefill".to_string(),
                        compute_dtype: "f32".to_string(),
                        kv_dtype: "none".to_string(),
                        cache_namespace: "none".to_string(),
                        adapter: Some(adapter_key.clone()),
                    },
                    batch_mode: NativeBatchMode::Static,
                    max_batch_size: 2,
                    estimate: ResourceVector::zero(),
                    stage: Some(stage.clone()),
                },
            );
        }
        let requests = scheduled
            .iter()
            .map(|item| core.requests.get(&item.request_id).unwrap().clone())
            .collect::<Vec<_>>();

        let joined = core
            .form_physical_batches(&requests, &scheduled, &mut Vec::new())
            .unwrap();
        assert_eq!(joined.len(), 1);
        assert_eq!(joined[0].physical_batch().rows.len(), 2);
        assert_eq!(
            joined[0].physical_batch().workspace,
            ResourceVector {
                host_bytes: ResourceAmount::Known(10),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            joined[0].physical_batch().lane.model_instance,
            super::super::ModelInstanceId::new(2)
        );

        core.active_plans
            .get_mut(&scheduled[1].plan_id)
            .unwrap()
            .batch_key
            .adapter
            .as_mut()
            .unwrap()
            .model_instance_id = super::super::ModelInstanceId::new(9);
        let split = core
            .form_physical_batches(&requests, &scheduled, &mut Vec::new())
            .unwrap();
        assert_eq!(split.len(), 2);
        assert!(split
            .iter()
            .all(|batch| batch.physical_batch().rows.len() == 1));
    }

    #[test]
    fn cuda_batch_workspace_preserves_host_and_device_domains() {
        let lane = BatchLaneKey {
            execution_group: super::super::ExecutionGroupId::new(1),
            model_instance: super::super::ModelInstanceId::new(2),
            adapter_instance: super::super::AdapterInstanceId::new(3),
            adapter_abi: super::super::AdapterAbiRevision::new(1),
            capability_id: "chat".to_string(),
            stage_id: super::super::StageId::new(4),
            backend: BackendKind::Cuda,
            device_ordinal: None,
            compute_dtype: "f16".to_string(),
            state_dtype: "f16".to_string(),
            tensor_layout: "ragged".to_string(),
            quantization: "none".to_string(),
            state_schema: "test".to_string(),
            kernel_mode: "continuous".to_string(),
            semantic_mode: "decode".to_string(),
            shape_bucket: "ragged".to_string(),
        };
        let row = ReadyQuantum {
            plan_id: 1,
            session: super::super::SessionKey::new("workspace".to_string(), 1),
            lane,
            work: WorkUnit::SequenceStep {
                phase: super::super::SequencePhase::Decode,
                input: super::super::InputRange { start: 1, end: 2 },
                max_output_steps: 1,
            },
            cost: WorkCost::with_workspace(
                1,
                1,
                ResourceVector {
                    host_bytes: ResourceAmount::Known(32),
                    temporary_bytes: ResourceAmount::Known(8_192),
                    ..ResourceVector::zero()
                },
            ),
            managed_cache: None,
        };

        assert_eq!(
            PhysicalBatchAssembly::workspace_resources(
                BackendKind::Cuda,
                16,
                std::slice::from_ref(&row),
            ),
            Some(ResourceVector {
                host_bytes: ResourceAmount::Known(32),
                device_bytes: ResourceAmount::Known(8_208),
                ..ResourceVector::zero()
            })
        );
    }

    #[test]
    fn independent_compatibility_rows_form_bounded_parallel_batches() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
            .expect("core");
        let ids = [
            "parallel-a",
            "parallel-b",
            "parallel-c",
            "parallel-d",
            "parallel-e",
        ];
        for id in ids {
            let mut request = EngineCoreRequest::tts(format!("independent {id}"));
            request.id = id.to_string();
            core.add_request(request).unwrap();
        }
        let scheduled = ids
            .into_iter()
            .map(|id| {
                let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
                scheduled_prefill(id, epoch)
            })
            .collect::<Vec<_>>();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.max_batch_size = 2;
        let stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(6),
            "tts.compatibility",
            &profile,
            NativeBatchMode::None,
        );
        let adapter_key = super::super::AdapterBindingKey {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: super::super::ModelInstanceId::new(2),
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            capability_id: "tts".to_string(),
            stage_id: stage.id,
        };
        for item in &scheduled {
            core.active_plans.insert(
                item.plan_id,
                ExecutionPlan {
                    plan_id: item.plan_id,
                    session: item.session_key(),
                    work: item.work.clone(),
                    batch_key: BatchKey {
                        backend: BackendKind::Cpu,
                        model_variant: None,
                        task_type: TaskType::TTS,
                        work_kind: "prefill".to_string(),
                        compute_dtype: "f32".to_string(),
                        kv_dtype: "none".to_string(),
                        cache_namespace: "none".to_string(),
                        adapter: Some(adapter_key.clone()),
                    },
                    batch_mode: NativeBatchMode::None,
                    max_batch_size: 2,
                    estimate: ResourceVector::zero(),
                    stage: Some(stage.clone()),
                },
            );
        }
        let requests = scheduled
            .iter()
            .map(|item| core.requests.get(&item.request_id).unwrap().clone())
            .collect::<Vec<_>>();

        let batches = core
            .form_physical_batches(&requests, &scheduled, &mut Vec::new())
            .unwrap();
        let widths = batches
            .iter()
            .map(|batch| batch.physical_batch().rows.len())
            .collect::<Vec<_>>();
        assert_eq!(widths, vec![2, 2, 1]);
        assert!(batches
            .iter()
            .all(|batch| batch.physical_batch().mode == NativeBatchMode::None));
        assert!(batches
            .iter()
            .all(|batch| { batch.physical_batch().lane.shape_bucket == "independent" }));
    }

    #[test]
    fn prepared_exact_shapes_split_static_physical_batches() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
            .expect("core");
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.prefill_batch = NativeBatchMode::Static;
        profile.max_batch_size = 2;
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(7),
            "tts.generate.tensor_static",
            &profile,
            NativeBatchMode::Static,
        );
        stage.selector = super::super::StageWorkSelector::Atomic;
        stage.shape_policy = StageShapePolicy::Exact;
        stage.max_padding_basis_points = 0;
        stage.max_work_units = 2;
        stage.max_workspace_bytes = 64;
        let binding = super::super::ExecutionAdapterBinding {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: super::super::ModelInstanceId::new(2),
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage.clone()]),
        };
        for (id, tensor_elements) in [("shape-a", 8), ("shape-b", 9)] {
            let mut request =
                EngineCoreRequest::tts("exact static shape").with_model_variant(variant);
            request.id = id.to_string();
            request.bind_execution_adapter(binding.clone()).unwrap();
            request
                .install_prepared_stage_cost(
                    stage.id,
                    WorkCost::new(1, tensor_elements, tensor_elements),
                )
                .unwrap();
            core.add_request(request).unwrap();
        }
        let scheduled = ["shape-a", "shape-b"]
            .into_iter()
            .map(|id| {
                let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
                scheduled_prefill(id, epoch)
            })
            .collect::<Vec<_>>();
        let adapter_key = binding.key_for_stage(stage.id).unwrap();
        for item in &scheduled {
            core.active_plans.insert(
                item.plan_id,
                ExecutionPlan {
                    plan_id: item.plan_id,
                    session: item.session_key(),
                    work: WorkUnit::AtomicJob {
                        kind: "tts".to_string(),
                    },
                    batch_key: BatchKey {
                        backend: BackendKind::Cpu,
                        model_variant: Some(variant),
                        task_type: TaskType::TTS,
                        work_kind: "tts".to_string(),
                        compute_dtype: "f32".to_string(),
                        kv_dtype: "none".to_string(),
                        cache_namespace: "none".to_string(),
                        adapter: Some(adapter_key.clone()),
                    },
                    batch_mode: NativeBatchMode::Static,
                    max_batch_size: 2,
                    estimate: ResourceVector::zero(),
                    stage: Some(stage.clone()),
                },
            );
        }
        let requests = scheduled
            .iter()
            .map(|item| core.requests.get(&item.request_id).unwrap().clone())
            .collect::<Vec<_>>();

        let batches = core
            .form_physical_batches(&requests, &scheduled, &mut Vec::new())
            .unwrap();
        assert_eq!(batches.len(), 2);
        assert!(batches
            .iter()
            .all(|batch| batch.physical_batch().rows.len() == 1));
        assert_eq!(
            batches[0]
                .physical_batch()
                .workspace
                .workspace_bytes()
                .unwrap(),
            8
        );
        assert_eq!(
            batches[1]
                .physical_batch()
                .workspace
                .workspace_bytes()
                .unwrap(),
            9
        );
    }

    #[test]
    fn one_token_continuous_budget_admits_multiple_rows() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
            .expect("core");
        for id in ["continuous-a", "continuous-b"] {
            let mut request = EngineCoreRequest::tts("continuous batch fixture");
            request.id = id.to_string();
            core.add_request(request).unwrap();
        }
        let scheduled = ["continuous-a", "continuous-b"]
            .into_iter()
            .map(|id| {
                let epoch = core.get_session_key(&id.to_string()).unwrap().epoch;
                scheduled_decode(id, epoch)
            })
            .collect::<Vec<_>>();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Sequence);
        profile.max_batch_size = 2;
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(5),
            "chat.decode.tensor_continuous",
            &profile,
            NativeBatchMode::Continuous,
        );
        stage.max_work_units = 2;
        let adapter_key = super::super::AdapterBindingKey {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: super::super::ModelInstanceId::new(2),
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(1),
            capability_id: "chat".to_string(),
            stage_id: stage.id,
        };
        for item in &scheduled {
            core.active_plans.insert(
                item.plan_id,
                ExecutionPlan {
                    plan_id: item.plan_id,
                    session: item.session_key(),
                    work: item.work.clone(),
                    batch_key: BatchKey {
                        backend: BackendKind::Cpu,
                        model_variant: None,
                        task_type: TaskType::Chat,
                        work_kind: "decode".to_string(),
                        compute_dtype: "f32".to_string(),
                        kv_dtype: "f32".to_string(),
                        cache_namespace: "continuous".to_string(),
                        adapter: Some(adapter_key.clone()),
                    },
                    batch_mode: NativeBatchMode::Continuous,
                    max_batch_size: 2,
                    estimate: ResourceVector::zero(),
                    stage: Some(stage.clone()),
                },
            );
        }
        let requests = scheduled
            .iter()
            .map(|item| core.requests.get(&item.request_id).unwrap().clone())
            .collect::<Vec<_>>();

        let batches = core
            .form_physical_batches(&requests, &scheduled, &mut Vec::new())
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0].physical_batch().mode,
            NativeBatchMode::Continuous
        );
        assert_eq!(batches[0].physical_batch().rows.len(), 2);
        assert_eq!(batches[0].physical_batch().budget.max_logical_units, 2);
    }

    #[tokio::test]
    async fn executor_progress_is_validated_before_scheduler_mutation() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("transaction");
        request.id = "transaction".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();

        let mut invalid = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: scheduled.num_tokens + 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        );
        invalid.disposition = ExecutionDisposition::Progress;
        let output = core
            .commit_executor_result(invalid, 1.0)
            .await
            .expect("invalid result must emit a terminal output")
            .output;

        assert!(output.finished);
        assert!(output
            .error
            .as_deref()
            .is_some_and(|message| message.contains("beyond the scheduled quantum")));
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((0, 0)),
            "invalid progress must not reach scheduler accounting"
        );
    }

    fn retryable_step_result(
        scheduled: &ScheduledRequest,
        retry: RetryDisposition,
    ) -> ExecutorStepResult {
        ExecutorStepResult::from_session(
            scheduled,
            super::super::executor::ModelSessionResult {
                output: ExecutorOutput {
                    request_id: scheduled.request_id.clone(),
                    audio: None,
                    text: None,
                    input_transcription: None,
                    tokens_processed: 0,
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: Some("transient backend failure".to_string()),
                },
                disposition: ExecutionDisposition::Failed(ExecutionFailure {
                    kind: super::super::execution::FailureKind::Backend,
                    scope: super::super::execution::FailureScope::Row,
                    retry,
                    health: super::super::execution::HealthImpact::Degraded,
                    message: "transient backend failure".to_string(),
                }),
                safe_point: true,
                provenance: OutcomeProvenance::failure(
                    FailureOrigin::Model,
                    DispatchState::Started,
                ),
                staged_stream_outputs: Vec::new(),
                managed_cache_completions: Vec::new(),
            },
        )
    }

    #[tokio::test]
    async fn same_session_retry_releases_quantum_without_committing_progress() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("retry");
        request.id = "same-session-retry".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        let session = scheduled.session_key();
        core.begin_execution_plan(&scheduled).await.unwrap();

        let emitted = core
            .commit_executor_result(
                retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
                1.0,
            )
            .await;
        assert!(
            emitted.is_none(),
            "a retry must not terminalize the request"
        );
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((0, 0))
        );

        let mut retry_schedule = core.scheduler.schedule();
        assert_eq!(
            retry_schedule.prefill_requests.len(),
            1,
            "same-session retry was not rescheduled: {retry_schedule:?}"
        );
        let retry = retry_schedule.prefill_requests.remove(0);
        assert_eq!(retry.session_key(), session);
        assert_ne!(retry.plan_id, scheduled.plan_id);
    }

    #[tokio::test]
    async fn committed_stream_progress_disables_all_executor_retries() {
        for (suffix, retry) in [
            ("same-session", RetryDisposition::RetrySameSession),
            ("recompute", RetryDisposition::Recompute),
        ] {
            let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
                Mutex::new(Vec::new()),
            ))));
            let mut core = EngineCore::new_with_unified_executor(
                EngineCoreConfig {
                    max_batch_size: 1,
                    max_tokens_per_step: 8,
                    ..Default::default()
                },
                executor,
            )
            .unwrap();
            core.retry_policy.execution_backoff_base = Duration::ZERO;
            core.retry_policy.execution_backoff_max = Duration::ZERO;
            let mut request = EngineCoreRequest::tts("visible retry");
            request.id = format!("visible-retry-{suffix}");
            request.prompt_tokens = vec![1];
            core.add_request(request).unwrap();
            core.refresh_scheduler_execution_profiles().await;
            let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
            let session = scheduled.session_key();
            core.begin_execution_plan(&scheduled).await.unwrap();
            core.incremental_stream_sessions.insert(session.clone());

            let committed = core
                .commit_executor_result(retryable_step_result(&scheduled, retry), 1.0)
                .await
                .expect("visible progress must turn a retry into a terminal failure");
            assert_eq!(committed.session, session);
            assert!(committed
                .output
                .error
                .as_deref()
                .is_some_and(|message| message.contains("retry is unsafe")));
            assert!(matches!(
                committed.disposition,
                ExecutionDisposition::Failed(ExecutionFailure {
                    retry: RetryDisposition::Never,
                    ..
                })
            ));
            let retry_schedule = core.scheduler.schedule();
            assert!(retry_schedule.prefill_requests.is_empty());
            assert!(retry_schedule.decode_requests.is_empty());
        }
    }

    #[tokio::test]
    async fn retry_budget_exhaustion_terminalizes_the_exact_session() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.max_execution_retries = 2;
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("retry budget");
        request.id = "retry-budget".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;

        let mut scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        let session = scheduled.session_key();
        for attempt in 1..=3 {
            core.begin_execution_plan(&scheduled).await.unwrap();
            let committed = core
                .commit_executor_result(
                    retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
                    1.0,
                )
                .await;
            if attempt <= 2 {
                assert!(committed.is_none());
                scheduled = core.scheduler.schedule().prefill_requests.remove(0);
                assert_eq!(scheduled.session_key(), session);
            } else {
                let committed = committed.expect("retry budget must terminalize");
                assert_eq!(committed.session, session);
                assert!(committed
                    .output
                    .error
                    .as_deref()
                    .is_some_and(|message| message.contains("retry budget exhausted")));
                assert!(matches!(
                    committed.disposition,
                    ExecutionDisposition::Failed(ExecutionFailure {
                        retry: RetryDisposition::Never,
                        ..
                    })
                ));
            }
        }
    }

    #[tokio::test]
    async fn recompute_retry_releases_capability_authoritative_session_state() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("recompute");
        request.id = "recompute-retry".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;

        let prefill = core.scheduler.schedule().prefill_requests.remove(0);
        let session = prefill.session_key();
        core.begin_execution_plan(&prefill).await.unwrap();
        core.commit_executor_result(
            ExecutorStepResult::new(
                &prefill,
                MockExecutor::build_outputs(std::slice::from_ref(&prefill)).remove(0),
            ),
            1.0,
        )
        .await
        .expect("prefill progress must be emitted");

        let decode = core.scheduler.schedule().decode_requests.remove(0);
        core.begin_execution_plan(&decode).await.unwrap();
        let emitted = core
            .commit_executor_result(
                retryable_step_result(&decode, RetryDisposition::Recompute),
                1.0,
            )
            .await;
        assert!(
            emitted.is_none(),
            "recompute must not terminalize the request"
        );
        assert!(cleanup_calls
            .lock()
            .unwrap()
            .iter()
            .any(|request_id| request_id == "recompute-retry"));
        assert_eq!(
            core.scheduler.get_running_info(&decode.request_id),
            Some((0, 0))
        );
        assert!(!core.execution_trackers.contains_key(&decode.request_id));

        let retry = core.scheduler.schedule().prefill_requests.remove(0);
        assert_eq!(retry.session_key(), session);
        assert_ne!(retry.plan_id, decode.plan_id);
    }

    #[tokio::test]
    async fn full_prefill_plan_covers_the_model_owned_prompt_operation() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 4,
                enable_chunked_prefill: true,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("full prefill");
        request.id = "full-prefill".to_string();
        request.prompt_tokens = (0..16).collect();
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        assert_eq!(scheduled.num_computed_tokens, 0);
        assert_eq!(scheduled.num_tokens, 16);
        let mut partial = scheduled.clone();
        partial.num_tokens = 4;
        assert!(core.begin_execution_plan(&partial).await.is_err());
        core.begin_execution_plan(&scheduled).await.unwrap();
        let plan = core.active_plans.get(&scheduled.plan_id).unwrap();
        assert!(matches!(
            plan.work,
            WorkUnit::SequenceStep {
                input: super::super::InputRange { start: 0, end: 16 },
                max_output_steps: 1,
                ..
            }
        ));

        let result = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 16,
                tokens_generated: 1,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        );
        let output = core
            .commit_executor_result(result, 1.0)
            .await
            .expect("valid progress must emit an output")
            .output;
        assert!(output.error.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((16, 1))
        );
    }

    #[tokio::test]
    async fn cache_plan_estimate_and_executor_observation_remain_distinct() {
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            Arc::new(Mutex::new(Vec::new())),
            true,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                block_size: 1,
                max_blocks: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("cache contract");
        request.id = "cache-contract".to_string();
        request.prompt_tokens = vec![1, 2];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();
        assert_eq!(
            core.active_plans[&scheduled.plan_id].estimate.kv_bytes,
            ResourceAmount::Known(0),
            "opaque executor caches must not inherit fabricated scheduler KV geometry"
        );

        let observed = ResourceVector {
            kv_bytes: ResourceAmount::Known(123),
            ..ResourceVector::zero()
        };
        let result = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: scheduled.num_tokens,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        )
        .with_observed_resources(observed);
        let report = EngineCore::report_from_result(&result);
        assert_eq!(report.observed_resources, observed);
    }

    #[tokio::test]
    async fn valid_executor_progress_commits_exactly_once() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("transaction");
        request.id = "valid-transaction".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();
        let result = ExecutorStepResult::new(
            &scheduled,
            MockExecutor::build_outputs(std::slice::from_ref(&scheduled)).remove(0),
        );

        let output = core
            .commit_executor_result(result.clone(), 1.0)
            .await
            .expect("valid progress must emit an output")
            .output;
        assert!(output.error.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((scheduled.num_tokens, 0))
        );

        assert!(core.commit_executor_result(result, 1.0).await.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((scheduled.num_tokens, 0)),
            "duplicate results must not advance progress twice"
        );
    }

    #[tokio::test]
    async fn unresolved_executor_is_planned_as_atomic_and_must_finish() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("atomic");
        request.id = "atomic".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        let scheduled = core.scheduler.schedule().prefill_requests.remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();
        assert!(matches!(
            core.active_plans
                .get(&scheduled.plan_id)
                .map(|plan| &plan.work),
            Some(WorkUnit::AtomicJob { .. })
        ));

        let result = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        );
        let output = core
            .commit_executor_result(result, 1.0)
            .await
            .expect("invalid atomic result must emit an error")
            .output;
        assert!(output
            .error
            .as_deref()
            .is_some_and(|message| message.contains("atomic execution must finish")));
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((0, 0))
        );
    }

    #[tokio::test]
    async fn executor_subbatch_failure_is_isolated_and_reconciled() {
        let executor = UnifiedExecutor::new_for_test(Box::new(SelectiveFailureExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 2,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut bad = EngineCoreRequest::tts("bad");
        bad.id = "bad".to_string();
        bad.prompt_tokens = vec![1];
        bad.model_variant = Some(ModelVariant::Qwen34BGguf);
        let mut good = EngineCoreRequest::tts("good");
        good.id = "good".to_string();
        good.prompt_tokens = vec![1];
        good.model_variant = Some(ModelVariant::Qwen38BGguf);
        core.add_request(bad).unwrap();
        core.add_request(good).unwrap();

        let outputs = core.step().await.unwrap();

        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|output| output.request_id == "bad" && output.error.is_some()));
        assert!(outputs
            .iter()
            .any(|output| output.request_id == "good" && output.error.is_none()));
        assert!(!core.has_request(&"bad".to_string()));
        assert!(!core.has_request(&"good".to_string()));
    }

    #[tokio::test]
    async fn hard_deadline_returns_one_terminal_output_with_original_sequence() {
        let metrics_before = crate::engine::engine_batch_metrics_snapshot();
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::tts("expired")
            .with_deadline(Some(Instant::now() - std::time::Duration::from_millis(1)));
        request.id = "expired".to_string();
        core.add_request(request).unwrap();

        let outputs = core.step().await.unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, "expired");
        assert_eq!(outputs[0].sequence_id, 0);
        assert!(outputs[0].is_finished);
        assert_eq!(outputs[0].num_tokens, 0);
        assert_eq!(outputs[0].error.as_deref(), Some(REQUEST_DEADLINE_EXCEEDED));
        assert_eq!(
            outputs[0].provenance,
            OutcomeProvenance::deadline(DeadlinePhase::SchedulerQueue, DispatchState::NotStarted,)
        );
        let metrics_after = crate::engine::engine_batch_metrics_snapshot();
        assert!(
            metrics_after.dispatch_states.not_started > metrics_before.dispatch_states.not_started
        );
        assert!(
            metrics_after.deadline_phases.scheduler_queue
                > metrics_before.deadline_phases.scheduler_queue
        );
        assert!(!core.has_request(&"expired".to_string()));
        assert_eq!(
            cleanup_calls
                .lock()
                .unwrap()
                .iter()
                .filter(|id| id.as_str() == "expired")
                .count(),
            1,
            "deadline cleanup must occur exactly once"
        );
    }

    #[tokio::test]
    async fn deadline_cleanup_precedes_newly_scheduled_execution() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            events.clone(),
            true,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                min_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 2,
                enable_chunked_prefill: false,
                enable_adaptive_batching: false,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut expired = EngineCoreRequest::tts("expires after prefill");
        expired.id = "expired-running".to_string();
        expired.prompt_tokens = vec![1];
        core.add_request(expired).unwrap();
        core.step().await.unwrap();
        assert!(core.scheduler.set_hard_deadline_for_test(
            &"expired-running".to_string(),
            Instant::now() - std::time::Duration::from_millis(1),
        ));

        let mut next = EngineCoreRequest::tts("new work");
        next.id = "new-work".to_string();
        next.prompt_tokens = vec![2];
        core.add_request(next).unwrap();
        core.step().await.unwrap();

        let events = events.lock().unwrap();
        let cleanup_index = events
            .iter()
            .position(|event| event.starts_with("cleanup:expired-running:"))
            .expect("expired session cleanup was not requested");
        let execution_index = events
            .iter()
            .position(|event| event == "execute-prefill:new-work")
            .expect("new work was not executed");
        assert!(
            cleanup_index < execution_index,
            "new work executed before expired cache cleanup: {events:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.starts_with("cleanup:expired-running:"))
                .count(),
            1,
            "expired cleanup must not be repeated during terminal processing"
        );
    }

    #[tokio::test]
    async fn unconfirmed_deadline_cleanup_fences_id_without_blocking_unrelated_work() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            events.clone(),
            false,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                min_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 1,
                enable_chunked_prefill: false,
                enable_prefix_caching: false,
                enable_adaptive_batching: false,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut expired = EngineCoreRequest::tts("expires with model-owned state");
        expired.id = "expired-running".to_string();
        expired.prompt_tokens = vec![1];
        core.add_request(expired).unwrap();
        core.step().await.unwrap();
        assert!(core.scheduler.set_hard_deadline_for_test(
            &"expired-running".to_string(),
            Instant::now() - std::time::Duration::from_millis(1),
        ));

        let mut waiting = EngineCoreRequest::tts("independent work");
        waiting.id = "waiting".to_string();
        waiting.prompt_tokens = vec![1];
        core.add_request(waiting).unwrap();
        let outputs = core.step().await.unwrap();

        assert!(outputs.iter().any(|output| {
            output.request_id == "expired-running"
                && output.error.as_deref() == Some(REQUEST_DEADLINE_EXCEEDED)
        }));
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|event| event == "execute-prefill:waiting"));

        let mut reused = EngineCoreRequest::tts("must remain fenced");
        reused.id = "expired-running".to_string();
        reused.prompt_tokens = vec![3];
        assert!(core.add_request(reused).is_err());
    }

    #[tokio::test]
    async fn test_step_records_exact_prompt_tokens_and_ttft() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let config = EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 1,
            max_blocks: 8,
            ..Default::default()
        };
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }]);
        request.id = "chat-req".to_string();
        request.model_variant = Some(ModelVariant::Qwen306B);
        request
            .install_chat_execution_preparation(
                ModelVariant::Qwen306B,
                vec![11, 22, 33, 44],
                None,
                4096,
            )
            .unwrap();

        core.add_request(request).unwrap();
        let outputs = core.step().await.unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].token_stats.prompt_tokens, 4);
        let latency = outputs[0]
            .latency_breakdown
            .as_ref()
            .expect("latency breakdown");
        assert!(latency.ttft_ms.is_some());
        assert!(latency.ttft_ms.unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn managed_kv_transaction_follows_live_prepare_commit_and_cancel_lifecycle() {
        let model_instance = ModelInstanceId::new(91);
        let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                enable_chunked_prefill: false,
                block_size: 16,
                max_blocks: 4,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .expect("core");
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "managed".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.id = "managed-core".to_string();
        request
            .install_chat_execution_preparation(
                ModelVariant::Qwen306B,
                vec![11, 12, 13, 14],
                None,
                4096,
            )
            .expect("prepare prompt");
        request
            .bind_model_instance(model_instance)
            .expect("model instance");
        bind_managed_test_state(&mut core, &mut request, model_instance);
        core.add_request(request).expect("add request");
        let session = core
            .get_session_key(&"managed-core".to_string())
            .expect("session");

        core.step().await.expect("managed step");
        let snapshot = core
            .managed_kv_cache
            .snapshot(model_instance, &session, crate::kv::CacheDomainId::new(1))
            .expect("committed table");
        assert_eq!(snapshot.committed_tokens, 4);
        assert_eq!(snapshot.version, 1);

        assert!(core.abort_request_session(&session).await);
        assert!(core
            .managed_kv_cache
            .snapshot(model_instance, &session, crate::kv::CacheDomainId::new(1))
            .is_none());
    }

    #[tokio::test]
    async fn managed_kv_capacity_defers_without_failing_or_sticking_prefill() {
        let model_instance = ModelInstanceId::new(92);
        let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                enable_chunked_prefill: false,
                block_size: 16,
                max_blocks: 1,
                max_batch_size: 1,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .expect("core");
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let owner = managed_test_request(
            &mut core,
            model_instance,
            "capacity-owner",
            vec![1, 2, 3, 4],
        );
        core.add_request(owner).expect("add owner");
        core.step().await.expect("commit owner prefill");
        let owner_session = core
            .get_session_key(&"capacity-owner".to_string())
            .expect("owner session");
        core.scheduler.finish_request(&"capacity-owner".to_string());

        let waiter = managed_test_request(
            &mut core,
            model_instance,
            "capacity-waiter",
            vec![5, 6, 7, 8],
        );
        core.add_request(waiter).expect("add waiter");
        assert!(core
            .prepare_step()
            .await
            .expect("capacity is scheduler backpressure")
            .is_none());
        assert!(core.active_plans.is_empty());
        assert!(core.active_managed_cache.is_empty());

        core.managed_kv_cache
            .release_session(&owner_session)
            .expect("release owner capacity");
        core.step()
            .await
            .expect("waiter retries after capacity release");
        let waiter_session = core
            .get_session_key(&"capacity-waiter".to_string())
            .expect("waiter session");
        let snapshot = core
            .managed_kv_cache
            .snapshot(
                model_instance,
                &waiter_session,
                crate::kv::CacheDomainId::new(1),
            )
            .expect("waiter committed table");
        assert_eq!(snapshot.committed_tokens, 4);
    }

    #[tokio::test]
    async fn managed_kv_backpressure_defers_only_the_blocked_row() {
        let blocked_model = ModelInstanceId::new(93);
        let runnable_model = ModelInstanceId::new(94);
        let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                enable_chunked_prefill: false,
                block_size: 16,
                max_blocks: 1,
                max_batch_size: 2,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .expect("core");
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;

        let owner = managed_test_request(
            &mut core,
            blocked_model,
            "isolated-capacity-owner",
            vec![1, 2, 3, 4],
        );
        core.add_request(owner).expect("add capacity owner");
        core.step().await.expect("commit capacity owner");
        core.scheduler
            .finish_request(&"isolated-capacity-owner".to_string());

        let blocked = managed_test_request(
            &mut core,
            blocked_model,
            "isolated-capacity-blocked",
            vec![5, 6, 7, 8],
        );
        let runnable = managed_test_request(
            &mut core,
            runnable_model,
            "isolated-capacity-runnable",
            vec![9, 10, 11, 12],
        );
        core.add_request(blocked).expect("add blocked row");
        core.add_request(runnable).expect("add runnable row");

        core.step().await.expect("execute independent runnable row");
        let runnable_session = core
            .get_session_key(&"isolated-capacity-runnable".to_string())
            .expect("runnable session");
        assert!(core
            .managed_kv_cache
            .snapshot(
                runnable_model,
                &runnable_session,
                crate::kv::CacheDomainId::new(1),
            )
            .is_some());
        let blocked_session = core
            .get_session_key(&"isolated-capacity-blocked".to_string())
            .expect("blocked session");
        let blocked_snapshot = core
            .managed_kv_cache
            .snapshot(
                blocked_model,
                &blocked_session,
                crate::kv::CacheDomainId::new(1),
            )
            .expect("blocked row owns an empty logical table");
        assert_eq!(blocked_snapshot.committed_tokens, 0);
        assert!(core
            .active_plans
            .values()
            .all(|plan| plan.session != blocked_session));
    }

    #[tokio::test]
    async fn blocked_prefill_does_not_discard_an_already_prepared_decode() {
        let saturated_model = ModelInstanceId::new(95);
        let decode_model = ModelInstanceId::new(96);
        let executor = UnifiedExecutor::new_for_test(Box::new(ManagedReceiptExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                enable_chunked_prefill: false,
                block_size: 16,
                max_blocks: 1,
                max_batch_size: 2,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .expect("core");
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;

        let owner = managed_test_request(
            &mut core,
            saturated_model,
            "decode-isolation-owner",
            vec![1, 2, 3, 4],
        );
        let decoder = managed_test_request(
            &mut core,
            decode_model,
            "decode-isolation-runnable",
            vec![5, 6, 7, 8],
        );
        core.add_request(owner).expect("add capacity owner");
        core.add_request(decoder).expect("add decoder");
        core.step().await.expect("commit both prefills");
        core.scheduler
            .finish_request(&"decode-isolation-owner".to_string());

        let blocked = managed_test_request(
            &mut core,
            saturated_model,
            "decode-isolation-blocked",
            vec![9, 10, 11, 12],
        );
        core.add_request(blocked).expect("add blocked prefill");
        core.step()
            .await
            .expect("commit decode despite later blocked prefill");

        let decode_session = core
            .get_session_key(&"decode-isolation-runnable".to_string())
            .expect("decode session");
        assert_eq!(
            core.managed_kv_cache
                .snapshot(
                    decode_model,
                    &decode_session,
                    crate::kv::CacheDomainId::new(1),
                )
                .expect("decode snapshot")
                .committed_tokens,
            5
        );
        let blocked_session = core
            .get_session_key(&"decode-isolation-blocked".to_string())
            .expect("blocked session");
        assert_eq!(
            core.managed_kv_cache
                .snapshot(
                    saturated_model,
                    &blocked_session,
                    crate::kv::CacheDomainId::new(1),
                )
                .expect("blocked logical table")
                .committed_tokens,
            0
        );
    }

    #[test]
    fn unbound_request_does_not_allocate_managed_cache() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "unbound".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1], None, 4096)
            .unwrap();
        request
            .bind_model_instance(ModelInstanceId::new(43))
            .unwrap();

        core.add_request(request).unwrap();
        assert_eq!(core.managed_kv_cache.model_count(), 0);
        assert!(core
            .requests
            .values()
            .next()
            .unwrap()
            .managed_cache_runtime()
            .is_none());
    }

    #[test]
    fn managed_model_cannot_publish_ready_after_failed_load_time_negotiation() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_blocks: 0,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let error = core
            .load_managed_model_cache(
                ModelInstanceId::new(44),
                &crate::kv::InferenceStateCapability::Managed(crate::kv::test_contract()),
                None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("capacity"));
        assert!(core.requests.is_empty());
        assert_eq!(core.managed_kv_cache.model_count(), 0);
    }

    #[test]
    fn core_rejects_unprepared_and_mismatched_chat_accounting() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.prompt_tokens = vec![11, 22, 33, 44];

        let error = core
            .add_request(request)
            .expect_err("public prompt tokens must not authorize scheduler accounting");
        assert!(error
            .to_string()
            .contains("missing exact model prompt preparation"));

        let mut mismatched = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        mismatched
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![11, 22], None, 4096)
            .unwrap();
        mismatched.prompt_tokens[0] = 99;
        let error = core
            .add_request(mismatched)
            .expect_err("mutated exact preparation must not reach the scheduler");
        assert!(error
            .to_string()
            .contains("changed after exact prompt preparation"));
    }

    #[test]
    fn core_enforces_exact_chat_context_after_preparation() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_seq_len: 4,
                ..EngineCoreConfig::default()
            },
            executor,
        )
        .unwrap();

        let mut full = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "full".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        full.install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3, 4], None, 4)
            .unwrap();
        assert!(core
            .add_request(full)
            .expect_err("a full context must leave no output allocation")
            .to_string()
            .contains("leaves no output capacity"));

        let mut bounded = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "bounded".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        bounded.id = "bounded-context".to_string();
        bounded.params.max_tokens = 100;
        bounded
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2], None, 4)
            .unwrap();
        core.add_request(bounded).unwrap();
        assert_eq!(core.requests["bounded-context"].params.max_tokens, 2);
    }

    #[tokio::test]
    async fn serialized_physical_batches_only_charge_each_request_its_own_elapsed_time() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let config = EngineCoreConfig {
            max_batch_size: 2,
            max_tokens_per_step: 2,
            min_tokens_per_step: 1,
            enable_chunked_prefill: false,
            enable_adaptive_batching: false,
            block_size: 1,
            max_blocks: 8,
            ..Default::default()
        };
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();
        for request_id in ["fast-batch", "slow-batch"] {
            let mut request = EngineCoreRequest::tts(request_id);
            request.id = request_id.to_string();
            request.prompt_tokens = vec![1];
            core.add_request(request).unwrap();
        }

        let prepared = core
            .prepare_step()
            .await
            .unwrap()
            .expect("prepared physical batches");
        let mut executed = core.execute_prepared_with_progress(prepared).await.unwrap();
        assert_eq!(executed.batches.len(), 2);
        for batch in &mut executed.batches {
            let request_id = batch.results[0].session.request_id.as_str();
            let elapsed = match request_id {
                "fast-batch" => Duration::from_millis(7),
                "slow-batch" => Duration::from_millis(31),
                other => panic!("unexpected batch row {other}"),
            };
            batch.report.elapsed = elapsed;
            for row in &mut batch.report.rows {
                row.execution.elapsed = elapsed;
            }
        }

        let committed = core.commit_step(executed).await.unwrap();
        let prefill_ms = committed
            .outputs
            .iter()
            .map(|output| {
                (
                    output.request_id.as_str(),
                    output
                        .latency_breakdown
                        .as_ref()
                        .expect("latency breakdown")
                        .prefill_ms,
                )
            })
            .collect::<HashMap<_, _>>();

        assert!((prefill_ms["fast-batch"] - 7.0).abs() < 0.001);
        assert!((prefill_ms["slow-batch"] - 31.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_step_preserves_optional_executor_phase_timings() {
        let executor = UnifiedExecutor::new_for_test(Box::new(PhaseTimingExecutor));
        let config = EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 1,
            max_blocks: 8,
            ..Default::default()
        };
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut request = EngineCoreRequest::tts("phase timing");
        request.id = "phase-req".to_string();
        request.prompt_tokens = vec![1, 2, 3];

        core.add_request(request).unwrap();
        let outputs = core.step().await.unwrap();

        let latency = outputs[0]
            .latency_breakdown
            .as_ref()
            .expect("latency breakdown");
        assert_eq!(latency.media_decode_ms, Some(12.5));
        assert_eq!(latency.sampling_ms, Some(1.25));
        assert!(
            latency.prefill_ms >= 0.0,
            "scheduler prefill timing remains available"
        );
        assert_eq!(latency.decode_ms, 0.0);
    }
}
