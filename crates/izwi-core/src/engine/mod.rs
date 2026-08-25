//! Production-ready inference engine following vLLM architecture patterns.
//!
//! This module implements a high-throughput audio inference engine with:
//! - Request scheduling with FCFS/priority policies
//! - Continuous batching for improved throughput
//! - Paged KV-cache memory management
//! - Streaming output support
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         Engine                                   │
//! │  ┌──────────────┐  ┌───────────┐  ┌──────────────────────────┐ │
//! │  │   Request    │  │           │  │      Engine Core          │ │
//! │  │  Processor   │──│ Scheduler │──│  ┌────────────────────┐  │ │
//! │  │              │  │           │  │  │  Model Executor    │  │ │
//! │  └──────────────┘  └───────────┘  │  │  (Native Rust)     │  │ │
//! │                                    │  └────────────────────┘  │ │
//! │  ┌──────────────┐                 │  ┌────────────────────┐  │ │
//! │  │   Output     │◄────────────────│  │  KV Cache Manager  │  │ │
//! │  │  Processor   │                 │  └────────────────────┘  │ │
//! │  └──────────────┘                 └──────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod cache;
#[cfg(test)]
pub(crate) use cache::managed::{plan_managed_state_capacity, ManagedStateCapacityRequest};
mod config;
mod core;
pub mod execution;
mod execution_group;
mod executor;
pub(crate) use executor::{decode_request_audio_with_rate, qwen3_asr_requires_long_form};
pub mod metrics;
mod output;
mod request;
pub mod resources;
mod scheduler;
pub mod signal_frontend;
mod types;

#[allow(unused_imports)]
pub(crate) use cache::invocation::{
    InvocationPagedKvCompletion, InvocationPagedKvLease, InvocationPagedKvPoolHandle,
    InvocationPagedKvPoolId, InvocationPagedKvPoolOwner, InvocationPagedKvSlotRef,
};
#[allow(unused_imports)]
pub(crate) use cache::invocation_tensor::{
    InvocationStaticAttentionLease, InvocationTensorLease, InvocationTensorPoolHandle,
    InvocationTensorPoolId, InvocationTensorPoolOwner, InvocationTensorSlotRef,
};

pub(crate) use cache::composite::{
    CompositeRetainedStateRuntimeIdV2, CompositeRetainedStateRuntimeV2,
};
#[cfg(test)]
pub(crate) use cache::managed::ManagedKvCacheManager;
pub(crate) use cache::managed::ManagedKvModelRuntime;
pub use cache::managed::{
    ManagedKvArenaRuntimeSnapshot, ManagedKvCoordinatorSnapshot, ManagedKvModelRuntimeSnapshot,
    ManagedKvOperationSnapshot, ManagedKvRuntimeSnapshot, ManagedKvRuntimeTotalsSnapshot,
};
#[cfg(test)]
pub(crate) use cache::physical::PhysicalStateManager;
pub(crate) use cache::physical::{RetainedTensorStateRuntimeIdV2, RetainedTensorStateRuntimeV2};
#[allow(unused_imports)]
pub(crate) use cache::retained_static_attention::{
    RetainedStaticAttentionBatchRow, RetainedStaticAttentionRuntimeIdV2,
    RetainedStaticAttentionRuntimeV2, RetainedStaticAttentionSequenceId,
    RetainedStaticAttentionTransactionId,
};
pub use cache::telemetry::ManagedKvTelemetrySnapshot;
pub(crate) use config::resolve_backend_model_context;
pub(crate) use config::tts_explicit_output_limit;
pub use config::EngineCoreConfig;
pub use core::EngineCore;
pub use execution::{
    AdapterAbiRevision, AdapterBindingKey, AdapterInstanceId, BatchBudget, BatchDispatch,
    BatchDispatchKind, BatchId, BatchKey, BatchLaneKey, CacheMode, CancellationGranularity,
    ConcurrencyClass, DeadlinePhase, DispatchState, ExecutionAdapterBinding, ExecutionCapabilities,
    ExecutionDisposition, ExecutionDomain, ExecutionFailure, ExecutionGroupId, ExecutionMode,
    ExecutionPlan, ExecutionProfile, ExecutionReport, ExecutionState, ExecutionTracker,
    FailureKind, FailureOrigin, FailureScope, FinishReason, HealthImpact, InputRange,
    ManagedCacheDomainReceipt, ManagedCacheDomainReservation, ManagedCacheReceipt,
    ManagedCacheReservation, ManagedSessionGeneration, ManagedTensorStateReservation,
    MembershipSafePoint, ModelInstanceId, NativeBatchMode, OutcomeProvenance, OutputVisibility,
    PhysicalBatch, PhysicalBatchReport, PhysicalBatchRowReport, PhysicalLaunchPolicy, PlanId,
    PrefillMode, ReadyQuantum, RetryDisposition, SequencePhase, SessionEpoch, SessionKey,
    StageDescriptor, StageId, StageProgressKind, StageShapePolicy, StageWorkSelector,
    StateDisposition, TerminalOutcome, WorkCost, WorkUnit, YieldReason,
};
pub use executor::{
    CacheReleaseOutcome, CacheReleaseReport, ExecutorOutput, ExecutorStepResult, ModelExecutor,
    ModelSessionResult, PhysicalBatchExecution, PhysicalDispatchError, PhysicalDispatchResult,
    WorkerConfig, REQUEST_DEADLINE_EXCEEDED,
};
pub use metrics::{
    engine_batch_metrics_snapshot, engine_metric_catalog,
    engine_physical_execution_metrics_snapshot, engine_request_parallel_batches_total,
    engine_stream_backpressure_total, engine_stream_metrics_snapshot,
    engine_tensor_batch_max_width, engine_tensor_batches_total, prometheus_engine_metric_name,
    prometheus_engine_metric_type, BenchmarkResult, EngineBatchMetricsSnapshot,
    EngineDeadlinePhaseMetricsSnapshot, EngineDispatchStateMetricsSnapshot,
    EngineDurationMetricsSnapshot, EngineFailureOriginMetricsSnapshot, EngineMetricDescriptor,
    EnginePhysicalDeferMetricsSnapshot, EnginePhysicalDeferReason,
    EnginePhysicalExecutionMetricsSnapshot, EnginePhysicalExecutionMode,
    EnginePhysicalFallbackMetricsSnapshot, EnginePhysicalFallbackReason,
    EngineStreamMetricsSnapshot, EngineWorkspaceDomainMetricsSnapshot, MetricsCollector,
    MetricsSnapshot, ENGINE_EXECUTOR_BATCH_WORKSPACE_BYTES_TOTAL,
    ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL,
    ENGINE_EXECUTOR_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS_TOTAL,
    ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL, ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL,
    ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL, ENGINE_EXECUTOR_MODEL_DECODE_CALLS_TOTAL,
    ENGINE_EXECUTOR_MODEL_SCALAR_ROW_DISPATCHES_TOTAL, ENGINE_EXECUTOR_MODEL_TENSOR_BATCHES_TOTAL,
    ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_MAX_WIDTH, ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_ROWS_TOTAL,
    ENGINE_EXECUTOR_MODEL_TENSOR_MULTIROW_CALLS_TOTAL, ENGINE_EXECUTOR_PHYSICAL_BATCHES_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_BATCH_CAPACITY_ROWS_TOTAL, ENGINE_EXECUTOR_PHYSICAL_BATCH_FILL_RATIO,
    ENGINE_EXECUTOR_PHYSICAL_BATCH_MATERIALIZED_ELEMENTS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_BATCH_MAX_WIDTH, ENGINE_EXECUTOR_PHYSICAL_BATCH_PADDING_RATIO,
    ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL, ENGINE_EXECUTOR_PHYSICAL_BATCH_ROWS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_BATCH_USEFUL_ELEMENTS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_COHORT_WAIT_OBSERVATIONS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_COHORT_WAIT_SECONDS_MAX,
    ENGINE_EXECUTOR_PHYSICAL_COHORT_WAIT_SECONDS_TOTAL, ENGINE_EXECUTOR_PHYSICAL_DEFERS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_DISPATCHES_COMPLETED_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_DISPATCHES_IN_FLIGHT,
    ENGINE_EXECUTOR_PHYSICAL_DISPATCHES_MAX_IN_FLIGHT,
    ENGINE_EXECUTOR_PHYSICAL_DISPATCHES_STARTED_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_DISPATCH_OBSERVATIONS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_DISPATCH_SECONDS_MAX, ENGINE_EXECUTOR_PHYSICAL_DISPATCH_SECONDS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_EXECUTION_CAP, ENGINE_EXECUTOR_PHYSICAL_EXECUTION_MODE,
    ENGINE_EXECUTOR_PHYSICAL_FALLBACKS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_PERMIT_WAIT_OBSERVATIONS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_PERMIT_WAIT_SECONDS_MAX,
    ENGINE_EXECUTOR_PHYSICAL_PERMIT_WAIT_SECONDS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_WORKSPACE_CURRENT_BYTES,
    ENGINE_EXECUTOR_PHYSICAL_WORKSPACE_HIGH_WATER_BYTES,
    ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_BATCH_CAPACITY_ROWS_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO,
    ENGINE_EXECUTOR_TENSOR_BATCH_MATERIALIZED_ELEMENTS_TOTAL,
    ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH, ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO,
    ENGINE_EXECUTOR_TENSOR_BATCH_ROWS_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCH_USEFUL_ELEMENTS_TOTAL,
    ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_CONTINUOUS_MULTIROW_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL, ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
    ENGINE_KV_CACHE_EVICTIONS_TOTAL, ENGINE_KV_CACHE_FREE_BLOCKS,
    ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS, ENGINE_KV_CACHE_HITS_TOTAL,
    ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES, ENGINE_KV_CACHE_MEMORY_USED_BYTES,
    ENGINE_KV_CACHE_MISSES_TOTAL, ENGINE_KV_CACHE_UTILIZATION_RATIO, ENGINE_METRIC_CATALOG,
    ENGINE_SCHEDULER_INCREMENTAL_PREFILL_QUANTA_COMMITTED_TOTAL,
    ENGINE_SCHEDULER_INCREMENTAL_PREFILL_TOKENS_COMMITTED_TOTAL,
    ENGINE_SCHEDULER_MULTISPAN_PREFILL_REQUESTS_TOTAL, ENGINE_SCHEDULER_QUEUE_DEPTH,
    ENGINE_SCHEDULER_RUNNING_REQUESTS, ENGINE_SCHEDULER_STEP_TOKENS_TOTAL,
    ENGINE_STREAM_BACKPRESSURE_TOTAL, ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL,
    ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL, ENGINE_STREAM_DELIVERY_FAILURES_TOTAL,
};
pub use output::{AsrProgress, AsrProgressPhase, OutputProcessor, StreamingOutput};
pub use request::{
    AsrEngineInput, AudioChatEngineInput, ChatEngineInput, EngineAudioInput, EngineCoreRequest,
    EngineStreamPolicy, EngineTask, RequestProcessor, RequestStatus, TtsEngineInput, WorkloadClass,
};
pub use resources::{
    BatchWorkspaceLease, CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot,
    ReservationClass, ReservationId, ReservationOwner, ResourceAmount, ResourceAuthority,
    ResourceAuthoritySnapshot, ResourceEstimate, ResourceLease, ResourceLedger,
    ResourceReservation, ResourceVector,
};
pub use scheduler::{ScheduleResult, Scheduler, SchedulerConfig, SchedulingPolicy};
pub use types::FinishReason as OutputFinishReason;
pub use types::{
    AudioOutput, EngineMetrics, EngineOutput, GenerationParams, Priority, RequestId, SequenceId,
    TaskType, TokenId,
};

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::asr::{Qwen3AsrAudioBatchRow, Qwen3AsrPreparedAudio};
use crate::models::registry::{ChatModelLease, ModelRegistry, NativeChatPreparedPrompt};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot, Mutex, Notify, RwLock, Semaphore};
use tracing::{debug, info, warn};

/// Main inference engine - the primary interface for audio generation.
///
/// The engine orchestrates all components and provides both synchronous
/// and asynchronous interfaces for inference.
struct RequestControl {
    session_epoch: SequenceId,
    cancellation: Arc<std::sync::atomic::AtomicBool>,
    model_variant: Option<ModelVariant>,
}

struct CompletionMailbox {
    registration_id: u64,
    session_epoch: Option<SequenceId>,
    sender: oneshot::Sender<EngineOutput>,
}

struct CompletionRegistration<'a> {
    engine: &'a Engine,
    request_id: RequestId,
    registration_id: u64,
}

impl Drop for CompletionRegistration<'_> {
    fn drop(&mut self) {
        let session_epoch = {
            let mut mailboxes = self
                .engine
                .completion_mailboxes
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if mailboxes
                .get(&self.request_id)
                .is_some_and(|mailbox| mailbox.registration_id == self.registration_id)
            {
                mailboxes
                    .remove(&self.request_id)
                    .and_then(|mailbox| mailbox.session_epoch)
            } else {
                None
            }
        };
        let Some(session_epoch) = session_epoch else {
            return;
        };

        let session = SessionKey::new(self.request_id.clone(), session_epoch);
        if let Some(control) = self
            .engine
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&self.request_id)
        {
            if control.session_epoch == session_epoch {
                control
                    .cancellation
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        self.engine.wake_notify.notify_one();

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let core = Arc::downgrade(&self.engine.core);
        let step_gate = self.engine.step_gate.clone();
        let controls = self.engine.request_controls.clone();
        let wake_notify = self.engine.wake_notify.clone();
        handle.spawn(async move {
            let initial_delay = {
                let _step = step_gate.lock().await;
                let Some(core) = core.upgrade() else {
                    return;
                };
                let mut core = core.write().await;
                if !core.abandon_request_session(&session).await {
                    return;
                }
                core.abandoned_session_cleanup_delay(&session)
            };

            {
                let mut controls = controls.lock().unwrap_or_else(|poison| poison.into_inner());
                if controls
                    .get(&session.request_id)
                    .is_some_and(|control| control.session_epoch == session.epoch)
                {
                    controls.remove(&session.request_id);
                }
            }
            wake_notify.notify_one();

            let mut retry_delay = initial_delay;
            while let Some(delay) = retry_delay {
                tokio::time::sleep(delay).await;
                let _step = step_gate.lock().await;
                retry_delay = {
                    let Some(core) = core.upgrade() else {
                        return;
                    };
                    let mut core = core.write().await;
                    core.retry_abandoned_session_cleanup(&session).await
                };
            }
        });
    }
}

pub struct Engine {
    /// Engine core handles the actual inference loop
    core: Arc<RwLock<EngineCore>>,
    /// Serializes one complete prepare/execute/commit transaction without
    /// keeping the mutable engine state locked during device execution.
    step_gate: Arc<Mutex<()>>,
    /// Request processor validates and preprocesses inputs
    request_processor: RequestProcessor,
    /// Output processor formats results for clients
    output_processor: OutputProcessor,
    /// Configuration
    config: EngineCoreConfig,
    /// Loaded models used to prepare exact public chat prompts before admission.
    model_registry: Option<Arc<ModelRegistry>>,
    /// Bounds direct request preprocessing that runs outside Runtime admission.
    direct_request_preparation_permits: Arc<Semaphore>,
    /// Whether the engine is running
    running: std::sync::atomic::AtomicBool,
    /// Metrics collector
    metrics: Arc<RwLock<EngineMetrics>>,
    /// Event-driven wakeup for run-loop when new requests arrive.
    wake_notify: Arc<Notify>,
    /// Session-fenced cooperative cancellation signals available without the core lock.
    request_controls: Arc<std::sync::Mutex<HashMap<RequestId, RequestControl>>>,
    /// Exact-session terminal outputs for synchronous public callers.
    completion_mailboxes: Arc<std::sync::Mutex<HashMap<RequestId, CompletionMailbox>>>,
    /// Distinguishes a cancelled registration from a later reuse of the public ID.
    next_completion_registration: std::sync::atomic::AtomicU64,
}

/// Cloneable state for one owned engine transaction. The task holding this
/// context is intentionally detached from the caller's future so cancellation
/// cannot interrupt the prepare/execute/commit sequence.
struct OwnedStepContext {
    core: Arc<RwLock<EngineCore>>,
    step_gate: Arc<Mutex<()>>,
    metrics: Arc<RwLock<EngineMetrics>>,
    request_controls: Arc<std::sync::Mutex<HashMap<RequestId, RequestControl>>>,
    completion_mailboxes: Arc<std::sync::Mutex<HashMap<RequestId, CompletionMailbox>>>,
}

struct OwnedRunnerRecoveryGuard {
    core: Arc<RwLock<EngineCore>>,
    recovery: Option<execution_group::PreparedStepRecovery>,
    abort: tokio::task::AbortHandle,
}

impl OwnedRunnerRecoveryGuard {
    async fn recover_now(&mut self) {
        self.abort.abort();
        let Some(recovery) = self.recovery.take() else {
            return;
        };
        recovery.wait_for_task_drain().await;
        self.core
            .write()
            .await
            .rollback_in_flight_dispatches(recovery.batch_ids());
    }

    fn disarm(&mut self) {
        self.recovery = None;
    }
}

impl Drop for OwnedRunnerRecoveryGuard {
    fn drop(&mut self) {
        let Some(recovery) = self.recovery.take() else {
            return;
        };
        self.abort.abort();
        let core = self.core.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                recovery.wait_for_task_drain().await;
                core.write()
                    .await
                    .rollback_in_flight_dispatches(recovery.batch_ids());
            });
        }
    }
}

impl OwnedStepContext {
    fn take_completion_sender(
        &self,
        session: &SessionKey,
    ) -> Option<oneshot::Sender<EngineOutput>> {
        let mut mailboxes = self
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let owns_session = mailboxes
            .get(&session.request_id)
            .is_some_and(|mailbox| mailbox.session_epoch == Some(session.epoch));
        owns_session
            .then(|| mailboxes.remove(&session.request_id))
            .flatten()
            .map(|mailbox| mailbox.sender)
    }

    fn cancel_failed_stream(&self, failure: &executor::StreamDeliveryFailure) {
        let controls = self
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(control) = controls.get(&failure.session.request_id) {
            if control.session_epoch == failure.session.epoch {
                control
                    .cancellation
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
    }

    async fn commit_incremental_progress(
        &self,
        progress: request::FencedStreamProgress,
    ) -> std::result::Result<executor::CommittedStreamDelivery, executor::StreamDeliveryFailure>
    {
        let session = progress.session.clone();
        let res = {
            let mut core = self.core.write().await;
            core.commit_incremental_stream_progress(progress)
        };
        match res {
            Ok(delivery) => Ok(delivery),
            Err(error) => {
                warn!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    error = %error,
                    "Rejecting invalid incremental stream progress"
                );
                Err(executor::StreamDeliveryFailure {
                    session,
                    kind: error.kind,
                })
            }
        }
    }

    fn record_stream_failure(
        &self,
        failure: executor::StreamDeliveryFailure,
        failures: &mut HashMap<SessionKey, executor::StreamDeliveryFailure>,
        deliveries: &mut executor::IncrementalStreamDeliveryWorkers,
    ) {
        self.cancel_failed_stream(&failure);
        deliveries.abandon_session(&failure.session);
        failures.entry(failure.session.clone()).or_insert(failure);
    }

    async fn enqueue_incremental_progress(
        &self,
        progress: request::FencedStreamProgress,
        failures: &mut HashMap<SessionKey, executor::StreamDeliveryFailure>,
        deliveries: &mut executor::IncrementalStreamDeliveryWorkers,
    ) {
        if failures.contains_key(&progress.session) {
            return;
        }
        let result = match self.commit_incremental_progress(progress).await {
            Ok(delivery) => deliveries.enqueue(delivery),
            Err(failure) => Err(failure),
        };
        if let Err(failure) = result {
            self.record_stream_failure(failure, failures, deliveries);
        }
    }

    async fn execute_prepared(
        &self,
        prepared: execution_group::PreparedEngineStep,
        defer_unregistered_terminal_ack: bool,
    ) -> Result<Vec<EngineOutput>> {
        let (progress_tx, mut progress_rx) = mpsc::channel(request::STREAM_PROGRESS_QUEUE_CAPACITY);
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel();
        let progress_budget =
            request::StreamProgressBudget::new(request::STREAM_PROGRESS_MAX_BUFFERED_BYTES);
        let recovery = prepared.recovery();
        let runner_registration = recovery.register_runner();
        let mut runner = tokio::spawn(async move {
            execution_group::ExecutionGroupRunner::execute(
                prepared,
                runner_registration,
                progress_tx,
                progress_budget,
                Some(completion_tx),
            )
            .await
        });
        let mut recovery_guard = OwnedRunnerRecoveryGuard {
            core: self.core.clone(),
            recovery: Some(recovery.clone()),
            abort: runner.abort_handle(),
        };
        let (mut deliveries, mut delivery_failures) =
            executor::IncrementalStreamDeliveryWorkers::new();
        let mut failures = HashMap::new();
        let mut progress_closed = false;
        let mut completion_closed = false;
        let mut delivery_failures_closed = false;
        let mut runner_finished = false;
        let mut fallback_batches = Vec::new();
        let mut completed_outputs = HashMap::<BatchId, Vec<EngineOutput>>::new();

        while !runner_finished || !completion_closed {
            tokio::select! {
                result = &mut runner, if !runner_finished => {
                    match result {
                        Ok(executed) => fallback_batches.extend(executed.batches),
                        Err(error) if error.is_panic() => {
                            recovery_guard.recover_now().await;
                            std::panic::resume_unwind(error.into_panic())
                        }
                        Err(error) => {
                            recovery_guard.recover_now().await;
                            return Err(Error::InferenceError(format!(
                                "execution group task was cancelled: {error}"
                            )));
                        }
                    }
                    runner_finished = true;
                }
                progress = progress_rx.recv(), if !progress_closed => {
                    match progress {
                        Some(progress) => {
                            self.enqueue_incremental_progress(
                                progress,
                                &mut failures,
                                &mut deliveries,
                            ).await;
                        }
                        None => progress_closed = true,
                    }
                }
                completion = completion_rx.recv(), if !completion_closed => {
                    match completion {
                        Some(batch) => {
                            let batch_id = batch.physical_batch.batch_id;
                            let outputs = self.commit_completed_dispatch(
                                batch,
                                &mut progress_rx,
                                &mut delivery_failures,
                                &mut failures,
                                &mut deliveries,
                                defer_unregistered_terminal_ack,
                            ).await?;
                            completed_outputs.insert(batch_id, outputs);
                        }
                        None => completion_closed = true,
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
        }

        for batch in fallback_batches {
            let batch_id = batch.physical_batch.batch_id;
            let outputs = self
                .commit_completed_dispatch(
                    batch,
                    &mut progress_rx,
                    &mut delivery_failures,
                    &mut failures,
                    &mut deliveries,
                    defer_unregistered_terminal_ack,
                )
                .await?;
            completed_outputs.insert(batch_id, outputs);
        }

        while let Some(progress) = progress_rx.recv().await {
            self.enqueue_incremental_progress(progress, &mut failures, &mut deliveries)
                .await;
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

        // An empty prepared step can still own durable terminal outbox rows.
        let tail = {
            let mut core = self.core.write().await;
            core.commit_step(execution_group::ExecutedEngineStep {
                batches: Vec::new(),
            })
            .await?
        };
        let mut tail_outputs = self
            .deliver_and_route_committed(tail, defer_unregistered_terminal_ack)
            .await?;

        let mut outputs = Vec::new();
        for batch_id in recovery.batch_ids() {
            if let Some(mut batch_outputs) = completed_outputs.remove(batch_id) {
                outputs.append(&mut batch_outputs);
            }
        }
        for mut unordered in completed_outputs.into_values() {
            outputs.append(&mut unordered);
        }
        outputs.append(&mut tail_outputs);
        recovery_guard.disarm();
        Ok(outputs)
    }

    async fn commit_completed_dispatch(
        &self,
        batch: execution_group::ExecutedPhysicalBatch,
        progress_rx: &mut mpsc::Receiver<request::FencedStreamProgress>,
        delivery_failures: &mut mpsc::UnboundedReceiver<executor::StreamDeliveryFailure>,
        failures: &mut HashMap<SessionKey, executor::StreamDeliveryFailure>,
        deliveries: &mut executor::IncrementalStreamDeliveryWorkers,
        defer_unregistered_terminal_ack: bool,
    ) -> Result<Vec<EngineOutput>> {
        while let Ok(progress) = progress_rx.try_recv() {
            self.enqueue_incremental_progress(progress, failures, deliveries)
                .await;
        }
        let sessions = batch
            .results
            .iter()
            .map(|result| result.session.clone())
            .collect::<HashSet<_>>();
        for failure in deliveries.finish_sessions(&sessions).await {
            self.record_stream_failure(failure, failures, deliveries);
        }
        while let Ok(failure) = delivery_failures.try_recv() {
            self.record_stream_failure(failure, failures, deliveries);
        }
        let dispatch_failures = failures
            .values()
            .filter(|failure| sessions.contains(&failure.session))
            .cloned()
            .collect::<Vec<_>>();
        let mut executed = execution_group::ExecutedEngineStep {
            batches: vec![batch],
        };
        executed.apply_stream_delivery_failures(&dispatch_failures);
        let committed = {
            let mut core = self.core.write().await;
            core.commit_step(executed).await?
        };
        self.deliver_and_route_committed(committed, defer_unregistered_terminal_ack)
            .await
    }

    async fn deliver_and_route_committed(
        &self,
        committed: core::CommittedEngineStep,
        defer_unregistered_terminal_ack: bool,
    ) -> Result<Vec<EngineOutput>> {
        let mut outputs = committed.outputs;
        let failed_streams = executor::deliver_committed_streams(committed.stream_deliveries).await;
        if !failed_streams.is_empty() {
            let mut core = self.core.write().await;
            core.reconcile_stream_delivery_failures(&mut outputs, failed_streams)
                .await;
        }

        let mut core = self.core.write().await;
        for output in outputs.iter().filter(|output| output.is_finished) {
            let session = SessionKey::new(output.request_id.clone(), output.sequence_id);
            let routed_to_mailbox = if let Some(sender) = self.take_completion_sender(&session) {
                let _ = sender.send(output.clone());
                true
            } else {
                false
            };
            if (routed_to_mailbox || !defer_unregistered_terminal_ack)
                && !core.acknowledge_terminal_output(&session)
            {
                warn!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    "Terminal output had no matching delivery fence"
                );
            }
            let mut controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if controls
                .get(&output.request_id)
                .is_some_and(|control| control.session_epoch == output.sequence_id)
            {
                controls.remove(&output.request_id);
            }
        }
        Ok(outputs)
    }

    async fn run(self, defer_unregistered_terminal_ack: bool) -> Result<Vec<EngineOutput>> {
        let _step = self.step_gate.lock().await;
        let prepared = {
            let mut core = self.core.write().await;
            core.prepare_step().await?
        };
        let outputs = match prepared {
            Some(prepared) => {
                self.execute_prepared(prepared, defer_unregistered_terminal_ack)
                    .await?
            }
            None => Vec::new(),
        };

        // Keep every await before terminal dispatch. Once a completion sender
        // is notified, routing and exact-session acknowledgement finish
        // synchronously inside this owned transaction.
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_steps += 1;
            metrics.requests_processed += outputs.len() as u64;
        }

        Ok(outputs)
    }
}

impl Engine {
    fn physical_execution_telemetry_mode(
        mode: crate::config::PhysicalExecutionMode,
    ) -> EnginePhysicalExecutionMode {
        match mode {
            crate::config::PhysicalExecutionMode::Serial => EnginePhysicalExecutionMode::Serial,
            crate::config::PhysicalExecutionMode::Shadow => EnginePhysicalExecutionMode::Shadow,
            crate::config::PhysicalExecutionMode::Concurrent => {
                EnginePhysicalExecutionMode::Concurrent
            }
        }
    }

    fn physical_execution_telemetry_policy(
        config: &EngineCoreConfig,
        admitted_capacity: Option<usize>,
    ) -> (EnginePhysicalExecutionMode, usize) {
        let capacity = config.resolved_physical_execution_capacity();
        (
            Self::physical_execution_telemetry_mode(config.physical_execution_mode),
            admitted_capacity
                .unwrap_or_else(|| capacity.physical_launch_limit.get())
                .max(1),
        )
    }

    fn queue_capacity_from_env(key: &str) -> Option<usize> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn streaming_queue_capacity(request: &EngineCoreRequest) -> usize {
        let default_capacity = match request.task_type {
            TaskType::TTS => 8usize,
            // Unified speech-to-speech emits bursty interleaved text and audio
            // chunks, so it needs a deeper queue than plain TTS.
            TaskType::SpeechToSpeech => 64usize,
            // ASR can emit per-character deltas in streaming mode.
            // Use a deeper default queue to absorb bursty decode emission.
            TaskType::ASR => 4096usize,
            TaskType::Chat => 64usize,
        };

        let task_override = match request.task_type {
            TaskType::TTS | TaskType::SpeechToSpeech => {
                Self::queue_capacity_from_env("IZWI_STREAM_AUDIO_QUEUE_CAPACITY")
            }
            TaskType::ASR | TaskType::Chat => {
                Self::queue_capacity_from_env("IZWI_STREAM_TEXT_QUEUE_CAPACITY")
            }
        };

        task_override
            .or_else(|| Self::queue_capacity_from_env("IZWI_STREAM_QUEUE_CAPACITY"))
            .unwrap_or(default_capacity)
    }

    /// Create a new inference engine with the given configuration.
    pub fn new(config: EngineCoreConfig) -> Result<Self> {
        let worker_config = WorkerConfig::from(&config);
        Self::new_with_worker(config, worker_config)
    }

    /// Create a new inference engine with explicit worker configuration.
    pub fn new_with_worker(config: EngineCoreConfig, worker_config: WorkerConfig) -> Result<Self> {
        info!("Initializing inference engine");

        let model_registry = worker_config.model_registry.clone();
        let admitted_capacity = worker_config
            .physical_execution_admission
            .as_ref()
            .map(|admission| admission.capacity());
        let core = EngineCore::new_with_worker(config.clone(), worker_config)?;
        let (physical_execution_mode, effective_physical_execution_cap) =
            Self::physical_execution_telemetry_policy(&config, admitted_capacity);
        metrics::set_engine_effective_physical_execution(
            physical_execution_mode,
            effective_physical_execution_cap,
        );
        let request_processor = RequestProcessor::new(config.clone());
        let output_processor = OutputProcessor::new(config.sample_rate);
        let direct_request_preparation_capacity = config.max_batch_size.max(1);

        Ok(Self {
            core: Arc::new(RwLock::new(core)),
            step_gate: Arc::new(Mutex::new(())),
            request_processor,
            output_processor,
            config,
            model_registry,
            direct_request_preparation_permits: Arc::new(Semaphore::new(
                direct_request_preparation_capacity,
            )),
            running: std::sync::atomic::AtomicBool::new(false),
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
            wake_notify: Arc::new(Notify::new()),
            request_controls: Arc::new(std::sync::Mutex::new(HashMap::new())),
            completion_mailboxes: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_completion_registration: std::sync::atomic::AtomicU64::new(1),
        })
    }

    fn register_completion_mailbox(
        &self,
        request_id: RequestId,
    ) -> Result<(CompletionRegistration<'_>, oneshot::Receiver<EngineOutput>)> {
        use std::collections::hash_map::Entry;
        use std::sync::atomic::Ordering;

        let registration_id = self
            .next_completion_registration
            .fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        let mut mailboxes = self
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match mailboxes.entry(request_id.clone()) {
            Entry::Occupied(_) => {
                return Err(crate::error::Error::InvalidInput(format!(
                    "Request {request_id} already has a completion waiter"
                )));
            }
            Entry::Vacant(entry) => {
                entry.insert(CompletionMailbox {
                    registration_id,
                    session_epoch: None,
                    sender,
                });
            }
        }
        drop(mailboxes);

        Ok((
            CompletionRegistration {
                engine: self,
                request_id,
                registration_id,
            },
            receiver,
        ))
    }

    fn bind_completion_mailbox(
        &self,
        request_id: &RequestId,
        registration_id: u64,
        session_epoch: SequenceId,
    ) {
        let mut mailboxes = self
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mailbox = mailboxes
            .get_mut(request_id)
            .filter(|mailbox| mailbox.registration_id == registration_id)
            .expect("completion registration must remain live while its request is admitted");
        mailbox.session_epoch = Some(session_epoch);
    }

    fn resolve_generation_output(
        request_id: &RequestId,
        output: EngineOutput,
    ) -> Result<EngineOutput> {
        if output.request_id != *request_id {
            return Err(crate::error::Error::InferenceError(format!(
                "Completion mailbox for {request_id} received output for {}",
                output.request_id
            )));
        }
        if output.finish_reason == Some(types::FinishReason::Aborted) {
            return Err(crate::error::Error::Cancelled(request_id.clone()));
        }
        if let Some(err) = output.error.clone() {
            return Err(crate::error::Error::InferenceError(err));
        }
        Ok(output)
    }

    async fn prepare_direct_chat_request_with<F>(
        request: EngineCoreRequest,
        preparation_permits: Arc<Semaphore>,
        prepare: F,
    ) -> Result<EngineCoreRequest>
    where
        F: FnOnce(
                &EngineCoreRequest,
            ) -> Result<(
                Vec<TokenId>,
                Option<NativeChatPreparedPrompt>,
                Option<(ChatModelLease, usize)>,
            )> + Send
            + 'static,
    {
        if request.task_type != TaskType::Chat {
            return Ok(request);
        }
        if request.has_chat_execution_preparation() {
            request.validate_chat_execution_preparation()?;
            return Ok(request);
        }
        if !request.chat_config.media_inputs.is_empty() {
            return Err(Error::InvalidInput(
                "Direct Engine multimodal chat is not supported; use RuntimeService so media preparation is resource-admitted"
                    .to_string(),
            ));
        }
        let model_variant = request.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Chat request {} is missing a model variant for prompt preparation",
                request.id
            ))
        })?;
        if request
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Error::Timeout(request.id.clone()));
        }

        let request_id = request.id.clone();
        let deadline = request.deadline;
        let acquire_permit = preparation_permits.acquire_owned();
        let permit = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire_permit)
                .await
                .map_err(|_| Error::Timeout(request_id.clone()))?,
            None => acquire_permit.await,
        }
        .map_err(|_| {
            Error::InferenceError("Direct chat preparation queue is unavailable".to_string())
        })?;

        let worker = tokio::task::spawn_blocking(move || -> Result<EngineCoreRequest> {
            // Keep the permit inside the blocking closure: timeout/cancellation
            // drops the JoinHandle but cannot stop native/tokenizer work already
            // running on Tokio's blocking pool.
            let _permit = permit;
            let mut request = request;
            let (prompt_tokens, prepared_chat_prompt, model) = prepare(&request)?;
            if let Some((model, context_limit)) = model {
                request.install_chat_execution_preparation_with_model(
                    model_variant,
                    prompt_tokens,
                    prepared_chat_prompt,
                    model,
                    context_limit,
                )?;
            } else {
                #[cfg(test)]
                request.install_chat_execution_preparation(
                    model_variant,
                    prompt_tokens,
                    prepared_chat_prompt,
                    4096,
                )?;
                #[cfg(not(test))]
                return Err(Error::InferenceError(format!(
                    "Chat request {} preparation did not retain its model instance",
                    request.id
                )));
            }
            Ok(request)
        });
        let prepared = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), worker)
                .await
                .map_err(|_| Error::Timeout(request_id.clone()))?,
            None => worker.await,
        }
        .map_err(|join_error| {
            Error::InferenceError(format!(
                "Chat request {request_id} prompt preparation worker failed: {join_error}"
            ))
        })??;
        if prepared
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Error::Timeout(prepared.id));
        }
        Ok(prepared)
    }

    async fn prepare_direct_non_chat_request_with<F>(
        request: EngineCoreRequest,
        preparation_permits: Arc<Semaphore>,
        prepare: F,
    ) -> Result<EngineCoreRequest>
    where
        F: FnOnce(EngineCoreRequest) -> Result<EngineCoreRequest> + Send + 'static,
    {
        debug_assert_ne!(request.task_type, TaskType::Chat);
        if request
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Error::Timeout(request.id));
        }

        let request_id = request.id.clone();
        let deadline = request.deadline;
        let acquire_permit = preparation_permits.acquire_owned();
        let permit = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire_permit)
                .await
                .map_err(|_| Error::Timeout(request_id.clone()))?,
            None => acquire_permit.await,
        }
        .map_err(|_| {
            Error::InferenceError("Direct request preparation queue is unavailable".to_string())
        })?;

        let worker = tokio::task::spawn_blocking(move || {
            // A timed-out caller cannot cancel blocking work. Retain the permit
            // until the owned request and its validation scan are fully dropped.
            let _permit = permit;
            prepare(request)
        });
        let prepared = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), worker)
                .await
                .map_err(|_| Error::Timeout(request_id.clone()))?,
            None => worker.await,
        }
        .map_err(|join_error| {
            Error::InferenceError(format!(
                "Direct request {request_id} preparation worker failed: {join_error}"
            ))
        })??;
        if prepared
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Error::Timeout(prepared.id));
        }
        Ok(prepared)
    }

    async fn prepare_direct_non_chat_request_for_execution(
        &self,
        request: EngineCoreRequest,
    ) -> Result<EngineCoreRequest> {
        if request.task_type == TaskType::Chat {
            return Ok(request);
        }
        let max_seq_len = self.config.max_seq_len;
        let processor = RequestProcessor::new(self.config.clone());
        Self::prepare_direct_non_chat_request_with(
            request,
            self.direct_request_preparation_permits.clone(),
            move |mut request| {
                request.canonicalize_direct_payloads(max_seq_len)?;
                processor.process_canonicalized(request)
            },
        )
        .await
    }

    async fn prepare_chat_request_for_execution(
        &self,
        request: EngineCoreRequest,
    ) -> Result<EngineCoreRequest> {
        if request.task_type != TaskType::Chat {
            return Ok(request);
        }
        if request.has_chat_execution_preparation() {
            request.validate_chat_execution_preparation()?;
            return Ok(request);
        }
        if !request.chat_config.media_inputs.is_empty() {
            return Err(Error::InvalidInput(
                "Direct Engine multimodal chat is not supported; use RuntimeService so media preparation is resource-admitted"
                    .to_string(),
            ));
        }
        let registry = self.model_registry.clone().ok_or_else(|| {
            Error::InvalidInput(
                "Direct Engine chat requires a configured ModelRegistry with the routed model loaded"
                    .to_string(),
            )
        })?;
        let backend = self.config.backend;
        let configured_context_limit = self.config.max_seq_len;

        Self::prepare_direct_chat_request_with(
            request,
            self.direct_request_preparation_permits.clone(),
            move |request| {
                let variant = request.model_variant.ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Chat request {} is missing a model variant for prompt preparation",
                        request.id
                    ))
                })?;
                let messages = request.chat_messages.as_deref().ok_or_else(|| {
                    Error::InvalidInput(format!("Chat request {} is missing messages", request.id))
                })?;
                let model = registry.blocking_get_chat(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!("Chat model {variant} is not loaded"))
                })?;
                let context_limit =
                    registry
                        .effective_context(variant)
                        .unwrap_or(resolve_backend_model_context(
                            backend,
                            configured_context_limit,
                            model.max_context_tokens()?,
                        )?);
                let (prompt_tokens, prepared_chat_prompt) = model
                    .prepare_prompt_for_execution(messages, &request.chat_generation_config())?;
                Ok((
                    prompt_tokens,
                    prepared_chat_prompt,
                    Some((model, context_limit)),
                ))
            },
        )
        .await
    }

    async fn retain_incremental_model_identity(
        &self,
        mut request: EngineCoreRequest,
    ) -> Result<EngineCoreRequest> {
        let Some(registry) = self.model_registry.as_ref() else {
            return Ok(request);
        };
        let variant = request.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Request {} is missing a model variant for execution",
                request.id
            ))
        })?;

        match request.task_type {
            TaskType::ASR if variant.family() != crate::catalog::ModelFamily::Voxtral => {
                let model = registry.get_asr_lease(variant).await.ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
                })?;
                if variant.family() == crate::catalog::ModelFamily::Qwen3Asr
                    && request.prepared_asr_execution_shape().is_none()
                {
                    let model_for_shape = model.clone();
                    let request_id = request.id.clone();
                    let deadline = request.deadline;
                    let context_limit = registry
                        .effective_context(variant)
                        .unwrap_or(self.config.max_seq_len);
                    let acquire_permit = self
                        .direct_request_preparation_permits
                        .clone()
                        .acquire_owned();
                    let permit = match deadline {
                        Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire_permit)
                            .await
                            .map_err(|_| Error::Timeout(request_id.clone()))?,
                        None => acquire_permit.await,
                    }
                    .map_err(|_| {
                        Error::InferenceError(
                            "Direct Qwen3 ASR preparation queue is unavailable".to_string(),
                        )
                    })?;
                    let worker = tokio::task::spawn_blocking(move || {
                        // The tower's transient host/device work is bounded by
                        // the same Engine-owned preparation capacity as direct
                        // payload processing. Its immutable result moves into
                        // EngineCoreRequest and is thereafter bounded by the
                        // Engine's retained-sequence/request capacity.
                        let _permit = permit;
                        let request = request;
                        let (samples, sample_rate) =
                            executor::decode_request_audio_with_rate(&request)?;
                        let long_form = executor::qwen3_asr_requires_long_form(
                            &samples,
                            sample_rate,
                            model_for_shape.max_audio_seconds_hint(),
                        );
                        let input_tokens = (!long_form)
                            .then(|| {
                                model_for_shape.incremental_prompt_token_count(
                                    &samples,
                                    sample_rate,
                                    request.asr_language_for_execution(),
                                    request.asr_prompt_for_execution(),
                                )
                            })
                            .transpose()?;
                        let prepared_encoder = if long_form {
                            None
                        } else {
                            let geometry = model_for_shape
                                .audio_preparation_row_geometry(samples.len(), sample_rate)?;
                            let rows = [Qwen3AsrAudioBatchRow {
                                audio: &samples,
                                sample_rate,
                            }];
                            let mut artifacts =
                                model_for_shape.prepare_qwen3_audio_tower_batch(&rows)?;
                            let artifact = artifacts.pop().ok_or_else(|| {
                                Error::InferenceError(
                                    "Direct Qwen3 ASR width-one encoder returned no artifact"
                                        .to_string(),
                                )
                            })?;
                            if !artifacts.is_empty()
                                || artifact.audio_tokens()? != geometry.audio_tokens
                                || artifact.resident_tensor_bytes()?
                                    != geometry.retained_artifact_bytes
                            {
                                return Err(Error::InferenceError(
                                    "Direct Qwen3 ASR encoder artifact disagrees with admitted geometry"
                                        .to_string(),
                                ));
                            }
                            Some(Arc::new(artifact))
                        };
                        Self::finalize_direct_qwen_asr_preparation(
                            request,
                            variant,
                            samples,
                            sample_rate,
                            context_limit,
                            input_tokens,
                            prepared_encoder,
                        )
                    });
                    let prepared = match deadline {
                        Some(deadline) => tokio::time::timeout_at(deadline.into(), worker)
                            .await
                            .map_err(|_| Error::Timeout(request_id.clone()))?,
                        None => worker.await,
                    }
                    .map_err(|error| {
                        Error::InferenceError(format!(
                            "ASR request {request_id} sequence-shape worker failed: {error}"
                        ))
                    })??;
                    request = prepared;
                }
                request.install_asr_execution_model(variant, model)?;
            }
            TaskType::TTS if variant.family() == crate::catalog::ModelFamily::Qwen3Tts => {
                let model = registry.get_qwen_tts_lease(variant).await.ok_or_else(|| {
                    Error::ModelNotFound(format!("Qwen TTS model {variant} is not loaded"))
                })?;
                let reference = if request.has_tts_reference_for_execution() {
                    let encoded = request
                        .tts_reference_audio_for_execution()
                        .ok_or_else(|| {
                            Error::InvalidInput(
                                "reference_audio and reference_text must both be provided"
                                    .to_string(),
                            )
                        })?
                        .to_string();
                    let text = request
                        .tts_reference_text_for_execution()
                        .filter(|text| !text.trim().is_empty())
                        .ok_or_else(|| {
                            Error::InvalidInput("reference_text cannot be empty".to_string())
                        })?
                        .to_string();
                    let request_id = request.id.clone();
                    let (audio_samples, sample_rate) = tokio::task::spawn_blocking(move || {
                        crate::runtime::audio_io::decode_reference_audio_base64(&encoded)
                    })
                    .await
                    .map_err(|error| {
                        Error::InferenceError(format!(
                            "Qwen TTS request {request_id} reference worker failed: {error}"
                        ))
                    })??;
                    Some(Arc::new(
                        crate::models::architectures::qwen3::tts::SpeakerReference {
                            audio_samples,
                            text,
                            sample_rate,
                        },
                    ))
                } else {
                    None
                };
                let context_limit = registry
                    .effective_context(variant)
                    .unwrap_or(self.config.max_seq_len);
                request.install_qwen_tts_execution_model(
                    variant,
                    model,
                    reference,
                    context_limit,
                )?;
            }
            TaskType::ASR | TaskType::TTS | TaskType::Chat | TaskType::SpeechToSpeech => {}
        }
        Ok(request)
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_direct_qwen_asr_preparation(
        mut request: EngineCoreRequest,
        variant: ModelVariant,
        samples: Vec<f32>,
        sample_rate: u32,
        context_limit: usize,
        input_tokens: Option<usize>,
        encoder_artifact: Option<Arc<Qwen3AsrPreparedAudio>>,
    ) -> Result<EngineCoreRequest> {
        request.install_prepared_asr_audio(variant, samples, sample_rate)?;
        match (input_tokens, encoder_artifact) {
            (None, None) => request.install_prepared_asr_long_form_atomic()?,
            (Some(input_tokens), Some(artifact)) => {
                request.install_prepared_sequence_input_tokens(input_tokens, context_limit)?;
                request.install_prepared_asr_encoder_artifact(variant, artifact)?;
            }
            _ => {
                return Err(Error::InferenceError(
                    "Direct Qwen3 ASR route and encoder artifact disagree".to_string(),
                ));
            }
        }
        Ok(request)
    }

    async fn add_request_with_completion(
        &self,
        request: EngineCoreRequest,
        completion_registration: Option<u64>,
    ) -> Result<(RequestId, SessionKey)> {
        let processed = if request.task_type == TaskType::Chat {
            // Apply the raw-input guard before a blocking tokenizer renders tool
            // JSON. Runtime-prepared chat skips model preparation below.
            request.validate_direct_chat_preparation_input(self.config.max_seq_len)?;
            self.request_processor.process(request)?
        } else {
            // Base64 source scans and canonicalization are O(n). Keep them off
            // async workers and behind one bounded, deadline-aware permit.
            self.prepare_direct_non_chat_request_for_execution(request)
                .await?
        };
        let processed = self.prepare_chat_request_for_execution(processed).await?;
        let mut processed = self.retain_incremental_model_identity(processed).await?;
        let request_id = processed.id.clone();
        let model_variant = processed.model_variant;
        let cancellation = Arc::new(std::sync::atomic::AtomicBool::new(false));
        processed.set_cancellation_signal(cancellation.clone());

        // Add to engine core. The core write lock also makes binding a pending
        // completion registration atomic with respect to every engine step.
        let mut core = self.core.write().await;
        core.add_request(processed)?;
        let session = core.get_session_key(&request_id).ok_or_else(|| {
            crate::error::Error::InferenceError(format!(
                "request {request_id} is missing its scheduler session"
            ))
        })?;
        self.request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(
                request_id.clone(),
                RequestControl {
                    session_epoch: session.epoch,
                    cancellation,
                    model_variant,
                },
            );
        if let Some(registration_id) = completion_registration {
            self.bind_completion_mailbox(&request_id, registration_id, session.epoch);
        }
        self.wake_notify.notify_one();

        debug!("Added request {} to engine", request_id);
        Ok((request_id, session))
    }

    /// Add a request to the engine for processing.
    ///
    /// The request will be validated, preprocessed, and added to the scheduler's
    /// waiting queue. Returns a request ID that can be used to track the request.
    pub async fn add_request(&self, request: EngineCoreRequest) -> Result<RequestId> {
        self.add_request_with_completion(request, None)
            .await
            .map(|(request_id, _)| request_id)
    }

    /// Add a request and return the scheduler incarnation established by that
    /// same atomic admission. Runtime dispatchers use this to bind their
    /// completion waiter before a fast terminal step can erase active state.
    pub(crate) async fn add_request_with_session(
        &self,
        request: EngineCoreRequest,
    ) -> Result<SessionKey> {
        self.add_request_with_completion(request, None)
            .await
            .map(|(_, session)| session)
    }

    /// Generate audio synchronously (blocking until complete).
    ///
    /// This is a convenience method that adds a request and waits for completion.
    pub async fn generate(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        let request_id = request.id.clone();
        let (registration, mut completion) =
            self.register_completion_mailbox(request_id.clone())?;
        let _ = self
            .add_request_with_completion(request, Some(registration.registration_id))
            .await?;
        let mut idle_backoff_ms = 1u64;

        // Run steps until this request completes
        loop {
            let outputs = tokio::select! {
                biased;
                completion = &mut completion => {
                    let output = completion.map_err(|_| {
                        crate::error::Error::InferenceError(format!(
                            "Completion mailbox for {request_id} closed before delivery"
                        ))
                    })?;
                    return Self::resolve_generation_output(&request_id, output);
                }
                outputs = self.step() => outputs?,
            };
            let step_was_idle = outputs.is_empty();

            // Check if request is still in the system
            let core = self.core.read().await;
            if !core.has_request(&request_id) && !core.has_pending_terminal_output(&request_id) {
                drop(core);
                return match completion.try_recv() {
                    Ok(output) => Self::resolve_generation_output(&request_id, output),
                    Err(oneshot::error::TryRecvError::Closed) => {
                        Err(crate::error::Error::InferenceError(format!(
                            "Completion mailbox for {request_id} closed before delivery"
                        )))
                    }
                    Err(oneshot::error::TryRecvError::Empty) => {
                        Err(crate::error::Error::InferenceError(format!(
                            "Request {request_id} was removed unexpectedly"
                        )))
                    }
                };
            }
            drop(core);

            if step_was_idle {
                tokio::select! {
                    biased;
                    completion = &mut completion => {
                        let output = completion.map_err(|_| {
                            crate::error::Error::InferenceError(format!(
                                "Completion mailbox for {request_id} closed before delivery"
                            ))
                        })?;
                        return Self::resolve_generation_output(&request_id, output);
                    }
                    _ = self.wake_notify.notified() => {},
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(idle_backoff_ms)) => {},
                }
                idle_backoff_ms = idle_backoff_ms.saturating_mul(2).min(50);
            } else {
                idle_backoff_ms = 1;
            }
        }
    }

    /// Generate audio with streaming output.
    ///
    /// Returns a channel receiver that will receive audio chunks as they're generated.
    pub async fn generate_streaming(
        &self,
        request: EngineCoreRequest,
    ) -> Result<(RequestId, mpsc::Receiver<StreamingOutput>)> {
        self.generate_streaming_with_session(request)
            .await
            .map(|(session, receiver)| (session.request_id, receiver))
    }

    /// Start streaming and return the exact scheduler incarnation admitted for
    /// the request. This keeps outer completion routing session-safe even when
    /// a request finishes immediately on another worker thread.
    pub(crate) async fn generate_streaming_with_session(
        &self,
        request: EngineCoreRequest,
    ) -> Result<(SessionKey, mpsc::Receiver<StreamingOutput>)> {
        let capacity = Self::streaming_queue_capacity(&request);
        let (tx, rx) = mpsc::channel(capacity);

        // Add request with streaming callback
        let mut streaming_request = request;
        streaming_request.streaming = true;
        streaming_request.streaming_tx = Some(tx);

        let session = self.add_request_with_session(streaming_request).await?;

        Ok((session, rx))
    }

    /// Execute one step of the inference loop.
    ///
    /// This is the core loop that:
    /// 1. Schedules requests (decides what to process this step)
    /// 2. Runs forward pass on scheduled requests
    /// 3. Processes outputs (sampling, stop conditions)
    ///
    /// Returns outputs for any completed or streaming requests.
    pub async fn step(&self) -> Result<Vec<EngineOutput>> {
        self.step_with_terminal_ack(false).await
    }

    /// Execute a step while retaining unregistered terminal fences for an
    /// outer dispatcher. `Engine::generate` mailboxes are still delivered and
    /// acknowledged here.
    pub(crate) async fn step_for_dispatch(&self) -> Result<Vec<EngineOutput>> {
        self.step_with_terminal_ack(true).await
    }

    async fn step_with_terminal_ack(
        &self,
        defer_unregistered_terminal_ack: bool,
    ) -> Result<Vec<EngineOutput>> {
        let context = OwnedStepContext {
            core: self.core.clone(),
            step_gate: self.step_gate.clone(),
            metrics: self.metrics.clone(),
            request_controls: self.request_controls.clone(),
            completion_mailboxes: self.completion_mailboxes.clone(),
        };
        match tokio::spawn(async move { context.run(defer_unregistered_terminal_ack).await }).await
        {
            Ok(result) => result,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(error) => Err(Error::InferenceError(format!(
                "owned engine step task was cancelled: {error}"
            ))),
        }
    }

    /// Confirm delivery after an outer dispatcher has attempted to route a
    /// terminal output to its exact request consumer.
    pub(crate) async fn acknowledge_dispatched_terminal(&self, output: &EngineOutput) -> bool {
        if !output.is_finished {
            return false;
        }
        let session = SessionKey::new(output.request_id.clone(), output.sequence_id);
        self.core
            .write()
            .await
            .acknowledge_terminal_output(&session)
    }

    /// Run the engine continuously, processing requests as they arrive.
    ///
    /// This should be called in a separate task. It will run until `stop()` is called.
    pub async fn run(&self) -> Result<()> {
        use std::sync::atomic::Ordering;

        self.running.store(true, Ordering::SeqCst);
        info!("Engine started");
        let mut idle_backoff_ms = 1u64;

        while self.running.load(Ordering::SeqCst) {
            // Check if there are requests to process
            let has_work = {
                let core = self.core.read().await;
                core.has_pending_work()
            };

            if has_work {
                match self.step().await {
                    Ok(outputs) if outputs.is_empty() => {
                        tokio::select! {
                            _ = self.wake_notify.notified() => {},
                            _ = tokio::time::sleep(tokio::time::Duration::from_millis(idle_backoff_ms)) => {},
                        }
                        idle_backoff_ms = idle_backoff_ms.saturating_mul(2).min(50);
                    }
                    Ok(_) => idle_backoff_ms = 1,
                    Err(e) => {
                        warn!("Engine step error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(idle_backoff_ms))
                            .await;
                        idle_backoff_ms = idle_backoff_ms.saturating_mul(2).min(50);
                    }
                }
            } else {
                // Event-driven wait to avoid hot polling on local/edge devices.
                tokio::select! {
                    _ = self.wake_notify.notified() => {},
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {},
                }
                idle_backoff_ms = 1;
            }
        }

        info!("Engine stopped");
        Ok(())
    }

    /// Stop the engine.
    pub fn stop(&self) {
        use std::sync::atomic::Ordering;
        self.running.store(false, Ordering::SeqCst);
        self.wake_notify.notify_waiters();
    }

    /// Check if the engine is running.
    pub fn is_running(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.running.load(Ordering::SeqCst)
    }

    /// Get engine metrics.
    pub async fn metrics(&self) -> EngineMetrics {
        self.metrics.read().await.clone()
    }

    /// Get current configuration.
    pub fn config(&self) -> &EngineCoreConfig {
        &self.config
    }

    /// Abort a specific request.
    pub async fn abort_request(&self, request_id: &RequestId) -> Result<bool> {
        if let Some(control) = self
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(request_id)
        {
            control
                .cancellation
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let _step = self.step_gate.lock().await;
        let mut core = self.core.write().await;
        let aborted = core.abort_request(request_id).await;
        drop(core);
        if aborted {
            self.request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .remove(request_id);
            self.wake_notify.notify_one();
        }
        Ok(aborted)
    }

    /// Read the session-fenced identity for the active request ID.
    pub async fn request_session_key(&self, request_id: &RequestId) -> Option<SessionKey> {
        self.core.read().await.get_session_key(request_id)
    }

    #[cfg(test)]
    pub(crate) async fn hold_core_step_lock_for_test(
        &self,
        entered: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    ) {
        let _core = self.core.write().await;
        let _ = entered.send(());
        let _ = release.await;
    }

    #[cfg(test)]
    pub(crate) async fn set_request_hard_deadline_for_test(
        &self,
        request_id: &RequestId,
        deadline: Instant,
    ) -> bool {
        self.core
            .write()
            .await
            .set_hard_deadline_for_test(request_id, deadline)
    }

    /// Abort only the request incarnation named by `session`.
    pub async fn abort_request_session(&self, session: &SessionKey) -> Result<bool> {
        let signaled = {
            let controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            controls
                .get(&session.request_id)
                .filter(|control| control.session_epoch == session.epoch)
                .map(|control| {
                    control
                        .cancellation
                        .store(true, std::sync::atomic::Ordering::Release);
                    true
                })
                .unwrap_or(false)
        };
        let _step = self.step_gate.lock().await;
        let mut core = self.core.write().await;
        let aborted = core.abort_request_session(session).await;
        drop(core);
        let accepted = signaled || aborted;
        if accepted {
            let mut controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if controls
                .get(&session.request_id)
                .is_some_and(|control| control.session_epoch == session.epoch)
            {
                controls.remove(&session.request_id);
            }
            drop(controls);
            self.wake_notify.notify_one();
        }
        Ok(accepted)
    }

    /// Abort all requests currently routed to a specific model variant.
    pub async fn abort_requests_for_variant(&self, variant: ModelVariant) -> Vec<RequestId> {
        {
            let controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for control in controls.values() {
                if control.model_variant == Some(variant) {
                    control
                        .cancellation
                        .store(true, std::sync::atomic::Ordering::Release);
                }
            }
        }
        let _step = self.step_gate.lock().await;
        let mut core = self.core.write().await;
        let aborted = core.abort_requests_for_variant(variant).await;
        self.request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retain(|_, control| control.model_variant != Some(variant));
        if !aborted.is_empty() {
            self.wake_notify.notify_one();
        }
        aborted
    }

    /// Purge reusable executor cache state owned by one model variant.
    pub async fn purge_model_cache(&self, variant: ModelVariant) -> CacheReleaseReport {
        let _step = self.step_gate.lock().await;
        self.core.write().await.purge_model_cache(variant).await
    }

    pub(crate) async fn synchronize_worker_device(&self) -> Result<()> {
        let _step = self.step_gate.lock().await;
        self.core.read().await.synchronize_worker_device()
    }

    /// Admit and allocate managed physical state before the model generation
    /// becomes Ready.
    pub(crate) async fn load_managed_model_cache(
        &self,
        model_instance: ModelInstanceId,
        capability: &crate::kv::InferenceStateCapability,
        logical_context_tokens: Option<usize>,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        self.load_managed_model_cache_with_capacity_policy(
            model_instance,
            capability,
            logical_context_tokens,
            None,
            false,
        )
        .await
    }

    pub(crate) async fn load_managed_model_cache_with_capacity_policy(
        &self,
        model_instance: ModelInstanceId,
        capability: &crate::kv::InferenceStateCapability,
        logical_context_tokens: Option<usize>,
        staged_transaction_rows: Option<u32>,
        fit_cuda_resident_context: bool,
    ) -> Result<Option<Arc<ManagedKvModelRuntime>>> {
        let _step = self.step_gate.lock().await;
        self.core
            .write()
            .await
            .load_managed_model_cache_with_capacity_policy(
                model_instance,
                capability,
                logical_context_tokens,
                staged_transaction_rows,
                fit_cuda_resident_context,
            )
    }

    pub(crate) async fn load_managed_model_state(
        &self,
        model_instance: ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        let _step = self.step_gate.lock().await;
        self.core.write().await.load_managed_model_state(
            model_instance,
            retained_state,
            logical_context_tokens,
        )
    }

    pub(crate) async fn load_managed_model_state_with_portable_copies(
        &self,
        model_instance: ModelInstanceId,
        retained_state: &crate::kv::v2::InferenceStateContract,
        logical_context_tokens: Option<usize>,
        portable_state_copies: u32,
    ) -> Result<Arc<ManagedKvModelRuntime>> {
        let _step = self.step_gate.lock().await;
        self.core
            .write()
            .await
            .load_managed_model_state_with_portable_copies(
                model_instance,
                retained_state,
                logical_context_tokens,
                portable_state_copies,
            )
    }

    pub(crate) async fn load_retained_tensor_state(
        &self,
        model_instance: ModelInstanceId,
        contract: &crate::kv::v2::InferenceStateContract,
        sequence_capacity: u32,
    ) -> Result<Arc<RetainedTensorStateRuntimeV2>> {
        let _step = self.step_gate.lock().await;
        self.core.write().await.load_retained_tensor_state(
            model_instance,
            contract,
            sequence_capacity,
        )
    }

    pub(crate) async fn resolve_and_load_invocation_workspace(
        &self,
        model_instance: ModelInstanceId,
        adapter_instance: AdapterInstanceId,
        stage_graph: [u8; 32],
        stage: StageId,
        contract: &crate::kv::v2::InferenceStateContract,
        domain: &crate::kv::v2::InvocationWorkspaceDomain,
        slot_count: u32,
    ) -> Result<Arc<dyn crate::kv::v2::InvocationWorkspaceBackingV2>> {
        let _step = self.step_gate.lock().await;
        self.core
            .write()
            .await
            .resolve_and_load_invocation_workspace(
                model_instance,
                adapter_instance,
                stage_graph,
                stage,
                contract,
                domain,
                slot_count,
            )
    }

    /// Retire the managed KV arenas for one exact loaded-model generation.
    pub async fn unload_managed_model_cache(
        &self,
        model_instance: ModelInstanceId,
    ) -> Result<bool> {
        let _step = self.step_gate.lock().await;
        self.core
            .write()
            .await
            .unload_managed_model_cache(model_instance)
    }

    /// Abort every request currently tracked by the engine.
    pub async fn abort_all_requests(&self) -> Vec<RequestId> {
        {
            let controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for control in controls.values() {
                control
                    .cancellation
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        let _step = self.step_gate.lock().await;
        let mut core = self.core.write().await;
        let aborted = core.abort_all_requests().await;
        self.request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        if !aborted.is_empty() {
            self.wake_notify.notify_one();
        }
        aborted
    }

    /// Check if a request is still tracked by the engine core.
    pub async fn has_request(&self, request_id: &RequestId) -> bool {
        let core = self.core.read().await;
        core.has_request(request_id)
    }

    /// Get model variants currently referenced by active engine requests.
    pub async fn active_model_variants(&self) -> HashSet<ModelVariant> {
        let core = self.core.read().await;
        core.active_model_variants()
    }

    /// Get the number of pending requests.
    pub async fn pending_requests(&self) -> usize {
        let core = self.core.read().await;
        core.pending_request_count()
    }

    /// Get the number of running requests.
    pub async fn running_requests(&self) -> usize {
        let core = self.core.read().await;
        core.running_request_count()
    }

    /// Snapshot exact physical arena backing, page ownership, and counters.
    pub async fn kv_cache_snapshot(&self) -> ManagedKvRuntimeSnapshot {
        self.core.read().await.managed_kv_runtime_snapshot()
    }

    /// Check if scheduler currently has runnable or queued work.
    pub async fn has_pending_work(&self) -> bool {
        let core = self.core.read().await;
        core.has_pending_work()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::scheduler::ScheduledRequest;
    use super::*;
    use crate::backends::{BackendKind, DeviceProfile};
    use crate::error::Error;
    use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind, ChatMessage, ChatRole};

    #[test]
    fn non_streaming_qwen_asr_shape_replaces_the_placeholder_before_admission() {
        let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
        assert!(!request.streaming);
        assert_eq!(
            request.num_prompt_tokens(),
            1,
            "fixture starts with a placeholder"
        );

        request
            .install_prepared_sequence_input_tokens(37, 64)
            .expect("exact multimodal shape");
        assert_eq!(request.num_prompt_tokens(), 37);
        assert!(request.uses_asr_retained_sequence());
        assert!(request.install_prepared_asr_long_form_atomic().is_err());
    }

    #[test]
    fn direct_qwen_asr_normal_route_installs_the_encoder_artifact_before_core_admission() {
        let variant = ModelVariant::Qwen3Asr06BGguf;
        let request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
        let artifact =
            Arc::new(Qwen3AsrPreparedAudio::for_test(7, 16).expect("test encoder artifact"));

        let prepared = Engine::finalize_direct_qwen_asr_preparation(
            request,
            variant,
            vec![0.0; 16_000],
            16_000,
            4_096,
            Some(32),
            Some(artifact.clone()),
        )
        .expect("normal direct Qwen3 ASR preparation");

        assert!(prepared.uses_asr_retained_sequence());
        assert_eq!(prepared.num_prompt_tokens(), 32);
        assert_eq!(
            prepared
                .prepared_asr_encoder_artifact_retained_bytes()
                .unwrap(),
            7 * 16 * std::mem::size_of::<f32>() as u64
        );
        assert!(Arc::ptr_eq(
            &prepared
                .prepared_asr_encoder_artifact_for_executor()
                .unwrap()
                .expect("installed artifact"),
            &artifact,
        ));
    }

    #[test]
    fn direct_qwen_asr_long_form_route_retains_no_encoder_artifact() {
        let variant = ModelVariant::Qwen3Asr06BGguf;
        let request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);

        let prepared = Engine::finalize_direct_qwen_asr_preparation(
            request,
            variant,
            vec![0.0; 16_000],
            16_000,
            4_096,
            None,
            None,
        )
        .expect("long-form direct Qwen3 ASR preparation");

        assert!(prepared.uses_asr_long_form_atomic());
        assert!(prepared
            .prepared_asr_encoder_artifact_for_executor()
            .unwrap()
            .is_none());
        assert_eq!(
            prepared
                .prepared_asr_encoder_artifact_retained_bytes()
                .unwrap(),
            0
        );
    }

    struct EndlessSequenceExecutor;

    impl EndlessSequenceExecutor {
        fn outputs(scheduled: &[ScheduledRequest]) -> Vec<ExecutorStepResult> {
            scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::new(
                        entry,
                        ExecutorOutput {
                            request_id: entry.request_id.clone(),
                            audio: None,
                            text: None,
                            input_transcription: None,
                            tokens_processed: usize::from(entry.is_prefill) * entry.num_tokens,
                            tokens_generated: usize::from(!entry.is_prefill),
                            finished: false,
                            phase_timing_override: None,
                            asr_diagnostics: None,
                            error: None,
                        },
                    )
                })
                .collect()
        }
    }

    impl ModelExecutor for EndlessSequenceExecutor {
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
            Ok(Self::outputs(scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Self::outputs(scheduled))
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

        fn cleanup_request(&self, _request_id: &str) -> executor::CacheReleaseReport {
            executor::CacheReleaseReport::confirmed(1)
        }
    }

    struct ImmediateTerminalExecutor {
        max_batch_width: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ImmediateTerminalExecutor {
        fn new(max_batch_width: Arc<std::sync::atomic::AtomicUsize>) -> Self {
            Self { max_batch_width }
        }

        fn outputs(&self, scheduled: &[ScheduledRequest]) -> Vec<ExecutorStepResult> {
            use std::sync::atomic::Ordering;

            self.max_batch_width
                .fetch_max(scheduled.len(), Ordering::Relaxed);
            let dispatch = if scheduled.len() > 1 {
                BatchDispatch::new(BatchDispatchKind::TensorStatic, scheduled.len())
            } else {
                BatchDispatch::serial()
            };
            scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::new(
                        entry,
                        ExecutorOutput {
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
                        },
                    )
                    .with_dispatch(dispatch)
                })
                .collect()
        }
    }

    impl ModelExecutor for ImmediateTerminalExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            profile.prefill_batch = NativeBatchMode::Static;
            profile.decode_batch = NativeBatchMode::Static;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = 8;
            profile.resolved_from_loaded_model = true;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.outputs(scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.outputs(scheduled))
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

        fn cleanup_request(&self, _request_id: &str) -> executor::CacheReleaseReport {
            executor::CacheReleaseReport::confirmed(1)
        }
    }

    struct BlockingForwardExecutor {
        entered: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    struct IncrementalBlockingExecutor {
        emitted: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        variant: ModelVariant,
    }

    struct ReverseTerminalExecutor {
        slow_request: RequestId,
        slow_entered: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        cleanup: Arc<std::sync::Mutex<Vec<RequestId>>>,
        variant: ModelVariant,
    }

    impl ModelExecutor for ReverseTerminalExecutor {
        fn execution_profile(&self, _request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                Some(self.variant),
                ExecutionMode::Atomic,
            );
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.physical_launch_policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
            profile.max_batch_size = 2;
            profile.resolved_from_loaded_model = true;
            Some(profile)
        }

        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            execution.validate().expect("test physical batch");
            let request = execution.requests[0];
            if request.id == self.slow_request {
                if let Some(entered) = self
                    .slow_entered
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take()
                {
                    let _ = entered.send(());
                }
                let (released, wake) = self.release.as_ref();
                let mut released = released.lock().unwrap_or_else(|poison| poison.into_inner());
                while !*released {
                    released = wake
                        .wait(released)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
            } else {
                request
                    .stream_staging_buffer()
                    .push_with_policy(
                        StreamingOutput {
                            request_id: request.id.clone(),
                            sequence: 0,
                            samples: Vec::new(),
                            sample_rate: 0,
                            is_final: false,
                            text: Some("fast-progress".to_string()),
                            stats: None,
                            asr_progress: None,
                        },
                        request.stream_policy,
                    )
                    .expect("publish fast progress");
            }
            let dispatch = execution.expected_dispatch();
            Ok(execution
                .scheduled
                .iter()
                .map(|scheduled| {
                    ExecutorStepResult::new(
                        scheduled,
                        ExecutorOutput::terminal(scheduled.request_id.clone()),
                    )
                    .with_dispatch(dispatch)
                })
                .collect())
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            unreachable!("physical boundary must own dispatch")
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            unreachable!("physical boundary must own dispatch")
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

        fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
            self.cleanup
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push(session.request_id.clone());
            CacheReleaseReport::confirmed(1)
        }
    }

    impl IncrementalBlockingExecutor {
        fn execute(
            &self,
            requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            let request = requests
                .first()
                .copied()
                .ok_or_else(|| Error::InferenceError("missing test request".to_string()))?;
            let scheduled = scheduled
                .first()
                .ok_or_else(|| Error::InferenceError("missing test schedule".to_string()))?;
            let staging = request.stream_staging_buffer();
            if !staging.has_incremental_binding() {
                return Err(Error::InferenceError(
                    "test request was not bound for incremental publication".to_string(),
                ));
            }
            staging.push_with_policy(
                StreamingOutput {
                    request_id: request.id.clone(),
                    sequence: 0,
                    samples: Vec::new(),
                    sample_rate: 0,
                    is_final: false,
                    text: Some("first delta".to_string()),
                    stats: None,
                    asr_progress: None,
                },
                request.stream_policy,
            )?;
            if let Some(emitted) = self
                .emitted
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .take()
            {
                let _ = emitted.send(());
            }

            tokio::task::block_in_place(|| {
                let (released, wake) = self.release.as_ref();
                let mut released = released.lock().unwrap_or_else(|poison| poison.into_inner());
                while !*released {
                    released = wake
                        .wait(released)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
            });

            staging.push_with_policy(
                StreamingOutput {
                    request_id: request.id.clone(),
                    sequence: 1,
                    samples: Vec::new(),
                    sample_rate: 0,
                    is_final: true,
                    text: None,
                    stats: None,
                    asr_progress: None,
                },
                request.stream_policy,
            )?;
            let mut result = ExecutorStepResult::new(
                scheduled,
                ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: None,
                    text: Some("first delta".to_string()),
                    input_transcription: None,
                    tokens_processed: scheduled.num_tokens.max(1),
                    tokens_generated: 1,
                    finished: true,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                },
            );
            result.staged_stream_outputs = request.take_staged_stream_outputs()?;
            Ok(vec![result])
        }
    }

    impl ModelExecutor for IncrementalBlockingExecutor {
        fn execution_profile(&self, _request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            Some(ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                Some(self.variant),
                ExecutionMode::Atomic,
            ))
        }

        fn execute_prefill(
            &self,
            requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            self.execute(requests, scheduled)
        }

        fn execute_decode(
            &self,
            requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            self.execute(requests, scheduled)
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

        fn cleanup_session(&self, _session: &SessionKey) -> CacheReleaseReport {
            CacheReleaseReport::confirmed(1)
        }
    }

    impl BlockingForwardExecutor {
        fn execute(&self, scheduled: &[ScheduledRequest]) -> Vec<ExecutorStepResult> {
            if let Some(entered) = self
                .entered
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .take()
            {
                let _ = entered.send(());
                let (released, wake) = self.release.as_ref();
                let mut released = released.lock().unwrap_or_else(|poison| poison.into_inner());
                while !*released {
                    released = wake
                        .wait(released)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
            }

            scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::new(
                        entry,
                        ExecutorOutput {
                            request_id: entry.request_id.clone(),
                            audio: None,
                            text: Some("done".to_string()),
                            input_transcription: None,
                            tokens_processed: entry.num_tokens.max(1),
                            tokens_generated: 1,
                            finished: true,
                            phase_timing_override: None,
                            asr_diagnostics: None,
                            error: None,
                        },
                    )
                })
                .collect()
        }
    }

    impl ModelExecutor for BlockingForwardExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.execute(scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.execute(scheduled))
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

        fn cleanup_request(&self, _request_id: &str) -> executor::CacheReleaseReport {
            executor::CacheReleaseReport::confirmed(1)
        }
    }

    fn immediate_terminal_request(id: &str) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::tts(format!("terminal output for {id}"));
        request.id = id.to_string();
        request.prompt_tokens = vec![1];
        request
    }

    fn engine_with_test_executor(executor: Box<dyn ModelExecutor>) -> Engine {
        let config = EngineCoreConfig::default();
        let core = EngineCore::new_with_unified_executor(
            config.clone(),
            executor::UnifiedExecutor::new_for_test(executor),
        )
        .unwrap();
        Engine {
            core: Arc::new(RwLock::new(core)),
            step_gate: Arc::new(Mutex::new(())),
            request_processor: RequestProcessor::new(config.clone()),
            output_processor: OutputProcessor::new(config.sample_rate),
            direct_request_preparation_permits: Arc::new(Semaphore::new(
                config.max_batch_size.max(1),
            )),
            config,
            model_registry: None,
            running: std::sync::atomic::AtomicBool::new(false),
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
            wake_notify: Arc::new(Notify::new()),
            request_controls: Arc::new(std::sync::Mutex::new(HashMap::new())),
            completion_mailboxes: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_completion_registration: std::sync::atomic::AtomicU64::new(1),
        }
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineCoreConfig::default();
        let engine = Engine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn physical_execution_telemetry_mode_preserves_the_bounded_rollout_axis() {
        for (configured, expected) in [
            (
                crate::config::PhysicalExecutionMode::Serial,
                EnginePhysicalExecutionMode::Serial,
            ),
            (
                crate::config::PhysicalExecutionMode::Shadow,
                EnginePhysicalExecutionMode::Shadow,
            ),
            (
                crate::config::PhysicalExecutionMode::Concurrent,
                EnginePhysicalExecutionMode::Concurrent,
            ),
        ] {
            assert_eq!(
                Engine::physical_execution_telemetry_mode(configured),
                expected
            );
        }

        let mut shadow = EngineCoreConfig::default();
        shadow.physical_execution_mode = crate::config::PhysicalExecutionMode::Shadow;
        shadow.max_physical_in_flight = crate::config::PhysicalInFlightLimit::new(4).unwrap();
        assert_eq!(
            Engine::physical_execution_telemetry_policy(&shadow, None),
            (EnginePhysicalExecutionMode::Shadow, 1)
        );

        shadow.physical_execution_mode = crate::config::PhysicalExecutionMode::Concurrent;
        assert_eq!(
            Engine::physical_execution_telemetry_policy(&shadow, None),
            (EnginePhysicalExecutionMode::Concurrent, 4)
        );
        assert_eq!(
            Engine::physical_execution_telemetry_policy(&shadow, Some(1)),
            (EnginePhysicalExecutionMode::Concurrent, 1),
            "effective telemetry must report the actual shared admission cap"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_forward_does_not_hold_the_engine_state_lock() {
        let (entered_tx, entered_rx) = oneshot::channel();
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let engine = Arc::new(engine_with_test_executor(Box::new(
            BlockingForwardExecutor {
                entered: std::sync::Mutex::new(Some(entered_tx)),
                release: release.clone(),
            },
        )));
        let request_id = "forward-lock-release".to_string();
        engine
            .core
            .write()
            .await
            .add_request(immediate_terminal_request(&request_id))
            .unwrap();

        let stepping_engine = engine.clone();
        let step = tokio::spawn(async move { stepping_engine.step().await });
        tokio::time::timeout(Duration::from_secs(1), entered_rx)
            .await
            .expect("executor did not enter the model forward")
            .expect("executor entry signal was dropped");

        let visible =
            tokio::time::timeout(Duration::from_millis(100), engine.has_request(&request_id))
                .await
                .expect("model forward retained the engine state lock");
        assert!(visible);

        let (released, wake) = release.as_ref();
        *released.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
        let outputs = tokio::time::timeout(Duration::from_secs(1), step)
            .await
            .expect("engine step did not complete")
            .expect("engine step task panicked")
            .expect("engine step failed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, request_id);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn incremental_atomic_delta_is_delivered_before_model_completion() {
        let (emitted_tx, emitted_rx) = oneshot::channel();
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let engine = Arc::new(engine_with_test_executor(Box::new(
            IncrementalBlockingExecutor {
                emitted: std::sync::Mutex::new(Some(emitted_tx)),
                release: release.clone(),
                variant,
            },
        )));
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.prefill = PrefillMode::None;
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(41),
            "test.atomic.incremental",
            &profile,
            NativeBatchMode::None,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.output_visibility = OutputVisibility::IncrementalCommitted;
        let binding = ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(11),
            model_instance_id: ModelInstanceId::new(12),
            adapter_instance_id: AdapterInstanceId::new(13),
            adapter_abi_revision: AdapterAbiRevision::new(8),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage]),
        };
        let mut request =
            EngineCoreRequest::tts("stream while running").with_model_variant(variant);
        request.id = "incremental-before-completion".to_string();
        request.prompt_tokens = vec![1];
        request.bind_execution_adapter(binding).unwrap();
        request.streaming = true;
        let (stream_tx, mut stream_rx) = mpsc::channel(8);
        request.streaming_tx = Some(stream_tx);
        engine.core.write().await.add_request(request).unwrap();

        let stepping_engine = engine.clone();
        let step = tokio::spawn(async move { stepping_engine.step().await });
        let emitted = tokio::time::timeout(Duration::from_secs(1), emitted_rx).await;
        let first = tokio::time::timeout(Duration::from_secs(1), stream_rx.recv()).await;
        let completed_early = step.is_finished();

        // Always release the blocking fake before asserting so a failing
        // temporal check cannot strand a Tokio worker during test teardown.
        let (released, wake) = release.as_ref();
        *released.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();

        let outputs = tokio::time::timeout(Duration::from_secs(1), step)
            .await
            .expect("engine step did not complete")
            .expect("engine step task panicked")
            .expect("engine step failed");
        emitted
            .expect("model did not emit its first delta")
            .expect("model emission signal was dropped");
        let first = first.unwrap_or_else(|error| {
            panic!("delta remained buffered until model completion: {error}; outputs={outputs:?}")
        });
        let first = first.expect("stream closed before its first delta");
        assert_eq!(first.sequence, 0);
        assert_eq!(first.text.as_deref(), Some("first delta"));
        assert!(!first.is_final);
        assert!(!completed_early, "model completed before it was released");
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finished);

        let final_output = tokio::time::timeout(Duration::from_secs(1), stream_rx.recv())
            .await
            .expect("final marker was not delivered")
            .expect("stream closed before its final marker");
        assert_eq!(final_output.sequence, 1);
        assert!(final_output.is_final);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_step_future_does_not_abandon_the_owned_transaction() {
        let (entered_tx, entered_rx) = oneshot::channel();
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let engine = Arc::new(engine_with_test_executor(Box::new(
            BlockingForwardExecutor {
                entered: std::sync::Mutex::new(Some(entered_tx)),
                release: release.clone(),
            },
        )));
        let request_id = "cancelled-step-owner".to_string();
        engine
            .core
            .write()
            .await
            .add_request(immediate_terminal_request(&request_id))
            .unwrap();

        let stepping_engine = engine.clone();
        let caller = tokio::spawn(async move { stepping_engine.step().await });
        tokio::time::timeout(Duration::from_secs(1), entered_rx)
            .await
            .expect("executor did not enter the model forward")
            .expect("executor entry signal was dropped");
        caller.abort();

        let (released, wake) = release.as_ref();
        *released.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();

        tokio::time::timeout(Duration::from_secs(1), async {
            while engine.has_request(&request_id).await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("owned step transaction did not finish after its caller was dropped");
        assert!(engine.step().await.unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_text_chat_preparation_runs_off_thread_and_authorizes_exact_tokens() {
        let caller_thread = std::thread::current().id();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "prepare me".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.prompt_tokens = vec![999];

        let prepared = Engine::prepare_direct_chat_request_with(
            request,
            Arc::new(Semaphore::new(1)),
            move |_| {
                assert_ne!(std::thread::current().id(), caller_thread);
                Ok((vec![10, 20, 30], None, None))
            },
        )
        .await
        .expect("direct text preparation should succeed");

        assert_eq!(prepared.prompt_tokens, vec![10, 20, 30]);
        prepared
            .validate_chat_execution_preparation()
            .expect("exact tokens should carry private execution authorization");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_chat_deadline_bounds_running_blocking_preparation() {
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let finished_in_worker = finished.clone();
        let request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "deadline-bound preparation".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B)
        .with_deadline(Some(Instant::now() + Duration::from_millis(30)));

        let started = Instant::now();
        let error = Engine::prepare_direct_chat_request_with(
            request,
            Arc::new(Semaphore::new(1)),
            move |_| {
                std::thread::sleep(Duration::from_millis(200));
                finished_in_worker.store(true, std::sync::atomic::Ordering::Release);
                Ok((vec![10], None, None))
            },
        )
        .await
        .expect_err("blocking preparation must respect the request deadline");

        assert!(matches!(error, Error::Timeout(_)));
        assert!(started.elapsed() < Duration::from_millis(150));
        assert!(!finished.load(std::sync::atomic::Ordering::Acquire));
        tokio::time::timeout(Duration::from_secs(1), async {
            while !finished.load(std::sync::atomic::Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("timed-out blocking worker did not finish");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_chat_deadline_bounds_preparation_queue_wait() {
        let permits = Arc::new(Semaphore::new(1));
        let held = permits.clone().acquire_owned().await.expect("test permit");
        let worker_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_started_in_closure = worker_started.clone();
        let request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "queue-bound preparation".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B)
        .with_deadline(Some(Instant::now() + Duration::from_millis(30)));

        let error = Engine::prepare_direct_chat_request_with(request, permits, move |_| {
            worker_started_in_closure.store(true, std::sync::atomic::Ordering::Release);
            Ok((vec![10], None, None))
        })
        .await
        .expect_err("preparation queue wait must respect the request deadline");

        assert!(matches!(error, Error::Timeout(_)));
        assert!(!worker_started.load(std::sync::atomic::Ordering::Acquire));
        drop(held);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_non_chat_preflight_runs_off_thread_and_obeys_running_deadline() {
        let caller_thread = std::thread::current().id();
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let finished_in_worker = finished.clone();
        let request = EngineCoreRequest::asr_bytes(vec![1])
            .with_deadline(Some(Instant::now() + Duration::from_millis(30)));

        let started = Instant::now();
        let error = Engine::prepare_direct_non_chat_request_with(
            request,
            Arc::new(Semaphore::new(1)),
            move |request| {
                assert_ne!(std::thread::current().id(), caller_thread);
                std::thread::sleep(Duration::from_millis(200));
                finished_in_worker.store(true, std::sync::atomic::Ordering::Release);
                Ok(request)
            },
        )
        .await
        .expect_err("non-chat blocking preflight must respect the request deadline");

        assert!(matches!(error, Error::Timeout(_)));
        assert!(started.elapsed() < Duration::from_millis(150));
        assert!(!finished.load(std::sync::atomic::Ordering::Acquire));
        tokio::time::timeout(Duration::from_secs(1), async {
            while !finished.load(std::sync::atomic::Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("timed-out non-chat blocking worker did not finish");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_non_chat_deadline_bounds_preparation_permit_wait() {
        let permits = Arc::new(Semaphore::new(1));
        let held = permits.clone().acquire_owned().await.expect("test permit");
        let worker_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_started_in_closure = worker_started.clone();
        let request = EngineCoreRequest::asr_bytes(vec![1])
            .with_deadline(Some(Instant::now() + Duration::from_millis(30)));

        let error =
            Engine::prepare_direct_non_chat_request_with(request, permits, move |request| {
                worker_started_in_closure.store(true, std::sync::atomic::Ordering::Release);
                Ok(request)
            })
            .await
            .expect_err("non-chat preparation queue wait must respect the request deadline");

        assert!(matches!(error, Error::Timeout(_)));
        assert!(!worker_started.load(std::sync::atomic::Ordering::Acquire));
        drop(held);
    }

    #[tokio::test]
    async fn direct_engine_rejects_oversized_chat_before_model_lookup_or_tokenization() {
        let engine = engine_with_test_executor(Box::new(ImmediateTerminalExecutor::new(Arc::new(
            std::sync::atomic::AtomicUsize::new(0),
        ))));
        let request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "x".repeat(300_000),
        }])
        .with_model_variant(ModelVariant::Qwen306B);

        let error = engine
            .add_request(request)
            .await
            .expect_err("oversized direct input must fail before model lookup");
        assert!(error.to_string().contains("preparation input exceeds"));
        assert_eq!(engine.pending_requests().await, 0);
    }

    #[tokio::test]
    async fn direct_engine_rejects_oversized_non_chat_metadata_before_processing_or_model_lookup() {
        let engine = engine_with_test_executor(Box::new(ImmediateTerminalExecutor::new(Arc::new(
            std::sync::atomic::AtomicUsize::new(0),
        ))));
        let metadata_limit = engine.config.max_seq_len * 8;
        let request = EngineCoreRequest::tts("x".repeat(metadata_limit + 1))
            .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase);

        let error = engine
            .add_request(request)
            .await
            .expect_err("oversized direct metadata must fail before Qwen model lookup");
        assert!(error.to_string().contains("TTS metadata"));
        assert_eq!(engine.pending_requests().await, 0);
    }

    #[tokio::test]
    async fn direct_engine_media_requires_resource_admitted_runtime() {
        let engine = engine_with_test_executor(Box::new(ImmediateTerminalExecutor::new(Arc::new(
            std::sync::atomic::AtomicUsize::new(0),
        ))));
        let request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "<|image_pad|>".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen359BGguf)
        .with_chat_config(crate::models::shared::chat::ChatRequestConfig {
            media_inputs: vec![ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: "data:image/png;base64,AA==".to_string(),
            }],
            ..Default::default()
        });

        let error = engine
            .add_request(request)
            .await
            .expect_err("direct media chat must not bypass runtime resource admission");
        assert!(error.to_string().contains("use RuntimeService"));
        assert_eq!(engine.pending_requests().await, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_engine_chat_uses_its_configured_registry() {
        let config = EngineCoreConfig::default();
        let mut worker_config = WorkerConfig::from(&config);
        worker_config.model_registry = Some(Arc::new(ModelRegistry::new(
            config.models_dir.clone(),
            DeviceProfile::cpu(),
        )));
        let engine = Engine::new_with_worker(config, worker_config).unwrap();
        let request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "not loaded".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);

        let error = engine
            .add_request(request)
            .await
            .expect_err("the configured but empty registry should report the missing model");
        assert!(error.to_string().contains("Chat model"));
        assert!(error.to_string().contains("not loaded"));
        assert_eq!(engine.pending_requests().await, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_generate_callers_receive_their_own_terminal_output() {
        let engine = Arc::new(engine_with_test_executor(Box::new(
            ImmediateTerminalExecutor::new(Arc::new(std::sync::atomic::AtomicUsize::new(0))),
        )));

        // Hold the core until both generate futures install their mailboxes.
        // Bounded blocking preflight may finish independently, so shared-step
        // batching is covered by the deterministic run-dispatch test below;
        // this regression owns concurrent caller/mailbox routing only.
        let admission_gate = engine.core.write().await;
        let first_engine = engine.clone();
        let first = tokio::spawn(async move {
            first_engine
                .generate(immediate_terminal_request("generate-first"))
                .await
        });
        let second_engine = engine.clone();
        let second = tokio::spawn(async move {
            second_engine
                .generate(immediate_terminal_request("generate-second"))
                .await
        });
        tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                let mailbox_count = engine
                    .completion_mailboxes
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .len();
                if mailbox_count == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both generate callers must register before admission opens");
        drop(admission_gate);

        let first = tokio::time::timeout(tokio::time::Duration::from_secs(1), first)
            .await
            .expect("first generate timed out")
            .expect("first generate task panicked")
            .expect("first generation failed");
        let second = tokio::time::timeout(tokio::time::Duration::from_secs(1), second)
            .await
            .expect("second generate timed out")
            .expect("second generate task panicked")
            .expect("second generation failed");

        assert_eq!(first.request_id, "generate-first");
        assert_eq!(first.text.as_deref(), Some("done-generate-first"));
        assert_eq!(second.request_id, "generate-second");
        assert_eq!(second.text.as_deref(), Some("done-generate-second"));
        assert!(
            engine
                .completion_mailboxes
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .is_empty(),
            "terminal routing must consume both exact-session mailboxes"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fast_terminal_mailbox_and_cache_cleanup_do_not_wait_for_slow_peer() {
        let variant = ModelVariant::Kokoro82M;
        let (slow_entered_tx, slow_entered_rx) = oneshot::channel();
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let cleanup = Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = EngineCoreConfig {
            max_batch_size: 2,
            max_tokens_per_step: 2,
            min_tokens_per_step: 1,
            enable_adaptive_batching: false,
            physical_execution_mode: crate::config::PhysicalExecutionMode::Concurrent,
            max_physical_in_flight: crate::config::PhysicalInFlightLimit::new(2).unwrap(),
            ..Default::default()
        };
        let core = EngineCore::new_with_unified_executor(
            config.clone(),
            executor::UnifiedExecutor::new_for_test(Box::new(ReverseTerminalExecutor {
                slow_request: "slow-mailbox".to_string(),
                slow_entered: std::sync::Mutex::new(Some(slow_entered_tx)),
                release: release.clone(),
                cleanup: cleanup.clone(),
                variant,
            })),
        )
        .unwrap();
        let engine = Arc::new(Engine {
            core: Arc::new(RwLock::new(core)),
            step_gate: Arc::new(Mutex::new(())),
            request_processor: RequestProcessor::new(config.clone()),
            output_processor: OutputProcessor::new(config.sample_rate),
            direct_request_preparation_permits: Arc::new(Semaphore::new(2)),
            config,
            model_registry: None,
            running: std::sync::atomic::AtomicBool::new(false),
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
            wake_notify: Arc::new(Notify::new()),
            request_controls: Arc::new(std::sync::Mutex::new(HashMap::new())),
            completion_mailboxes: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_completion_registration: std::sync::atomic::AtomicU64::new(1),
        });

        let policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.concurrency = ConcurrencyClass::Batchable;
        profile.physical_launch_policy = policy;
        profile.max_batch_size = 2;
        profile.resolved_from_loaded_model = true;
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(88),
            "test.reverse.terminal",
            &profile,
            NativeBatchMode::None,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.output_visibility = OutputVisibility::IncrementalCommitted;
        let slow_binding = ExecutionAdapterBinding {
            execution_group_id: ExecutionGroupId::new(88),
            model_instance_id: ModelInstanceId::new(88),
            adapter_instance_id: AdapterInstanceId::new(88),
            adapter_abi_revision: AdapterAbiRevision::new(1),
            model_variant: variant,
            capability_id: "tts".to_string(),
            stages: Arc::from([stage]),
        };
        let mut fast_binding = slow_binding.clone();
        fast_binding.execution_group_id = ExecutionGroupId::new(89);
        fast_binding.model_instance_id = ModelInstanceId::new(89);
        fast_binding.adapter_instance_id = AdapterInstanceId::new(89);
        let mut slow = EngineCoreRequest::tts("slow").with_model_variant(variant);
        slow.id = "slow-mailbox".to_string();
        slow.prompt_tokens = vec![1];
        slow.bind_execution_adapter(slow_binding).unwrap();
        let mut fast = EngineCoreRequest::tts("fast").with_model_variant(variant);
        fast.id = "fast-mailbox".to_string();
        fast.prompt_tokens = vec![1];
        fast.bind_execution_adapter(fast_binding).unwrap();
        fast.streaming = true;
        let (fast_stream_tx, mut fast_stream_rx) = mpsc::channel(4);
        fast.streaming_tx = Some(fast_stream_tx);

        let (slow_registration, slow_rx) =
            engine.register_completion_mailbox(slow.id.clone()).unwrap();
        let (fast_registration, fast_rx) =
            engine.register_completion_mailbox(fast.id.clone()).unwrap();
        {
            let mut core = engine.core.write().await;
            core.add_request(slow).unwrap();
            core.add_request(fast).unwrap();
            let slow_session = core.get_session_key(&"slow-mailbox".to_string()).unwrap();
            let fast_session = core.get_session_key(&"fast-mailbox".to_string()).unwrap();
            engine.bind_completion_mailbox(
                &slow_session.request_id,
                slow_registration.registration_id,
                slow_session.epoch,
            );
            engine.bind_completion_mailbox(
                &fast_session.request_id,
                fast_registration.registration_id,
                fast_session.epoch,
            );
        }

        let stepping = engine.clone();
        let step = tokio::spawn(async move { stepping.step().await });
        slow_entered_rx.await.expect("slow dispatch did not enter");
        let fast_output = tokio::time::timeout(Duration::from_secs(1), fast_rx).await;
        let progress = fast_stream_rx.try_recv();
        let step_finished_early = step.is_finished();
        let fast_still_active = engine.has_request(&"fast-mailbox".to_string()).await;
        let slow_still_active = engine.has_request(&"slow-mailbox".to_string()).await;
        let fast_cleaned = cleanup
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains(&"fast-mailbox".to_string());

        let (released, wake) = release.as_ref();
        *released.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
        let slow_output = slow_rx.await.expect("slow mailbox closed");
        assert_eq!(slow_output.request_id, "slow-mailbox");
        step.await.unwrap().unwrap();
        let fast_output = fast_output
            .expect("fast mailbox waited for slow peer")
            .expect("fast mailbox closed");
        assert_eq!(fast_output.request_id, "fast-mailbox");
        assert!(!step_finished_early, "slow peer unexpectedly completed");
        let progress = progress.expect("terminal mailbox overtook committed progress delivery");
        assert_eq!(progress.text.as_deref(), Some("fast-progress"));
        assert!(!fast_still_active);
        assert!(slow_still_active);
        assert!(fast_cleaned);
        drop((slow_registration, fast_registration));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_routes_scalar_terminal_outputs_to_registered_mailboxes() {
        let max_batch_width = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = Arc::new(engine_with_test_executor(Box::new(
            ImmediateTerminalExecutor::new(max_batch_width.clone()),
        )));
        let first_request = immediate_terminal_request("run-first");
        let second_request = immediate_terminal_request("run-second");

        let (first_registration, first_completion) = engine
            .register_completion_mailbox(first_request.id.clone())
            .unwrap();
        engine
            .add_request_with_completion(first_request, Some(first_registration.registration_id))
            .await
            .unwrap();
        let (second_registration, second_completion) = engine
            .register_completion_mailbox(second_request.id.clone())
            .unwrap();
        engine
            .add_request_with_completion(second_request, Some(second_registration.registration_id))
            .await
            .unwrap();

        let run_engine = engine.clone();
        let runner = tokio::spawn(async move { run_engine.run().await });
        let (first, second) = tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            tokio::join!(first_completion, second_completion)
        })
        .await
        .expect("run loop did not route both completions");
        engine.stop();
        tokio::time::timeout(tokio::time::Duration::from_secs(1), runner)
            .await
            .expect("run loop did not stop")
            .expect("run task panicked")
            .expect("run loop failed");

        let first = first.expect("first mailbox closed");
        let second = second.expect("second mailbox closed");
        assert_eq!(first.request_id, "run-first");
        assert_eq!(second.request_id, "run-second");
        assert_eq!(
            max_batch_width.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "unbound direct callers must remain on width-one compatibility execution"
        );

        // Exact-session acknowledgement happens after the mailboxes are routed,
        // so the public ID is reusable once delivery completes.
        engine
            .add_request(immediate_terminal_request("run-first"))
            .await
            .expect("delivered session must release its public ID fence");
        engine.abort_all_requests().await;
    }

    #[tokio::test]
    async fn outer_dispatcher_acknowledges_only_after_routing_terminal_output() {
        let engine = engine_with_test_executor(Box::new(ImmediateTerminalExecutor::new(Arc::new(
            std::sync::atomic::AtomicUsize::new(0),
        ))));
        let request = immediate_terminal_request("outer-dispatch");
        engine.add_request(request.clone()).await.unwrap();

        let outputs = engine.step_for_dispatch().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finished);
        assert!(
            engine.add_request(request.clone()).await.is_err(),
            "returning a batch to the outer dispatcher must retain the ID fence"
        );

        assert!(engine.acknowledge_dispatched_terminal(&outputs[0]).await);
        engine
            .add_request(request)
            .await
            .expect("the exact ID becomes reusable after outer routing acknowledgement");
        engine.abort_all_requests().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_generate_returns_cancelled_after_exact_abort() {
        let engine = Arc::new(engine_with_test_executor(Box::new(EndlessSequenceExecutor)));
        let mut request = EngineCoreRequest::tts("cancel direct generation");
        request.id = "direct-generate-abort".to_string();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = usize::MAX;
        let request_id = request.id.clone();
        let generating_engine = engine.clone();
        let generating = tokio::spawn(async move { generating_engine.generate(request).await });

        let session = tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                if let Some(session) = engine.request_session_key(&request_id).await {
                    break session;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("request was not admitted");
        assert!(engine.abort_request_session(&session).await.unwrap());

        let result = tokio::time::timeout(tokio::time::Duration::from_secs(1), generating)
            .await
            .expect("generate did not observe cancellation")
            .expect("generate task panicked");
        assert!(matches!(
            result,
            Err(Error::Cancelled(id)) if id == request_id
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_direct_generate_aborts_and_releases_its_exact_session() {
        let engine = Arc::new(engine_with_test_executor(Box::new(EndlessSequenceExecutor)));
        let mut request = EngineCoreRequest::tts("drop direct generation");
        request.id = "direct-generate-drop".to_string();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = usize::MAX;
        let request_id = request.id.clone();
        let generating_engine = engine.clone();
        let generating = tokio::spawn(async move { generating_engine.generate(request).await });

        let abandoned_session = tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                if let Some(session) = engine.request_session_key(&request_id).await {
                    break session;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("request was not admitted");

        generating.abort();
        assert!(generating
            .await
            .expect_err("generate task should be cancelled")
            .is_cancelled());

        tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                if engine.request_session_key(&request_id).await.is_none() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("abandoned exact session was not cleaned up");

        let mut replacement = EngineCoreRequest::tts("reuse abandoned request id");
        replacement.id = request_id.clone();
        replacement.prompt_tokens = vec![2];
        engine
            .add_request(replacement)
            .await
            .expect("abandoned request ID must be reusable after cleanup");
        let replacement_session = engine
            .request_session_key(&request_id)
            .await
            .expect("replacement session");
        assert_ne!(replacement_session.epoch, abandoned_session.epoch);
        engine.abort_all_requests().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_direct_generate_after_external_abort_releases_queued_terminal() {
        let engine = engine_with_test_executor(Box::new(EndlessSequenceExecutor));
        let mut request = EngineCoreRequest::tts("abort before dropping direct generation");
        request.id = "direct-generate-abort-drop".to_string();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = usize::MAX;
        let request_id = request.id.clone();

        // Install the same mailbox/request ownership that `generate` holds,
        // but keep the abort-before-drop ordering deterministic.
        let (registration, completion) = engine
            .register_completion_mailbox(request_id.clone())
            .expect("completion registration");
        engine
            .add_request_with_completion(request, Some(registration.registration_id))
            .await
            .expect("request admission");
        let abandoned_session = engine
            .request_session_key(&request_id)
            .await
            .expect("request session");

        assert!(engine
            .abort_request_session(&abandoned_session)
            .await
            .expect("exact abort"));
        drop(registration);
        assert!(completion.await.is_err());

        let mut replacement = EngineCoreRequest::tts("reuse externally aborted request id");
        replacement.id = request_id.clone();
        replacement.prompt_tokens = vec![2];
        tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                match engine.add_request(replacement.clone()).await {
                    Ok(_) => break,
                    Err(_) => tokio::task::yield_now().await,
                }
            }
        })
        .await
        .expect("abandoned queued terminal kept the request ID fenced");
        let replacement_session = engine
            .request_session_key(&request_id)
            .await
            .expect("replacement session");
        assert_ne!(replacement_session.epoch, abandoned_session.epoch);
        engine.abort_all_requests().await;
    }

    #[test]
    fn speech_to_speech_streaming_queue_defaults_deeper_than_tts() {
        let tts_request = EngineCoreRequest::tts("hello");
        let speech_to_speech_request = EngineCoreRequest::speech_to_speech("audio");

        assert_eq!(Engine::streaming_queue_capacity(&tts_request), 8);
        assert_eq!(
            Engine::streaming_queue_capacity(&speech_to_speech_request),
            64
        );
    }

    #[test]
    fn asr_streaming_queue_default_handles_character_level_deltas() {
        let asr_request = EngineCoreRequest::asr("audio");
        assert_eq!(Engine::streaming_queue_capacity(&asr_request), 4096);
    }

    #[tokio::test]
    async fn bulk_abort_signals_and_removes_matching_request_controls() {
        use std::sync::atomic::Ordering;

        let engine = Engine::new(EngineCoreConfig::default()).unwrap();
        let mut first = EngineCoreRequest::tts("first");
        first.id = "first".to_string();
        first.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
        let mut second = EngineCoreRequest::tts("second");
        second.id = "second".to_string();
        second.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        engine.add_request(first).await.unwrap();
        engine.add_request(second).await.unwrap();

        let (first_signal, second_signal) = {
            let controls = engine
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            (
                controls["first"].cancellation.clone(),
                controls["second"].cancellation.clone(),
            )
        };

        assert_eq!(
            engine
                .abort_requests_for_variant(ModelVariant::Qwen3Tts12Hz06BBase)
                .await,
            vec!["first".to_string()]
        );
        assert!(first_signal.load(Ordering::Acquire));
        assert!(!second_signal.load(Ordering::Acquire));
        {
            let controls = engine
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            assert!(!controls.contains_key("first"));
            assert!(controls.contains_key("second"));
        }

        assert_eq!(
            engine.abort_all_requests().await,
            vec!["second".to_string()]
        );
        assert!(second_signal.load(Ordering::Acquire));
        assert!(engine
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_empty());
    }
}
