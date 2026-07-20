//! Serialized physical execution for one engine execution group.
//!
//! The scheduler and lifecycle state live in [`super::core::EngineCore`], but
//! model forwards must not run while that mutable state is locked. A prepared
//! step owns immutable request snapshots and exact scheduler transactions; the
//! runner consumes those batches serially and returns results for a later
//! fenced commit.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tracing::warn;

use super::execution::{
    DeadlinePhase, DispatchState, ExecutionDisposition, ExecutionFailure, ExecutionReport,
    FailureKind, FailureOrigin, FailureScope, FinishReason, HealthImpact, OutcomeProvenance,
    OutputVisibility, PhysicalBatch, PhysicalBatchReport, PhysicalBatchRowReport, RetryDisposition,
    StateDisposition,
};
use super::executor::{
    ExecutorOutput, ExecutorStepResult, PhysicalDispatchResult, StreamDeliveryFailure,
    StreamDeliveryFailureKind, UnifiedExecutor,
};
use super::request::{
    EngineCoreRequest, FencedStreamProgress, StreamBindingGuard, StreamProgressBudget,
};
use super::scheduler::ScheduledRequest;
#[cfg(test)]
use crate::error::Result;

/// One compatibility-checked physical executor call.
pub(super) struct PreparedExecutionBatch {
    physical_batch: PhysicalBatch,
    requests: Vec<Arc<EngineCoreRequest>>,
    scheduled: Vec<ScheduledRequest>,
    output_visibility: OutputVisibility,
}

impl PreparedExecutionBatch {
    pub(super) fn new(
        physical_batch: PhysicalBatch,
        requests: Vec<Arc<EngineCoreRequest>>,
        scheduled: Vec<ScheduledRequest>,
        output_visibility: OutputVisibility,
    ) -> Self {
        Self {
            physical_batch,
            requests,
            scheduled,
            output_visibility,
        }
    }

    #[cfg(test)]
    pub(super) fn physical_batch(&self) -> &PhysicalBatch {
        &self.physical_batch
    }
}

/// Immutable work detached from the mutable engine state.
pub(super) struct PreparedEngineStep {
    executor: UnifiedExecutor,
    decode_batches: Vec<PreparedExecutionBatch>,
    prefill_batches: Vec<PreparedExecutionBatch>,
}

impl PreparedEngineStep {
    pub(super) fn new(
        executor: UnifiedExecutor,
        decode_batches: Vec<PreparedExecutionBatch>,
        prefill_batches: Vec<PreparedExecutionBatch>,
    ) -> Self {
        Self {
            executor,
            decode_batches,
            prefill_batches,
        }
    }
}

/// Results that can only be applied by the engine's commit phase.
pub(super) struct ExecutedEngineStep {
    pub(super) batches: Vec<ExecutedPhysicalBatch>,
}

impl ExecutedEngineStep {
    pub(super) fn apply_stream_delivery_failures(&mut self, failures: &[StreamDeliveryFailure]) {
        let failed = failures
            .iter()
            .map(|failure| (failure.session.clone(), failure.kind))
            .collect::<HashMap<_, _>>();
        for batch in &mut self.batches {
            let mut changed = false;
            for result in &mut batch.results {
                let Some(kind) = failed.get(&result.session).copied() else {
                    continue;
                };
                changed = true;
                result.safe_point = true;
                result.staged_stream_outputs.clear();
                match kind {
                    StreamDeliveryFailureKind::Delivery => {
                        let message = "committed stream delivery failed";
                        result.output =
                            ExecutorOutput::error(result.session.request_id.clone(), message);
                        result.disposition =
                            ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
                        result.provenance = OutcomeProvenance::failure(
                            FailureOrigin::StreamDelivery,
                            DispatchState::ProducedOutput,
                        );
                    }
                    StreamDeliveryFailureKind::Deadline => {
                        result.output = ExecutorOutput::terminal(result.session.request_id.clone());
                        result.disposition = ExecutionDisposition::Finished(FinishReason::TimedOut);
                        result.provenance = OutcomeProvenance::deadline(
                            DeadlinePhase::StreamDelivery,
                            DispatchState::ProducedOutput,
                        );
                    }
                    StreamDeliveryFailureKind::Cancelled => {
                        result.output =
                            ExecutorOutput::cancelled(result.session.request_id.clone());
                        result.disposition =
                            ExecutionDisposition::Finished(FinishReason::Cancelled);
                        result.provenance = OutcomeProvenance::produced_output();
                    }
                    StreamDeliveryFailureKind::RequestDeadline => {
                        result.output = ExecutorOutput::terminal(result.session.request_id.clone());
                        result.disposition = ExecutionDisposition::Finished(FinishReason::TimedOut);
                        result.provenance = OutcomeProvenance::deadline(
                            DeadlinePhase::ModelExecution,
                            DispatchState::ProducedOutput,
                        );
                    }
                    StreamDeliveryFailureKind::InvalidProgress => {
                        let message = "executor emitted invalid incremental stream progress";
                        result.output =
                            ExecutorOutput::error(result.session.request_id.clone(), message);
                        result.disposition =
                            ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
                        result.provenance = OutcomeProvenance::failure(
                            FailureOrigin::ExecutorValidation,
                            DispatchState::ProducedOutput,
                        );
                    }
                }
            }
            if changed {
                batch.report.rows = batch
                    .results
                    .iter()
                    .map(|result| PhysicalBatchRowReport {
                        execution: execution_report_from_result(result, batch.report.elapsed),
                        state: state_disposition(&result.disposition),
                    })
                    .collect();
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExecutionPhase {
    Decode,
    Prefill,
}

impl ExecutionPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

pub(super) struct ExecutedPhysicalBatch {
    pub(super) phase: ExecutionPhase,
    pub(super) physical_batch: PhysicalBatch,
    pub(super) report: PhysicalBatchReport,
    pub(super) results: Vec<ExecutorStepResult>,
}

/// The sole owner of model-forward dispatch within one engine step.
pub(super) struct ExecutionGroupRunner;

impl ExecutionGroupRunner {
    pub(super) async fn execute(
        prepared: PreparedEngineStep,
        progress_tx: mpsc::Sender<FencedStreamProgress>,
        progress_budget: Arc<StreamProgressBudget>,
    ) -> ExecutedEngineStep {
        // Physical device work is deliberately serialized for every backend.
        // Tensor adapters may still fan out inside one physical batch.
        let mut batches = execute_batches(
            &prepared.executor,
            ExecutionPhase::Decode,
            prepared.decode_batches,
            &progress_tx,
            &progress_budget,
        )
        .await;
        let mut prefill_batches = execute_batches(
            &prepared.executor,
            ExecutionPhase::Prefill,
            prepared.prefill_batches,
            &progress_tx,
            &progress_budget,
        )
        .await;
        batches.append(&mut prefill_batches);

        ExecutedEngineStep { batches }
    }
}

async fn execute_batches(
    executor: &UnifiedExecutor,
    phase: ExecutionPhase,
    batches: Vec<PreparedExecutionBatch>,
    progress_tx: &mpsc::Sender<FencedStreamProgress>,
    progress_budget: &Arc<StreamProgressBudget>,
) -> Vec<ExecutedPhysicalBatch> {
    if batches.is_empty() {
        return Vec::new();
    }

    let mut executed = Vec::new();
    for batch in batches {
        let batch_started = Instant::now();
        if let Some(results) = pre_dispatch_deadline_results(&batch, Instant::now()) {
            executed.push(executed_batch(
                phase,
                batch,
                results,
                batch_started.elapsed(),
                super::ResourceVector::zero(),
            ));
            continue;
        }
        let workspace = match executor.reserve_batch_workspace(&batch.physical_batch) {
            Ok(workspace) => workspace,
            Err(error) => {
                let dispatch = super::BatchDispatch::not_dispatched(batch.scheduled.len());
                let results = batch
                    .scheduled
                    .iter()
                    .map(|scheduled| {
                        let mut result = failed_step_result(
                            scheduled,
                            format!("physical batch workspace admission failed: {error}"),
                        );
                        result.dispatch = dispatch;
                        result.provenance = OutcomeProvenance::failure(
                            FailureOrigin::WorkspaceAdmission,
                            DispatchState::NotStarted,
                        );
                        result
                    })
                    .collect();
                executed.push(executed_batch(
                    phase,
                    batch,
                    results,
                    batch_started.elapsed(),
                    super::ResourceVector::zero(),
                ));
                continue;
            }
        };
        if let Some(results) = pre_dispatch_deadline_results(&batch, Instant::now()) {
            drop(workspace);
            executed.push(executed_batch(
                phase,
                batch,
                results,
                batch_started.elapsed(),
                super::ResourceVector::zero(),
            ));
            continue;
        }
        let stream_bindings = match bind_stream_quantum(&batch, progress_tx, progress_budget) {
            Ok(bindings) => bindings,
            Err(error) => {
                drop(workspace);
                let dispatch = super::BatchDispatch::not_dispatched(batch.scheduled.len().max(1));
                let results = batch
                    .scheduled
                    .iter()
                    .map(|scheduled| {
                        failed_step_result(
                            scheduled,
                            format!("stream quantum binding failed: {error}"),
                        )
                        .with_dispatch(dispatch)
                        .with_provenance(OutcomeProvenance::failure(
                            FailureOrigin::ExecutorValidation,
                            DispatchState::NotStarted,
                        ))
                    })
                    .collect();
                executed.push(executed_batch(
                    phase,
                    batch,
                    results,
                    batch_started.elapsed(),
                    super::ResourceVector::zero(),
                ));
                continue;
            }
        };
        let observed_workspace = batch.physical_batch.workspace;
        let expected_dispatch = batch.physical_batch.expected_dispatch();
        let request_refs: Vec<_> = batch.requests.iter().map(Arc::as_ref).collect();
        let result = executor
            .execute_physical_batch(&batch.physical_batch, &request_refs, &batch.scheduled)
            .await;
        let mut results =
            reconcile_executor_outputs(phase.label(), &batch.scheduled, expected_dispatch, result);
        apply_post_dispatch_deadlines(&batch, Instant::now(), &mut results);
        drop(stream_bindings);
        executed.push(executed_batch(
            phase,
            batch,
            results,
            batch_started.elapsed(),
            observed_workspace,
        ));
        drop(workspace);
    }
    executed
}

fn bind_stream_quantum(
    batch: &PreparedExecutionBatch,
    progress_tx: &mpsc::Sender<FencedStreamProgress>,
    progress_budget: &Arc<StreamProgressBudget>,
) -> crate::error::Result<Vec<StreamBindingGuard>> {
    if batch.requests.len() != batch.physical_batch.rows.len()
        || batch.scheduled.len() != batch.physical_batch.rows.len()
    {
        return Err(crate::error::Error::InferenceError(
            "physical batch rows do not match stream binding inputs".to_string(),
        ));
    }

    batch
        .requests
        .iter()
        .zip(&batch.physical_batch.rows)
        .map(|(request, row)| {
            request.bind_stream_quantum(
                batch.physical_batch.batch_id,
                batch.physical_batch.lane.clone(),
                row.plan_id,
                row.session.clone(),
                batch.output_visibility,
                progress_tx.clone(),
                progress_budget.clone(),
            )
        })
        .collect()
}

fn pre_dispatch_deadline_results(
    batch: &PreparedExecutionBatch,
    now: Instant,
) -> Option<Vec<ExecutorStepResult>> {
    if batch.requests.len() != batch.scheduled.len() {
        return None;
    }
    let deadlines = batch
        .requests
        .iter()
        .map(|request| (request.id.as_str(), request.deadline))
        .collect::<HashMap<_, _>>();
    if deadlines.len() != batch.requests.len() {
        return None;
    }
    let expired = batch
        .scheduled
        .iter()
        .map(|scheduled| {
            deadlines
                .get(scheduled.request_id.as_str())
                .is_some_and(|deadline| deadline.is_some_and(|deadline| now >= deadline))
        })
        .collect::<Vec<_>>();
    if !expired.iter().any(|expired| *expired) {
        return None;
    }

    let dispatch = super::BatchDispatch::not_dispatched(batch.scheduled.len().max(1));
    Some(
        batch
            .scheduled
            .iter()
            .zip(expired)
            .map(|(scheduled, expired)| {
                if expired {
                    let mut result = ExecutorStepResult::new(
                        scheduled,
                        ExecutorOutput::terminal(scheduled.request_id.clone()),
                    );
                    result.disposition = ExecutionDisposition::Finished(FinishReason::TimedOut);
                    result.dispatch = dispatch;
                    result.provenance = OutcomeProvenance::deadline(
                        DeadlinePhase::DispatchWait,
                        DispatchState::NotStarted,
                    );
                    result
                } else {
                    let message =
                        "physical batch dispatch deferred because a peer deadline expired";
                    let mut output =
                        ExecutorOutput::error(scheduled.request_id.clone(), message.to_string());
                    output.finished = false;
                    let mut result = ExecutorStepResult::new(scheduled, output);
                    result.disposition = ExecutionDisposition::Failed(ExecutionFailure {
                        kind: FailureKind::Internal,
                        scope: FailureScope::PhysicalBatch,
                        retry: RetryDisposition::RetrySameSession,
                        health: HealthImpact::None,
                        message: message.to_string(),
                    });
                    result.dispatch = dispatch;
                    result.provenance = OutcomeProvenance::failure(
                        FailureOrigin::DispatchCoordination,
                        DispatchState::NotStarted,
                    );
                    result
                }
            })
            .collect(),
    )
}

fn apply_post_dispatch_deadlines(
    batch: &PreparedExecutionBatch,
    now: Instant,
    results: &mut [ExecutorStepResult],
) {
    let deadlines = batch
        .requests
        .iter()
        .map(|request| (request.id.as_str(), request.deadline))
        .collect::<HashMap<_, _>>();
    for result in results {
        let Some(Some(deadline)) = deadlines.get(result.session.request_id.as_str()) else {
            continue;
        };
        if now < *deadline {
            continue;
        }
        let (phase, dispatch_state) = match result.provenance.dispatch_state {
            DispatchState::NotStarted => (DeadlinePhase::DispatchWait, DispatchState::NotStarted),
            DispatchState::Started => (DeadlinePhase::ModelExecution, DispatchState::Started),
            DispatchState::ProducedOutput => {
                (DeadlinePhase::ModelExecution, DispatchState::ProducedOutput)
            }
        };
        result.output = ExecutorOutput::terminal(result.session.request_id.clone());
        result.disposition = ExecutionDisposition::Finished(FinishReason::TimedOut);
        result.safe_point = true;
        result.provenance = OutcomeProvenance::deadline(phase, dispatch_state);
        result.staged_stream_outputs.clear();
    }
}

fn executed_batch(
    phase: ExecutionPhase,
    batch: PreparedExecutionBatch,
    results: Vec<ExecutorStepResult>,
    elapsed: Duration,
    observed_resources: super::ResourceVector,
) -> ExecutedPhysicalBatch {
    let dispatch = results
        .first()
        .map(|result| result.dispatch)
        .unwrap_or_default();
    let rows = results
        .iter()
        .map(|result| PhysicalBatchRowReport {
            execution: execution_report_from_result(result, elapsed),
            state: state_disposition(&result.disposition),
        })
        .collect();
    let report = PhysicalBatchReport {
        batch_id: batch.physical_batch.batch_id,
        lane: batch.physical_batch.lane.clone(),
        dispatch,
        observed_resources,
        elapsed,
        rows,
    };
    ExecutedPhysicalBatch {
        phase,
        physical_batch: batch.physical_batch,
        report,
        results,
    }
}

fn execution_report_from_result(result: &ExecutorStepResult, elapsed: Duration) -> ExecutionReport {
    ExecutionReport {
        plan_id: result.plan_id,
        session: result.session.clone(),
        input_consumed: result.output.tokens_processed,
        output_produced: result.output.tokens_generated,
        observed_resources: result.observed_resources,
        dispatch: result.dispatch,
        provenance: result.provenance,
        elapsed,
        safe_point: result.safe_point,
        disposition: result.disposition.clone(),
        output_finished: result.output.finished,
        output_has_error: result.output.error.is_some(),
    }
}

fn state_disposition(disposition: &ExecutionDisposition) -> StateDisposition {
    match disposition {
        ExecutionDisposition::Progress | ExecutionDisposition::Yielded(_) => {
            StateDisposition::ValidNext
        }
        ExecutionDisposition::Failed(ExecutionFailure {
            retry: RetryDisposition::RetrySameSession,
            ..
        }) => StateDisposition::Unchanged,
        ExecutionDisposition::Failed(ExecutionFailure {
            retry: RetryDisposition::Recompute,
            ..
        }) => StateDisposition::RolledBack,
        ExecutionDisposition::Failed(_) => StateDisposition::Poisoned,
        ExecutionDisposition::Finished(_) => StateDisposition::Unchanged,
    }
}

fn failed_step_result(
    scheduled: &ScheduledRequest,
    message: impl Into<String>,
) -> ExecutorStepResult {
    ExecutorStepResult::new(
        scheduled,
        ExecutorOutput::error(scheduled.request_id.clone(), message),
    )
}

pub(super) fn reconcile_executor_outputs(
    phase: &str,
    scheduled: &[ScheduledRequest],
    expected_dispatch: super::BatchDispatch,
    result: PhysicalDispatchResult,
) -> Vec<ExecutorStepResult> {
    let expected: HashSet<_> = scheduled
        .iter()
        .map(|entry| (entry.plan_id, entry.session_key()))
        .collect();
    let outputs = match result {
        Ok(outputs) => outputs,
        Err(err) => {
            return scheduled
                .iter()
                .map(|entry| {
                    failed_step_result(entry, format!("{phase} executor failed: {}", err.error))
                        .with_dispatch(err.dispatch)
                        .with_provenance(err.provenance)
                })
                .collect();
        }
    };

    let dispatch = outputs
        .first()
        .map(|output| output.dispatch)
        .unwrap_or(expected_dispatch);
    if outputs.iter().any(|output| output.dispatch != dispatch) {
        return scheduled
            .iter()
            .map(|entry| {
                failed_step_result(entry, format!("{phase} executor returned mixed dispatches"))
                    .with_dispatch(expected_dispatch)
                    .with_provenance(OutcomeProvenance::failure(
                        FailureOrigin::ExecutorValidation,
                        DispatchState::Started,
                    ))
            })
            .collect();
    }

    let mut by_transaction = HashMap::new();
    let mut duplicates = HashSet::new();
    for mut result in outputs {
        let key = (result.plan_id, result.session.clone());
        if !expected.contains(&key) {
            warn!(
                phase,
                plan_id = result.plan_id,
                request_id = %result.session.request_id,
                session_epoch = result.session.epoch,
                "Ignoring executor output for an unknown or stale transaction"
            );
            continue;
        }
        if result.output.request_id != result.session.request_id {
            result.output = ExecutorOutput::error(
                result.session.request_id.clone(),
                format!("{phase} executor output request ID did not match its session"),
            );
            result.disposition = ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                format!("{phase} executor output request ID did not match its session"),
            ));
            result.safe_point = true;
            result.provenance = OutcomeProvenance::failure(
                FailureOrigin::ExecutorValidation,
                dispatch_state_for(dispatch),
            );
            result.staged_stream_outputs.clear();
        }
        if by_transaction.insert(key.clone(), result).is_some() {
            duplicates.insert(key);
        }
    }

    scheduled
        .iter()
        .map(|entry| {
            let key = (entry.plan_id, entry.session_key());
            if duplicates.contains(&key) {
                return failed_step_result(
                    entry,
                    format!("{phase} executor returned duplicate outputs"),
                )
                .with_dispatch(dispatch)
                .with_provenance(OutcomeProvenance::failure(
                    FailureOrigin::ExecutorValidation,
                    dispatch_state_for(dispatch),
                ));
            }
            by_transaction.remove(&key).unwrap_or_else(|| {
                failed_step_result(
                    entry,
                    format!("{phase} executor did not return a scheduled output"),
                )
                .with_dispatch(dispatch)
                .with_provenance(OutcomeProvenance::failure(
                    FailureOrigin::ExecutorValidation,
                    dispatch_state_for(dispatch),
                ))
            })
        })
        .collect()
}

fn dispatch_state_for(dispatch: super::BatchDispatch) -> DispatchState {
    if dispatch.kind == super::BatchDispatchKind::NotDispatched {
        DispatchState::NotStarted
    } else {
        DispatchState::Started
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::backends::BackendKind;
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchBudget, BatchDispatch, BatchDispatchKind,
        BatchId, BatchLaneKey, ExecutionGroupId, InputRange, ModelExecutor, ModelInstanceId,
        NativeBatchMode, PhysicalBatchExecution, PhysicalDispatchError, PhysicalDispatchResult,
        ReadyQuantum, ResourceVector, SequencePhase, SessionKey, StageId, WorkCost, WorkUnit,
    };

    struct CountingExecutor {
        calls: Arc<AtomicUsize>,
    }

    impl ModelExecutor for CountingExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Vec::new())
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
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

    struct PhysicalBoundaryExecutor {
        physical_calls: Arc<AtomicUsize>,
        legacy_calls: Arc<AtomicUsize>,
    }

    struct SleepingPhysicalExecutor {
        physical_calls: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl ModelExecutor for SleepingPhysicalExecutor {
        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            execution.validate().expect("test physical batch");
            self.physical_calls.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(self.delay);
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
    }

    impl ModelExecutor for PhysicalBoundaryExecutor {
        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            execution.validate().expect("test physical batch");
            self.physical_calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(execution.batch.batch_id, BatchId::new(9));
            Ok(execution
                .scheduled
                .iter()
                .map(|scheduled| failed_step_result(scheduled, "physical boundary observed"))
                .collect())
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            self.legacy_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Vec::new())
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            self.legacy_calls.fetch_add(1, Ordering::Relaxed);
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

    fn lane() -> BatchLaneKey {
        BatchLaneKey {
            execution_group: ExecutionGroupId::new(1),
            model_instance: ModelInstanceId::new(2),
            adapter_instance: AdapterInstanceId::new(3),
            adapter_abi: AdapterAbiRevision::new(1),
            capability_id: "test".to_string(),
            stage_id: StageId::new(4),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: "exact".to_string(),
            quantization: "none".to_string(),
            state_schema: "none".to_string(),
            kernel_mode: "test".to_string(),
            semantic_mode: "test".to_string(),
            shape_bucket: "exact.1".to_string(),
        }
    }

    fn scheduled(request_id: &str, plan_id: u64, epoch: u64) -> ScheduledRequest {
        ScheduledRequest {
            plan_id,
            request_id: request_id.to_string(),
            sequence_id: epoch,
            num_tokens: 1,
            is_prefill: true,
            block_ids: Vec::new(),
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        }
    }

    fn prepared_batch(
        batch_id: u64,
        request: EngineCoreRequest,
        scheduled: ScheduledRequest,
    ) -> PreparedExecutionBatch {
        let lane = lane();
        PreparedExecutionBatch::new(
            PhysicalBatch {
                batch_id: BatchId::new(batch_id),
                lane: lane.clone(),
                mode: NativeBatchMode::None,
                budget: BatchBudget::width_one(),
                rows: vec![ReadyQuantum {
                    plan_id: scheduled.plan_id,
                    session: scheduled.session_key(),
                    lane,
                    work: scheduled.work.clone(),
                    cost: WorkCost::new(1, 1, 0),
                }],
                materialized_tensor_elements: 1,
                workspace: ResourceVector::zero(),
            },
            vec![Arc::new(request)],
            vec![scheduled],
            OutputVisibility::AfterQuantumCommit,
        )
    }

    async fn execute_prepared(prepared: PreparedEngineStep) -> ExecutedEngineStep {
        let (progress_tx, _progress_rx) = mpsc::channel(64);
        ExecutionGroupRunner::execute(
            prepared,
            progress_tx,
            StreamProgressBudget::new(1024 * 1024),
        )
        .await
    }

    #[tokio::test]
    async fn stream_progress_failures_keep_their_typed_terminal_outcomes() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(CountingExecutor { calls }));
        let mut request = EngineCoreRequest::tts("typed progress failure");
        request.id = "typed-progress".to_string();
        let scheduled = scheduled("typed-progress", 9, 0);
        let session = scheduled.session_key();
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![prepared_batch(9, request, scheduled)],
        );
        let mut executed = execute_prepared(prepared).await;

        executed.apply_stream_delivery_failures(&[StreamDeliveryFailure {
            session: session.clone(),
            kind: StreamDeliveryFailureKind::Cancelled,
        }]);
        let result = &executed.batches[0].results[0];
        assert_eq!(
            result.disposition,
            ExecutionDisposition::Finished(FinishReason::Cancelled)
        );
        assert_eq!(result.provenance, OutcomeProvenance::produced_output());

        executed.apply_stream_delivery_failures(&[StreamDeliveryFailure {
            session: session.clone(),
            kind: StreamDeliveryFailureKind::RequestDeadline,
        }]);
        let result = &executed.batches[0].results[0];
        assert_eq!(
            result.disposition,
            ExecutionDisposition::Finished(FinishReason::TimedOut)
        );
        assert_eq!(
            result.provenance,
            OutcomeProvenance::deadline(
                DeadlinePhase::ModelExecution,
                DispatchState::ProducedOutput,
            )
        );

        executed.apply_stream_delivery_failures(&[StreamDeliveryFailure {
            session,
            kind: StreamDeliveryFailureKind::InvalidProgress,
        }]);
        let result = &executed.batches[0].results[0];
        assert!(matches!(
            result.disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                kind: FailureKind::InvalidOutput,
                retry: RetryDisposition::Never,
                ..
            })
        ));
        assert_eq!(
            result.provenance,
            OutcomeProvenance::failure(
                FailureOrigin::ExecutorValidation,
                DispatchState::ProducedOutput,
            )
        );
    }

    #[test]
    fn keyed_reconciliation_rejects_duplicate_unknown_and_missing_transactions() {
        let scheduled = vec![scheduled("req-a", 1, 0), scheduled("req-b", 2, 1)];
        let first = ExecutorStepResult::new(
            &scheduled[0],
            ExecutorOutput::terminal(scheduled[0].request_id.clone()),
        );
        let duplicate = first.clone();
        let mut unknown = first.clone();
        unknown.plan_id = 999;
        unknown.session = SessionKey::new("unknown".to_string(), 999);
        unknown.output.request_id = "unknown".to_string();

        let reconciled = reconcile_executor_outputs(
            "prefill",
            &scheduled,
            BatchDispatch::serial(),
            Ok(vec![first, duplicate, unknown]),
        );

        assert_eq!(
            reconciled
                .iter()
                .map(|result| result.output.request_id.as_str())
                .collect::<Vec<_>>(),
            vec!["req-a", "req-b"]
        );
        assert!(reconciled[0]
            .output
            .error
            .as_deref()
            .unwrap()
            .contains("duplicate"));
        assert!(reconciled[1]
            .output
            .error
            .as_deref()
            .unwrap()
            .contains("did not return"));
    }

    #[test]
    fn executor_batch_error_is_reported_as_one_failed_physical_dispatch() {
        let scheduled = vec![scheduled("req-a", 1, 0), scheduled("req-b", 2, 1)];

        let reconciled = reconcile_executor_outputs(
            "decode",
            &scheduled,
            BatchDispatch::new(BatchDispatchKind::TensorStatic, 2),
            Err(PhysicalDispatchError::started(
                crate::error::Error::InferenceError("tensor kernel failed".to_string()),
                BatchDispatch::new(BatchDispatchKind::TensorStatic, 2),
                FailureOrigin::Model,
            )),
        );

        assert_eq!(reconciled.len(), 2);
        assert!(reconciled.iter().all(|result| {
            result.dispatch.kind == BatchDispatchKind::TensorStatic
                && result.dispatch.width == 2
                && result.provenance.dispatch_state == DispatchState::Started
                && result.provenance.failure_origin == Some(FailureOrigin::Model)
                && result
                    .output
                    .error
                    .as_deref()
                    .is_some_and(|message| message.contains("tensor kernel failed"))
        }));
    }

    #[tokio::test]
    async fn workspace_rejection_never_enters_the_model_executor() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(CountingExecutor {
            calls: calls.clone(),
        }));
        let mut request = EngineCoreRequest::tts("workspace");
        request.id = "workspace".to_string();
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: 1,
            is_prefill: true,
            block_ids: Vec::new(),
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        };
        let lane = lane();
        let physical_batch = PhysicalBatch {
            batch_id: BatchId::new(1),
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: BatchBudget::width_one(),
            rows: vec![ReadyQuantum {
                plan_id: scheduled.plan_id,
                session: SessionKey::new(request.id.clone(), scheduled.sequence_id),
                lane,
                work: scheduled.work.clone(),
                cost: WorkCost::new(1, 1, 1),
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::temporary_workspace(1),
        };
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![PreparedExecutionBatch::new(
                physical_batch,
                vec![Arc::new(request)],
                vec![scheduled],
                OutputVisibility::AfterQuantumCommit,
            )],
        );

        let executed = execute_prepared(prepared).await;
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(executed.batches.len(), 1);
        assert_eq!(
            executed.batches[0].report.dispatch.kind,
            BatchDispatchKind::NotDispatched
        );
        assert!(executed.batches[0].results[0]
            .output
            .error
            .as_deref()
            .unwrap()
            .contains("workspace admission failed"));
        assert_eq!(
            executed.batches[0].results[0].provenance,
            OutcomeProvenance::failure(
                FailureOrigin::WorkspaceAdmission,
                DispatchState::NotStarted,
            )
        );
    }

    #[tokio::test]
    async fn deadline_expiring_behind_an_earlier_batch_never_enters_the_executor() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(SleepingPhysicalExecutor {
            physical_calls: calls.clone(),
            delay: Duration::from_millis(60),
        }));
        let mut first = EngineCoreRequest::tts("first");
        first.id = "first".to_string();
        let mut expired = EngineCoreRequest::tts("expired");
        expired.id = "expired".to_string();
        expired.deadline = Some(Instant::now() + Duration::from_millis(20));
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![
                prepared_batch(11, first, scheduled("first", 11, 1)),
                prepared_batch(12, expired, scheduled("expired", 12, 1)),
            ],
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(executed.batches.len(), 2);
        let expired = &executed.batches[1].results[0];
        assert_eq!(expired.dispatch.kind, BatchDispatchKind::NotDispatched);
        assert_eq!(
            expired.disposition,
            ExecutionDisposition::Finished(FinishReason::TimedOut)
        );
        assert_eq!(
            expired.provenance,
            OutcomeProvenance::deadline(DeadlinePhase::DispatchWait, DispatchState::NotStarted,)
        );
    }

    #[tokio::test]
    async fn expired_tensor_peer_defers_live_rows_without_changing_the_envelope() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(SleepingPhysicalExecutor {
            physical_calls: calls.clone(),
            delay: Duration::ZERO,
        }));
        let mut expired_request = EngineCoreRequest::tts("expired");
        expired_request.id = "expired-peer".to_string();
        expired_request.deadline = Some(Instant::now() - Duration::from_millis(1));
        let mut live_request = EngineCoreRequest::tts("live");
        live_request.id = "live-peer".to_string();
        live_request.deadline = Some(Instant::now() + Duration::from_secs(1));
        let expired = scheduled("expired-peer", 21, 1);
        let live = scheduled("live-peer", 22, 1);
        let lane = lane();
        let physical_batch = PhysicalBatch {
            batch_id: BatchId::new(21),
            lane: lane.clone(),
            mode: NativeBatchMode::Static,
            budget: BatchBudget {
                max_rows: 2,
                max_logical_units: 2,
                max_tensor_elements: 2,
                max_workspace_bytes: 0,
                max_padding_basis_points: 0,
                max_formation_delay: Duration::ZERO,
            },
            rows: [&expired, &live]
                .into_iter()
                .map(|scheduled| ReadyQuantum {
                    plan_id: scheduled.plan_id,
                    session: scheduled.session_key(),
                    lane: lane.clone(),
                    work: scheduled.work.clone(),
                    cost: WorkCost::new(1, 1, 0),
                })
                .collect(),
            materialized_tensor_elements: 2,
            workspace: ResourceVector::zero(),
        };
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![PreparedExecutionBatch::new(
                physical_batch,
                vec![Arc::new(expired_request), Arc::new(live_request)],
                vec![expired, live],
                OutputVisibility::AfterQuantumCommit,
            )],
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let results = &executed.batches[0].results;
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].disposition,
            ExecutionDisposition::Finished(FinishReason::TimedOut)
        );
        assert!(matches!(
            &results[1].disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                retry: RetryDisposition::RetrySameSession,
                scope: FailureScope::PhysicalBatch,
                ..
            })
        ));
        assert!(!results[1].output.finished);
        assert_eq!(
            results[1].provenance,
            OutcomeProvenance::failure(
                FailureOrigin::DispatchCoordination,
                DispatchState::NotStarted,
            )
        );
        assert!(results
            .iter()
            .all(|result| result.dispatch.kind == BatchDispatchKind::NotDispatched));
    }

    #[tokio::test]
    async fn deadline_expiring_during_model_work_records_actual_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(SleepingPhysicalExecutor {
            physical_calls: calls.clone(),
            delay: Duration::from_millis(60),
        }));
        let mut request = EngineCoreRequest::tts("during-model");
        request.id = "during-model".to_string();
        request.deadline = Some(Instant::now() + Duration::from_millis(20));
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![prepared_batch(
                13,
                request,
                scheduled("during-model", 13, 1),
            )],
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let expired = &executed.batches[0].results[0];
        assert_eq!(expired.dispatch.kind, BatchDispatchKind::Serial);
        assert_eq!(
            expired.disposition,
            ExecutionDisposition::Finished(FinishReason::TimedOut)
        );
        assert_eq!(
            expired.provenance,
            OutcomeProvenance::deadline(
                DeadlinePhase::ModelExecution,
                DispatchState::ProducedOutput,
            )
        );
    }

    #[tokio::test]
    async fn runner_dispatches_the_exact_physical_batch_envelope() {
        let physical_calls = Arc::new(AtomicUsize::new(0));
        let legacy_calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(PhysicalBoundaryExecutor {
            physical_calls: physical_calls.clone(),
            legacy_calls: legacy_calls.clone(),
        }));
        let mut request = EngineCoreRequest::tts("physical");
        request.id = "physical".to_string();
        let scheduled = ScheduledRequest {
            plan_id: 5,
            request_id: request.id.clone(),
            sequence_id: 2,
            num_tokens: 1,
            is_prefill: true,
            block_ids: Vec::new(),
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        };
        let lane = lane();
        let physical_batch = PhysicalBatch {
            batch_id: BatchId::new(9),
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: BatchBudget::width_one(),
            rows: vec![ReadyQuantum {
                plan_id: scheduled.plan_id,
                session: scheduled.session_key(),
                lane,
                work: scheduled.work.clone(),
                cost: WorkCost::new(1, 1, 0),
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        };
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![PreparedExecutionBatch::new(
                physical_batch,
                vec![Arc::new(request)],
                vec![scheduled],
                OutputVisibility::AfterQuantumCommit,
            )],
        );

        let executed = execute_prepared(prepared).await;
        assert_eq!(physical_calls.load(Ordering::Relaxed), 1);
        assert_eq!(legacy_calls.load(Ordering::Relaxed), 0);
        assert_eq!(executed.batches.len(), 1);
        assert!(executed.batches[0].results[0]
            .output
            .error
            .as_deref()
            .unwrap()
            .contains("physical boundary observed"));
    }
}
