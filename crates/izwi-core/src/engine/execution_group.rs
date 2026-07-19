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

use tracing::warn;

use super::execution::{
    ExecutionDisposition, ExecutionFailure, ExecutionReport, PhysicalBatch, PhysicalBatchReport,
    PhysicalBatchRowReport, RetryDisposition, StateDisposition,
};
use super::executor::{ExecutorOutput, ExecutorStepResult, UnifiedExecutor};
use super::request::EngineCoreRequest;
use super::scheduler::ScheduledRequest;
use super::types::RequestId;
use crate::error::Result;

/// One compatibility-checked physical executor call.
pub(super) struct PreparedExecutionBatch {
    physical_batch: PhysicalBatch,
    requests: Vec<Arc<EngineCoreRequest>>,
    scheduled: Vec<ScheduledRequest>,
}

impl PreparedExecutionBatch {
    pub(super) fn new(
        physical_batch: PhysicalBatch,
        requests: Vec<Arc<EngineCoreRequest>>,
        scheduled: Vec<ScheduledRequest>,
    ) -> Self {
        Self {
            physical_batch,
            requests,
            scheduled,
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
    pub(super) decode_ids: HashSet<RequestId>,
    pub(super) prefill_ids: HashSet<RequestId>,
    pub(super) decode_elapsed: Duration,
    pub(super) prefill_elapsed: Duration,
}

pub(super) struct ExecutedPhysicalBatch {
    pub(super) physical_batch: PhysicalBatch,
    pub(super) report: PhysicalBatchReport,
    pub(super) results: Vec<ExecutorStepResult>,
}

/// The sole owner of model-forward dispatch within one engine step.
pub(super) struct ExecutionGroupRunner;

impl ExecutionGroupRunner {
    pub(super) async fn execute(prepared: PreparedEngineStep) -> ExecutedEngineStep {
        let decode_ids = request_ids(&prepared.decode_batches);
        let prefill_ids = request_ids(&prepared.prefill_batches);

        // Physical device work is deliberately serialized for every backend.
        // Tensor adapters may still fan out inside one physical batch.
        let (mut batches, decode_elapsed) =
            execute_batches(&prepared.executor, "decode", prepared.decode_batches, false).await;
        let (mut prefill_batches, prefill_elapsed) = execute_batches(
            &prepared.executor,
            "prefill",
            prepared.prefill_batches,
            true,
        )
        .await;
        batches.append(&mut prefill_batches);

        ExecutedEngineStep {
            batches,
            decode_ids,
            prefill_ids,
            decode_elapsed,
            prefill_elapsed,
        }
    }
}

fn request_ids(batches: &[PreparedExecutionBatch]) -> HashSet<RequestId> {
    batches
        .iter()
        .flat_map(|batch| batch.scheduled.iter())
        .map(|scheduled| scheduled.request_id.clone())
        .collect()
}

async fn execute_batches(
    executor: &UnifiedExecutor,
    phase: &'static str,
    batches: Vec<PreparedExecutionBatch>,
    is_prefill: bool,
) -> (Vec<ExecutedPhysicalBatch>, Duration) {
    if batches.is_empty() {
        return (Vec::new(), Duration::ZERO);
    }

    let started = Instant::now();
    let mut executed = Vec::new();
    for batch in batches {
        let batch_started = Instant::now();
        let request_refs: Vec<_> = batch.requests.iter().map(Arc::as_ref).collect();
        let result = if is_prefill {
            executor
                .execute_prefill(&request_refs, &batch.scheduled)
                .await
        } else {
            executor
                .execute_decode(&request_refs, &batch.scheduled)
                .await
        };
        let results = reconcile_executor_outputs(phase, &batch.scheduled, result);
        let elapsed = batch_started.elapsed();
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
            observed_resources: super::ResourceVector::zero(),
            elapsed,
            rows,
        };
        executed.push(ExecutedPhysicalBatch {
            physical_batch: batch.physical_batch,
            report,
            results,
        });
    }
    (executed, started.elapsed())
}

fn execution_report_from_result(result: &ExecutorStepResult, elapsed: Duration) -> ExecutionReport {
    ExecutionReport {
        plan_id: result.plan_id,
        session: result.session.clone(),
        input_consumed: result.output.tokens_processed,
        output_produced: result.output.tokens_generated,
        observed_resources: result.observed_resources,
        dispatch: result.dispatch,
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
    result: Result<Vec<ExecutorStepResult>>,
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
                .map(|entry| failed_step_result(entry, format!("{phase} executor failed: {err}")))
                .collect();
        }
    };

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
                );
            }
            by_transaction.remove(&key).unwrap_or_else(|| {
                failed_step_result(
                    entry,
                    format!("{phase} executor did not return a scheduled output"),
                )
            })
        })
        .collect()
}
