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
            execute_batches(&prepared.executor, "decode", prepared.decode_batches).await;
        let (mut prefill_batches, prefill_elapsed) =
            execute_batches(&prepared.executor, "prefill", prepared.prefill_batches).await;
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
) -> (Vec<ExecutedPhysicalBatch>, Duration) {
    if batches.is_empty() {
        return (Vec::new(), Duration::ZERO);
    }

    let started = Instant::now();
    let mut executed = Vec::new();
    for batch in batches {
        let batch_started = Instant::now();
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
                        result
                    })
                    .collect();
                executed.push(executed_batch(batch, results, batch_started.elapsed()));
                continue;
            }
        };
        let request_refs: Vec<_> = batch.requests.iter().map(Arc::as_ref).collect();
        let result = executor
            .execute_physical_batch(&batch.physical_batch, &request_refs, &batch.scheduled)
            .await;
        let results = reconcile_executor_outputs(phase, &batch.scheduled, result);
        executed.push(executed_batch(batch, results, batch_started.elapsed()));
        drop(workspace);
    }
    (executed, started.elapsed())
}

fn executed_batch(
    batch: PreparedExecutionBatch,
    results: Vec<ExecutorStepResult>,
    elapsed: Duration,
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
        observed_resources: super::ResourceVector::zero(),
        elapsed,
        rows,
    };
    ExecutedPhysicalBatch {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::backends::BackendKind;
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchBudget, BatchDispatchKind, BatchId,
        BatchLaneKey, ExecutionGroupId, InputRange, ModelExecutor, ModelInstanceId,
        NativeBatchMode, PhysicalBatchExecution, ReadyQuantum, SequencePhase, SessionKey, StageId,
        WorkCost, WorkUnit,
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

    impl ModelExecutor for PhysicalBoundaryExecutor {
        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> Result<Vec<ExecutorStepResult>> {
            execution.validate()?;
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

        let reconciled =
            reconcile_executor_outputs("prefill", &scheduled, Ok(vec![first, duplicate, unknown]));

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
            workspace_bytes: 1,
        };
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![PreparedExecutionBatch::new(
                physical_batch,
                vec![Arc::new(request)],
                vec![scheduled],
            )],
        );

        let executed = ExecutionGroupRunner::execute(prepared).await;
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
            workspace_bytes: 0,
        };
        let prepared = PreparedEngineStep::new(
            executor,
            Vec::new(),
            vec![PreparedExecutionBatch::new(
                physical_batch,
                vec![Arc::new(request)],
                vec![scheduled],
            )],
        );

        let executed = ExecutionGroupRunner::execute(prepared).await;
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
