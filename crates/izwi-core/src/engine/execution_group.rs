//! Bounded physical execution for one engine execution group.
//!
//! The scheduler and lifecycle state live in [`super::core::EngineCore`], but
//! model forwards must not run while that mutable state is locked. A prepared
//! step owns immutable request snapshots and exact scheduler transactions; the
//! runner consumes those batches according to their sealed launch policies and
//! returns results for a later fenced commit.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::stream::{FuturesUnordered, StreamExt};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, warn};

use crate::config::{PhysicalExecutionMode, PhysicalInFlightLimit};

use super::execution::{
    DeadlinePhase, DispatchState, ExecutionDisposition, ExecutionFailure, ExecutionReport,
    FailureKind, FailureOrigin, FailureScope, FinishReason, HealthImpact, OutcomeProvenance,
    OutputVisibility, PhysicalBatch, PhysicalBatchReport, PhysicalBatchRowReport,
    PhysicalLaunchPolicy, RetryDisposition, StateDisposition,
};
use super::executor::{
    ExecutorOutput, ExecutorStepResult, ModelSessionResult, PhysicalDispatchResult,
    PhysicalExecutionAdmissionOutcome, StreamDeliveryFailure, StreamDeliveryFailureKind,
    UnifiedExecutor,
};
use super::metrics::{
    begin_engine_physical_workspace, record_engine_physical_defer, record_engine_physical_fallback,
    EnginePhysicalDeferReason, EnginePhysicalFallbackReason,
};
use super::request::{
    EngineCoreRequest, FencedStreamProgress, StreamBindingGuard, StreamProgressBudget,
};
use super::scheduler::ScheduledRequest;
#[cfg(test)]
use crate::error::Result;

/// One immutable, compatibility-checked physical executor call.
///
/// The ticket owns every input needed after the core lock is released. Its
/// physical envelope carries exact batch, lane/model, row/session/plan, work,
/// workspace, and managed-cache fences; the remaining fields retain immutable
/// request/model leases, scheduler inputs, phase, and progress visibility.
#[derive(Clone)]
pub(super) struct PreparedPhysicalDispatch {
    phase: ExecutionPhase,
    physical_batch: PhysicalBatch,
    requests: Vec<Arc<EngineCoreRequest>>,
    scheduled: Vec<ScheduledRequest>,
    output_visibility: OutputVisibility,
    managed_cache_reservations: Vec<super::ManagedCacheReservation>,
    launch_policy: PhysicalLaunchPolicy,
}

impl PreparedPhysicalDispatch {
    pub(super) fn new(
        phase: ExecutionPhase,
        physical_batch: PhysicalBatch,
        requests: Vec<Arc<EngineCoreRequest>>,
        scheduled: Vec<ScheduledRequest>,
        output_visibility: OutputVisibility,
        managed_cache_reservations: Vec<super::ManagedCacheReservation>,
        launch_policy: PhysicalLaunchPolicy,
    ) -> crate::error::Result<Self> {
        physical_batch.validate()?;
        if requests.len() != physical_batch.rows.len()
            || scheduled.len() != physical_batch.rows.len()
        {
            return Err(crate::error::Error::InvalidInput(
                "physical dispatch ticket inputs do not match its row count".to_string(),
            ));
        }

        let mut sessions = HashSet::with_capacity(physical_batch.rows.len());
        let mut plans = HashSet::with_capacity(physical_batch.rows.len());
        for ((request, scheduled), row) in requests.iter().zip(&scheduled).zip(&physical_batch.rows)
        {
            if request.id != scheduled.request_id
                || scheduled.session_key() != row.session
                || scheduled.plan_id != row.plan_id
                || scheduled.work != row.work
            {
                return Err(crate::error::Error::InvalidInput(
                    "physical dispatch ticket row inputs cross an exact request or plan fence"
                        .to_string(),
                ));
            }
            if scheduled.is_prefill != (phase == ExecutionPhase::Prefill) {
                return Err(crate::error::Error::InvalidInput(
                    "physical dispatch ticket phase disagrees with its scheduled row".to_string(),
                ));
            }
            if !sessions.insert(row.session.clone()) || !plans.insert(row.plan_id) {
                return Err(crate::error::Error::InvalidInput(
                    "physical dispatch ticket contains a duplicate session or plan quantum"
                        .to_string(),
                ));
            }
            if let Some(binding) = request.execution_adapter_binding() {
                let adapter = binding.key_for_stage(physical_batch.lane.stage_id)?;
                if adapter.execution_group_id != physical_batch.lane.execution_group
                    || adapter.model_instance_id != physical_batch.lane.model_instance
                    || adapter.adapter_instance_id != physical_batch.lane.adapter_instance
                    || adapter.adapter_abi_revision != physical_batch.lane.adapter_abi
                    || adapter.capability_id != physical_batch.lane.capability_id
                {
                    return Err(crate::error::Error::InvalidInput(
                        "physical dispatch ticket crossed its loaded adapter/model fence"
                            .to_string(),
                    ));
                }
            }
        }

        let rows = physical_batch
            .rows
            .iter()
            .map(|row| ((row.session.clone(), row.plan_id), row))
            .collect::<HashMap<_, _>>();
        let mut attached = HashSet::with_capacity(managed_cache_reservations.len());
        for reservation in &managed_cache_reservations {
            let key = (reservation.session.clone(), reservation.txn_id);
            if !attached.insert(key.clone()) {
                return Err(crate::error::Error::InvalidInput(
                    "physical dispatch ticket contains a duplicate managed-cache reservation"
                        .to_string(),
                ));
            }
            let row = rows.get(&key).ok_or_else(|| {
                crate::error::Error::InvalidInput(
                    "physical dispatch ticket contains a foreign managed-cache reservation"
                        .to_string(),
                )
            })?;
            reservation.validate_for_row(row)?;
            if row.managed_cache.as_ref() != Some(reservation) {
                return Err(crate::error::Error::InvalidInput(
                    "physical dispatch managed-cache metadata differs from its row envelope"
                        .to_string(),
                ));
            }
        }
        if rows.values().any(|row| {
            row.managed_cache.as_ref().is_some_and(|reservation| {
                !attached.contains(&(reservation.session.clone(), reservation.txn_id))
            })
        }) {
            return Err(crate::error::Error::InvalidInput(
                "physical dispatch ticket omitted a row managed-cache reservation".to_string(),
            ));
        }

        Ok(Self {
            phase,
            physical_batch,
            requests,
            scheduled,
            output_visibility,
            managed_cache_reservations,
            launch_policy,
        })
    }

    pub(super) const fn phase(&self) -> ExecutionPhase {
        self.phase
    }

    pub(super) fn physical_batch(&self) -> &PhysicalBatch {
        &self.physical_batch
    }

    pub(super) fn scheduled(&self) -> &[ScheduledRequest] {
        &self.scheduled
    }

    pub(super) const fn output_visibility(&self) -> OutputVisibility {
        self.output_visibility
    }

    pub(super) fn managed_cache_reservations(&self) -> &[super::ManagedCacheReservation] {
        &self.managed_cache_reservations
    }

    pub(super) const fn launch_policy(&self) -> PhysicalLaunchPolicy {
        self.launch_policy
    }
}

/// Immutable work detached from the mutable engine state.
pub(super) struct PreparedEngineStep {
    executor: UnifiedExecutor,
    dispatches: Vec<PreparedPhysicalDispatch>,
    physical_execution_mode: PhysicalExecutionMode,
    max_physical_in_flight: PhysicalInFlightLimit,
    recovery: PreparedStepRecovery,
}

/// Exact recovery fence retained outside the runner task. The caller registers
/// the runner before spawning it, and every nested physical launch registers
/// before its own spawn, so cancellation cannot make recovery race native work.
#[derive(Clone)]
pub(super) struct PreparedStepRecovery {
    batch_ids: Arc<[super::BatchId]>,
    task_drain: PhysicalTaskDrainTracker,
}

impl PreparedStepRecovery {
    pub(super) fn batch_ids(&self) -> &[super::BatchId] {
        &self.batch_ids
    }

    pub(super) fn register_runner(&self) -> PhysicalTaskDrainRegistration {
        self.task_drain.register(&self.batch_ids)
    }

    pub(super) async fn wait_for_task_drain(&self) {
        self.task_drain.wait_for(&self.batch_ids).await;
    }
}

#[derive(Clone, Default)]
struct PhysicalTaskDrainTracker {
    inner: Arc<PhysicalTaskDrainState>,
}

#[derive(Default)]
struct PhysicalTaskDrainState {
    active: Mutex<HashMap<super::BatchId, usize>>,
    changed: Notify,
}

pub(super) struct PhysicalTaskDrainRegistration {
    tracker: PhysicalTaskDrainTracker,
    batch_ids: Arc<[super::BatchId]>,
}

impl PhysicalTaskDrainTracker {
    fn register(&self, batch_ids: &[super::BatchId]) -> PhysicalTaskDrainRegistration {
        let batch_ids = batch_ids.iter().copied().collect::<HashSet<_>>();
        {
            let mut active = self
                .inner
                .active
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for batch_id in &batch_ids {
                *active.entry(*batch_id).or_default() += 1;
            }
        }
        PhysicalTaskDrainRegistration {
            tracker: self.clone(),
            batch_ids: batch_ids.into_iter().collect::<Vec<_>>().into(),
        }
    }

    async fn wait_for(&self, batch_ids: &[super::BatchId]) {
        loop {
            let changed = self.inner.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let drained = {
                let active = self
                    .inner
                    .active
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                batch_ids
                    .iter()
                    .all(|batch_id| !active.contains_key(batch_id))
            };
            if drained {
                return;
            }
            changed.await;
        }
    }
}

impl Drop for PhysicalTaskDrainRegistration {
    fn drop(&mut self) {
        {
            let mut active = self
                .tracker
                .inner
                .active
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for batch_id in self.batch_ids.iter() {
                let remove = active.get_mut(batch_id).is_some_and(|count| {
                    *count = count.saturating_sub(1);
                    *count == 0
                });
                if remove {
                    active.remove(batch_id);
                }
            }
        }
        self.tracker.inner.changed.notify_waiters();
    }
}

impl PreparedEngineStep {
    pub(super) fn new(
        executor: UnifiedExecutor,
        dispatches: Vec<PreparedPhysicalDispatch>,
    ) -> Self {
        Self::with_execution_policy(
            executor,
            dispatches,
            PhysicalExecutionMode::Serial,
            PhysicalInFlightLimit::default(),
        )
    }

    pub(super) fn with_execution_policy(
        executor: UnifiedExecutor,
        dispatches: Vec<PreparedPhysicalDispatch>,
        physical_execution_mode: PhysicalExecutionMode,
        max_physical_in_flight: PhysicalInFlightLimit,
    ) -> Self {
        let batch_ids = dispatches
            .iter()
            .map(|dispatch| dispatch.physical_batch.batch_id)
            .collect::<Vec<_>>();
        debug_assert_eq!(
            batch_ids.iter().copied().collect::<HashSet<_>>().len(),
            batch_ids.len(),
            "prepared engine step contains duplicate physical batch IDs"
        );
        Self {
            executor,
            dispatches,
            physical_execution_mode,
            max_physical_in_flight,
            recovery: PreparedStepRecovery {
                batch_ids: batch_ids.into(),
                task_drain: PhysicalTaskDrainTracker::default(),
            },
        }
    }

    pub(super) fn recovery(&self) -> PreparedStepRecovery {
        self.recovery.clone()
    }
}

/// Results that can only be applied by the engine's commit phase.
#[derive(Clone)]
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
                result.managed_cache = None;
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
                        managed_cache: result.managed_cache.clone(),
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

#[derive(Clone)]
pub(super) struct ExecutedPhysicalBatch {
    pub(super) phase: ExecutionPhase,
    pub(super) physical_batch: PhysicalBatch,
    pub(super) report: PhysicalBatchReport,
    pub(super) results: Vec<ExecutorStepResult>,
    #[allow(dead_code)]
    pub(super) managed_cache_reservations: Vec<super::ManagedCacheReservation>,
}

/// The sole owner of model-forward dispatch within one engine step.
pub(super) struct ExecutionGroupRunner;

impl ExecutionGroupRunner {
    pub(super) async fn execute(
        prepared: PreparedEngineStep,
        runner_registration: PhysicalTaskDrainRegistration,
        progress_tx: mpsc::Sender<FencedStreamProgress>,
        progress_budget: Arc<StreamProgressBudget>,
    ) -> ExecutedEngineStep {
        let _runner_registration = runner_registration;
        let task_drain = prepared.recovery.task_drain.clone();
        let batches = execute_dispatches(
            &prepared.executor,
            prepared.dispatches,
            prepared.physical_execution_mode,
            prepared.max_physical_in_flight,
            &progress_tx,
            &progress_budget,
            &task_drain,
        )
        .await;

        ExecutedEngineStep { batches }
    }
}

async fn execute_dispatches(
    executor: &UnifiedExecutor,
    dispatches: Vec<PreparedPhysicalDispatch>,
    mode: PhysicalExecutionMode,
    max_physical_in_flight: PhysicalInFlightLimit,
    progress_tx: &mpsc::Sender<FencedStreamProgress>,
    progress_budget: &Arc<StreamProgressBudget>,
    task_drain: &PhysicalTaskDrainTracker,
) -> Vec<ExecutedPhysicalBatch> {
    if mode == PhysicalExecutionMode::Shadow {
        observe_shadow_launch_plan(&dispatches, max_physical_in_flight);
        return execute_dispatches_serial(executor, dispatches, progress_tx, progress_budget).await;
    }
    if mode != PhysicalExecutionMode::Concurrent
        || max_physical_in_flight.get() == 1
        || dispatches.len() < 2
    {
        return execute_dispatches_serial(executor, dispatches, progress_tx, progress_budget).await;
    }

    execute_dispatches_concurrent(
        executor,
        dispatches,
        max_physical_in_flight,
        progress_tx,
        progress_budget,
        task_drain,
    )
    .await
}

#[derive(Clone, Copy)]
struct ActivePhysicalLaunch {
    index: usize,
    execution_group: super::ExecutionGroupId,
    model_instance: super::ModelInstanceId,
    policy: PhysicalLaunchPolicy,
    units: usize,
}

fn physical_launch_units(dispatch: &PreparedPhysicalDispatch) -> usize {
    if matches!(
        dispatch.launch_policy,
        PhysicalLaunchPolicy::Concurrent { .. }
    ) && dispatch.physical_batch.mode == super::NativeBatchMode::None
    {
        dispatch.physical_batch.rows.len().max(1)
    } else {
        1
    }
}

fn launch_is_compatible(
    candidate: &PreparedPhysicalDispatch,
    active: &[ActivePhysicalLaunch],
    engine_limit: PhysicalInFlightLimit,
) -> bool {
    let candidate_units = physical_launch_units(candidate);
    let active_units = active.iter().map(|launch| launch.units).sum::<usize>();
    if candidate_units > engine_limit.get()
        || active_units.saturating_add(candidate_units) > engine_limit.get()
    {
        return false;
    }
    let execution_group = candidate.physical_batch.lane.execution_group;
    if active.iter().any(|launch| {
        launch.execution_group == execution_group
            && launch.policy == PhysicalLaunchPolicy::ExecutionGroupExclusive
    }) {
        return false;
    }

    let model_instance = candidate.physical_batch.lane.model_instance;
    match candidate.launch_policy {
        PhysicalLaunchPolicy::ExecutionGroupExclusive => active
            .iter()
            .all(|launch| launch.execution_group != execution_group),
        PhysicalLaunchPolicy::ModelExclusive => !active
            .iter()
            .any(|launch| launch.model_instance == model_instance),
        policy @ PhysicalLaunchPolicy::Concurrent { .. } => {
            let same_model = active
                .iter()
                .filter(|launch| launch.model_instance == model_instance)
                .collect::<Vec<_>>();
            let same_model_limit = same_model.iter().fold(
                policy.effective_max_in_flight_per_model(engine_limit),
                |limit, launch| {
                    limit.min(
                        launch
                            .policy
                            .effective_max_in_flight_per_model(engine_limit),
                    )
                },
            );
            !same_model
                .iter()
                .any(|launch| launch.policy == PhysicalLaunchPolicy::ModelExclusive)
                && same_model.iter().map(|launch| launch.units).sum::<usize>() + candidate_units
                    <= same_model_limit
        }
    }
}

/// Evaluate the same bounded compatibility rules used by concurrent mode while
/// retaining serial execution. Existing bounded fallback/defer counters expose
/// the dry-run result without adding request or model identities as labels.
fn observe_shadow_launch_plan(
    dispatches: &[PreparedPhysicalDispatch],
    engine_limit: PhysicalInFlightLimit,
) {
    let mut active = Vec::new();
    for (index, dispatch) in dispatches.iter().enumerate() {
        let units = physical_launch_units(dispatch);
        if units > engine_limit.get() {
            record_engine_physical_fallback(EnginePhysicalFallbackReason::ResourcePressure);
            debug!(
                batch_id = dispatch.physical_batch.batch_id.get(),
                required_units = units,
                available_units = engine_limit.get(),
                decision = "fallback",
                reason = "resource_pressure",
                "Shadow physical launch decision"
            );
            continue;
        }

        if launch_is_compatible(dispatch, &active, engine_limit) {
            active.push(ActivePhysicalLaunch {
                index,
                execution_group: dispatch.physical_batch.lane.execution_group,
                model_instance: dispatch.physical_batch.lane.model_instance,
                policy: dispatch.launch_policy,
                units,
            });
            // Shadow deliberately falls back to serial after proving that this
            // candidate would have been launchable under the effective policy.
            record_engine_physical_fallback(EnginePhysicalFallbackReason::PolicyDisabled);
            debug!(
                batch_id = dispatch.physical_batch.batch_id.get(),
                decision = "would_launch",
                "Shadow physical launch decision"
            );
        } else {
            record_engine_physical_defer(EnginePhysicalDeferReason::ExecutionCapacity);
            debug!(
                batch_id = dispatch.physical_batch.batch_id.get(),
                decision = "defer",
                reason = "execution_capacity_or_policy_conflict",
                "Shadow physical launch decision"
            );
        }
    }
}

type PhysicalLaunchFuture =
    Pin<Box<dyn Future<Output = (ActivePhysicalLaunch, ExecutedPhysicalBatch)> + Send>>;

fn failed_physical_task_completion(
    dispatch: PreparedPhysicalDispatch,
    message: String,
) -> ExecutedPhysicalBatch {
    let expected_dispatch = dispatch.physical_batch.expected_dispatch();
    let results = dispatch
        .scheduled
        .iter()
        .map(|scheduled| {
            failed_step_result(scheduled, message.clone())
                .with_dispatch(expected_dispatch)
                .with_provenance(OutcomeProvenance::failure(
                    FailureOrigin::Panic,
                    DispatchState::Started,
                ))
        })
        .collect();
    executed_batch(
        dispatch,
        results,
        Duration::ZERO,
        super::ResourceVector::zero(),
    )
}

fn rejected_physical_dispatch_completion(
    dispatch: PreparedPhysicalDispatch,
    message: String,
) -> ExecutedPhysicalBatch {
    let width = dispatch.scheduled.len().max(1);
    let results = dispatch
        .scheduled
        .iter()
        .map(|scheduled| {
            failed_step_result(scheduled, message.clone())
                .with_dispatch(super::BatchDispatch::not_dispatched(width))
                .with_provenance(OutcomeProvenance::failure(
                    FailureOrigin::DispatchCoordination,
                    DispatchState::NotStarted,
                ))
        })
        .collect();
    executed_batch(
        dispatch,
        results,
        Duration::ZERO,
        super::ResourceVector::zero(),
    )
}

async fn execute_dispatches_concurrent(
    executor: &UnifiedExecutor,
    dispatches: Vec<PreparedPhysicalDispatch>,
    engine_limit: PhysicalInFlightLimit,
    progress_tx: &mpsc::Sender<FencedStreamProgress>,
    progress_budget: &Arc<StreamProgressBudget>,
    task_drain: &PhysicalTaskDrainTracker,
) -> Vec<ExecutedPhysicalBatch> {
    let result_count = dispatches.len();
    let mut pending = dispatches
        .into_iter()
        .enumerate()
        .collect::<std::collections::VecDeque<_>>();
    let mut launches = FuturesUnordered::<PhysicalLaunchFuture>::new();
    let mut active = Vec::<ActivePhysicalLaunch>::new();
    let mut completed = vec![None; result_count];

    while !pending.is_empty() || !launches.is_empty() {
        while let Some((index, dispatch)) = pending.front() {
            if physical_launch_units(dispatch) > engine_limit.get() {
                let index = *index;
                let (_, dispatch) = pending
                    .pop_front()
                    .expect("front dispatch disappeared before rejection");
                completed[index] = Some(rejected_physical_dispatch_completion(
                    dispatch,
                    "physical dispatch width exceeds the configured launch capacity".to_string(),
                ));
                continue;
            }
            if !launch_is_compatible(dispatch, &active, engine_limit) {
                break;
            }
            let index = *index;
            let (_, dispatch) = pending
                .pop_front()
                .expect("front dispatch disappeared before launch");
            let launch = ActivePhysicalLaunch {
                index,
                execution_group: dispatch.physical_batch.lane.execution_group,
                model_instance: dispatch.physical_batch.lane.model_instance,
                policy: dispatch.launch_policy,
                units: physical_launch_units(&dispatch),
            };
            active.push(launch);
            let recovery = dispatch.clone();
            let task_registration = task_drain.register(&[dispatch.physical_batch.batch_id]);
            let task_executor = executor.clone();
            let task_progress_tx = progress_tx.clone();
            let task_progress_budget = progress_budget.clone();
            let task = tokio::spawn(async move {
                let _task_registration = task_registration;
                let mut executed = execute_dispatches_serial(
                    &task_executor,
                    vec![dispatch],
                    &task_progress_tx,
                    &task_progress_budget,
                )
                .await;
                executed
                    .pop()
                    .expect("one prepared dispatch must produce one completion")
            });
            launches.push(Box::pin(async move {
                let batch = match task.await {
                    Ok(batch) => batch,
                    Err(error) => failed_physical_task_completion(
                        recovery,
                        format!("physical dispatch task failed: {error}"),
                    ),
                };
                (launch, batch)
            }));
        }

        let Some((launch, batch)) = launches.next().await else {
            break;
        };
        active.retain(|candidate| candidate.index != launch.index);
        completed[launch.index] = Some(batch);
    }

    completed
        .into_iter()
        .enumerate()
        .map(|(index, batch)| {
            batch.unwrap_or_else(|| panic!("physical dispatch {index} produced no completion"))
        })
        .collect()
}

async fn execute_dispatches_serial(
    executor: &UnifiedExecutor,
    dispatches: Vec<PreparedPhysicalDispatch>,
    progress_tx: &mpsc::Sender<FencedStreamProgress>,
    progress_budget: &Arc<StreamProgressBudget>,
) -> Vec<ExecutedPhysicalBatch> {
    if dispatches.is_empty() {
        return Vec::new();
    }

    let mut executed = Vec::new();
    for dispatch in dispatches {
        let batch_started = Instant::now();
        if let Some(results) = pre_device_entry_results(&dispatch, Instant::now()) {
            executed.push(executed_batch(
                dispatch,
                results,
                batch_started.elapsed(),
                super::ResourceVector::zero(),
            ));
            continue;
        }
        let expected_dispatch = dispatch.physical_batch.expected_dispatch();
        let request_refs: Vec<_> = dispatch.requests.iter().map(Arc::as_ref).collect();
        let admitted = match executor
            .acquire_physical_execution(
                &dispatch.physical_batch,
                &request_refs,
                &dispatch.scheduled,
            )
            .await
        {
            Ok(PhysicalExecutionAdmissionOutcome::Admitted(admitted)) => admitted,
            Ok(PhysicalExecutionAdmissionOutcome::Cancelled) => {
                let results = pre_dispatch_all_cancelled_results(&dispatch)
                    .unwrap_or_else(|| cancelled_before_dispatch_results(&dispatch.scheduled));
                executed.push(executed_batch(
                    dispatch,
                    results,
                    batch_started.elapsed(),
                    super::ResourceVector::zero(),
                ));
                continue;
            }
            Err(error) => {
                let mut results = reconcile_executor_outputs(
                    dispatch.phase.label(),
                    &dispatch.scheduled,
                    expected_dispatch,
                    Err(error),
                );
                apply_post_dispatch_deadlines(&dispatch, Instant::now(), &mut results);
                executed.push(executed_batch(
                    dispatch,
                    results,
                    batch_started.elapsed(),
                    super::ResourceVector::zero(),
                ));
                continue;
            }
        };
        if let Some(results) = pre_device_entry_results(&dispatch, Instant::now()) {
            drop(admitted);
            executed.push(executed_batch(
                dispatch,
                results,
                batch_started.elapsed(),
                super::ResourceVector::zero(),
            ));
            continue;
        }
        let workspace = match executor.reserve_batch_workspace(&dispatch.physical_batch) {
            Ok(workspace) => workspace,
            Err(error) => {
                let batch_dispatch = super::BatchDispatch::not_dispatched(dispatch.scheduled.len());
                let results = dispatch
                    .scheduled
                    .iter()
                    .map(|scheduled| {
                        let mut result = failed_step_result(
                            scheduled,
                            format!("physical batch workspace admission failed: {error}"),
                        );
                        result.dispatch = batch_dispatch;
                        result.provenance = OutcomeProvenance::failure(
                            FailureOrigin::WorkspaceAdmission,
                            DispatchState::NotStarted,
                        );
                        result
                    })
                    .collect();
                executed.push(executed_batch(
                    dispatch,
                    results,
                    batch_started.elapsed(),
                    super::ResourceVector::zero(),
                ));
                continue;
            }
        };
        let _workspace_metrics = workspace
            .as_ref()
            .map(|_| begin_engine_physical_workspace(dispatch.physical_batch.workspace));
        if let Some(results) = pre_device_entry_results(&dispatch, Instant::now()) {
            drop(admitted);
            drop(workspace);
            executed.push(executed_batch(
                dispatch,
                results,
                batch_started.elapsed(),
                super::ResourceVector::zero(),
            ));
            continue;
        }
        let stream_bindings = match bind_stream_quantum(&dispatch, progress_tx, progress_budget) {
            Ok(bindings) => bindings,
            Err(error) => {
                drop(workspace);
                let batch_dispatch =
                    super::BatchDispatch::not_dispatched(dispatch.scheduled.len().max(1));
                let results = dispatch
                    .scheduled
                    .iter()
                    .map(|scheduled| {
                        failed_step_result(
                            scheduled,
                            format!("stream quantum binding failed: {error}"),
                        )
                        .with_dispatch(batch_dispatch)
                        .with_provenance(OutcomeProvenance::failure(
                            FailureOrigin::ExecutorValidation,
                            DispatchState::NotStarted,
                        ))
                    })
                    .collect();
                executed.push(executed_batch(
                    dispatch,
                    results,
                    batch_started.elapsed(),
                    super::ResourceVector::zero(),
                ));
                continue;
            }
        };
        if let Some(results) = pre_device_entry_results(&dispatch, Instant::now()) {
            drop(stream_bindings);
            drop(workspace);
            drop(admitted);
            executed.push(executed_batch(
                dispatch,
                results,
                batch_started.elapsed(),
                super::ResourceVector::zero(),
            ));
            continue;
        }
        let observed_workspace = dispatch.physical_batch.workspace;
        let result = executor
            .execute_admitted_physical_batch(
                admitted,
                &dispatch.physical_batch,
                &request_refs,
                &dispatch.scheduled,
            )
            .await;
        let mut results = reconcile_executor_outputs(
            dispatch.phase.label(),
            &dispatch.scheduled,
            expected_dispatch,
            result,
        );
        apply_post_dispatch_deadlines(&dispatch, Instant::now(), &mut results);
        drop(stream_bindings);
        executed.push(executed_batch(
            dispatch,
            results,
            batch_started.elapsed(),
            observed_workspace,
        ));
        drop(workspace);
    }
    executed
}

fn bind_stream_quantum(
    batch: &PreparedPhysicalDispatch,
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
    batch: &PreparedPhysicalDispatch,
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

fn pre_dispatch_all_cancelled_results(
    batch: &PreparedPhysicalDispatch,
) -> Option<Vec<ExecutorStepResult>> {
    if batch.requests.is_empty()
        || batch.requests.len() != batch.scheduled.len()
        || !batch.requests.iter().all(|request| request.is_cancelled())
    {
        return None;
    }
    Some(cancelled_before_dispatch_results(&batch.scheduled))
}

fn cancelled_before_dispatch_results(scheduled: &[ScheduledRequest]) -> Vec<ExecutorStepResult> {
    let dispatch = super::BatchDispatch::not_dispatched(scheduled.len().max(1));
    scheduled
        .iter()
        .map(|scheduled| {
            ExecutorStepResult::from_session(
                scheduled,
                ModelSessionResult::cancelled_before_dispatch(ExecutorOutput::cancelled(
                    scheduled.request_id.clone(),
                )),
            )
            .with_dispatch(dispatch)
        })
        .collect()
}

fn pre_device_entry_results(
    batch: &PreparedPhysicalDispatch,
    now: Instant,
) -> Option<Vec<ExecutorStepResult>> {
    pre_dispatch_deadline_results(batch, now).or_else(|| pre_dispatch_all_cancelled_results(batch))
}

fn apply_post_dispatch_deadlines(
    batch: &PreparedPhysicalDispatch,
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
        result.managed_cache = None;
    }
}

fn executed_batch(
    batch: PreparedPhysicalDispatch,
    results: Vec<ExecutorStepResult>,
    elapsed: Duration,
    observed_resources: super::ResourceVector,
) -> ExecutedPhysicalBatch {
    let phase = batch.phase;
    let dispatch = results
        .first()
        .map(|result| result.dispatch)
        .unwrap_or_default();
    let rows = results
        .iter()
        .map(|result| PhysicalBatchRowReport {
            execution: execution_report_from_result(result, elapsed),
            state: state_disposition(&result.disposition),
            managed_cache: result.managed_cache.clone(),
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
        managed_cache_reservations: batch.managed_cache_reservations,
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::backends::BackendKind;
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchBudget, BatchDispatch, BatchDispatchKind,
        BatchId, BatchLaneKey, ExecutionAdapterBinding, ExecutionGroupId, ExecutionMode,
        ExecutionProfile, InputRange, ModelExecutor, ModelInstanceId, NativeBatchMode,
        PhysicalBatchExecution, PhysicalDispatchError, PhysicalDispatchResult, ReadyQuantum,
        ResourceAmount, ResourceVector, SequencePhase, SessionKey, StageDescriptor, StageId,
        WorkCost, WorkUnit,
    };
    use crate::model::ModelVariant;
    use crate::runtime::InferenceCoordinator;

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

    struct ObservedPhysicalExecutor {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        delay: Duration,
    }

    struct ReverseCompletionExecutor {
        completion_order: Arc<std::sync::Mutex<Vec<BatchId>>>,
        slow_batch: BatchId,
    }

    struct BlockingDrainExecutor {
        entered: Arc<AtomicUsize>,
        exited: Arc<AtomicUsize>,
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    struct PanickingPhysicalExecutor;

    impl ModelExecutor for BlockingDrainExecutor {
        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            execution.validate().expect("test physical batch");
            self.entered.fetch_add(1, Ordering::SeqCst);
            let (released, wake) = self.release.as_ref();
            let mut released = released.lock().unwrap_or_else(|poison| poison.into_inner());
            while !*released {
                released = wake
                    .wait(released)
                    .unwrap_or_else(|poison| poison.into_inner());
            }
            self.exited.fetch_add(1, Ordering::SeqCst);
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

    impl ModelExecutor for PanickingPhysicalExecutor {
        fn execute_physical_batch(
            &self,
            _execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            panic!("intentional physical task panic")
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

    impl ModelExecutor for ReverseCompletionExecutor {
        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            execution.validate().expect("test physical batch");
            if execution.batch.batch_id == self.slow_batch {
                std::thread::sleep(Duration::from_millis(50));
            } else {
                std::thread::sleep(Duration::from_millis(5));
            }
            self.completion_order
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push(execution.batch.batch_id);
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

    impl ModelExecutor for ObservedPhysicalExecutor {
        fn execute_physical_batch(
            &self,
            execution: PhysicalBatchExecution<'_>,
        ) -> PhysicalDispatchResult {
            execution.validate().expect("test physical batch");
            let width = execution.scheduled.len().max(1);
            let active = self.active.fetch_add(width, Ordering::SeqCst) + width;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(self.delay);
            self.active.fetch_sub(width, Ordering::SeqCst);
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
    ) -> PreparedPhysicalDispatch {
        let lane = lane();
        PreparedPhysicalDispatch::new(
            ExecutionPhase::Prefill,
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
                    managed_cache: None,
                }],
                materialized_tensor_elements: 1,
                workspace: ResourceVector::zero(),
            },
            vec![Arc::new(request)],
            vec![scheduled],
            OutputVisibility::AfterQuantumCommit,
            Vec::new(),
            PhysicalLaunchPolicy::ExecutionGroupExclusive,
        )
        .unwrap()
    }

    fn prepared_batch_with_policy(
        batch_id: u64,
        request: EngineCoreRequest,
        scheduled: ScheduledRequest,
        policy: PhysicalLaunchPolicy,
    ) -> PreparedPhysicalDispatch {
        let mut dispatch = prepared_batch(batch_id, request, scheduled);
        dispatch.launch_policy = policy;
        dispatch
    }

    fn rebind_execution_group(
        mut dispatch: PreparedPhysicalDispatch,
        execution_group: u64,
    ) -> PreparedPhysicalDispatch {
        let execution_group = ExecutionGroupId::new(execution_group);
        let model_instance = ModelInstanceId::new(execution_group.get());
        dispatch.physical_batch.lane.execution_group = execution_group;
        dispatch.physical_batch.lane.model_instance = model_instance;
        for row in &mut dispatch.physical_batch.rows {
            row.lane.execution_group = execution_group;
            row.lane.model_instance = model_instance;
        }
        dispatch
    }

    fn prepared_rows_with_policy(
        batch_id: u64,
        request_prefix: &str,
        width: usize,
        policy: PhysicalLaunchPolicy,
    ) -> PreparedPhysicalDispatch {
        let lane = lane();
        let scheduled = (0..width)
            .map(|row| {
                scheduled(
                    &format!("{request_prefix}-{row}"),
                    batch_id * 10 + row as u64,
                    row as u64 + 1,
                )
            })
            .collect::<Vec<_>>();
        let requests = scheduled
            .iter()
            .map(|scheduled| {
                let mut request = EngineCoreRequest::tts("parallel rows");
                request.id = scheduled.request_id.clone();
                Arc::new(request)
            })
            .collect::<Vec<_>>();
        let rows = scheduled
            .iter()
            .map(|scheduled| ReadyQuantum {
                plan_id: scheduled.plan_id,
                session: scheduled.session_key(),
                lane: lane.clone(),
                work: scheduled.work.clone(),
                cost: WorkCost::new(1, 1, 0),
                managed_cache: None,
            })
            .collect::<Vec<_>>();
        PreparedPhysicalDispatch::new(
            ExecutionPhase::Prefill,
            PhysicalBatch {
                batch_id: BatchId::new(batch_id),
                lane,
                mode: NativeBatchMode::None,
                budget: BatchBudget {
                    max_rows: width,
                    max_logical_units: width as u64,
                    max_tensor_elements: width as u64,
                    max_workspace_bytes: 0,
                    max_padding_basis_points: 0,
                    max_formation_delay: Duration::ZERO,
                },
                rows,
                materialized_tensor_elements: width as u64,
                workspace: ResourceVector::zero(),
            },
            requests,
            scheduled,
            OutputVisibility::AfterQuantumCommit,
            Vec::new(),
            policy,
        )
        .unwrap()
    }

    fn prepared_bound_rows_with_policy(
        batch_id: u64,
        execution_group: ExecutionGroupId,
        model_instance: ModelInstanceId,
        width: usize,
        policy: PhysicalLaunchPolicy,
        cancellation: Arc<AtomicBool>,
    ) -> PreparedPhysicalDispatch {
        let mut lane = lane();
        lane.execution_group = execution_group;
        lane.model_instance = model_instance;
        let variant = ModelVariant::Kokoro82M;
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.concurrency = super::super::ConcurrencyClass::Batchable;
        profile.physical_launch_policy = policy;
        profile.max_batch_size = width;
        profile.resolved_from_loaded_model = true;
        let stage = StageDescriptor::from_execution_profile(
            lane.stage_id,
            "test.concurrent.scalar",
            &profile,
            NativeBatchMode::None,
        );
        let binding = ExecutionAdapterBinding {
            execution_group_id: lane.execution_group,
            model_instance_id: lane.model_instance,
            adapter_instance_id: lane.adapter_instance,
            adapter_abi_revision: lane.adapter_abi,
            model_variant: variant,
            capability_id: lane.capability_id.clone(),
            stages: Arc::from([stage]),
        };
        let scheduled = (0..width)
            .map(|row| {
                scheduled(
                    &format!("cancelled-wide-{row}"),
                    batch_id * 10 + row as u64,
                    row as u64 + 1,
                )
            })
            .collect::<Vec<_>>();
        let requests = scheduled
            .iter()
            .map(|scheduled| {
                let mut request = EngineCoreRequest::tts("cancelled wide FIFO waiter")
                    .with_model_variant(variant);
                request.id = scheduled.request_id.clone();
                request.set_cancellation_signal(cancellation.clone());
                request
                    .bind_execution_adapter(binding.clone())
                    .expect("test execution binding");
                Arc::new(request)
            })
            .collect::<Vec<_>>();
        let rows = scheduled
            .iter()
            .map(|scheduled| ReadyQuantum {
                plan_id: scheduled.plan_id,
                session: scheduled.session_key(),
                lane: lane.clone(),
                work: scheduled.work.clone(),
                cost: WorkCost::new(1, 1, 0),
                managed_cache: None,
            })
            .collect::<Vec<_>>();
        PreparedPhysicalDispatch::new(
            ExecutionPhase::Prefill,
            PhysicalBatch {
                batch_id: BatchId::new(batch_id),
                lane,
                mode: NativeBatchMode::None,
                budget: BatchBudget {
                    max_rows: width,
                    max_logical_units: width as u64,
                    max_tensor_elements: width as u64,
                    max_workspace_bytes: 0,
                    max_padding_basis_points: 0,
                    max_formation_delay: Duration::ZERO,
                },
                rows,
                materialized_tensor_elements: width as u64,
                workspace: ResourceVector::zero(),
            },
            requests,
            scheduled,
            OutputVisibility::AfterQuantumCommit,
            Vec::new(),
            policy,
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_mode_overlaps_certified_disjoint_physical_tickets() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active,
            max_active: max_active.clone(),
            delay: Duration::from_millis(50),
        }));
        let policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
        let mut first = EngineCoreRequest::tts("first concurrent");
        first.id = "first-concurrent".to_string();
        let mut second = EngineCoreRequest::tts("second concurrent");
        second.id = "second-concurrent".to_string();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_batch_with_policy(
                    101,
                    first,
                    scheduled("first-concurrent", 101, 1),
                    policy,
                ),
                prepared_batch_with_policy(
                    102,
                    second,
                    scheduled("second-concurrent", 102, 1),
                    policy,
                ),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(executed.batches.len(), 2);
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_mode_keeps_same_group_exclusive_tickets_serial() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active,
            max_active: max_active.clone(),
            delay: Duration::from_millis(25),
        }));
        let mut first = EngineCoreRequest::tts("first exclusive");
        first.id = "first-exclusive".to_string();
        let mut second = EngineCoreRequest::tts("second exclusive");
        second.id = "second-exclusive".to_string();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_batch(103, first, scheduled("first-exclusive", 103, 1)),
                prepared_batch(104, second, scheduled("second-exclusive", 104, 1)),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(executed.batches.len(), 2);
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_mode_overlaps_exclusive_tickets_from_distinct_groups() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active,
            max_active: max_active.clone(),
            delay: Duration::from_millis(25),
        }));
        let mut first = EngineCoreRequest::tts("first group");
        first.id = "first-group".to_string();
        let mut second = EngineCoreRequest::tts("second group");
        second.id = "second-group".to_string();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                rebind_execution_group(
                    prepared_batch(112, first, scheduled("first-group", 112, 1)),
                    11,
                ),
                rebind_execution_group(
                    prepared_batch(113, second, scheduled("second-group", 113, 1)),
                    12,
                ),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(executed.batches.len(), 2);
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn shadow_mode_observes_compatibility_but_executes_serially() {
        let before = super::super::metrics::engine_physical_execution_metrics_snapshot();
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active,
            max_active: max_active.clone(),
            delay: Duration::from_millis(10),
        }));
        let policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
        let mut first = EngineCoreRequest::tts("first shadow");
        first.id = "first-shadow".to_string();
        let mut second = EngineCoreRequest::tts("second shadow");
        second.id = "second-shadow".to_string();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_batch_with_policy(114, first, scheduled("first-shadow", 114, 1), policy),
                prepared_batch_with_policy(115, second, scheduled("second-shadow", 115, 1), policy),
            ],
            PhysicalExecutionMode::Shadow,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;
        let after = super::super::metrics::engine_physical_execution_metrics_snapshot();

        assert_eq!(executed.batches.len(), 2);
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
        assert!(
            after.fallbacks.policy_disabled >= before.fallbacks.policy_disabled.saturating_add(2)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn shadow_mode_records_same_group_policy_deferral() {
        let before = super::super::metrics::engine_physical_execution_metrics_snapshot();
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
            delay: Duration::ZERO,
        }));
        let mut first = EngineCoreRequest::tts("first shadow exclusive");
        first.id = "first-shadow-exclusive".to_string();
        let mut second = EngineCoreRequest::tts("second shadow exclusive");
        second.id = "second-shadow-exclusive".to_string();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_batch(116, first, scheduled("first-shadow-exclusive", 116, 1)),
                prepared_batch(117, second, scheduled("second-shadow-exclusive", 117, 1)),
            ],
            PhysicalExecutionMode::Shadow,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;
        let after = super::super::metrics::engine_physical_execution_metrics_snapshot();

        assert_eq!(executed.batches.len(), 2);
        assert!(
            after.defers.execution_capacity >= before.defers.execution_capacity.saturating_add(1)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_scalar_tickets_respect_weighted_row_capacity() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active,
            max_active: max_active.clone(),
            delay: Duration::from_millis(40),
        }));
        let policy = PhysicalLaunchPolicy::concurrent(3).unwrap();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_rows_with_policy(105, "weighted-two", 2, policy),
                prepared_rows_with_policy(106, "weighted-one", 1, policy),
                prepared_rows_with_policy(107, "weighted-tail", 2, policy),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(3).unwrap(),
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(executed.batches.len(), 3);
        assert_eq!(max_active.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn exclusive_wide_ticket_runs_alone_instead_of_being_rejected() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(ObservedPhysicalExecutor {
            active,
            max_active: max_active.clone(),
            delay: Duration::from_millis(20),
        }));
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_rows_with_policy(
                    108,
                    "wide-exclusive",
                    4,
                    PhysicalLaunchPolicy::ExecutionGroupExclusive,
                ),
                prepared_rows_with_policy(
                    109,
                    "narrow-exclusive",
                    1,
                    PhysicalLaunchPolicy::ExecutionGroupExclusive,
                ),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(executed.batches.len(), 2);
        assert!(executed
            .batches
            .iter()
            .all(|batch| batch.report.dispatch.kind != BatchDispatchKind::NotDispatched));
        assert_eq!(max_active.load(Ordering::SeqCst), 4);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reverse_physical_completion_preserves_prepared_commit_order() {
        let completion_order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(ReverseCompletionExecutor {
            completion_order: completion_order.clone(),
            slow_batch: BatchId::new(110),
        }));
        let policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_rows_with_policy(110, "slow-first", 1, policy),
                prepared_rows_with_policy(111, "fast-second", 1, policy),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );

        let executed = execute_prepared(prepared).await;

        assert_eq!(
            *completion_order
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()),
            vec![BatchId::new(111), BatchId::new(110)]
        );
        assert_eq!(
            executed
                .batches
                .iter()
                .map(|batch| batch.physical_batch.batch_id)
                .collect::<Vec<_>>(),
            vec![BatchId::new(110), BatchId::new(111)]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn outer_runner_abort_drains_detached_launches_before_workspace_and_permit_release() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 2, 4));
        let authority = coordinator.resource_authority();
        let admission = coordinator.physical_execution_admission();
        let execution_group = coordinator.execution_group_id();
        let model_instance = ModelInstanceId::new(991);
        let policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
        let entered = Arc::new(AtomicUsize::new(0));
        let exited = Arc::new(AtomicUsize::new(0));
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let executor = UnifiedExecutor::new_for_test_with_physical_context(
            Box::new(BlockingDrainExecutor {
                entered: entered.clone(),
                exited: exited.clone(),
                release: release.clone(),
            }),
            BackendKind::Cpu,
            authority.clone(),
            admission.clone(),
        );
        let cancellation = Arc::new(AtomicBool::new(false));
        let mut dispatches = [301, 302]
            .into_iter()
            .map(|batch_id| {
                prepared_bound_rows_with_policy(
                    batch_id,
                    execution_group,
                    model_instance,
                    1,
                    policy,
                    cancellation.clone(),
                )
            })
            .collect::<Vec<_>>();
        for dispatch in &mut dispatches {
            dispatch.physical_batch.workspace.host_bytes = ResourceAmount::Known(8);
            dispatch.physical_batch.budget.max_workspace_bytes = 8;
            dispatch.physical_batch.rows[0].cost = WorkCost::new(1, 1, 8);
        }
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            dispatches,
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );
        let recovery = prepared.recovery();
        let runner_registration = recovery.register_runner();
        let (progress_tx, _progress_rx) = mpsc::channel(8);
        let runner = tokio::spawn(async move {
            ExecutionGroupRunner::execute(
                prepared,
                runner_registration,
                progress_tx,
                StreamProgressBudget::new(1024),
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(1), async {
            while entered.load(Ordering::SeqCst) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both physical launches did not enter native execution");
        assert_eq!(authority.snapshot().reservations, 2);

        runner.abort();
        let runner_error = match runner.await {
            Ok(_) => panic!("aborted runner unexpectedly completed"),
            Err(error) => error,
        };
        assert!(runner_error.is_cancelled());
        let drain_recovery = recovery.clone();
        let drain = tokio::spawn(async move { drain_recovery.wait_for_task_drain().await });
        tokio::task::yield_now().await;
        assert!(
            !drain.is_finished(),
            "drain completed while native calls ran"
        );
        assert_eq!(exited.load(Ordering::SeqCst), 0);
        assert_eq!(authority.snapshot().reservations, 2);

        let waiter = tokio::spawn(async move {
            admission
                .acquire_dispatch(
                    execution_group,
                    model_instance,
                    policy,
                    NativeBatchMode::None,
                    1,
                    None,
                )
                .await
        });
        tokio::task::yield_now().await;
        assert!(
            !waiter.is_finished(),
            "physical permits were released early"
        );

        let (released, wake) = release.as_ref();
        *released.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
        tokio::time::timeout(Duration::from_secs(1), drain)
            .await
            .expect("physical task drain did not finish")
            .expect("drain task panicked");
        assert_eq!(exited.load(Ordering::SeqCst), 2);
        assert_eq!(authority.snapshot().reservations, 0);
        drop(
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .expect("permit did not become available after drain")
                .expect("permit waiter panicked")
                .expect("permit waiter failed"),
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn panicking_physical_tasks_leave_the_prepared_recovery_fence_drained() {
        let executor = UnifiedExecutor::new_for_test(Box::new(PanickingPhysicalExecutor));
        let policy = PhysicalLaunchPolicy::concurrent(2).unwrap();
        let prepared = PreparedEngineStep::with_execution_policy(
            executor,
            vec![
                prepared_rows_with_policy(303, "panic-a", 1, policy),
                prepared_rows_with_policy(304, "panic-b", 1, policy),
            ],
            PhysicalExecutionMode::Concurrent,
            PhysicalInFlightLimit::new(2).unwrap(),
        );
        let recovery = prepared.recovery();
        let runner_registration = recovery.register_runner();
        let (progress_tx, _progress_rx) = mpsc::channel(8);
        let executed = tokio::spawn(async move {
            ExecutionGroupRunner::execute(
                prepared,
                runner_registration,
                progress_tx,
                StreamProgressBudget::new(1024),
            )
            .await
        })
        .await
        .expect("child physical panics must be reconciled by the runner");

        tokio::time::timeout(Duration::from_millis(100), recovery.wait_for_task_drain())
            .await
            .expect("panic recovery left a physical task registered");
        assert_eq!(executed.batches.len(), 2);
        assert!(executed.batches.iter().all(|batch| {
            batch.results.iter().all(|result| {
                result.provenance.failure_origin == Some(FailureOrigin::Panic)
                    && result.provenance.dispatch_state == DispatchState::Started
            })
        }));
    }

    #[test]
    fn prepared_physical_dispatch_rejects_duplicate_session_or_plan_quantum() {
        let build = |batch_id: u64, scheduled: Vec<ScheduledRequest>| {
            let lane = lane();
            let requests = scheduled
                .iter()
                .map(|scheduled| {
                    let mut request = EngineCoreRequest::tts("duplicate ticket fence");
                    request.id = scheduled.request_id.clone();
                    Arc::new(request)
                })
                .collect::<Vec<_>>();
            let rows = scheduled
                .iter()
                .map(|scheduled| ReadyQuantum {
                    plan_id: scheduled.plan_id,
                    session: scheduled.session_key(),
                    lane: lane.clone(),
                    work: scheduled.work.clone(),
                    cost: WorkCost::new(1, 1, 0),
                    managed_cache: None,
                })
                .collect::<Vec<_>>();
            PreparedPhysicalDispatch::new(
                ExecutionPhase::Prefill,
                PhysicalBatch {
                    batch_id: BatchId::new(batch_id),
                    lane,
                    mode: NativeBatchMode::None,
                    budget: BatchBudget {
                        max_rows: 2,
                        max_logical_units: 2,
                        max_tensor_elements: 2,
                        max_workspace_bytes: 0,
                        max_padding_basis_points: 0,
                        max_formation_delay: Duration::ZERO,
                    },
                    rows,
                    materialized_tensor_elements: 2,
                    workspace: ResourceVector::zero(),
                },
                requests,
                scheduled,
                OutputVisibility::AfterQuantumCommit,
                Vec::new(),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
            )
            .err()
            .expect("duplicate ticket must be rejected")
        };

        let duplicate_session = build(
            90,
            vec![
                scheduled("same-session", 1, 7),
                scheduled("same-session", 2, 7),
            ],
        );
        assert!(duplicate_session
            .to_string()
            .contains("duplicate session or plan"));

        let duplicate_plan = build(
            91,
            vec![scheduled("plan-a", 3, 8), scheduled("plan-b", 3, 9)],
        );
        assert!(duplicate_plan
            .to_string()
            .contains("duplicate session or plan"));
    }

    async fn execute_prepared(prepared: PreparedEngineStep) -> ExecutedEngineStep {
        let (progress_tx, _progress_rx) = mpsc::channel(64);
        let runner_registration = prepared.recovery().register_runner();
        ExecutionGroupRunner::execute(
            prepared,
            runner_registration,
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
        let prepared =
            PreparedEngineStep::new(executor, vec![prepared_batch(9, request, scheduled)]);
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
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::temporary_workspace(1),
        };
        let prepared = PreparedEngineStep::new(
            executor,
            vec![PreparedPhysicalDispatch::new(
                ExecutionPhase::Prefill,
                physical_batch,
                vec![Arc::new(request)],
                vec![scheduled],
                OutputVisibility::AfterQuantumCommit,
                Vec::new(),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
            )
            .unwrap()],
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
    async fn queued_engine_dispatch_does_not_reserve_workspace_before_physical_admission() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let authority = coordinator.resource_authority();
        let admission = coordinator.physical_execution_admission();
        let group = coordinator.execution_group_id();
        let blocker = admission
            .acquire_dispatch(
                group,
                ModelInstanceId::new(700),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
                NativeBatchMode::None,
                1,
                None,
            )
            .await
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test_with_physical_context(
            Box::new(SleepingPhysicalExecutor {
                physical_calls: calls.clone(),
                delay: Duration::ZERO,
            }),
            BackendKind::Cpu,
            authority.clone(),
            admission,
        );
        let request = EngineCoreRequest::tts("workspace after permit");
        let request_id = request.id.clone();
        let scheduled = scheduled(&request_id, 700, 1);
        let mut dispatch =
            rebind_execution_group(prepared_batch(700, request, scheduled), group.get());
        dispatch.physical_batch.workspace.host_bytes = ResourceAmount::Known(8);
        dispatch.physical_batch.rows[0].cost = WorkCost::new(1, 1, 8);
        let task = tokio::spawn(execute_prepared(PreparedEngineStep::new(
            executor,
            vec![dispatch],
        )));

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!task.is_finished());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(authority.snapshot().reservations, 0);

        drop(blocker);
        let executed = task.await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(executed.batches.len(), 1);
    }

    #[tokio::test]
    async fn deadline_expiring_while_queued_for_a_permit_never_enters_the_executor() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let admission = coordinator.physical_execution_admission();
        let group = coordinator.execution_group_id();
        let blocker = admission
            .acquire_dispatch(
                group,
                ModelInstanceId::new(710),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
                NativeBatchMode::None,
                1,
                None,
            )
            .await
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test_with_physical_context(
            Box::new(SleepingPhysicalExecutor {
                physical_calls: calls.clone(),
                delay: Duration::ZERO,
            }),
            BackendKind::Cpu,
            coordinator.resource_authority(),
            admission,
        );
        let mut request = EngineCoreRequest::tts("permit deadline");
        request.deadline = Some(Instant::now() + Duration::from_millis(20));
        let request_id = request.id.clone();
        let scheduled = scheduled(&request_id, 710, 1);
        let dispatch = rebind_execution_group(prepared_batch(710, request, scheduled), group.get());
        let executed = execute_prepared(PreparedEngineStep::new(executor, vec![dispatch])).await;

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            executed.batches[0].results[0].disposition,
            ExecutionDisposition::Finished(FinishReason::TimedOut)
        );
        assert_eq!(
            executed.batches[0].results[0].provenance,
            OutcomeProvenance::deadline(DeadlinePhase::DispatchWait, DispatchState::NotStarted)
        );
        drop(blocker);
    }

    #[tokio::test]
    async fn fully_cancelled_batch_queued_for_a_permit_never_enters_the_executor() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let admission = coordinator.physical_execution_admission();
        let group = coordinator.execution_group_id();
        let blocker = admission
            .acquire_dispatch(
                group,
                ModelInstanceId::new(720),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
                NativeBatchMode::None,
                1,
                None,
            )
            .await
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test_with_physical_context(
            Box::new(SleepingPhysicalExecutor {
                physical_calls: calls.clone(),
                delay: Duration::ZERO,
            }),
            BackendKind::Cpu,
            coordinator.resource_authority(),
            admission,
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut request = EngineCoreRequest::tts("permit cancellation");
        request.set_cancellation_signal(cancelled.clone());
        let request_id = request.id.clone();
        let scheduled = scheduled(&request_id, 720, 1);
        let dispatch = rebind_execution_group(prepared_batch(720, request, scheduled), group.get());
        let task = tokio::spawn(execute_prepared(PreparedEngineStep::new(
            executor,
            vec![dispatch],
        )));
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancelled.store(true, Ordering::Release);
        drop(blocker);
        let executed = task.await.unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            executed.batches[0].results[0].disposition,
            ExecutionDisposition::Finished(FinishReason::Cancelled)
        );
        assert_eq!(
            executed.batches[0].results[0].dispatch.kind,
            BatchDispatchKind::NotDispatched
        );
    }

    #[tokio::test]
    async fn fully_cancelled_wide_waiter_leaves_fifo_before_capacity_releases() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 3, 8));
        let admission = coordinator.physical_execution_admission();
        let group = coordinator.execution_group_id();
        let policy = PhysicalLaunchPolicy::concurrent(3).unwrap();
        let blocker = admission
            .acquire_dispatch(
                group,
                ModelInstanceId::new(740),
                policy,
                NativeBatchMode::None,
                1,
                None,
            )
            .await
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test_with_physical_context(
            Box::new(SleepingPhysicalExecutor {
                physical_calls: calls.clone(),
                delay: Duration::ZERO,
            }),
            BackendKind::Cpu,
            coordinator.resource_authority(),
            admission.clone(),
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let wide_dispatch = prepared_bound_rows_with_policy(
            741,
            group,
            ModelInstanceId::new(741),
            3,
            policy,
            cancelled.clone(),
        );
        let wide = tokio::spawn(execute_prepared(PreparedEngineStep::new(
            executor,
            vec![wide_dispatch],
        )));
        tokio::task::yield_now().await;

        let follower_admission = admission.clone();
        let follower = tokio::spawn(async move {
            follower_admission
                .acquire_dispatch(
                    group,
                    ModelInstanceId::new(742),
                    policy,
                    NativeBatchMode::None,
                    1,
                    None,
                )
                .await
        });
        tokio::task::yield_now().await;
        assert!(
            !follower.is_finished(),
            "the live narrow follower must initially remain behind the wide FIFO waiter"
        );

        cancelled.store(true, Ordering::Release);
        let follower = tokio::time::timeout(Duration::from_millis(250), follower)
            .await
            .expect("cancelled wide waiter should leave FIFO within the polling bound")
            .unwrap()
            .unwrap();
        assert_eq!(
            coordinator.snapshot().active_executions,
            2,
            "the follower acquires alongside the still-held original blocker"
        );
        let executed = tokio::time::timeout(Duration::from_millis(250), wide)
            .await
            .expect("cancelled engine waiter")
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(executed.batches[0].results.iter().all(|result| {
            result.disposition == ExecutionDisposition::Finished(FinishReason::Cancelled)
                && result.dispatch.kind == BatchDispatchKind::NotDispatched
        }));

        drop((follower, blocker));
        assert_eq!(coordinator.snapshot().active_executions, 0);
    }

    #[tokio::test]
    async fn one_cancelled_peer_does_not_cancel_a_live_physical_batch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = UnifiedExecutor::new_for_test(Box::new(SleepingPhysicalExecutor {
            physical_calls: calls.clone(),
            delay: Duration::ZERO,
        }));
        let cancelled = Arc::new(AtomicBool::new(true));
        let mut cancelled_request = EngineCoreRequest::tts("cancelled peer");
        cancelled_request.id = "cancelled-peer".to_string();
        cancelled_request.set_cancellation_signal(cancelled);
        let mut live_request = EngineCoreRequest::tts("live peer");
        live_request.id = "live-peer-cancel".to_string();
        let cancelled_scheduled = scheduled("cancelled-peer", 730, 1);
        let live_scheduled = scheduled("live-peer-cancel", 731, 1);
        let lane = lane();
        let physical_batch = PhysicalBatch {
            batch_id: BatchId::new(730),
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
            rows: [&cancelled_scheduled, &live_scheduled]
                .into_iter()
                .map(|scheduled| ReadyQuantum {
                    plan_id: scheduled.plan_id,
                    session: scheduled.session_key(),
                    lane: lane.clone(),
                    work: scheduled.work.clone(),
                    cost: WorkCost::new(1, 1, 0),
                    managed_cache: None,
                })
                .collect(),
            materialized_tensor_elements: 2,
            workspace: ResourceVector::zero(),
        };
        let dispatch = PreparedPhysicalDispatch::new(
            ExecutionPhase::Prefill,
            physical_batch,
            vec![Arc::new(cancelled_request), Arc::new(live_request)],
            vec![cancelled_scheduled, live_scheduled],
            OutputVisibility::AfterQuantumCommit,
            Vec::new(),
            PhysicalLaunchPolicy::ExecutionGroupExclusive,
        )
        .unwrap();
        let executed = execute_prepared(PreparedEngineStep::new(executor, vec![dispatch])).await;

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(executed.batches[0]
            .results
            .iter()
            .all(|result| result.dispatch.kind != BatchDispatchKind::NotDispatched));
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
                    managed_cache: None,
                })
                .collect(),
            materialized_tensor_elements: 2,
            workspace: ResourceVector::zero(),
        };
        let prepared = PreparedEngineStep::new(
            executor,
            vec![PreparedPhysicalDispatch::new(
                ExecutionPhase::Prefill,
                physical_batch,
                vec![Arc::new(expired_request), Arc::new(live_request)],
                vec![expired, live],
                OutputVisibility::AfterQuantumCommit,
                Vec::new(),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
            )
            .unwrap()],
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
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        };
        let prepared = PreparedEngineStep::new(
            executor,
            vec![PreparedPhysicalDispatch::new(
                ExecutionPhase::Prefill,
                physical_batch,
                vec![Arc::new(request)],
                vec![scheduled],
                OutputVisibility::AfterQuantumCommit,
                Vec::new(),
                PhysicalLaunchPolicy::ExecutionGroupExclusive,
            )
            .unwrap()],
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
