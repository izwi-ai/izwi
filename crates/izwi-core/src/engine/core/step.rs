//! One engine step: prepare, execute and commit, with incremental progress.

use super::*;

impl EngineCore {
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
        self.execute_and_commit_prepared_with_progress(prepared)
            .await
    }

    pub(super) async fn execute_and_commit_prepared_with_progress(
        &mut self,
        prepared: PreparedEngineStep,
    ) -> Result<Vec<EngineOutput>> {
        let (progress_tx, mut progress_rx) = mpsc::channel(STREAM_PROGRESS_QUEUE_CAPACITY);
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel();
        let progress_budget = StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES);
        let recovery = prepared.recovery();
        debug_assert!(self.pending_runner_recovery.is_none());
        self.pending_runner_recovery = Some(recovery.clone());
        let runner_registration = recovery.register_runner();
        let mut runner = tokio::spawn(async move {
            ExecutionGroupRunner::execute(
                prepared,
                runner_registration,
                progress_tx,
                progress_budget,
                Some(completion_tx),
            )
            .await
        });
        let (mut deliveries, mut delivery_failures) = IncrementalStreamDeliveryWorkers::new();
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
                            recovery.wait_for_task_drain().await;
                            self.rollback_in_flight_dispatches(recovery.batch_ids());
                            self.pending_runner_recovery = None;
                            std::panic::resume_unwind(error.into_panic())
                        }
                        Err(error) => {
                            recovery.wait_for_task_drain().await;
                            self.rollback_in_flight_dispatches(recovery.batch_ids());
                            self.pending_runner_recovery = None;
                            return Err(Error::InferenceError(format!(
                                "execution group task was cancelled: {error}"
                            )));
                        }
                    }
                    runner_finished = true;
                }
                progress = progress_rx.recv(), if !progress_closed => {
                    match progress {
                        Some(progress) => self.enqueue_incremental_progress(
                            progress,
                            &mut failures,
                            &mut deliveries,
                        ),
                        None => progress_closed = true,
                    }
                }
                completion = completion_rx.recv(), if !completion_closed => {
                    match completion {
                        Some(batch) => {
                            let batch_id = batch.physical_batch.batch_id;
                            let outputs = self.commit_completed_dispatch_with_progress(
                                batch,
                                &mut progress_rx,
                                &mut delivery_failures,
                                &mut failures,
                                &mut deliveries,
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
                .commit_completed_dispatch_with_progress(
                    batch,
                    &mut progress_rx,
                    &mut delivery_failures,
                    &mut failures,
                    &mut deliveries,
                )
                .await?;
            completed_outputs.insert(batch_id, outputs);
        }
        while let Some(progress) = progress_rx.recv().await {
            self.enqueue_incremental_progress(progress, &mut failures, &mut deliveries);
        }
        for failure in deliveries.finish().await {
            self.cancel_failed_stream(&failure);
            failures.entry(failure.session.clone()).or_insert(failure);
        }
        while let Ok(failure) = delivery_failures.try_recv() {
            self.cancel_failed_stream(&failure);
            failures.entry(failure.session.clone()).or_insert(failure);
        }

        let tail = self
            .commit_step(ExecutedEngineStep {
                batches: Vec::new(),
            })
            .await?;
        let mut tail_outputs = tail.outputs;
        let failed_streams = deliver_committed_streams(tail.stream_deliveries).await;
        self.reconcile_stream_delivery_failures(&mut tail_outputs, failed_streams)
            .await;

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
        self.pending_runner_recovery = None;
        Ok(outputs)
    }

    pub(super) async fn commit_completed_dispatch_with_progress(
        &mut self,
        batch: super::super::execution_group::ExecutedPhysicalBatch,
        progress_rx: &mut mpsc::Receiver<FencedStreamProgress>,
        delivery_failures: &mut mpsc::UnboundedReceiver<StreamDeliveryFailure>,
        failures: &mut HashMap<super::super::SessionKey, StreamDeliveryFailure>,
        deliveries: &mut IncrementalStreamDeliveryWorkers,
    ) -> Result<Vec<EngineOutput>> {
        while let Ok(progress) = progress_rx.try_recv() {
            self.enqueue_incremental_progress(progress, failures, deliveries);
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
        let mut executed = ExecutedEngineStep {
            batches: vec![batch],
        };
        executed.apply_stream_delivery_failures(&dispatch_failures);
        let committed = self.commit_step(executed).await?;
        let mut outputs = committed.outputs;
        let failed_streams = deliver_committed_streams(committed.stream_deliveries).await;
        self.reconcile_stream_delivery_failures(&mut outputs, failed_streams)
            .await;
        Ok(outputs)
    }

    #[cfg(test)]
    pub(super) async fn execute_prepared_with_progress(
        &mut self,
        prepared: PreparedEngineStep,
    ) -> Result<ExecutedEngineStep> {
        let (progress_tx, mut progress_rx) = mpsc::channel(STREAM_PROGRESS_QUEUE_CAPACITY);
        let progress_budget = StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES);
        let recovery = prepared.recovery();
        let runner_registration = recovery.register_runner();
        let mut runner = tokio::spawn(async move {
            ExecutionGroupRunner::execute(
                prepared,
                runner_registration,
                progress_tx,
                progress_budget,
                None,
            )
            .await
        });
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
                            recovery.wait_for_task_drain().await;
                            self.rollback_in_flight_dispatches(recovery.batch_ids());
                            std::panic::resume_unwind(error.into_panic())
                        }
                        Err(error) => {
                            recovery.wait_for_task_drain().await;
                            self.rollback_in_flight_dispatches(recovery.batch_ids());
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

    /// Prepare an immutable execution transaction under the engine state lock.
    pub(in crate::engine) async fn prepare_step(&mut self) -> Result<Option<PreparedEngineStep>> {
        self.prepare_step_for(None).await
    }

    /// [`Self::prepare_step`] for the rows `scope` serves: the whole engine,
    /// or one lane for that lane's step driver.
    pub(in crate::engine) async fn prepare_step_for(
        &mut self,
        scope: LaneScope,
    ) -> Result<Option<PreparedEngineStep>> {
        if let Some(recovery) = self.pending_runner_recovery.take() {
            recovery.wait_for_task_drain().await;
            self.rollback_in_flight_dispatches(recovery.batch_ids());
        }
        // Ensure initialized
        if !self.initialized {
            self.initialize().await?;
        }

        // Decode and prefill of one lane share one physical-quantum
        // ownership fence: a lane with a registered ticket is not polled
        // again until it commits or rolls back. Other lanes schedule.
        let lanes = LaneSelection::scoped(scope).excluding(self.busy_lanes());

        let (blocked, other_lanes): (Vec<_>, Vec<_>) = std::mem::take(&mut self.workspace_pressure)
            .into_iter()
            .partition(|(_, row)| lanes.admits(self.request_lane(&row.request_id)));
        self.workspace_pressure.extend(other_lanes);
        let blocked = blocked
            .into_iter()
            .map(|(_, row)| row)
            .filter(|row| {
                self.requests
                    .get(&row.request_id)
                    .is_some_and(|r| !r.is_cancelled())
            })
            .collect::<Vec<_>>();
        if !blocked.is_empty()
            && self
                .suspend_published_session_for_capacity(&blocked, &HashSet::new())
                .await?
        {
            // Reclaimed state changes the physical resource envelope. Keep
            // the original streams and let the protected survivor try again.
            for row in &blocked {
                self.execution_retry_attempts.remove(&row.session_key());
                self.workspace_waits.remove(&row.session_key());
            }
        }

        // Phase 1: Schedule
        self.refresh_scheduler_execution_profiles().await;
        self.reconcile_due_cleanup().await;
        let (managed_kv_cache, requests) = (&mut self.managed_kv_cache, &mut self.requests);
        let schedule_result = self.scheduler.schedule_with_prefix_probe(
            &mut |session| reprobe_managed_prefix(managed_kv_cache, requests, session),
            &lanes,
        );

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
            super::super::metrics::record_kv_blocked_requests(0);
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
        let mut contract_failures = Vec::new();
        let mut exhausted = Vec::new();
        for scheduled in &all_scheduled {
            let session = scheduled.session_key();
            match self.begin_execution_plan(scheduled).await {
                Ok(()) => {
                    self.plan_begin_attempts.remove(&session);
                }
                Err(PlanBeginError::AdapterContract(error)) => {
                    contract_failures.push((session, error));
                }
                Err(PlanBeginError::Engine(error)) => {
                    if plan_begin_attempts_exhausted(&mut self.plan_begin_attempts, &session) {
                        exhausted.push((session, error));
                        continue;
                    }
                    self.rollback_unexecuted_schedule(&all_scheduled);
                    return Err(error);
                }
            }
        }
        if !contract_failures.is_empty() || !exhausted.is_empty() {
            // Rescheduling would select the same unsupported work on every
            // step. Fail those requests now and commit their terminal outputs;
            // the rest of this schedule runs on the next step.
            self.rollback_unexecuted_schedule(&all_scheduled);
            for (session, error) in contract_failures {
                self.fail_adapter_contract_mismatch(&session, error).await;
            }
            for (session, error) in exhausted {
                self.fail_unstartable_plan(&session, error).await;
            }
            let capacity = self.config.resolved_physical_execution_capacity();
            return Ok(Some(PreparedEngineStep::with_execution_policy(
                self.executor.clone(),
                Vec::new(),
                self.config.physical_execution_mode,
                capacity.physical_launch_limit,
            )));
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

        // Rows whose plan bound a mixed stage run as one mixed batch on the
        // decode side of the candidate slot: decode rows first, then prefill
        // spans. That is the token layout the model needs, and it reserves
        // managed KV in the order the scheduler allocated the plans.
        let active_plans = &self.active_plans;
        let is_mixed = |scheduled: &super::super::scheduler::ScheduledRequest| {
            active_plans
                .get(&scheduled.plan_id)
                .is_some_and(|plan| plan.batch_mode == NativeBatchMode::Mixed)
        };
        let (mixed_decode, decode_scheduled): (Vec<_>, Vec<_>) =
            decode_scheduled.into_iter().partition(is_mixed);
        let (mixed_prefill, prefill_scheduled): (Vec<_>, Vec<_>) =
            prefill_scheduled.into_iter().partition(is_mixed);
        let mixed_scheduled = mixed_decode
            .into_iter()
            .chain(mixed_prefill)
            .collect::<Vec<_>>();
        let requests_of = |scheduled: &[super::super::scheduler::ScheduledRequest]| {
            scheduled
                .iter()
                .filter_map(|s| self.requests.get(&s.request_id).cloned())
                .collect::<Vec<Arc<EngineCoreRequest>>>()
        };
        let prefill_requests = requests_of(&prefill_scheduled);
        let decode_requests = requests_of(&decode_scheduled);
        let mixed_requests = requests_of(&mixed_scheduled);
        let decode_side_empty = decode_scheduled.is_empty() && mixed_scheduled.is_empty();

        if prefill_requests.is_empty()
            && decode_requests.is_empty()
            && mixed_requests.is_empty()
            && self.pending_terminal_outputs.is_empty()
        {
            super::super::metrics::record_kv_blocked_requests(0);
            return Ok(None);
        }

        let mut capacity_blocked = Vec::new();
        let mut transaction_deferred = Vec::new();
        let mut prepare_failures = Vec::new();
        let candidate_capacity = self
            .config
            .resolved_physical_execution_capacity()
            .candidate_dispatch_limit
            .get();
        let both_phases = !decode_side_empty && !prefill_scheduled.is_empty();
        let candidate_phase = self
            .next_single_candidate_phase
            .get(&scope)
            .copied()
            .unwrap_or(SequencePhase::Decode);
        let (decode_quota, prefill_quota) = Self::candidate_phase_quotas(
            candidate_capacity,
            decode_scheduled.len() + mixed_scheduled.len(),
            prefill_scheduled.len(),
            candidate_phase,
        );
        let mut decode_batches = Vec::new();
        let mut prefill_batches = Vec::new();

        // With multiple candidate slots, form the reserved prefill service
        // first. Decode then consumes every unused slot, preserving utilization
        // without allowing its larger runnable population to erase prefill.
        if prefill_quota > 0 {
            prefill_batches = match self.form_physical_batches(
                ExecutionPhase::Prefill,
                &prefill_requests,
                &prefill_scheduled,
                prefill_quota,
                &mut capacity_blocked,
                &mut transaction_deferred,
                &mut prepare_failures,
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
        }

        let decode_limit = candidate_capacity.saturating_sub(self.dispatches_in_flight_in(&lanes));
        if decode_quota > 0 && decode_limit > 0 {
            decode_batches = match self.form_decode_side(
                &mixed_requests,
                &mixed_scheduled,
                &decode_requests,
                &decode_scheduled,
                decode_limit,
                &mut capacity_blocked,
                &mut transaction_deferred,
                &mut prepare_failures,
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
        } else if !(decode_side_empty || candidate_capacity == 1 && both_phases) {
            transaction_deferred.extend(mixed_scheduled.iter().cloned());
            transaction_deferred.extend(decode_scheduled.iter().cloned());
        }

        // A single candidate alternates only while both phases are runnable.
        // If the preferred phase cannot form a ticket, immediately offer the
        // slot to its peer rather than idling the physical lane.
        if candidate_capacity == 1 {
            let preferred_admitted = match candidate_phase {
                SequencePhase::Decode => !decode_batches.is_empty(),
                SequencePhase::Prefill => !prefill_batches.is_empty(),
            };
            if both_phases && !preferred_admitted && self.dispatches_in_flight_in(&lanes) == 0 {
                match candidate_phase {
                    SequencePhase::Decode => {
                        prefill_batches = match self.form_physical_batches(
                            ExecutionPhase::Prefill,
                            &prefill_requests,
                            &prefill_scheduled,
                            1,
                            &mut capacity_blocked,
                            &mut transaction_deferred,
                            &mut prepare_failures,
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
                    }
                    SequencePhase::Prefill => {
                        decode_batches = match self.form_decode_side(
                            &mixed_requests,
                            &mixed_scheduled,
                            &decode_requests,
                            &decode_scheduled,
                            1,
                            &mut capacity_blocked,
                            &mut transaction_deferred,
                            &mut prepare_failures,
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
                    }
                }
            }

            let next = Self::next_candidate_preference(
                candidate_phase,
                both_phases,
                !decode_batches.is_empty(),
                !prefill_batches.is_empty(),
            );
            self.next_single_candidate_phase.insert(scope, next);
        }

        if prefill_quota == 0 && !prefill_scheduled.is_empty() && prefill_batches.is_empty() {
            transaction_deferred.extend(prefill_scheduled.iter().cloned());
        }
        if decode_quota == 0 && !decode_side_empty && decode_batches.is_empty() {
            transaction_deferred.extend(mixed_scheduled.iter().cloned());
            transaction_deferred.extend(decode_scheduled.iter().cloned());
        }
        self.rollback_unexecuted_schedule(&transaction_deferred);
        self.defer_unexecuted_schedule_for_capacity(&capacity_blocked);
        super::super::metrics::record_kv_blocked_requests(capacity_blocked.len());
        self.resolve_managed_prepare_failures(prepare_failures)
            .await;
        if decode_batches.is_empty() && prefill_batches.is_empty() {
            let deadlocked = self.claim_deadlocked_models(&capacity_blocked);
            if self
                .suspend_published_session_for_capacity(&capacity_blocked, &deadlocked)
                .await?
                || self
                    .preempt_unpublished_session_for_capacity(&capacity_blocked)
                    .await?
            {
                return Ok(None);
            }
            self.fail_claim_deadlocked_session(deadlocked).await;
        }
        if decode_batches.is_empty()
            && prefill_batches.is_empty()
            && self.pending_terminal_outputs.is_empty()
        {
            return Ok(None);
        }

        let mut dispatches = decode_batches;
        dispatches.extend(prefill_batches);
        let capacity = self.config.resolved_physical_execution_capacity();
        Ok(Some(PreparedEngineStep::with_execution_policy(
            self.executor.clone(),
            dispatches,
            self.config.physical_execution_mode,
            capacity.physical_launch_limit,
        )))
    }

    /// Commit one completed execution transaction under the engine state lock.
    pub(in crate::engine) async fn commit_step(
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
        let mut step_model_calls = 0;
        for mut batch in batches {
            let batch_id = batch.physical_batch.batch_id;
            let Some(in_flight) = self.in_flight_dispatches.remove(&batch_id) else {
                for result in &batch.results {
                    if let Err(error) = self.executor.finalize_pending_quantum(
                        result.plan_id,
                        &result.session,
                        PendingQuantumDecision::Abort,
                    ) {
                        warn!(
                            plan_id = result.plan_id,
                            request_id = %result.session.request_id,
                            error = %error,
                            "Failed to abort executor state for a stale physical completion"
                        );
                    }
                }
                warn!(
                    batch_id = batch_id.get(),
                    "Ignoring a stale or duplicate physical completion"
                );
                continue;
            };
            let active_stream_fence = self.active_stream_batches.remove(&batch_id);
            if let Err(error) = in_flight.validate_completion(&batch) {
                warn!(
                    batch_id = batch_id.get(),
                    error = %error,
                    "Rolling back a physical completion that crossed its dispatch ticket"
                );
                self.rollback_unexecuted_schedule(&in_flight.scheduled);
                continue;
            }
            if active_stream_fence != in_flight.stream_fence() {
                warn!(
                    batch_id = batch_id.get(),
                    "Rolling back a physical completion with a mismatched progress fence"
                );
                self.rollback_unexecuted_schedule(&in_flight.scheduled);
                continue;
            }
            if let Err(error) = batch
                .report
                .validate_against(&in_flight.physical_batch, &self.active_plans)
            {
                warn!(
                    batch_id = batch.physical_batch.batch_id.get(),
                    error = %error,
                    "Rejecting an invalid physical batch report before state commit"
                );
                let message = format!("invalid physical batch report: {error}");
                let dispatch_state =
                    if batch.report.dispatch.kind == super::super::BatchDispatchKind::NotDispatched {
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
                step_model_calls += model_calls_for_dispatch(batch.report.dispatch);
                if batch.phase == ExecutionPhase::Mixed {
                    let prefill_rows = in_flight
                        .scheduled
                        .iter()
                        .filter(|row| row.is_prefill)
                        .count();
                    record_engine_mixed_batch(
                        in_flight.scheduled.len() - prefill_rows,
                        prefill_rows,
                        in_flight.scheduled.iter().map(|row| row.num_tokens).sum(),
                    );
                }
            }

            for (scheduled, result) in in_flight.scheduled.iter().filter_map(|scheduled| {
                batch
                    .results
                    .iter()
                    .find(|result| result.plan_id == scheduled.plan_id)
                    .map(|result| (scheduled, result))
            }) {
                if Self::is_workspace_backpressure(result) {
                    self.adapt_workspace_batch(scheduled, batch.physical_batch.rows.len());
                }
            }

            let mut committed_service_rows = Vec::new();
            let mut committed_decode_work = Vec::new();
            let step_elapsed = batch.report.elapsed;
            for result in batch.results {
                let plan_id = result.plan_id;
                // Per-row facts key on the row's scheduled phase: a mixed
                // batch holds rows of both.
                let prefill_row = in_flight
                    .scheduled
                    .iter()
                    .find(|row| row.plan_id == plan_id)
                    .is_some_and(|row| row.is_prefill);
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
                        if prefill_row {
                            timing.prefill_ms += step_time_ms;
                            timing.prefill_steps = timing.prefill_steps.saturating_add(1);
                        } else {
                            // Use an explicit monotonic pre-execution boundary, not
                            // elapsed service time subtracted from a later timestamp.
                            timing
                                .decode_started_at
                                .get_or_insert(in_flight.registered_at);
                            timing.decode_ms += step_time_ms;
                            timing.decode_steps = timing.decode_steps.saturating_add(1);
                        }
                    }
                }
                match self.commit_executor_result(result, step_time_ms).await {
                    Some(committed) => {
                        let progressed = matches!(
                            committed.disposition,
                            ExecutionDisposition::Progress
                                | ExecutionDisposition::Yielded(_)
                                | ExecutionDisposition::Finished(_)
                        );
                        if !prefill_row && progressed {
                            committed_decode_work.extend(
                                in_flight
                                    .scheduled
                                    .iter()
                                    .filter(|row| row.plan_id == plan_id)
                                    .map(|row| (&row.request_id, &row.work)),
                            );
                        }
                        if (committed.output.tokens_processed > 0
                            || committed.output.tokens_generated > 0)
                            && progressed
                        {
                            let request_id = &committed.session.request_id;
                            let tokens = committed.output.tokens_generated;
                            let gap =
                                self.request_phase_timings
                                    .get_mut(request_id)
                                    .and_then(|timing| {
                                        timing.record_token_commit(
                                            !prefill_row,
                                            tokens,
                                            Instant::now(),
                                        )
                                    });
                            if let (Some(gap), Some(request)) = (gap, self.requests.get(request_id))
                            {
                                super::super::metrics::record_engine_inter_token_gap(
                                    request
                                        .model_variant
                                        .map_or("unknown", |variant| variant.dir_name()),
                                    request.task_type.as_str(),
                                    gap,
                                    tokens,
                                );
                            }
                            let phase = if prefill_row {
                                SequencePhase::Prefill
                            } else {
                                SequencePhase::Decode
                            };
                            committed_service_rows
                                .push((committed.session.request_id.clone(), phase));
                        } else if prefill_row
                            && matches!(
                                committed.disposition,
                                ExecutionDisposition::Yielded(super::super::YieldReason::QuantumExhausted)
                            )
                        {
                            // One operation of a resumable atomic job (a
                            // long-form ASR chunk) commits no token, but it
                            // is the waiting prefill's service: it pays the
                            // starvation debt that forced it.
                            committed_service_rows.push((
                                committed.session.request_id.clone(),
                                SequencePhase::Prefill,
                            ));
                            self.scheduler
                                .yield_prefill_to_waiting(&committed.session.request_id);
                        }
                        executor_outputs.push(committed);
                    }
                    None => record_engine_execution_outcome(provenance),
                }
            }
            self.scheduler
                .record_committed_batch_service(&committed_service_rows);
            self.scheduler
                .record_committed_decode_service(&committed_decode_work, step_elapsed);
        }
        record_engine_step_model_calls(step_model_calls);
        #[cfg(test)]
        if step_model_calls > 0 {
            self.step_model_calls.push(step_model_calls);
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
                        stream_deliveries.push(
                            CommittedStreamDelivery::new(
                                session.clone(),
                                tx,
                                request.stream_policy,
                                staged_stream_outputs,
                            )
                            .with_audio_credits(request.take_audio_output_credits()),
                        );
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

            // Process output. Only these two facts are read afterwards, so the
            // output (audio, text, logprobs) moves instead of being copied.
            let finished = exec_output.finished;
            let user_visible = Self::has_user_visible_output(&exec_output);
            let mut engine_output = self.output_processor.process_execution(
                exec_output,
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
            // ASR executors report generated audio, which is empty for
            // transcription; the prepared request records the input duration.
            engine_output.input_audio_duration_secs = self
                .requests
                .get(&request_id)
                .and_then(|request| request.prepared_asr_input_duration_secs());
            // DS9.1: the managed prefix depth this request actually ran with.
            // The scheduler's cursor copy is authoritative — it is cleared
            // when the probed prefix is lost and the row replans from zero.
            engine_output.token_stats.cached_prefix_tokens =
                self.scheduler.managed_prefix_cursor(&request_id);
            if finished {
                if let Some((_, total_generated)) = self.scheduler.get_running_info(&request_id) {
                    let resolved_total = total_generated.max(engine_output.num_tokens);
                    engine_output.num_tokens = resolved_total;
                    engine_output.token_stats.generated_tokens = resolved_total;
                }
                if let Some(request) = self.requests.get(&request_id) {
                    engine_output.finish_reason = chat_budget_finish_reason(
                        engine_output.finish_reason,
                        request.answers_chat(),
                        request.params.max_tokens,
                        engine_output.num_tokens,
                    );
                }
            }
            if user_visible {
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
                    pre_engine_ms: phase.pre_engine_ms,
                    queue_wait_ms: phase.queue_wait_ms,
                    media_decode_ms: phase.media_decode_ms,
                    normalization_ms: phase.normalization_ms,
                    prefill_ms: phase.prefill_ms,
                    decode_ms: phase.decode_ms,
                    decode_wall_ms: phase.decode_wall_ms(),
                    decode_tokens: phase.last_decode_commit_at.map(|_| phase.decode_tokens),
                    post_first_token_ms: phase.post_first_token_ms(),
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
                if let ExecutionDisposition::Failed(failure) = &disposition {
                    tracing::error!(
                        request_id = %request_id,
                        kind = ?failure.kind,
                        message = %failure.message,
                        "row failed terminally"
                    );
                }
                self.begin_terminal_release(&session, cause).await;
                self.requests.remove(&request_id);
                self.execution_profiles.remove(&request_id);
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
}
