//! Committing executor results: realtime stage authentication, terminal
//! replacement and the per-request commit of one executed dispatch.

use super::*;

impl EngineCore {
    pub(super) fn report_from_result(result: &ExecutorStepResult) -> ExecutionReport {
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

    pub(super) fn realtime_work_identity(
        work: &WorkUnit,
    ) -> Option<(super::super::RealtimeOperationId, RealtimeSubphase)> {
        match work {
            WorkUnit::RealtimePreparation { operation_id, .. } => {
                Some((*operation_id, RealtimeSubphase::Preparation))
            }
            WorkUnit::RealtimePromptPrefill {
                operation_id,
                cache_append,
                ..
            } => Some((
                *operation_id,
                RealtimeSubphase::PromptPrefill {
                    cache_append: *cache_append,
                },
            )),
            WorkUnit::RealtimeDecodeContinuation { operation_id, .. } => {
                Some((*operation_id, RealtimeSubphase::DecodeContinuation))
            }
            WorkUnit::RealtimeCompletion { operation_id } => {
                Some((*operation_id, RealtimeSubphase::Completion))
            }
            _ => None,
        }
    }

    pub(super) fn authenticate_realtime_stage_outcome(
        plan: &ExecutionPlan,
        outcome: Option<&RealtimeStageOutcome>,
        authoritative_commit: bool,
    ) -> Result<()> {
        match (Self::realtime_work_identity(&plan.work), outcome) {
            (None, None) => Ok(()),
            (None, Some(_)) => Err(Error::InferenceError(
                "non-realtime executor result carried a realtime stage outcome".into(),
            )),
            (Some(_), None) if !authoritative_commit => Ok(()),
            (Some(_), None) => Err(Error::InferenceError(
                "authoritative realtime executor result omitted its stage outcome".into(),
            )),
            (Some((operation_id, subphase)), Some(outcome))
                if outcome.plan_id == plan.plan_id
                    && outcome.operation_id == operation_id
                    && outcome.completed == subphase =>
            {
                Ok(())
            }
            (Some(_), Some(_)) => Err(Error::InferenceError(
                "realtime stage outcome crossed its exact plan/operation/subphase fence".into(),
            )),
        }
    }

    pub(super) fn replace_active_result_with_terminal(
        &mut self,
        plan: &ExecutionPlan,
        result: &mut ExecutorStepResult,
        output: ExecutorOutput,
        disposition: ExecutionDisposition,
        provenance: OutcomeProvenance,
    ) -> Result<()> {
        result.output = output;
        result.disposition = disposition;
        result.safe_point = true;
        result.provenance = provenance;
        result.staged_stream_outputs.clear();
        result.managed_cache = None;
        result.managed_cache_completions.clear();
        let report = Self::report_from_result(result);
        self.execution_trackers
            .get_mut(&plan.session.request_id)
            .ok_or_else(|| {
                Error::InferenceError("execution tracker is missing for active plan".into())
            })?
            .commit(plan, &report)
    }

    pub(super) fn canonical_failure_dispatch(
        plan: &ExecutionPlan,
        result: &ExecutorStepResult,
    ) -> (BatchDispatch, DispatchState) {
        let width = result.dispatch.width.clamp(1, plan.max_batch_size.max(1));
        if result.dispatch.kind == super::super::BatchDispatchKind::NotDispatched {
            return (
                BatchDispatch::not_dispatched(width),
                DispatchState::NotStarted,
            );
        }
        let dispatch = BatchDispatch::for_batch(plan.batch_mode, width);
        let dispatch_state = match result.provenance.dispatch_state {
            DispatchState::NotStarted | DispatchState::Started => DispatchState::Started,
            DispatchState::ProducedOutput => DispatchState::ProducedOutput,
        };
        (dispatch, dispatch_state)
    }

    pub(super) async fn commit_executor_result(
        &mut self,
        mut result: ExecutorStepResult,
        step_time_ms: f64,
    ) -> Option<CommittedExecutorOutput> {
        let Some(plan) = self.active_plans.remove(&result.plan_id) else {
            if let Err(error) = self.executor.finalize_pending_quantum(
                result.plan_id,
                &result.session,
                PendingQuantumDecision::Abort,
            ) {
                warn!(
                    plan_id = result.plan_id,
                    request_id = %result.session.request_id,
                    error = %error,
                    "Failed to abort executor state for an inactive result"
                );
            }
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

        if (self
            .executor
            .has_pending_quantum(plan.plan_id, &plan.session)
            || Self::is_workspace_backpressure(&result))
            && self
                .requests
                .get(&plan.session.request_id)
                .is_some_and(|request| request.is_cancelled())
        {
            result.output = ExecutorOutput::cancelled(plan.session.request_id.clone());
            result.disposition = ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled);
            result.provenance.failure_origin = None;
            result.provenance.deadline_phase = None;
            result.safe_point = true;
            result.staged_stream_outputs.clear();
            result.managed_cache = None;
            result.managed_cache_completions.clear();
        }

        if self.incremental_stream_sessions.contains(&plan.session)
            && !Self::is_workspace_backpressure(&result)
        {
            match &mut result.disposition {
                ExecutionDisposition::Failed(failure)
                    if failure.retry != RetryDisposition::Never =>
                {
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
                ExecutionDisposition::RestartSequence(_) => {
                    let message =
                        "sequence restart is unsafe after committed incremental stream output";
                    result.output = ExecutorOutput::error(plan.session.request_id.clone(), message);
                    result.disposition =
                        ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
                    result.safe_point = true;
                    result.provenance = OutcomeProvenance::failure(
                        FailureOrigin::ExecutorValidation,
                        result.provenance.dispatch_state,
                    );
                    result.staged_stream_outputs.clear();
                    result.managed_cache_completions.clear();
                }
                _ => {}
            }
        }

        if matches!(result.disposition, ExecutionDisposition::RestartSequence(_))
            && (!result.staged_stream_outputs.is_empty()
                || result.managed_cache.is_some()
                || !result.managed_cache_completions.is_empty()
                || result.output.audio.is_some()
                || result.output.text.is_some()
                || result.output.input_transcription.is_some()
                || result.output.phase_timing_override.is_some()
                || result.output.asr_diagnostics.is_some())
        {
            let message =
                "sequence restart cannot publish output or managed-cache completion state";
            result.output = ExecutorOutput::error(plan.session.request_id.clone(), message);
            result.disposition =
                ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
            result.safe_point = true;
            result.provenance = OutcomeProvenance::failure(
                FailureOrigin::ExecutorValidation,
                result.provenance.dispatch_state,
            );
            result.staged_stream_outputs.clear();
            result.managed_cache_completions.clear();
            result.managed_cache = None;
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

        let workspace_backpressure = Self::is_workspace_backpressure(&result);
        if workspace_backpressure {
            self.observe_workspace_retry(&plan.session);
        }
        let retry_attempt = match &mut result.disposition {
            ExecutionDisposition::Failed(failure)
                if workspace_backpressure && failure.retry != RetryDisposition::Never =>
            {
                // The refusal came before device entry, so the step is retried
                // exactly. It waits for capacity on a wall-clock budget: a few
                // attempts milliseconds apart would all read one cached
                // capacity sample, and fail a running stream for nothing.
                let wait = self
                    .workspace_waits
                    .entry(plan.session.clone())
                    .or_insert_with(WorkspaceWait::start);
                wait.attempts = wait.attempts.saturating_add(1);
                let (attempts, waited) = (wait.attempts, wait.started.elapsed());
                if waited < self.retry_policy.workspace_wait_budget {
                    record_engine_workspace_wait(EngineWorkspaceWaitOutcome::Retried);
                    Some(attempts)
                } else {
                    self.workspace_waits.remove(&plan.session);
                    record_engine_workspace_wait(EngineWorkspaceWaitOutcome::Exhausted);
                    let message = format!(
                        "executor retry budget exhausted after {attempts} attempts over {:.1}s of workspace backpressure: {}",
                        waited.as_secs_f64(),
                        failure.message
                    );
                    failure.retry = RetryDisposition::Never;
                    failure.message = message.clone();
                    result.output.finished = true;
                    result.output.error = Some(message);
                    None
                }
            }
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
                if self.workspace_waits.remove(&plan.session).is_some()
                    && result.provenance.dispatch_state != DispatchState::NotStarted
                {
                    record_engine_workspace_wait(EngineWorkspaceWaitOutcome::Recovered);
                }
                None
            }
        };
        let prospective_authoritative_commit = matches!(
            result.disposition,
            ExecutionDisposition::Progress
                | ExecutionDisposition::Yielded(_)
                | ExecutionDisposition::Finished(ExecutionFinishReason::Completed)
        );
        if let Err(error) = Self::authenticate_realtime_stage_outcome(
            &plan,
            result.realtime_stage_outcome.as_ref(),
            prospective_authoritative_commit,
        ) {
            let message = error.to_string();
            result.output = ExecutorOutput::error(plan.session.request_id.clone(), message.clone());
            result.disposition =
                ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
            result.safe_point = true;
            result.provenance = OutcomeProvenance::failure(
                FailureOrigin::ExecutorValidation,
                result.provenance.dispatch_state,
            );
            result.staged_stream_outputs.clear();
            result.managed_cache_completions.clear();
            result.realtime_stage_outcome = None;
        }
        let realtime_stage_outcome = result.realtime_stage_outcome;
        let managed_reservation = self.active_managed_cache.remove(&plan.plan_id);
        let report = Self::report_from_result(&result);
        // Validate the execution transition on a clone before publishing the
        // physical cache table. This keeps invalid/duplicate reports from
        // advancing KV state, while the serialized core lock guarantees that
        // installing the validated tracker cannot race another row commit.
        let prospective_state = self
            .execution_trackers
            .get(&plan.session.request_id)
            .cloned()
            .ok_or_else(|| {
                Error::InferenceError("execution tracker is missing for active plan".to_string())
            })
            .and_then(|mut tracker| {
                tracker.commit(&plan, &report)?;
                let prepared_realtime = match realtime_stage_outcome {
                    Some(outcome) if prospective_authoritative_commit => {
                        let prepared = self
                            .scheduler
                            .prepare_realtime_stage_outcome(&plan.session, outcome)?;
                        if outcome.next.is_none() {
                            self.requests
                                .get(&plan.session.request_id)
                                .ok_or_else(|| {
                                    Error::InferenceError(
                                        "committed realtime operation lost its request mailbox"
                                            .into(),
                                    )
                                })?
                                .realtime_asr_operation(outcome.operation_id)?;
                        }
                        Some(prepared)
                    }
                    _ => None,
                };
                Ok((tracker, prepared_realtime))
            });

        let (prospective_tracker, prepared_realtime) = match prospective_state {
            Ok(state) => state,
            Err(err) => {
                if let Err(abort_error) = self.executor.finalize_pending_quantum(
                    plan.plan_id,
                    &plan.session,
                    PendingQuantumDecision::Abort,
                ) {
                    warn!(
                        plan_id = plan.plan_id,
                        request_id = %plan.session.request_id,
                        error = %abort_error,
                        "Failed to abort executor state after result validation rejection"
                    );
                }
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

        // Completed terminal quanta commit their final append before normal
        // terminal cleanup releases the managed sequence. The executor's host
        // transaction and the managed table must use this exact same decision.
        let authoritative_commit = matches!(
            result.disposition,
            ExecutionDisposition::Progress
                | ExecutionDisposition::Yielded(_)
                | ExecutionDisposition::Finished(ExecutionFinishReason::Completed)
        );
        let pending_decision = if authoritative_commit {
            PendingQuantumDecision::Commit
        } else {
            PendingQuantumDecision::Abort
        };
        let prepared =
            self.executor
                .prepare_pending_quantum(plan.plan_id, &plan.session, pending_decision);
        let pending_was_prepared = matches!(&prepared, Ok(PendingQuantumFinalizeStatus::Finalized));
        let pending_auth = match prepared {
            Ok(PendingQuantumFinalizeStatus::Finalized) if result.pending_quantum_required => {
                Ok(())
            }
            Ok(PendingQuantumFinalizeStatus::NotFound) if !result.pending_quantum_required => {
                Ok(())
            }
            Ok(PendingQuantumFinalizeStatus::NotFound) => Err(Error::InferenceError(
                "executor result requires a pending quantum, but none was registered".to_string(),
            )),
            Ok(PendingQuantumFinalizeStatus::Finalized) => Err(Error::InferenceError(
                "executor registered a pending quantum without authenticating it in the result"
                    .to_string(),
            )),
            Err(error) => Err(error),
        };
        let mut finalize_managed = |commit| match managed_reservation.as_ref() {
            Some(reservation) => {
                self.managed_kv_cache
                    .finalize(reservation, result.managed_cache.as_ref(), commit)
            }
            None if result.managed_cache.is_some() => Err(Error::InferenceError(
                "executor returned an unplanned managed KV receipt".to_string(),
            )),
            None => Ok(()),
        };
        let managed_result = match pending_auth {
            Ok(()) => finalize_managed(authoritative_commit),
            Err(error) => match finalize_managed(false) {
                Ok(()) => Err(error),
                Err(abort_error) => Err(Error::InferenceError(format!(
                    "{error}; managed KV abort also failed: {abort_error}"
                ))),
            },
        };
        let pending_result = if pending_was_prepared {
            if managed_result.is_ok() {
                match self
                    .executor
                    .publish_pending_quantum(plan.plan_id, &plan.session)
                {
                    Ok(PendingQuantumFinalizeStatus::Finalized) => Ok(()),
                    Ok(PendingQuantumFinalizeStatus::NotFound) => Err(Error::InferenceError(
                        "prepared executor quantum disappeared before publication".to_string(),
                    )),
                    Err(error) => Err(error),
                }
            } else {
                self.executor
                    .discard_prepared_quantum(plan.plan_id, &plan.session);
                Ok(())
            }
        } else {
            Ok(())
        };
        if let Err(error) = managed_result {
            let message = match pending_result.as_ref() {
                Ok(_) => format!("managed KV commit failed: {error}"),
                Err(abort_error) => format!(
                    "managed KV commit failed: {error}; executor pending-quantum abort also failed: {abort_error}"
                ),
            };
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
        } else if let Err(error) = pending_result {
            let message = format!("executor pending-quantum finalization failed: {error}");
            result.output = ExecutorOutput::error(plan.session.request_id.clone(), message.clone());
            result.disposition =
                ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message));
            result.safe_point = true;
            result.provenance = OutcomeProvenance::failure(
                FailureOrigin::StateCommit,
                result.provenance.dispatch_state,
            );
            result.staged_stream_outputs.clear();
            result.managed_cache_completions.clear();
            let failure_report = Self::report_from_result(&result);
            if let Some(tracker) = self.execution_trackers.get_mut(&plan.session.request_id) {
                let _ = tracker.commit(&plan, &failure_report);
            }
        } else if authoritative_commit {
            if let Some(prepared) = prepared_realtime {
                if let Some(outcome) =
                    realtime_stage_outcome.filter(|outcome| outcome.next.is_none())
                {
                    self.requests
                        .get(&plan.session.request_id)
                        .expect("prepared realtime mailbox remains present until publication")
                        .complete_realtime_asr_operation(outcome.operation_id)
                        .expect("prepared realtime operation remains present until publication");
                }
                self.scheduler
                    .publish_prepared_realtime_stage_outcome(&plan.session, prepared);
            }
            if !matches!(result.disposition, ExecutionDisposition::RestartSequence(_)) {
                self.execution_trackers
                    .insert(plan.session.request_id.clone(), prospective_tracker);
            }
        } else if !matches!(result.disposition, ExecutionDisposition::RestartSequence(_)) {
            self.execution_trackers
                .insert(plan.session.request_id.clone(), prospective_tracker);
        }
        if matches!(result.disposition, ExecutionDisposition::RestartSequence(_)) {
            let restart_failure = if let Some(reservation) = managed_reservation.as_ref() {
                let request = self.requests.get(&plan.session.request_id).cloned();
                if request
                    .as_ref()
                    .is_some_and(|request| request.is_cancelled())
                {
                    if let Err(error) = self.replace_active_result_with_terminal(
                        &plan,
                        &mut result,
                        ExecutorOutput::cancelled(plan.session.request_id.clone()),
                        ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled),
                        OutcomeProvenance::started(),
                    ) {
                        Some(format!(
                            "sequence restart cancellation commit failed: {error}"
                        ))
                    } else {
                        self.release_managed_session_if_ready(&plan.session);
                        result.output.request_id = plan.session.request_id.clone();
                        return Some(CommittedExecutorOutput {
                            session: plan.session,
                            output: result.output,
                            disposition: result.disposition,
                            provenance: result.provenance,
                            staged_stream_outputs: Vec::new(),
                        });
                    }
                } else if let Some(request) = request {
                    if let Some(runtime) = request.managed_cache_runtime().cloned() {
                        match self.managed_kv_cache.reset_session_generation(
                            &runtime,
                            &plan.session,
                            reservation.session_generation,
                        ) {
                            Ok(_) => {
                                #[cfg(test)]
                                if let Some(hook) = self.restart_after_reset_hook.as_ref() {
                                    hook();
                                }
                                if request.is_cancelled() {
                                    if let Err(error) = self.replace_active_result_with_terminal(
                                        &plan,
                                        &mut result,
                                        ExecutorOutput::cancelled(plan.session.request_id.clone()),
                                        ExecutionDisposition::Finished(
                                            ExecutionFinishReason::Cancelled,
                                        ),
                                        OutcomeProvenance::started(),
                                    ) {
                                        Some(format!(
                                            "post-reset cancellation commit failed: {error}"
                                        ))
                                    } else {
                                        self.release_managed_session_if_ready(&plan.session);
                                        result.output.request_id = plan.session.request_id.clone();
                                        return Some(CommittedExecutorOutput {
                                            session: plan.session,
                                            output: result.output,
                                            disposition: result.disposition,
                                            provenance: result.provenance,
                                            staged_stream_outputs: Vec::new(),
                                        });
                                    }
                                } else if self
                                    .scheduler
                                    .restart_request_for_recompute(&plan.session)
                                {
                                    self.execution_trackers.insert(
                                        plan.session.request_id.clone(),
                                        ExecutionTracker::new(plan.session.clone()),
                                    );
                                    self.execution_retry_attempts.remove(&plan.session);
                                    return None;
                                } else {
                                    Some(
                                        "scheduler rejected a managed sequence restart".to_string(),
                                    )
                                }
                            }
                            Err(error) => Some(format!("managed sequence reset failed: {error}")),
                        }
                    } else {
                        Some("sequence restart request lost its managed-cache runtime".to_string())
                    }
                } else {
                    Some("sequence restart request is no longer active".to_string())
                }
            } else {
                Some("sequence restart requires a managed-cache reservation".to_string())
            };

            let message =
                restart_failure.unwrap_or_else(|| "managed sequence restart failed".to_string());
            let dispatch_state = result.provenance.dispatch_state;
            if let Err(error) = self.replace_active_result_with_terminal(
                &plan,
                &mut result,
                ExecutorOutput::error(plan.session.request_id.clone(), message.clone()),
                ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message)),
                OutcomeProvenance::failure(FailureOrigin::StateCommit, dispatch_state),
            ) {
                result.output = ExecutorOutput::error(
                    plan.session.request_id.clone(),
                    format!("sequence restart failure commit failed: {error}"),
                );
                result.disposition =
                    ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        "sequence restart failure could not be committed",
                    ));
                result.provenance =
                    OutcomeProvenance::failure(FailureOrigin::StateCommit, dispatch_state);
            }
        }
        self.release_managed_session_if_ready(&plan.session);
        if matches!(
            result.disposition,
            ExecutionDisposition::Progress
                | ExecutionDisposition::Yielded(_)
                | ExecutionDisposition::Finished(ExecutionFinishReason::Completed)
        ) && result.provenance.dispatch_state != DispatchState::NotStarted
        {
            self.workspace_progress_epoch = self.workspace_progress_epoch.wrapping_add(1);
        }

        match &result.disposition {
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::RetrySameSession =>
            {
                if let Some((operation_id, subphase)) = Self::realtime_work_identity(&plan.work) {
                    if self.scheduler.release_realtime_operation_for_retry(
                        &plan.session,
                        plan.plan_id,
                        operation_id,
                        subphase,
                    ) {
                        return None;
                    }
                    let message = "scheduler rejected an exact realtime stage retry";
                    return Some(CommittedExecutorOutput {
                        session: plan.session.clone(),
                        output: ExecutorOutput::error(
                            plan.session.request_id.clone(),
                            message.to_string(),
                        ),
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
            | ExecutionDisposition::RestartSequence(_)
            | ExecutionDisposition::Finished(_) => {
                if Self::realtime_work_identity(&plan.work).is_none() {
                    self.scheduler.update_after_step(
                        &plan.session.request_id,
                        result.output.tokens_processed,
                        result.output.tokens_generated,
                        step_time_ms,
                    );
                    let audio_transition =
                        match result.disposition {
                            ExecutionDisposition::Yielded(
                                super::super::YieldReason::AwaitingAudioDecode { max_frames },
                            ) => Some(self.scheduler.request_sequence_audio_decode(
                                &plan.session.request_id,
                                max_frames,
                            )),
                            ExecutionDisposition::Yielded(
                                super::super::YieldReason::AwaitingAudioOutput { max_frames },
                            ) => {
                                // The batch entered the adapter, but this row stopped
                                // at output admission before codec execution. The
                                // active-plan removal above fences this refund against
                                // duplicate results and the unexecuted rollback path.
                                self.scheduler.refund_unexecuted_service(
                                    &plan.session,
                                    Scheduler::service_units_for_work(1, &plan.work),
                                );
                                Some(
                                    self.scheduler
                                        .defer_audio_output(&plan.session.request_id, max_frames),
                                )
                            }
                            _ => None,
                        };
                    if let Some(Err(error)) = audio_transition {
                        return Some(CommittedExecutorOutput {
                            session: plan.session,
                            output: ExecutorOutput::error(
                                result.output.request_id,
                                error.to_string(),
                            ),
                            disposition: ExecutionDisposition::Failed(
                                ExecutionFailure::invalid_output("audio decode transition failed"),
                            ),
                            provenance: OutcomeProvenance::failure(
                                FailureOrigin::StateCommit,
                                result.provenance.dispatch_state,
                            ),
                            staged_stream_outputs: Vec::new(),
                        });
                    }
                    if matches!(
                        result.disposition,
                        ExecutionDisposition::Yielded(super::super::YieldReason::AwaitingFinalization)
                    ) {
                        if let Err(error) = self
                            .scheduler
                            .request_sequence_finalize(&plan.session.request_id)
                        {
                            warn!(
                                request_id = %plan.session.request_id,
                                error = %error,
                                "Committed sequence could not enter finalization"
                            );
                            return Some(CommittedExecutorOutput {
                                session: plan.session,
                                output: ExecutorOutput::error(
                                    result.output.request_id,
                                    format!("sequence finalization transition failed: {error}"),
                                ),
                                disposition: ExecutionDisposition::Failed(
                                    ExecutionFailure::invalid_output(
                                        "sequence finalization transition failed",
                                    ),
                                ),
                                provenance: OutcomeProvenance::failure(
                                    FailureOrigin::StateCommit,
                                    result.provenance.dispatch_state,
                                ),
                                staged_stream_outputs: Vec::new(),
                            });
                        }
                    }
                }
            }
            ExecutionDisposition::Failed(_) => {}
        }
        result.output.request_id = plan.session.request_id.clone();
        if !result.staged_stream_outputs.is_empty() {
            // AfterQuantumCommit PCM/text is just as irreversible as an
            // IncrementalCommitted event. Fence recomputation before handing
            // the committed outbox to delivery, which may already be audible
            // when the next model quantum fails or needs capacity.
            if result.staged_stream_outputs.iter().any(|output| {
                !output.samples.is_empty()
                    || output.text.as_ref().is_some_and(|text| !text.is_empty())
                    || output.asr_progress.is_some()
            }) {
                self.incremental_stream_sessions
                    .insert(plan.session.clone());
            }
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
}
