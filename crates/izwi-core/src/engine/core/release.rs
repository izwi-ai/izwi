//! Request release: terminal and abandoned-session cleanup, managed session
//! release and aborts.

use super::*;

impl EngineCore {
    pub(super) fn terminal_release_cause(disposition: &ExecutionDisposition) -> Option<TerminalReleaseCause> {
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

    pub(super) fn clear_exact_execution_state(&mut self, session: &super::super::SessionKey) {
        if self
            .in_flight_dispatches
            .values()
            .any(|dispatch| dispatch.contains_session(session))
        {
            return;
        }
        self.clear_execution_state_for_capacity_replay(session);
        self.workspace_retry_progress.remove(session);
        self.workspace_row_limits.remove(session);
        self.workspace_pressure.remove(session);
        self.stream_sequence_cursors.remove(session);
        self.incremental_stream_sessions.remove(session);
    }

    // Capacity replay retains the same client-visible stream incarnation.
    pub(super) fn clear_execution_state_for_capacity_replay(&mut self, session: &super::super::SessionKey) {
        if self
            .in_flight_dispatches
            .values()
            .any(|dispatch| dispatch.contains_session(session))
        {
            return;
        }
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
        self.plan_begin_attempts.remove(session);
        self.workspace_waits.remove(session);
    }

    pub(super) fn request_managed_session_release(&mut self, session: &super::super::SessionKey) -> bool {
        if self
            .active_managed_cache
            .values()
            .any(|reservation| &reservation.session == session)
        {
            self.pending_managed_releases.insert(session.clone());
            return false;
        }
        match self.managed_kv_cache.release_session(session) {
            Ok(()) => {
                self.pending_managed_releases.remove(session);
                true
            }
            Err(error) => {
                self.pending_managed_releases.insert(session.clone());
                warn!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    error = %error,
                    "Failed to release a managed KV session table; retaining cleanup quarantine"
                );
                false
            }
        }
    }

    pub(super) fn release_managed_session_if_ready(&mut self, session: &super::super::SessionKey) {
        if self.pending_managed_releases.contains(session)
            && !self
                .active_managed_cache
                .values()
                .any(|reservation| &reservation.session == session)
        {
            let _ = self.request_managed_session_release(session);
        }
    }

    pub(super) fn exact_session_cleanup_absent(&self, session: &super::super::SessionKey) -> bool {
        self.scheduler.get_sequence_id(&session.request_id) != Some(session.epoch)
            && self
                .scheduler
                .pending_release_confirmation_required(session)
                .is_none()
            && !self.pending_managed_releases.contains(session)
            && !self
                .active_managed_cache
                .values()
                .any(|reservation| &reservation.session == session)
    }

    pub(crate) fn register_cleanup_confirmation(
        &mut self,
        session: &super::super::SessionKey,
    ) -> oneshot::Receiver<Result<()>> {
        let (sender, receiver) = oneshot::channel();
        if self.exact_session_cleanup_absent(session) {
            let _ = sender.send(Ok(()));
        } else {
            self.cleanup_confirmation_waiters
                .entry(session.clone())
                .or_default()
                .push(sender);
        }
        receiver
    }

    pub(super) fn resolve_cleanup_waiters(&mut self, session: &super::super::SessionKey) {
        if let Some(waiters) = self.cleanup_confirmation_waiters.remove(session) {
            for waiter in waiters {
                let _ = waiter.send(Ok(()));
            }
        }
    }

    pub(super) fn resolve_all_cleanup_waiters(&mut self) {
        for (_, waiters) in self.cleanup_confirmation_waiters.drain() {
            for waiter in waiters {
                let _ = waiter.send(Ok(()));
            }
        }
    }

    pub(super) fn fail_all_cleanup_waiters(&mut self, message: &str) {
        for (_, waiters) in self.cleanup_confirmation_waiters.drain() {
            for waiter in waiters {
                let _ = waiter.send(Err(Error::InferenceError(message.to_string())));
            }
        }
    }

    pub(super) fn record_unconfirmed_cleanup(&mut self, session: &super::super::SessionKey) {
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

    pub(super) async fn attempt_pending_release_cleanup(&mut self, session: &super::super::SessionKey) {
        let Some(confirmation_required) = self
            .scheduler
            .pending_release_confirmation_required(session)
        else {
            if self.exact_session_cleanup_absent(session) {
                self.resolve_cleanup_waiters(session);
            }
            return;
        };
        let release = self.executor.cleanup_session(session).await;
        if (release.confirmed || !confirmation_required)
            && self.request_managed_session_release(session)
        {
            self.scheduler.confirm_session_release(session);
            self.resolve_cleanup_waiters(session);
        } else {
            self.record_unconfirmed_cleanup(session);
        }
    }

    pub(super) async fn begin_terminal_release(
        &mut self,
        session: &super::super::SessionKey,
        cause: TerminalReleaseCause,
    ) {
        if let Some(request) = self
            .requests
            .get(&session.request_id)
            .filter(|_| self.scheduler.get_sequence_id(&session.request_id) == Some(session.epoch))
        {
            let terminal = match cause {
                TerminalReleaseCause::Completed => RealtimeAsrTerminalOutcome::Completed,
                TerminalReleaseCause::Cancelled => RealtimeAsrTerminalOutcome::Cancelled,
                TerminalReleaseCause::TimedOut => RealtimeAsrTerminalOutcome::TimedOut,
                TerminalReleaseCause::Failed => {
                    RealtimeAsrTerminalOutcome::Failed(Arc::from("realtime ASR session failed"))
                }
            };
            if request.is_realtime_asr_session() {
                let _ = request.terminate_realtime_asr_ingress(terminal);
            }
        }
        if matches!(
            self.scheduler.begin_terminal_release(session, cause),
            BeginTerminalRelease::Started { .. }
        ) {
            self.attempt_pending_release_cleanup(session).await;
        }
        let _ = self.request_managed_session_release(session);
        self.clear_exact_execution_state(session);
        if self.exact_session_cleanup_absent(session) {
            self.resolve_cleanup_waiters(session);
        }
    }

    pub(super) async fn reconcile_due_cleanup(&mut self) {
        let sessions = self
            .scheduler
            .due_cleanup_sessions(Instant::now(), self.retry_policy.cleanup_budget_per_step);
        for session in sessions {
            self.attempt_pending_release_cleanup(&session).await;
        }
    }

    /// Confirm that a terminal output has been routed outside the core.
    ///
    /// The exact scheduler session remains fenced after [`Self::step`] returns
    /// its terminal output. Callers must acknowledge that same session only
    /// after placing the output in their delivery channel or return batch.
    pub fn acknowledge_terminal_output(&mut self, session: &super::super::SessionKey) -> bool {
        self.scheduler.mark_terminal_delivered(session)
    }

    /// Cancel a synchronous caller's exact session after its completion
    /// receiver is abandoned. There is no consumer for the cancellation event,
    /// so discard that exact terminal output and mark delivery complete while
    /// retaining the scheduler quarantine until executor cleanup is confirmed.
    pub(crate) async fn abandon_request_session(&mut self, session: &super::super::SessionKey) -> bool {
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
        session: &super::super::SessionKey,
    ) -> Option<Duration> {
        self.attempt_pending_release_cleanup(session).await;
        self.scheduler
            .pending_cleanup_attempts(session)
            .map(|attempt| self.retry_policy.cleanup_delay(attempt.max(1)))
    }

    /// Arm an exact cleanup confirmation before cancellation and drive the
    /// same idempotent quarantine cleanup path for active, terminal, and
    /// already-confirmed sessions.
    pub(crate) async fn begin_confirmed_session_cleanup(
        &mut self,
        session: &super::super::SessionKey,
    ) -> oneshot::Receiver<Result<()>> {
        let confirmation = self.register_cleanup_confirmation(session);
        let _ = self.abort_request_session(session).await;
        self.attempt_pending_release_cleanup(session).await;
        confirmation
    }

    pub(crate) fn abandoned_session_cleanup_delay(
        &self,
        session: &super::super::SessionKey,
    ) -> Option<Duration> {
        self.scheduler
            .pending_cleanup_attempts(session)
            .map(|attempt| self.retry_policy.cleanup_delay(attempt.max(1)))
    }

    pub(super) async fn terminate_request_session(
        &mut self,
        session: &super::super::SessionKey,
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
        self.execution_profiles.remove(&session.request_id);
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

    /// Abort only if the caller still owns the exact request incarnation.
    pub async fn abort_request_session(&mut self, session: &super::super::SessionKey) -> bool {
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
            if let Some(request) = self.requests.get(&request_id) {
                if request.is_realtime_asr_session() {
                    let _ = request
                        .terminate_realtime_asr_ingress(RealtimeAsrTerminalOutcome::Unloaded);
                }
            }
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

    /// Shutdown the engine core.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down engine core");

        // Abort all pending requests
        let request_ids: Vec<_> = self.requests.keys().cloned().collect();
        for id in request_ids {
            self.abort_request(&id).await;
        }

        // Executor shutdown is the final model-owned physical-state fence.
        // Settle every registered receipt explicitly if that fence fails;
        // dropping the Core must never be the notification mechanism.
        if let Err(error) = self.executor.shutdown().await {
            self.fail_all_cleanup_waiters(&format!(
                "Engine executor shutdown failed before realtime cleanup confirmation: {error}"
            ));
            return Err(error);
        }

        // Abort any managed transactions that were still owned by prepared
        // plans, then release the exact session tables named by cleanup
        // receipts. Executor shutdown has already guaranteed that no model
        // kernel can publish another completion into these transactions.
        let mut managed_release_errors = Vec::new();
        for (_, reservation) in std::mem::take(&mut self.active_managed_cache) {
            if let Err(error) = self.managed_kv_cache.finalize(&reservation, None, false) {
                managed_release_errors.push(format!(
                    "{}:{} transaction abort failed: {error}",
                    reservation.session.request_id, reservation.session.epoch
                ));
            }
        }
        let cleanup_sessions = self
            .cleanup_confirmation_waiters
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        for session in &cleanup_sessions {
            if let Err(error) = self.managed_kv_cache.release_session(session) {
                managed_release_errors.push(format!(
                    "{}:{} managed release failed: {error}",
                    session.request_id, session.epoch
                ));
            }
        }

        // A successful executor shutdown is the final physical-release fence.
        self.scheduler.force_release_all_after_executor_shutdown();
        self.pending_managed_releases.clear();
        self.requests.clear();
        self.request_start_times.clear();
        self.request_phase_timings.clear();
        self.execution_trackers.clear();
        self.active_plans.clear();
        self.in_flight_dispatches.clear();
        self.active_stream_batches.clear();
        self.stream_sequence_cursors.clear();
        self.incremental_stream_sessions.clear();
        self.pending_terminal_outputs.clear();
        self.execution_retry_attempts.clear();
        self.plan_begin_attempts.clear();
        self.workspace_retry_progress.clear();
        self.workspace_waits.clear();
        self.workspace_row_limits.clear();
        self.workspace_pressure.clear();
        self.initialized = false;

        if managed_release_errors.is_empty() {
            self.resolve_all_cleanup_waiters();
        } else {
            let message = format!(
                "Engine shutdown could not confirm managed realtime cleanup: {}",
                managed_release_errors.join("; ")
            );
            self.fail_all_cleanup_waiters(&message);
            return Err(Error::InferenceError(message));
        }

        info!("Engine core shutdown complete");

        Ok(())
    }
}
