//! Stream delivery: staged stream outputs, incremental progress and stream
//! delivery failures.

use super::*;

impl EngineCore {
    pub(super) fn validate_staged_stream_outputs(
        &self,
        session: &super::super::SessionKey,
        outputs: &[super::super::output::StreamingOutput],
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

    pub(in crate::engine) fn commit_incremental_stream_progress(
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

        let dispatch = self
            .in_flight_dispatches
            .get(&progress.batch_id)
            .ok_or_else(|| {
                StreamProgressRejection::invalid(
                    "stream progress references an inactive physical batch",
                )
            })?;
        if dispatch.output_visibility != OutputVisibility::IncrementalCommitted
            || dispatch.physical_batch.lane != progress.lane
            || !dispatch
                .physical_batch
                .rows
                .iter()
                .any(|row| row.plan_id == progress.plan_id && row.session == progress.session)
        {
            return Err(StreamProgressRejection::invalid(
                "stream progress physical batch fence does not match its in-flight dispatch ticket",
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

    pub(super) fn commit_progress_delivery(
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

    pub(super) fn cancel_failed_stream(&self, failure: &StreamDeliveryFailure) {
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

    pub(super) fn record_stream_failure(
        &self,
        failure: StreamDeliveryFailure,
        failures: &mut HashMap<super::super::SessionKey, StreamDeliveryFailure>,
        deliveries: &mut IncrementalStreamDeliveryWorkers,
    ) {
        self.cancel_failed_stream(&failure);
        deliveries.abandon_session(&failure.session);
        failures.entry(failure.session.clone()).or_insert(failure);
    }

    pub(super) fn enqueue_incremental_progress(
        &mut self,
        progress: FencedStreamProgress,
        failures: &mut HashMap<super::super::SessionKey, StreamDeliveryFailure>,
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

    pub(super) fn overlay_terminal_stream_delivery_failure(
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
}
