//! Voxtral offline ASR as closed realtime streams. A request's audio was
//! prepared once (`VoxtralOfflinePreparedAudio`); its row prefills the
//! transcription prompt alone, then decodes one audio frame per step beside
//! the other rows, in one language-model forward. Each row's quantum stays
//! pending until Core commits it, as a realtime session's does, and every row
//! settles alone: a cancelled row rolls back only its own quantum.

use super::*;
use crate::models::architectures::voxtral::realtime::{
    VoxtralRealtimeCheckpoint, VoxtralRealtimeStep,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// What one row of a Voxtral offline step does once its model work
/// returned: seal its quantum for Core's decision, or roll it back, alone.
pub(super) trait VoxtralOfflineStepRow {
    type Sealed;

    fn cancelled(&self) -> bool;

    /// Finish the row's step (fold an exhausted completion in, stage its
    /// text) and seal its quantum. A row cancelled while it seals returns
    /// `Error::Cancelled` and stays rollback-capable.
    fn seal(&mut self, step: VoxtralRealtimeStep) -> Result<Self::Sealed>;

    /// Restore the row's state and pages to its quantum's start.
    fn rollback(&mut self) -> Result<()>;
}

#[derive(Debug, PartialEq)]
pub(super) enum VoxtralOfflineRowOutcome<S> {
    Sealed(S),
    Cancelled,
    Failed(String),
}

/// Settle every row of one step alone, in order: a row cancelled before or
/// while it seals rolls back its own quantum and is cancelled; a row whose
/// seal fails rolls back and fails alone. The other rows seal as if it had
/// never been in the batch.
pub(super) fn settle_voxtral_offline_rows<R: VoxtralOfflineStepRow>(
    rows: &mut [R],
    steps: Vec<VoxtralRealtimeStep>,
) -> Vec<VoxtralOfflineRowOutcome<R::Sealed>> {
    if steps.len() != rows.len() {
        let message = format!(
            "Voxtral offline step returned {} rows for a {}-row batch",
            steps.len(),
            rows.len()
        );
        return rows
            .iter_mut()
            .map(|row| match row.rollback() {
                Ok(()) => VoxtralOfflineRowOutcome::Failed(message.clone()),
                Err(rollback) => VoxtralOfflineRowOutcome::Failed(format!(
                    "{message}; rollback also failed: {rollback}"
                )),
            })
            .collect();
    }
    rows.iter_mut()
        .zip(steps)
        .map(|(row, step)| {
            let error = if row.cancelled() {
                Error::Cancelled("Voxtral offline row cancelled before its seal".into())
            } else {
                match row.seal(step) {
                    Ok(sealed) => return VoxtralOfflineRowOutcome::Sealed(sealed),
                    Err(error) => error,
                }
            };
            let cancelled = matches!(error, Error::Cancelled(_)) || row.cancelled();
            match row.rollback() {
                Ok(()) if cancelled => VoxtralOfflineRowOutcome::Cancelled,
                Ok(()) => VoxtralOfflineRowOutcome::Failed(error.to_string()),
                Err(rollback) => VoxtralOfflineRowOutcome::Failed(format!(
                    "{error}; Voxtral offline rollback also failed: {rollback}"
                )),
            }
        })
        .collect()
}

/// One row's quantum: its state lease, its pages and the checkpoint that
/// rolls both back, until it seals into a pending quantum.
struct VoxtralOfflineRow<'a> {
    index: usize,
    request: &'a EngineCoreRequest,
    plan_id: crate::engine::PlanId,
    session: SessionKey,
    lease: Option<ExecutorStateLease<'a, ActiveVoxtralRealtime>>,
    cache: Option<PhysicalPagedKvCache>,
    checkpoint: Option<VoxtralRealtimeCheckpoint>,
    prior_last_tokens_generated: usize,
    prior_stream_sequence: usize,
    cache_start: usize,
    /// Pages the step appends: the prompt for a prefill, one for a frame
    /// the step decodes, none for a row that only completes.
    expected_append: usize,
    /// Tokens the step processes, as the scheduler planned them.
    tokens_processed: usize,
    /// Model predictions the step makes (one per forward).
    tokens_generated: usize,
    duration_secs: f32,
}

type SealedVoxtralOfflineRow = (
    crate::engine::PlanId,
    super::super::PendingVoxtralRealtimeQuantum,
    ModelSessionResult,
);

impl VoxtralOfflineRow<'_> {
    fn active_mut(&mut self) -> Result<&mut ActiveVoxtralRealtime> {
        self.lease
            .as_mut()
            .ok_or_else(|| Error::InferenceError("Voxtral offline row lost its state".into()))?
            .require_state_mut()
    }

    /// The row's lease, released: the request ends with this step.
    fn release(mut self) -> Result<()> {
        match self.lease.take() {
            Some(lease) => lease.release(),
            None => Ok(()),
        }
    }
}

impl VoxtralOfflineStepRow for VoxtralOfflineRow<'_> {
    type Sealed = SealedVoxtralOfflineRow;

    fn cancelled(&self) -> bool {
        self.request.is_cancelled()
    }

    fn seal(&mut self, step: VoxtralRealtimeStep) -> Result<Self::Sealed> {
        let request = self.request;
        let cache_start = self.cache_start;
        let expected_append = self.expected_append;
        let (lease, cache) = match (self.lease.as_mut(), self.cache.as_mut()) {
            (Some(lease), Some(cache)) => (lease, cache),
            _ => {
                return Err(Error::InferenceError(
                    "Voxtral offline row was sealed twice".into(),
                ))
            }
        };
        let active = lease.require_state_mut()?;
        let step = active
            .model
            .finish_exhausted_offline_row(&mut active.state, step)?;
        let appended = cache
            .context_len()
            .checked_sub(cache_start)
            .filter(|appended| *appended == expected_append)
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "Voxtral offline row appended {} pages, expected {expected_append}",
                    cache.context_len().saturating_sub(cache_start)
                ))
            })?;
        if let Some(tx) = NativeExecutor::stream_sender(request) {
            if !step.delta.is_empty() {
                NativeExecutor::stream_text_with_policy(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active.stream_sequence,
                    step.delta.clone(),
                )?;
            }
            if step.finished {
                NativeExecutor::stream_final_marker_with_policy(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active.stream_sequence,
                )?;
            }
        }
        let staged = request.take_staged_stream_outputs()?;
        if request.is_cancelled() {
            return Err(Error::Cancelled(
                "Voxtral offline row cancelled during its seal".into(),
            ));
        }
        let text = active.state.text().to_string();
        let finished = active.state.is_finished();
        let sample_rate = active.input_sample_rate;
        // A finishing row publishes its diagnostics once.
        let diagnostics = finished.then(|| {
            active.clock.diagnostics(Some(serde_json::json!({
                "model_family": "voxtral_realtime",
                "audio": { "prepared_frames": active.state.prepared_audio_frames() },
                "decode": { "tokens_generated": step.tokens_generated },
            })))
        });
        lease.validate_defer()?;
        // Nothing below fails: the row leaves its rollback-capable lease.
        let completions = cache.take_completed_writes();
        let active = self
            .lease
            .take()
            .expect("validated offline lease")
            .defer_validated();
        let pending = super::super::PendingVoxtralRealtimeQuantum {
            session: self.session.clone(),
            active,
            cache: self.cache.take().expect("validated offline cache"),
            checkpoint: self.checkpoint.take().expect("armed offline checkpoint"),
            prior_last_tokens_generated: self.prior_last_tokens_generated,
            prior_stream_sequence: self.prior_stream_sequence,
            prior_input_sample_rate: sample_rate,
            finished,
        };
        let result = ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: self.duration_secs,
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: self.tokens_processed,
            tokens_generated: self.tokens_generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: diagnostics,
            error: None,
            logprobs: Vec::new(),
        })
        .with_staged_stream_outputs(staged)
        .with_managed_cache_completions(completions)
        .with_managed_cache_append(appended)
        .requiring_pending_quantum();
        Ok((self.plan_id, pending, result))
    }

    fn rollback(&mut self) -> Result<()> {
        let _ = self.request.take_staged_stream_outputs();
        let Some(mut checkpoint) = self.checkpoint.take() else {
            return Ok(());
        };
        let prior_tokens = self.prior_last_tokens_generated;
        let prior_stream = self.prior_stream_sequence;
        let mut cache = self
            .cache
            .take()
            .ok_or_else(|| Error::InferenceError("Voxtral offline row lost its pages".into()))?;
        let active = self.active_mut()?;
        active.last_tokens_generated = prior_tokens;
        active.stream_sequence = prior_stream;
        active
            .model
            .rollback_realtime_quantum(&mut active.state, &mut cache, &mut checkpoint)?;
        if let Some(lease) = self.lease.as_mut() {
            lease.mark_clean();
        }
        Ok(())
    }
}

impl NativeExecutor {
    /// Check a row out of the Voxtral state store and open its quantum. A
    /// prefill starts the row from its prepared audio; a decode continues
    /// the row its prefill started.
    fn voxtral_offline_row<'a>(
        &'a self,
        index: usize,
        request: &'a EngineCoreRequest,
        scheduled: &ScheduledRequest,
        managed: Option<super::super::RetainedRowManagedState>,
    ) -> Result<VoxtralOfflineRow<'a>> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Voxtral || !request.uses_asr_retained_sequence() {
            return Err(Error::InvalidInput(
                "Voxtral offline row crossed its retained route".into(),
            ));
        }
        let mut retained = managed.ok_or_else(|| {
            Error::InferenceError("Voxtral offline row lost its retained pages".into())
        })?;
        if retained.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Voxtral offline row unexpectedly received tensor state".into(),
            ));
        }
        let cache = retained.take_only_paged()?;
        retained.ensure_all_paged_consumed()?;
        let session = scheduled.session_key();
        let mut lease = ExecutorStateLease::checkout(
            &self.voxtral_realtime.states,
            session.clone(),
            variant,
            "Voxtral offline ASR",
        )?;
        let (expected_append, tokens_processed, tokens_generated) = if scheduled.is_prefill {
            let prompt = request.prepared_sequence_input_tokens.ok_or_else(|| {
                Error::InferenceError("Voxtral offline row lost its prompt".into())
            })?;
            if scheduled.num_tokens != prompt || cache.context_len() != 0 {
                return Err(Error::InvalidInput(format!(
                    "Voxtral offline prompt prefills whole: {} of {prompt} tokens scheduled at cursor {}",
                    scheduled.num_tokens,
                    cache.context_len()
                )));
            }
            // A prefill starts the row afresh from its prepared audio.
            lease.discard_state();
            let model = request
                .prepared_voxtral_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Voxtral offline row lost its model".into())
                })?;
            let artifact = request
                .prepared_voxtral_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Voxtral offline row lost its prepared audio".into())
                })?;
            let state =
                model.start_offline_state(&artifact, request.asr_language_for_execution())?;
            lease.install_state(ActiveVoxtralRealtime {
                variant,
                model,
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: artifact.source_sample_rate(),
                clock: Default::default(),
            })?;
            (prompt, prompt, 1)
        } else {
            if scheduled.num_tokens != 1 {
                return Err(Error::InvalidInput(
                    "Voxtral offline decode steps one frame per row".into(),
                ));
            }
            let active = lease.require_state_mut()?;
            if active.variant != variant {
                return Err(Error::InvalidInput(
                    "Voxtral offline row crossed its model variant".into(),
                ));
            }
            let advances = active.state.next_audio_frame() < active.state.prepared_audio_frames();
            let step = usize::from(advances);
            (step, step, step)
        };
        let active = lease.require_state_mut()?;
        let prior_last_tokens_generated = active.last_tokens_generated;
        let prior_stream_sequence = active.stream_sequence;
        let cache_start = cache.context_len();
        let checkpoint = active
            .model
            .begin_realtime_quantum(&mut active.state, &cache)?;
        lease.mark_dirty();
        Ok(VoxtralOfflineRow {
            index,
            request,
            plan_id: scheduled.plan_id,
            session,
            lease: Some(lease),
            cache: Some(cache),
            checkpoint: Some(checkpoint),
            prior_last_tokens_generated,
            prior_stream_sequence,
            cache_start,
            expected_append,
            tokens_processed,
            tokens_generated,
            duration_secs: request.prepared_asr_input_duration_secs().unwrap_or(0.0),
        })
    }

    /// One row's scalar prompt prefill: the transcription prompt over the
    /// first audio frames, predicting the row's first token.
    pub(super) fn voxtral_asr_prefill_row(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        managed: Option<super::super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        if !scheduled.is_prefill {
            return Err(Error::InvalidInput(
                "Voxtral offline prefill received decode work".into(),
            ));
        }
        if request.is_cancelled() {
            return Ok(ModelSessionResult::cancelled_before_dispatch(
                ExecutorOutput::cancelled(request.id.clone()),
            ));
        }
        let mut row = self.voxtral_offline_row(0, request, scheduled, managed)?;
        let prefill_started = std::time::Instant::now();
        let prefill = Self::run_blocking(|| {
            let VoxtralOfflineRow { lease, cache, .. } = &mut row;
            let active = lease
                .as_mut()
                .ok_or_else(|| Error::InferenceError("Voxtral offline row lost its state".into()))?
                .require_state_mut()?;
            let cache = cache.as_mut().ok_or_else(|| {
                Error::InferenceError("Voxtral offline row lost its pages".into())
            })?;
            active
                .model
                .prefill_realtime_in_quantum(&mut active.state, cache)
        });
        let step = match prefill {
            Ok(step) => {
                row.active_mut()?
                    .clock
                    .record_prefill(prefill_started.elapsed());
                step
            }
            Err(error) => {
                return match row.rollback() {
                    Ok(()) => Err(error),
                    Err(rollback) => Err(Error::InferenceError(format!(
                        "{error}; Voxtral offline prefill rollback also failed: {rollback}"
                    ))),
                };
            }
        };
        let mut rows = [row];
        let outcome = settle_voxtral_offline_rows(&mut rows, vec![step])
            .pop()
            .ok_or_else(|| Error::InferenceError("Voxtral offline prefill lost its row".into()))?;
        let [row] = rows;
        self.publish_voxtral_offline_outcomes(vec![(row, outcome)])?
            .pop()
            .ok_or_else(|| Error::InferenceError("Voxtral offline prefill lost its output".into()))
    }

    /// One continuous decode step of Voxtral offline rows: every row that
    /// still has audio frames decodes its next one in one language-model
    /// forward; a row whose frames ran out completes without one. Each row
    /// then settles alone (a cancelled row rolls back only its own quantum).
    pub(in crate::engine::executor) fn voxtral_asr_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Voxtral offline decode batch rows do not match".into(),
            ));
        }
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
        let mut rows = Vec::with_capacity(scheduled.len());
        for (index, (request, row)) in requests.iter().zip(scheduled).enumerate() {
            if row.is_prefill {
                return Err(Error::InvalidInput(
                    "Voxtral offline decode batch received prefill work".into(),
                ));
            }
            if request.is_cancelled() {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(request.id.clone()),
                ));
                continue;
            }
            match self.voxtral_offline_row(index, request, row, managed[index].take()) {
                Ok(row) => rows.push(row),
                Err(error) => {
                    outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                        request.id.clone(),
                        format!("Voxtral offline row could not start its step: {error}"),
                    )));
                }
            }
        }
        // A row cancelled since it was scheduled, or loaded by another model
        // instance, leaves the batch before the forward, alone.
        let model = match rows.first_mut() {
            Some(row) => Some(row.active_mut()?.model.clone()),
            None => None,
        };
        let mut settled = Vec::new();
        let mut live = Vec::with_capacity(rows.len());
        for mut row in rows {
            let foreign = match (&model, row.active_mut()) {
                (Some(model), Ok(active)) => {
                    !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
                }
                _ => true,
            };
            if row.cancelled() || foreign {
                let outcome = match row.rollback() {
                    Ok(()) if !foreign => VoxtralOfflineRowOutcome::Cancelled,
                    Ok(()) => VoxtralOfflineRowOutcome::Failed(
                        "Voxtral offline batch crossed model identity".into(),
                    ),
                    Err(error) => VoxtralOfflineRowOutcome::Failed(format!(
                        "Voxtral offline rollback failed: {error}"
                    )),
                };
                settled.push((row, outcome));
            } else {
                live.push(row);
            }
        }
        if let Some(model) = model.filter(|_| !live.is_empty()) {
            let decode_started = std::time::Instant::now();
            let forward = {
                let mut batch = live
                    .iter_mut()
                    .map(|row| {
                        let VoxtralOfflineRow { lease, cache, .. } = row;
                        let active = lease
                            .as_mut()
                            .ok_or_else(|| {
                                Error::InferenceError("Voxtral offline row lost its state".into())
                            })?
                            .require_state_mut()?;
                        let cache = cache.as_mut().ok_or_else(|| {
                            Error::InferenceError("Voxtral offline row lost its pages".into())
                        })?;
                        Ok(VoxtralRealtimeDecodeBatchRow {
                            state: &mut active.state,
                            cache,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Self::run_blocking(|| model.decode_realtime_step_batch(&mut batch))
            };
            let steps = match forward {
                Ok(steps) => steps,
                Err(error) => {
                    // The forward failed for every row it held: roll each
                    // back to its quantum's start.
                    let mut rollback_errors = Vec::new();
                    for row in &mut live {
                        if let Err(rollback) = row.rollback() {
                            rollback_errors.push(rollback.to_string());
                        }
                    }
                    return if rollback_errors.is_empty() {
                        Err(error)
                    } else {
                        Err(Error::InferenceError(format!(
                            "{error}; Voxtral offline rollback also failed: {}",
                            rollback_errors.join("; ")
                        )))
                    };
                }
            };
            let decode_elapsed = decode_started.elapsed();
            for row in &mut live {
                row.active_mut()?.clock.record_decode(decode_elapsed);
            }
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::NativeTensor {
                    mode: crate::engine::NativeBatchMode::Continuous,
                    rows: live.len(),
                },
            );
            let outcomes = settle_voxtral_offline_rows(&mut live, steps);
            settled.extend(live.into_iter().zip(outcomes));
        }
        for (index, output) in self.publish_voxtral_offline_settled(settled)? {
            outputs[index] = Some(output);
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("Voxtral offline row produced no result".into())
                })
            })
            .collect()
    }

    /// Register every sealed row's quantum with the realtime coordinator,
    /// which commits or aborts it on Core's decision, and release the rows
    /// that ended. A registration failure aborts every sealed quantum.
    fn publish_voxtral_offline_settled(
        &self,
        settled: Vec<(
            VoxtralOfflineRow<'_>,
            VoxtralOfflineRowOutcome<SealedVoxtralOfflineRow>,
        )>,
    ) -> Result<Vec<(usize, ModelSessionResult)>> {
        let mut pending = Vec::new();
        let mut outputs = Vec::with_capacity(settled.len());
        for (row, outcome) in settled {
            let index = row.index;
            let request_id = row.request.id.clone();
            match outcome {
                VoxtralOfflineRowOutcome::Sealed((plan_id, quantum, result)) => {
                    pending.push((plan_id, quantum));
                    outputs.push((index, result));
                }
                VoxtralOfflineRowOutcome::Cancelled => {
                    row.release()?;
                    outputs.push((
                        index,
                        ModelSessionResult::cancelled(ExecutorOutput::cancelled(request_id)),
                    ));
                }
                VoxtralOfflineRowOutcome::Failed(message) => {
                    // A failed rollback leaves the row's lease dirty, which
                    // fences its session until cleanup.
                    drop(row);
                    outputs.push((
                        index,
                        ModelSessionResult::sequence(ExecutorOutput::error(request_id, message)),
                    ));
                }
            }
        }
        if let Err((error, pending)) = self.voxtral_realtime.register_batch(pending) {
            let mut rollback_errors = Vec::new();
            for (_, pending) in pending {
                let session = pending.session.clone();
                if let Err(rollback) = self
                    .voxtral_realtime
                    .resolve_pending(pending, super::super::PendingQuantumDecision::Abort)
                    .and_then(|prepared| {
                        self.voxtral_realtime
                            .replace_in_flight(&session, prepared.replacement)
                    })
                {
                    rollback_errors.push(rollback.to_string());
                }
            }
            return if rollback_errors.is_empty() {
                Err(error)
            } else {
                Err(Error::InferenceError(format!(
                    "{error}; Voxtral offline abort also failed: {}",
                    rollback_errors.join("; ")
                )))
            };
        }
        Ok(outputs)
    }

    fn publish_voxtral_offline_outcomes(
        &self,
        settled: Vec<(
            VoxtralOfflineRow<'_>,
            VoxtralOfflineRowOutcome<SealedVoxtralOfflineRow>,
        )>,
    ) -> Result<Vec<ModelSessionResult>> {
        Ok(self
            .publish_voxtral_offline_settled(settled)?
            .into_iter()
            .map(|(_, output)| output)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::{settle_voxtral_offline_rows, VoxtralOfflineRowOutcome, VoxtralOfflineStepRow};
    use crate::error::{Error, Result};
    use crate::models::architectures::voxtral::realtime::VoxtralRealtimeStep;

    /// A row of a step: cancelled before its seal, cancelled while it
    /// seals, or failing its seal, with counts of what happened to it.
    #[derive(Default)]
    struct FakeRow {
        cancelled: Arc<AtomicBool>,
        cancel_while_sealing: bool,
        seal_fails: bool,
        seals: usize,
        rollbacks: usize,
    }

    impl VoxtralOfflineStepRow for FakeRow {
        type Sealed = (String, bool);

        fn cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Acquire)
        }

        fn seal(&mut self, step: VoxtralRealtimeStep) -> Result<Self::Sealed> {
            self.seals += 1;
            if self.cancel_while_sealing {
                self.cancelled.store(true, Ordering::Release);
                return Err(Error::Cancelled("cancelled while sealing".into()));
            }
            if self.seal_fails {
                return Err(Error::InferenceError("stream staging failed".into()));
            }
            Ok((step.text, step.finished))
        }

        fn rollback(&mut self) -> Result<()> {
            self.rollbacks += 1;
            Ok(())
        }
    }

    fn step(text: &str, finished: bool) -> VoxtralRealtimeStep {
        VoxtralRealtimeStep {
            delta: text.into(),
            text: text.into(),
            tokens_generated: 1,
            finished,
        }
    }

    /// One step's rows settle alone: a continuing row and a finishing row
    /// seal; a row cancelled before its seal, a row cancelled while it
    /// seals and a row whose seal fails each roll back their own quantum
    /// once, and no other row rolls back.
    #[test]
    fn offline_rows_settle_and_roll_back_alone() {
        let mut rows = vec![
            FakeRow::default(),
            FakeRow::default(),
            FakeRow::default(),
            FakeRow {
                cancel_while_sealing: true,
                ..FakeRow::default()
            },
            FakeRow {
                seal_fails: true,
                ..FakeRow::default()
            },
        ];
        rows[2].cancelled.store(true, Ordering::Release);
        let outcomes = settle_voxtral_offline_rows(
            &mut rows,
            vec![
                step("going on", false),
                step("finished", true),
                step("cancelled", false),
                step("cancelled late", false),
                step("failed", false),
            ],
        );
        assert_eq!(
            outcomes,
            vec![
                VoxtralOfflineRowOutcome::Sealed(("going on".into(), false)),
                VoxtralOfflineRowOutcome::Sealed(("finished".into(), true)),
                VoxtralOfflineRowOutcome::Cancelled,
                VoxtralOfflineRowOutcome::Cancelled,
                VoxtralOfflineRowOutcome::Failed("Inference error: stream staging failed".into()),
            ]
        );
        assert_eq!(
            rows.iter().map(|row| row.rollbacks).collect::<Vec<_>>(),
            [0, 0, 1, 1, 1]
        );
        // A row cancelled before its seal never seals.
        assert_eq!(
            rows.iter().map(|row| row.seals).collect::<Vec<_>>(),
            [1, 1, 0, 1, 1]
        );
    }

    /// A step that returns the wrong number of rows rolls every row back
    /// and fails each.
    #[test]
    fn offline_step_with_missing_rows_rolls_back_every_row() {
        let mut rows = vec![FakeRow::default(), FakeRow::default()];
        let outcomes = settle_voxtral_offline_rows(&mut rows, vec![step("one", false)]);
        assert!(outcomes
            .iter()
            .all(|outcome| matches!(outcome, VoxtralOfflineRowOutcome::Failed(_))));
        assert!(rows.iter().all(|row| row.rollbacks == 1 && row.seals == 0));
    }
}
