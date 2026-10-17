//! Parakeet ASR: continuous recurrent prefill and decode batches.

use super::*;

pub(super) fn late_cancelled_parakeet_rows(cancelled: &[bool], armed: &[bool]) -> Vec<usize> {
    debug_assert_eq!(cancelled.len(), armed.len());
    cancelled
        .iter()
        .zip(armed)
        .enumerate()
        .filter_map(|(row, (cancelled, armed))| (*cancelled && *armed).then_some(row))
        .collect()
}

pub(super) fn rollback_parakeet_host_state<T>(state: &mut T, checkpoint: &mut Option<T>) -> bool {
    let Some(checkpoint) = checkpoint.take() else {
        return false;
    };
    *state = checkpoint;
    true
}

pub(super) struct ContinuousParakeetAsrRow<'a> {
    pub(super) index: usize,
    pub(super) session: SessionKey,
    pub(super) lease: ExecutorStateLease<'a, ActiveParakeetAsrDecode>,
    pub(super) checkpoint: Option<ParakeetRetainedDecodeState>,
}

pub(super) struct ContinuousParakeetAsrStateBatch<'a> {
    pub(super) rows: Vec<ContinuousParakeetAsrRow<'a>>,
    pub(super) armed: bool,
}

impl<'a> ContinuousParakeetAsrStateBatch<'a> {
    pub(super) fn new(rows: Vec<ContinuousParakeetAsrRow<'a>>) -> Self {
        Self { rows, armed: true }
    }

    pub(super) fn rollback_row(&mut self, row: usize) -> Result<usize> {
        let row = self.rows.get_mut(row).ok_or_else(|| {
            Error::InferenceError("continuous Parakeet ASR rollback row is out of range".into())
        })?;
        let active = row.lease.require_state_mut()?;
        if !rollback_parakeet_host_state(&mut active.state, &mut row.checkpoint) {
            return Err(Error::InferenceError(
                "continuous Parakeet ASR row has no armed checkpoint".into(),
            ));
        }
        row.lease.mark_clean();
        Ok(row.index)
    }

    pub(super) fn validate_publication(&self) -> Result<()> {
        for row in &self.rows {
            row.lease.validate_defer()?;
        }
        Ok(())
    }

    pub(super) fn commit(mut self) -> Result<Vec<(usize, ExecutorStateLease<'a, ActiveParakeetAsrDecode>)>> {
        for row in &mut self.rows {
            row.checkpoint.take();
            row.lease.mark_clean();
        }
        self.armed = false;
        Ok(std::mem::take(&mut self.rows)
            .into_iter()
            .map(|row| (row.index, row.lease))
            .collect())
    }
}

impl Drop for ContinuousParakeetAsrStateBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            let rollback = row
                .lease
                .require_state_mut()
                .map(|active| rollback_parakeet_host_state(&mut active.state, &mut row.checkpoint));
            match rollback {
                Ok(true) => row.lease.mark_clean(),
                Ok(false) => {}
                Err(error) => tracing::error!(
                    request_id = %row.session.request_id,
                    epoch = row.session.epoch,
                    %error,
                    "continuous Parakeet ASR rollback failed; state fenced until cleanup"
                ),
            }
        }
        self.rows.clear();
    }
}

pub(super) struct ParakeetPrefillHostBatch<'a> {
    pub(super) rows: Vec<(usize, ExecutorStateLease<'a, ActiveParakeetAsrDecode>)>,
    pub(super) armed: bool,
}

impl<'a> ParakeetPrefillHostBatch<'a> {
    pub(super) fn new() -> Self {
        Self {
            rows: Vec::new(),
            armed: true,
        }
    }

    pub(super) fn push(&mut self, index: usize, lease: ExecutorStateLease<'a, ActiveParakeetAsrDecode>) {
        self.rows.push((index, lease));
    }

    pub(super) fn validate_publication(&self) -> Result<()> {
        for (_, lease) in &self.rows {
            lease.validate_defer()?;
        }
        Ok(())
    }

    pub(super) fn commit(mut self) -> Vec<(usize, ExecutorStateLease<'a, ActiveParakeetAsrDecode>)> {
        self.armed = false;
        std::mem::take(&mut self.rows)
    }
}

impl Drop for ParakeetPrefillHostBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for (_, lease) in &mut self.rows {
            lease.discard_state();
            lease.mark_clean();
        }
        self.rows.clear();
    }
}

impl NativeExecutor {
    pub(in crate::engine::executor) fn parakeet_asr_prefill_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Parakeet prefill batch geometry is inconsistent".into(),
            ));
        }
        let mut outputs = (0..requests.len()).map(|_| None).collect::<Vec<_>>();
        let live = requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| {
                if request.is_cancelled() {
                    outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    None
                } else {
                    Some(index)
                }
            })
            .collect::<Vec<_>>();
        if live.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("Parakeet prefill lost cancelled output".into())
                    })
                })
                .collect();
        }
        let model_lease = requests[live[0]]
            .prepared_asr_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Parakeet prefill lost model residency".into()))?;
        let model_arc = model_lease.model_arc();
        let NativeAsrModel::Parakeet(model) = model_arc.as_ref() else {
            return Err(Error::InvalidInput(
                "Parakeet prefill crossed model family".into(),
            ));
        };
        let mut rows = Vec::with_capacity(live.len());
        for index in live.iter().copied() {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Parakeet prefill row lost model residency".into())
                })?;
            if variant.family() != ModelFamily::ParakeetAsr
                || !Arc::ptr_eq(&row_model.model_arc(), &model_arc)
                || !scheduled[index].is_prefill
                || scheduled[index].num_computed_tokens != 0
            {
                return Err(Error::InvalidInput(
                    "Parakeet prefill batch crossed its loaded sequence identity".into(),
                ));
            }
            if managed[index].is_some() {
                return Err(Error::InferenceError(
                    "cacheless Parakeet prefill received managed state".into(),
                ));
            }
            let artifact = request
                .prepared_parakeet_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Parakeet prefill lost encoder artifact".into())
                })?;
            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError("Parakeet prefill lost decoded audio".into())
                })?;
            let lease = ExecutorStateLease::checkout(
                &self.parakeet_asr_decode_states,
                scheduled[index].session_key(),
                variant,
                "Parakeet ASR prefill",
            )?;
            if lease.state().is_some() {
                return Err(Error::InferenceError(
                    "Parakeet initial prefill found an existing session state".into(),
                ));
            }
            rows.push((
                index,
                variant,
                row_model,
                artifact,
                samples.len(),
                sample_rate,
                lease,
            ));
        }
        let states = Self::run_blocking(|| model.start_retained_decode_batch(rows.len()))?;
        if states.len() != rows.len() {
            return Err(Error::InferenceError(
                "Parakeet native prefill returned the wrong row count".into(),
            ));
        }
        if let Some(call) =
            retained_asr_batch_model_call(crate::engine::NativeBatchMode::Static, states.len())
        {
            crate::engine::metrics::record_engine_model_call(call);
        }
        let mut cancelled = vec![false; requests.len()];
        for row in &rows {
            cancelled[row.0] = requests[row.0].is_cancelled();
        }
        for row in &rows {
            cancelled[row.0] |= requests[row.0].is_cancelled();
        }

        let mut host = ParakeetPrefillHostBatch::new();
        let mut continuing = vec![false; requests.len()];
        for ((index, variant, row_model, artifact, sample_count, sample_rate, mut lease), state) in
            rows.into_iter().zip(states)
        {
            let request = requests[index];
            if cancelled[index] {
                lease.release()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                continue;
            }
            lease.install_state(ActiveParakeetAsrDecode {
                variant,
                model: row_model,
                artifact,
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: sample_count,
            })?;
            continuing[index] = true;
            host.push(index, lease);
            outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(Vec::new(), sample_rate)),
                text: Some(String::new()),
                input_transcription: None,
                tokens_processed: scheduled[index].num_tokens,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            }));
        }
        let host_cancelled = host
            .rows
            .iter()
            .map(|(index, _)| requests[*index].is_cancelled())
            .collect::<Vec<_>>();
        let host_armed = host
            .rows
            .iter()
            .map(|(_, lease)| lease.state().is_some())
            .collect::<Vec<_>>();
        for row in late_cancelled_parakeet_rows(&host_cancelled, &host_armed) {
            let index = host.rows[row].0;
            continuing[index] = false;
            outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                requests[index].id.clone(),
            )));
        }
        host.validate_publication()?;
        for (index, lease) in host.commit() {
            if continuing[index] {
                lease.restore()?;
            } else {
                lease.release()?;
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| Error::InferenceError("Parakeet prefill lost output".into()))
            })
            .collect()
    }

    pub(super) fn parakeet_asr_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        let mut outputs = (0..requests.len()).map(|_| None).collect::<Vec<_>>();
        let live = requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| {
                if request.is_cancelled() {
                    outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    None
                } else {
                    Some(index)
                }
            })
            .collect::<Vec<_>>();
        if live.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("Parakeet decode lost cancelled output".into())
                    })
                })
                .collect();
        }
        let model_lease = requests[live[0]]
            .prepared_asr_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Parakeet decode lost model residency".into()))?;
        let model_arc = model_lease.model_arc();
        let NativeAsrModel::Parakeet(model) = model_arc.as_ref() else {
            return Err(Error::InvalidInput(
                "Parakeet decode crossed model family".into(),
            ));
        };
        let mut rows = Vec::with_capacity(live.len());
        for index in live.iter().copied() {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Parakeet decode row lost model residency".into())
                })?;
            if variant.family() != ModelFamily::ParakeetAsr
                || !Arc::ptr_eq(&row_model.model_arc(), &model_arc)
                || scheduled[index].is_prefill
            {
                return Err(Error::InvalidInput(
                    "Parakeet decode batch crossed its loaded sequence identity".into(),
                ));
            }
            if managed[index].is_some() {
                return Err(Error::InferenceError(
                    "cacheless Parakeet decode received managed state".into(),
                ));
            }
            let mut lease = ExecutorStateLease::checkout(
                &self.parakeet_asr_decode_states,
                scheduled[index].session_key(),
                variant,
                "Parakeet ASR decode",
            )?;
            let active = lease.require_state_mut()?;
            if active.variant != variant
                || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
                || !Arc::ptr_eq(
                    &active.artifact,
                    &request
                        .prepared_parakeet_artifact_for_executor()?
                        .ok_or_else(|| {
                            Error::InferenceError("Parakeet decode lost encoder artifact".into())
                        })?,
                )
            {
                return Err(Error::InferenceError(
                    "Parakeet decode state identity differs from its request".into(),
                ));
            }
            let checkpoint = active.state.clone();
            lease.mark_dirty();
            rows.push(ContinuousParakeetAsrRow {
                index,
                session: scheduled[index].session_key(),
                lease,
                checkpoint: Some(checkpoint),
            });
        }
        let mut active_states = ContinuousParakeetAsrStateBatch::new(rows);
        let mut batch = active_states
            .rows
            .iter_mut()
            .map(|row| {
                let max_tokens = requests[row.index].params.max_tokens;
                let active = row.lease.require_state_mut()?;
                Ok(ParakeetRetainedDecodeBatchRow {
                    artifact: active.artifact.as_ref(),
                    state: &mut active.state,
                    max_tokens,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let steps = Self::run_blocking(|| model.retained_decode_step_batch(&mut batch));
        drop(batch);
        let steps = steps?;
        if steps.len() != active_states.rows.len() {
            return Err(Error::InferenceError(
                "Parakeet native decode returned the wrong row count".into(),
            ));
        }
        if let Some(call) =
            retained_asr_batch_model_call(crate::engine::NativeBatchMode::Continuous, steps.len())
        {
            crate::engine::metrics::record_engine_model_call(call);
        }
        let mut steps = steps.into_iter().map(Some).collect::<Vec<_>>();
        let mut continuing = vec![false; requests.len()];
        let sealing = (|| -> Result<()> {
            for row in 0..active_states.rows.len() {
                let index = active_states.rows[row].index;
                let request = requests[index];
                if request.is_cancelled() {
                    let index = active_states.rollback_row(row)?;
                    outputs[index] = Some(ModelSessionResult::cancelled(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    continue;
                }
                let step = steps[row].take().ok_or_else(|| {
                    Error::InferenceError("Parakeet decode lost a native batch row".into())
                })?;
                let active = active_states.rows[row].lease.require_state_mut()?;
                let generated = step
                    .tokens_generated
                    .saturating_sub(active.last_tokens_generated);
                active.last_tokens_generated = step.tokens_generated;
                if let Some(tx) = Self::stream_sender(request) {
                    if !step.delta.is_empty() {
                        Self::stream_text_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active.stream_sequence,
                            step.delta.clone(),
                        )?;
                    }
                    if step.finished {
                        Self::stream_final_marker_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active.stream_sequence,
                        )?;
                    }
                }
                let diagnostics = step
                    .finished
                    .then(|| {
                        model
                            .retained_decode_output(active.artifact.as_ref(), &active.state)
                            .diagnostics
                    })
                    .flatten();
                let sample_rate = active.input_sample_rate;
                let sample_count = active.input_sample_count;
                continuing[index] = !step.finished;
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::new(Vec::new(), sample_rate)),
                    text: Some(step.text),
                    input_transcription: None,
                    tokens_processed: scheduled[index].num_tokens,
                    tokens_generated: generated,
                    finished: step.finished,
                    phase_timing_override: None,
                    asr_diagnostics: diagnostics,
                    error: None,
                    logprobs: Vec::new(),
                }));
                let _ = sample_count;
            }
            Ok(())
        })();
        if let Err(error) = sealing {
            for row in &active_states.rows {
                let _ = requests[row.index].take_staged_stream_outputs();
            }
            return Err(error);
        }
        // A peer row may finish its native call or stream staging after an
        // earlier row was cancelled. Recheck every armed checkpoint before any
        // host state is made visible; cancellation restores only that row while
        // an error still rolls the complete cohort back through the guard.
        let host_cancelled = active_states
            .rows
            .iter()
            .map(|row| requests[row.index].is_cancelled())
            .collect::<Vec<_>>();
        let host_armed = active_states
            .rows
            .iter()
            .map(|row| row.checkpoint.is_some())
            .collect::<Vec<_>>();
        for row in late_cancelled_parakeet_rows(&host_cancelled, &host_armed) {
            let index = active_states.rows[row].index;
            let _ = requests[index].take_staged_stream_outputs()?;
            let index = active_states.rollback_row(row)?;
            continuing[index] = false;
            outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                requests[index].id.clone(),
            )));
        }
        if let Err(error) = active_states.validate_publication() {
            for row in &active_states.rows {
                let _ = requests[row.index].take_staged_stream_outputs();
            }
            return Err(error);
        }
        let committed = active_states.commit()?;
        for (index, lease) in committed {
            if continuing[index] {
                lease.restore()?;
            } else {
                lease.release()?;
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| Error::InferenceError("Parakeet decode lost output".into()))
            })
            .collect()
    }
}
