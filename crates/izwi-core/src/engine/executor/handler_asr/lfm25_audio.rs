//! LFM2.5-Audio ASR: long-form, sequence and continuous prefill/decode batches.

use super::*;

pub(super) fn late_cancelled_lfm_asr_prefill_rows(cancelled: &[bool], armed: &[bool]) -> Vec<usize> {
    cancelled
        .iter()
        .zip(armed)
        .enumerate()
        .filter_map(|(row, (cancelled, armed))| (*cancelled && *armed).then_some(row))
        .collect()
}

/// LFM2.5 Audio sizes each long-form chunk's decode budget from the chunk's
/// duration. Only a caller-set limit overrides that; the request default would
/// otherwise replace it for every chunk.
pub(super) fn lfm25_audio_asr_long_form_max_tokens(request: &EngineCoreRequest) -> Option<usize> {
    (!request.asr_auto_max_tokens).then_some(request.params.max_tokens)
}

pub(super) struct ContinuousLfm25AsrRow<'a> {
    pub(super) index: usize,
    pub(super) session: SessionKey,
    pub(super) lease: ExecutorStateLease<'a, ActiveLfm25AsrDecode>,
    pub(super) cache: crate::models::shared::attention::physical::PhysicalPagedKvCache,
    pub(super) checkpoint: Option<Lfm25AudioAsrQuantumCheckpoint>,
    pub(super) prior_last_tokens_generated: usize,
    pub(super) prior_stream_sequence: usize,
}

pub(super) struct ContinuousLfm25AsrStateBatch<'a> {
    pub(super) rows: Vec<ContinuousLfm25AsrRow<'a>>,
    pub(super) armed: bool,
}

impl<'a> ContinuousLfm25AsrStateBatch<'a> {
    pub(super) fn new(rows: Vec<ContinuousLfm25AsrRow<'a>>) -> Self {
        Self { rows, armed: true }
    }

    pub(super) fn rollback_row(&mut self, row: usize) -> Result<usize> {
        let row = self.rows.get_mut(row).ok_or_else(|| {
            Error::InferenceError("continuous LFM2.5 ASR rollback row is out of range".into())
        })?;
        let checkpoint = row.checkpoint.take().ok_or_else(|| {
            Error::InferenceError("continuous LFM2.5 ASR row has no armed checkpoint".into())
        })?;
        let state = row.lease.require_state_mut()?;
        state.state.rollback_quantum(&mut row.cache, &checkpoint)?;
        state.last_tokens_generated = row.prior_last_tokens_generated;
        state.stream_sequence = row.prior_stream_sequence;
        row.lease.mark_clean();
        Ok(row.index)
    }

    pub(super) fn commit(
        mut self,
    ) -> Result<
        Vec<(
            usize,
            ExecutorStateLease<'a, ActiveLfm25AsrDecode>,
            Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
        )>,
    > {
        let mut completed = Vec::with_capacity(self.rows.len());
        for row in &mut self.rows {
            let Some(checkpoint) = row.checkpoint.take() else {
                continue;
            };
            let completions = row.cache.take_completed_writes();
            row.lease
                .require_state_mut()?
                .state
                .commit_quantum(&row.cache, &checkpoint)?;
            row.lease.mark_clean();
            completed.push((row.index, completions));
        }
        self.armed = false;
        let mut rows = std::mem::take(&mut self.rows);
        Ok(completed
            .into_iter()
            .map(|(index, completions)| {
                let position = rows
                    .iter()
                    .position(|row| row.index == index)
                    .expect("committed LFM2.5 row remains present");
                let row = rows.swap_remove(position);
                (index, row.lease, completions)
            })
            .collect())
    }
}

impl Drop for ContinuousLfm25AsrStateBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            let Some(checkpoint) = row.checkpoint.take() else {
                continue;
            };
            let rollback = row.lease.require_state_mut().and_then(|state| {
                state.state.rollback_quantum(&mut row.cache, &checkpoint)?;
                state.last_tokens_generated = row.prior_last_tokens_generated;
                state.stream_sequence = row.prior_stream_sequence;
                Ok(())
            });
            match rollback {
                Ok(()) => row.lease.mark_clean(),
                Err(error) => tracing::error!(
                    request_id = %row.session.request_id,
                    epoch = row.session.epoch,
                    %error,
                    "continuous LFM2.5 ASR rollback failed; state fenced until cleanup"
                ),
            }
        }
        self.rows.clear();
    }
}

impl NativeExecutor {
    /// Run audio that LFM2.5 Audio preparation planned as several chunks. The
    /// model chunks, decodes and merges internally, using the prepared lease
    /// and the atomic stage's main-backbone invocation workspace.
    pub(super) fn lfm25_audio_asr_long_form_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        let model = request
            .prepared_lfm25_audio_asr_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio long-form ASR lost model residency".into())
            })?;
        // One chunk per dispatch: resume the run the committed chunks left.
        let session = scheduled.session_key();
        let resumed = self.long_form_asr.committed(&session);
        let (execution_audio, sample_rate, audio_decode_ms) = match resumed.as_ref() {
            Some(resumed) => (
                AsrExecutionAudio::Prepared(resumed.samples.clone()),
                resumed.sample_rate,
                resumed.audio_decode_ms,
            ),
            None => resolve_asr_execution_audio(request, ModelFamily::Lfm25Audio, || {
                Self::run_blocking(|| decode_request_audio_with_rate(request))
            })?,
        };
        let samples = execution_audio.samples();
        let max_tokens = lfm25_audio_asr_long_form_max_tokens(request);
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let mut leases =
            super::super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
        let chunk_plan = asr_execution_route_plan(request, samples, sample_rate, || {
            Ok(model.asr_route_capabilities())
        })?;
        let transcription = Self::run_blocking(|| {
            let transcribe_chunk =
                |chunk_audio: &[f32],
                 sr: u32,
                 _prefix_text: &str,
                 partial: Option<&mut dyn FnMut(&str)>| {
                    let chunk_secs = chunk_audio.len() as f32 / sr.max(1) as f32;
                    let options = NativeAsrGenerationOptions {
                        max_new_tokens: model
                            .lfm25_audio_asr_chunk_max_new_tokens(chunk_secs, max_tokens),
                        ..NativeAsrGenerationOptions::default()
                    };
                    let mut no_partials = |_: &str| {};
                    let emit: &mut dyn FnMut(&str) = match partial {
                        Some(emit) => emit,
                        None => &mut no_partials,
                    };
                    let output = model
                        .transcribe_single_pass_with_callback_and_options_from_invocation_workspace(
                            chunk_audio,
                            sr,
                            options,
                            &mut leases,
                            emit,
                        )?;
                    Ok(AsrChunkTranscription {
                        text: output.text,
                        diagnostics: output.diagnostics,
                    })
                };
            if !chunk_plan.chunks.is_empty() {
                return Self::run_long_form_chunk(
                    request,
                    resumed,
                    &execution_audio,
                    sample_rate,
                    audio_decode_ms,
                    stream_tx.as_ref(),
                    stream_policy,
                    &chunk_plan,
                    transcribe_chunk,
                );
            }
            let mut sequence = 0usize;
            let chunked = Self::transcribe_with_chunk_plan_with_streaming_details_and_options(
                &request.id,
                &|| request.is_cancelled(),
                stream_tx.as_ref(),
                stream_policy,
                &mut sequence,
                samples,
                sample_rate,
                &chunk_plan.chunks,
                &chunk_plan.config,
                Self::asr_chunk_stream_options(),
                transcribe_chunk,
            )?;
            Ok(Self::complete_chunked_asr(&chunk_plan, chunked))
        });
        let Some(transcription) = settle_atomic_asr_cancellation(request, transcription)? else {
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        };
        let completions = leases.release()?;
        if completions.len() != 2 {
            return Err(Error::InferenceError(
                "LFM2.5 Audio long-form ASR returned an incomplete physical-state completion set"
                    .into(),
            ));
        }
        let (text, diagnostics) = match transcription {
            AsrTranscriptionStep::Complete(text, diagnostics) => (text, diagnostics),
            AsrTranscriptionStep::Yielded(state) => {
                return self.yield_long_form_chunk(request, scheduled, session, *state);
            }
        };
        let samples_len = samples.len();

        Ok(ModelSessionResult::atomic(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: samples_len as f32 / sample_rate as f32,
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (samples_len / 256).max(1),
            finished: true,
            phase_timing_override: Some(ExecutorPhaseTiming::with_media_decode_ms(audio_decode_ms)),
            asr_diagnostics: Self::with_audio_decode_timing(diagnostics, audio_decode_ms),
            error: None,
            logprobs: Vec::new(),
        }))
    }

    pub(super) fn lfm25_audio_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Lfm25Audio || request.uses_asr_long_form_atomic() {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio retained executor received a foreign or long-form request".into(),
            ));
        }
        let model = request
            .prepared_lfm25_audio_asr_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio ASR sequence lost model residency".into())
            })?;
        let session = scheduled.session_key();
        let mut state_lease = ExecutorStateLease::checkout(
            &self.lfm25_asr_decode_states,
            session,
            variant,
            "LFM2.5 Audio ASR decode",
        )?;
        if state_lease.state().is_some_and(|active| {
            active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        }) {
            state_lease.discard_state();
        }

        let mut retained = retained.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio ASR lost retained physical state".into())
        })?;
        let tensor_reservation = retained.tensor_state.clone().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio ASR lost its ShortConv reservation".into())
        })?;
        let tensor_arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state())
            .ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio ASR lost its ShortConv arena".into())
            })?;
        let mut cache = retained.take_only_paged()?;
        retained.ensure_all_paged_consumed()?;

        let mut fresh_state = false;
        if state_lease.state().is_none() {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "LFM2.5 Audio ASR sequence lost state before initial prefill".into(),
                ));
            }
            let artifact = request
                .prepared_lfm25_audio_asr_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "LFM2.5 Audio ASR sequence lost its prepared prompt artifact".into(),
                    )
                })?;
            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError("LFM2.5 Audio ASR sequence lost decoded audio".into())
                })?;
            if artifact.prompt_tokens != request.num_prompt_tokens()
                || artifact.source_samples != samples.len()
                || artifact.source_sample_rate != sample_rate
            {
                return Err(Error::InferenceError(
                    "LFM2.5 Audio ASR artifact differs from scheduler admission".into(),
                ));
            }
            let state = model
                .new_lfm25_audio_retained_asr_state(artifact, request.params.max_tokens.max(1))?;
            state_lease.install_state(ActiveLfm25AsrDecode {
                variant,
                model: model.clone(),
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: samples.len(),
            })?;
            fresh_state = true;
        }

        let prior = {
            let active = state_lease.require_state_mut()?;
            if active.variant != variant
                || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
                || active.state.prompt_tokens() != request.num_prompt_tokens()
            {
                return Err(Error::InferenceError(
                    "LFM2.5 Audio ASR state crossed loaded model identity".into(),
                ));
            }
            active
                .state
                .bind_tensor_sequence(tensor_reservation.sequence)?;
            active.state.restore_shortconv(tensor_arena)?;
            (active.last_tokens_generated, active.stream_sequence)
        };
        let checkpoint = state_lease
            .require_state_mut()?
            .state
            .begin_quantum(&cache)?;
        state_lease.mark_dirty();
        if request.is_cancelled() {
            let active = state_lease.require_state_mut()?;
            active.state.rollback_quantum(&mut cache, &checkpoint)?;
            active.last_tokens_generated = prior.0;
            active.stream_sequence = prior.1;
            state_lease.mark_clean();
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        let execution = (|| -> Result<Lfm25AudioAsrDecodeStep> {
            let active = state_lease.require_state_mut()?;
            let step = if scheduled.is_prefill {
                let (start, end) =
                    resumable_asr_prefill_span(scheduled, active.state.prompt_tokens())?;
                if active.state.prefill_cursor() != start {
                    return Err(Error::InferenceError(
                        "LFM2.5 Audio ASR prefill cursor differs from scheduler admission".into(),
                    ));
                }
                let prefill = Self::run_blocking(|| {
                    model.lfm25_audio_asr_prefill_step(
                        &mut active.state,
                        &mut cache,
                        &checkpoint,
                        end - start,
                    )
                })?;
                if prefill.consumed_tokens != end - start
                    || prefill.prefill_cursor != end
                    || prefill.prompt_tokens != active.state.prompt_tokens()
                    || prefill.complete != (end == prefill.prompt_tokens)
                {
                    return Err(Error::InferenceError(
                        "LFM2.5 Audio ASR prefill returned inconsistent progress".into(),
                    ));
                }
                Lfm25AudioAsrDecodeStep {
                    cache_append_token: None,
                    visible_token: None,
                    delta: String::new(),
                    text: active.state.text().to_string(),
                    tokens_generated: active.last_tokens_generated,
                    finished: prefill.finished,
                    stop_reason: prefill
                        .finished
                        .then_some(Lfm25AudioAsrStopReason::StopToken),
                    pending_token: prefill.pending_token,
                }
            } else {
                Self::run_blocking(|| {
                    model.lfm25_audio_asr_decode_step(&mut active.state, &mut cache, &checkpoint)
                })?
            };
            if request.is_cancelled() {
                return Err(Error::Cancelled(request.id.clone()));
            }
            active
                .state
                .stage_shortconv(tensor_arena, scheduled.plan_id)?;
            Ok(step)
        })();

        let step = match execution {
            Ok(step) if !request.is_cancelled() => step,
            Ok(_) | Err(Error::Cancelled(_)) => {
                let _ = request.take_staged_stream_outputs()?;
                let active = state_lease.require_state_mut()?;
                active.state.rollback_quantum(&mut cache, &checkpoint)?;
                active.last_tokens_generated = prior.0;
                active.stream_sequence = prior.1;
                state_lease.mark_clean();
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            Err(error) => {
                let _ = request.take_staged_stream_outputs()?;
                let active = state_lease.require_state_mut()?;
                active.state.rollback_quantum(&mut cache, &checkpoint)?;
                active.last_tokens_generated = prior.0;
                active.stream_sequence = prior.1;
                state_lease.mark_clean();
                if fresh_state {
                    state_lease.discard_state();
                }
                return Err(error);
            }
        };

        let (generated, stream_result, sample_rate, sample_count) = {
            let active = state_lease.require_state_mut()?;
            let generated = step
                .tokens_generated
                .saturating_sub(active.last_tokens_generated);
            active.last_tokens_generated = step.tokens_generated;
            let stream_result = (|| -> Result<()> {
                if scheduled.is_prefill {
                    return Ok(());
                }
                let Some(tx) = Self::stream_sender(request) else {
                    return Ok(());
                };
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
                Ok(())
            })();
            (
                generated,
                stream_result,
                active.input_sample_rate,
                active.input_sample_count,
            )
        };
        if let Err(error) = stream_result {
            let _ = request.take_staged_stream_outputs()?;
            let active = state_lease.require_state_mut()?;
            active.state.rollback_quantum(&mut cache, &checkpoint)?;
            active.last_tokens_generated = prior.0;
            active.stream_sequence = prior.1;
            state_lease.mark_clean();
            return Err(error);
        }
        if request.is_cancelled() {
            let _ = request.take_staged_stream_outputs()?;
            let active = state_lease.require_state_mut()?;
            active.state.rollback_quantum(&mut cache, &checkpoint)?;
            active.last_tokens_generated = prior.0;
            active.stream_sequence = prior.1;
            state_lease.mark_clean();
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        let completions = cache.take_completed_writes();
        state_lease
            .require_state_mut()?
            .state
            .commit_quantum(&cache, &checkpoint)?;
        state_lease.mark_clean();
        if step.finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate > 0 {
                    sample_count as f32 / sample_rate as f32
                } else {
                    0.0
                },
            }),
            text: Some(step.text),
            input_transcription: None,
            tokens_processed: scheduled.num_tokens,
            tokens_generated: generated,
            finished: step.finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        })
        .with_managed_cache_completions(completions))
    }

    pub(in crate::engine::executor) fn lfm25_audio_asr_prefill_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio ASR prefill batch rows do not match".into(),
            ));
        }
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
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
                    output
                        .ok_or_else(|| Error::InferenceError("LFM ASR prefill lost output".into()))
                })
                .collect();
        }
        let model = requests[live[0]]
            .prepared_lfm25_audio_asr_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("LFM ASR prefill lost model".into()))?;
        let model_arc = model.model_arc();
        let mut rows = Vec::with_capacity(live.len());
        for index in live.iter().copied() {
            let request = requests[index];
            if !scheduled[index].is_prefill || request.uses_asr_long_form_atomic() {
                return Err(Error::InvalidInput(
                    "LFM ASR prefill batch crossed stage identity".into(),
                ));
            }
            let variant = Self::resolve_variant(request)?;
            if variant.family() != ModelFamily::Lfm25Audio {
                return Err(Error::InvalidInput(
                    "LFM ASR prefill batch crossed model family".into(),
                ));
            }
            let row_model = request
                .prepared_lfm25_audio_asr_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("LFM ASR prefill row lost model".into()))?;
            if !Arc::ptr_eq(&model_arc, &row_model.model_arc()) {
                return Err(Error::InferenceError(
                    "LFM ASR prefill crossed model identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("LFM ASR prefill lost retained state".into())
            })?;
            let tensor = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError("LFM ASR prefill lost ShortConv reservation".into())
            })?;
            let arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("LFM ASR prefill lost ShortConv arena".into())
                })?;
            let cache = retained.take_only_paged()?;
            retained.ensure_all_paged_consumed()?;
            let session = scheduled[index].session_key();
            let mut lease = ExecutorStateLease::checkout(
                &self.lfm25_asr_decode_states,
                session.clone(),
                variant,
                "batched LFM2.5 Audio ASR prefill",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "LFM ASR prefill lost state after initial quantum".into(),
                    ));
                }
                let artifact = request
                    .prepared_lfm25_audio_asr_artifact_for_executor()?
                    .ok_or_else(|| Error::InferenceError("LFM ASR prefill lost prompt".into()))?;
                let (samples, sample_rate) =
                    request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                        Error::InferenceError("LFM ASR prefill lost decoded audio".into())
                    })?;
                if artifact.prompt_tokens != request.num_prompt_tokens()
                    || artifact.source_samples != samples.len()
                    || artifact.source_sample_rate != sample_rate
                {
                    return Err(Error::InferenceError(
                        "LFM ASR prefill artifact differs from admission".into(),
                    ));
                }
                let state = model.new_lfm25_audio_retained_asr_state(
                    artifact,
                    request.params.max_tokens.max(1),
                )?;
                lease.install_state(ActiveLfm25AsrDecode {
                    variant,
                    model: model.clone(),
                    state,
                    last_tokens_generated: 0,
                    stream_sequence: 0,
                    input_sample_rate: sample_rate,
                    input_sample_count: samples.len(),
                })?;
            }
            let active = lease.require_state_mut()?;
            if active.state.prompt_tokens() != request.num_prompt_tokens() {
                return Err(Error::InferenceError(
                    "LFM ASR prefill state differs from admission".into(),
                ));
            }
            active.state.bind_tensor_sequence(tensor.sequence)?;
            active.state.restore_shortconv(arena)?;
            if active.state.prefill_cursor() != scheduled[index].num_computed_tokens {
                return Err(Error::InferenceError(
                    "LFM ASR prefill cursor differs from admission".into(),
                ));
            }
            let prior_last_tokens_generated = active.last_tokens_generated;
            let prior_stream_sequence = active.stream_sequence;
            let checkpoint = active.state.begin_quantum(&cache)?;
            lease.mark_dirty();
            rows.push(ContinuousLfm25AsrRow {
                index,
                session,
                lease,
                cache,
                checkpoint: Some(checkpoint),
                prior_last_tokens_generated,
                prior_stream_sequence,
            });
        }
        let mut batch = ContinuousLfm25AsrStateBatch::new(rows);
        for row in 0..batch.rows.len() {
            let index = batch.rows[row].index;
            if requests[index].is_cancelled() {
                batch.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    requests[index].id.clone(),
                )));
            }
        }
        let call_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, state)| state.checkpoint.is_some().then_some(row))
            .collect::<Vec<_>>();
        if !call_rows.is_empty() {
            let mut states = Vec::with_capacity(call_rows.len());
            let mut caches = Vec::with_capacity(call_rows.len());
            let mut checkpoints = Vec::with_capacity(call_rows.len());
            let mut spans = Vec::with_capacity(call_rows.len());
            for (row_index, row) in batch.rows.iter_mut().enumerate() {
                if !call_rows.contains(&row_index) {
                    continue;
                }
                states.push(&mut row.lease.require_state_mut()?.state);
                caches.push(&mut row.cache);
                checkpoints.push(
                    row.checkpoint
                        .as_ref()
                        .expect("armed LFM ASR prefill checkpoint"),
                );
                spans.push(scheduled[row.index].num_tokens);
            }
            let native = Self::run_blocking(|| {
                model.lfm25_audio_asr_prefill_batch(&mut states, &mut caches, &checkpoints, &spans)
            })?;
            if native.steps.len() != call_rows.len() {
                return Err(Error::InferenceError(
                    "LFM ASR prefill returned wrong row count".into(),
                ));
            }
            for width in native.launch_widths {
                if let Some(call) =
                    retained_asr_batch_model_call(crate::engine::NativeBatchMode::Static, width)
                {
                    crate::engine::metrics::record_engine_model_call(call);
                }
            }
            for (row, step) in call_rows.iter().copied().zip(native.steps) {
                let index = batch.rows[row].index;
                let expected_end = scheduled[index]
                    .num_computed_tokens
                    .checked_add(scheduled[index].num_tokens)
                    .ok_or_else(|| Error::InvalidInput("LFM ASR prefill span overflowed".into()))?;
                if step.consumed_tokens != scheduled[index].num_tokens
                    || step.prefill_cursor != expected_end
                    || step.prompt_tokens != requests[index].num_prompt_tokens()
                    || step.complete != (expected_end == step.prompt_tokens)
                    || step.pending_token.is_some() != (step.complete && !step.finished)
                {
                    return Err(Error::InferenceError(
                        "LFM ASR prefill progress differs from admission".into(),
                    ));
                }
            }
        }
        for row in 0..batch.rows.len() {
            let index = batch.rows[row].index;
            if batch.rows[row].checkpoint.is_none() {
                continue;
            }
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("LFM ASR prefill lost ShortConv arena".into())
                })?;
            batch.rows[row]
                .lease
                .require_state_mut()?
                .state
                .stage_shortconv(arena, scheduled[index].plan_id)?;
        }
        let cancelled = batch
            .rows
            .iter()
            .map(|row| requests[row.index].is_cancelled())
            .collect::<Vec<_>>();
        let armed = batch
            .rows
            .iter()
            .map(|row| row.checkpoint.is_some())
            .collect::<Vec<_>>();
        for row in late_cancelled_lfm_asr_prefill_rows(&cancelled, &armed) {
            let index = batch.rollback_row(row)?;
            outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                requests[index].id.clone(),
            )));
        }
        let committed = batch.commit()?;
        for (index, lease, completions) in committed {
            let active = lease.state().ok_or_else(|| {
                Error::InferenceError("committed LFM ASR prefill lost state".into())
            })?;
            let sample_rate = active.input_sample_rate;
            let sample_count = active.input_sample_count;
            let finished = active.state.finished();
            if finished {
                lease.release()?;
            } else {
                lease.restore()?;
            }
            outputs[index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: requests[index].id.clone(),
                    audio: Some(AudioOutput {
                        streamed_samples: None,
                        samples: Vec::new(),
                        sample_rate,
                        duration_secs: if sample_rate > 0 {
                            sample_count as f32 / sample_rate as f32
                        } else {
                            0.0
                        },
                    }),
                    text: Some(String::new()),
                    input_transcription: None,
                    tokens_processed: scheduled[index].num_tokens,
                    tokens_generated: 0,
                    finished,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                })
                .with_managed_cache_completions(completions),
            );
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| Error::InferenceError("LFM ASR prefill lost output".into()))
            })
            .collect()
    }

    pub(super) fn lfm25_audio_asr_decode_batch_with_managed(
        &self,
        ordered_requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed_states: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        let mut outputs = (0..scheduled.len())
            .map(|_| None)
            .collect::<Vec<Option<ModelSessionResult>>>();
        let live_indices = ordered_requests
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
        if live_indices.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("cancelled LFM2.5 ASR row produced no result".into())
                    })
                })
                .collect();
        }

        let model = ordered_requests[live_indices[0]]
            .prepared_lfm25_audio_asr_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("continuous LFM2.5 ASR lost model residency".into())
            })?;
        let model_arc = model.model_arc();
        let mut rows = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let variant = Self::resolve_variant(request)?;
            if variant.family() != ModelFamily::Lfm25Audio || request.uses_asr_long_form_atomic() {
                return Err(Error::InvalidInput(
                    "continuous LFM2.5 ASR batch crossed route identity".into(),
                ));
            }
            let row_model = request
                .prepared_lfm25_audio_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("continuous LFM2.5 ASR row lost model residency".into())
                })?;
            if !Arc::ptr_eq(&model_arc, &row_model.model_arc()) {
                return Err(Error::InferenceError(
                    "continuous LFM2.5 ASR batch spans loaded model instances".into(),
                ));
            }
            let session = scheduled[index].session_key();
            let mut lease = ExecutorStateLease::checkout(
                &self.lfm25_asr_decode_states,
                session.clone(),
                variant,
                "continuous LFM2.5 Audio ASR decode",
            )?;
            let active = lease.state().ok_or_else(|| {
                Error::InferenceError(format!(
                    "continuous LFM2.5 ASR session {}:{} has no active state",
                    session.request_id, session.epoch
                ))
            })?;
            if active.variant != variant
                || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
                || active.state.prompt_tokens() != request.num_prompt_tokens()
            {
                return Err(Error::InferenceError(
                    "continuous LFM2.5 ASR state identity does not match its request".into(),
                ));
            }
            let mut retained = managed_states[index].take().ok_or_else(|| {
                Error::InferenceError("continuous LFM2.5 ASR lost retained physical state".into())
            })?;
            let tensor_reservation = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError("continuous LFM2.5 ASR lost ShortConv reservation".into())
            })?;
            let tensor_arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("continuous LFM2.5 ASR lost ShortConv arena".into())
                })?;
            let cache = retained.take_only_paged()?;
            retained.ensure_all_paged_consumed()?;
            let active = lease.require_state_mut()?;
            active
                .state
                .bind_tensor_sequence(tensor_reservation.sequence)?;
            active.state.restore_shortconv(tensor_arena)?;
            let prior_last_tokens_generated = active.last_tokens_generated;
            let prior_stream_sequence = active.stream_sequence;
            let checkpoint = active.state.begin_quantum(&cache)?;
            lease.mark_dirty();
            rows.push(ContinuousLfm25AsrRow {
                index,
                session,
                lease,
                cache,
                checkpoint: Some(checkpoint),
                prior_last_tokens_generated,
                prior_stream_sequence,
            });
        }
        let mut batch = ContinuousLfm25AsrStateBatch::new(rows);

        for row in 0..batch.rows.len() {
            let index = batch.rows[row].index;
            if ordered_requests[index].is_cancelled() {
                let index = batch.rollback_row(row)?;
                let _ = ordered_requests[index].take_staged_stream_outputs()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    ordered_requests[index].id.clone(),
                )));
            }
        }

        let call_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, state)| outputs[state.index].is_none().then_some(row))
            .collect::<Vec<_>>();
        let mut steps = Vec::<(usize, Lfm25AudioAsrDecodeStep)>::with_capacity(call_rows.len());
        let mut append_rows = Vec::new();
        let mut scalar_rows = 0usize;
        for row in call_rows.iter().copied() {
            let will_append = model.lfm25_audio_asr_decode_will_append(
                &batch.rows[row].lease.require_state_mut()?.state,
            )?;
            if will_append {
                append_rows.push(row);
            } else {
                scalar_rows += 1;
                let ContinuousLfm25AsrRow {
                    lease,
                    cache,
                    checkpoint,
                    ..
                } = &mut batch.rows[row];
                let checkpoint = checkpoint
                    .as_ref()
                    .expect("active LFM2.5 row has checkpoint");
                let step = Self::run_blocking(|| {
                    model.lfm25_audio_asr_decode_step(
                        &mut lease.require_state_mut()?.state,
                        cache,
                        checkpoint,
                    )
                })?;
                steps.push((row, step));
            }
        }
        for row in append_rows.iter().copied() {
            let index = batch.rows[row].index;
            if ordered_requests[index].is_cancelled() {
                let index = batch.rollback_row(row)?;
                let _ = ordered_requests[index].take_staged_stream_outputs()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    ordered_requests[index].id.clone(),
                )));
            }
        }
        append_rows.retain(|row| batch.rows[*row].checkpoint.is_some());
        if !append_rows.is_empty() {
            let native_width = append_rows.len();
            let mut state_refs = Vec::with_capacity(append_rows.len());
            let mut cache_refs = Vec::with_capacity(append_rows.len());
            let mut checkpoint_refs = Vec::with_capacity(append_rows.len());
            for (row_index, row) in batch.rows.iter_mut().enumerate() {
                if !append_rows.contains(&row_index) {
                    continue;
                }
                state_refs.push(&mut row.lease.require_state_mut()?.state);
                cache_refs.push(&mut row.cache);
                checkpoint_refs.push(
                    row.checkpoint
                        .as_ref()
                        .expect("active LFM2.5 row has checkpoint"),
                );
            }
            let native_steps = Self::run_blocking(|| {
                model.lfm25_audio_asr_decode_append_batch(
                    &mut state_refs,
                    &mut cache_refs,
                    &checkpoint_refs,
                )
            })?;
            if native_steps.len() != append_rows.len() {
                return Err(Error::InferenceError(
                    "LFM2.5 ASR native decode returned the wrong row count".into(),
                ));
            }
            steps.extend(append_rows.into_iter().zip(native_steps));
            if let Some(call) = retained_asr_batch_model_call(
                crate::engine::NativeBatchMode::Continuous,
                native_width,
            ) {
                crate::engine::metrics::record_engine_model_call(call);
            }
        }
        if scalar_rows > 0 {
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::ScalarRows {
                    envelope: crate::engine::NativeBatchMode::Continuous,
                    rows: scalar_rows,
                },
            );
        }

        for row in call_rows.iter().copied() {
            let index = batch.rows[row].index;
            if batch.rows[row].checkpoint.is_some() && ordered_requests[index].is_cancelled() {
                let index = batch.rollback_row(row)?;
                let _ = ordered_requests[index].take_staged_stream_outputs()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    ordered_requests[index].id.clone(),
                )));
            }
        }

        let mut continuing = vec![false; scheduled.len()];
        for (row, step) in steps {
            let index = batch.rows[row].index;
            if outputs[index].is_some() {
                continue;
            }
            let request = ordered_requests[index];
            let tensor_arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("continuous LFM2.5 ASR lost ShortConv arena".into())
                })?;
            let (generated, sample_rate, sample_count, stream_result) = {
                let active = batch.rows[row].lease.require_state_mut()?;
                active
                    .state
                    .stage_shortconv(tensor_arena, scheduled[index].plan_id)?;
                let generated = step
                    .tokens_generated
                    .saturating_sub(active.last_tokens_generated);
                active.last_tokens_generated = step.tokens_generated;
                let stream_result = (|| -> Result<()> {
                    let Some(tx) = Self::stream_sender(request) else {
                        return Ok(());
                    };
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
                    Ok(())
                })();
                (
                    generated,
                    active.input_sample_rate,
                    active.input_sample_count,
                    stream_result,
                )
            };
            if let Err(error) = stream_result {
                let _ = request.take_staged_stream_outputs()?;
                let index = batch.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    request.id.clone(),
                    format!("continuous LFM2.5 ASR stream staging failed: {error}"),
                )));
                continue;
            }
            if request.is_cancelled() {
                let _ = request.take_staged_stream_outputs()?;
                let index = batch.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                continue;
            }
            outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput {
                    streamed_samples: None,
                    samples: Vec::new(),
                    sample_rate,
                    duration_secs: if sample_rate > 0 {
                        sample_count as f32 / sample_rate as f32
                    } else {
                        0.0
                    },
                }),
                text: Some(step.text),
                input_transcription: None,
                tokens_processed: 1,
                tokens_generated: generated,
                finished: step.finished,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            }));
            continuing[index] = !step.finished;
        }

        for row in 0..batch.rows.len() {
            let index = batch.rows[row].index;
            if outputs[index].is_some()
                && batch.rows[row].checkpoint.is_some()
                && ordered_requests[index].is_cancelled()
            {
                let _ = ordered_requests[index].take_staged_stream_outputs()?;
                let index = batch.rollback_row(row)?;
                continuing[index] = false;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    ordered_requests[index].id.clone(),
                )));
            }
        }

        let committed = batch.commit()?;
        for (index, lease, completions) in committed {
            let result = outputs[index].take().ok_or_else(|| {
                Error::InferenceError("committed LFM2.5 ASR row produced no output".into())
            })?;
            outputs[index] = Some(result.with_managed_cache_completions(completions));
            if continuing[index] {
                lease.restore()?;
            } else {
                lease.release()?;
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("continuous LFM2.5 ASR row produced no result".into())
                })
            })
            .collect()
    }
}
