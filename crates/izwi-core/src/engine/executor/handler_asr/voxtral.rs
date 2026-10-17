//! Voxtral realtime ASR: the managed realtime batch and staged session steps.

use super::*;

impl NativeExecutor {
    pub(in crate::engine::executor) fn voxtral_realtime_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Voxtral staged batch width mismatch".into(),
            ));
        }
        if scheduled.len() == 1 {
            return Ok(vec![self.voxtral_realtime_staged_request(
                requests[0],
                &scheduled[0],
                managed[0].take().ok_or_else(|| {
                    Error::InferenceError("Voxtral scalar batch lost retained state".into())
                })?,
            )?]);
        }
        if scheduled
            .iter()
            .all(|row| matches!(row.work, WorkUnit::RealtimePreparation { .. }))
        {
            let mut payloads = Vec::with_capacity(scheduled.len());
            for (request, row) in requests.iter().zip(scheduled) {
                let operation_id = match row.work {
                    WorkUnit::RealtimePreparation { operation_id, .. } => operation_id,
                    _ => unreachable!(),
                };
                payloads.push(request.realtime_asr_operation(operation_id)?);
            }
            let mut leases = Vec::with_capacity(scheduled.len());
            for ((request, row), payload) in requests.iter().zip(scheduled).zip(&payloads) {
                let variant = Self::resolve_variant(request)?;
                let session = row.session_key();
                let mut lease = ExecutorStateLease::checkout(
                    &self.voxtral_realtime.states,
                    session,
                    variant,
                    "Voxtral preparation batch",
                )?;
                if lease.state().is_none() {
                    if !matches!(
                        (&row.work, payload),
                        (
                            WorkUnit::RealtimePreparation {
                                mode: RealtimePreparationMode::Push,
                                ..
                            },
                            RealtimeAsrOperationPayload::Push { .. }
                        )
                    ) {
                        return Err(Error::InferenceError(
                            "Voxtral preparation batch cannot start with finish".into(),
                        ));
                    }
                    let model = self.with_registry(|registry| {
                        registry.try_get_voxtral_lease(variant).ok_or_else(|| {
                            Error::ModelNotFound(format!("Voxtral model {variant} is not loaded"))
                        })
                    })?;
                    lease.install_state(ActiveVoxtralRealtime {
                        variant,
                        state: model.start_realtime_state(request.asr_language_for_execution()),
                        model,
                        last_tokens_generated: 0,
                        stream_sequence: 0,
                        input_sample_rate: 0,
                    })?;
                }
                leases.push(lease);
            }
            let model = leases[0]
                .state()
                .expect("preparation state installed")
                .model
                .clone();
            let mut batch_rows = Vec::with_capacity(scheduled.len());
            for ((lease, row), payload) in leases.iter().zip(scheduled).zip(&payloads) {
                let active = lease.state().expect("preparation state installed");
                if !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc()) {
                    return Err(Error::InvalidInput(
                        "Voxtral preparation batch crossed model identity".into(),
                    ));
                }
                let (samples, sample_rate, mode) = match (&row.work, payload) {
                    (
                        WorkUnit::RealtimePreparation {
                            mode: RealtimePreparationMode::Push,
                            ..
                        },
                        RealtimeAsrOperationPayload::Push {
                            samples,
                            sample_rate,
                        },
                    ) => (
                        samples.as_ref(),
                        *sample_rate,
                        VoxtralRealtimePreparationMode::Push,
                    ),
                    (
                        WorkUnit::RealtimePreparation {
                            mode: RealtimePreparationMode::Finish,
                            ..
                        },
                        RealtimeAsrOperationPayload::Finish,
                    ) => (
                        &[][..],
                        active.input_sample_rate.max(1),
                        VoxtralRealtimePreparationMode::Finish,
                    ),
                    _ => {
                        return Err(Error::InferenceError(
                            "Voxtral preparation batch payload mismatch".into(),
                        ));
                    }
                };
                batch_rows.push(VoxtralRealtimePreparationBatchRow {
                    state: &active.state,
                    appended_samples: samples,
                    sample_rate,
                    mode,
                });
            }
            let prepared = model.prepare_realtime_audio_batch(&batch_rows)?;
            drop(batch_rows);
            for lease in leases {
                lease.restore()?;
            }
            if prepared.len() != scheduled.len() {
                return Err(Error::InferenceError(format!(
                    "Voxtral preparation batch returned {} rows for a {}-row cohort",
                    prepared.len(),
                    scheduled.len()
                )));
            }
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::NativeTensor {
                    mode: crate::engine::NativeBatchMode::Static,
                    rows: prepared.len(),
                },
            );
            let mut rows = Vec::with_capacity(scheduled.len());
            let acquisition = (|| -> Result<()> {
                for ((request, row), managed) in
                    requests.iter().zip(scheduled).zip(managed.iter_mut())
                {
                    let mut retained = managed.take().ok_or_else(|| {
                        Error::InferenceError("Voxtral preparation row lost retained state".into())
                    })?;
                    let cache = retained.take_only_paged()?;
                    retained.ensure_all_paged_consumed()?;
                    let session = row.session_key();
                    let variant = Self::resolve_variant(request)?;
                    let mut lease = ExecutorStateLease::checkout(
                        &self.voxtral_realtime.states,
                        session.clone(),
                        variant,
                        "Voxtral preparation batch install",
                    )?;
                    let active = lease.require_state_mut()?;
                    if active.variant != variant
                        || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
                    {
                        return Err(Error::InvalidInput(
                            "Voxtral preparation batch install crossed model identity".into(),
                        ));
                    }
                    let prior = (
                        active.last_tokens_generated,
                        active.stream_sequence,
                        active.input_sample_rate,
                        cache.context_len(),
                    );
                    let checkpoint = active
                        .model
                        .begin_realtime_quantum(&mut active.state, &cache)?;
                    lease.mark_dirty();
                    rows.push((request, row, session, lease, cache, checkpoint, prior));
                }
                Ok(())
            })();
            if let Err(error) = acquisition {
                let mut rollback_errors = Vec::new();
                for (request, _, _, lease, cache, checkpoint, prior) in &mut rows {
                    let _ = request.take_staged_stream_outputs();
                    match lease.require_state_mut() {
                        Ok(active) => {
                            active.last_tokens_generated = prior.0;
                            active.stream_sequence = prior.1;
                            active.input_sample_rate = prior.2;
                            match active.model.rollback_realtime_quantum(
                                &mut active.state,
                                cache,
                                checkpoint,
                            ) {
                                Ok(()) => lease.mark_clean(),
                                Err(rollback) => rollback_errors.push(rollback.to_string()),
                            }
                        }
                        Err(rollback) => rollback_errors.push(rollback.to_string()),
                    }
                }
                return if rollback_errors.is_empty() {
                    Err(error)
                } else {
                    Err(Error::InferenceError(format!(
                        "{error}; Voxtral preparation cohort acquisition rollback also failed: {}",
                        rollback_errors.join("; ")
                    )))
                };
            }

            let sealing = (|| -> Result<_> {
                let mut sealed = Vec::with_capacity(rows.len());
                for (((request, scheduled, _, lease, cache, _, prior), prepared), payload) in
                    rows.iter_mut().zip(prepared).zip(payloads.iter())
                {
                    let (operation_id, max_output_steps, max_cache_append, mode) =
                        match scheduled.work {
                            WorkUnit::RealtimePreparation {
                                operation_id,
                                mode,
                                max_output_steps,
                                max_cache_append,
                                ..
                            } => (operation_id, max_output_steps, max_cache_append, mode),
                            _ => unreachable!(),
                        };
                    let active = lease.require_state_mut()?;
                    match (mode, payload) {
                        (
                            RealtimePreparationMode::Push,
                            RealtimeAsrOperationPayload::Push { sample_rate, .. },
                        ) => {
                            active
                                .model
                                .install_realtime_audio_preparation(&mut active.state, prepared)?;
                            active.input_sample_rate = *sample_rate;
                        }
                        (RealtimePreparationMode::Finish, RealtimeAsrOperationPayload::Finish) => {
                            active
                                .model
                                .install_realtime_audio_preparation(&mut active.state, prepared)?;
                        }
                        _ => {
                            return Err(Error::InferenceError(
                                "Voxtral preparation batch payload changed during install".into(),
                            ));
                        }
                    }
                    if request.is_cancelled() {
                        return Err(Error::Cancelled(
                            "Voxtral preparation cohort cancelled during install".into(),
                        ));
                    }
                    let appended = cache.context_len().checked_sub(prior.3).ok_or_else(|| {
                        Error::InferenceError(
                            "Voxtral preparation batch cache cursor regressed".into(),
                        )
                    })?;
                    if appended != 0 || appended > max_cache_append {
                        return Err(Error::InferenceError(
                            "Voxtral preparation batch unexpectedly appended KV state".into(),
                        ));
                    }
                    let next = if active.state.is_finished() {
                        None
                    } else if let Some(prompt) =
                        active.model.realtime_prompt_cache_append(&active.state)?
                    {
                        Some(RealtimeSubphase::PromptPrefill {
                            cache_append: prompt,
                        })
                    } else if active.model.realtime_decode_ready(&active.state)
                        && max_output_steps > 0
                    {
                        Some(RealtimeSubphase::DecodeContinuation)
                    } else if matches!(payload, RealtimeAsrOperationPayload::Finish) {
                        Some(RealtimeSubphase::Completion)
                    } else {
                        None
                    };
                    let input_consumed = if next.is_none() {
                        match payload {
                            RealtimeAsrOperationPayload::Push { samples, .. } => samples.len(),
                            RealtimeAsrOperationPayload::Finish => 0,
                        }
                    } else {
                        0
                    };
                    let staged = request.take_staged_stream_outputs()?;
                    if request.is_cancelled() {
                        return Err(Error::Cancelled(
                            "Voxtral preparation cohort cancelled during seal".into(),
                        ));
                    }
                    sealed.push((
                        operation_id,
                        next,
                        input_consumed,
                        active.state.text().to_string(),
                        active.state.is_finished(),
                        active.input_sample_rate,
                        active.state.source_sample_count(),
                        staged,
                    ));
                }
                if rows.iter().any(|(request, ..)| request.is_cancelled()) {
                    return Err(Error::Cancelled(
                        "Voxtral preparation cohort cancelled before registration".into(),
                    ));
                }
                for (_, _, _, lease, _, _, _) in &rows {
                    lease.validate_defer()?;
                }
                Ok(sealed)
            })();
            let sealed = match sealing {
                Ok(sealed) => sealed,
                Err(error) => {
                    let mut rollback_errors = Vec::new();
                    for (request, _, _, lease, cache, checkpoint, prior) in &mut rows {
                        let _ = request.take_staged_stream_outputs();
                        match lease.require_state_mut() {
                            Ok(active) => {
                                active.last_tokens_generated = prior.0;
                                active.stream_sequence = prior.1;
                                active.input_sample_rate = prior.2;
                                match active.model.rollback_realtime_quantum(
                                    &mut active.state,
                                    cache,
                                    checkpoint,
                                ) {
                                    Ok(()) => lease.mark_clean(),
                                    Err(rollback) => rollback_errors.push(rollback.to_string()),
                                }
                            }
                            Err(rollback) => rollback_errors.push(rollback.to_string()),
                        }
                    }
                    return if rollback_errors.is_empty() {
                        Err(error)
                    } else {
                        Err(Error::InferenceError(format!(
                            "{error}; Voxtral preparation cohort rollback also failed: {}",
                            rollback_errors.join("; ")
                        )))
                    };
                }
            };

            let mut pending = Vec::with_capacity(rows.len());
            let mut outputs = Vec::with_capacity(rows.len());
            for ((request, scheduled, session, lease, mut cache, checkpoint, prior), seal) in
                rows.into_iter().zip(sealed)
            {
                let (
                    operation_id,
                    next,
                    input_consumed,
                    text,
                    finished,
                    sample_rate,
                    sample_count,
                    staged,
                ) = seal;
                let completions = cache.take_completed_writes();
                let active = lease.defer_validated();
                pending.push((
                    scheduled.plan_id,
                    super::super::PendingVoxtralRealtimeQuantum {
                        session,
                        active,
                        cache,
                        checkpoint,
                        prior_last_tokens_generated: prior.0,
                        prior_stream_sequence: prior.1,
                        prior_input_sample_rate: prior.2,
                        finished,
                    },
                ));
                outputs.push(
                    ModelSessionResult::sequence(ExecutorOutput {
                        request_id: request.id.clone(),
                        audio: Some(AudioOutput {
                            streamed_samples: None,
                            samples: Vec::new(),
                            sample_rate,
                            duration_secs: if sample_rate == 0 {
                                0.0
                            } else {
                                sample_count as f32 / sample_rate as f32
                            },
                        }),
                        text: Some(text),
                        input_transcription: None,
                        tokens_processed: 0,
                        tokens_generated: 0,
                        finished,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                        logprobs: Vec::new(),
                    })
                    .with_staged_stream_outputs(staged)
                    .with_managed_cache_completions(completions)
                    .with_managed_cache_append(0)
                    .requiring_pending_quantum()
                    .with_realtime_stage_outcome(RealtimeStageOutcome {
                        plan_id: scheduled.plan_id,
                        operation_id,
                        completed: RealtimeSubphase::Preparation,
                        next,
                        input_consumed,
                        output_steps: 0,
                        cache_append: 0,
                    }),
                );
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
                        "{error}; Voxtral preparation cohort rollback also failed: {}",
                        rollback_errors.join("; ")
                    )))
                };
            }
            return Ok(outputs);
        }
        if scheduled
            .iter()
            .all(|row| matches!(row.work, WorkUnit::RealtimeDecodeContinuation { .. }))
        {
            let mut rows = Vec::with_capacity(scheduled.len());
            for ((request, scheduled), managed) in
                requests.iter().zip(scheduled).zip(managed.iter_mut())
            {
                let mut retained = managed.take().ok_or_else(|| {
                    Error::InferenceError("Voxtral decode row lost retained state".into())
                })?;
                let cache = retained.take_only_paged()?;
                retained.ensure_all_paged_consumed()?;
                let session = scheduled.session_key();
                let variant = Self::resolve_variant(request)?;
                let mut lease = ExecutorStateLease::checkout(
                    &self.voxtral_realtime.states,
                    session.clone(),
                    variant,
                    "Voxtral decode batch",
                )?;
                let active = lease.require_state_mut()?;
                let prior = (
                    active.last_tokens_generated,
                    active.stream_sequence,
                    active.input_sample_rate,
                    cache.context_len(),
                );
                let checkpoint = active
                    .model
                    .begin_realtime_quantum(&mut active.state, &cache)?;
                lease.mark_dirty();
                rows.push((request, scheduled, session, lease, cache, checkpoint, prior));
            }
            let model = rows[0].3.require_state_mut()?.model.clone();
            let execution = (|| {
                let mut batch = Vec::with_capacity(rows.len());
                for (_, _, _, lease, cache, _, _) in &mut rows {
                    let active = lease.require_state_mut()?;
                    if !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc()) {
                        return Err(Error::InvalidInput(
                            "Voxtral decode batch crossed model identity".into(),
                        ));
                    }
                    batch.push(VoxtralRealtimeDecodeBatchRow {
                        state: &mut active.state,
                        cache,
                    });
                }
                let steps = model.decode_realtime_step_batch(&mut batch)?;
                drop(batch);
                if rows.iter().any(|(request, ..)| request.is_cancelled()) {
                    return Err(Error::Cancelled(
                        "Voxtral decode batch cancelled before seal".into(),
                    ));
                }
                Ok(steps)
            })();
            let steps = match execution {
                Ok(steps) => steps,
                Err(error) => {
                    let mut rollback_errors = Vec::new();
                    for (_, _, _, lease, cache, checkpoint, prior) in &mut rows {
                        match lease.require_state_mut() {
                            Ok(active) => {
                                active.last_tokens_generated = prior.0;
                                active.stream_sequence = prior.1;
                                active.input_sample_rate = prior.2;
                                match active.model.rollback_realtime_quantum(
                                    &mut active.state,
                                    cache,
                                    checkpoint,
                                ) {
                                    Ok(()) => lease.mark_clean(),
                                    Err(rollback) => rollback_errors.push(rollback.to_string()),
                                }
                            }
                            Err(rollback) => rollback_errors.push(rollback.to_string()),
                        }
                    }
                    return if rollback_errors.is_empty() {
                        Err(error)
                    } else {
                        Err(Error::InferenceError(format!(
                            "{error}; Voxtral decode batch rollback also failed: {}",
                            rollback_errors.join("; ")
                        )))
                    };
                }
            };
            if steps.len() != rows.len() {
                let error = Error::InferenceError(format!(
                    "Voxtral decode batch returned {} rows for a {}-row cohort",
                    steps.len(),
                    rows.len()
                ));
                let mut rollback_errors = Vec::new();
                for (request, _, _, lease, cache, checkpoint, prior) in &mut rows {
                    let _ = request.take_staged_stream_outputs();
                    match lease.require_state_mut() {
                        Ok(active) => {
                            active.last_tokens_generated = prior.0;
                            active.stream_sequence = prior.1;
                            active.input_sample_rate = prior.2;
                            match active.model.rollback_realtime_quantum(
                                &mut active.state,
                                cache,
                                checkpoint,
                            ) {
                                Ok(()) => lease.mark_clean(),
                                Err(rollback) => rollback_errors.push(rollback.to_string()),
                            }
                        }
                        Err(rollback) => rollback_errors.push(rollback.to_string()),
                    }
                }
                return if rollback_errors.is_empty() {
                    Err(error)
                } else {
                    Err(Error::InferenceError(format!(
                        "{error}; Voxtral decode cohort rollback also failed: {}",
                        rollback_errors.join("; ")
                    )))
                };
            }
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::NativeTensor {
                    mode: crate::engine::NativeBatchMode::Continuous,
                    rows: steps.len(),
                },
            );
            let sealing = (|| -> Result<_> {
                let mut sealed = Vec::with_capacity(rows.len());
                for ((request, scheduled, _, lease, cache, _, prior), step) in
                    rows.iter_mut().zip(steps.iter())
                {
                    let (operation_id, max_output_steps) = match scheduled.work {
                        WorkUnit::RealtimeDecodeContinuation {
                            operation_id,
                            max_output_steps,
                            ..
                        } => (operation_id, max_output_steps),
                        _ => unreachable!(),
                    };
                    let payload = request.realtime_asr_operation(operation_id)?;
                    let active = lease.require_state_mut()?;
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
                    }
                    let staged = request.take_staged_stream_outputs()?;
                    if request.is_cancelled() {
                        return Err(Error::Cancelled(
                            "Voxtral decode cohort cancelled during seal".into(),
                        ));
                    }
                    let next = if active.model.realtime_decode_ready(&active.state)
                        && max_output_steps > 1
                    {
                        Some(RealtimeSubphase::DecodeContinuation)
                    } else if matches!(payload, RealtimeAsrOperationPayload::Finish) {
                        Some(RealtimeSubphase::Completion)
                    } else {
                        None
                    };
                    let input_consumed = if next.is_none() {
                        match &payload {
                            RealtimeAsrOperationPayload::Push { samples, .. } => samples.len(),
                            RealtimeAsrOperationPayload::Finish => 0,
                        }
                    } else {
                        0
                    };
                    let appended = cache.context_len().checked_sub(prior.3).ok_or_else(|| {
                        Error::InferenceError("Voxtral decode batch cache cursor regressed".into())
                    })?;
                    if appended != 1 {
                        return Err(Error::InferenceError(
                            "Voxtral decode batch did not append exactly one token".into(),
                        ));
                    }
                    sealed.push((
                        operation_id,
                        next,
                        input_consumed,
                        active.state.text().to_string(),
                        active.state.is_finished(),
                        active.input_sample_rate,
                        active.state.source_sample_count(),
                        staged,
                    ));
                }
                if rows.iter().any(|(request, ..)| request.is_cancelled()) {
                    return Err(Error::Cancelled(
                        "Voxtral decode cohort cancelled before registration".into(),
                    ));
                }
                for (_, _, _, lease, _, _, _) in &rows {
                    lease.validate_defer()?;
                }
                Ok(sealed)
            })();
            let sealed = match sealing {
                Ok(sealed) => sealed,
                Err(error) => {
                    let mut rollback_errors = Vec::new();
                    for (request, _, _, lease, cache, checkpoint, prior) in &mut rows {
                        let _ = request.take_staged_stream_outputs();
                        match lease.require_state_mut() {
                            Ok(active) => {
                                active.last_tokens_generated = prior.0;
                                active.stream_sequence = prior.1;
                                active.input_sample_rate = prior.2;
                                match active.model.rollback_realtime_quantum(
                                    &mut active.state,
                                    cache,
                                    checkpoint,
                                ) {
                                    Ok(()) => lease.mark_clean(),
                                    Err(rollback) => rollback_errors.push(rollback.to_string()),
                                }
                            }
                            Err(rollback) => rollback_errors.push(rollback.to_string()),
                        }
                    }
                    return if rollback_errors.is_empty() {
                        Err(error)
                    } else {
                        Err(Error::InferenceError(format!(
                            "{error}; Voxtral decode cohort rollback also failed: {}",
                            rollback_errors.join("; ")
                        )))
                    };
                }
            };
            let mut pending = Vec::with_capacity(rows.len());
            let mut outputs = Vec::with_capacity(rows.len());
            for ((request, scheduled, session, lease, mut cache, checkpoint, prior), seal) in
                rows.into_iter().zip(sealed)
            {
                let (
                    operation_id,
                    next,
                    input_consumed,
                    text,
                    finished,
                    sample_rate,
                    sample_count,
                    staged,
                ) = seal;
                let completions = cache.take_completed_writes();
                let active = lease.defer_validated();
                pending.push((
                    scheduled.plan_id,
                    super::super::PendingVoxtralRealtimeQuantum {
                        session,
                        active,
                        cache,
                        checkpoint,
                        prior_last_tokens_generated: prior.0,
                        prior_stream_sequence: prior.1,
                        prior_input_sample_rate: prior.2,
                        finished,
                    },
                ));
                outputs.push(
                    ModelSessionResult::sequence(ExecutorOutput {
                        request_id: request.id.clone(),
                        audio: Some(AudioOutput {
                            streamed_samples: None,
                            samples: Vec::new(),
                            sample_rate,
                            duration_secs: if sample_rate == 0 {
                                0.0
                            } else {
                                sample_count as f32 / sample_rate as f32
                            },
                        }),
                        text: Some(text),
                        input_transcription: None,
                        tokens_processed: 0,
                        tokens_generated: 1,
                        finished,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                        logprobs: Vec::new(),
                    })
                    .with_staged_stream_outputs(staged)
                    .with_managed_cache_completions(completions)
                    .with_managed_cache_append(1)
                    .requiring_pending_quantum()
                    .with_realtime_stage_outcome(RealtimeStageOutcome {
                        plan_id: scheduled.plan_id,
                        operation_id,
                        completed: RealtimeSubphase::DecodeContinuation,
                        next,
                        input_consumed,
                        output_steps: 1,
                        cache_append: 1,
                    }),
                );
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
                        "{error}; Voxtral decode cohort rollback also failed: {}",
                        rollback_errors.join("; ")
                    )))
                };
            }
            return Ok(outputs);
        }
        // Decode batching is handled below once all rows have authenticated
        // retained checkpoints; other staged work is intentionally scalar.
        let mut outputs = Vec::with_capacity(scheduled.len());
        for ((request, row), state) in requests.iter().zip(scheduled).zip(managed.iter_mut()) {
            outputs.push(self.voxtral_realtime_staged_request(
                request,
                row,
                state.take().ok_or_else(|| {
                    Error::InferenceError("Voxtral staged row lost retained state".into())
                })?,
            )?);
        }
        Ok(outputs)
    }

    pub(super) fn voxtral_realtime_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: super::super::RetainedRowManagedState,
    ) -> Result<ModelSessionResult> {
        if matches!(
            scheduled.work,
            WorkUnit::RealtimePreparation { .. }
                | WorkUnit::RealtimePromptPrefill { .. }
                | WorkUnit::RealtimeDecodeContinuation { .. }
                | WorkUnit::RealtimeCompletion { .. }
        ) {
            return self.voxtral_realtime_staged_request(request, scheduled, managed_state);
        }
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Voxtral || !request.is_realtime_asr_session() {
            return Err(Error::InvalidInput(
                "Voxtral realtime work crossed its exact ASR session route".into(),
            ));
        }
        if request.asr_prompt_for_execution().is_some() {
            return Err(Error::InvalidInput(
                "Voxtral realtime ASR does not support an initial text prompt".into(),
            ));
        }
        if managed_state.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Voxtral realtime row unexpectedly received tensor state".into(),
            ));
        }
        let mut cache = managed_state.take_only_paged()?;
        managed_state.ensure_all_paged_consumed()?;
        let (operation_id, max_output_steps, max_cache_append) = match &scheduled.work {
            crate::engine::WorkUnit::RealtimePush {
                operation_id,
                max_output_steps,
                max_cache_append,
                ..
            }
            | crate::engine::WorkUnit::RealtimeFinish {
                operation_id,
                max_output_steps,
                max_cache_append,
            } => (*operation_id, *max_output_steps, *max_cache_append),
            _ => {
                return Err(Error::InvalidInput(
                    "Voxtral realtime executor received non-realtime work".into(),
                ));
            }
        };
        let payload = request.realtime_asr_operation(operation_id)?;
        let session = scheduled.session_key();
        let mut state_lease = ExecutorStateLease::checkout(
            &self.voxtral_realtime.states,
            session.clone(),
            variant,
            "Voxtral realtime ASR",
        )?;
        if state_lease
            .state()
            .is_some_and(|active| active.variant != variant)
        {
            state_lease.discard_state();
        }
        if state_lease.state().is_none() {
            if !matches!(payload, RealtimeAsrOperationPayload::Push { .. }) {
                return Err(Error::InferenceError(
                    "Voxtral realtime session cannot start with finish".into(),
                ));
            }
            let model = self.with_registry(|registry| {
                registry.try_get_voxtral_lease(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!(
                        "Voxtral model {variant} is not loaded in registry"
                    ))
                })
            })?;
            let state = model.start_realtime_state(request.asr_language_for_execution());
            state_lease.install_state(ActiveVoxtralRealtime {
                variant,
                model,
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: 0,
            })?;
        }

        let model_max_output_steps = state_lease
            .require_state_mut()?
            .model
            .realtime_max_output_steps()?;
        if max_output_steps == 0 || max_output_steps > model_max_output_steps {
            return Err(Error::InvalidInput(format!(
                "Voxtral realtime quantum requested {max_output_steps} output steps outside the model ceiling {model_max_output_steps}"
            )));
        }
        if max_cache_append == 0 {
            return Err(Error::InvalidInput(
                "Voxtral realtime quantum requires a non-zero cache append bound".into(),
            ));
        }

        let (prior_tokens, prior_stream_sequence, prior_input_sample_rate, prior_cache_len) = {
            let active = state_lease.require_state_mut()?;
            (
                active.last_tokens_generated,
                active.stream_sequence,
                active.input_sample_rate,
                cache.context_len(),
            )
        };
        let active = state_lease.require_state_mut()?;
        let mut checkpoint = active
            .model
            .begin_realtime_quantum(&mut active.state, &cache)?;
        state_lease.mark_dirty();

        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let mut should_cancel = || request.is_cancelled();
            let steps = match (&scheduled.work, payload) {
                (
                    crate::engine::WorkUnit::RealtimePush { input, .. },
                    RealtimeAsrOperationPayload::Push {
                        samples,
                        sample_rate,
                    },
                ) if input.start == active.state.source_sample_count()
                    && input.len() == samples.len()
                    && sample_rate > 0 =>
                {
                    if active.input_sample_rate != 0 && active.input_sample_rate != sample_rate {
                        return Err(Error::InvalidInput(
                            "Voxtral realtime sample rate changed within one session".into(),
                        ));
                    }
                    active.input_sample_rate = sample_rate;
                    active.model.apply_realtime_push_physical(
                        &mut active.state,
                        &mut cache,
                        samples.as_ref(),
                        sample_rate,
                        max_output_steps,
                        &mut should_cancel,
                    )?
                }
                (
                    crate::engine::WorkUnit::RealtimeFinish { .. },
                    RealtimeAsrOperationPayload::Finish,
                ) => active.model.apply_realtime_finish_physical(
                    &mut active.state,
                    &mut cache,
                    max_output_steps,
                    &mut should_cancel,
                )?,
                _ => {
                    return Err(Error::InferenceError(
                        "Voxtral realtime work and retained operation payload disagree".into(),
                    ));
                }
            };
            let appended = cache
                .context_len()
                .checked_sub(prior_cache_len)
                .ok_or_else(|| {
                    Error::InferenceError("Voxtral realtime cache cursor regressed".into())
                })?;
            if appended > max_cache_append {
                return Err(Error::InferenceError(format!(
                    "Voxtral realtime quantum appended {appended} cache tokens beyond its {max_cache_append}-token bound"
                )));
            }
            if request.is_cancelled() {
                return Err(Error::Cancelled(
                    "Voxtral realtime quantum cancelled before commit".into(),
                ));
            }
            if let Some(tx) = Self::stream_sender(request) {
                for step in &steps {
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
            }
            if request.is_cancelled() {
                return Err(Error::Cancelled(
                    "Voxtral realtime quantum cancelled before seal".into(),
                ));
            }
            let staged = request.take_staged_stream_outputs()?;
            if request.is_cancelled() {
                return Err(Error::Cancelled(
                    "Voxtral realtime quantum cancelled during seal".into(),
                ));
            }
            Ok(staged)
        })();

        let staged = match execution {
            Ok(staged) => staged,
            Err(error) => {
                let _ = request.take_staged_stream_outputs();
                let active = state_lease.require_state_mut()?;
                active.stream_sequence = prior_stream_sequence;
                active.last_tokens_generated = prior_tokens;
                active.input_sample_rate = prior_input_sample_rate;
                active.model.rollback_realtime_quantum(
                    &mut active.state,
                    &mut cache,
                    &mut checkpoint,
                )?;
                let cancelled = matches!(error, Error::Cancelled(_));
                if cancelled {
                    state_lease.release()?;
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                state_lease.restore()?;
                return Err(error);
            }
        };

        if request.is_cancelled() {
            let _ = request.take_staged_stream_outputs();
            let active = state_lease.require_state_mut()?;
            active.stream_sequence = prior_stream_sequence;
            active.last_tokens_generated = prior_tokens;
            active.input_sample_rate = prior_input_sample_rate;
            active.model.rollback_realtime_quantum(
                &mut active.state,
                &mut cache,
                &mut checkpoint,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        let (tokens_generated, text, finished, sample_rate, sample_count) = {
            let active = state_lease.require_state_mut()?;
            let tokens_generated = active.state.tokens_generated().saturating_sub(prior_tokens);
            (
                tokens_generated,
                active.state.text().to_string(),
                active.state.is_finished(),
                active.input_sample_rate,
                active.state.source_sample_count(),
            )
        };
        let completions = cache.take_completed_writes();
        let cache_append = cache.context_len().saturating_sub(prior_cache_len);
        let active = state_lease.defer()?;
        let pending = super::super::PendingVoxtralRealtimeQuantum {
            session: session.clone(),
            active,
            cache,
            checkpoint,
            prior_last_tokens_generated: prior_tokens,
            prior_stream_sequence,
            prior_input_sample_rate,
            finished,
        };
        if let Err((register_error, pending)) =
            self.voxtral_realtime.register(scheduled.plan_id, pending)
        {
            let pending_session = pending.session.clone();
            let rollback = self
                .voxtral_realtime
                .resolve_pending(pending, super::super::PendingQuantumDecision::Abort)
                .and_then(|prepared| {
                    self.voxtral_realtime
                        .replace_in_flight(&pending_session, prepared.replacement)
                });
            return Err(match rollback {
                Ok(()) => register_error,
                Err(rollback_error) => Error::InferenceError(format!(
                    "{register_error}; Voxtral pending-quantum rollback also failed: {rollback_error}"
                )),
            });
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: match &scheduled.work {
                crate::engine::WorkUnit::RealtimePush { input, .. } => input.len(),
                crate::engine::WorkUnit::RealtimeFinish { .. } => 0,
                _ => unreachable!("realtime work authenticated above"),
            },
            tokens_generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        })
        .with_staged_stream_outputs(staged)
        .with_managed_cache_completions(completions)
        .with_managed_cache_append(cache_append)
        .requiring_pending_quantum())
    }

    pub(super) fn voxtral_realtime_staged_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        managed_state: super::super::RetainedRowManagedState,
    ) -> Result<ModelSessionResult> {
        self.voxtral_realtime_staged_request_inner(request, scheduled, managed_state, None)
    }

    pub(super) fn voxtral_realtime_staged_request_inner(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: super::super::RetainedRowManagedState,
        mut supplied_preparation: Option<
            crate::models::architectures::voxtral::realtime::VoxtralRealtimePreparedAudio,
        >,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Voxtral || !request.is_realtime_asr_session() {
            return Err(Error::InvalidInput(
                "Voxtral staged realtime work crossed its session route".into(),
            ));
        }
        if managed_state.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Voxtral staged realtime row unexpectedly received tensor state".into(),
            ));
        }
        let mut cache = managed_state.take_only_paged()?;
        managed_state.ensure_all_paged_consumed()?;
        let (operation_id, completed, max_output_steps, max_cache_append) = match scheduled.work {
            WorkUnit::RealtimePreparation {
                operation_id,
                max_output_steps,
                max_cache_append,
                ..
            } => (
                operation_id,
                RealtimeSubphase::Preparation,
                max_output_steps,
                max_cache_append,
            ),
            WorkUnit::RealtimePromptPrefill {
                operation_id,
                max_output_steps,
                cache_append,
            } => (
                operation_id,
                RealtimeSubphase::PromptPrefill { cache_append },
                max_output_steps,
                cache_append,
            ),
            WorkUnit::RealtimeDecodeContinuation {
                operation_id,
                max_output_steps,
                max_cache_append,
                ..
            } => (
                operation_id,
                RealtimeSubphase::DecodeContinuation,
                max_output_steps,
                max_cache_append,
            ),
            WorkUnit::RealtimeCompletion { operation_id } => {
                (operation_id, RealtimeSubphase::Completion, 0, 0)
            }
            _ => unreachable!("staged work authenticated by caller"),
        };
        let payload = request.realtime_asr_operation(operation_id)?;
        let session = scheduled.session_key();
        let mut state_lease = ExecutorStateLease::checkout(
            &self.voxtral_realtime.states,
            session.clone(),
            variant,
            "Voxtral staged realtime ASR",
        )?;
        if state_lease.state().is_none() {
            if !matches!(
                (&scheduled.work, &payload),
                (
                    WorkUnit::RealtimePreparation {
                        mode: RealtimePreparationMode::Push,
                        ..
                    },
                    RealtimeAsrOperationPayload::Push { .. }
                )
            ) {
                return Err(Error::InferenceError(
                    "Voxtral realtime state must start with push preparation".into(),
                ));
            }
            let model = self.with_registry(|registry| {
                registry.try_get_voxtral_lease(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!("Voxtral model {variant} is not loaded"))
                })
            })?;
            let state = model.start_realtime_state(request.asr_language_for_execution());
            state_lease.install_state(ActiveVoxtralRealtime {
                variant,
                model,
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: 0,
            })?;
        }
        let (prior_tokens, prior_stream_sequence, prior_input_sample_rate, prior_cache_len) = {
            let active = state_lease.require_state_mut()?;
            (
                active.last_tokens_generated,
                active.stream_sequence,
                active.input_sample_rate,
                cache.context_len(),
            )
        };
        let active = state_lease.require_state_mut()?;
        let mut checkpoint = active
            .model
            .begin_realtime_quantum(&mut active.state, &cache)?;
        state_lease.mark_dirty();

        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let mut step = None;
            match (&scheduled.work, &payload) {
                (
                    WorkUnit::RealtimePreparation { mode, input, .. },
                    RealtimeAsrOperationPayload::Push {
                        samples,
                        sample_rate,
                    },
                ) if *mode == RealtimePreparationMode::Push
                    && input.start == active.state.source_sample_count()
                    && input.len() == samples.len() =>
                {
                    let prepared = match supplied_preparation.take() {
                        Some(prepared) => prepared,
                        None => active
                            .model
                            .prepare_realtime_audio_batch(&[VoxtralRealtimePreparationBatchRow {
                                state: &active.state,
                                appended_samples: samples,
                                sample_rate: *sample_rate,
                                mode: VoxtralRealtimePreparationMode::Push,
                            }])?
                            .pop()
                            .ok_or_else(|| {
                                Error::InferenceError("Voxtral preparation omitted its row".into())
                            })?,
                    };
                    active
                        .model
                        .install_realtime_audio_preparation(&mut active.state, prepared)?;
                    active.input_sample_rate = *sample_rate;
                }
                (
                    WorkUnit::RealtimePreparation {
                        mode: RealtimePreparationMode::Finish,
                        ..
                    },
                    RealtimeAsrOperationPayload::Finish,
                ) => {
                    let sample_rate = active.input_sample_rate.max(1);
                    let prepared = match supplied_preparation.take() {
                        Some(prepared) => prepared,
                        None => active
                            .model
                            .prepare_realtime_audio_batch(&[VoxtralRealtimePreparationBatchRow {
                                state: &active.state,
                                appended_samples: &[],
                                sample_rate,
                                mode: VoxtralRealtimePreparationMode::Finish,
                            }])?
                            .pop()
                            .ok_or_else(|| {
                                Error::InferenceError(
                                    "Voxtral finish preparation omitted its row".into(),
                                )
                            })?,
                    };
                    active
                        .model
                        .install_realtime_audio_preparation(&mut active.state, prepared)?;
                }
                (WorkUnit::RealtimePromptPrefill { .. }, _) => {
                    step = Some(
                        active
                            .model
                            .prefill_realtime_in_quantum(&mut active.state, &mut cache)?,
                    );
                }
                (WorkUnit::RealtimeDecodeContinuation { .. }, _) => {
                    let mut rows = [VoxtralRealtimeDecodeBatchRow {
                        state: &mut active.state,
                        cache: &mut cache,
                    }];
                    step = active.model.decode_realtime_step_batch(&mut rows)?.pop();
                }
                (WorkUnit::RealtimeCompletion { .. }, RealtimeAsrOperationPayload::Finish) => {
                    step = Some(
                        active
                            .model
                            .complete_realtime_in_quantum(&mut active.state, &cache)?,
                    );
                }
                _ => {
                    return Err(Error::InferenceError(
                        "Voxtral realtime work and operation payload disagree".into(),
                    ));
                }
            }
            if request.is_cancelled() {
                return Err(Error::Cancelled("Voxtral staged quantum cancelled".into()));
            }
            let next = if active.state.is_finished() {
                None
            } else if let Some(prompt) = active.model.realtime_prompt_cache_append(&active.state)? {
                Some(RealtimeSubphase::PromptPrefill {
                    cache_append: prompt,
                })
            } else if active.model.realtime_decode_ready(&active.state)
                && max_output_steps > usize::from(step.is_some())
            {
                Some(RealtimeSubphase::DecodeContinuation)
            } else if matches!(payload, RealtimeAsrOperationPayload::Finish)
                && !matches!(completed, RealtimeSubphase::Completion)
            {
                Some(RealtimeSubphase::Completion)
            } else {
                None
            };
            if let (Some(tx), Some(step)) = (Self::stream_sender(request), step.as_ref()) {
                if !step.delta.is_empty() {
                    Self::stream_text_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut active.stream_sequence,
                        step.delta.clone(),
                    )?;
                }
                if matches!(completed, RealtimeSubphase::Completion) {
                    Self::stream_final_marker_with_policy(
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
                    "Voxtral staged quantum cancelled during seal".into(),
                ));
            }
            Ok((step, next, staged))
        })();
        let (_step, next, staged) = match execution {
            Ok(output) => output,
            Err(error) => {
                let _ = request.take_staged_stream_outputs();
                let rollback = (|| {
                    let active = state_lease.require_state_mut()?;
                    active.last_tokens_generated = prior_tokens;
                    active.stream_sequence = prior_stream_sequence;
                    active.input_sample_rate = prior_input_sample_rate;
                    active.model.rollback_realtime_quantum(
                        &mut active.state,
                        &mut cache,
                        &mut checkpoint,
                    )?;
                    state_lease.mark_clean();
                    Ok(())
                })()
                .and_then(|()| state_lease.restore());
                return match rollback {
                    Ok(()) => Err(error),
                    Err(rollback) => Err(Error::InferenceError(format!(
                        "{error}; Voxtral staged rollback also failed: {rollback}"
                    ))),
                };
            }
        };
        let appended = match cache.context_len().checked_sub(prior_cache_len) {
            Some(appended) if appended <= max_cache_append => appended,
            _ => {
                let error =
                    Error::InferenceError("Voxtral staged cache append violated its bound".into());
                let rollback = (|| {
                    let active = state_lease.require_state_mut()?;
                    active.last_tokens_generated = prior_tokens;
                    active.stream_sequence = prior_stream_sequence;
                    active.input_sample_rate = prior_input_sample_rate;
                    active.model.rollback_realtime_quantum(
                        &mut active.state,
                        &mut cache,
                        &mut checkpoint,
                    )?;
                    state_lease.mark_clean();
                    Ok(())
                })()
                .and_then(|()| state_lease.restore());
                return match rollback {
                    Ok(()) => Err(error),
                    Err(rollback) => Err(Error::InferenceError(format!(
                        "{error}; Voxtral staged rollback also failed: {rollback}"
                    ))),
                };
            }
        };
        let output_steps = usize::from(matches!(
            completed,
            RealtimeSubphase::PromptPrefill { .. } | RealtimeSubphase::DecodeContinuation
        ));
        let input_consumed = if next.is_none() {
            match &payload {
                RealtimeAsrOperationPayload::Push { samples, .. } => samples.len(),
                RealtimeAsrOperationPayload::Finish => 0,
            }
        } else {
            0
        };
        let (text, finished, sample_rate, sample_count) = {
            let active = state_lease.require_state_mut()?;
            (
                active.state.text().to_string(),
                active.state.is_finished(),
                active.input_sample_rate,
                active.state.source_sample_count(),
            )
        };
        if let Err(error) = state_lease.validate_defer() {
            let rollback = (|| {
                let active = state_lease.require_state_mut()?;
                active.last_tokens_generated = prior_tokens;
                active.stream_sequence = prior_stream_sequence;
                active.input_sample_rate = prior_input_sample_rate;
                active.model.rollback_realtime_quantum(
                    &mut active.state,
                    &mut cache,
                    &mut checkpoint,
                )?;
                state_lease.mark_clean();
                Ok(())
            })()
            .and_then(|()| state_lease.restore());
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback) => Err(Error::InferenceError(format!(
                    "{error}; Voxtral staged rollback also failed: {rollback}"
                ))),
            };
        }
        let completions = cache.take_completed_writes();
        let active = state_lease.defer_validated();
        let pending = super::super::PendingVoxtralRealtimeQuantum {
            session: session.clone(),
            active,
            cache,
            checkpoint,
            prior_last_tokens_generated: prior_tokens,
            prior_stream_sequence,
            prior_input_sample_rate,
            finished,
        };
        if let Err((error, pending)) = self.voxtral_realtime.register(scheduled.plan_id, pending) {
            let rollback = self
                .voxtral_realtime
                .resolve_pending(pending, super::super::PendingQuantumDecision::Abort)
                .and_then(|prepared| {
                    self.voxtral_realtime
                        .replace_in_flight(&session, prepared.replacement)
                });
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback) => Err(Error::InferenceError(format!(
                    "{error}; rollback also failed: {rollback}"
                ))),
            };
        }
        let outcome = RealtimeStageOutcome {
            plan_id: scheduled.plan_id,
            operation_id,
            completed,
            next,
            input_consumed,
            output_steps,
            cache_append: appended,
        };
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: output_steps,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        })
        .with_staged_stream_outputs(staged)
        .with_managed_cache_completions(completions)
        .with_managed_cache_append(appended)
        .requiring_pending_quantum()
        .with_realtime_stage_outcome(outcome))
    }
}
