//! Execution-plan begin and rollback: adapter contracts, scheduler profiles,
//! workspace backpressure and the capacity paths (suspend, preempt, defer).

use super::*;

impl EngineCore {
    pub(super) fn candidate_phase_quotas(
        capacity: usize,
        decode_work: usize,
        prefill_work: usize,
        single_slot_preference: SequencePhase,
    ) -> (usize, usize) {
        let capacity = capacity.max(1);
        match (decode_work > 0, prefill_work > 0) {
            (false, false) => (0, 0),
            (true, false) => (capacity, 0),
            (false, true) => (0, capacity),
            (true, true) if capacity == 1 => match single_slot_preference {
                SequencePhase::Decode => (1, 0),
                SequencePhase::Prefill => (0, 1),
            },
            (true, true) => {
                // Reserve enough candidate slots for the scheduler's decode
                // selection while preserving at least one prefill slot. Prefill
                // forms first, so any unused reservation is immediately
                // available to decode formation below.
                let decode = decode_work.min(capacity - 1).max(1);
                let prefill = prefill_work.min(capacity - decode).max(1);
                (decode, prefill)
            }
        }
    }

    pub(super) fn alternate_candidate_phase(phase: SequencePhase) -> SequencePhase {
        match phase {
            SequencePhase::Decode => SequencePhase::Prefill,
            SequencePhase::Prefill => SequencePhase::Decode,
        }
    }

    pub(super) fn next_candidate_preference(
        current: SequencePhase,
        both_phases: bool,
        decode_admitted: bool,
        prefill_admitted: bool,
    ) -> SequencePhase {
        let preferred_admitted = match current {
            SequencePhase::Decode => decode_admitted,
            SequencePhase::Prefill => prefill_admitted,
        };
        if both_phases && preferred_admitted {
            Self::alternate_candidate_phase(current)
        } else {
            current
        }
    }

    pub(in crate::engine) fn apply_adapter_execution_contract(
        request: &EngineCoreRequest,
        mut profile: ExecutionProfile,
    ) -> Result<ExecutionProfile> {
        let Some(binding) = request.execution_adapter_binding() else {
            return Ok(Self::force_scalar_execution(profile));
        };

        let prefill_work = WorkUnit::SequenceStep {
            phase: super::super::SequencePhase::Prefill,
            input: super::super::InputRange { start: 0, end: 0 },
            max_output_steps: 1,
            auxiliary_state: None,
        };
        let decode_work = WorkUnit::SequenceStep {
            phase: super::super::SequencePhase::Decode,
            input: super::super::InputRange { start: 0, end: 0 },
            max_output_steps: 1,
            auxiliary_state: None,
        };
        let atomic_work = WorkUnit::AtomicJob {
            kind: format!("{:?}", request.task_type).to_ascii_lowercase(),
        };
        let pipeline_work = WorkUnit::PipelineStage {
            name: format!("{:?}", request.task_type).to_ascii_lowercase(),
            ordinal: 0,
        };

        let (prefill_stage, decode_stage) = match profile.mode {
            ExecutionMode::Sequence | ExecutionMode::Realtime => (
                Some(binding.stage_for_work(&prefill_work)?),
                Some(binding.stage_for_work(&decode_work)?),
            ),
            ExecutionMode::Atomic | ExecutionMode::Artifact => {
                (Some(binding.stage_for_work(&atomic_work)?), None)
            }
            ExecutionMode::Pipeline => (Some(binding.stage_for_work(&pipeline_work)?), None),
        };

        let verify_mode = |phase: &str,
                           declared: NativeBatchMode,
                           stage: Option<&super::super::StageDescriptor>|
         -> Result<NativeBatchMode> {
            let Some(stage) = stage else {
                return Ok(NativeBatchMode::None);
            };
            if stage.batch_mode == NativeBatchMode::None {
                return Ok(NativeBatchMode::None);
            }
            if stage.batch_mode != declared {
                return Err(Error::InferenceError(format!(
                    "loaded adapter stage {} advertises {:?} {phase} batching, but the executor declared {:?}",
                    stage.name, stage.batch_mode, declared
                )));
            }
            Ok(stage.batch_mode)
        };

        // One mixed stage serving both phases runs resumable prefill spans
        // beside continuous decode rows in one model call; the executor keeps
        // declaring those, and the profile records the mixed step.
        profile.mixed_step = match (prefill_stage, decode_stage) {
            (Some(prefill), Some(decode))
                if prefill.batch_mode == NativeBatchMode::Mixed && prefill.id == decode.id =>
            {
                if profile.prefill != PrefillMode::Incremental
                    || profile.decode_batch != NativeBatchMode::Continuous
                {
                    return Err(Error::InferenceError(format!(
                        "loaded adapter stage {} runs mixed steps, but the executor declared {:?} prefill and {:?} decode batching",
                        prefill.name, profile.prefill, profile.decode_batch
                    )));
                }
                true
            }
            _ => false,
        };
        if !profile.mixed_step {
            profile.prefill_batch = verify_mode("prefill", profile.prefill_batch, prefill_stage)?;
            profile.decode_batch = verify_mode("decode", profile.decode_batch, decode_stage)?;
        }
        let max_batch_size = [prefill_stage, decode_stage]
            .into_iter()
            .flatten()
            .map(|stage| {
                if stage.concurrency == ConcurrencyClass::Batchable {
                    stage.max_batch_size
                } else {
                    1
                }
            })
            .max();
        profile.max_batch_size = max_batch_size
            .map(|maximum| profile.max_batch_size.min(maximum).max(1))
            .unwrap_or(1);
        profile.concurrency = if [prefill_stage, decode_stage]
            .into_iter()
            .flatten()
            .any(|stage| stage.concurrency == ConcurrencyClass::Batchable)
        {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        Ok(profile)
    }

    /// The session's execution profile under its adapter contract, or the
    /// executor's profile with the contract error. A profile resolved from
    /// the loaded model is cached for the session; any other profile (a model
    /// not loaded yet, or a contract mismatch) is resolved again next time.
    pub(super) async fn session_execution_profile(
        &mut self,
        request: &EngineCoreRequest,
        epoch: u64,
    ) -> std::result::Result<Arc<ExecutionProfile>, Box<(ExecutionProfile, Error)>> {
        if let Some(cached) = self
            .execution_profiles
            .get(&request.id)
            .filter(|cached| cached.epoch == epoch)
        {
            return Ok(cached.profile.clone());
        }
        let raw_profile = self
            .executor
            .execution_profile(request)
            .await
            .unwrap_or_else(|| {
                ExecutionProfile::fail_closed(
                    self.config.backend,
                    request.model_variant,
                    ExecutionMode::Atomic,
                )
            });
        let cacheable = raw_profile.resolved_from_loaded_model;
        match Self::apply_adapter_execution_contract(request, raw_profile.clone()) {
            Ok(profile) => {
                let profile = Arc::new(profile);
                if cacheable {
                    self.execution_profiles.insert(
                        request.id.clone(),
                        CachedExecutionProfile {
                            epoch,
                            profile: profile.clone(),
                        },
                    );
                }
                Ok(profile)
            }
            Err(error) => Err(Box::new((raw_profile, error))),
        }
    }

    /// Publish each session's execution profile to the scheduler. A cached
    /// profile was published when it was resolved, so only sessions without
    /// one are resolved here.
    pub(super) async fn refresh_scheduler_execution_profiles(&mut self) {
        let unresolved: Vec<_> = self
            .requests
            .values()
            .filter_map(|request| {
                let epoch = self.scheduler.get_sequence_id(&request.id)?;
                let cached = self
                    .execution_profiles
                    .get(&request.id)
                    .is_some_and(|cached| cached.epoch == epoch);
                (!cached).then(|| (request.clone(), epoch))
            })
            .collect();
        for (request, epoch) in unresolved {
            let session = super::super::SessionKey::new(request.id.clone(), epoch);
            match self.session_execution_profile(&request, epoch).await {
                Ok(profile) => {
                    self.scheduler.update_execution_profile(&session, &profile);
                }
                Err(refused) => {
                    let (raw_profile, error) = *refused;
                    warn!(
                        request_id = %request.id,
                        error = %error,
                        "Loaded adapter contract disagrees with executor; forcing scalar scheduling"
                    );
                    self.scheduler.update_execution_profile(
                        &session,
                        &Self::force_scalar_execution(raw_profile),
                    );
                }
            }
        }
    }

    pub(super) async fn begin_execution_plan(
        &mut self,
        scheduled: &super::super::scheduler::ScheduledRequest,
    ) -> std::result::Result<(), PlanBeginError> {
        let session = scheduled.session_key();
        if self.active_plans.contains_key(&scheduled.plan_id)
            || self.active_plans.contains_session(&session)
            || self.in_flight_dispatches.values().any(|dispatch| {
                dispatch.contains_plan(scheduled.plan_id) || dispatch.contains_session(&session)
            })
        {
            return Err(PlanBeginError::Engine(Error::InferenceError(format!(
                "execution plan {} duplicates an active session or plan quantum",
                scheduled.plan_id
            ))));
        }
        let request = self
            .requests
            .get(&scheduled.request_id)
            .cloned()
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "scheduled request {} is missing from the engine",
                    scheduled.request_id
                ))
            })?;
        let profile = self
            .session_execution_profile(&request, session.epoch)
            .await
            .map_err(|refused| PlanBeginError::AdapterContract(refused.1))?;
        if scheduled.is_prefill
            && profile.prefill != PrefillMode::Incremental
            && (scheduled.num_computed_tokens != 0
                || scheduled.num_tokens < request.num_prompt_tokens())
        {
            return Err(PlanBeginError::Engine(Error::InferenceError(format!(
                "full-prefill request {} was scheduled as a partial prompt quantum",
                scheduled.request_id
            ))));
        }
        let mut work = match profile.mode {
            // A full prefill writes the whole prompt in one quantum, or a
            // capacity replay's whole journal (prompt plus generated tokens).
            ExecutionMode::Sequence | ExecutionMode::Realtime
                if scheduled.is_prefill && profile.prefill == PrefillMode::Full =>
            {
                WorkUnit::SequenceStep {
                    phase: super::super::SequencePhase::Prefill,
                    input: super::super::InputRange {
                        start: 0,
                        end: scheduled.num_tokens,
                    },
                    max_output_steps: 1,
                    auxiliary_state: None,
                }
            }
            ExecutionMode::Sequence | ExecutionMode::Realtime => scheduled.work.clone(),
            ExecutionMode::Atomic | ExecutionMode::Artifact => WorkUnit::AtomicJob {
                kind: format!("{:?}", request.task_type).to_ascii_lowercase(),
            },
            ExecutionMode::Pipeline => WorkUnit::PipelineStage {
                name: format!("{:?}", request.task_type).to_ascii_lowercase(),
                ordinal: 0,
            },
        };
        // Managed arenas are registered once at model scope, while opaque
        // caches are authorized and reconciled from executor-owned tensors.
        // A row therefore never reserves either physical allocation again.
        let estimate = ResourceVector::zero();
        let bound_stage = request
            .execution_adapter_binding()
            .map(|binding| binding.stage_for_work(&work).cloned())
            .transpose()
            .map_err(PlanBeginError::AdapterContract)?;
        if let WorkUnit::SequenceStep {
            input,
            auxiliary_state,
            ..
        } = &mut work
        {
            if auxiliary_state.is_some() {
                return Err(PlanBeginError::Engine(Error::InferenceError(
                    "scheduler attempted to author retained auxiliary state".into(),
                )));
            }
            if let Some(stage) = bound_stage.as_ref() {
                let projected = request.project_auxiliary_state_spans(stage, *input)?;
                if let Some(spans) = projected.as_deref().filter(|spans| !spans.is_empty()) {
                    let runtime = request.managed_cache_runtime().ok_or_else(|| {
                        Error::InferenceError(
                            "clocked retained-state stage has no exact managed runtime".into(),
                        )
                    })?;
                    let tensor = runtime.tensor_state().ok_or_else(|| {
                        Error::InferenceError(
                            "clocked retained-state stage has no tensor-state arena".into(),
                        )
                    })?;
                    let mut previous_group = None;
                    for span in spans {
                        if span.input().is_empty()
                            || previous_group.is_some_and(|previous| previous >= span.group().get())
                        {
                            return Err(PlanBeginError::Engine(Error::InferenceError(
                                "projected clocked-state spans are not canonical and unique".into(),
                            )));
                        }
                        tensor.validate_group_clock(span.group(), span.clock())?;
                        previous_group = Some(span.group().get());
                    }
                }
                *auxiliary_state = projected;
            }
        }
        let realtime_state = match &mut work {
            WorkUnit::RealtimePreparation {
                retained_state_input,
                auxiliary_state,
                ..
            }
            | WorkUnit::RealtimeDecodeContinuation {
                retained_state_input,
                auxiliary_state,
                ..
            } => Some((*retained_state_input, auxiliary_state)),
            _ => None,
        };
        if let Some((state_input, auxiliary_state)) = realtime_state {
            if auxiliary_state.is_some() {
                return Err(PlanBeginError::Engine(Error::InferenceError(
                    "scheduler attempted to author retained realtime state".into(),
                )));
            }
            if let Some(selections) = bound_stage
                .as_ref()
                .and_then(|stage| stage.retained_state_selections.as_deref())
            {
                let runtime = request.managed_cache_runtime().ok_or_else(|| {
                    Error::InferenceError(
                        "retained realtime stage has no exact managed runtime".into(),
                    )
                })?;
                let tensor = runtime.tensor_state().ok_or_else(|| {
                    Error::InferenceError(
                        "retained realtime stage has no tensor-state arena".into(),
                    )
                })?;
                let mut spans = Vec::with_capacity(selections.len());
                let mut previous_group = None;
                for selection in selections {
                    if previous_group.is_some_and(|previous| previous >= selection.group().get()) {
                        return Err(PlanBeginError::Engine(Error::InferenceError(
                            "retained realtime selections are not canonical and unique".into(),
                        )));
                    }
                    tensor.validate_group_clock(selection.group(), selection.clock())?;
                    spans.push(ClockedStateSpan::new(
                        selection.group(),
                        selection.clock().clone(),
                        state_input,
                    )?);
                    previous_group = Some(selection.group().get());
                }
                *auxiliary_state = Some(Arc::from(spans));
            }
        }
        let batch_mode = bound_stage
            .as_ref()
            .map_or(NativeBatchMode::None, |stage| stage.batch_mode);
        let work_kind = match &work {
            WorkUnit::PreSequencePreparation { kind } => kind.clone(),
            // Both phases of a mixed stage share one lane.
            WorkUnit::SequenceStep { .. } if batch_mode == NativeBatchMode::Mixed => {
                "sequence_step".to_string()
            }
            WorkUnit::SequenceStep { phase, .. } => format!("{phase:?}").to_ascii_lowercase(),
            WorkUnit::SequenceAudioDecode { .. } => "sequence_audio_decode".to_string(),
            WorkUnit::SequenceFinalize { .. } => "sequence_finalize".to_string(),
            WorkUnit::RealtimePush { .. } => "realtime_push".to_string(),
            WorkUnit::RealtimeFinish { .. } => "realtime_finish".to_string(),
            WorkUnit::RealtimePreparation { mode, .. } => {
                format!(
                    "realtime_{}_preparation",
                    format!("{mode:?}").to_ascii_lowercase()
                )
            }
            WorkUnit::RealtimePromptPrefill { .. } => "realtime_prompt_prefill".to_string(),
            WorkUnit::RealtimeDecodeContinuation { .. } => {
                "realtime_decode_continuation".to_string()
            }
            WorkUnit::RealtimeCompletion { .. } => "realtime_completion".to_string(),
            WorkUnit::AtomicJob { kind } => kind.clone(),
            WorkUnit::PipelineStage { name, ordinal } => format!("{name}:{ordinal}"),
        };
        let bound_adapter = request
            .execution_adapter_binding()
            .zip(bound_stage.as_ref())
            .map(|(binding, stage)| binding.key_for_stage(stage.id))
            .transpose()?;
        let max_batch_size = bound_stage.as_ref().map_or(1, |stage| {
            if stage.concurrency == ConcurrencyClass::Batchable {
                stage.max_batch_size.min(profile.max_batch_size).max(1)
            } else {
                1
            }
        });
        let plan = ExecutionPlan {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            work,
            batch_key: BatchKey {
                backend: profile.backend,
                model_variant: profile.model_variant,
                task_type: request.task_type,
                work_kind,
                compute_dtype: profile.compute_dtype.clone(),
                kv_dtype: profile.kv_dtype.clone(),
                cache_namespace: profile
                    .cache_namespace
                    .clone()
                    .unwrap_or_else(|| "none".to_string()),
                adapter: bound_adapter,
            },
            batch_mode,
            max_batch_size,
            estimate,
            stage: bound_stage,
        };

        let tracker = self
            .execution_trackers
            .entry(scheduled.request_id.clone())
            .or_insert_with(|| ExecutionTracker::new(plan.session.clone()));
        if tracker.session() != &plan.session {
            return Err(PlanBeginError::Engine(Error::InferenceError(format!(
                "request {} already has a different active session",
                scheduled.request_id
            ))));
        }
        if tracker.state() == ExecutionState::Queued {
            tracker.transition(ExecutionState::Admitted)?;
        }
        let running_state = match plan.work {
            WorkUnit::PreSequencePreparation { .. } => {
                return Err(PlanBeginError::Engine(Error::InferenceError(
                    "pre-sequence preparation cannot enter EngineCore execution".to_string(),
                )));
            }
            WorkUnit::SequenceStep {
                phase: super::super::SequencePhase::Prefill,
                ..
            } => ExecutionState::Prefilling,
            WorkUnit::SequenceStep {
                phase: super::super::SequencePhase::Decode,
                ..
            } => ExecutionState::Decoding,
            WorkUnit::SequenceAudioDecode { .. } | WorkUnit::SequenceFinalize { .. } => {
                ExecutionState::Decoding
            }
            WorkUnit::RealtimePush { .. } => ExecutionState::RealtimeRunning,
            WorkUnit::RealtimeFinish { .. } => ExecutionState::RealtimeFinishing,
            WorkUnit::RealtimePreparation {
                mode: super::super::execution::RealtimePreparationMode::Push,
                ..
            } => ExecutionState::RealtimeRunning,
            WorkUnit::RealtimePreparation {
                mode: super::super::execution::RealtimePreparationMode::Finish,
                ..
            } => ExecutionState::RealtimeFinishing,
            WorkUnit::RealtimePromptPrefill { .. }
            | WorkUnit::RealtimeDecodeContinuation { .. }
            | WorkUnit::RealtimeCompletion { .. } => {
                if tracker.state() == ExecutionState::RealtimeFinishing {
                    ExecutionState::RealtimeFinishing
                } else {
                    ExecutionState::RealtimeRunning
                }
            }
            WorkUnit::AtomicJob { .. } => ExecutionState::AtomicRunning,
            WorkUnit::PipelineStage { .. } => ExecutionState::PipelineRunning,
        };
        if tracker.state() != running_state {
            tracker.transition(running_state)?;
        }
        tracker.begin_plan(&plan)?;
        let replaced = self.active_plans.insert(plan.plan_id, plan);
        debug_assert!(
            replaced.is_none(),
            "active plan was preflight-checked above"
        );
        Ok(())
    }

    pub(super) fn rollback_unexecuted_schedule(&mut self, scheduled: &[super::super::scheduler::ScheduledRequest]) {
        let requested_plans = scheduled
            .iter()
            .map(|scheduled| scheduled.plan_id)
            .collect::<HashSet<_>>();
        let registered_batches = self
            .in_flight_dispatches
            .iter()
            .filter(|(_, dispatch)| {
                dispatch
                    .physical_batch
                    .rows
                    .iter()
                    .any(|row| requested_plans.contains(&row.plan_id))
            })
            .map(|(batch_id, _)| *batch_id)
            .collect::<Vec<_>>();
        let mut exact_schedule = scheduled.to_vec();
        let mut exact_plans = requested_plans;
        for batch_id in registered_batches {
            let Some(dispatch) = self.in_flight_dispatches.remove(&batch_id) else {
                continue;
            };
            self.active_stream_batches.remove(&batch_id);
            for row in dispatch.scheduled {
                if exact_plans.insert(row.plan_id) {
                    exact_schedule.push(row);
                }
            }
        }

        let rolled_back_plans = exact_plans;
        self.active_stream_batches.retain(|_, batch| {
            !batch
                .rows
                .keys()
                .any(|plan_id| rolled_back_plans.contains(plan_id))
        });
        self.rollback_schedule_state(&exact_schedule);
    }

    pub(super) fn rollback_schedule_state(&mut self, exact_schedule: &[super::super::scheduler::ScheduledRequest]) {
        for scheduled in exact_schedule {
            let session = scheduled.session_key();
            if let Err(error) = self.executor.finalize_pending_quantum(
                scheduled.plan_id,
                &session,
                PendingQuantumDecision::Abort,
            ) {
                warn!(
                    plan_id = scheduled.plan_id,
                    request_id = %scheduled.request_id,
                    error = %error,
                    "Failed to abort executor pending quantum during schedule rollback"
                );
            }
            if let Some(reservation) = self.active_managed_cache.remove(&scheduled.plan_id) {
                if let Err(error) = self.managed_kv_cache.finalize(&reservation, None, false) {
                    warn!(
                        plan_id = scheduled.plan_id,
                        error = %error,
                        "Failed to abort an unexecuted managed KV transaction"
                    );
                }
            }
            if let Some(plan) = self.active_plans.remove(&scheduled.plan_id) {
                if plan.session != session {
                    warn!(
                        plan_id = scheduled.plan_id,
                        request_id = %scheduled.request_id,
                        "Unexecuted plan rollback found a mismatched session fence"
                    );
                }
                if !self.scheduler.refund_unexecuted_service(
                    &session,
                    Scheduler::service_units_for_work(scheduled.num_tokens, &scheduled.work),
                ) {
                    warn!(
                        plan_id = scheduled.plan_id,
                        request_id = %scheduled.request_id,
                        "Scheduler rejected unexecuted weighted-service refund"
                    );
                }
            }
            if let Some(tracker) = self.execution_trackers.get_mut(&scheduled.request_id) {
                if tracker.session() == &session {
                    tracker.rollback_unexecuted_plan(scheduled.plan_id);
                }
            }
            self.scheduler.release_unexecuted_quantum(scheduled);
            self.release_managed_session_if_ready(&session);
        }
    }

    /// Roll back every registered physical ticket. The main engine driver uses
    /// this hook when a prepared dispatch can no longer produce a completion
    /// (for example, a cancelled/join-failed runner task).
    #[cfg(test)]
    pub(in crate::engine) fn rollback_all_in_flight_dispatches(&mut self) -> usize {
        let batch_ids = self
            .in_flight_dispatches
            .keys()
            .copied()
            .collect::<Vec<_>>();
        self.rollback_in_flight_dispatches(&batch_ids)
    }

    /// Roll back only the exact physical tickets owned by one abandoned
    /// prepared step. Later tickets may already be present by the time recovery
    /// acquires the core lock and must remain untouched.
    pub(in crate::engine) fn rollback_in_flight_dispatches(&mut self, batch_ids: &[BatchId]) -> usize {
        let mut scheduled = Vec::new();
        let mut removed = 0;
        for batch_id in batch_ids.iter().copied().collect::<HashSet<_>>() {
            let Some(dispatch) = self.in_flight_dispatches.remove(&batch_id) else {
                continue;
            };
            removed += 1;
            self.active_stream_batches.remove(&batch_id);
            scheduled.extend(dispatch.scheduled);
        }
        self.rollback_schedule_state(&scheduled);
        removed
    }

    pub(super) fn defer_unexecuted_schedule_for_capacity(
        &mut self,
        scheduled: &[super::super::scheduler::ScheduledRequest],
    ) {
        self.rollback_unexecuted_schedule(scheduled);
        self.scheduler.yield_capacity_blocked(scheduled);
        let retry_at = Instant::now() + self.retry_policy.execution_delay(1);
        for scheduled in scheduled {
            let session = scheduled.session_key();
            if !self.scheduler.defer_execution_retry(&session, retry_at) {
                warn!(
                    request_id = %session.request_id,
                    session_epoch = session.epoch,
                    "Scheduler rejected managed KV capacity backpressure"
                );
            }
        }
    }

    /// Only a rejected admission can retry a published stream without model
    /// rollback: no device entry, output, or state receipts may have occurred.
    pub(super) fn is_workspace_backpressure(result: &ExecutorStepResult) -> bool {
        result.provenance
            == OutcomeProvenance::failure(
                FailureOrigin::WorkspaceAdmission,
                DispatchState::NotStarted,
            )
            && result.dispatch.kind == super::super::BatchDispatchKind::NotDispatched
            && result.safe_point
            && result.output.tokens_processed == 0
            && result.output.tokens_generated == 0
            && !result.output.finished
            && result.output.error.is_some()
            && result.output.text.is_none()
            && result.output.audio.is_none()
            && result.output.input_transcription.is_none()
            && result.output.phase_timing_override.is_none()
            && result.output.asr_diagnostics.is_none()
            && result.staged_stream_outputs.is_empty()
            && result.managed_cache.is_none()
            && result.managed_cache_completions.is_empty()
            && result.managed_cache_append.is_none()
            && !result.pending_quantum_required
            && result.realtime_stage_outcome.is_none()
            && result.clocked_state_completion.is_none()
            && matches!(&result.disposition, ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::RetrySameSession
                    && failure.kind == super::super::execution::FailureKind::ResourceExhausted
                    && failure.scope == super::super::execution::FailureScope::PhysicalBatch
                    && failure.health == super::super::execution::HealthImpact::None)
    }

    pub(super) fn observe_workspace_retry(&mut self, session: &super::super::SessionKey) {
        let previous = self
            .workspace_retry_progress
            .insert(session.clone(), self.workspace_progress_epoch);
        if previous.is_some_and(|epoch| epoch != self.workspace_progress_epoch) {
            // Other work can release live workspace/request reservations. A
            // progressing owner must not exhaust a waiting stream's budget.
            self.execution_retry_attempts.remove(session);
            self.workspace_waits.remove(session);
        }
    }

    pub(super) fn adapt_workspace_batch(
        &mut self,
        scheduled: &super::super::scheduler::ScheduledRequest,
        width: usize,
    ) -> bool {
        let session = scheduled.session_key();
        let reduced = if width > 1 {
            let next = (width / 2).max(1);
            let cap = self
                .workspace_row_limits
                .entry(session.clone())
                .or_insert(width);
            let reduced = next < *cap;
            *cap = (*cap).min(next);
            reduced
        } else if scheduled.is_prefill {
            self.scheduler
                .reduce_workspace_prefill_quantum(&session, scheduled.num_tokens)
        } else {
            false
        };
        if reduced {
            // A smaller physical shape is a new attempt at fitting; only
            // unchanged failures consume the bounded no-progress retry budget.
            self.execution_retry_attempts.remove(&session);
            if let Some(wait) = self.workspace_waits.get_mut(&session) {
                // Retry the smaller shape promptly; the wait keeps its clock.
                wait.attempts = 0;
            }
        } else {
            self.workspace_pressure.insert(session, scheduled.clone());
        }
        reduced
    }

    /// How a request's managed-KV admission claim is sized. Chat claims grow
    /// with generation (TTS/ASR retained sessions keep full claims so their
    /// sealed floors and frame budgets stay guaranteed).
    pub(super) fn managed_claim_policy(&self, request: &EngineCoreRequest) -> ManagedClaimPolicy {
        if self.uses_incremental_chat_capacity(request) {
            return ManagedClaimPolicy::Incremental;
        }
        match u32::try_from(self.config.managed_chat_claim_quantum_tokens) {
            Ok(quantum_tokens)
                if quantum_tokens > 0 && request.task_type == super::super::TaskType::Chat =>
            {
                ManagedClaimPolicy::Chunked { quantum_tokens }
            }
            _ => ManagedClaimPolicy::Full,
        }
    }

    pub(super) fn uses_incremental_chat_capacity(&self, request: &EngineCoreRequest) -> bool {
        self.config.cuda_incremental_chat_enabled()
            && request.task_type == super::super::TaskType::Chat
            && request
                .prepared_chat_model_for_executor()
                .is_ok_and(|model| model.admits_incrementally_on_cuda())
    }

    /// Whether a session released for capacity is rebuilt by replaying its
    /// tokens: its loaded contract declares token replay and capacity replay
    /// is on (always for the CUDA incremental route, whose claims rely on it).
    pub(super) fn replays_for_capacity(&self, request_id: &RequestId) -> bool {
        (self.config.enable_capacity_replay || self.config.cuda_incremental_chat_enabled())
            && self.scheduler.replays_tokens(request_id)
    }

    /// Suspend a replayable claim holder so a blocked request can grow. A
    /// row on the CUDA incremental route (claims of exactly its next span)
    /// protects itself as soon as it blocks. Chunked claim holders act only
    /// once their model is in `deadlocked`: then every holder has been seen
    /// blocked, possibly in different steps, so the survivor and victim are
    /// chosen among all of them, and a row blocked while its peers still make
    /// progress waits for them instead of evicting one.
    pub(super) async fn suspend_published_session_for_capacity(
        &mut self,
        blocked: &[super::super::scheduler::ScheduledRequest],
        deadlocked: &HashSet<ModelInstanceId>,
    ) -> Result<bool> {
        let deadlocked_holders = deadlocked
            .iter()
            .filter_map(|model| self.claim_waiters.get(model))
            .flat_map(|waiters| waiters.holders.iter().cloned());
        let candidates = blocked
            .iter()
            .filter(|row| {
                self.requests
                    .get(&row.request_id)
                    .is_some_and(|request| self.uses_incremental_chat_capacity(request))
            })
            .map(|row| row.session_key())
            .chain(deadlocked_holders)
            .filter(|session| self.replays_for_capacity(&session.request_id))
            .collect::<Vec<_>>();
        let Some(survivor) = self.scheduler.capacity_survivor(candidates) else {
            return Ok(false);
        };
        let model = self
            .requests
            .get(&survivor.request_id)
            .and_then(|request| request.managed_cache_runtime())
            .map(|runtime| runtime.plan().model_instance)
            .ok_or_else(|| Error::InferenceError("capacity survivor lost runtime".into()))?;
        let candidates = self
            .managed_kv_cache
            .capacity_claim_sessions(model)
            .into_iter()
            .filter(|session| *session != survivor)
            .filter(|session| self.replays_for_capacity(&session.request_id))
            .filter(|session| {
                !self.active_plans.contains_session(session)
                    && !self
                        .active_managed_cache
                        .values()
                        .any(|reservation| reservation.session == *session)
            })
            .collect::<Vec<_>>();
        let Some(victim) = self
            .scheduler
            .published_capacity_victim(candidates.iter().cloned(), &survivor)
        else {
            if candidates
                .iter()
                .any(|session| self.scheduler.capacity_replays_exhausted(session))
            {
                super::super::metrics::record_capacity_replay_capped();
            }
            return Ok(false);
        };
        let Some(replay_tokens) = self.executor.suspend_session_for_capacity(&victim).await? else {
            return Ok(false);
        };
        self.managed_kv_cache.release_session(&victim)?;
        if !self
            .scheduler
            .suspend_for_replay(&victim, replay_tokens, survivor)
        {
            return Err(Error::InferenceError(
                "capacity replay lost scheduler session".into(),
            ));
        }
        self.clear_execution_state_for_capacity_replay(&victim);
        super::super::metrics::record_capacity_suspension();
        Ok(true)
    }

    /// Chunked chat claims grow during generation, so every claim holder of a
    /// model can end up waiting to grow at once while none can finish. Record
    /// this step's blocked holders and return the models whose every claim
    /// holder has been seen blocked since the holder set last changed. Decode
    /// selection is clamped to the physical width, so the holders need not
    /// all be blocked in the same step.
    pub(super) fn claim_deadlocked_models(
        &mut self,
        blocked: &[super::super::scheduler::ScheduledRequest],
    ) -> HashSet<ModelInstanceId> {
        let mut by_model = HashMap::<ModelInstanceId, Vec<super::super::SessionKey>>::new();
        for row in blocked {
            let Some(request) = self.requests.get(&row.request_id) else {
                continue;
            };
            if !matches!(
                self.managed_claim_policy(request),
                ManagedClaimPolicy::Chunked { .. }
            ) {
                continue;
            }
            let Some(model) = request
                .managed_cache_runtime()
                .map(|runtime| runtime.plan().model_instance)
            else {
                continue;
            };
            by_model.entry(model).or_default().push(row.session_key());
        }
        let mut deadlocked = HashSet::new();
        for (model, blocked_sessions) in by_model {
            let mut holders = self.managed_kv_cache.capacity_claim_sessions(model);
            holders.sort_by(|left, right| {
                (&left.request_id, left.epoch).cmp(&(&right.request_id, right.epoch))
            });
            let waiters = self.claim_waiters.entry(model).or_default();
            if waiters.holders != holders {
                waiters.holders = holders.clone();
                waiters.blocked.clear();
            }
            waiters.blocked.extend(blocked_sessions);
            if holders.len() > 1
                && holders
                    .iter()
                    .all(|holder| waiters.blocked.contains(holder))
            {
                deadlocked.insert(model);
            }
        }
        deadlocked
    }

    /// When no row ran this step and suspension and preemption freed
    /// nothing, fail the youngest lowest-priority claim holder of each
    /// deadlocked model so the others can complete.
    pub(super) async fn fail_claim_deadlocked_session(&mut self, deadlocked: HashSet<ModelInstanceId>) {
        for model in deadlocked {
            let holders = self
                .claim_waiters
                .get(&model)
                .map(|waiters| waiters.holders.clone())
                .unwrap_or_default();
            // Session epochs are scheduler sequence ids, so the highest epoch
            // among the lowest-priority holders is the youngest.
            let Some(victim) = holders
                .iter()
                .min_by_key(|session| {
                    (
                        self.requests
                            .get(&session.request_id)
                            .map(|request| request.priority),
                        std::cmp::Reverse(session.epoch),
                    )
                })
                .cloned()
            else {
                continue;
            };
            self.claim_waiters.remove(&model);
            warn!(
                request_id = %victim.request_id,
                session_epoch = victim.epoch,
                claim_holders = holders.len(),
                "Every managed KV claim holder is waiting to grow; failing the youngest"
            );
            super::super::metrics::record_claim_deadlock_abort();
            let message =
                "managed KV capacity is exhausted by concurrent requests; retry later".to_string();
            self.terminate_request_session(
                &victim,
                ExecutorOutput::error(victim.request_id.clone(), message.clone()),
                ExecutionDisposition::Failed(ExecutionFailure {
                    kind: FailureKind::ResourceExhausted,
                    scope: FailureScope::Row,
                    retry: RetryDisposition::Never,
                    health: HealthImpact::None,
                    message,
                }),
                OutcomeProvenance::failure(
                    FailureOrigin::AdapterPlanning,
                    DispatchState::NotStarted,
                ),
            )
            .await;
        }
    }

    pub(super) async fn preempt_unpublished_session_for_capacity(
        &mut self,
        blocked: &[super::super::scheduler::ScheduledRequest],
    ) -> Result<bool> {
        let Some((blocked_request, blocked_model)) = blocked
            .iter()
            .filter_map(|scheduled| {
                let request = self.requests.get(&scheduled.request_id)?.clone();
                let model = request.managed_cache_runtime()?.plan().model_instance;
                Some((request, model))
            })
            .max_by_key(|(request, _)| request.priority)
        else {
            return Ok(false);
        };
        let candidates = self
            .managed_kv_cache
            .capacity_claim_sessions(blocked_model)
            .into_iter()
            .filter(|session| session.request_id != blocked_request.id)
            .filter(|session| {
                !self.active_plans.contains_session(session)
                    && !self
                        .active_managed_cache
                        .values()
                        .any(|reservation| &reservation.session == session)
            });
        let Some(victim) = self
            .scheduler
            .capacity_preemption_candidate(candidates, blocked_request.priority)
        else {
            return Ok(false);
        };

        let release = self.executor.cleanup_session(&victim).await;
        if !release.confirmed {
            warn!(
                request_id = %victim.request_id,
                session_epoch = victim.epoch,
                "Executor could not confirm capacity-preemption cleanup"
            );
            return Ok(false);
        }
        self.managed_kv_cache.release_session(&victim)?;
        if !self.scheduler.restart_request_for_recompute(&victim) {
            return Err(Error::InferenceError(
                "scheduler rejected a validated capacity-preemption restart".into(),
            ));
        }
        self.clear_exact_execution_state(&victim);
        let retry_at = Instant::now()
            + self
                .retry_policy
                .execution_delay(2)
                .max(Duration::from_millis(1));
        if !self.scheduler.defer_execution_retry(&victim, retry_at) {
            return Err(Error::InferenceError(
                "scheduler rejected a capacity-preemption retry fence".into(),
            ));
        }
        debug!(
            request_id = %victim.request_id,
            session_epoch = victim.epoch,
            blocked_request_id = %blocked_request.id,
            "Released an unpublished lower-priority session for managed KV capacity"
        );
        super::super::metrics::record_recompute_preemption();
        Ok(true)
    }

    /// Fail a request whose bound adapter cannot run the work its executor
    /// schedules. Nothing was dispatched, so the session is released like any
    /// other terminal failure.
    /// A session whose plan failed to begin on [`MAX_PLAN_BEGIN_ATTEMPTS`]
    /// consecutive steps fails terminally instead of retrying forever.
    /// Roll back rows whose managed-state preparation failed and decide their
    /// fate without failing the step for every other row. `Overloaded` means
    /// the request can never fit (or a device allocation failed after
    /// admission), so it fails now; any other error is retried on later
    /// steps up to [`MAX_PLAN_BEGIN_ATTEMPTS`] and then fails the request.
    pub(super) async fn resolve_managed_prepare_failures(&mut self, failures: Vec<ManagedPrepareFailure>) {
        if failures.is_empty() {
            return;
        }
        let scheduled = failures
            .iter()
            .map(|failure| failure.scheduled.clone())
            .collect::<Vec<_>>();
        self.rollback_unexecuted_schedule(&scheduled);
        for ManagedPrepareFailure { session, error, .. } in failures {
            let terminal = matches!(error, Error::Overloaded(_))
                || plan_begin_attempts_exhausted(&mut self.managed_prepare_attempts, &session);
            if !terminal {
                debug!(
                    request_id = %session.request_id,
                    error = %error,
                    "Managed KV preparation failed for one row; retrying it on a later step"
                );
                continue;
            }
            self.managed_prepare_attempts.remove(&session);
            let resource_exhausted = matches!(error, Error::Overloaded(_));
            warn!(
                request_id = %session.request_id,
                session_epoch = session.epoch,
                error = %error,
                "Managed KV state could not be prepared; failing the request"
            );
            let message = format!("managed KV state could not be prepared: {error}");
            self.terminate_request_session(
                &session,
                ExecutorOutput::error(session.request_id.clone(), message.clone()),
                ExecutionDisposition::Failed(ExecutionFailure {
                    kind: if resource_exhausted {
                        FailureKind::ResourceExhausted
                    } else {
                        FailureKind::Internal
                    },
                    scope: FailureScope::Row,
                    retry: RetryDisposition::Never,
                    health: HealthImpact::None,
                    message,
                }),
                OutcomeProvenance::failure(
                    FailureOrigin::AdapterPlanning,
                    DispatchState::NotStarted,
                ),
            )
            .await;
        }
    }

    pub(super) async fn fail_unstartable_plan(&mut self, session: &super::super::SessionKey, error: Error) {
        warn!(
            request_id = %session.request_id,
            session_epoch = session.epoch,
            error = %error,
            attempts = MAX_PLAN_BEGIN_ATTEMPTS,
            "Execution plan could not begin; failing the request"
        );
        self.plan_begin_attempts.remove(session);
        let message =
            format!("execution could not begin after {MAX_PLAN_BEGIN_ATTEMPTS} attempts: {error}");
        self.terminate_request_session(
            session,
            ExecutorOutput::error(session.request_id.clone(), message.clone()),
            ExecutionDisposition::Failed(ExecutionFailure {
                kind: FailureKind::Internal,
                scope: FailureScope::Row,
                retry: RetryDisposition::Never,
                health: HealthImpact::None,
                message,
            }),
            OutcomeProvenance::failure(FailureOrigin::AdapterPlanning, DispatchState::NotStarted),
        )
        .await;
    }

    pub(super) async fn fail_adapter_contract_mismatch(&mut self, session: &super::super::SessionKey, error: Error) {
        warn!(
            request_id = %session.request_id,
            session_epoch = session.epoch,
            error = %error,
            "Loaded adapter cannot run the scheduled work; failing the request"
        );
        let message = format!("execution adapter contract mismatch: {error}");
        self.terminate_request_session(
            session,
            ExecutorOutput::error(session.request_id.clone(), message.clone()),
            ExecutionDisposition::Failed(ExecutionFailure {
                kind: FailureKind::Internal,
                scope: FailureScope::Row,
                retry: RetryDisposition::Never,
                health: HealthImpact::None,
                message,
            }),
            OutcomeProvenance::failure(FailureOrigin::AdapterPlanning, DispatchState::NotStarted),
        )
        .await;
    }
}
