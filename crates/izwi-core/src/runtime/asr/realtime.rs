//! Realtime ASR streams on RuntimeService: start, push, segment rollover and
//! finish, on the engine or the direct path.

use super::*;

impl RuntimeService {
    pub(super) async fn start_engine_realtime_stream(
        &self,
        variant: ModelVariant,
        language: Option<&str>,
        prompt: Option<&str>,
        context: RuntimeRequestContext,
        metadata_bytes: usize,
        session_lease: Arc<RealtimeAsrSessionLease>,
        absolute_deadline: Instant,
        idle_timeout: Duration,
    ) -> Result<RuntimeAsrRealtimeStream> {
        if variant.family() == ModelFamily::Voxtral
            && prompt.is_some_and(|prompt| !prompt.trim().is_empty())
        {
            return Err(Error::InvalidInput(
                "Voxtral realtime ASR does not support an initial text prompt".into(),
            ));
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let initial_spec = self.coordinator_job_for_input(
            request_id.clone(),
            CoordinatorLane::Realtime,
            context,
            metadata_bytes,
        );
        let initial_job = self
            .coordinator
            .admit_observed(initial_spec, host_input_observation(metadata_bytes)?)
            .await?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &initial_job,
                variant,
                CapabilityKind::RealtimeAsr,
                true,
                ExecutionTargetKind::RealtimeRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let (
            model,
            max_samples,
            max_host_bytes,
            max_tensor_bytes,
            max_output_steps,
            max_cache_append,
            voxtral_model,
        ) = match variant.family() {
            ModelFamily::Voxtral => {
                let model = self
                    .model_registry
                    .get_voxtral_lease(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelNotFound(format!("Voxtral model {variant} is not loaded"))
                    })?;
                let peak = model.realtime_stream_peak_reservation()?;
                let max_output_steps = model.realtime_max_output_steps()?;
                let max_cache_append = state_binding
                    .state
                    .managed_kv_runtime()
                    .map(|runtime| runtime.logical_token_reach())
                    .filter(|tokens| *tokens > 0)
                    .and_then(|tokens| usize::try_from(tokens).ok())
                    .ok_or_else(|| {
                        Error::ModelLoadError(
                            "Voxtral realtime state has no finite physical KV-token ceiling".into(),
                        )
                    })?;
                (
                    model.clone(),
                    peak.max_source_samples,
                    peak.max_host_bytes,
                    peak.max_tensor_bytes,
                    max_output_steps,
                    max_cache_append,
                    Some(model),
                )
            }
            _ => {
                return Err(Error::InvalidInput(
                    "unsupported Engine realtime ASR family".into(),
                ))
            }
        };
        let stream_resources = realtime_stream_resource_vector(
            self.backend_context().backend_kind,
            max_host_bytes
                .checked_add(u64::try_from(metadata_bytes).map_err(|_| {
                    Error::Overloaded("realtime ASR retained metadata usage exceeds u64".into())
                })?)
                .ok_or_else(|| {
                    Error::Overloaded(
                        "realtime ASR transactional host reservation overflowed".into(),
                    )
                })?,
            max_tensor_bytes,
        )?;
        let mut execution_spec = initial_job.spec.clone();
        let bridge = self.coordinator.bridge_preparation_admission(initial_job)?;
        execution_spec.resources = execution_spec.resources.checked_add(stream_resources)?;
        let job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                execution_spec,
                host_input_observation(metadata_bytes)?,
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };

        let mut request = EngineCoreRequest::asr_bytes(Vec::new()).with_model_variant(variant);
        if let Some(language) = language {
            request = request.with_language(language);
        }
        if let Some(prompt) = prompt {
            request = request.with_asr_prompt(prompt);
        }
        request.id = request_id;
        request.workload_class = context.workload_class;
        request.priority = context.priority;
        request.admission_ms = context.admission_ms;
        request.deadline = Some(absolute_deadline);
        request.params.max_tokens = max_cache_append;
        request.bind_execution_adapter(state_binding.execution.clone())?;
        request.bind_v2_state_runtime(
            state_binding.state.clone(),
            state_binding.state.state_fingerprint,
            self.backend_context().backend_kind,
        )?;
        let admission = self.core_engine.start_realtime_asr_session(request);
        let handle = tokio::time::timeout_at(absolute_deadline.into(), admission)
            .await
            .map_err(|_| Error::Timeout("realtime ASR Engine admission".into()))??;
        let engine_session = RuntimeEngineRealtimeSession {
            engine: self.core_engine.clone(),
            handle,
            model,
            retained_metadata_bytes: u64::try_from(metadata_bytes).map_err(|_| {
                Error::Overloaded("realtime ASR retained metadata usage exceeds u64".into())
            })?,
            max_output_steps,
            max_cache_append,
        };
        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: None,
            state: None,
            execution_contract: Some(execution_contract),
            residency_lease: Some(residency_lease),
            job: Some(job),
            session_lease: Some(session_lease),
            engine_session: Some(engine_session),
            voxtral_model,
            absolute_deadline,
            idle_timeout,
            last_activity: Instant::now(),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity.clone());
        Ok(RuntimeAsrRealtimeStream {
            variant,
            noise_gate: realtime_noise_gate(variant),
            resources,
            activity,
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples,
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
            segment: RealtimeEngineSegment::new(language, prompt),
        })
    }

    pub async fn try_start_asr_realtime_stream(
        &self,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<Option<RuntimeAsrRealtimeStream>> {
        let Some(variant) = resolve_asr_realtime_stream_variant(model_id) else {
            return Ok(None);
        };

        self.observe_broker_capability_request(CapabilityKind::RealtimeAsr, Some(variant), true)?;
        let started_at = Instant::now();
        let limits = self.asr_realtime_sessions.limits;
        let session_lease = self.asr_realtime_sessions.try_acquire()?;
        let absolute_deadline = started_at.checked_add(limits.max_lifetime).ok_or_else(|| {
            Error::ConfigError("realtime ASR absolute lifetime overflowed".to_string())
        })?;
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Realtime)
            .with_deadline(absolute_deadline);
        let metadata_bytes =
            language.map(str::len).unwrap_or_default() + prompt.map(str::len).unwrap_or_default();
        if realtime_asr_execution_path(variant) == RealtimeAsrPath::Engine {
            return self
                .start_engine_realtime_stream(
                    variant,
                    language,
                    prompt,
                    context,
                    metadata_bytes,
                    session_lease,
                    absolute_deadline,
                    limits.idle_timeout,
                )
                .await
                .map(Some);
        }
        let reservation = NativeAsrModel::conservative_realtime_stream_resource_reservation(
            variant, language, prompt, None,
        )?;
        let mut job_spec = self.coordinator_job_for_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Realtime,
            context,
            metadata_bytes,
        );
        job_spec.resources = add_realtime_stream_reservation(
            job_spec.resources,
            self.backend_context().backend_kind,
            reservation,
        )?;
        let job = self
            .coordinator
            .admit_observed(job_spec, host_input_observation(metadata_bytes)?)
            .await?;
        let (lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::RealtimeAsr,
                true,
                ExecutionTargetKind::RealtimeRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let physical_runtime = state_binding
            .state
            .retained_tensor_state_runtime()
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "realtime ASR did not bind its retained tensor runtime".into(),
                )
            })?;
        let model =
            self.model_registry.get_asr(variant).await.ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
        if !model.supports_realtime_stream_decode() {
            return Ok(None);
        }
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not resident")))?;
        let task_model = model.clone();
        let language = language.map(ToOwned::to_owned);
        let prompt = prompt.map(ToOwned::to_owned);
        let retained_metadata_bytes = metadata_bytes
            .checked_add(language.as_ref().map_or(0, String::capacity))
            .and_then(|bytes| bytes.checked_add(prompt.as_ref().map_or(0, String::capacity)))
            .ok_or_else(|| Error::Overloaded("realtime ASR metadata overflowed".to_string()))?;
        job.record_materialized_usage(retained_host_observation(&[
            metadata_bytes,
            language.as_ref().map_or(0, String::capacity),
            prompt.as_ref().map_or(0, String::capacity),
        ])?)?;
        let task_session_lease = session_lease.clone();
        let (state, language, prompt) = self
            .coordinator
            .run_loaded_blocking_stage(
                &job,
                execution_contract.clone(),
                WorkUnit::AtomicJob {
                    kind: "asr.realtime.start".to_string(),
                },
                move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = task_session_lease;
                    let native = task_model.start_realtime_stream_state_with_reservation(
                        language.as_deref(),
                        prompt.as_deref(),
                        None,
                        reservation,
                    )?;
                    let state = RuntimeAsrRealtimeState::new(
                        task_model.as_ref(),
                        native,
                        physical_runtime,
                    )?;
                    Ok((state, language, prompt))
                },
            )
            .await?;
        let steady_usage = realtime_state_observation(&state.native)?;
        job.record_materialized_usage(add_retained_host_bytes(
            steady_usage,
            retained_metadata_bytes,
        )?)?;
        job.prepare_materialized_release(steady_usage)?;
        drop((language, prompt));

        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: Some(model),
            state: Some(Arc::new(StdMutex::new(state))),
            execution_contract: Some(execution_contract),
            residency_lease: Some(lease),
            job: Some(job),
            session_lease: Some(session_lease),
            engine_session: None,
            voxtral_model: None,
            absolute_deadline,
            idle_timeout: limits.idle_timeout,
            last_activity: Instant::now(),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity.clone());

        Ok(Some(RuntimeAsrRealtimeStream {
            variant,
            noise_gate: realtime_noise_gate(variant),
            resources,
            activity,
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples: reservation.max_samples(),
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
            segment: RealtimeEngineSegment::default(),
        }))
    }

    /// Push input audio. For a model that hallucinates on non-speech the
    /// audio first passes the stream's noise gate, which releases it after a
    /// short lookahead with non-speech replaced by silence.
    pub async fn push_asr_realtime_samples(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        validate_realtime_input_copy(samples.len(), stream.max_samples)?;
        match stream.noise_gate.as_mut() {
            Some(gate) => {
                let gated = gate.push(samples, sample_rate);
                self.push_asr_realtime_model_samples(stream, &gated, sample_rate)
                    .await
            }
            None => {
                self.push_asr_realtime_model_samples(stream, samples, sample_rate)
                    .await
            }
        }
    }

    async fn push_asr_realtime_model_samples(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        // Reject an impossible single packet before acquiring residency or
        // allocating the owned copy needed by the blocking worker.
        validate_realtime_input_copy(samples.len(), stream.max_samples)?;
        if samples.is_empty() {
            stream.ensure_open()?;
            return Ok(Vec::new());
        }
        if stream.execution_path() == RealtimeAsrPath::Engine {
            let mut events = Vec::new();
            if realtime_segment_rollover_due(
                stream.committed_samples,
                samples.len(),
                stream.segment.trailing_quiet_samples,
                stream.max_samples,
                sample_rate,
            ) {
                // A failed rollover has already finished or detached the
                // current session, so the stream ends instead of rejecting
                // every later push.
                events = match self
                    .roll_over_engine_realtime_segment(stream, sample_rate)
                    .await
                {
                    Ok(events) => events,
                    Err(error) => {
                        stream.close();
                        return Err(error);
                    }
                };
            }
            let total_samples = stream
                .committed_samples
                .checked_add(samples.len())
                .ok_or_else(|| Error::Overloaded("realtime ASR sample clock overflowed".into()))?;
            validate_realtime_input_copy(total_samples, stream.max_samples)?;
            let handles = stream.begin_engine_operation(true)?;
            let operation_lease = self
                .model_lifecycle
                .try_acquire_ready_lease(stream.variant)
                .ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
                });
            let operation_lease = match operation_lease {
                Ok(lease) => lease,
                Err(error) => {
                    stream.close();
                    return Err(error);
                }
            };
            let RealtimeAsrEngineOperationHandles {
                session,
                job,
                session_lease,
                _guard: operation_guard,
            } = handles;
            if stream
                .engine_sample_rate
                .is_some_and(|established| established != sample_rate)
            {
                drop((operation_lease, session_lease, operation_guard));
                return Err(Error::InvalidInput(
                    "realtime ASR sample rate changed within one stream".into(),
                ));
            }
            // Voxtral recomputes its preparation over the committed prefix.
            let geometry = session
                .model
                .realtime_preparation_geometry_for_source_samples(
                    total_samples,
                    sample_rate,
                    VoxtralRealtimePreparationMode::Push,
                )?;
            let preparation_cost = voxtral_realtime_preparation_cost(&session.model, geometry)?;
            let committed_observation = voxtral_realtime_committed_observation(
                session.retained_metadata_bytes,
                session.model.realtime_prepared_resource_usage(geometry)?,
            )?;
            let ordering = match self
                .coordinator
                .acquire_job_ordering(&job, stream.operation_gate.clone())
                .await
            {
                Ok(ordering) => ordering,
                Err(error) => {
                    drop((operation_lease, session_lease, operation_guard));
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            let operation = session
                .engine
                .push_realtime_asr_samples_with_outputs_and_cost(
                    &session.handle,
                    samples.to_vec(),
                    sample_rate,
                    session.max_output_steps,
                    session.max_cache_append,
                    preparation_cost,
                );
            let result = match job.spec.deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline.into(), operation).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout("realtime ASR Engine push".into())),
                },
                None => operation.await,
            };
            drop((ordering, operation_lease, session_lease, operation_guard));
            let (ack, outputs) = match result {
                Ok(result) => result,
                Err(error) => {
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            if ack.accepted_samples() != samples.len() {
                stream.close();
                return Err(Error::InferenceError(
                    "realtime ASR Engine acknowledgement reported the wrong sample span".into(),
                ));
            }
            if let Err(error) = job.record_materialized_usage(committed_observation) {
                stream.close();
                return Err(error);
            }
            stream.committed_samples = total_samples;
            stream.engine_sample_rate = Some(sample_rate);
            stream.segment.trailing_quiet_samples =
                realtime_trailing_quiet_samples(stream.segment.trailing_quiet_samples, samples);
            events.extend(stream.map_engine_outputs(outputs));
            return Ok(events);
        }
        let handles = stream.begin_operation(true)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(err) => {
                stream.close();
                return Err(err);
            }
        };
        let operation_gate = stream.operation_gate.clone();
        let RealtimeAsrOperationHandles {
            model,
            state,
            execution_contract,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let result = async {
            // The ordering guard is held before this allocation, bounding
            // each stream to one owned packet even after caller timeout.
            let ordering = self
                .coordinator
                .acquire_job_ordering(&job, operation_gate)
                .await?;
            self.push_realtime_batched(RealtimePush {
                model,
                contract: execution_contract,
                job,
                state,
                samples: samples.to_vec(),
                sample_rate,
                _holds: RealtimePushHolds {
                    _ordering: ordering,
                    _residency: operation_lease,
                    _session: session_lease,
                    _operation: operation_guard,
                },
            })
            .await
        }
        .await;
        if let Err(err) = &result {
            if matches!(err, Error::Timeout(_)) {
                stream.close_due_to_timeout();
            } else {
                stream.close();
            }
        }
        let events = result?;
        Ok(map_native_realtime_events(events))
    }

    /// Flushes the current Engine session's audio and returns its final
    /// outputs. Success leaves the stream open so the caller decides between
    /// ending it and rolling over into a fresh session.
    pub(super) async fn finish_engine_realtime_segment(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<StreamingOutput>> {
        let handles = stream.begin_engine_operation(false)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(error) => {
                stream.close();
                return Err(error);
            }
        };
        let RealtimeAsrEngineOperationHandles {
            session,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let sample_rate = stream.engine_sample_rate.ok_or_else(|| {
            Error::InvalidInput("realtime ASR finish requires committed audio".into())
        })?;
        let geometry = session
            .model
            .realtime_preparation_geometry_for_source_samples(
                stream.committed_samples,
                sample_rate,
                VoxtralRealtimePreparationMode::Finish,
            )?;
        let preparation_cost = voxtral_realtime_preparation_cost(&session.model, geometry)?;
        let committed_observation = voxtral_realtime_committed_observation(
            session.retained_metadata_bytes,
            session.model.realtime_prepared_resource_usage(geometry)?,
        )?;
        let ordering = match self
            .coordinator
            .acquire_job_ordering(&job, stream.operation_gate.clone())
            .await
        {
            Ok(ordering) => ordering,
            Err(error) => {
                drop((operation_lease, session_lease, operation_guard));
                if matches!(error, Error::Timeout(_)) {
                    stream.close_due_to_timeout();
                } else {
                    stream.close();
                }
                return Err(error);
            }
        };
        let operation = session
            .engine
            .finish_realtime_asr_session_with_outputs_and_cost(
                &session.handle,
                session.max_output_steps,
                session.max_cache_append,
                preparation_cost,
            );
        let result = match job.spec.deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline.into(), operation).await {
                Ok(result) => result,
                Err(_) => Err(Error::Timeout("realtime ASR Engine finish".into())),
            },
            None => operation.await,
        };
        drop((ordering, operation_lease, session_lease, operation_guard));
        let (_ack, outputs) = match result {
            Ok(result) => result,
            Err(error) => {
                if matches!(error, Error::Timeout(_)) {
                    stream.close_due_to_timeout();
                } else {
                    stream.close();
                }
                return Err(error);
            }
        };
        if let Err(error) = job.record_materialized_usage(committed_observation) {
            stream.close();
            return Err(error);
        }
        Ok(outputs)
    }

    /// Ends the current Engine session at a segment boundary and continues
    /// the stream in a fresh one. The flushed text is returned as ordinary
    /// (non-final) events; the stream's transcript stays continuous.
    pub(super) async fn roll_over_engine_realtime_segment(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        if stream
            .engine_sample_rate
            .is_some_and(|established| established != sample_rate)
        {
            return Err(Error::InvalidInput(
                "realtime ASR sample rate changed within one stream".into(),
            ));
        }
        let (session_lease, absolute_deadline, idle_timeout) = stream.segment_authorities()?;
        let outputs = self.finish_engine_realtime_segment(stream).await?;
        let mut events = stream.map_engine_outputs(outputs);
        for event in &mut events {
            event.is_final = false;
        }
        // The session quota slot travels with `session_lease`; the finished
        // session's memory and residency are released before the next one
        // is admitted.
        tokio::time::timeout_at(
            absolute_deadline.into(),
            release_realtime_asr_cleanup(stream.detach()),
        )
        .await
        .map_err(|_| Error::Timeout("realtime ASR segment cleanup".into()))??;
        let language = stream.segment.language.clone();
        let prompt = stream.segment.prompt.clone();
        let metadata_bytes = language.as_deref().map(str::len).unwrap_or_default()
            + prompt.as_deref().map(str::len).unwrap_or_default();
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Realtime)
            .with_deadline(absolute_deadline);
        let mut next = self
            .start_engine_realtime_stream(
                stream.variant,
                language.as_deref(),
                prompt.as_deref(),
                context,
                metadata_bytes,
                session_lease,
                absolute_deadline,
                idle_timeout,
            )
            .await?;
        stream.continue_on(&mut next);
        Ok(events)
    }

    /// Finish the stream, first releasing the audio its noise gate holds.
    pub async fn finish_asr_realtime_stream(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        let mut events = Vec::new();
        if let Some((held, sample_rate)) = stream.noise_gate.take().and_then(|mut gate| {
            let rate = gate.sample_rate()?;
            Some((gate.flush(), rate))
        }) {
            events = self
                .push_asr_realtime_model_samples(stream, &held, sample_rate)
                .await?;
        }
        events.extend(self.finish_asr_realtime_model_stream(stream).await?);
        Ok(events)
    }

    async fn finish_asr_realtime_model_stream(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        if stream.execution_path() == RealtimeAsrPath::Engine {
            let outputs = self.finish_engine_realtime_segment(stream).await?;
            stream.close();
            return Ok(stream.map_engine_outputs(outputs));
        }
        let handles = stream.begin_operation(false)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(err) => {
                stream.close();
                return Err(err);
            }
        };
        let operation_gate = stream.operation_gate.clone();
        let RealtimeAsrOperationHandles {
            model,
            state,
            execution_contract,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let observation_job = job.clone();
        let result = run_realtime_blocking_operation(
            &self.coordinator,
            &job,
            execution_contract,
            operation_gate,
            "asr.realtime.finish",
            || {
                Ok(move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = session_lease;
                    let _operation_guard = operation_guard;
                    let mut state = state.lock().map_err(|_| {
                        Error::InferenceError("realtime ASR state mutex poisoned".to_string())
                    })?;
                    observation_job
                        .prepare_materialized_release(JobResourceObservation::default())?;
                    state.transact(model.as_ref(), |native| {
                        model.finish_realtime_stream(native)
                    })
                })
            },
        )
        .await;
        if matches!(&result, Err(Error::Timeout(_))) {
            stream.close_due_to_timeout();
        } else {
            stream.close();
        }
        let events = result?;
        Ok(map_native_realtime_events(events))
    }

    pub fn asr_realtime_stream_variant(&self, stream: &RuntimeAsrRealtimeStream) -> ModelVariant {
        stream.variant
    }
}
