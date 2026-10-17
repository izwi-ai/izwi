//! Request admission and execution: coordinator jobs, capability loads, engine
//! admission and the non-streaming and streaming request paths.

use super::*;

impl RuntimeService {
    pub(crate) fn coordinator_job_for_input(
        &self,
        request_id: impl Into<String>,
        lane: CoordinatorLane,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
    ) -> JobSpec {
        let resources =
            transient_resources(self.backend_router.context().backend_kind, input_bytes);
        JobSpec {
            request_id: request_id.into(),
            lane,
            priority: runtime_context.priority,
            workload_class: runtime_context.workload_class,
            deadline: runtime_context.deadline,
            resources,
        }
    }

    pub(crate) fn coordinator_job_for_audio_input(
        &self,
        request_id: impl Into<String>,
        lane: CoordinatorLane,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
    ) -> Result<JobSpec> {
        let mut spec =
            self.coordinator_job_for_input(request_id, lane, runtime_context, input_bytes);
        spec.resources = spec.resources.checked_add(audio_decode_resources(
            self.backend_router.context().backend_kind,
        ))?;
        Ok(spec)
    }

    pub(crate) async fn prepare_engine_request_blocking<F>(
        &self,
        variant: ModelVariant,
        task_type: TaskType,
        streaming: bool,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
        additional_resources: ResourceVector,
        build: F,
    ) -> Result<AdmittedEngineRequest>
    where
        F: FnOnce(Arc<ModelRegistry>) -> Result<EngineCoreRequest> + Send + 'static,
    {
        self.prepare_engine_request_blocking_with_input(
            variant,
            task_type,
            streaming,
            runtime_context,
            input_bytes,
            additional_resources,
            |_job| async { Ok(()) },
            move |registry, ()| build(registry),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn prepare_engine_request_blocking_with_input<P, PFut, T, F>(
        &self,
        variant: ModelVariant,
        task_type: TaskType,
        streaming: bool,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
        additional_resources: ResourceVector,
        prepare_input: P,
        build: F,
    ) -> Result<AdmittedEngineRequest>
    where
        P: FnOnce(JobLease) -> PFut,
        PFut: Future<Output = Result<T>>,
        T: Send + 'static,
        F: FnOnce(Arc<ModelRegistry>, T) -> Result<EngineCoreRequest> + Send + 'static,
    {
        let mut effective_context = runtime_context;
        if streaming && effective_context.workload_class == WorkloadClass::Online {
            effective_context.workload_class = WorkloadClass::Streaming;
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let initial_lane = if task_type == TaskType::ASR
            && matches!(
                variant.family(),
                ModelFamily::Qwen3Asr
                    | ModelFamily::WhisperAsr
                    | ModelFamily::VibeVoiceAsr
                    | ModelFamily::GraniteSpeechAsr
                    | ModelFamily::Lfm25Audio
            ) {
            CoordinatorLane::Atomic
        } else {
            coordinator_lane_for_metadata(
                task_type,
                Some(variant),
                streaming,
                effective_context.workload_class,
            )
        };
        let mut spec = self.coordinator_job_for_input(
            request_id,
            initial_lane,
            effective_context,
            input_bytes,
        );
        if task_type == TaskType::ASR {
            // The audio length is unknown before decoding; one full chunk
            // bounds the request whatever its length.
            if let Some(resources) = asr_transient_resources(
                self.backend_router.context().backend_kind,
                variant,
                input_bytes,
                None,
            )? {
                spec.resources = resources;
            }
        }
        spec.resources = spec.resources.checked_add(additional_resources)?;
        if task_decodes_audio(task_type) {
            spec.resources = spec.resources.checked_add(audio_decode_resources(
                self.backend_router.context().backend_kind,
            ))?;
        }
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;

        let prepared_input = match job.spec.deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), prepare_input(job.clone()))
                .await
                .map_err(|_| Error::Timeout(job.spec.request_id.clone()))??,
            None => prepare_input(job.clone()).await?,
        };
        let residency_lease = match job.spec.deadline {
            Some(deadline) => {
                tokio::time::timeout_at(deadline.into(), self.load_model_for_inference(variant))
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))??
            }
            None => self.load_model_for_inference(variant).await?,
        };
        let registry = self.model_registry.clone();
        // Input preparation can temporarily retain both the caller-provided
        // payload and a job-owned copy. Restore the pending claim before that
        // temporary representation is moved into or replaced by the canonical
        // request. The final retained request is observed below, so shrinking
        // and growing representations follow one ordered transition.
        job.prepare_materialized_release(JobResourceObservation::default())?;
        let (residency_lease, mut request) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let request = build(registry, prepared_input)?;
                Ok((residency_lease, request))
            })
            .await?;
        if request.task_type != task_type || request.model_variant != Some(variant) {
            return Err(Error::InvalidInput(
                "prepared engine request does not match its admitted task/model".to_string(),
            ));
        }
        request.id = job.spec.request_id.clone();
        request.streaming = streaming;
        request.workload_class = effective_context.workload_class;
        request.priority = effective_context.priority;
        request.admission_ms = effective_context.admission_ms;
        request.deadline = effective_context.deadline;
        let retained_request_host_bytes =
            u64::try_from(retained_engine_request_input_bytes(&request)?).map_err(|_| {
                Error::Overloaded("prepared engine request retained input exceeds u64".into())
            })?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_host_bytes))?;
        Ok(AdmittedEngineRequest {
            request,
            job,
            residency_lease,
        })
    }

    pub(super) fn coordinator_job_for_request(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<(JobSpec, JobResourceObservation)> {
        let input_bytes = retained_engine_request_input_bytes(request)?;
        let mut spec = self.coordinator_job_for_input(
            request.id.clone(),
            coordinator_lane_for_request(request),
            RuntimeRequestContext {
                workload_class: request.workload_class,
                tenant_key: request.tenant_key,
                admission_ms: request.admission_ms,
                priority: request.priority,
                deadline: request.deadline,
            },
            input_bytes,
        );
        if request.task_type == TaskType::ASR {
            if let Some(variant) = request.model_variant {
                if let Some(resources) = asr_transient_resources(
                    self.backend_router.context().backend_kind,
                    variant,
                    input_bytes,
                    request.prepared_asr_input_duration_secs(),
                )? {
                    spec.resources = resources;
                }
            }
        }
        // Price the durable prompt/position/token journal and overlapping
        // suspension/restoration copies in the request's existing host lease.
        // This is separate from the generic request/text/workspace allowance.
        if request.task_type == TaskType::Chat
            && request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::Qwen38Chat)
        {
            if let Some(runtime) = request.managed_cache_runtime() {
                let bytes = runtime
                    .maximum_sequence_tokens()
                    .checked_mul(256)
                    .ok_or_else(|| {
                        Error::Overloaded("chat replay host reservation overflow".into())
                    })?;
                let mut journal = ResourceVector::zero();
                match self.backend_router.context().backend_kind {
                    BackendKind::Metal => journal.unified_bytes = ResourceAmount::Known(bytes),
                    BackendKind::Cpu | BackendKind::Cuda => {
                        journal.host_bytes = ResourceAmount::Known(bytes)
                    }
                }
                spec.resources = spec.resources.checked_add(journal)?;
            }
        }
        let audio_decode_required = task_decodes_audio(request.task_type)
            && !(request.task_type == TaskType::ASR
                && request.model_variant.is_some_and(|variant| {
                    matches!(
                        variant.family(),
                        ModelFamily::Qwen3Asr
                            | ModelFamily::WhisperAsr
                            | ModelFamily::VibeVoiceAsr
                            | ModelFamily::GraniteSpeechAsr
                            | ModelFamily::Lfm25Audio
                    )
                })
                && request.prepared_asr_audio_for_executor()?.is_some());
        if audio_decode_required
            || (request.task_type == TaskType::TTS && request.has_tts_reference_for_execution())
        {
            spec.resources = spec.resources.checked_add(audio_decode_resources(
                self.backend_router.context().backend_kind,
            ))?;
        }
        let (asr_encoder_host_bytes, asr_encoder_device_bytes) =
            request.prepared_asr_encoder_artifact_retained_resources()?;
        if asr_encoder_host_bytes > 0 || asr_encoder_device_bytes > 0 {
            spec.resources = spec.resources.checked_add(asr_encoder_retained_resources(
                self.backend_router.context().backend_kind,
                asr_encoder_host_bytes,
                asr_encoder_device_bytes,
            )?)?;
        }
        if request.task_type == TaskType::TTS
            && request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::Qwen3Tts)
        {
            let frames =
                u64::try_from(request.qwen_tts_generation_params().max_frames).map_err(|_| {
                    Error::Overloaded("Qwen3-TTS output frame bound exceeds u64".into())
                })?;
            let output_bytes = frames
                .checked_mul(1_920)
                .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
                .ok_or_else(|| Error::Overloaded("Qwen3-TTS output reservation overflow".into()))?;
            let mut output = ResourceVector::zero();
            match self.backend_router.context().backend_kind {
                BackendKind::Metal => output.unified_bytes = ResourceAmount::Known(output_bytes),
                BackendKind::Cpu | BackendKind::Cuda => {
                    output.host_bytes = ResourceAmount::Known(output_bytes)
                }
            }
            spec.resources = spec.resources.checked_add(output)?;
        }
        if request.task_type == TaskType::TTS
            && request.model_variant == Some(ModelVariant::FishAudioS2Pro)
        {
            let frames = request
                .params
                .max_tokens
                .clamp(1, ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES);
            let output_bytes = if request.collect_audio_samples {
                (frames as u64).checked_mul(2048 * 4 * 3).ok_or_else(|| {
                    Error::Overloaded("Fish S2 output reservation overflow".into())
                })?
            } else {
                request.streaming_audio_buffer_bytes(
                    crate::models::architectures::fish_s2::FISH_S2_AUDIO_CHUNK_FRAMES * 2048,
                )?
            };
            let mut output = ResourceVector::zero();
            match self.backend_router.context().backend_kind {
                BackendKind::Metal => output.unified_bytes = ResourceAmount::Known(output_bytes),
                BackendKind::Cpu | BackendKind::Cuda => {
                    output.host_bytes = ResourceAmount::Known(output_bytes)
                }
            }
            spec.resources = spec.resources.checked_add(output)?;
            // Current history and immutable rollback history coexist during a push.
            let history_bytes =
                crate::models::architectures::fish_s2::dac::FishS2DacConfig::current()
                    .streaming_history_bound_bytes()?
                    .checked_mul(2)
                    .ok_or_else(|| {
                        Error::Overloaded("Fish codec history reservation overflow".into())
                    })?;
            spec.resources = spec.resources.checked_add(asr_encoder_retained_resources(
                self.backend_router.context().backend_kind,
                0,
                history_bytes,
            )?)?;
        }
        if request.task_type == TaskType::Chat && !request.chat_config.media_inputs.is_empty() {
            if !request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::Qwen35Chat)
            {
                return Err(Error::InvalidInput(
                    "Only Qwen3.5 chat requests may include media inputs".to_string(),
                ));
            }
            let estimate = crate::models::architectures::qwen35::media_resource_estimate(
                &request.chat_config.media_inputs,
            )?;
            spec.resources = spec.resources.checked_add(media_preparation_resources(
                self.backend_router.context().backend_kind,
                estimate,
            )?)?;
        }
        let mut observation = host_input_observation(input_bytes)?;
        if request.task_type == TaskType::TTS
            && request.model_variant == Some(ModelVariant::FishAudioS2Pro)
        {
            if let Some(artifact) = request.prepared_fish_s2_tts_artifact_for_executor()? {
                add_fish_s2_artifact_to_admission(
                    self.backend_router.context().backend_kind,
                    &mut spec,
                    &mut observation,
                    artifact.retained_bytes()?,
                )?;
            }
        }
        Ok((spec, observation))
    }

    /// Load or pin a model under the admitted request's absolute deadline.
    /// The detached lifecycle transaction remains cancellation-safe, while the
    /// caller never waits beyond its end-to-end budget.
    pub(crate) async fn load_model_for_job(
        &self,
        job: &JobLease,
        variant: ModelVariant,
    ) -> Result<ModelResidencyLease> {
        match job.spec.deadline {
            Some(deadline) => {
                tokio::time::timeout_at(deadline.into(), self.load_model_for_inference(variant))
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
            }
            None => self.load_model_for_inference(variant).await,
        }
    }

    /// Load and pin the exact model generation selected for a non-engine
    /// capability stage, then resolve its immutable loaded-adapter contract.
    /// Returning both values makes it impossible for direct runners to execute
    /// against catalog metadata or a different load generation.
    pub(crate) async fn load_capability_for_job(
        &self,
        job: &JobLease,
        variant: ModelVariant,
        capability: CapabilityKind,
        streaming_required: bool,
        expected_target: ExecutionTargetKind,
    ) -> Result<(ModelResidencyLease, LoadedExecutionContract)> {
        let lease = self.load_model_for_job(job, variant).await?;
        let bundle = self.model_lifecycle.try_get_ready_bundle(variant);
        let contract = loaded_contract_for_residency(
            &lease,
            bundle.as_deref(),
            capability,
            streaming_required,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(expected_target),
        )?;
        Ok((lease, contract))
    }

    pub(crate) async fn load_capability_with_state_for_job(
        &self,
        job: &JobLease,
        variant: ModelVariant,
        capability: CapabilityKind,
        streaming_required: bool,
        expected_target: ExecutionTargetKind,
        options: CapabilityLoadOptions,
    ) -> Result<(
        ModelResidencyLease,
        LoadedExecutionContract,
        LoadedCapabilityBinding,
    )> {
        let lease = self.load_model_for_job(job, variant).await?;
        let bundle = self.model_lifecycle.try_get_ready_bundle(variant);
        let (contract, binding) = loaded_binding_for_residency(
            &lease,
            bundle.as_deref(),
            capability,
            streaming_required,
            options.asr_long_form,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(expected_target),
        )?;
        Ok((lease, contract, binding))
    }

    /// Bound the pre-session Engine admission transaction by the request's
    /// absolute deadline. Cancelling this future before the core write lock is
    /// acquired cannot create a scheduler session; once that lock is acquired,
    /// Engine admission contains no further await point and returns its exact
    /// session atomically.
    pub(super) async fn await_engine_admission_for_job<T, F>(&self, job: &JobLease, admission: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match job.spec.deadline {
            Some(deadline) => {
                if deadline <= Instant::now() {
                    return Err(Error::Timeout(job.spec.request_id.clone()));
                }
                tokio::time::timeout_at(deadline.into(), admission)
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
            }
            None => admission.await,
        }
    }

    pub(crate) async fn run_request(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        self.observe_broker_request(&request)?;
        let (spec, observation) = self.coordinator_job_for_request(&request)?;
        let job = self.coordinator.admit_observed(spec, observation).await?;
        let _residency_lease = match request.model_variant {
            Some(variant) => Some(self.load_model_for_job(&job, variant).await?),
            None => None,
        };
        self.run_request_after_admission(request, job, _residency_lease)
            .await
    }

    pub(crate) async fn run_admitted_request(
        &self,
        admitted: AdmittedEngineRequest,
    ) -> Result<EngineOutput> {
        let AdmittedEngineRequest {
            request,
            job,
            residency_lease,
        } = admitted;
        self.observe_broker_request(&request)?;
        self.run_request_after_admission(request, job, Some(residency_lease))
            .await
    }

    pub(super) async fn run_request_after_admission(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<ModelResidencyLease>,
    ) -> Result<EngineOutput> {
        let (request, no_speech) = self.asr_no_speech_gate(request, &job).await?;
        if let Some(gate) = no_speech {
            let output = gate.engine_output(&request.id);
            self.record_engine_output_observation(&request, &output, false);
            return Ok(output);
        }
        let (mut request, job) = self
            .prepare_request_for_binding(request, job, residency_lease.as_ref())
            .await?;
        let loaded_bundle = residency_lease
            .as_ref()
            .and_then(|lease| self.model_lifecycle.try_get_ready_bundle(lease.variant()));
        bind_request_to_residency(
            &mut request,
            residency_lease.as_ref(),
            loaded_bundle.as_deref(),
            false,
        )?;
        if job.spec.request_id != request.id || job.spec.deadline != request.deadline {
            return Err(Error::InvalidInput(
                "engine request does not match its coordinator admission".to_string(),
            ));
        }
        request.runtime_admitted_at = Some(job.admitted_at());
        let observation_request = request.clone();
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            workload_class = ?request.workload_class,
            streaming = false
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let (waiter_registration_id, completion_rx) = self.register_waiter(&request_id).await?;
        let mut waiter_guard = WaiterRegistrationGuard::new(
            request_id.clone(),
            waiter_registration_id,
            self.completion_waiters.clone(),
        );

        let session = match self
            .await_engine_admission_for_job(
                &job,
                self.core_engine.add_request_with_session(request),
            )
            .await
        {
            Ok(session) => session,
            Err(err) => {
                self.remove_waiter(&request_id, waiter_registration_id)
                    .await;
                waiter_guard.disarm();
                self.record_engine_error_observation(&observation_request, false, err.to_string());
                return Err(err);
            }
        };
        // Establish exact-session cancellation ownership before the next await.
        // A caller may drop this future while waiter binding is contended; the
        // admitted request and its coordinator lease must still be reclaimed.
        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            waiter_registration_id,
            self.telemetry.clone(),
            job,
            residency_lease,
        );
        // If Engine admission became ready in the same poll as the deadline,
        // the exact session now exists. Establish cleanup ownership first, then
        // fail the caller without leaving that session or its waiter behind.
        if observation_request
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            let err = Error::Timeout(request_id.clone());
            guard.defer_cleanup();
            self.record_engine_error_observation(&observation_request, false, err.to_string());
            return Err(err);
        }
        self.bind_waiter(&request_id, waiter_registration_id, guard.session.epoch)
            .await?;
        waiter_guard.disarm();
        self.telemetry.record_request_queued(&request_id).await;
        self.step_driver_wakeup.notify_one();
        let completion = self
            .await_completion(&request_id, completion_rx, observation_request.deadline)
            .await;
        match completion.as_ref() {
            Ok(output) => {
                self.record_engine_output_observation(&observation_request, output, false)
            }
            Err(err) => {
                self.record_engine_error_observation(&observation_request, false, err.to_string())
            }
        }
        let output = completion?;
        guard.disarm();
        Ok(output)
    }

    pub(super) fn defer_streaming_failure(
        &self,
        guard: &mut PendingRequestGuard,
        observation_request: &EngineCoreRequest,
        err: Error,
    ) -> Error {
        // The exact abort may need the same core write lock held by a running
        // native step. Hand cleanup off before returning to the transport;
        // DeferredRequestOwnership retains admission and residency until that
        // abort proves the exact session is no longer executing.
        guard.defer_cleanup();
        self.record_engine_error_observation(observation_request, true, err.to_string());
        err
    }

    pub(super) async fn deliver_streaming_chunk_before_deadline<F, Fut>(
        &self,
        on_chunk: &mut F,
        chunk: StreamingOutput,
        deadline: Option<Instant>,
        stream_request_id: &str,
        observation_request: &EngineCoreRequest,
        guard: &mut PendingRequestGuard,
    ) -> Result<()>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        // Calling `on_chunk` constructs the future and may itself perform
        // synchronous transport work. Reject an expired request before that
        // callback can observe or emit another chunk.
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let err = Error::Timeout(stream_request_id.to_string());
            return Err(self.defer_streaming_failure(guard, observation_request, err));
        }
        let delivery = match deadline {
            Some(deadline) => {
                match tokio::time::timeout_at(deadline.into(), on_chunk(chunk)).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout(stream_request_id.to_string())),
                }
            }
            None => on_chunk(chunk).await,
        };

        delivery.map_err(|err| self.defer_streaming_failure(guard, observation_request, err))?;
        // `timeout_at` and the callback can become ready in the same poll. Do
        // not let select/poll ordering turn an already-expired absolute
        // request deadline into a successful transport delivery.
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let err = Error::Timeout(stream_request_id.to_string());
            return Err(self.defer_streaming_failure(guard, observation_request, err));
        }
        Ok(())
    }

    pub(crate) async fn run_streaming_request<F, Fut>(
        &self,
        request: EngineCoreRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_streaming_request_with_broker_streaming(request, on_chunk, true)
            .await
    }

    pub(crate) async fn run_admitted_streaming_request<F, Fut>(
        &self,
        admitted: AdmittedEngineRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_admitted_streaming_request_with_broker_streaming(admitted, on_chunk, true)
            .await
    }

    pub(crate) async fn run_admitted_transport_streaming_request<F, Fut>(
        &self,
        admitted: AdmittedEngineRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_admitted_streaming_request_with_broker_streaming(admitted, on_chunk, false)
            .await
    }

    pub(super) async fn run_admitted_streaming_request_with_broker_streaming<F, Fut>(
        &self,
        admitted: AdmittedEngineRequest,
        on_chunk: F,
        broker_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let AdmittedEngineRequest {
            request,
            job,
            residency_lease,
        } = admitted;
        if broker_streaming_required {
            self.observe_broker_request(&request)?;
        } else {
            self.observe_broker_request_with_transport_streaming(&request)?;
        }
        self.run_streaming_request_after_admission(
            request,
            on_chunk,
            job,
            Some(residency_lease),
            broker_streaming_required,
        )
        .await
    }

    pub(super) async fn run_streaming_request_with_broker_streaming<F, Fut>(
        &self,
        mut request: EngineCoreRequest,
        on_chunk: F,
        broker_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        request.streaming = true;
        if request.model_variant == Some(ModelVariant::FishAudioS2Pro) {
            request.mark_audio_streaming_only();
        }
        if request.workload_class == WorkloadClass::Online {
            request.workload_class = WorkloadClass::Streaming;
        }
        if broker_streaming_required {
            self.observe_broker_request(&request)?;
        } else {
            self.observe_broker_request_with_transport_streaming(&request)?;
        }
        let (spec, observation) = self.coordinator_job_for_request(&request)?;
        let job = self.coordinator.admit_observed(spec, observation).await?;
        let _residency_lease = match request.model_variant {
            Some(variant) => Some(self.load_model_for_job(&job, variant).await?),
            None => None,
        };
        self.run_streaming_request_after_admission(
            request,
            on_chunk,
            job,
            _residency_lease,
            broker_streaming_required,
        )
        .await
    }

    pub(super) async fn run_streaming_request_after_admission<F, Fut>(
        &self,
        request: EngineCoreRequest,
        mut on_chunk: F,
        job: JobLease,
        residency_lease: Option<ModelResidencyLease>,
        model_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let (request, no_speech) = self.asr_no_speech_gate(request, &job).await?;
        if let Some(gate) = no_speech {
            // Close the stream as the engine does: a terminal marker, no text.
            on_chunk(AsrNoSpeechGate::final_stream_chunk(&request.id)).await?;
            let output = gate.engine_output(&request.id);
            self.record_engine_output_observation(&request, &output, true);
            return Ok(output);
        }
        let (mut request, job) = self
            .prepare_request_for_binding(request, job, residency_lease.as_ref())
            .await?;
        let loaded_bundle = residency_lease
            .as_ref()
            .and_then(|lease| self.model_lifecycle.try_get_ready_bundle(lease.variant()));
        bind_request_to_residency(
            &mut request,
            residency_lease.as_ref(),
            loaded_bundle.as_deref(),
            model_streaming_required,
        )?;
        if job.spec.request_id != request.id || job.spec.deadline != request.deadline {
            return Err(Error::InvalidInput(
                "streaming engine request does not match its coordinator admission".to_string(),
            ));
        }
        request.runtime_admitted_at = Some(job.admitted_at());
        if request.model_variant == Some(ModelVariant::FishAudioS2Pro)
            && (request.collect_audio_samples
                || request.audio_stream_engine_queue_capacity.is_none())
        {
            return Err(Error::InvalidInput(
                "Fish streaming output capacity must be frozen before admission".into(),
            ));
        }
        let observation_request = request.clone();
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            workload_class = ?request.workload_class,
            streaming = true
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let (waiter_registration_id, mut completion_rx) = self.register_waiter(&request_id).await?;
        let mut waiter_guard = WaiterRegistrationGuard::new(
            request_id.clone(),
            waiter_registration_id,
            self.completion_waiters.clone(),
        );
        let (session, mut stream_rx) = match self
            .await_engine_admission_for_job(
                &job,
                self.core_engine.generate_streaming_with_session(request),
            )
            .await
        {
            Ok(v) => v,
            Err(err) => {
                self.remove_waiter(&request_id, waiter_registration_id)
                    .await;
                waiter_guard.disarm();
                self.record_engine_error_observation(&observation_request, true, err.to_string());
                return Err(err);
            }
        };
        let stream_request_id = session.request_id.clone();
        debug_assert_eq!(stream_request_id, request_id);
        // Establish exact-session cancellation ownership before the next await.
        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            waiter_registration_id,
            self.telemetry.clone(),
            job,
            residency_lease,
        );
        // As above, a session returned on the deadline boundary must be
        // cancelled through exact-session ownership rather than treated as a
        // pre-admission timeout.
        if observation_request
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            let err = Error::Timeout(stream_request_id.clone());
            return Err(self.defer_streaming_failure(&mut guard, &observation_request, err));
        }
        self.bind_waiter(
            &stream_request_id,
            waiter_registration_id,
            guard.session.epoch,
        )
        .await?;
        waiter_guard.disarm();
        self.telemetry.record_request_queued(&request_id).await;
        self.step_driver_wakeup.notify_one();
        let mut completion_result: Option<EngineOutput> = None;
        let deadline = observation_request.deadline;
        let deadline_wait = async move {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline_wait);
        let mut stream_order = StreamOutputOrder::new(observation_request.stream_policy);

        loop {
            tokio::select! {
                maybe_chunk = stream_rx.recv() => {
                    let Some(chunk) = maybe_chunk else {
                        break;
                    };

                    if let Err(err) = stream_order.observe(&stream_request_id, &chunk) {
                        return Err(self.defer_streaming_failure(
                            &mut guard,
                            &observation_request,
                            err,
                        ));
                    }

                    self
                        .deliver_streaming_chunk_before_deadline(
                            &mut on_chunk,
                            chunk,
                            deadline,
                            &stream_request_id,
                            &observation_request,
                            &mut guard,
                        )
                        .await?
                }
                completion = &mut completion_rx, if completion_result.is_none() => {
                    let completion = match completion {
                        Ok(completion) => completion,
                        Err(_) => {
                            let err = Error::InferenceError(format!(
                                "Request {stream_request_id} completion channel closed unexpectedly"
                            ));
                            return Err(self.defer_streaming_failure(
                                &mut guard,
                                &observation_request,
                                err,
                            ));
                        }
                    };

                    match completion {
                        Ok(output) => {
                            completion_result = Some(output);
                        }
                        Err(err) => {
                            // If engine worker panics, fail fast so streaming callers
                            // don't hang waiting for a chunk channel that may never close.
                            return Err(self.defer_streaming_failure(
                                &mut guard,
                                &observation_request,
                                err,
                            ));
                        }
                    }
                }
                _ = &mut deadline_wait => {
                    let err = Error::Timeout(stream_request_id.clone());
                    return Err(self.defer_streaming_failure(
                        &mut guard,
                        &observation_request,
                        err,
                    ));
                }
            }
        }

        let output = if let Some(output) = completion_result {
            output
        } else {
            match self
                .await_completion(
                    &stream_request_id,
                    completion_rx,
                    observation_request.deadline,
                )
                .await
            {
                Ok(output) => output,
                Err(err) => {
                    return Err(self.defer_streaming_failure(
                        &mut guard,
                        &observation_request,
                        err,
                    ));
                }
            }
        };
        // Completion and stream closure can be ready in the same select poll
        // as the deadline. Preserve an absolute end-to-end deadline regardless
        // of which ready branch Tokio chooses first.
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let err = Error::Timeout(stream_request_id.clone());
            return Err(self.defer_streaming_failure(&mut guard, &observation_request, err));
        }
        if let Err(err) = stream_order.require_final(&stream_request_id) {
            return Err(self.defer_streaming_failure(&mut guard, &observation_request, err));
        }
        self.record_engine_output_observation(&observation_request, &output, true);
        guard.disarm();
        // Allow pending tasks to progress before returning to upper layers.
        yield_now().await;
        Ok(output)
    }
}
