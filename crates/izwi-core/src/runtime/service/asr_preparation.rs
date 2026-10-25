//! ASR request preparation: decoding, encoder batching and the exact execution
//! shape each ASR family binds before admission.

use super::*;

pub(super) fn asr_encoder_retained_resources(
    backend: BackendKind,
    host_bytes: u64,
    accelerator_bytes: u64,
) -> Result<ResourceVector> {
    retained_artifact_resources(backend, host_bytes, accelerator_bytes)
}

/// Parakeet's decode budget is the input's emittable-label bound, not the
/// portable token context: its recurrent predictor keeps no per-token state,
/// so the context limits nothing, while long dense audio exceeds 4096 labels.
/// A caller's `max_tokens` is kept below the bound by the sequence clamp.
pub(super) fn install_parakeet_decode_budget(
    request: &mut EngineCoreRequest,
    max_emittable_tokens: usize,
) -> Result<()> {
    if request.asr_auto_max_tokens {
        request.params.max_tokens = max_emittable_tokens;
    }
    let sequence_tokens = max_emittable_tokens
        .checked_add(1)
        .ok_or_else(|| Error::Overloaded("Parakeet decode budget overflowed".into()))?;
    request.install_prepared_sequence_input_tokens(1, sequence_tokens)
}

impl RuntimeService {
    pub(super) async fn prepare_qwen3_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::Qwen3Asr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Qwen3 ASR variant");
        let model = self
            .model_registry
            .get_asr_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not loaded")))?;
        let residency_lease = residency_lease.ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 ASR preparation requires authoritative model residency".to_string(),
            )
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency_lease.variant());
        let encoder_contract = loaded_contract_for_residency(
            residency_lease,
            loaded_bundle.as_deref(),
            CapabilityKind::Asr,
            false,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(ExecutionTargetKind::TokenEngine),
        )?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Qwen3 ASR model {variant} has no load-sealed effective context"
                ))
            })?;
        let model_for_shape = model.clone();
        let (prepared, geometry) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let capabilities = crate::engine::AsrRouteCapabilities::for_native_model(
                    &model_for_shape,
                    variant,
                    &request,
                    Some(context_limit),
                )?;
                let plan = crate::engine::asr_route_plan(
                    &samples,
                    sample_rate,
                    &capabilities,
                    request.streaming,
                );
                let long_form = plan.requires_chunk_path();
                let geometry = (!long_form)
                    .then(|| {
                        model_for_shape.audio_preparation_row_geometry(samples.len(), sample_rate)
                    })
                    .transpose()?;
                let input_tokens = (!long_form)
                    .then(|| {
                        model_for_shape.incremental_prompt_token_count(
                            &samples,
                            sample_rate,
                            request.asr_language_for_execution(),
                            request.asr_prompt_for_execution(),
                        )
                    })
                    .transpose()?;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                if long_form {
                    request.install_prepared_asr_long_form_atomic(plan)?;
                } else {
                    request.install_prepared_sequence_input_tokens(
                        input_tokens.expect("normal Qwen3 ASR shape"),
                        context_limit,
                    )?;
                }
                Ok((request, geometry))
            })
            .await?;
        let retained_host_bytes = u64::try_from(retained_engine_request_input_bytes(&prepared)?)
            .map_err(|_| Error::Overloaded("Qwen3 ASR retained host input exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_host_bytes))?;
        let initial_bridge = self.coordinator.bridge_preparation_admission(job)?;

        if prepared.uses_asr_long_form_atomic() {
            let (execution, observation) = self.coordinator_job_for_request(&prepared)?;
            let execution_job = match self
                .coordinator
                .admit_observed_from_preparation(initial_bridge, execution, observation)
                .await
            {
                Ok(job) => job,
                Err(failure) => {
                    drop(prepared);
                    let error = failure.error;
                    drop(failure.bridge);
                    return Err(error);
                }
            };
            return Ok((prepared, execution_job));
        }

        let geometry = geometry.ok_or_else(|| {
            Error::InferenceError("normal Qwen3 ASR route lost encoder geometry".to_string())
        })?;
        let encoder_resources = asr_encoder_retained_resources(
            self.backend_router.context().backend_kind,
            retained_host_bytes,
            geometry.retained_artifact_bytes,
        )?;
        let encoder_spec = JobSpec {
            request_id: prepared.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: prepared.priority,
            workload_class: prepared.workload_class,
            deadline: prepared.deadline,
            resources: encoder_resources,
        };
        let encoder_job = match self
            .coordinator
            .admit_observed_from_preparation(
                initial_bridge,
                encoder_spec,
                JobResourceObservation::host(retained_host_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        let (samples, sample_rate) = prepared
            .prepared_asr_audio_for_executor()?
            .ok_or_else(|| Error::InferenceError("Qwen3 ASR decoded audio was lost".into()))?;
        let outcome = self
            .qwen_asr_encoder_batcher
            .submit(QwenAsrEncoderPending {
                job: encoder_job,
                contract: encoder_contract,
                model,
                samples,
                sample_rate,
                geometry,
                retained_host_bytes,
                cancellation: PreparationCancellation::default(),
                response: None,
            })
            .await?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact.value, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(prepared.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(prepared.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let artifact_audio_tokens = match artifact.audio_tokens() {
            Ok(tokens) => tokens,
            Err(error) => {
                drop(artifact);
                drop(bridge);
                return Err(error);
            }
        };
        if artifact_audio_tokens != geometry.audio_tokens {
            drop(artifact);
            drop(bridge);
            return Err(Error::InferenceError(
                "Qwen3 ASR encoder artifact token geometry drifted after admission".to_string(),
            ));
        }
        let accelerator_bytes = match artifact.resident_tensor_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(artifact);
                drop(bridge);
                return Err(error);
            }
        };
        let artifact = Arc::new(artifact);
        let mut prepared = prepared;
        if let Err(error) =
            prepared.install_prepared_asr_encoder_artifact(variant, artifact.clone())
        {
            drop(artifact);
            drop(prepared);
            drop(bridge);
            return Err(error);
        }
        let execution = match self.coordinator_job_for_request(&prepared) {
            Ok((execution, _)) => execution,
            Err(error) => {
                drop(artifact);
                drop(prepared);
                drop(bridge);
                return Err(error);
            }
        };
        let observation = JobResourceObservation {
            host_bytes: retained_host_bytes,
            accelerator_bytes,
        };
        let execution_job = match self
            .coordinator
            .admit_observed_from_preparation(bridge, execution, observation)
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                drop(artifact);
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        Ok((prepared, execution_job))
    }

    pub(super) async fn prepare_whisper_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::WhisperAsr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Whisper variant");
        let model = self
            .model_registry
            .get_asr_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not loaded")))?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Whisper preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let encoder_contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Asr,
            false,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(ExecutionTargetKind::TokenEngine),
        )?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!("Whisper model {variant} has no effective context"))
            })?;
        let model_for_shape = model.clone();
        let (prepared, geometry) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let capabilities = crate::engine::AsrRouteCapabilities::for_native_model(
                    &model_for_shape,
                    variant,
                    &request,
                    Some(context_limit),
                )?;
                let plan = crate::engine::asr_route_plan(
                    &samples,
                    sample_rate,
                    &capabilities,
                    request.streaming,
                );
                let long_form = plan.requires_chunk_path();
                let geometry = (!long_form)
                    .then(|| {
                        model_for_shape.whisper_window_preparation_geometry(&samples, sample_rate)
                    })
                    .transpose()?;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                if long_form {
                    request.install_prepared_asr_long_form_atomic(plan)?;
                }
                Ok((request, geometry))
            })
            .await?;
        let retained_host_bytes = u64::try_from(retained_engine_request_input_bytes(&prepared)?)
            .map_err(|_| Error::Overloaded("Whisper retained input exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_host_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        if prepared.uses_asr_long_form_atomic() {
            let (spec, observation) = match self.coordinator_job_for_request(&prepared) {
                Ok(value) => value,
                Err(error) => {
                    drop(prepared);
                    drop(bridge);
                    return Err(error);
                }
            };
            return match self
                .coordinator
                .admit_observed_from_preparation(bridge, spec, observation)
                .await
            {
                Ok(job) => Ok((prepared, job)),
                Err(failure) => {
                    drop(prepared);
                    let error = failure.error;
                    drop(failure.bridge);
                    Err(error)
                }
            };
        }
        let geometry = match geometry {
            Some(geometry) => geometry,
            None => {
                drop(prepared);
                drop(bridge);
                return Err(Error::InferenceError("Whisper geometry was lost".into()));
            }
        };
        let resources = match asr_encoder_retained_resources(
            self.backend_router.context().backend_kind,
            retained_host_bytes,
            geometry.retained_artifact_bytes,
        ) {
            Ok(resources) => resources,
            Err(error) => {
                drop(prepared);
                drop(bridge);
                return Err(error);
            }
        };
        let prep_spec = JobSpec {
            request_id: prepared.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: prepared.priority,
            workload_class: prepared.workload_class,
            deadline: prepared.deadline,
            resources,
        };
        let prep_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                prep_spec,
                JobResourceObservation::host(retained_host_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        let (samples, sample_rate) = prepared
            .prepared_asr_audio_for_executor()?
            .ok_or_else(|| Error::InferenceError("Whisper decoded audio was lost".into()))?;
        let outcome = self
            .whisper_encoder_batcher
            .submit(WhisperEncoderPending {
                job: prep_job,
                contract: encoder_contract,
                model: model.clone(),
                samples,
                sample_rate,
                geometry,
                retained_host_bytes,
                cancellation: PreparationCancellation::default(),
                response: None,
            })
            .await?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact.value, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(prepared.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(prepared.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let accelerator_bytes = match artifact.resident_tensor_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(artifact);
                drop(prepared);
                drop(bridge);
                return Err(error);
            }
        };
        if artifact.cross_memory_tokens() != geometry.cross_memory_tokens
            || accelerator_bytes != geometry.retained_artifact_bytes
        {
            drop(artifact);
            drop(prepared);
            drop(bridge);
            return Err(Error::InferenceError(
                "Whisper encoder artifact geometry drifted after admission".into(),
            ));
        }
        let input_tokens = match model.whisper_incremental_prompt_token_count(
            &artifact,
            prepared.asr_language_for_execution(),
            prepared.asr_prompt_for_execution(),
        ) {
            Ok(tokens) => tokens,
            Err(error) => {
                drop(artifact);
                drop(prepared);
                drop(bridge);
                return Err(error);
            }
        };
        let mut prepared = prepared;
        if let Err(error) =
            prepared.install_prepared_sequence_input_tokens(input_tokens, context_limit)
        {
            drop(artifact);
            drop(prepared);
            drop(bridge);
            return Err(error);
        }
        if let Err(error) = prepared.install_prepared_whisper_window(variant, Arc::new(artifact)) {
            drop(prepared);
            drop(bridge);
            return Err(error);
        }
        let (execution, _) = match self.coordinator_job_for_request(&prepared) {
            Ok(value) => value,
            Err(error) => {
                drop(prepared);
                drop(bridge);
                return Err(error);
            }
        };
        let observation = JobResourceObservation {
            host_bytes: retained_host_bytes,
            accelerator_bytes,
        };
        match self
            .coordinator
            .admit_observed_from_preparation(bridge, execution, observation)
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                Err(error)
            }
        }
    }

    pub(super) async fn prepare_vibevoice_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::VibeVoiceAsr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated VibeVoice variant");
        let model = self
            .model_registry
            .get_asr_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not loaded")))?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("VibeVoice preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let preparation_contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Asr,
            false,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(ExecutionTargetKind::TokenEngine),
        )?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "VibeVoice ASR model {variant} has no effective context"
                ))
            })?;
        let model_for_shape = model.clone();
        let (prepared, geometry) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let decision = model_for_shape.vibevoice_retained_preparation_decision(
                    samples.len(),
                    sample_rate,
                    request.asr_language_for_execution(),
                    request.asr_prompt_for_execution(),
                )?;
                let audio_seconds = samples.len() as f32 / sample_rate.max(1) as f32;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                match decision {
                    VibeVoiceAsrPreparationDecision::Retained(geometry) => {
                        // Admission prices `max_tokens`: narrow an omitted
                        // budget to this audio's duration first.
                        request.params.max_tokens =
                            crate::runtime::asr::vibevoice_asr_planned_max_tokens(
                                &request,
                                audio_seconds,
                            );
                        request.install_prepared_sequence_input_tokens(
                            geometry.prompt_tokens,
                            context_limit,
                        )?;
                        Ok((request, Some(geometry)))
                    }
                    VibeVoiceAsrPreparationDecision::LegacyInvocation => {
                        let capabilities = crate::engine::AsrRouteCapabilities::for_native_model(
                            &model_for_shape,
                            variant,
                            &request,
                            Some(context_limit),
                        )?;
                        let (samples, sample_rate) =
                            request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                                Error::InferenceError(
                                    "VibeVoice ASR preparation lost its decoded audio".into(),
                                )
                            })?;
                        let plan = crate::engine::asr_route_plan(
                            &samples,
                            sample_rate,
                            &capabilities,
                            request.streaming,
                        );
                        request.install_prepared_asr_long_form_atomic(plan)?;
                        Ok((request, None))
                    }
                }
            })
            .await?;
        let retained_request_host_bytes =
            u64::try_from(retained_engine_request_input_bytes(&prepared)?).map_err(|_| {
                Error::Overloaded("VibeVoice ASR retained input exceeds u64".into())
            })?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_host_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        if prepared.uses_asr_long_form_atomic() {
            let (spec, observation) = match self.coordinator_job_for_request(&prepared) {
                Ok(value) => value,
                Err(error) => {
                    drop(prepared);
                    drop(bridge);
                    return Err(error);
                }
            };
            return match self
                .coordinator
                .admit_observed_from_preparation(bridge, spec, observation)
                .await
            {
                Ok(job) => Ok((prepared, job)),
                Err(failure) => {
                    drop(prepared);
                    let error = failure.error;
                    drop(failure.bridge);
                    Err(error)
                }
            };
        }
        let geometry = geometry.ok_or_else(|| {
            Error::InferenceError("VibeVoice normal route lost preparation geometry".into())
        })?;
        let retained_host_bytes = retained_request_host_bytes
            .checked_add(geometry.retained_host_bytes)
            .ok_or_else(|| Error::Overloaded("VibeVoice retained host bytes overflow".into()))?;
        let resources = asr_encoder_retained_resources(
            self.backend_router.context().backend_kind,
            retained_host_bytes,
            geometry.retained_device_bytes,
        )?;
        let preparation_spec = JobSpec {
            request_id: prepared.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: prepared.priority,
            workload_class: prepared.workload_class,
            deadline: prepared.deadline,
            resources,
        };
        let preparation_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                preparation_spec,
                JobResourceObservation::host(retained_request_host_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        let (samples, sample_rate) = prepared
            .prepared_asr_audio_for_executor()?
            .ok_or_else(|| Error::InferenceError("VibeVoice decoded audio was lost".into()))?;
        let outcome = self
            .vibevoice_encoder_batcher
            .submit(VibeVoiceEncoderPending {
                job: preparation_job,
                contract: preparation_contract,
                model: model.clone(),
                samples,
                sample_rate,
                language: prepared.asr_language_for_execution().map(str::to_owned),
                prompt: prepared.asr_prompt_for_execution().map(str::to_owned),
                geometry,
                retained_request_host_bytes,
                cancellation: PreparationCancellation::default(),
            })
            .await?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact.value, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(prepared.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(prepared.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        if artifact.geometry() != geometry
            || artifact.resident_host_bytes() != geometry.retained_host_bytes
            || artifact.resident_tensor_bytes() != geometry.retained_device_bytes
        {
            drop(artifact);
            drop(prepared);
            drop(bridge);
            return Err(Error::InferenceError(
                "VibeVoice ASR preparation artifact drifted from admitted geometry".into(),
            ));
        }
        let mut prepared = prepared;
        if let Err(error) =
            prepared.install_prepared_vibevoice_artifact(variant, Arc::new(artifact))
        {
            drop(prepared);
            drop(bridge);
            return Err(error);
        }
        let (execution, _) = match self.coordinator_job_for_request(&prepared) {
            Ok(value) => value,
            Err(error) => {
                drop(prepared);
                drop(bridge);
                return Err(error);
            }
        };
        let observation = JobResourceObservation {
            host_bytes: retained_host_bytes,
            accelerator_bytes: geometry.retained_device_bytes,
        };
        match self
            .coordinator
            .admit_observed_from_preparation(bridge, execution, observation)
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                Err(error)
            }
        }
    }

    pub(super) async fn prepare_granite_speech_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::GraniteSpeechAsr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated Granite Speech variant");
        let model = self
            .model_registry
            .get_asr_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not loaded")))?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Granite Speech preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let preparation_contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Asr,
            false,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(ExecutionTargetKind::TokenEngine),
        )?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Granite Speech model {variant} has no effective context"
                ))
            })?;
        let model_for_shape = model.clone();
        let (prepared, geometry) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let capabilities = crate::engine::AsrRouteCapabilities::for_native_model(
                    &model_for_shape,
                    variant,
                    &request,
                    Some(context_limit),
                )?;
                let plan = crate::engine::asr_route_plan(
                    &samples,
                    sample_rate,
                    &capabilities,
                    request.streaming,
                );
                let long_form = plan.requires_chunk_path();
                // Granite's retained decoder is not quality-certified: the
                // real model can collapse to tokenizer id 0 for every output
                // step while its invocation-scoped physical route produces
                // the expected transcript. Keep retained preparation behind
                // the model's truthful capability declaration so production
                // requests use the correctness-proven atomic route.
                // A native-timestamp request needs the atomic route, which
                // selects the timestamp task; retained preparation is ASR-only.
                let retained = !long_form
                    && !request.asr_native_word_timestamps
                    && model_for_shape.supports_resumable_prefill();
                let geometry = retained
                    .then(|| {
                        model_for_shape.granite_speech_retained_preparation_geometry(
                            &samples,
                            sample_rate,
                            request.asr_language_for_execution(),
                            request.asr_prompt_for_execution(),
                        )
                    })
                    .transpose()?;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                if let Some(geometry) = geometry {
                    request.install_prepared_sequence_input_tokens(
                        geometry.prompt_tokens,
                        context_limit,
                    )?;
                    Ok((request, Some(geometry)))
                } else {
                    request.install_prepared_asr_long_form_atomic(plan)?;
                    Ok((request, None))
                }
            })
            .await?;
        let retained_host_bytes = u64::try_from(retained_engine_request_input_bytes(&prepared)?)
            .map_err(|_| Error::Overloaded("Granite Speech retained input exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_host_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        if prepared.uses_asr_long_form_atomic() {
            let (spec, observation) = self.coordinator_job_for_request(&prepared)?;
            return match self
                .coordinator
                .admit_observed_from_preparation(bridge, spec, observation)
                .await
            {
                Ok(job) => Ok((prepared, job)),
                Err(failure) => {
                    drop(prepared);
                    let error = failure.error;
                    drop(failure.bridge);
                    Err(error)
                }
            };
        }
        let geometry = geometry.ok_or_else(|| {
            Error::InferenceError("Granite Speech normal route lost preparation geometry".into())
        })?;
        let artifact_host_bytes = u64::try_from(
            prepared.asr_language_for_execution().map_or(0, str::len),
        )
        .map_err(|_| Error::Overloaded("Granite Speech language bytes exceed u64".into()))?;
        let total_retained_host_bytes = retained_host_bytes
            .checked_add(artifact_host_bytes)
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech retained host bytes overflow".into())
            })?;
        let resources = asr_encoder_retained_resources(
            self.backend_router.context().backend_kind,
            total_retained_host_bytes,
            geometry.retained_device_bytes,
        )?;
        let prep_spec = JobSpec {
            request_id: prepared.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: prepared.priority,
            workload_class: prepared.workload_class,
            deadline: prepared.deadline,
            resources,
        };
        let prep_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                prep_spec,
                JobResourceObservation::host(retained_host_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        let (samples, sample_rate) = prepared
            .prepared_asr_audio_for_executor()?
            .ok_or_else(|| Error::InferenceError("Granite Speech decoded audio was lost".into()))?;
        let outcome = self
            .granite_speech_encoder_batcher
            .submit(GraniteSpeechEncoderPending {
                job: prep_job,
                contract: preparation_contract,
                model: model.clone(),
                samples,
                sample_rate,
                language: prepared.asr_language_for_execution().map(str::to_owned),
                prompt: prepared.asr_prompt_for_execution().map(str::to_owned),
                geometry,
                retained_host_bytes: total_retained_host_bytes,
                artifact_host_bytes,
                cancellation: PreparationCancellation::default(),
                response: None,
            })
            .await?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact.value, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(prepared.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(prepared.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        if artifact.prompt_tokens() != geometry.prompt_tokens
            || artifact.audio_tokens() != geometry.audio_tokens
            || artifact.resident_host_bytes() != artifact_host_bytes
            || artifact.resident_tensor_bytes()? != geometry.retained_device_bytes
        {
            drop(artifact);
            drop(prepared);
            drop(bridge);
            return Err(Error::InferenceError(
                "Granite Speech artifact drifted from admitted geometry".into(),
            ));
        }
        let mut prepared = prepared;
        prepared.install_prepared_granite_speech_artifact(variant, artifact)?;
        let (execution, _) = self.coordinator_job_for_request(&prepared)?;
        match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                execution,
                JobResourceObservation {
                    host_bytes: total_retained_host_bytes,
                    accelerator_bytes: geometry.retained_device_bytes,
                },
            )
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                Err(error)
            }
        }
    }

    pub(super) async fn prepare_lfm25_audio_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::Lfm25Audio)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated LFM2.5 Audio variant");
        let model = self
            .model_registry
            .get_lfm25_audio_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("LFM2.5 Audio model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio ASR preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let preparation_contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Asr,
            false,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(ExecutionTargetKind::TokenEngine),
        )?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "LFM2.5 Audio model {variant} has no effective context"
                ))
            })?;
        let model_for_shape = model.clone();
        let (prepared, envelope) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let plan = crate::engine::asr_route_plan(
                    &samples,
                    sample_rate,
                    &model_for_shape.asr_route_capabilities(),
                    request.streaming,
                );
                let long_form = plan.requires_chunk_path();
                let envelope = (!long_form)
                    .then(|| {
                        model_for_shape.lfm25_audio_asr_preparation_resource_envelope(
                            samples.len(),
                            sample_rate,
                        )
                    })
                    .transpose()?;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                if let Some(envelope) = envelope {
                    request.install_prepared_sequence_input_tokens(
                        envelope.geometry.prompt_tokens,
                        context_limit,
                    )?;
                    Ok((request, Some(envelope)))
                } else {
                    request.install_prepared_asr_long_form_atomic(plan)?;
                    Ok((request, None))
                }
            })
            .await?;
        let retained_request_host_bytes =
            u64::try_from(retained_engine_request_input_bytes(&prepared)?).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio ASR retained input exceeds u64".into())
            })?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_host_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        if prepared.uses_asr_long_form_atomic() {
            let (spec, observation) = self.coordinator_job_for_request(&prepared)?;
            return match self
                .coordinator
                .admit_observed_from_preparation(bridge, spec, observation)
                .await
            {
                Ok(job) => Ok((prepared, job)),
                Err(failure) => {
                    drop(prepared);
                    let error = failure.error;
                    drop(failure.bridge);
                    Err(error)
                }
            };
        }
        let envelope = envelope.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio normal route lost preparation envelope".into())
        })?;
        if envelope.backend != self.backend_router.context().backend_kind {
            drop(prepared);
            drop(bridge);
            return Err(Error::InferenceError(
                "LFM2.5 Audio preparation envelope used the wrong backend domain".into(),
            ));
        }
        let preparation_resources = asr_encoder_retained_resources(
            self.backend_router.context().backend_kind,
            retained_request_host_bytes,
            envelope.max_retained_resident_bytes,
        )?;
        let preparation_spec = JobSpec {
            request_id: prepared.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: prepared.priority,
            workload_class: prepared.workload_class,
            deadline: prepared.deadline,
            resources: preparation_resources,
        };
        let preparation_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                preparation_spec,
                JobResourceObservation::host(retained_request_host_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        let (samples, sample_rate) = prepared
            .prepared_asr_audio_for_executor()?
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio decoded audio was lost".into()))?;
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.lfm25_audio".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            envelope.max_work_units,
            envelope.max_materialized_tensor_elements,
            ResourceVector {
                host_bytes: ResourceAmount::Known(envelope.max_host_workspace_bytes),
                device_bytes: ResourceAmount::Known(envelope.max_device_workspace_bytes),
                unified_bytes: ResourceAmount::Known(envelope.max_unified_workspace_bytes),
                ..ResourceVector::zero()
            },
        );
        let cancellation = PreparationCancellation::default();
        let row = self.coordinator.seal_preparation_row(
            preparation_job,
            &preparation_contract,
            &work,
            cost,
            envelope.max_materialized_tensor_elements,
            cancellation.clone(),
        )?;
        let model_for_preparation = model.clone();
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(
                vec![row],
                preparation_contract,
                work,
                move |live| {
                    if live != [0] {
                        return Err(Error::InferenceError(
                            "LFM2.5 Audio scalar preparation received a non-scalar live set".into(),
                        ));
                    }
                    let artifact = model_for_preparation
                        .prepare_lfm25_audio_asr_artifact(samples.as_ref(), sample_rate)?;
                    let retained_host_bytes = retained_request_host_bytes
                        .checked_add(artifact.retained_host_bytes)
                        .ok_or_else(|| {
                            Error::Overloaded(
                                "LFM2.5 Audio ASR retained host accounting overflow".into(),
                            )
                        })?;
                    Ok(vec![Ok(PreparationArtifact {
                        retained: JobResourceObservation {
                            host_bytes: retained_host_bytes,
                            accelerator_bytes: artifact.retained_resident_bytes,
                        },
                        value: artifact,
                    })])
                },
            )
            .await?;
        cancellation_guard.armed = false;
        let outcome = outcomes.pop().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio scalar preparation returned no outcome".into())
        })?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact.value, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(prepared.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(prepared.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        if artifact.source_samples != envelope.geometry.source_samples
            || artifact.source_sample_rate != envelope.geometry.source_sample_rate
            || artifact.resampled_samples != envelope.geometry.resampled_samples
            || artifact.mel_frames != envelope.geometry.total_mel_frames
            || artifact.effective_feature_frames != envelope.geometry.effective_feature_frames
            || artifact.audio_tokens != envelope.geometry.encoder_frames
            || artifact.prompt_tokens != envelope.geometry.prompt_tokens
            || artifact.materialized_tensor_elements
                != envelope.geometry.materialized_tensor_elements
            || artifact.retained_resident_bytes != envelope.geometry.retained_resident_bytes
        {
            drop(artifact);
            drop(prepared);
            drop(bridge);
            return Err(Error::InferenceError(
                "LFM2.5 Audio ASR artifact drifted from admitted geometry".into(),
            ));
        }
        let retained_host_bytes = retained_request_host_bytes
            .checked_add(artifact.retained_host_bytes)
            .ok_or_else(|| {
                Error::Overloaded("LFM2.5 Audio ASR retained host accounting overflow".into())
            })?;
        let accelerator_bytes = artifact.retained_resident_bytes;
        let mut prepared = prepared;
        prepared.install_prepared_lfm25_audio_asr_artifact(variant, artifact)?;
        let (execution, _) = self.coordinator_job_for_request(&prepared)?;
        match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                execution,
                JobResourceObservation {
                    host_bytes: retained_host_bytes,
                    accelerator_bytes,
                },
            )
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                Err(error)
            }
        }
    }

    pub(super) async fn prepare_parakeet_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::ParakeetAsr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Parakeet variant");
        let model = self
            .model_registry
            .get_asr_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not loaded")))?;
        let model_arc = model.model_arc();
        let crate::models::registry::NativeAsrModel::Parakeet(parakeet) = model_arc.as_ref() else {
            return Err(Error::ModelLoadError(
                "Parakeet registry lease crossed model family".into(),
            ));
        };
        let max_symbols = parakeet.max_symbols_per_frame();
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Parakeet preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Asr,
            false,
            false,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(ExecutionTargetKind::TokenEngine),
        )?;
        let prepared = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let max_emittable_tokens =
                    crate::models::architectures::parakeet::asr::parakeet_max_emittable_tokens(
                        samples.len(),
                        sample_rate,
                        max_symbols,
                    )?;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                install_parakeet_decode_budget(&mut request, max_emittable_tokens)?;
                Ok(request)
            })
            .await?;
        let retained_host_bytes = u64::try_from(retained_engine_request_input_bytes(&prepared)?)
            .map_err(|_| Error::Overloaded("Parakeet retained input exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_host_bytes))?;
        let (samples, sample_rate) = prepared
            .prepared_asr_audio_for_executor()?
            .ok_or_else(|| Error::InferenceError("Parakeet decoded audio was lost".into()))?;
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Parakeet sample rate must be greater than zero".into(),
            ));
        }
        let resampled = samples
            .len()
            .checked_mul(16_000)
            .and_then(|value| value.checked_add(sample_rate as usize - 1))
            .map(|value| value / sample_rate as usize)
            .ok_or_else(|| Error::Overloaded("Parakeet resampled length overflowed".into()))?;
        let feature_frames = resampled.div_ceil(160).max(1);
        let encoded_frames = feature_frames.div_ceil(8).max(1);
        let retained_device_ceiling = u64::try_from(encoded_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(1024 * 4))
            .ok_or_else(|| {
                Error::Overloaded("Parakeet retained tensor estimate overflowed".into())
            })?;
        let materialized_elements = u64::try_from(feature_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(1024 * 16))
            .ok_or_else(|| Error::Overloaded("Parakeet preparation geometry overflowed".into()))?;
        // Windowed encoding caps the quadratic attention terms at one window;
        // only the host features and the stitched output grow with length.
        let preparation_bytes =
            crate::models::architectures::parakeet::asr::parakeet_preparation_bytes(
                resampled,
                self.backend_router.context().backend_kind,
            )?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        let prep_spec = JobSpec {
            request_id: prepared.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: prepared.priority,
            workload_class: prepared.workload_class,
            deadline: prepared.deadline,
            resources: asr_encoder_retained_resources(
                self.backend_router.context().backend_kind,
                retained_host_bytes,
                retained_device_ceiling,
            )?,
        };
        let prep_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                prep_spec,
                JobResourceObservation::host(retained_host_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => return Err(failure.error),
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.parakeet".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            u64::try_from(samples.len())
                .map_err(|_| Error::Overloaded("Parakeet sample work exceeds u64".into()))?,
            materialized_elements,
            retained_artifact_resources(
                self.backend_router.context().backend_kind,
                preparation_bytes.host_bytes,
                preparation_bytes.device_bytes()?,
            )?,
        );
        let cancellation = PreparationCancellation::default();
        let row = self.coordinator.seal_preparation_row(
            prep_job,
            &contract,
            &work,
            cost,
            materialized_elements,
            cancellation.clone(),
        )?;
        let language = prepared.asr_language_for_execution().map(str::to_owned);
        let model_for_preparation = model.clone();
        let encoder_cancellation = cancellation.clone();
        let encoder_request_id = prepared.id.clone();
        let encoder_deadline = prepared.deadline;
        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "Parakeet encoder preparation must remain scalar".into(),
                    ));
                }
                let model_arc = model_for_preparation.model_arc();
                let crate::models::registry::NativeAsrModel::Parakeet(model) = model_arc.as_ref()
                else {
                    return Err(Error::InferenceError(
                        "Parakeet preparation crossed model family".into(),
                    ));
                };
                // Checked between encoder windows, so a disconnected or
                // expired request releases the device within one window.
                let checkpoint = || {
                    if encoder_cancellation.is_cancelled() {
                        return Err(Error::Cancelled(encoder_request_id.clone()));
                    }
                    if encoder_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        return Err(Error::Timeout(encoder_request_id.clone()));
                    }
                    Ok(())
                };
                let artifact = match model.prepare_retained_encoder(
                    samples.as_ref(),
                    sample_rate,
                    language.as_deref(),
                    &checkpoint,
                ) {
                    Ok(artifact) => Arc::new(artifact),
                    // A row error keeps its kind; a batch error is reported
                    // as an inference failure.
                    Err(error @ (Error::Cancelled(_) | Error::Timeout(_))) => {
                        return Ok(vec![Err(error)]);
                    }
                    Err(error) => return Err(error),
                };
                let accelerator_bytes = artifact.resident_tensor_bytes()?;
                if accelerator_bytes > retained_device_ceiling {
                    return Err(Error::InferenceError(format!(
                        "Parakeet encoder artifact exceeded its admitted ceiling: {accelerator_bytes} > {retained_device_ceiling}"
                    )));
                }
                Ok(vec![Ok(PreparationArtifact {
                    retained: JobResourceObservation {
                        host_bytes: retained_host_bytes,
                        accelerator_bytes,
                    },
                    value: artifact,
                })])
            })
            .await?;
        guard.armed = false;
        let outcome = outcomes.pop().ok_or_else(|| {
            Error::InferenceError("Parakeet preparation returned no outcome".into())
        })?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(prepared.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(prepared.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let mut prepared = prepared;
        prepared.install_prepared_parakeet_artifact(variant, artifact.value)?;
        let (execution, _) = self.coordinator_job_for_request(&prepared)?;
        match self
            .coordinator
            .admit_observed_from_preparation(bridge, execution, artifact.retained)
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => Err(failure.error),
        }
    }

    /// Granite Speech 5.0: decode the audio and plan its chunks on a host
    /// worker, then dispatch the chunked atomic stage one chunk at a time, so
    /// a long file no longer holds the device for its whole length and the
    /// upload is never decoded on the device thread.
    pub(super) async fn prepare_granite_speech5_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::GraniteSpeech5Asr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated Granite Speech 5.0 variant");
        let model = self
            .model_registry
            .get_asr_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not loaded")))?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Granite Speech 5.0 model {variant} has no effective context"
                ))
            })?;
        let prepared = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let capabilities = crate::engine::AsrRouteCapabilities::for_native_model(
                    &model,
                    variant,
                    &request,
                    Some(context_limit),
                )?;
                let plan = crate::engine::asr_route_plan(
                    &samples,
                    sample_rate,
                    &capabilities,
                    request.streaming,
                );
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                request.install_prepared_asr_long_form_atomic(plan)?;
                Ok(request)
            })
            .await?;
        let retained_host_bytes = u64::try_from(retained_engine_request_input_bytes(&prepared)?)
            .map_err(|_| {
                Error::Overloaded("Granite Speech 5.0 retained input exceeds u64".into())
            })?;
        job.record_materialized_usage(JobResourceObservation::host(retained_host_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        let (spec, observation) = self.coordinator_job_for_request(&prepared)?;
        match self
            .coordinator
            .admit_observed_from_preparation(bridge, spec, observation)
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => {
                drop(prepared);
                let error = failure.error;
                drop(failure.bridge);
                Err(error)
            }
        }
    }

    pub(super) async fn prepare_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        let (request, job) = self
            .prepare_qwen3_asr_shape_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_whisper_asr_shape_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_vibevoice_asr_shape_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_granite_speech_asr_shape_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_granite_speech5_asr_shape_for_binding(request, job)
            .await?;
        let (request, job) = self
            .prepare_parakeet_asr_shape_for_binding(request, job, residency_lease)
            .await?;
        self.prepare_lfm25_audio_asr_shape_for_binding(request, job, residency_lease)
            .await
    }

    /// Decide from the canonical decoded audio, before the request's route is
    /// prepared, whether the whole input holds no speech for a model family
    /// that hallucinates on non-speech (PIPE-18). The decode is transient:
    /// route preparation decodes its own copy, because families without a
    /// preparer must not carry prepared audio.
    pub(super) async fn asr_no_speech_gate(
        &self,
        request: EngineCoreRequest,
        job: &JobLease,
    ) -> Result<(EngineCoreRequest, Option<AsrNoSpeechGate>)> {
        if !asr_no_speech_gate_applies(&request) {
            return Ok((request, None));
        }
        let (request, gate) = self
            .coordinator
            .run_host_blocking_stage(job, move || {
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let gate = AsrNoSpeechGate::evaluate(&samples, sample_rate);
                Ok((request, gate))
            })
            .await?;
        if gate.is_some() {
            self.telemetry.record_asr_no_speech_gated();
        }
        Ok((request, gate))
    }
}
