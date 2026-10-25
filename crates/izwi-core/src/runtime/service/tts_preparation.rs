//! TTS request preparation: the per-family synthesis inputs, artifacts and
//! resources each TTS family binds before admission.

use super::*;

/// VibeVoice sampling for a request: unset fields keep the reference demo's
/// CFG 1.3 and `default_diffusion_steps` (10) (TPIPE-24, VVT-5).
pub(super) fn vibevoice_tts_request_params(
    params: &GenerationParams,
    text: &str,
    default_diffusion_steps: usize,
) -> VibeVoiceTtsGenerationParams {
    let auto_frame_budget = params.max_tokens == 0;
    VibeVoiceTtsGenerationParams {
        cfg_scale: params
            .speech
            .cfg_scale
            .unwrap_or(VIBEVOICE_TTS_DEFAULT_CFG_SCALE),
        diffusion_steps: params
            .speech
            .diffusion_steps
            .unwrap_or(default_diffusion_steps)
            .max(1),
        max_frames: if auto_frame_budget {
            vibevoice_tts_auto_max_frames_for_text(text)
        } else {
            params
                .max_tokens
                .clamp(1, ModelVariant::VIBEVOICE_TTS_MAX_OUTPUT_FRAMES)
        },
        auto_frame_budget,
    }
}

/// Fish S2 sampling for a request: unset fields keep fish-speech's serving
/// defaults, and the seed defaults to one derived from the request id.
pub(super) fn fish_s2_request_params(params: &GenerationParams, request_id: &str) -> FishS2GenerationParams {
    let defaults = FishS2GenerationParams::default();
    FishS2GenerationParams {
        max_frames: if params.max_tokens == 0 {
            defaults.max_frames
        } else {
            params
                .max_tokens
                .clamp(1, ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
        },
        temperature: params.speech.temperature.unwrap_or(defaults.temperature),
        top_p: params.speech.top_p.unwrap_or(defaults.top_p),
        top_k: params
            .audio_top_k
            .or(params.speech.top_k)
            .unwrap_or(defaults.top_k),
        seed: params
            .speech
            .seed
            .unwrap_or_else(|| EngineCoreRequest::chat_request_seed(request_id)),
        ..defaults
    }
}

/// Voxtral sampling for a request: unset fields keep Voxtral's defaults
/// (greedy semantic codes), and the flow-matching seed defaults to one
/// derived from the request id, so a retried frame redraws the same noise.
pub(super) fn voxtral_tts_request_params(
    params: &GenerationParams,
    request_id: &str,
    context_limit: usize,
) -> VoxtralTtsGenerationParams {
    let speech = &params.speech;
    let defaults = VoxtralTtsGenerationParams::default();
    VoxtralTtsGenerationParams {
        max_frames: params
            .max_tokens
            .max(1)
            .min(context_limit.saturating_sub(1).max(1)),
        temperature: speech.temperature.unwrap_or(defaults.temperature).max(0.0),
        top_p: speech.top_p.unwrap_or(defaults.top_p).clamp(0.0, 1.0),
        top_k: speech.top_k.unwrap_or(defaults.top_k),
        seed: speech
            .seed
            .unwrap_or_else(|| EngineCoreRequest::chat_request_seed(request_id).max(1)),
        ..defaults
    }
}

pub(super) fn retained_lfm25_audio_tts_artifact_host_bytes(messages: &[ChatMessage]) -> Result<u64> {
    let mut bytes = 0usize;
    add_chat_messages_allocations(&mut bytes, messages, messages.len())?;
    u64::try_from(bytes)
        .map_err(|_| Error::Overloaded("LFM2.5 Audio TTS retained prompt exceeds u64".into()))
}

pub(super) fn lfm25_audio_tts_preparation_workspace(backend: BackendKind, bytes: u64) -> ResourceVector {
    let mut workspace = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => workspace.host_bytes = ResourceAmount::Known(bytes),
        BackendKind::Metal => workspace.unified_bytes = ResourceAmount::Known(bytes),
        BackendKind::Cuda => workspace.device_bytes = ResourceAmount::Known(bytes),
    }
    workspace
}

/// What one engine Kokoro request reserves, from its prepared chunks
/// (KOK-8/9): the widest chunk's decoder workspace (chunks run one after
/// another) and the joined output of every chunk, each at the planning pace
/// for its own text rather than the model's 50-bins-per-token ceiling.
pub(super) fn kokoro_synthesis_resources(
    backend: BackendKind,
    prepared: &crate::engine::KokoroPreparedText,
) -> Result<ResourceVector> {
    use crate::models::architectures::kokoro::admission::{
        kokoro_chunk_frame_bound, kokoro_chunk_price,
    };
    let overflow = || Error::Overloaded("Kokoro reservation overflowed".into());
    let mut widest: Option<crate::models::architectures::kokoro::admission::KokoroChunkPrice> =
        None;
    let mut output_samples = 0u64;
    for (index, chunk) in prepared.chunks.iter().enumerate() {
        let frames = kokoro_chunk_frame_bound(&chunk.source_text, usize::MAX, chunk.speed);
        let price = kokoro_chunk_price(frames, backend)?;
        let pause = if index == 0 {
            0
        } else {
            KOKORO_JOIN_PAUSE_SAMPLES
        };
        output_samples = output_samples
            .checked_add(price.output_samples)
            .and_then(|samples| samples.checked_add(pause))
            .ok_or_else(overflow)?;
        if widest.is_none_or(|current| price.frames > current.frames) {
            widest = Some(price);
        }
    }
    let widest =
        widest.ok_or_else(|| Error::InvalidInput("Kokoro request has no chunks".into()))?;
    let host_bytes = output_samples
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .and_then(|bytes| bytes.checked_add(widest.host_bytes))
        .ok_or_else(overflow)?;
    super::super::tts::direct_tts_physical_resources(
        backend,
        host_bytes,
        widest.cpu_tensor_bytes,
        widest.accelerator_tensor_bytes,
    )
}

/// Silence the executor inserts between joined Kokoro chunks (40 ms at
/// 24 kHz).
pub(super) const KOKORO_JOIN_PAUSE_SAMPLES: u64 = 960;

impl RuntimeService {
    pub(super) async fn prepare_kokoro_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::KokoroTts)
            || request
                .prepared_kokoro_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Kokoro TTS variant");
        let model = self
            .model_registry
            .get_kokoro_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Kokoro TTS model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Kokoro TTS preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Tts,
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
                    "Kokoro TTS model {variant} has no effective context"
                ))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Kokoro TTS request is missing text".into()))?
            .to_string();
        let speaker = request.tts_speaker_for_execution().map(str::to_string);
        let language = request.language.clone();
        let speed = request.params.speed;
        let seed = request.params.speech.seed;
        let budget = kokoro_output_budget(&text, speed)?;
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("Kokoro retained request exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        let artifact_host_ceiling = retained_request_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                bytes.checked_add(
                    u64::try_from(budget.max_model_tokens)
                        .ok()?
                        .checked_mul(8)?,
                )
            })
            .ok_or_else(|| Error::Overloaded("Kokoro artifact ceiling overflowed".into()))?;
        let preparation_spec = JobSpec {
            request_id: request.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: request.priority,
            workload_class: request.workload_class,
            deadline: request.deadline,
            // One retained style row (256 F32) per chunk the text can split
            // into: a single row's authorization refused long texts (B2).
            resources: retained_artifact_resources(
                self.backend_router.context().backend_kind,
                artifact_host_ceiling,
                u64::try_from(budget.max_chunks)
                    .ok()
                    .and_then(|chunks| chunks.checked_mul(256 * std::mem::size_of::<f32>() as u64))
                    .ok_or_else(|| Error::Overloaded("Kokoro style rows overflowed".into()))?,
            )?,
        };
        let preparation_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                preparation_spec,
                JobResourceObservation::host(retained_request_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => return Err(failure.error),
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.kokoro".into(),
        };
        let max_model_tokens = u64::try_from(budget.max_model_tokens)
            .map_err(|_| Error::Overloaded("Kokoro token budget exceeds u64".into()))?;
        let cost = crate::engine::WorkCost::new(1, max_model_tokens, 0);
        let cancellation = PreparationCancellation::default();
        let row = self.coordinator.seal_preparation_row(
            preparation_job,
            &contract,
            &work,
            cost,
            max_model_tokens,
            cancellation.clone(),
        )?;
        let model_for_preparation = model.clone();
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "Kokoro preparation must remain scalar".into(),
                    ));
                }
                let artifact = Arc::new(crate::runtime::kokoro::prepare_kokoro_text(
                    &*model_for_preparation,
                    &text,
                    speaker.as_deref(),
                    language.as_deref(),
                    speed,
                    seed,
                )?);
                let retained = JobResourceObservation {
                    host_bytes: retained_request_bytes
                        .checked_add(artifact.retained_host_bytes()?)
                        .ok_or_else(|| {
                            Error::Overloaded("Kokoro retained bytes overflowed".into())
                        })?,
                    accelerator_bytes: artifact.retained_tensor_bytes()?,
                };
                Ok(vec![Ok(PreparationArtifact {
                    retained,
                    value: artifact,
                })])
            })
            .await?;
        cancellation_guard.armed = false;
        let (artifact, bridge) = match outcomes
            .pop()
            .ok_or_else(|| Error::InferenceError("Kokoro preparation returned no outcome".into()))?
        {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let retained = artifact.retained;
        let mut prepared = request;
        prepared.install_kokoro_tts_execution_model(
            variant,
            model,
            artifact.value,
            context_limit,
        )?;
        let (mut execution, _) = self.coordinator_job_for_request(&prepared)?;
        let prepared_text = prepared
            .prepared_kokoro_tts_artifact_for_executor()?
            .ok_or_else(|| Error::InferenceError("prepared Kokoro request lost its text".into()))?;
        execution.resources = execution.resources.checked_add(kokoro_synthesis_resources(
            self.backend_router.context().backend_kind,
            &prepared_text,
        )?)?;
        match self
            .coordinator
            .admit_observed_from_preparation(bridge, execution, retained)
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => Err(failure.error),
        }
    }

    pub(super) async fn prepare_vibevoice_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::VibeVoiceTts)
            || request
                .prepared_vibevoice_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated VibeVoice TTS variant");
        let model = self
            .model_registry
            .get_vibevoice_tts_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("VibeVoice TTS model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Tts,
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
                    "VibeVoice TTS model {variant} has no effective context"
                ))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("VibeVoice TTS request is missing text".into()))?
            .trim()
            .to_string();
        let reference_audio = request.tts_reference_audio_for_execution().ok_or_else(|| {
            Error::InvalidInput("VibeVoice TTS requires reference_audio and reference_text".into())
        })?;
        let reference_text = request
            .tts_reference_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::InvalidInput(
                    "VibeVoice TTS requires reference_audio and reference_text".into(),
                )
            })?;
        let (audio_samples, sample_rate) = decode_reference_audio_base64(reference_audio)?;
        let reference = VibeVoiceSpeakerReference {
            audio_samples,
            sample_rate,
            text: reference_text.to_string(),
        };
        let requested_speaker = request.tts_speaker_for_execution().map(str::to_string);
        let params =
            vibevoice_tts_request_params(&request.params, &text, model.default_diffusion_steps());
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("VibeVoice TTS retained request exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        let preparation_spec = JobSpec {
            request_id: request.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: request.priority,
            workload_class: request.workload_class,
            deadline: request.deadline,
            resources: asr_encoder_retained_resources(
                self.backend_router.context().backend_kind,
                retained_request_bytes,
                512 * 1024 * 1024,
            )?,
        };
        let preparation_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                preparation_spec,
                JobResourceObservation::host(retained_request_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => return Err(failure.error),
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.vibevoice".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            u64::try_from(context_limit)
                .map_err(|_| Error::Overloaded("VibeVoice context exceeds u64".into()))?,
            u64::try_from(context_limit).unwrap_or(u64::MAX),
            lfm25_audio_tts_preparation_workspace(
                self.backend_router.context().backend_kind,
                512 * 1024 * 1024,
            ),
        );
        let cancellation = PreparationCancellation::default();
        let row = self.coordinator.seal_preparation_row(
            preparation_job,
            &contract,
            &work,
            cost,
            u64::try_from(context_limit).unwrap_or(u64::MAX),
            cancellation.clone(),
        )?;
        let model_for_preparation = model.clone();
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "VibeVoice TTS preparation must remain scalar".into(),
                    ));
                }
                let artifact = model_for_preparation.prepare_retained_artifact(
                    &text,
                    &reference,
                    requested_speaker.as_deref(),
                )?;
                let accelerator_bytes = artifact.retained_tensor_bytes()?;
                Ok(vec![Ok(PreparationArtifact {
                    retained: JobResourceObservation {
                        host_bytes: retained_request_bytes,
                        accelerator_bytes,
                    },
                    value: artifact,
                })])
            })
            .await?;
        cancellation_guard.armed = false;
        let (artifact, bridge) = match outcomes.pop().ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS preparation returned no outcome".into())
        })? {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let accelerator_bytes = artifact.retained.accelerator_bytes;
        let mut prepared = request;
        prepared.install_vibevoice_tts_execution_model(
            variant,
            model,
            artifact.value,
            params,
            context_limit,
        )?;
        let (execution, _) = self.coordinator_job_for_request(&prepared)?;
        match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                execution,
                JobResourceObservation {
                    host_bytes: retained_request_bytes,
                    accelerator_bytes,
                },
            )
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => Err(failure.error),
        }
    }

    pub(super) async fn prepare_fish_s2_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::FishS2Tts)
            || request
                .prepared_fish_s2_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated Fish S2 TTS variant");
        let model = self
            .model_registry
            .get_fish_s2_tts_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Fish S2 TTS model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Fish S2 TTS preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Tts,
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
                    "Fish S2 TTS model {variant} has no effective context"
                ))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Fish S2 TTS request is missing text".into()))?
            .trim()
            .to_string();
        let reference_audio = request.tts_reference_audio_for_execution().ok_or_else(|| {
            Error::InvalidInput("Fish S2 requires reference_audio and reference_text".into())
        })?;
        let reference_text = request
            .tts_reference_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::InvalidInput("Fish S2 requires reference_audio and reference_text".into())
            })?;
        let (audio_samples, sample_rate) = decode_reference_audio_base64(reference_audio)?;
        let reference = FishS2Reference {
            audio_samples,
            sample_rate,
            text: reference_text.to_string(),
        };
        let params = fish_s2_request_params(&request.params, &request.id);
        params.validate()?;
        let memory = model.preparation_memory(&text, &reference, context_limit)?;
        // The text clone and decoded reference were constructed under the initial
        // input/audio-decode lease. Keep their ownership covered across the bridge;
        // subsequent codec/tokenizer allocations begin only after admission.
        let preparation_host_bytes = memory
            .host_bytes()?
            .checked_add(u64::try_from(text.capacity()).map_err(|_| {
                Error::Overloaded("Fish S2 preparation text capacity exceeds u64".into())
            })?)
            .ok_or_else(|| Error::Overloaded("Fish S2 preparation host bytes overflow".into()))?;
        let codec_workspace = memory.accelerator_workspace_bytes;
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("Fish S2 TTS retained request exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_bytes))?;
        let preparation_spec = JobSpec {
            request_id: request.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: request.priority,
            workload_class: request.workload_class,
            deadline: request.deadline,
            resources: fish_s2_preparation_resources(
                self.backend_router.context().backend_kind,
                retained_request_bytes,
                preparation_host_bytes,
            )?,
        };
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        let preparation_job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                preparation_spec,
                JobResourceObservation::host(retained_request_bytes),
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                return Err(release_failed_preparation(
                    failure,
                    (request, text, reference),
                ));
            }
        };
        let request_id = request.id.clone();
        let codec_deadline = request.deadline;
        let owned_inputs =
            PreparationOwnedInputs::new((request, text, reference), preparation_job.clone());
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.fish_s2".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            u64::try_from(context_limit)
                .map_err(|_| Error::Overloaded("Fish S2 context exceeds u64".into()))?,
            u64::try_from(context_limit).unwrap_or(u64::MAX),
            lfm25_audio_tts_preparation_workspace(
                self.backend_router.context().backend_kind,
                codec_workspace,
            ),
        );
        let cancellation = PreparationCancellation::default();
        let row = self.coordinator.seal_preparation_row(
            preparation_job,
            &contract,
            &work,
            cost,
            u64::try_from(context_limit).unwrap_or(u64::MAX),
            cancellation.clone(),
        )?;
        let model_for_preparation = model.clone();
        let cancellation_for_codec = cancellation.clone();
        let codec_request_id = request_id.clone();
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "Fish S2 TTS preparation must remain scalar".into(),
                    ));
                }
                owned_inputs.run(|(request, text, reference)| {
                    let artifact = model_for_preparation.prepare_retained_artifact_with_cancel(
                        &text,
                        reference,
                        context_limit,
                        &|| {
                            if cancellation_for_codec.is_cancelled() {
                                return Err(Error::Cancelled(codec_request_id.clone()));
                            }
                            if codec_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                                return Err(Error::Timeout(codec_request_id.clone()));
                            }
                            Ok(())
                        },
                    )?;
                    let retained_host_bytes = retained_request_bytes
                        .checked_add(artifact.retained_bytes()?)
                        .ok_or_else(|| {
                            Error::Overloaded("Fish S2 retained bytes overflow".into())
                        })?;
                    Ok(vec![Ok(PreparationArtifact {
                        retained: JobResourceObservation::host(retained_host_bytes),
                        value: (request, artifact),
                    })])
                })
            })
            .await?;
        cancellation_guard.armed = false;
        let (artifact, bridge) = match outcomes.pop().ok_or_else(|| {
            Error::InferenceError("Fish S2 TTS preparation returned no outcome".into())
        })? {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request_id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request_id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let retained = artifact.retained;
        let (mut prepared, prepared_artifact) = artifact.value;
        prepared.install_fish_s2_tts_execution_model(
            variant,
            model,
            prepared_artifact,
            params,
            context_limit,
        )?;
        let (execution, _) = self.coordinator_job_for_request(&prepared)?;
        match self
            .coordinator
            .admit_observed_from_preparation(bridge, execution, retained)
            .await
        {
            Ok(job) => Ok((prepared, job)),
            Err(failure) => Err(release_failed_preparation(failure, prepared)),
        }
    }

    pub(super) async fn prepare_voxtral_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::VoxtralTts)
            || request
                .prepared_voxtral_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated Voxtral TTS variant");
        let model = self
            .model_registry
            .get_voxtral_tts_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Voxtral TTS model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Voxtral TTS preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Tts,
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
                    "Voxtral TTS model {variant} has no effective context"
                ))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS request is missing text".into()))?
            .trim()
            .to_string();
        let voice = request
            .tts_speaker_for_execution()
            .map(str::to_string)
            .or_else(|| model.available_speakers().into_iter().next())
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS has no available voice".into()))?;
        let params = voxtral_tts_request_params(&request.params, &request.id, context_limit);
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("Voxtral TTS retained request exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        let preparation_job = self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                JobSpec {
                    request_id: request.id.clone(),
                    lane: CoordinatorLane::Atomic,
                    priority: request.priority,
                    workload_class: request.workload_class,
                    deadline: request.deadline,
                    resources: asr_encoder_retained_resources(
                        self.backend_router.context().backend_kind,
                        retained_request_bytes,
                        512 * 1024 * 1024,
                    )?,
                },
                JobResourceObservation::host(retained_request_bytes),
            )
            .await
            .map_err(|failure| failure.error)?;
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.voxtral".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            context_limit as u64,
            context_limit as u64,
            lfm25_audio_tts_preparation_workspace(
                self.backend_router.context().backend_kind,
                512 * 1024 * 1024,
            ),
        );
        let cancellation = PreparationCancellation::default();
        let row = self.coordinator.seal_preparation_row(
            preparation_job,
            &contract,
            &work,
            cost,
            context_limit as u64,
            cancellation.clone(),
        )?;
        let model_for_preparation = model.clone();
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "Voxtral TTS preparation must remain scalar".into(),
                    ));
                }
                let artifact = model_for_preparation.prepare_retained_artifact(&text, &voice)?;
                Ok(vec![Ok(PreparationArtifact {
                    retained: JobResourceObservation {
                        host_bytes: retained_request_bytes,
                        accelerator_bytes: artifact.retained_resident_bytes,
                    },
                    value: artifact,
                })])
            })
            .await?;
        cancellation_guard.armed = false;
        let (artifact, bridge) = match outcomes.pop().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS preparation returned no outcome".into())
        })? {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let retained = artifact.retained;
        let mut prepared = request;
        prepared.install_voxtral_tts_execution_model(
            variant,
            model,
            artifact.value,
            params,
            context_limit,
        )?;
        let (execution, _) = self.coordinator_job_for_request(&prepared)?;
        self.coordinator
            .admit_observed_from_preparation(bridge, execution, retained)
            .await
            .map(|job| (prepared, job))
            .map_err(|failure| failure.error)
    }

    pub(super) async fn prepare_lfm25_audio_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::Lfm25Audio)
            || request
                .prepared_lfm25_audio_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request
            .model_variant
            .expect("validated LFM2.5 Audio TTS variant");
        let model = self
            .model_registry
            .get_lfm25_audio_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("LFM2.5 Audio model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio TTS preparation requires model residency".into())
        })?;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency.variant());
        let preparation_contract = loaded_contract_for_residency(
            residency,
            loaded_bundle.as_deref(),
            CapabilityKind::Tts,
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
        let ceiling = model.lfm25_audio_tts_stage_ceiling()?;
        let backend = self.backend_router.context().backend_kind;
        if ceiling.backend != backend {
            return Err(Error::InferenceError(
                "LFM2.5 Audio TTS preparation ceiling used the wrong backend domain".into(),
            ));
        }
        let retained_request_host_bytes =
            u64::try_from(retained_engine_request_input_bytes(&request)?).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio TTS retained input exceeds u64".into())
            })?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_host_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        // Priced from this request's prompt bound, never the model ceiling
        // (LFT-1): the ceiling's full-context prefill refused every request.
        let messages = request.lfm25_audio_tts_messages_for_preparation()?;
        let envelope = model.lfm25_audio_tts_preparation_envelope(&messages)?;
        if envelope.backend != backend || envelope.max_prompt_tokens > ceiling.max_prompt_tokens {
            return Err(Error::InferenceError(
                "LFM2.5 Audio TTS preparation envelope exceeds its loaded ceiling".into(),
            ));
        }
        let preparation_resources = asr_encoder_retained_resources(
            backend,
            retained_request_host_bytes,
            envelope.max_retained_resident_bytes,
        )?;
        let preparation_spec = JobSpec {
            request_id: request.id.clone(),
            lane: CoordinatorLane::Atomic,
            priority: request.priority,
            workload_class: request.workload_class,
            deadline: request.deadline,
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
                drop(request);
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.lfm25_audio".into(),
        };
        let workspace = lfm25_audio_tts_preparation_workspace(backend, envelope.workspace_bytes);
        let cost = crate::engine::WorkCost::with_workspace(
            u64::try_from(envelope.max_prompt_tokens).map_err(|_| {
                Error::Overloaded("LFM2.5 Audio TTS prompt bound exceeds u64".into())
            })?,
            envelope.max_materialized_tensor_elements,
            workspace,
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
                            "LFM2.5 Audio TTS preparation received a non-scalar live set".into(),
                        ));
                    }
                    let artifact =
                        model_for_preparation.prepare_lfm25_audio_tts_artifact(&messages)?;
                    let artifact_host_bytes = retained_lfm25_audio_tts_artifact_host_bytes(
                        artifact.source_messages.as_ref(),
                    )?;
                    let retained_host_bytes = retained_request_host_bytes
                        .checked_add(artifact_host_bytes)
                        .ok_or_else(|| {
                            Error::Overloaded(
                                "LFM2.5 Audio TTS retained host accounting overflow".into(),
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
            Error::InferenceError("LFM2.5 Audio TTS preparation returned no outcome".into())
        })?;
        let (artifact, bridge) = match outcome {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact.value, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        if artifact.prompt_tokens == 0
            || artifact.prompt_tokens > envelope.max_prompt_tokens
            || artifact.materialized_tensor_elements > envelope.max_materialized_tensor_elements
            || artifact.retained_resident_bytes > envelope.max_retained_resident_bytes
        {
            drop(artifact);
            drop(request);
            drop(bridge);
            return Err(Error::InferenceError(
                "LFM2.5 Audio TTS artifact exceeded its admitted preparation ceiling".into(),
            ));
        }
        let artifact_host_bytes =
            retained_lfm25_audio_tts_artifact_host_bytes(artifact.source_messages.as_ref())?;
        let retained_host_bytes = retained_request_host_bytes
            .checked_add(artifact_host_bytes)
            .ok_or_else(|| {
                Error::Overloaded("LFM2.5 Audio TTS retained host accounting overflow".into())
            })?;
        let accelerator_bytes = artifact.retained_resident_bytes;
        let mut prepared = request;
        prepared.install_lfm25_audio_tts_execution_model(
            variant,
            model,
            artifact,
            context_limit,
        )?;
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

    /// Every admitted execution mode must install the same model-bound artifacts
    /// before binding or engine admission. Keep the family pipeline in one place.
    pub(super) async fn prepare_request_for_binding(
        &self,
        mut request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        request.normalize_tts_params_before_preparation()?;
        let (request, job) = self
            .prepare_asr_shape_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_kokoro_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_vibevoice_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_fish_s2_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_voxtral_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_lfm25_audio_tts_for_binding(request, job, residency_lease)
            .await?;
        Ok((request, job))
    }
}
