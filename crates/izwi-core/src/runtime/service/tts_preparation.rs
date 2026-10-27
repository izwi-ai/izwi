//! TTS request preparation: the per-family synthesis inputs, artifacts and
//! resources each TTS family binds before admission.

use sha2::{Digest, Sha256};

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

/// Chatterbox generation for a request: unset fields keep upstream's
/// defaults (temperature 0.8, repetition penalty 1.2, min-p 0.05, top-p 1,
/// exaggeration 0.5, guidance 0.5); the speech-token budget is the
/// request's (the routes fill a text-aware one) within the 1,000-token
/// limit; the seed defaults to one derived from the request id.
pub(super) fn chatterbox_tts_request_params(
    params: &GenerationParams,
    request_id: &str,
) -> Result<ChatterboxGenerationParams> {
    let speech = &params.speech;
    let defaults = ChatterboxGenerationParams::defaults(1);
    let generation = ChatterboxGenerationParams {
        sampling: T3SamplingParams {
            temperature: speech.temperature.unwrap_or(defaults.sampling.temperature),
            repetition_penalty: speech
                .repetition_penalty
                .unwrap_or(defaults.sampling.repetition_penalty),
            min_p: speech.min_p.unwrap_or(defaults.sampling.min_p),
            top_p: speech.top_p.unwrap_or(defaults.sampling.top_p),
            cfg_weight: speech.cfg_weight.unwrap_or(defaults.sampling.cfg_weight),
        },
        exaggeration: speech.exaggeration.unwrap_or(defaults.exaggeration),
        max_speech_tokens: if params.max_tokens == 0 {
            defaults.max_speech_tokens
        } else {
            params.max_tokens.min(defaults.max_speech_tokens)
        },
        seed: speech
            .seed
            .filter(|seed| *seed != 0)
            .unwrap_or_else(|| EngineCoreRequest::chat_request_seed(request_id).max(1)),
        language: defaults.language,
    };
    generation.validate()?;
    Ok(generation)
}

/// Maya1 generation for a request: unset fields keep the card's sampling
/// (temperature 0.4, top-p 0.9, top-k 50, repetition penalty 1.1, at least
/// 28 tokens); the budget is the request's, else one sized to `text`, within
/// the card's 2,048 tokens; the seed defaults to one derived from the
/// request id.
pub(super) fn maya1_tts_request_params(
    params: &GenerationParams,
    request_id: &str,
    text: &str,
) -> Result<Maya1GenerationParams> {
    let speech = &params.speech;
    let defaults = Maya1SamplingParams::default();
    let generation = Maya1GenerationParams {
        sampling: Maya1SamplingParams {
            temperature: speech.temperature.unwrap_or(defaults.temperature),
            top_p: speech.top_p.unwrap_or(defaults.top_p),
            top_k: speech.top_k.unwrap_or(defaults.top_k),
            repetition_penalty: speech
                .repetition_penalty
                .unwrap_or(defaults.repetition_penalty),
            min_new_tokens: defaults.min_new_tokens,
        },
        max_new_tokens: if params.max_tokens == 0 {
            maya1_auto_max_new_tokens(text)
        } else {
            params.max_tokens.min(ModelVariant::MAYA1_MAX_OUTPUT_FRAMES)
        },
        seed: speech
            .seed
            .filter(|seed| *seed != 0)
            .unwrap_or_else(|| EngineCoreRequest::chat_request_seed(request_id).max(1)),
    };
    generation.validate()?;
    Ok(generation)
}

/// The description a Maya1 request speaks with: its own, else the one its
/// built-in voice (an OpenAI name aliased to one) stands for, else the
/// default voice's.
pub(super) fn maya1_voice_description(
    description: Option<&str>,
    speaker: Option<&str>,
) -> Result<String> {
    if let Some(description) = description.map(str::trim).filter(|value| !value.is_empty()) {
        return Ok(description.to_string());
    }
    let name = speaker
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(MAYA1_DEFAULT_VOICE);
    maya1_voice_preset(name)
        .map(|preset| preset.description.to_string())
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Maya1 has no voice '{name}'; use one of {} or describe the voice",
                MAYA1_VOICE_PRESETS
                    .iter()
                    .map(|preset| preset.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// Maya2 generation for a request: unset fields keep upstream's sampler
/// (codebook-0 temperature 0.9, top-k 64, no nucleus, presence penalty 0.8);
/// `repetition_penalty` is that presence penalty; `max_tokens` below the
/// 1,500-frame ceiling caps the frames (the server sends the ceiling when the
/// caller sets none, which leaves the per-piece budget to the frame prior);
/// the seed defaults to one derived from the request id.
pub(super) fn maya2_tts_request_params(
    params: &GenerationParams,
    request_id: &str,
) -> Result<Maya2GenerationParams> {
    let speech = &params.speech;
    let defaults = Maya2SamplingParams::default();
    let generation = Maya2GenerationParams {
        sampling: Maya2SamplingParams {
            temperature: speech.temperature.unwrap_or(defaults.temperature),
            top_k: speech.top_k.unwrap_or(defaults.top_k),
            top_p: speech.top_p.unwrap_or(defaults.top_p),
            presence_penalty: speech
                .repetition_penalty
                .unwrap_or(defaults.presence_penalty),
        },
        max_frames: (params.max_tokens != 0
            && params.max_tokens < ModelVariant::MAYA2_MAX_OUTPUT_FRAMES)
            .then_some(params.max_tokens),
        seed: speech
            .seed
            .filter(|seed| *seed != 0)
            .unwrap_or_else(|| EngineCoreRequest::chat_request_seed(request_id).max(1)),
    };
    generation.validate()?;
    Ok(generation)
}

/// Who speaks a Maya2 request, before any audio is decoded.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Maya2VoiceChoice {
    /// A built-in voice (its exact name).
    Named(&'static str),
    /// A clone of base64 reference audio with its transcript and language.
    Clone {
        audio: String,
        transcript: String,
        language: &'static str,
    },
}

/// The voice a Maya2 request names: a reference clip (which needs its
/// transcript; its language is the request's `reference_language`, else the
/// transcript's dominant script), else a built-in voice or an OpenAI name
/// aliased to one, else the default voice.
pub(super) fn maya2_voice_choice(
    reference_audio: Option<&str>,
    reference_text: Option<&str>,
    reference_language: Option<&str>,
    speaker: Option<&str>,
) -> Result<Maya2VoiceChoice> {
    if let Some(audio) = reference_audio.filter(|audio| !audio.trim().is_empty()) {
        let transcript = reference_text
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .ok_or_else(|| {
                Error::InvalidInput(
                    "Maya 2 Native clones a voice from reference audio and its exact \
                     transcript (reference_text)"
                        .into(),
                )
            })?;
        let language = resolve_language(reference_language, transcript)?;
        return Ok(Maya2VoiceChoice::Clone {
            audio: audio.to_string(),
            transcript: transcript.to_string(),
            language,
        });
    }
    let name = speaker
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(MAYA2_DEFAULT_VOICE);
    maya2_voice(name)
        .map(|voice| Maya2VoiceChoice::Named(voice.name))
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Maya 2 Native has no voice '{name}'; use one of {} or clone one from reference audio",
                MAYA2_VOICES
                    .iter()
                    .map(|voice| voice.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
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

    pub(super) async fn prepare_chatterbox_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::ChatterboxTts)
            || request
                .prepared_chatterbox_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Chatterbox variant");
        let model = self
            .model_registry
            .get_chatterbox_tts_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Chatterbox model {variant} is not loaded"))
            })?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Chatterbox preparation requires model residency".into())
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
                    "Chatterbox model {variant} has no effective context"
                ))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Chatterbox request is missing text".into()))?
            .trim()
            .to_string();
        // A reference wins over a voice name; otherwise any name the route
        // let through (an OpenAI preset name included) selects the one
        // built-in voice.
        let reference_audio = request
            .tts_reference_audio_for_execution()
            .map(str::to_string);
        let mut params = chatterbox_tts_request_params(&request.params, &request.id)?;
        params.language = model.runtime().language_tag(request.language.as_deref())?;
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("Chatterbox retained request exceeds u64".into()))?;
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
            kind: "tts.prepare.chatterbox".into(),
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
        let params_for_preparation = params.clone();
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "Chatterbox preparation must remain scalar".into(),
                    ));
                }
                let voice = reference_audio
                    .as_deref()
                    .map(|audio| {
                        let digest: [u8; 32] = Sha256::digest(audio.as_bytes()).into();
                        model_for_preparation
                            .reference_voice(digest, || decode_reference_audio_base64(audio))
                    })
                    .transpose()?;
                let artifact = model_for_preparation.prepare_retained_artifact(
                    &text,
                    voice,
                    &params_for_preparation,
                )?;
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
            Error::InferenceError("Chatterbox preparation returned no outcome".into())
        })? {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let retained = artifact.retained;
        let mut prepared = request;
        prepared.install_chatterbox_tts_execution_model(
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

    pub(super) async fn prepare_maya1_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::Maya1Tts)
            || request
                .prepared_maya1_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Maya1 variant");
        let model = self
            .model_registry
            .get_maya1_tts_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("Maya1 model {variant} is not loaded")))?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Maya1 preparation requires model residency".into())
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
                Error::ModelLoadError(format!("Maya1 model {variant} has no effective context"))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Maya1 request is missing text".into()))?
            .trim()
            .to_string();
        // A request's own description wins; else its built-in voice's.
        let description = maya1_voice_description(
            request.voice_description.as_deref(),
            request.tts_speaker_for_execution(),
        )?;
        let params = maya1_tts_request_params(&request.params, &request.id, &text)?;
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("Maya1 retained request exceeds u64".into()))?;
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
                        64 * 1024 * 1024,
                    )?,
                },
                JobResourceObservation::host(retained_request_bytes),
            )
            .await
            .map_err(|failure| failure.error)?;
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.maya1".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            context_limit as u64,
            context_limit as u64,
            lfm25_audio_tts_preparation_workspace(
                self.backend_router.context().backend_kind,
                64 * 1024 * 1024,
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
                        "Maya1 preparation must remain scalar".into(),
                    ));
                }
                let artifact =
                    model_for_preparation.prepare_retained_artifact(&description, &text)?;
                Ok(vec![Ok(PreparationArtifact {
                    retained: JobResourceObservation {
                        host_bytes: retained_request_bytes,
                        accelerator_bytes: 0,
                    },
                    value: artifact,
                })])
            })
            .await?;
        cancellation_guard.armed = false;
        let (artifact, bridge) = match outcomes
            .pop()
            .ok_or_else(|| Error::InferenceError("Maya1 preparation returned no outcome".into()))?
        {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let retained = artifact.retained;
        let mut prepared = request;
        prepared.install_maya1_tts_execution_model(
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

    pub(super) async fn prepare_maya2_tts_for_binding(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&ModelResidencyLease>,
    ) -> Result<(EngineCoreRequest, JobLease)> {
        if request.task_type != TaskType::TTS
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::Maya2Tts)
            || request
                .prepared_maya2_tts_artifact_for_executor()?
                .is_some()
        {
            return Ok((request, job));
        }
        let variant = request.model_variant.expect("validated Maya2 variant");
        let model = self
            .model_registry
            .get_maya2_tts_lease(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(format!("Maya2 model {variant} is not loaded")))?;
        let residency = residency_lease.ok_or_else(|| {
            Error::InferenceError("Maya2 preparation requires model residency".into())
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
                Error::ModelLoadError(format!("Maya2 model {variant} has no effective context"))
            })?;
        let text = request
            .tts_text_for_execution()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Maya2 request is missing text".into()))?
            .trim()
            .to_string();
        let language = resolve_language(request.language.as_deref(), &text)?;
        // A reference clip (with its transcript) wins; else a built-in voice.
        let voice = maya2_voice_choice(
            request.tts_reference_audio_for_execution(),
            request.tts_reference_text_for_execution(),
            request.reference_language.as_deref(),
            request.tts_speaker_for_execution(),
        )?;
        let params = maya2_tts_request_params(&request.params, &request.id)?;
        let retained_request_bytes = u64::try_from(retained_engine_request_input_bytes(&request)?)
            .map_err(|_| Error::Overloaded("Maya2 retained request exceeds u64".into()))?;
        job.record_materialized_usage(JobResourceObservation::host(retained_request_bytes))?;
        let bridge = self.coordinator.bridge_preparation_admission(job)?;
        // A clone encodes its reference with Mimi and ECAPA here (once per
        // distinct clip: the model caches prepared voices).
        let workspace = match voice {
            Maya2VoiceChoice::Named(_) => 64 * 1024 * 1024,
            Maya2VoiceChoice::Clone { .. } => 768 * 1024 * 1024,
        };
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
                        workspace,
                    )?,
                },
                JobResourceObservation::host(retained_request_bytes),
            )
            .await
            .map_err(|failure| failure.error)?;
        let work = WorkUnit::PreSequencePreparation {
            kind: "tts.prepare.maya2".into(),
        };
        let cost = crate::engine::WorkCost::with_workspace(
            context_limit as u64,
            context_limit as u64,
            lfm25_audio_tts_preparation_workspace(
                self.backend_router.context().backend_kind,
                workspace,
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
        let max_frames = params.max_frames;
        let mut cancellation_guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "Maya2 preparation must remain scalar".into(),
                    ));
                }
                let voice = match voice {
                    Maya2VoiceChoice::Named(name) => Maya2Voice::Named(name),
                    Maya2VoiceChoice::Clone {
                        audio,
                        transcript,
                        language,
                    } => {
                        let (samples, rate) = decode_reference_audio_base64(&audio)?;
                        let samples = crate::audio::resample_mono_high_quality(
                            &samples,
                            rate,
                            crate::models::architectures::maya2::SAMPLE_RATE,
                        )?;
                        Maya2Voice::Cloned(model_for_preparation.prepare_clone(
                            &samples,
                            &transcript,
                            language,
                        )?)
                    }
                };
                let artifact = model_for_preparation
                    .prepare_retained_artifact(voice, language, &text, max_frames)?;
                Ok(vec![Ok(PreparationArtifact {
                    retained: JobResourceObservation {
                        host_bytes: retained_request_bytes,
                        accelerator_bytes: 0,
                    },
                    value: artifact,
                })])
            })
            .await?;
        cancellation_guard.armed = false;
        let (artifact, bridge) = match outcomes
            .pop()
            .ok_or_else(|| Error::InferenceError("Maya2 preparation returned no outcome".into()))?
        {
            PreparationRowOutcome::Committed { artifact, bridge } => (artifact, bridge),
            PreparationRowOutcome::Cancelled => return Err(Error::Cancelled(request.id.clone())),
            PreparationRowOutcome::TimedOut => return Err(Error::Timeout(request.id.clone())),
            PreparationRowOutcome::Failed(error) => return Err(error),
        };
        let retained = artifact.retained;
        let mut prepared = request;
        prepared.install_maya2_tts_execution_model(
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
    ///
    /// Boxed: the pipeline awaits every family's preparation in turn, so its
    /// future holds the largest of them plus a request at each step (≈ 22 KB
    /// in debug builds). Inline, that size is repeated in every future that
    /// awaits it, up the chat and speech paths, until a debug build overflows
    /// a 2 MiB thread stack.
    pub(super) fn prepare_request_for_binding<'a>(
        &'a self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<&'a ModelResidencyLease>,
    ) -> futures::future::BoxFuture<'a, Result<(EngineCoreRequest, JobLease)>> {
        Box::pin(self.prepare_request_for_binding_inline(request, job, residency_lease))
    }

    async fn prepare_request_for_binding_inline(
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
            .prepare_chatterbox_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_maya1_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_maya2_tts_for_binding(request, job, residency_lease)
            .await?;
        let (request, job) = self
            .prepare_lfm25_audio_tts_for_binding(request, job, residency_lease)
            .await?;
        Ok((request, job))
    }
}

#[cfg(test)]
mod maya1_tests {
    use super::*;

    #[test]
    fn maya1_voice_is_the_description_else_a_preset() {
        let own = maya1_voice_description(Some("  A calm narrator.  "), Some("leo")).unwrap();
        assert_eq!(own, "A calm narrator.");
        let leo = maya1_voice_description(None, Some("leo")).unwrap();
        assert!(leo.starts_with("Realistic male voice"), "{leo}");
        let alloy = maya1_voice_description(Some(" "), Some("alloy")).unwrap();
        assert_eq!(
            alloy,
            maya1_voice_description(None, Some(MAYA1_DEFAULT_VOICE)).unwrap()
        );
        assert!(maya1_voice_description(None, None).is_ok());
        assert!(maya1_voice_description(None, Some("nobody")).is_err());
    }

    #[test]
    fn maya1_params_keep_the_card_defaults_and_size_the_budget() {
        let mut params = GenerationParams {
            max_tokens: 0,
            ..GenerationParams::default()
        };
        let auto = maya1_tts_request_params(&params, "request-1", "Hello there.").unwrap();
        assert_eq!(auto.sampling, Maya1SamplingParams::default());
        assert_eq!(
            auto.max_new_tokens,
            maya1_auto_max_new_tokens("Hello there.")
        );
        assert!(auto.seed > 0);
        params.max_tokens = 99_999;
        params.speech.temperature = Some(0.7);
        params.speech.seed = Some(42);
        let explicit = maya1_tts_request_params(&params, "request-1", "Hi.").unwrap();
        assert_eq!(
            explicit.max_new_tokens,
            ModelVariant::MAYA1_MAX_OUTPUT_FRAMES
        );
        assert_eq!(explicit.sampling.temperature, 0.7);
        assert_eq!(explicit.seed, 42);
        params.speech.temperature = Some(0.0);
        assert!(maya1_tts_request_params(&params, "request-1", "Hi.").is_err());
    }
}

#[cfg(test)]
mod maya2_tests {
    use super::*;

    #[test]
    fn maya2_voice_is_a_clone_else_a_named_voice() {
        assert_eq!(
            maya2_voice_choice(None, None, None, Some("ahana")).unwrap(),
            Maya2VoiceChoice::Named("Ahana")
        );
        // OpenAI names map to fixed voices; none at all is the default.
        assert_eq!(
            maya2_voice_choice(None, None, None, Some("alloy")).unwrap(),
            Maya2VoiceChoice::Named("Vaidehi")
        );
        assert_eq!(
            maya2_voice_choice(None, None, None, None).unwrap(),
            Maya2VoiceChoice::Named(MAYA2_DEFAULT_VOICE)
        );
        assert!(maya2_voice_choice(None, None, None, Some("nobody")).is_err());
        // A reference needs its transcript; its language comes from the
        // request or the transcript's script.
        assert!(maya2_voice_choice(Some("UklGRg=="), None, None, None).is_err());
        let clone =
            maya2_voice_choice(Some("UklGRg=="), Some(" नमस्ते "), None, Some("ahana")).unwrap();
        assert_eq!(
            clone,
            Maya2VoiceChoice::Clone {
                audio: "UklGRg==".into(),
                transcript: "नमस्ते".into(),
                language: "hi",
            }
        );
        let marathi =
            maya2_voice_choice(Some("UklGRg=="), Some("नमस्कार"), Some("mr"), None).unwrap();
        assert!(matches!(
            marathi,
            Maya2VoiceChoice::Clone { language: "mr", .. }
        ));
        assert!(maya2_voice_choice(Some("UklGRg=="), Some("hi"), Some("fr"), None).is_err());
    }

    #[test]
    fn maya2_params_keep_upstream_defaults() {
        let mut params = GenerationParams {
            max_tokens: ModelVariant::MAYA2_MAX_OUTPUT_FRAMES,
            ..GenerationParams::default()
        };
        let auto = maya2_tts_request_params(&params, "request-1").unwrap();
        assert_eq!(auto.sampling, Maya2SamplingParams::default());
        assert_eq!(auto.max_frames, None, "the ceiling is no caller limit");
        assert!(auto.seed > 0);
        params.max_tokens = 300;
        params.speech.temperature = Some(0.7);
        params.speech.repetition_penalty = Some(0.0);
        params.speech.seed = Some(42);
        let explicit = maya2_tts_request_params(&params, "request-1").unwrap();
        assert_eq!(explicit.max_frames, Some(300));
        assert_eq!(explicit.sampling.temperature, 0.7);
        assert_eq!(explicit.sampling.presence_penalty, 0.0);
        assert_eq!(explicit.seed, 42);
        params.speech.temperature = Some(0.0);
        assert!(maya2_tts_request_params(&params, "request-1").is_err());
    }
}
