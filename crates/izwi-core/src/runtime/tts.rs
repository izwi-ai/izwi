//! Text-to-speech runtime methods.

use std::time::Instant;

use tokio::sync::mpsc;
use tracing::info;

use crate::backends::BackendKind;
use crate::catalog::ModelFamily;
use crate::engine::{
    tts_explicit_output_limit, GenerationParams as CoreGenParams, ResourceAmount, ResourceVector,
    WorkUnit,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::kokoro::{kokoro_output_budget, kokoro_peak_workspace};
use crate::models::architectures::lfm25_audio::lfm25_audio_tts_system_prompt;
use crate::models::architectures::qwen3::tts::qwen_tts_cuda_chunked_codec_stream_enabled;
use crate::models::architectures::vibevoice::tts::{
    vibevoice_tts_auto_max_frames_for_text, VibeVoiceSpeakerReference, VibeVoiceTtsGenerationParams,
};
use crate::models::architectures::voxtral::tts::{
    voxtral_tts_auto_max_frames_for_text, VoxtralTtsGenerationParams,
};
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::runtime::adapters::{CapabilityKind, ExecutionTargetKind};
use crate::runtime::audio_io::decode_reference_audio_base64;
use crate::runtime::coordinator::{JobLease, JobResourceObservation};
use crate::runtime::request::TtsRuntimeRequest;
use crate::runtime::service::RuntimeService;
use crate::runtime::telemetry::{
    RuntimeObservationContext, RuntimeStageObservation, RuntimeStageOutcome,
    RuntimeStageOutputCounters, RuntimeStageTiming,
};
use crate::runtime::types::{
    AudioChunk, ChunkStats, GenerationConfig, GenerationRequest, GenerationResult,
};
use crate::runtime::CoordinatorLane;

const LFM25_AUDIO_DEFAULT_MAX_NEW_TOKENS: usize = 1024;
const DIRECT_TTS_STATE_BYTES_PER_UNIT: u64 = 64 * 1024;
// Reference decoding may simultaneously retain the base64-decoded source, a
// Symphonia-owned source copy, mono f32 output, and a model preprocessing copy.
const DIRECT_TTS_REFERENCE_HOST_PEAK_BYTES: u64 = 128 * 1024 * 1024;
// Reference encoders also materialize tensors on the selected accelerator.
const DIRECT_TTS_REFERENCE_ACCELERATOR_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectTtsGenerationShape {
    units: u64,
    max_output_samples: u64,
    has_reference_decode: bool,
    kokoro_max_chunk_expanded_frames: Option<u64>,
}

fn direct_tts_generation_shape(
    backend: BackendKind,
    request: &GenerationRequest,
    variant: ModelVariant,
    max_sequence_length: usize,
) -> Result<DirectTtsGenerationShape> {
    let explicit = request.config.options.max_tokens;
    if matches!(variant.family(), ModelFamily::KokoroTts) {
        let budget = kokoro_output_budget(&request.text, request.config.options.speed)?;
        return Ok(DirectTtsGenerationShape {
            units: u64::try_from(budget.max_model_tokens).map_err(|_| {
                Error::InvalidInput("Kokoro model-token budget exceeds u64".to_string())
            })?,
            max_output_samples: u64::try_from(budget.max_samples).map_err(|_| {
                Error::InvalidInput("Kokoro output-sample budget exceeds u64".to_string())
            })?,
            has_reference_decode: false,
            kokoro_max_chunk_expanded_frames: Some(
                u64::try_from(budget.max_chunk_expanded_frames).map_err(|_| {
                    Error::InvalidInput(
                        "Kokoro expanded-frame workspace budget exceeds u64".to_string(),
                    )
                })?,
            ),
        });
    }
    let (units, output_samples_per_unit, has_reference_decode) = match variant.family() {
        ModelFamily::KokoroTts => unreachable!("Kokoro shape handled above"),
        ModelFamily::Lfm25Audio => (
            if explicit == 0 {
                LFM25_AUDIO_DEFAULT_MAX_NEW_TOKENS.min(max_sequence_length.max(1))
            } else {
                explicit.min(tts_explicit_output_limit(
                    backend,
                    variant,
                    max_sequence_length,
                ))
            },
            1_920,
            false,
        ),
        ModelFamily::VoxtralTts => (
            if explicit == 0 {
                voxtral_tts_auto_max_frames_for_text(&request.text)
            } else {
                explicit.min(tts_explicit_output_limit(
                    backend,
                    variant,
                    max_sequence_length,
                ))
            },
            1_920,
            false,
        ),
        ModelFamily::VibeVoiceTts => (
            if explicit == 0 {
                vibevoice_tts_auto_max_frames_for_text(&request.text)
            } else {
                explicit.min(ModelVariant::VIBEVOICE_TTS_MAX_OUTPUT_FRAMES)
            },
            3_200,
            true,
        ),
        ModelFamily::FishS2Tts => (
            if explicit == 0 {
                ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES
            } else {
                explicit.min(tts_explicit_output_limit(
                    backend,
                    variant,
                    max_sequence_length,
                ))
            },
            2_048,
            true,
        ),
        _ => {
            return Err(Error::InvalidInput(format!(
                "model {variant} does not use the direct TTS runtime"
            )))
        }
    };
    let units = u64::try_from(units)
        .map_err(|_| Error::InvalidInput("TTS generation unit count exceeds u64".into()))?;
    let output_samples_per_unit = u64::try_from(output_samples_per_unit)
        .map_err(|_| Error::InvalidInput("TTS output shape exceeds u64".into()))?;
    Ok(DirectTtsGenerationShape {
        units,
        max_output_samples: units
            .checked_mul(output_samples_per_unit)
            .ok_or_else(|| Error::Overloaded("TTS output shape overflowed".to_string()))?,
        has_reference_decode,
        kokoro_max_chunk_expanded_frames: None,
    })
}

pub(super) fn direct_tts_physical_resources(
    backend: BackendKind,
    host_bytes: u64,
    cpu_tensor_bytes: u64,
    accelerator_tensor_bytes: u64,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes =
                ResourceAmount::Known(host_bytes.checked_add(cpu_tensor_bytes).ok_or_else(
                    || Error::Overloaded("TTS CPU reservation overflowed".to_string()),
                )?);
        }
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(
                host_bytes
                    .checked_add(accelerator_tensor_bytes)
                    .ok_or_else(|| {
                        Error::Overloaded("TTS Metal reservation overflowed".to_string())
                    })?,
            );
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host_bytes);
            resources.device_bytes = ResourceAmount::Known(accelerator_tensor_bytes);
        }
    }
    Ok(resources)
}

fn direct_tts_additional_resources(
    backend: BackendKind,
    request: &GenerationRequest,
    variant: ModelVariant,
    max_sequence_length: usize,
) -> Result<ResourceVector> {
    let max_text_chars = max_sequence_length
        .max(1)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidInput("TTS text length limit overflowed".to_string()))?;
    for (label, value) in [
        ("input", Some(request.text.as_str())),
        ("reference_text", request.reference_text.as_deref()),
        ("language", request.language.as_deref()),
        ("voice_description", request.voice_description.as_deref()),
        ("speaker", request.config.options.speaker.as_deref()),
        ("voice", request.config.options.voice.as_deref()),
    ] {
        let Some(value) = value else {
            continue;
        };
        let chars = value.chars().count();
        if chars > max_text_chars {
            return Err(Error::InvalidInput(format!(
                "TTS {label} contains {chars} characters, exceeding the {max_text_chars}-character runtime limit"
            )));
        }
    }

    let shape = direct_tts_generation_shape(backend, request, variant, max_sequence_length)?;
    let output_bytes = shape
        .max_output_samples
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| Error::Overloaded("TTS output reservation overflowed".to_string()))?;
    let (state_host_bytes, cpu_tensor_bytes, accelerator_tensor_bytes) =
        if let Some(max_chunk_expanded_frames) = shape.kokoro_max_chunk_expanded_frames {
            let workspace = kokoro_peak_workspace(max_chunk_expanded_frames)?;
            (
                workspace.host_bytes,
                workspace.cpu_tensor_bytes,
                workspace.accelerator_tensor_bytes,
            )
        } else {
            let state_bytes = shape
                .units
                .checked_mul(DIRECT_TTS_STATE_BYTES_PER_UNIT)
                .ok_or_else(|| {
                    Error::Overloaded("TTS session reservation overflowed".to_string())
                })?;
            (0, state_bytes, state_bytes)
        };
    let (reference_host_bytes, reference_accelerator_bytes) =
        if shape.has_reference_decode && request.reference_audio.is_some() {
            (
                DIRECT_TTS_REFERENCE_HOST_PEAK_BYTES,
                DIRECT_TTS_REFERENCE_ACCELERATOR_BYTES,
            )
        } else {
            (0, 0)
        };
    let host_bytes = output_bytes
        .checked_add(reference_host_bytes)
        .and_then(|value| value.checked_add(state_host_bytes))
        .ok_or_else(|| Error::Overloaded("TTS host reservation overflowed".to_string()))?;
    let cpu_tensor_bytes = cpu_tensor_bytes
        .checked_add(reference_accelerator_bytes)
        .ok_or_else(|| Error::Overloaded("TTS CPU tensor reservation overflowed".to_string()))?;
    let accelerator_tensor_bytes = accelerator_tensor_bytes
        .checked_add(reference_accelerator_bytes)
        .ok_or_else(|| Error::Overloaded("TTS accelerator reservation overflowed".to_string()))?;
    direct_tts_physical_resources(
        backend,
        host_bytes,
        cpu_tensor_bytes,
        accelerator_tensor_bytes,
    )
}

#[derive(Debug, Clone)]
struct DirectTtsObservationContext {
    request_id: String,
    correlation_id: Option<String>,
    model_variant: ModelVariant,
    workload_class: String,
    admission_ms: Option<f64>,
    streaming: bool,
    started: Instant,
}

impl DirectTtsObservationContext {
    fn new(request: &GenerationRequest, model_variant: ModelVariant, streaming: bool) -> Self {
        Self {
            request_id: request.id.clone(),
            correlation_id: request.correlation_id.clone(),
            model_variant,
            workload_class: request.runtime_context.workload_class.as_str().to_string(),
            admission_ms: request.runtime_context.admission_ms,
            streaming,
            started: Instant::now(),
        }
    }
}

fn uses_direct_tts_runtime(variant: ModelVariant) -> bool {
    matches!(variant.family(), ModelFamily::VoxtralTts)
}

fn uses_direct_streaming_tts_runtime(variant: ModelVariant) -> bool {
    uses_direct_tts_runtime(variant) || matches!(variant.family(), ModelFamily::KokoroTts)
}

pub(super) fn direct_tts_retained_input_bytes(request: &GenerationRequest) -> Result<usize> {
    fn add(total: &mut usize, bytes: usize) -> Result<()> {
        *total = total.checked_add(bytes).ok_or_else(|| {
            Error::Overloaded("TTS retained input storage overflowed".to_string())
        })?;
        Ok(())
    }

    let mut total = 0usize;
    for value in [
        Some(&request.id),
        request.correlation_id.as_ref(),
        Some(&request.text),
        request.language.as_ref(),
        request.reference_audio.as_ref(),
        request.reference_text.as_ref(),
        request.voice_description.as_ref(),
        request.config.options.speaker.as_ref(),
        request.config.options.voice.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        add(&mut total, value.capacity())?;
    }
    add(
        &mut total,
        request
            .config
            .options
            .stop_sequences
            .capacity()
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(|| Error::Overloaded("TTS stop-sequence storage overflowed".to_string()))?,
    )?;
    for stop in &request.config.options.stop_sequences {
        add(&mut total, stop.capacity())?;
    }
    add(
        &mut total,
        request
            .config
            .options
            .stop_token_ids
            .capacity()
            .checked_mul(std::mem::size_of::<crate::engine::TokenId>())
            .ok_or_else(|| Error::Overloaded("TTS stop-token storage overflowed".to_string()))?,
    )?;
    Ok(total)
}

fn vibevoice_reference_from_request(
    request: &GenerationRequest,
) -> Result<VibeVoiceSpeakerReference> {
    let ref_audio = request.reference_audio.as_deref().ok_or_else(|| {
        Error::InvalidInput(
            "VibeVoice TTS requires `reference_audio` and `reference_text`".to_string(),
        )
    })?;
    let ref_text = request.reference_text.as_deref().ok_or_else(|| {
        Error::InvalidInput(
            "VibeVoice TTS requires `reference_audio` and `reference_text`".to_string(),
        )
    })?;
    if ref_text.trim().is_empty() {
        return Err(Error::InvalidInput(
            "VibeVoice TTS reference_text cannot be empty".to_string(),
        ));
    }
    let (audio_samples, sample_rate) = decode_reference_audio_base64(ref_audio)?;
    Ok(VibeVoiceSpeakerReference {
        audio_samples,
        sample_rate,
        text: ref_text.to_string(),
    })
}

fn qwen_tts_streaming_uses_final_only(
    is_cuda: bool,
    variant: ModelVariant,
    chunked_codec_stream_enabled: bool,
) -> bool {
    matches!(variant.family(), ModelFamily::Qwen3Tts) && !(is_cuda && chunked_codec_stream_enabled)
}

fn lfm25_audio_prompt_messages(text: &str, speaker: Option<&str>) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: lfm25_audio_tts_system_prompt(speaker).to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: text.trim().to_string(),
        },
    ]
}

async fn send_direct_tts_chunk(
    job: &JobLease,
    chunk_tx: &mpsc::Sender<AudioChunk>,
    chunk: AudioChunk,
) -> Result<()> {
    let send = chunk_tx.send(chunk);
    match job.spec.deadline {
        Some(deadline) => tokio::time::timeout_at(deadline.into(), send)
            .await
            .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string())),
        None => send
            .await
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string())),
    }
}

impl RuntimeService {
    fn record_direct_tts_observation(
        &self,
        context: DirectTtsObservationContext,
        result: std::result::Result<Option<&GenerationResult>, &Error>,
    ) {
        let backend = self.backend_context();
        let outcome = if result.is_ok() {
            RuntimeStageOutcome::Completed
        } else {
            RuntimeStageOutcome::Failed
        };
        let mut observation = RuntimeStageObservation::new(
            RuntimeObservationContext {
                route_source: Some("direct_model".to_string()),
                capability: Some("tts".to_string()),
                model_variant: Some(context.model_variant.dir_name().to_string()),
                backend_kind: Some(backend.backend_kind.as_str().to_string()),
                pipeline_stage: Some(if context.streaming {
                    "tts.direct.streaming".to_string()
                } else {
                    "tts.direct.request".to_string()
                }),
                workload_class: Some(context.workload_class),
                request_id: Some(context.request_id),
                correlation_id: context.correlation_id,
                ..RuntimeObservationContext::default()
            },
            outcome,
        );
        observation.timing = RuntimeStageTiming {
            admission_ms: context.admission_ms,
            total_ms: Some(
                result
                    .as_ref()
                    .ok()
                    .and_then(|generation| generation.as_ref())
                    .map(|generation| f64::from(generation.total_time_ms))
                    .unwrap_or_else(|| context.started.elapsed().as_secs_f64() * 1000.0),
            ),
            ..RuntimeStageTiming::default()
        };
        match result {
            Ok(Some(generation)) => {
                observation.outputs = RuntimeStageOutputCounters {
                    generated_tokens: Some(generation.total_tokens as u64),
                    audio_samples: Some(generation.samples.len() as u64),
                    ..RuntimeStageOutputCounters::default()
                };
            }
            Ok(None) => {}
            Err(err) => observation.error_kind = Some(err.to_string()),
        }
        self.record_stage_observation(observation);
    }

    async fn resolve_tts_variant_for_request(
        &self,
        request: &GenerationRequest,
    ) -> Result<ModelVariant> {
        request
            .model_variant
            .or(*self.loaded_tts_variant.read().await)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))
            .and_then(|variant| {
                self.adapter_registry
                    .require(CapabilityKind::Tts, variant)
                    .map(|_| variant)
            })
    }

    async fn lfm25_audio_tts_generate(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        variant: ModelVariant,
        streaming_required: bool,
    ) -> Result<GenerationResult> {
        self.observe_broker_capability_request(
            CapabilityKind::Tts,
            Some(variant),
            streaming_required,
        )?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                job,
                variant,
                CapabilityKind::Tts,
                streaming_required,
                ExecutionTargetKind::DirectModel,
            )
            .await?;

        let text = request.text.trim().to_string();
        if text.is_empty() {
            return Err(Error::InvalidInput("TTS request missing text".to_string()));
        }

        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| Error::InferenceError("No LFM2.5 Audio model loaded".to_string()))?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .unwrap_or_else(|| self.config.portable_context_ceiling());
        let max_new_tokens = if request.config.options.max_tokens == 0 {
            LFM25_AUDIO_DEFAULT_MAX_NEW_TOKENS.min(context_limit.max(1))
        } else {
            request
                .config
                .options
                .max_tokens
                .min(tts_explicit_output_limit(
                    self.backend_router.context().backend_kind,
                    variant,
                    context_limit,
                ))
        };
        let requested_speaker = request
            .config
            .options
            .speaker
            .clone()
            .or_else(|| request.config.options.voice.clone());
        let request_id = request.id;
        self.coordinator
            .run_loaded_blocking_stage_with_invocation_workspace(
                job,
                execution_contract,
                state_binding,
                WorkUnit::AtomicJob {
                    kind: CapabilityKind::Tts.as_str().to_string(),
                },
                move |leases| {
                    let _residency_lease = residency_lease;
                    let started = Instant::now();
                    let output = model
                        .generate_sequential_with_callback_from_invocation_workspace(
                            &lfm25_audio_prompt_messages(&text, requested_speaker.as_deref()),
                            max_new_tokens,
                            leases,
                            &mut |_delta| {},
                        )?;
                    let total_time_ms = started.elapsed().as_secs_f32() * 1000.0;

                    Ok(GenerationResult {
                        request_id,
                        samples: output.samples,
                        sample_rate: output.sample_rate,
                        total_tokens: output.tokens_generated,
                        total_time_ms,
                        diagnostics: output.diagnostics,
                    })
                },
            )
            .await
    }

    async fn lfm25_audio_tts_generate_streaming(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        variant: ModelVariant,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self
            .lfm25_audio_tts_generate(job, request, variant, true)
            .await?;
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples)
            .with_sample_rate(result.sample_rate);
        chunk.is_final = true;
        send_direct_tts_chunk(job, &chunk_tx, chunk).await
    }

    async fn voxtral_tts_generate(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        variant: ModelVariant,
        streaming_required: bool,
    ) -> Result<GenerationResult> {
        self.observe_broker_capability_request(
            CapabilityKind::Tts,
            Some(variant),
            streaming_required,
        )?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                job,
                variant,
                CapabilityKind::Tts,
                streaming_required,
                ExecutionTargetKind::DirectModel,
            )
            .await?;

        let text = request.text.trim().to_string();
        if text.is_empty() {
            return Err(Error::InvalidInput("TTS request missing text".to_string()));
        }

        let model = self
            .model_registry
            .get_voxtral_tts(variant)
            .await
            .ok_or_else(|| Error::InferenceError("No Voxtral TTS model loaded".to_string()))?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .unwrap_or_else(|| self.config.portable_context_ceiling());
        let explicit_max_frames = tts_explicit_output_limit(
            self.backend_router.context().backend_kind,
            variant,
            context_limit,
        );
        let request_id = request.id;
        let config = request.config;
        self.coordinator
            .run_loaded_blocking_stage_with_invocation_paged(
                job,
                execution_contract,
                state_binding,
                WorkUnit::AtomicJob {
                    kind: CapabilityKind::Tts.as_str().to_string(),
                },
                move |leases| {
                    let _residency_lease = residency_lease;
                    let voice = config
                        .options
                        .speaker
                        .clone()
                        .or_else(|| config.options.voice.clone())
                        .or_else(|| model.available_speakers().into_iter().next())
                        .ok_or_else(|| {
                            Error::InferenceError(
                                "Voxtral TTS model exposes no preset voices".to_string(),
                            )
                        })?;
                    let params =
                        VoxtralTtsGenerationParams::from_generation_config_for_text_with_limit(
                            &config,
                            &text,
                            explicit_max_frames,
                        );
                    let started = Instant::now();
                    let domains = leases.domains().collect::<Vec<_>>();
                    let [domain] = domains.as_slice() else {
                        return Err(Error::InferenceError(format!(
                            "Voxtral TTS requires one invocation KV domain, found {}",
                            domains.len()
                        )));
                    };
                    let output = model.generate_with_voice_physical(
                        &text,
                        &voice,
                        params,
                        leases.cache_mut(*domain)?,
                    )?;
                    let total_time_ms = started.elapsed().as_secs_f32() * 1000.0;
                    let sample_rate = u32::try_from(output.sample_rate).map_err(|_| {
                        Error::InferenceError(format!(
                            "Voxtral TTS sample rate {} exceeds u32",
                            output.sample_rate
                        ))
                    })?;

                    Ok(GenerationResult {
                        request_id,
                        samples: output.samples,
                        sample_rate,
                        total_tokens: output.frames_generated,
                        total_time_ms,
                        diagnostics: None,
                    })
                },
            )
            .await
    }

    async fn voxtral_tts_generate_streaming(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        variant: ModelVariant,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self
            .voxtral_tts_generate(job, request, variant, true)
            .await?;
        let generation_time_ms = result.total_time_ms;
        let tokens_generated = result.total_tokens;
        let rtf = result.rtf();
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples)
            .with_sample_rate(result.sample_rate);
        chunk.stats = Some(ChunkStats {
            generation_time_ms,
            tokens_generated,
            rtf,
        });
        send_direct_tts_chunk(job, &chunk_tx, chunk).await
    }

    async fn vibevoice_tts_generate(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        variant: ModelVariant,
        streaming_required: bool,
    ) -> Result<GenerationResult> {
        self.observe_broker_capability_request(
            CapabilityKind::Tts,
            Some(variant),
            streaming_required,
        )?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                job,
                variant,
                CapabilityKind::Tts,
                streaming_required,
                ExecutionTargetKind::DirectModel,
            )
            .await?;

        let text = request.text.trim().to_string();
        if text.is_empty() {
            return Err(Error::InvalidInput("TTS request missing text".to_string()));
        }

        let model = self
            .model_registry
            .get_vibevoice_tts(variant)
            .await
            .ok_or_else(|| Error::InferenceError("No VibeVoice TTS model loaded".to_string()))?;
        self.coordinator
            .run_loaded_blocking_stage_with_invocation_workspace(
                job,
                execution_contract,
                state_binding,
                WorkUnit::AtomicJob {
                    kind: CapabilityKind::Tts.as_str().to_string(),
                },
                move |leases| {
                    let _residency_lease = residency_lease;
                    let reference = vibevoice_reference_from_request(&request)?;
                    let requested_speaker = request.config.options.speaker.as_deref().or(request
                        .config
                        .options
                        .voice
                        .as_deref());
                    let params = VibeVoiceTtsGenerationParams::from_generation_config_for_text(
                        &request.config,
                        &text,
                        model.default_diffusion_steps(),
                    );
                    let started = Instant::now();
                    let output = model.generate_with_reference_physical(
                        &text,
                        &reference,
                        requested_speaker,
                        params,
                        leases,
                    )?;
                    let total_time_ms = started.elapsed().as_secs_f32() * 1000.0;

                    Ok(GenerationResult {
                        request_id: request.id,
                        samples: output.samples,
                        sample_rate: output.sample_rate,
                        total_tokens: output.frames_generated,
                        total_time_ms,
                        diagnostics: None,
                    })
                },
            )
            .await
    }

    async fn vibevoice_tts_generate_streaming(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        variant: ModelVariant,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self
            .vibevoice_tts_generate(job, request, variant, true)
            .await?;
        let generation_time_ms = result.total_time_ms;
        let tokens_generated = result.total_tokens;
        let rtf = result.rtf();
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples)
            .with_sample_rate(result.sample_rate);
        chunk.stats = Some(ChunkStats {
            generation_time_ms,
            tokens_generated,
            rtf,
        });
        send_direct_tts_chunk(job, &chunk_tx, chunk).await
    }

    async fn qwen_tts_final_only_streaming(
        &self,
        mut request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let deadline = request.runtime_context.deadline;
        let request_id = request.id.clone();
        request.config.streaming = false;
        let result = self.generate(request).await?;
        let generation_time_ms = result.total_time_ms;
        let tokens_generated = result.total_tokens;
        let rtf = result.rtf();
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples)
            .with_sample_rate(result.sample_rate);
        chunk.stats = Some(ChunkStats {
            generation_time_ms,
            tokens_generated,
            rtf,
        });
        let send = chunk_tx.send(chunk);
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), send)
                .await
                .map_err(|_| Error::Timeout(request_id))?
                .map_err(|_| {
                    Error::InferenceError("Streaming output channel closed".to_string())
                })?,
            None => send.await.map_err(|_| {
                Error::InferenceError("Streaming output channel closed".to_string())
            })?,
        }

        info!(
            "Qwen3-TTS streaming emitted final-only audio in {:.1}ms (RTF {:.3}); enable IZWI_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM=1 for experimental progressive CUDA codec streaming",
            generation_time_ms, rtf
        );
        Ok(())
    }
}

impl RuntimeService {
    /// Generate audio from text using the unified core engine.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let resolved_variant = self.resolve_tts_variant_for_request(&request).await?;
        if uses_direct_tts_runtime(resolved_variant) {
            let observation = DirectTtsObservationContext::new(&request, resolved_variant, false);
            let retained_input_bytes = direct_tts_retained_input_bytes(&request)?;
            let observed_input_bytes = u64::try_from(retained_input_bytes).map_err(|_| {
                Error::InvalidInput("TTS retained input size exceeds u64".to_string())
            })?;
            let mut spec = self.coordinator_job_for_input(
                request.id.clone(),
                CoordinatorLane::Atomic,
                request.runtime_context,
                retained_input_bytes,
            );
            spec.resources = spec.resources.checked_add(direct_tts_additional_resources(
                self.backend_router.context().backend_kind,
                &request,
                resolved_variant,
                self.config.portable_context_ceiling(),
            )?)?;
            let job = self
                .coordinator
                .admit_observed(spec, JobResourceObservation::host(observed_input_bytes))
                .await?;
            let result = match resolved_variant.family() {
                ModelFamily::KokoroTts => self.kokoro_tts_generate(&job, request).await,
                ModelFamily::Lfm25Audio => {
                    self.lfm25_audio_tts_generate(&job, request, resolved_variant, false)
                        .await
                }
                ModelFamily::VoxtralTts => {
                    self.voxtral_tts_generate(&job, request, resolved_variant, false)
                        .await
                }
                ModelFamily::VibeVoiceTts => {
                    self.vibevoice_tts_generate(&job, request, resolved_variant, false)
                        .await
                }
                _ => unreachable!("direct TTS family checked above"),
            };
            self.record_direct_tts_observation(observation, result.as_ref().map(Some));
            return result;
        }
        let core_params = core_params_from_generation(&request.config);
        let core_request = TtsRuntimeRequest::from_generation(request, resolved_variant)?
            .into_engine_request(core_params);

        let output = self.run_request(core_request).await?;
        let samples = output.audio.samples;
        let sample_rate = output.audio.sample_rate;
        let total_tokens = output.num_tokens;
        let total_time_ms = output.generation_time.as_secs_f32() * 1000.0;

        info!(
            "Generated {} samples in {:.1}ms via core engine",
            samples.len(),
            total_time_ms
        );

        Ok(GenerationResult {
            request_id: output.request_id,
            samples,
            sample_rate,
            total_tokens,
            total_time_ms,
            diagnostics: None,
        })
    }

    /// Generate audio with streaming output.
    ///
    /// Streaming is emitted from engine outputs in chunked form so all synthesis
    /// execution still routes through the core engine.
    pub async fn generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let resolved_variant = self.resolve_tts_variant_for_request(&request).await?;
        if uses_direct_streaming_tts_runtime(resolved_variant) {
            let observation = DirectTtsObservationContext::new(&request, resolved_variant, true);
            let retained_input_bytes = direct_tts_retained_input_bytes(&request)?;
            let observed_input_bytes = u64::try_from(retained_input_bytes).map_err(|_| {
                Error::InvalidInput("TTS retained input size exceeds u64".to_string())
            })?;
            let mut spec = self.coordinator_job_for_input(
                request.id.clone(),
                CoordinatorLane::Atomic,
                request.runtime_context,
                retained_input_bytes,
            );
            spec.resources = spec.resources.checked_add(direct_tts_additional_resources(
                self.backend_router.context().backend_kind,
                &request,
                resolved_variant,
                self.config.portable_context_ceiling(),
            )?)?;
            let job = self
                .coordinator
                .admit_observed(spec, JobResourceObservation::host(observed_input_bytes))
                .await?;
            let result = match resolved_variant.family() {
                ModelFamily::KokoroTts => {
                    self.kokoro_tts_generate_streaming(&job, request, chunk_tx)
                        .await
                }
                ModelFamily::Lfm25Audio => {
                    self.lfm25_audio_tts_generate_streaming(
                        &job,
                        request,
                        resolved_variant,
                        chunk_tx,
                    )
                    .await
                }
                ModelFamily::VoxtralTts => {
                    self.voxtral_tts_generate_streaming(&job, request, resolved_variant, chunk_tx)
                        .await
                }
                ModelFamily::VibeVoiceTts => {
                    self.vibevoice_tts_generate_streaming(&job, request, resolved_variant, chunk_tx)
                        .await
                }
                _ => unreachable!("direct TTS family checked above"),
            };
            self.record_direct_tts_observation(observation, result.as_ref().map(|_| None));
            return result;
        }
        if qwen_tts_streaming_uses_final_only(
            self.device.kind.is_cuda(),
            resolved_variant,
            qwen_tts_cuda_chunked_codec_stream_enabled(),
        ) {
            return self.qwen_tts_final_only_streaming(request, chunk_tx).await;
        }
        let core_params = core_params_from_generation(&request.config);
        let core_request = TtsRuntimeRequest::from_generation(request, resolved_variant)?
            .into_engine_request(core_params);

        self.run_streaming_request(core_request, |stream_chunk| {
            let tx = chunk_tx.clone();
            async move {
                if stream_chunk.samples.is_empty() && !stream_chunk.is_final {
                    return Ok(());
                }

                let mut chunk = AudioChunk::new(
                    stream_chunk.request_id.clone(),
                    stream_chunk.sequence,
                    stream_chunk.samples,
                )
                .with_sample_rate(stream_chunk.sample_rate);
                chunk.is_final = stream_chunk.is_final;
                tx.send(chunk).await.map_err(|_| {
                    Error::InferenceError("Streaming output channel closed".to_string())
                })?;
                Ok(())
            }
        })
        .await?;

        info!("Streaming generation complete via core engine");
        Ok(())
    }
}

fn core_params_from_generation(config: &GenerationConfig) -> CoreGenParams {
    let mut params = config.options.clone();
    if params.voice.is_none() {
        params.voice = params.speaker.clone();
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fish_s2_public_tts_routes_always_use_retained_execution() {
        assert!(!uses_direct_tts_runtime(ModelVariant::FishAudioS2Pro));
        assert!(!uses_direct_streaming_tts_runtime(
            ModelVariant::FishAudioS2Pro
        ));
    }

    #[test]
    fn kokoro_uses_engine_for_atomic_generation_and_direct_native_streaming() {
        assert!(!uses_direct_tts_runtime(ModelVariant::Kokoro82M));
        assert!(uses_direct_streaming_tts_runtime(ModelVariant::Kokoro82M));
    }
    use crate::backends::BackendKind;
    use crate::engine::{Priority, ResourceAmount, ResourceVector, WorkloadClass};
    use crate::runtime::coordinator::{InferenceCoordinator, JobSpec};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn qwen_tts_cuda_streaming_defaults_to_final_only_without_chunked_codec() {
        assert!(qwen_tts_streaming_uses_final_only(
            true,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            false,
        ));
    }

    #[test]
    fn qwen_tts_cuda_streaming_respects_chunked_codec_opt_in() {
        assert!(!qwen_tts_streaming_uses_final_only(
            true,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            true,
        ));
    }

    #[test]
    fn qwen_tts_non_cuda_streaming_defaults_to_final_only() {
        assert!(qwen_tts_streaming_uses_final_only(
            false,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            false,
        ));
    }

    #[test]
    fn qwen_tts_final_only_streaming_policy_does_not_affect_non_qwen() {
        assert!(!qwen_tts_streaming_uses_final_only(
            true,
            ModelVariant::Kokoro82M,
            false,
        ));
    }

    #[test]
    fn direct_tts_retained_input_includes_reference_and_config_strings() {
        let mut request = GenerationRequest::new("speak");
        let baseline = direct_tts_retained_input_bytes(&request).unwrap();
        let mut reference_audio = String::with_capacity(8 * 1024);
        reference_audio.push_str("audio");
        let mut reference_text = String::with_capacity(1024);
        reference_text.push_str("transcript");
        let mut language = String::with_capacity(32);
        language.push_str("en");
        let mut speaker = String::with_capacity(64);
        speaker.push_str("speaker-1");
        let expected_growth = reference_audio.capacity()
            + reference_text.capacity()
            + language.capacity()
            + speaker.capacity();
        request.reference_audio = Some(reference_audio);
        request.reference_text = Some(reference_text);
        request.language = Some(language);
        request.config.options.speaker = Some(speaker);

        assert_eq!(
            direct_tts_retained_input_bytes(&request).unwrap(),
            baseline + expected_growth
        );
    }

    #[test]
    fn direct_tts_reservation_is_request_shaped_and_backend_aware() {
        let mut request = GenerationRequest::new("bounded voice clone");
        request.reference_audio = Some("AAAA".to_string());
        request.config.options.max_tokens = 100;
        let output_bytes = 100 * 3_200 * std::mem::size_of::<f32>() as u64;
        let state_bytes = 100 * DIRECT_TTS_STATE_BYTES_PER_UNIT;
        let host_bytes = output_bytes + DIRECT_TTS_REFERENCE_HOST_PEAK_BYTES;
        let accelerator_bytes = state_bytes + DIRECT_TTS_REFERENCE_ACCELERATOR_BYTES;

        let cpu = direct_tts_additional_resources(
            BackendKind::Cpu,
            &request,
            ModelVariant::VibeVoice15BTts,
            4096,
        )
        .unwrap();
        let metal = direct_tts_additional_resources(
            BackendKind::Metal,
            &request,
            ModelVariant::VibeVoice15BTts,
            4096,
        )
        .unwrap();
        let cuda = direct_tts_additional_resources(
            BackendKind::Cuda,
            &request,
            ModelVariant::VibeVoice15BTts,
            4096,
        )
        .unwrap();

        assert_eq!(
            cpu.host_bytes,
            ResourceAmount::Known(host_bytes + accelerator_bytes)
        );
        assert_eq!(
            metal.unified_bytes,
            ResourceAmount::Known(host_bytes + accelerator_bytes)
        );
        assert_eq!(cuda.host_bytes, ResourceAmount::Known(host_bytes));
        assert_eq!(cuda.device_bytes, ResourceAmount::Known(accelerator_bytes));
    }

    #[test]
    fn direct_lfm_tts_caps_untrusted_output_budget_to_runtime_sequence_limit() {
        let mut request = GenerationRequest::new("bounded LFM output");
        request.config.options.max_tokens = usize::MAX;
        let shape = direct_tts_generation_shape(
            BackendKind::Cpu,
            &request,
            ModelVariant::Lfm25Audio15BGguf,
            4096,
        )
        .unwrap();

        assert_eq!(shape.units, 4096);
        assert_eq!(shape.max_output_samples, 4096 * 1_920);
        assert!(!shape.has_reference_decode);
    }

    #[test]
    fn direct_cuda_tts_reserves_unlocked_explicit_context() {
        let mut request = GenerationRequest::new("long CUDA output");
        request.config.options.max_tokens = 5000;

        let lfm = direct_tts_generation_shape(
            BackendKind::Cuda,
            &request,
            ModelVariant::Lfm25Audio15BGguf,
            4096,
        )
        .unwrap();
        let fish = direct_tts_generation_shape(
            BackendKind::Cuda,
            &request,
            ModelVariant::FishAudioS2Pro,
            4096,
        )
        .unwrap();

        assert_eq!(lfm.units, 5000);
        assert_eq!(fish.units, 5000);
        assert_eq!(fish.max_output_samples, 5000 * 2048);
    }

    #[test]
    fn direct_lfm_tts_caps_default_output_budget_to_runtime_sequence_limit() {
        let request = GenerationRequest::new("bounded LFM default output");
        let shape = direct_tts_generation_shape(
            BackendKind::Cpu,
            &request,
            ModelVariant::Lfm25Audio15BGguf,
            64,
        )
        .unwrap();

        assert_eq!(shape.units, 64);
    }

    #[test]
    fn kokoro_max_text_at_minimum_speed_uses_the_hard_model_output_contract() {
        let max_sequence_length = 4096usize;
        let max_text_chars = max_sequence_length * 8;
        let mut request = GenerationRequest::new("x".repeat(max_text_chars));
        request.config.options.speed = 0.5;
        let budget = kokoro_output_budget(&request.text, request.config.options.speed).unwrap();
        let shape = direct_tts_generation_shape(
            BackendKind::Cpu,
            &request,
            ModelVariant::Kokoro82M,
            max_sequence_length,
        )
        .unwrap();

        let expected_frames = budget
            .max_model_tokens
            .checked_mul(100)
            .unwrap()
            .min(max_text_chars * 4_096);
        let expected_samples = expected_frames
            .checked_mul(600)
            .and_then(|samples| samples.checked_add((max_text_chars - 1) * 960))
            .unwrap();
        assert_eq!(budget.max_samples, expected_samples);
        assert_eq!(budget.max_chunk_expanded_frames, 4_096);
        assert_eq!(shape.units, budget.max_model_tokens as u64);
        assert_eq!(shape.max_output_samples, expected_samples as u64);
    }

    #[test]
    fn kokoro_worst_case_chunk_reserves_architecture_peak_on_each_backend() {
        // 29 input characters authorize at least 512 worst-case model tokens,
        // so the per-chunk frame ceiling, rather than total request tokens,
        // determines the live activation peak.
        let mut request = GenerationRequest::new("x".repeat(29));
        request.config.options.speed = 0.5;
        let budget = kokoro_output_budget(&request.text, request.config.options.speed).unwrap();
        assert_eq!(budget.max_chunk_expanded_frames, 4_096);

        let expanded_frames = 4_096u64;
        let generator_time_scale = 2 * 10 * 6;
        let final_stage_elements_per_frame = generator_time_scale * 128;
        let retained_elements_per_frame = generator_time_scale * 22 + 512 + 2 * 2;
        let cpu_tensor_bytes = expanded_frames
            * (final_stage_elements_per_frame * 10 + retained_elements_per_frame)
            * std::mem::size_of::<f32>() as u64;
        let accelerator_tensor_bytes = expanded_frames
            * (final_stage_elements_per_frame * 9 + retained_elements_per_frame)
            * std::mem::size_of::<f32>() as u64;
        let host_workspace_bytes =
            expanded_frames * generator_time_scale * 5 * 40 * std::mem::size_of::<f32>() as u64;
        let output_bytes = budget.max_samples as u64 * std::mem::size_of::<f32>() as u64;

        let cpu = direct_tts_additional_resources(
            BackendKind::Cpu,
            &request,
            ModelVariant::Kokoro82M,
            4_096,
        )
        .unwrap();
        let metal = direct_tts_additional_resources(
            BackendKind::Metal,
            &request,
            ModelVariant::Kokoro82M,
            4_096,
        )
        .unwrap();
        let cuda = direct_tts_additional_resources(
            BackendKind::Cuda,
            &request,
            ModelVariant::Kokoro82M,
            4_096,
        )
        .unwrap();

        assert_eq!(
            cpu.host_bytes,
            ResourceAmount::Known(output_bytes + host_workspace_bytes + cpu_tensor_bytes)
        );
        assert_eq!(
            metal.unified_bytes,
            ResourceAmount::Known(output_bytes + host_workspace_bytes + accelerator_tensor_bytes)
        );
        assert_eq!(
            cuda.host_bytes,
            ResourceAmount::Known(output_bytes + host_workspace_bytes)
        );
        assert_eq!(
            cuda.device_bytes,
            ResourceAmount::Known(accelerator_tensor_bytes)
        );
        assert!(
            output_bytes + host_workspace_bytes + cpu_tensor_bytes < 4 * 1024 * 1024 * 1024,
            "the enforced frame ceiling must keep a worst-case chunk admissible on practical hosts"
        );
    }

    #[test]
    fn direct_tts_backend_mapping_rejects_combined_domain_overflow() {
        assert!(matches!(
            direct_tts_physical_resources(BackendKind::Cpu, u64::MAX, 1, 0),
            Err(Error::Overloaded(message)) if message.contains("CPU reservation")
        ));
        assert!(matches!(
            direct_tts_physical_resources(BackendKind::Metal, u64::MAX, 0, 1),
            Err(Error::Overloaded(message)) if message.contains("Metal reservation")
        ));

        let cuda = direct_tts_physical_resources(BackendKind::Cuda, u64::MAX, 0, u64::MAX)
            .expect("CUDA domains remain independent");
        assert_eq!(cuda.host_bytes, ResourceAmount::Known(u64::MAX));
        assert_eq!(cuda.device_bytes, ResourceAmount::Known(u64::MAX));
    }

    #[test]
    fn direct_tts_rejects_text_beyond_runtime_sequence_contract() {
        let request = GenerationRequest::new("x".repeat(33));
        assert!(matches!(
            direct_tts_additional_resources(
                BackendKind::Cpu,
                &request,
                ModelVariant::Kokoro82M,
                4,
            ),
            Err(Error::InvalidInput(message)) if message.contains("32-character")
        ));
    }

    #[test]
    fn direct_tts_rejects_reference_text_beyond_runtime_sequence_contract() {
        let mut request = GenerationRequest::new("bounded target");
        request.reference_text = Some("x".repeat(33));
        assert!(matches!(
            direct_tts_additional_resources(
                BackendKind::Cpu,
                &request,
                ModelVariant::VibeVoice15BTts,
                4,
            ),
            Err(Error::InvalidInput(message)) if message.contains("reference_text") && message.contains("32-character")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn direct_stream_send_is_responsive_and_honors_the_job_deadline() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 2));
        let mut resources = ResourceVector::zero();
        resources.host_bytes = ResourceAmount::Known(1024);
        let job = coordinator
            .admit(JobSpec {
                request_id: "tts-stream-deadline".to_string(),
                lane: CoordinatorLane::Atomic,
                priority: Priority::Normal,
                workload_class: WorkloadClass::Online,
                deadline: Some(Instant::now() + Duration::from_millis(100)),
                resources,
            })
            .await
            .unwrap();
        let (chunk_tx, _chunk_rx) = mpsc::channel(1);
        chunk_tx
            .send(AudioChunk::new(
                "tts-stream-deadline".to_string(),
                0,
                vec![0.0],
            ))
            .await
            .unwrap();

        let (heartbeat_tx, heartbeat_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            let _ = heartbeat_tx.send(());
        });
        let send = send_direct_tts_chunk(
            &job,
            &chunk_tx,
            AudioChunk::new("tts-stream-deadline".to_string(), 1, vec![0.0]),
        );
        tokio::pin!(send);

        tokio::time::timeout(Duration::from_millis(50), heartbeat_rx)
            .await
            .expect("stalled bounded send must not block the Tokio worker")
            .unwrap();
        assert!(matches!(send.await, Err(Error::Timeout(_))));
    }
}
