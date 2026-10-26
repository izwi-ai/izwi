//! Text-to-speech runtime methods.

use std::time::Instant;

use tokio::sync::mpsc;
use tracing::info;

use crate::backends::BackendKind;
use crate::catalog::ModelFamily;
use crate::engine::{
    tts_explicit_output_limit, GenerationParams as CoreGenParams, ResourceAmount, ResourceVector,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::kokoro::admission::kokoro_chunk_price;
use crate::models::architectures::kokoro::kokoro_output_budget;
use crate::models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
use crate::models::architectures::voxtral::tts::voxtral_tts_auto_max_frames_for_text;
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::coordinator::JobResourceObservation;
use crate::runtime::request::TtsRuntimeRequest;
use crate::runtime::service::RuntimeService;
use crate::runtime::telemetry::{
    RuntimeObservationContext, RuntimeStageObservation, RuntimeStageOutcome,
    RuntimeStageOutputCounters, RuntimeStageTiming,
};
use crate::runtime::types::{
    AudioChunk, ChunkStats, GenerationConfig, GenerationRequest, GenerationResult, TtsStopReason,
    TtsStreamingMode,
};
use crate::runtime::CoordinatorLane;

#[cfg(test)]
use crate::runtime::adapters::ExecutionTargetKind;
#[cfg(test)]
use crate::runtime::coordinator::JobLease;

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
    /// Planned frames of one Kokoro stream chunk.
    kokoro_chunk_frames: Option<usize>,
}

fn direct_tts_generation_shape(
    backend: BackendKind,
    request: &GenerationRequest,
    variant: ModelVariant,
    max_sequence_length: usize,
) -> Result<DirectTtsGenerationShape> {
    let explicit = request.config.options.max_tokens;
    if matches!(variant.family(), ModelFamily::KokoroTts) {
        // The direct route streams: it synthesizes one planned chunk (at
        // most `KOKORO_STREAM_TARGET_CHARS`) at a time and hands its audio
        // on, so it holds one chunk's workspace and output (KOK-8).
        let budget = kokoro_output_budget(&request.text, request.config.options.speed)?;
        let frames = crate::runtime::kokoro::kokoro_stream_chunk_frames(
            &request.text,
            request.config.options.speed,
        );
        return Ok(DirectTtsGenerationShape {
            units: u64::try_from(budget.max_model_tokens).map_err(|_| {
                Error::InvalidInput("Kokoro model-token budget exceeds u64".to_string())
            })?,
            max_output_samples: kokoro_chunk_price(frames, backend)?.output_samples,
            has_reference_decode: false,
            kokoro_chunk_frames: Some(frames),
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
        kokoro_chunk_frames: None,
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
    let (state_host_bytes, cpu_tensor_bytes, accelerator_tensor_bytes) = if let Some(frames) =
        shape.kokoro_chunk_frames
    {
        let price = kokoro_chunk_price(frames, backend)?;
        (
            price.host_bytes,
            price.cpu_tensor_bytes,
            price.accelerator_tensor_bytes,
        )
    } else {
        let state_bytes = shape
            .units
            .checked_mul(DIRECT_TTS_STATE_BYTES_PER_UNIT)
            .ok_or_else(|| Error::Overloaded("TTS session reservation overflowed".to_string()))?;
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

/// Kokoro streams its planned chunks natively on the direct route; every
/// other TTS request, streaming or not, runs on the engine.
fn uses_direct_streaming_tts_runtime(variant: ModelVariant) -> bool {
    matches!(variant.family(), ModelFamily::KokoroTts)
}

/// The execution target the public TTS route actually uses for `variant`.
/// The adapter registry must advertise the same target, or no loaded
/// contract can serve the request (VXT-1).
#[cfg(test)]
pub(crate) fn tts_route_target(variant: ModelVariant, streaming: bool) -> ExecutionTargetKind {
    if streaming && uses_direct_streaming_tts_runtime(variant) {
        ExecutionTargetKind::DirectModel
    } else {
        ExecutionTargetKind::TokenEngine
    }
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

/// The streaming mode `variant` delivers (TPIPE-3). Every streaming family
/// publishes audio as it commits it: Qwen3-TTS per codec frame, LFM2.5
/// Audio per detokenized code run (LFT-5), VibeVoice per frame (VVT-10),
/// Voxtral per codec chunk, Fish per decoded frame batch and Kokoro per
/// sentence chunk.
fn tts_streaming_mode(variant: ModelVariant) -> TtsStreamingMode {
    if variant
        .speech_capabilities()
        .is_some_and(|capabilities| capabilities.supports_streaming)
    {
        TtsStreamingMode::Chunked
    } else {
        TtsStreamingMode::Unsupported
    }
}

impl RuntimeService {
    /// How a streaming request for `variant` delivers audio here (TPIPE-3).
    pub fn tts_streaming_mode(&self, variant: ModelVariant) -> TtsStreamingMode {
        tts_streaming_mode(variant)
    }
}

#[cfg(test)]
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

/// Preserve the absolute request deadline after engine cleanup, while delivering
/// terminal statistics. Dropping this future cancels the pending bounded send;
/// closing the receiver wakes it immediately without requiring an engine lease.
async fn send_terminal_tts_chunk(
    chunk_tx: &mpsc::Sender<AudioChunk>,
    chunk: AudioChunk,
    deadline: Option<Instant>,
) -> Result<()> {
    let request_id = chunk.request_id.clone();
    let sent = match deadline {
        Some(deadline) => {
            if deadline <= Instant::now() {
                return Err(Error::Timeout(request_id));
            }
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline.into()) => return Err(Error::Timeout(request_id)),
                sent = chunk_tx.send(chunk) => sent,
            }
        }
        None => chunk_tx.send(chunk).await,
    };
    sent.map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))
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
}

/// A finished request must carry audio at a real sample rate: an empty
/// result is an error, not a 200 with an empty WAV (TPIPE-18).
fn ensure_speech_audio(
    variant: ModelVariant,
    request_id: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<()> {
    if samples.is_empty() || sample_rate == 0 {
        return Err(Error::InferenceError(format!(
            "{variant} produced no audio for request {request_id}"
        )));
    }
    Ok(())
}

/// Speak numbers, currency, dates, times and units as words when the
/// request asks for it, or leaves it unset for a family that normalizes by
/// default (`batch.tts.normalize_text`, TPIPE-26). English text only.
fn normalize_speech_text(
    mut request: GenerationRequest,
    variant: ModelVariant,
) -> GenerationRequest {
    use crate::models::shared::text_normalize::{
        family_normalizes_by_default, is_english, normalize_english,
    };
    let enabled = request
        .config
        .options
        .speech
        .normalize_text
        .unwrap_or_else(|| family_normalizes_by_default(variant.family()));
    if enabled
        && is_english(
            request.language.as_deref(),
            request.config.options.speaker.as_deref(),
        )
    {
        request.text = normalize_english(&request.text);
    }
    request
}

impl RuntimeService {
    /// Generate audio from text using the unified core engine.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let resolved_variant = self.resolve_tts_variant_for_request(&request).await?;
        let request = normalize_speech_text(request, resolved_variant);
        let core_params = core_params_from_generation(&request.config);
        let core_request = TtsRuntimeRequest::from_generation(request, resolved_variant)?
            .into_engine_request(core_params);
        let budget_frames = core_request.params.max_tokens;

        let output = self.run_request(core_request).await?;
        let diagnostics = Some(serde_json::json!({
            "latency_breakdown": output.latency_breakdown,
            "timing_basis": "engine_execution",
            "token_unit": if resolved_variant == ModelVariant::FishAudioS2Pro {
                "semantic_frames"
            } else {
                "model_tokens"
            },
            "split_request_count": 1,
        }));
        let samples = output.audio.samples;
        let sample_rate = output.audio.sample_rate;
        ensure_speech_audio(resolved_variant, &output.request_id, &samples, sample_rate)?;
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
            diagnostics,
            stop_reason: tts_stop_reason(
                resolved_variant,
                budget_frames,
                &output.latency_breakdown,
                total_tokens,
            ),
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
        let request = normalize_speech_text(request, resolved_variant);
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
                _ => unreachable!("direct TTS family checked above"),
            };
            self.record_direct_tts_observation(observation, result.as_ref().map(|_| None));
            return result;
        }
        let core_params = core_params_from_generation(&request.config);
        let mut core_request = TtsRuntimeRequest::from_generation(request, resolved_variant)?
            .into_engine_request(core_params);
        if resolved_variant == ModelVariant::FishAudioS2Pro {
            core_request.audio_stream_external_queue_capacity = chunk_tx.max_capacity();
            core_request.mark_audio_streaming_only();
        }

        let terminal_deadline = core_request.deadline;
        let budget_frames = core_request.params.max_tokens;
        let mut next_sequence = 0usize;
        let mut output = self
            .run_streaming_request(core_request, |stream_chunk| {
                let tx = chunk_tx.clone();
                let sequence = next_sequence;
                if !stream_chunk.samples.is_empty() {
                    next_sequence += 1;
                }
                async move {
                    if stream_chunk.samples.is_empty() {
                        return Ok(());
                    }
                    let chunk = AudioChunk::new(
                        stream_chunk.request_id.clone(),
                        sequence,
                        stream_chunk.samples,
                    )
                    .with_sample_rate(stream_chunk.sample_rate);
                    tx.send(chunk).await.map_err(|_| {
                        Error::InferenceError("Streaming output channel closed".to_string())
                    })?;
                    Ok(())
                }
            })
            .await?;

        if let Some(samples) = unstreamed_terminal_pcm(&mut output.audio, next_sequence) {
            let chunk = AudioChunk::new(output.request_id.clone(), next_sequence, samples)
                .with_sample_rate(output.audio.sample_rate);
            next_sequence += 1;
            chunk_tx.send(chunk).await.map_err(|_| {
                Error::InferenceError("Streaming output channel closed".to_string())
            })?;
        }

        // Completion statistics are cumulative and arrive after all committed PCM,
        // including when the engine's terminal marker contains no audio.
        let mut terminal =
            AudioChunk::final_chunk(output.request_id.clone(), next_sequence, Vec::new())
                .with_sample_rate(output.audio.sample_rate);
        terminal.stats = Some(ChunkStats {
            generation_time_ms: output.generation_time.as_secs_f32() * 1000.0,
            tokens_generated: output.num_tokens,
            rtf: output.rtf(),
            stop_reason: tts_stop_reason(
                resolved_variant,
                budget_frames,
                &output.latency_breakdown,
                output.num_tokens,
            ),
        });
        send_terminal_tts_chunk(&chunk_tx, terminal, terminal_deadline).await?;
        info!("Streaming generation complete via core engine");
        Ok(())
    }
}

/// PCM a stream's terminal output holds that the stream never carried: a
/// final-only family renders its whole utterance there. A family that
/// streamed `streamed_chunks` may also keep the utterance there (Qwen3-TTS
/// does), so its terminal PCM is never sent again.
fn unstreamed_terminal_pcm(
    audio: &mut crate::engine::AudioOutput,
    streamed_chunks: usize,
) -> Option<Vec<f32>> {
    (streamed_chunks == 0 && !audio.samples.is_empty()).then(|| std::mem::take(&mut audio.samples))
}

/// Why an engine TTS request ended. The engine's decode accounting counts
/// committed audio frames (`decode_tokens`) and decode steps separately;
/// `num_tokens` (scheduler tokens) is not a frame count for every family,
/// so it is only the fallback. Only Qwen3-TTS is known to end on a decode
/// step that commits no frame, so only its step count separates end of
/// speech from a context stop.
fn tts_stop_reason(
    variant: ModelVariant,
    budget_frames: usize,
    breakdown: &Option<crate::engine::LatencyBreakdown>,
    num_tokens: usize,
) -> Option<TtsStopReason> {
    let frames = breakdown
        .as_ref()
        .and_then(|breakdown| breakdown.decode_tokens)
        .unwrap_or(num_tokens);
    let steps = breakdown
        .as_ref()
        .filter(|breakdown| {
            breakdown.decode_tokens.is_some() && variant.family() == ModelFamily::Qwen3Tts
        })
        .map(|breakdown| breakdown.decode_steps as usize);
    TtsStopReason::classify(budget_frames, frames, steps)
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
    fn stream_forwards_only_terminal_pcm_that_was_never_streamed() {
        let mut streamed = crate::engine::AudioOutput::streamed(48_000, 24_000);
        assert_eq!(unstreamed_terminal_pcm(&mut streamed, 3), None);
        // Qwen3-TTS streams its chunks and keeps the utterance as well.
        let mut kept = crate::engine::AudioOutput::new(vec![0.25; 4], 24_000);
        assert_eq!(unstreamed_terminal_pcm(&mut kept, 2), None);

        let mut final_only = crate::engine::AudioOutput::new(vec![0.25; 4], 24_000);
        assert_eq!(
            unstreamed_terminal_pcm(&mut final_only, 0),
            Some(vec![0.25; 4])
        );
        // The duration the terminal statistics report survives the move.
        assert!(final_only.samples.is_empty());
        assert!(final_only.duration_secs > 0.0);
    }

    #[test]
    fn fish_s2_public_tts_routes_always_use_retained_execution() {
        assert_eq!(
            tts_route_target(ModelVariant::FishAudioS2Pro, false),
            ExecutionTargetKind::TokenEngine
        );
        assert!(!uses_direct_streaming_tts_runtime(
            ModelVariant::FishAudioS2Pro
        ));
    }

    #[test]
    fn kokoro_uses_engine_for_atomic_generation_and_direct_native_streaming() {
        assert_eq!(
            tts_route_target(ModelVariant::Kokoro82M, false),
            ExecutionTargetKind::TokenEngine
        );
        assert!(uses_direct_streaming_tts_runtime(ModelVariant::Kokoro82M));
    }
    use crate::backends::BackendKind;
    use crate::engine::{Priority, ResourceAmount, ResourceVector, WorkloadClass};
    use crate::runtime::coordinator::{InferenceCoordinator, JobSpec};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// TPIPE-3: the advertised streaming mode of every enabled TTS variant
    /// is what its streaming route does. Every streaming family publishes
    /// audio as it commits it, so none is final-only: Qwen3-TTS on every
    /// backend (its per-frame stream replaced the CUDA-only chunked codec),
    /// and LFM2.5 Audio, VibeVoice and Voxtral, which used to run the
    /// non-stream route and send one final chunk.
    #[test]
    fn advertised_streaming_mode_is_what_the_route_does() {
        use crate::runtime::adapters::{RuntimeAdapterRegistry, StreamingMode};
        let registry = RuntimeAdapterRegistry::built_in();
        for variant in ModelVariant::all()
            .iter()
            .copied()
            .filter(|variant| variant.is_enabled() && variant.speech_capabilities().is_some())
        {
            // A stream runs on a streaming capability, or on the TTS
            // capability's own native-streaming graph, which the loader
            // seals only for a TTS adapter that streams in chunks.
            let streams = registry
                .require(CapabilityKind::StreamingTts, variant)
                .is_ok()
                || (variant.streams_on_its_tts_route()
                    && registry
                        .require(CapabilityKind::Tts, variant)
                        .is_ok_and(|tts| tts.streaming_mode == StreamingMode::Chunked));
            let expected = if streams {
                TtsStreamingMode::Chunked
            } else {
                TtsStreamingMode::Unsupported
            };
            assert_eq!(tts_streaming_mode(variant), expected, "{variant}");
        }
        for variant in [
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
            ModelVariant::Kokoro82M,
            ModelVariant::FishAudioS2Pro,
            ModelVariant::Lfm25Audio15BGguf,
            ModelVariant::VibeVoice15BTts,
            ModelVariant::Voxtral4BTts2603,
        ] {
            assert_eq!(
                tts_streaming_mode(variant),
                TtsStreamingMode::Chunked,
                "{variant}"
            );
            assert_eq!(
                tts_route_target(variant, true),
                if variant.is_kokoro() {
                    ExecutionTargetKind::DirectModel
                } else {
                    ExecutionTargetKind::TokenEngine
                },
                "{variant}"
            );
        }
    }

    /// The live CPU Qwen runs: short (54 steps / 53 frames) and long
    /// (983 / 982) ended on EOS but were reported as budget stops, because
    /// the scheduler token count is not a frame count; medium (313 / 313)
    /// really hit its budget. A family without the end-of-speech step
    /// signal never reads as a context stop.
    #[test]
    fn empty_speech_output_is_an_error_and_never_a_nan_duration() {
        let variant = ModelVariant::Kokoro82M;
        assert!(ensure_speech_audio(variant, "empty", &[], 24_000).is_err());
        assert!(ensure_speech_audio(variant, "rateless", &[0.1], 0).is_err());
        assert!(ensure_speech_audio(variant, "ok", &[0.1], 24_000).is_ok());
        let result = GenerationResult {
            request_id: "rateless".into(),
            samples: vec![0.1; 10],
            sample_rate: 0,
            total_tokens: 1,
            total_time_ms: 5.0,
            diagnostics: None,
            stop_reason: None,
        };
        assert_eq!(result.duration_secs(), 0.0);
        assert_eq!(result.rtf(), 0.0);
    }

    /// Kokoro speaks "$1,250.75" and "March 3rd" as words by default; a
    /// family that reads them itself is left alone unless the request asks,
    /// and a request can turn Kokoro's off.
    #[test]
    fn speech_text_normalization_follows_the_family_default_and_the_request() {
        let text = "It costs $1,250.75 on March 3rd.";
        let spoken = "It costs one thousand two hundred fifty dollars and seventy-five cents on March third.";
        let request = |normalize_text: Option<bool>| {
            let mut request = GenerationRequest::new(text);
            request.config.options.speech.normalize_text = normalize_text;
            request
        };
        assert_eq!(
            normalize_speech_text(request(None), ModelVariant::Kokoro82M).text,
            spoken
        );
        assert_eq!(
            normalize_speech_text(request(None), ModelVariant::Qwen3Tts12Hz06BCustomVoice).text,
            text
        );
        assert_eq!(
            normalize_speech_text(
                request(Some(true)),
                ModelVariant::Qwen3Tts12Hz06BCustomVoice
            )
            .text,
            spoken
        );
        assert_eq!(
            normalize_speech_text(request(Some(false)), ModelVariant::Kokoro82M).text,
            text
        );
        // Non-English text is never rewritten.
        let mut japanese = request(None);
        japanese.config.options.speaker = Some("jf_alpha".into());
        assert_eq!(
            normalize_speech_text(japanese, ModelVariant::Kokoro82M).text,
            text
        );
    }

    #[test]
    fn engine_tts_stop_reason_reads_frames_and_the_end_of_speech_step() {
        let breakdown = |steps: u32, frames: usize| {
            Some(crate::engine::LatencyBreakdown {
                pre_engine_ms: None,
                queue_wait_ms: 0.0,
                media_decode_ms: None,
                normalization_ms: None,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                decode_wall_ms: None,
                decode_tokens: Some(frames),
                post_first_token_ms: None,
                sampling_ms: None,
                codec_ms: None,
                postprocess_ms: None,
                ttft_ms: None,
                total_ms: 0.0,
                prefill_steps: 1,
                decode_steps: steps,
            })
        };
        let qwen = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let reason = |variant, budget, b: Option<_>, tokens| {
            tts_stop_reason(variant, budget, &b, tokens).unwrap()
        };
        assert_eq!(
            reason(qwen, 82, breakdown(54, 53), 864),
            TtsStopReason::EndOfSpeech
        );
        assert_eq!(
            reason(qwen, 1_500, breakdown(983, 982), 15_712),
            TtsStopReason::EndOfSpeech
        );
        assert_eq!(
            reason(qwen, 313, breakdown(313, 313), 5_008),
            TtsStopReason::OutputBudget
        );
        assert_eq!(
            reason(qwen, 313, breakdown(200, 200), 3_200),
            TtsStopReason::ContextLimit
        );
        let vibevoice = ModelVariant::VibeVoice15BTts;
        assert_eq!(
            reason(vibevoice, 384, breakdown(200, 200), 200),
            TtsStopReason::EndOfSpeech
        );
        assert_eq!(
            reason(vibevoice, 384, None, 384),
            TtsStopReason::OutputBudget
        );
        assert_eq!(tts_stop_reason(qwen, 0, &breakdown(5, 5), 5), None);
    }

    /// Phase 0 capability contract for every enabled TTS variant: the route
    /// the runtime takes for non-stream and stream requests is the one the
    /// registry advertises (VXT-1); streaming is advertised exactly when a
    /// streaming capability is published; a streamed request that runs on
    /// the engine belongs to a family whose executor publishes audio and a
    /// final marker (TPIPE-3/4); and the automatic
    /// frame budget grows with text and stays inside the model's maximum
    /// (TPIPE-2).
    #[test]
    fn every_tts_family_route_matches_its_advertised_capability_and_budget() {
        use crate::models::architectures::qwen3::tts::qwen3_tts_auto_max_frames_for_text;
        use crate::models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
        use crate::runtime::adapters::RuntimeAdapterRegistry;
        // Families whose engine executors stream committed audio and publish
        // the final marker.
        const ENGINE_STREAM_EMITTERS: [ModelFamily; 6] = [
            ModelFamily::Qwen3Tts,
            ModelFamily::FishS2Tts,
            ModelFamily::VoxtralTts,
            ModelFamily::Lfm25Audio,
            ModelFamily::VibeVoiceTts,
            ModelFamily::ChatterboxTts,
        ];
        let registry = RuntimeAdapterRegistry::built_in();
        let short = "Hello there.";
        let paragraph = "The committee reviewed every proposal in detail. ".repeat(6);
        let flood = "word ".repeat(20_000);
        let mut checked = 0;
        for variant in ModelVariant::all().iter().copied() {
            if !variant.is_enabled() || !variant.is_tts() {
                continue;
            }
            checked += 1;
            let capabilities = variant.speech_capabilities().expect("speech capabilities");
            let tts = registry
                .require(CapabilityKind::Tts, variant)
                .unwrap_or_else(|error| panic!("{variant}: {error}"));
            assert_eq!(
                tts.execution_target,
                tts_route_target(variant, false),
                "{variant}: non-stream route"
            );
            match registry.require(CapabilityKind::StreamingTts, variant) {
                Ok(streaming) => {
                    assert!(capabilities.supports_streaming, "{variant}");
                    assert_eq!(
                        streaming.execution_target,
                        tts_route_target(variant, true),
                        "{variant}: stream route"
                    );
                }
                Err(_) => assert!(
                    !capabilities.supports_streaming || variant.streams_on_its_tts_route(),
                    "{variant}"
                ),
            }
            let engine_stream = capabilities.supports_streaming
                && tts_route_target(variant, true) == ExecutionTargetKind::TokenEngine;
            assert!(
                !engine_stream || ENGINE_STREAM_EMITTERS.contains(&variant.family()),
                "{variant}: an engine stream would end without audio"
            );
            let budget: Option<fn(&str) -> usize> = match variant.family() {
                ModelFamily::Qwen3Tts => Some(qwen3_tts_auto_max_frames_for_text),
                ModelFamily::VibeVoiceTts => Some(vibevoice_tts_auto_max_frames_for_text),
                ModelFamily::VoxtralTts => Some(voxtral_tts_auto_max_frames_for_text),
                _ => None,
            };
            if let Some(budget) = budget {
                let maximum = variant.tts_max_output_frames_hint().expect("frame hint");
                assert!(budget(short) > 0, "{variant}");
                assert!(budget(&paragraph) > budget(short), "{variant}");
                assert!(budget(&flood) <= maximum, "{variant}");
            }
        }
        assert!(checked >= 8, "only {checked} TTS variants were checked");
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
    fn kokoro_max_text_keeps_the_hard_unit_contract_and_reserves_one_chunk_of_output() {
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

        assert_eq!(shape.units, budget.max_model_tokens as u64);
        // One 180-character chunk at 4 frames per character plus the
        // 64-frame chunk allowance, halved speed.
        assert_eq!(shape.kokoro_chunk_frames, Some(1_568));
        assert_eq!(shape.max_output_samples, 1_568 * 600);
    }

    /// KOK-8/9: a 45-word stream reserved 3.34 GB on Metal (4,096 frames of
    /// workspace plus the whole text's output at 50 bins per token) and was
    /// refused against 2.30 GB of live memory, although one chunk of real
    /// speech needs far less. The stream now reserves its largest planned
    /// chunk at the planning pace, with the final-stage im2col counted
    /// (rounded to a power of two on Metal), and one chunk of output.
    #[test]
    fn kokoro_stream_reserves_one_planned_chunk_on_each_backend() {
        use crate::models::architectures::kokoro::admission::kokoro_chunk_price;
        let paragraph = "Good morning. Today we are testing a speech synthesis system on a \
            laptop. The goal is to measure how fast it runs, how much memory it uses, and \
            whether every word in this paragraph comes out clearly, without skipping, \
            repeating, or adding anything at the end.";
        let request = GenerationRequest::new(paragraph);
        let resources = |backend| {
            direct_tts_additional_resources(backend, &request, ModelVariant::Kokoro82M, 4_096)
                .unwrap()
        };
        // Its heaviest 180-character window, plus the per-chunk allowance.
        let frames = crate::runtime::kokoro::kokoro_stream_chunk_frames(paragraph, 1.0);
        assert!((64 + 180 * 4..=64 + 180 * 8).contains(&frames), "{frames}");
        let price = |backend| kokoro_chunk_price(frames, backend).unwrap();
        let output_bytes = frames as u64 * 600 * 4;
        let metal = price(BackendKind::Metal);
        assert_eq!(
            resources(BackendKind::Metal).unified_bytes,
            ResourceAmount::Known(output_bytes + metal.host_bytes + metal.accelerator_tensor_bytes)
        );
        let cpu = price(BackendKind::Cpu);
        assert_eq!(
            resources(BackendKind::Cpu).host_bytes,
            ResourceAmount::Known(output_bytes + cpu.host_bytes + cpu.cpu_tensor_bytes)
        );
        let cuda = price(BackendKind::Cuda);
        assert_eq!(
            resources(BackendKind::Cuda).device_bytes,
            ResourceAmount::Known(cuda.accelerator_tensor_bytes)
        );
        let ResourceAmount::Known(metal_total) = resources(BackendKind::Metal).unified_bytes else {
            panic!("known Metal reservation");
        };
        // The tiled decoder (16 MiB per conv) holds one tile's im2col, not
        // the 512 MiB stage im2col: about 0.55 GB, from about 1.0 GB.
        assert!(
            metal_total < 600 * 1024 * 1024,
            "45-word stream reserves {metal_total} bytes on Metal"
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

    #[tokio::test]
    async fn terminal_statistics_send_honors_deadline_and_closed_receiver() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.send(AudioChunk::new("terminal".into(), 0, vec![0.0]))
            .await
            .unwrap();
        let terminal = || AudioChunk::final_chunk("terminal".into(), 1, Vec::new());
        assert!(matches!(
            send_terminal_tts_chunk(
                &tx,
                terminal(),
                Some(Instant::now() + Duration::from_millis(5))
            )
            .await,
            Err(Error::Timeout(_))
        ));
        assert!(!rx.recv().await.unwrap().is_final);
        assert!(
            rx.try_recv().is_err(),
            "timed-out terminal marker must not arrive later"
        );
        assert!(matches!(
            send_terminal_tts_chunk(&tx, terminal(), Some(Instant::now())).await,
            Err(Error::Timeout(_))
        ));
        drop(rx);
        assert!(matches!(
            send_terminal_tts_chunk(&tx, terminal(), None).await,
            Err(Error::InferenceError(_))
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
