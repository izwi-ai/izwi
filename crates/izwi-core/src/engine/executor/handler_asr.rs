use crate::backends::state::PhysicalStateTransactionId;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::granite_speech::asr::{
    granite_speech_request_plan, GraniteSpeechTask,
};
use crate::models::architectures::lfm25_audio::asr_retained::{
    Lfm25AudioAsrDecodeStep, Lfm25AudioAsrQuantumCheckpoint, Lfm25AudioAsrStopReason,
};
use crate::models::architectures::parakeet::asr::{
    ParakeetRetainedDecodeBatchRow, ParakeetRetainedDecodeState,
};
use crate::models::architectures::vibevoice::asr::VibeVoiceAsrRetainedTokenizerQuantum;
use crate::models::architectures::vibevoice::asr::{
    VibeVoiceAsrPreparedTokenizerSpan, VibeVoiceAsrRetainedPrefillBatchRow,
};
use crate::models::architectures::voxtral::realtime::{
    VoxtralRealtimeDecodeBatchRow, VoxtralRealtimePreparationBatchRow,
    VoxtralRealtimePreparationMode,
};
use crate::models::architectures::whisper::asr::WhisperTerminalTransition;
use crate::models::registry::{
    NativeAsrDecodeCheckpoint, NativeAsrDecodeStep, NativeAsrGenerationOptions, NativeAsrModel,
};
use crate::runtime::{
    granite_asr_planned_max_tokens, granite_auto_asr_max_tokens_for_duration,
    invocation_decode_room,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

use super::super::request::{EngineCoreRequest, RealtimeAsrOperationPayload};
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::{
    RealtimePreparationMode, RealtimeStageOutcome, RealtimeSubphase, SequenceRestartReason,
    SessionKey, WorkUnit,
};
use super::audio::{
    decode_request_audio_with_rate, timestamp_words_json, AsrChunkTranscription,
    AsrRouteCapabilities, ChunkStream, ChunkedAsrRun, ChunkedAsrTranscription, PlannedAsrChunks,
};
use super::long_form::LongFormAsrState;
use super::state::{
    ActiveAsrDecode, ActiveLfm25AsrDecode, ActiveParakeetAsrDecode, ActiveVoxtralRealtime,
};
use super::{
    ExecutorOutput, ExecutorPhaseTiming, ExecutorStateLease, ModelSessionResult, NativeExecutor,
};
use crate::models::architectures::granite_speech5::asr::{
    granite_speech5_auto_max_tokens_for_duration, GraniteSpeech5RequestOptions,
};
use granite_speech5::{recover_granite_speech5_chunk, ChunkRecoveryLimits};

#[path = "handler_asr/continuous.rs"]
mod continuous;
#[path = "handler_asr/granite.rs"]
mod granite;
#[path = "handler_asr/granite_speech5.rs"]
mod granite_speech5;
#[path = "handler_asr/lfm25_audio.rs"]
mod lfm25_audio;
#[path = "handler_asr/parakeet.rs"]
mod parakeet;
#[path = "handler_asr/qwen3.rs"]
mod qwen3;
#[path = "handler_asr/vibevoice.rs"]
mod vibevoice;
#[path = "handler_asr/voxtral.rs"]
mod voxtral;
#[path = "handler_asr/whisper.rs"]
mod whisper;
#[cfg(test)]
use granite::{
    granite_chunk_loop_signal, granite_chunk_truncated, granite_loop_recovery_max_tokens,
    granite_truncation_retry_max_tokens, recent_word_suffix, repeated_phrase_loop,
    trim_repeated_phrase_tail, with_granite_truncation_retry_diagnostics,
};
#[cfg(test)]
use lfm25_audio::{late_cancelled_lfm_asr_prefill_rows, lfm25_audio_asr_long_form_max_tokens};
#[cfg(test)]
use parakeet::{late_cancelled_parakeet_rows, rollback_parakeet_host_state};
#[cfg(test)]
use vibevoice::vibevoice_prefill_boundary_step;
use vibevoice::with_vibevoice_invocation_state;
use whisper::with_whisper_invocation_state;
#[cfg(test)]
use whisper::{
    apply_whisper_terminal_transition, begins_whisper_managed_generation,
    resolve_whisper_terminal_action, whisper_prefill_boundary_step, WhisperTerminalAction,
};

const MAX_ASR_NEW_TOKENS: usize = 512;
const GRANITE_ASR_PREFIX_REPLAY_WORDS: usize = 0;
const GRANITE_ASR_PREFIX_REPLAY_WORDS_MAX: usize = 240;

fn retained_asr_batch_model_call(
    mode: crate::engine::NativeBatchMode,
    live_kernel_rows: usize,
) -> Option<crate::engine::metrics::EngineModelCall> {
    match live_kernel_rows {
        0 => None,
        1 => Some(crate::engine::metrics::EngineModelCall::ScalarRows {
            envelope: mode,
            rows: 1,
        }),
        rows => Some(crate::engine::metrics::EngineModelCall::NativeTensor { mode, rows }),
    }
}

enum AsrExecutionAudio {
    Prepared(Arc<[f32]>),
    Decoded(Vec<f32>),
}

/// What one generic ASR dispatch produced: the transcript, or one chunk of
/// a long-form run that continues in a later dispatch.
enum AsrTranscriptionStep {
    Complete(String, Option<serde_json::Value>),
    Yielded(Box<LongFormAsrState>),
}

impl AsrExecutionAudio {
    fn samples(&self) -> &[f32] {
        match self {
            Self::Prepared(samples) => samples,
            Self::Decoded(samples) => samples,
        }
    }

    fn shared(&self) -> Arc<[f32]> {
        match self {
            Self::Prepared(samples) => samples.clone(),
            Self::Decoded(samples) => Arc::from(samples.as_slice()),
        }
    }
}

fn resolve_asr_execution_audio(
    request: &EngineCoreRequest,
    family: ModelFamily,
    decode: impl FnOnce() -> Result<(Vec<f32>, u32)>,
) -> Result<(AsrExecutionAudio, u32, f64)> {
    if matches!(
        family,
        ModelFamily::Qwen3Asr
            | ModelFamily::WhisperAsr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::GraniteSpeechAsr
            | ModelFamily::Lfm25Audio
    ) {
        if let Some((samples, sample_rate)) = request.prepared_asr_audio_for_executor()? {
            return Ok((AsrExecutionAudio::Prepared(samples), sample_rate, 0.0));
        }
        if matches!(
            family,
            ModelFamily::Qwen3Asr
                | ModelFamily::VibeVoiceAsr
                | ModelFamily::GraniteSpeechAsr
                | ModelFamily::Lfm25Audio
        ) {
            return Err(Error::InferenceError(format!(
                "{} execution lost its prepared decoded-audio artifact",
                match family {
                    ModelFamily::Qwen3Asr => "Qwen3 ASR",
                    ModelFamily::VibeVoiceAsr => "VibeVoice ASR",
                    ModelFamily::GraniteSpeechAsr => "Granite Speech ASR",
                    ModelFamily::Lfm25Audio => "LFM2.5 Audio ASR",
                    _ => unreachable!("guarded prepared-audio family"),
                }
            )));
        }
    }
    let started = Instant::now();
    let (samples, sample_rate) = decode()?;
    Ok((
        AsrExecutionAudio::Decoded(samples),
        sample_rate,
        started.elapsed().as_secs_f64() * 1000.0,
    ))
}

/// Executor path for an ASR request that no realtime or retained-sequence
/// executor claimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomicAsrRoute {
    /// The prepared LFM2.5 Audio lease chunks and decodes the audio itself.
    Lfm25AudioLongForm,
    /// The generic path, which resolves a `NativeAsrModel` (or Voxtral) lease.
    Generic,
}

/// Audio-chat models are loaded only as audio-chat leases, never into the
/// `NativeAsrModel` registry that the generic path resolves. Route their
/// long-form requests to the dedicated executor and fail closed otherwise,
/// instead of reporting a loaded model as missing.
fn atomic_asr_route(
    request: &EngineCoreRequest,
    variant: crate::model::ModelVariant,
) -> Result<AtomicAsrRoute> {
    if !variant.is_audio_chat() {
        return Ok(AtomicAsrRoute::Generic);
    }
    if variant.family() == ModelFamily::Lfm25Audio && request.uses_asr_long_form_atomic() {
        return Ok(AtomicAsrRoute::Lfm25AudioLongForm);
    }
    Err(Error::InvalidInput(format!(
        "ASR request {} for audio-chat model {variant} has no retained or long-form route",
        request.id
    )))
}

/// Physical model call the generic ASR path makes for a `NativeAsrModel`
/// family, in every mode (streaming or not, single pass or per chunk). Each
/// route runs against the lifecycle-owned invocation workspace. There is no
/// route to the registry's workspace-less APIs: Qwen3 cannot run there at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalAsrRoute {
    Qwen3,
    VibeVoice,
    GraniteSpeech,
    GraniteSpeech5,
    Whisper,
    Parakeet,
    Nemotron,
}

/// Every family is listed, so a new one has to choose a physical route here
/// or fail with a typed error before any model call.
fn physical_asr_route(family: ModelFamily) -> Result<PhysicalAsrRoute> {
    match family {
        ModelFamily::Qwen3Asr => Ok(PhysicalAsrRoute::Qwen3),
        ModelFamily::VibeVoiceAsr => Ok(PhysicalAsrRoute::VibeVoice),
        ModelFamily::GraniteSpeechAsr => Ok(PhysicalAsrRoute::GraniteSpeech),
        ModelFamily::GraniteSpeech5Asr => Ok(PhysicalAsrRoute::GraniteSpeech5),
        ModelFamily::WhisperAsr => Ok(PhysicalAsrRoute::Whisper),
        ModelFamily::ParakeetAsr => Ok(PhysicalAsrRoute::Parakeet),
        ModelFamily::NemotronAsr => Ok(PhysicalAsrRoute::Nemotron),
        // Voxtral and LFM2.5 Audio have their own executors; the forced
        // aligner and the remaining families do not transcribe.
        ModelFamily::Voxtral
        | ModelFamily::Lfm25Audio
        | ModelFamily::Qwen3ForcedAligner
        | ModelFamily::Qwen3Tts
        | ModelFamily::KokoroTts
        | ModelFamily::VoxtralTts
        | ModelFamily::VibeVoiceTts
        | ModelFamily::FishS2Tts
        | ModelFamily::SortformerDiarization
        | ModelFamily::Qwen3Chat
        | ModelFamily::Qwen3MoeChat
        | ModelFamily::Qwen35Chat
        | ModelFamily::Qwen35MoeChat
        | ModelFamily::Qwen38Chat
        | ModelFamily::Lfm2Chat
        | ModelFamily::Gemma3Chat
        | ModelFamily::Tokenizer => Err(Error::InvalidInput(format!(
            "{family:?} has no physical ASR route on the generic executor path"
        ))),
    }
}

/// The route plan the generic path executes. A long-form row runs the plan
/// its preparation stored, unchanged; a family without a preparation stage
/// (Nemotron, offline Voxtral) is planned here by the same rule.
fn asr_execution_route_plan(
    request: &EngineCoreRequest,
    samples: &[f32],
    sample_rate: u32,
    capabilities: impl FnOnce() -> Result<AsrRouteCapabilities>,
) -> Result<Arc<PlannedAsrChunks>> {
    if let Some(plan) = request.prepared_asr_route_plan() {
        if plan.input_samples != samples.len() || plan.sample_rate != sample_rate {
            return Err(Error::InferenceError(format!(
                "ASR request {} route plan does not match its prepared audio",
                request.id
            )));
        }
        return Ok(plan);
    }
    if request.uses_asr_long_form_atomic() {
        return Err(Error::InferenceError(format!(
            "long-form ASR request {} lost its prepared route plan",
            request.id
        )));
    }
    Ok(Arc::new(NativeExecutor::asr_route_plan(
        samples,
        sample_rate,
        &capabilities()?,
        request.streaming,
    )))
}

fn begins_resumable_asr_prefill_state(
    scheduled: &ScheduledRequest,
    resumable_prefill: bool,
) -> bool {
    scheduled.is_prefill && resumable_prefill && scheduled.num_computed_tokens == 0
}

fn resumable_asr_prefill_span(
    scheduled: &ScheduledRequest,
    prompt_tokens: usize,
) -> Result<(usize, usize)> {
    let start = scheduled.num_computed_tokens;
    let end = start.checked_add(scheduled.num_tokens).ok_or_else(|| {
        Error::InvalidInput("resumable ASR prefill span overflowed prompt accounting".into())
    })?;
    let crate::engine::WorkUnit::SequenceStep { phase, input, .. } = &scheduled.work else {
        return Err(Error::InvalidInput(
            "resumable ASR prefill requires sequence-prefill work".into(),
        ));
    };
    if *phase != crate::engine::SequencePhase::Prefill
        || input.start != start
        || input.end != end
        || start >= end
        || end > prompt_tokens
    {
        return Err(Error::InvalidInput(format!(
            "resumable ASR prefill work [{}, {}) disagrees with scheduler span [{start}, {end}) for {prompt_tokens} prompt tokens",
            input.start, input.end
        )));
    }
    Ok((start, end))
}

/// Run one physical ASR model call and observe cooperative cancellation before
/// any caller can publish output or detach the transaction checkpoint. Native
/// device calls are not preemptible, so this post-call edge is the first safe
/// point at which a cancellation that arrived during the call can be honored.
fn run_asr_model_call<T>(
    request: &EngineCoreRequest,
    run: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    let output = run()?;
    Ok((!request.is_cancelled()).then_some(output))
}

/// Settle an atomic ASR invocation against cooperative cancellation. The
/// chunk executor stops with `Cancelled` at the first chunk boundary after the
/// signal flips; a single-window model call cannot be preempted, so a
/// cancellation that arrived during it is honored here instead. Either way no
/// text or staged stream output is published, and `None` tells the caller to
/// finish the row as cancelled.
fn settle_atomic_asr_cancellation<T>(
    request: &EngineCoreRequest,
    result: Result<T>,
) -> Result<Option<T>> {
    match result {
        Err(Error::Cancelled(_)) => {}
        Ok(_) if request.is_cancelled() => {}
        other => return other.map(Some),
    }
    let _ = request.take_staged_stream_outputs();
    Ok(None)
}

fn late_cancelled_batch_row(cancelled: bool, checkpoint_armed: bool) -> bool {
    cancelled && checkpoint_armed
}

fn rollback_scalar_asr_quantum(
    state_lease: &mut ExecutorStateLease<'_, ActiveAsrDecode>,
    checkpoint: &mut Option<NativeAsrDecodeCheckpoint>,
    outer_checkpoint: Option<(usize, usize)>,
    fresh_state: bool,
) -> Result<()> {
    if let Some(checkpoint) = checkpoint.take() {
        let active_state = state_lease.require_state_mut()?;
        active_state.state.rollback_managed_quantum(checkpoint)?;
        if let Some((last_tokens_generated, stream_sequence)) = outer_checkpoint {
            active_state.last_tokens_generated = last_tokens_generated;
            active_state.stream_sequence = stream_sequence;
        }
        state_lease.mark_clean();
    } else if fresh_state {
        state_lease.discard_state();
        state_lease.mark_clean();
    }
    Ok(())
}

fn with_single_invocation_cache<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(&mut crate::models::shared::attention::physical::PhysicalPagedKvCache) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_paged_leases_for_atomic_scalar_row(request, scheduled)?;
    let domains = leases.domains().collect::<Vec<_>>();
    let [domain] = domains.as_slice() else {
        return Err(Error::InferenceError(format!(
            "ASR stage requires exactly one invocation KV domain, found {}",
            domains.len()
        )));
    };
    let result = run(leases.cache_mut(*domain)?);
    let output = leases.fence_on_error(result)?;
    let _completions = leases.release()?;
    Ok(output)
}

fn with_single_invocation_tensor<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(&mut crate::engine::InvocationTensorLease) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = {
        let tensor =
            leases.lease_exact_kind_mut(crate::kv::v2::InvocationStateBackingKindV2::Tensor)?;
        run(tensor.typed_mut::<crate::engine::InvocationTensorLease>()?)
    };
    let output = leases.fence_on_error(output)?;
    let completions = leases.release()?;
    if completions.len() != 1 {
        return Err(Error::InferenceError(
            "atomic ASR tensor state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

fn with_nemotron_offline_state<T>(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    run: impl FnOnce(
        &mut crate::engine::InvocationTensorLease,
        &mut crate::engine::InvocationTensorLease,
    ) -> Result<T>,
) -> Result<T> {
    let mut leases = super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
    let output = {
        let (predictor, acoustic) = leases.lease_exact_kind_pair_mut(
            crate::kv::v2::InvocationStateBackingKindV2::Tensor,
            crate::kv::v2::InvocationStateBackingKindV2::StaticTensor,
        )?;
        run(
            predictor.typed_mut::<crate::engine::InvocationTensorLease>()?,
            acoustic.typed_mut::<crate::engine::InvocationTensorLease>()?,
        )
    };
    let output = leases.fence_on_error(output)?;
    let completions = leases.release()?;
    if completions.len() != 2 {
        return Err(Error::InferenceError(
            "Nemotron offline state returned an incomplete completion set".to_string(),
        ));
    }
    Ok(output)
}

impl NativeExecutor {
    pub(super) fn transcribe_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.transcribe_request_with_managed_cache(request, scheduled, None)
    }

    pub(super) fn transcribe_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        if request.managed_cache_runtime().is_some() != managed_state.is_some() {
            return Err(Error::InferenceError(
                "ASR request and scheduler reservation disagree on retained-state authority".into(),
            ));
        }
        let variant = Self::resolve_variant(request)?;
        let family = variant.family();
        if family == ModelFamily::NemotronAsr && request.is_realtime_asr_session() {
            // The runtime streams Nemotron on its direct path; the engine
            // never admits a Nemotron realtime session (SPK-F14).
            return Err(Error::InvalidInput(
                "Nemotron realtime ASR runs on the direct streaming path, not the engine".into(),
            ));
        }
        if family == ModelFamily::Voxtral && request.is_realtime_asr_session() {
            let retained = managed_state.take().ok_or_else(|| {
                Error::InferenceError("Voxtral realtime ASR lost retained paged state".into())
            })?;
            return self.voxtral_realtime_request(request, scheduled, retained);
        }
        if family == ModelFamily::Qwen3Asr && request.uses_asr_retained_sequence() {
            return self.qwen3_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::WhisperAsr && request.uses_asr_retained_sequence() {
            return self.whisper_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::VibeVoiceAsr && request.uses_asr_retained_sequence() {
            return self.vibevoice_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::GraniteSpeechAsr && request.uses_asr_retained_sequence() {
            return self.granite_speech_asr_sequence_request(
                request,
                scheduled,
                managed_state.take(),
            );
        }
        if family == ModelFamily::Lfm25Audio && request.uses_asr_retained_sequence() {
            return self.lfm25_audio_asr_sequence_request(request, scheduled, managed_state.take());
        }
        if family == ModelFamily::ParakeetAsr && request.uses_asr_retained_sequence() {
            let mut outputs = if scheduled.is_prefill {
                self.parakeet_asr_prefill_batch_with_managed(
                    &[request],
                    std::slice::from_ref(scheduled),
                    vec![managed_state.take()],
                )?
            } else {
                self.parakeet_asr_decode_batch_with_managed(
                    &[request],
                    std::slice::from_ref(scheduled),
                    vec![managed_state.take()],
                )?
            };
            return outputs.pop().ok_or_else(|| {
                Error::InferenceError("Parakeet B1 fallback returned no output".into())
            });
        }
        if managed_state.is_some() {
            return Err(Error::InferenceError(
                "retained ASR state was routed outside a retained sequence executor".to_string(),
            ));
        }
        match atomic_asr_route(request, variant)? {
            AtomicAsrRoute::Lfm25AudioLongForm => {
                return self.lfm25_audio_asr_long_form_request(request, scheduled);
            }
            AtomicAsrRoute::Generic => {}
        }
        let language = request.asr_language_for_execution();
        let asr_prompt = request.asr_prompt_for_execution();
        let generation_options = Self::asr_generation_options(request);
        let granite_speech5_request = GraniteSpeech5RequestOptions {
            language: language.map(str::to_string),
            prompt: asr_prompt.map(str::to_string),
            ..GraniteSpeech5RequestOptions::default()
        };
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        // A long-form row of a family that dispatches its chunks one at a
        // time resumes the run its committed chunks left (`Some(None)` on
        // its first dispatch).
        let session = scheduled.session_key();
        let long_form = (request.uses_asr_long_form_atomic()
            && family.dispatches_long_form_asr_by_chunk())
        .then(|| self.long_form_asr.committed(&session));

        let (execution_audio, sample_rate, audio_decode_ms) =
            match long_form.as_ref().and_then(Option::as_ref) {
                Some(resumed) => (
                    AsrExecutionAudio::Prepared(resumed.samples.clone()),
                    resumed.sample_rate,
                    resumed.audio_decode_ms,
                ),
                None => resolve_asr_execution_audio(request, family, || {
                    Self::run_blocking(|| decode_request_audio_with_rate(request))
                })?,
            };
        let samples = execution_audio.samples();
        let samples_len = samples.len();

        let transcription = Self::run_blocking(|| {
            let mut sequence = 0usize;
            if matches!(family, ModelFamily::Voxtral) {
                let model = self.with_registry(|registry| {
                    registry.try_get_voxtral_lease(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!(
                            "Voxtral model {variant} is not loaded in registry"
                        ))
                    })
                })?;

                let chunk_plan = asr_execution_route_plan(request, samples, sample_rate, || {
                    Ok(model.asr_route_capabilities())
                })?;
                if chunk_plan.requires_chunk_path() {
                    let chunked =
                        Self::transcribe_with_chunk_plan_with_streaming_details_and_options(
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
                            |chunk_audio, sr, _prefix_text, partial| {
                                if let Some(emit) = partial {
                                    return with_single_invocation_cache(
                                        request,
                                        scheduled,
                                        |cache| {
                                            model.transcribe_with_callback_physical(
                                                chunk_audio,
                                                sr,
                                                language,
                                                emit,
                                                cache,
                                            )
                                        },
                                    )
                                    .map(|text| {
                                        AsrChunkTranscription {
                                            text,
                                            diagnostics: None,
                                        }
                                    });
                                }
                                with_single_invocation_cache(request, scheduled, |cache| {
                                    model.transcribe_with_details_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        cache,
                                    )
                                })
                                .map(|details| {
                                    AsrChunkTranscription {
                                        text: details.text,
                                        diagnostics: details.diagnostics,
                                    }
                                })
                            },
                        )?;
                    return Ok(Self::complete_chunked_asr(&chunk_plan, chunked));
                }

                if request.streaming {
                    if let Some(tx) = stream_tx.as_ref() {
                        let mut stream_err: Option<Error> = None;
                        let mut emit = |delta: &str| {
                            if stream_err.is_none() {
                                if let Err(err) = Self::stream_text_with_policy(
                                    tx,
                                    stream_policy,
                                    &request.id,
                                    &mut sequence,
                                    delta.to_string(),
                                ) {
                                    stream_err = Some(err);
                                }
                            }
                        };
                        let text = with_single_invocation_cache(request, scheduled, |cache| {
                            model.transcribe_with_callback_physical(
                                samples,
                                sample_rate,
                                language,
                                &mut emit,
                                cache,
                            )
                        })?;
                        if let Some(err) = stream_err {
                            return Err(err);
                        }
                        Self::stream_final_marker_with_policy(
                            tx,
                            stream_policy,
                            &request.id,
                            &mut sequence,
                        )?;
                        return Ok(AsrTranscriptionStep::Complete(text, None));
                    }
                }
                let details = with_single_invocation_cache(request, scheduled, |cache| {
                    model.transcribe_with_details_physical(samples, sample_rate, language, cache)
                })?;
                return Ok(AsrTranscriptionStep::Complete(
                    details.text,
                    details.diagnostics,
                ));
            }

            let route = physical_asr_route(family)?;
            let (model, _model_lease) = self.asr_model_for_request(request, variant)?;

            let granite_plan = matches!(family, ModelFamily::GraniteSpeechAsr)
                .then(|| granite_speech_request_plan(variant, request.asr_native_word_timestamps));
            let granite_task = granite_plan.map_or(GraniteSpeechTask::Asr, |plan| plan.task);
            let effective_context = self.asr_effective_context(variant);
            let invocation_context = effective_context.zip(model.audio_token_rate());
            let chunk_plan = asr_execution_route_plan(request, samples, sample_rate, || {
                AsrRouteCapabilities::for_native_model(&model, variant, request, effective_context)
            })?;
            if chunk_plan.requires_chunk_path() {
                let transcribe_chunk =
                    |chunk_audio: &[f32],
                     sr: u32,
                     prefix_text: &str,
                     partial: Option<&mut dyn FnMut(&str)>| {
                        let bounded_prefix_text = granite_plan
                            .filter(|plan| plan.prefix_replay)
                            .map(|_| Self::granite_asr_prefix_replay_text(prefix_text));
                        let prefix_text = bounded_prefix_text
                            .as_deref()
                            .filter(|value| !value.trim().is_empty());
                        let chunk_generation_options = Self::asr_chunk_generation_options(
                            request,
                            family,
                            chunk_audio.len(),
                            sr,
                            &generation_options,
                            invocation_context,
                        );
                        let mut details = match route {
                            PhysicalAsrRoute::Qwen3 => {
                                if let Some(emit) = partial {
                                    return with_single_invocation_cache(
                                        request,
                                        scheduled,
                                        |cache| {
                                            model
                                                .transcribe_qwen3_with_callback_and_prompt_physical(
                                                    chunk_audio,
                                                    sr,
                                                    language,
                                                    asr_prompt,
                                                    emit,
                                                    cache,
                                                )
                                        },
                                    )
                                    .map(|text| {
                                        AsrChunkTranscription {
                                            text,
                                            diagnostics: None,
                                        }
                                    });
                                }
                                with_single_invocation_cache(request, scheduled, |cache| {
                                    model.transcribe_qwen3_with_details_and_prompt_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        cache,
                                    )
                                })?
                            }
                            PhysicalAsrRoute::VibeVoice => {
                                if let Some(emit) = partial {
                                    return with_vibevoice_invocation_state(
                                        request,
                                        scheduled,
                                        |leases| {
                                            model.transcribe_vibevoice_with_callback_and_prompt_and_options_physical(
                                                chunk_audio,
                                                sr,
                                                language,
                                                asr_prompt,
                                                chunk_generation_options.clone(),
                                                leases,
                                                emit,
                                            )
                                        },
                                    )
                                    .map(|text| AsrChunkTranscription {
                                        text,
                                        diagnostics: None,
                                    });
                                }
                                with_vibevoice_invocation_state(request, scheduled, |leases| {
                                    model.transcribe_vibevoice_with_details_and_prompt_and_options_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        asr_prompt,
                                        chunk_generation_options.clone(),
                                        leases,
                                    )
                                })?
                            }
                            PhysicalAsrRoute::GraniteSpeech => {
                                with_single_invocation_cache(request, scheduled, |cache| {
                                    model.transcribe_granite_speech_with_details_prompt_prefix_and_options_physical(
                                        chunk_audio,
                                        sr,
                                        language,
                                        granite_task,
                                        asr_prompt,
                                        prefix_text,
                                        chunk_generation_options.clone(),
                                        cache,
                                    )
                                })?
                            }
                            // Its chunk text is final only once recovery has
                            // looked at it, so it never streams within a chunk.
                            PhysicalAsrRoute::GraniteSpeech5 => {
                                let redecode = |options: NativeAsrGenerationOptions| {
                                    with_single_invocation_cache(request, scheduled, |cache| {
                                        model.transcribe_granite_speech5_physical(
                                            chunk_audio,
                                            sr,
                                            &granite_speech5_request,
                                            options,
                                            cache,
                                            None,
                                        )
                                    })
                                };
                                let details = redecode(chunk_generation_options.clone())?;
                                recover_granite_speech5_chunk(
                                    details,
                                    chunk_audio.len() as f32 / sr.max(1) as f32,
                                    &chunk_generation_options,
                                    ChunkRecoveryLimits {
                                        auto_max_tokens: request.asr_auto_max_tokens,
                                        request_ceiling: request.params.max_tokens,
                                        invocation_context,
                                    },
                                    redecode,
                                )?
                            }
                            PhysicalAsrRoute::Whisper => {
                                if let Some(emit) = partial {
                                    return with_whisper_invocation_state(
                                        request,
                                        scheduled,
                                        |self_kv, cross_kv| {
                                            model.transcribe_whisper_with_callback_and_prompt_physical(
                                                chunk_audio,
                                                sr,
                                                language,
                                                asr_prompt,
                                                self_kv,
                                                cross_kv,
                                                emit,
                                            )
                                        },
                                    )
                                    .map(|text| AsrChunkTranscription {
                                        text,
                                        diagnostics: None,
                                    });
                                }
                                with_whisper_invocation_state(
                                    request,
                                    scheduled,
                                    |self_kv, cross_kv| {
                                        model.transcribe_whisper_with_details_and_prompt_physical(
                                            chunk_audio,
                                            sr,
                                            language,
                                            asr_prompt,
                                            self_kv,
                                            cross_kv,
                                        )
                                    },
                                )?
                            }
                            // Parakeet's windowed encoder covers any length in
                            // one pass, so its route plan never chunks.
                            PhysicalAsrRoute::Parakeet => {
                                return Err(Error::InferenceError(format!(
                                    "Parakeet ASR request {} reached the chunk route",
                                    request.id
                                )));
                            }
                            PhysicalAsrRoute::Nemotron => {
                                if let Some(emit) = partial {
                                    return with_nemotron_offline_state(
                                        request,
                                        scheduled,
                                        |predictor, acoustic| {
                                            model.transcribe_nemotron_with_callback_and_prompt_physical(
                                                chunk_audio,
                                                sr,
                                                language,
                                                asr_prompt,
                                                predictor,
                                                acoustic,
                                                emit,
                                            )
                                        },
                                    )
                                    .map(|text| AsrChunkTranscription {
                                        text,
                                        diagnostics: None,
                                    });
                                }
                                with_nemotron_offline_state(
                                    request,
                                    scheduled,
                                    |predictor, acoustic| {
                                        model.transcribe_nemotron_with_details_and_prompt_physical(
                                            chunk_audio,
                                            sr,
                                            language,
                                            asr_prompt,
                                            predictor,
                                            acoustic,
                                        )
                                    },
                                )?
                            }
                        };
                        if matches!(family, ModelFamily::GraniteSpeechAsr) {
                            details = Self::recover_granite_chunk(
                                &model,
                                chunk_audio,
                                sr,
                                language,
                                granite_task,
                                asr_prompt,
                                prefix_text,
                                &chunk_generation_options,
                                details,
                                request,
                                scheduled,
                                invocation_context,
                            )?;
                        }
                        Ok(AsrChunkTranscription {
                            text: details.text,
                            diagnostics: details.diagnostics,
                        })
                    };
                if let Some(resumed) = long_form.filter(|_| !chunk_plan.chunks.is_empty()) {
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
                return Ok(Self::complete_chunked_asr(&chunk_plan, chunked));
            }

            // Granite's public auto budget is admitted at its conservative
            // ceiling. Once the audio has been decoded under that admission,
            // narrow a single-segment decode to the actual duration just as
            // the chunked path does for each chunk.
            let segment_generation_options = Self::asr_chunk_generation_options(
                request,
                family,
                samples.len(),
                sample_rate,
                &generation_options,
                invocation_context,
            );

            if request.streaming {
                if let Some(tx) = stream_tx.as_ref() {
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) = Self::stream_text_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut sequence,
                                delta.to_string(),
                            ) {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let text = match route {
                        PhysicalAsrRoute::Qwen3 => {
                            with_single_invocation_cache(request, scheduled, |cache| {
                                model.transcribe_qwen3_with_callback_and_prompt_physical(
                                    samples,
                                    sample_rate,
                                    language,
                                    asr_prompt,
                                    &mut emit,
                                    cache,
                                )
                            })?
                        }
                        PhysicalAsrRoute::VibeVoice => {
                            with_vibevoice_invocation_state(request, scheduled, |leases| {
                                model
                                    .transcribe_vibevoice_with_callback_and_prompt_and_options_physical(
                                        samples,
                                        sample_rate,
                                        language,
                                        asr_prompt,
                                        segment_generation_options.clone(),
                                        leases,
                                        &mut emit,
                                    )
                            })?
                        }
                        PhysicalAsrRoute::GraniteSpeech => {
                            with_single_invocation_cache(request, scheduled, |cache| {
                                model
                                    .transcribe_granite_speech_with_callback_and_prompt_and_options_physical(
                                        samples,
                                        sample_rate,
                                        language,
                                        asr_prompt,
                                        segment_generation_options.clone(),
                                        cache,
                                        &mut emit,
                                    )
                            })?
                        }
                        PhysicalAsrRoute::GraniteSpeech5 => {
                            with_single_invocation_cache(request, scheduled, |cache| {
                                model
                                    .transcribe_granite_speech5_physical(
                                        samples,
                                        sample_rate,
                                        &granite_speech5_request,
                                        segment_generation_options.clone(),
                                        cache,
                                        Some(&mut emit),
                                    )
                                    .map(|details| details.text)
                            })?
                        }
                        PhysicalAsrRoute::Whisper => with_whisper_invocation_state(
                            request,
                            scheduled,
                            |self_kv, cross_kv| {
                                model.transcribe_whisper_with_callback_and_prompt_physical(
                                    samples,
                                    sample_rate,
                                    language,
                                    asr_prompt,
                                    self_kv,
                                    cross_kv,
                                    &mut emit,
                                )
                            },
                        )?,
                        PhysicalAsrRoute::Parakeet => {
                            with_single_invocation_tensor(request, scheduled, |state| {
                                model.transcribe_parakeet_with_callback_physical(
                                    samples,
                                    sample_rate,
                                    language,
                                    state,
                                    &mut emit,
                                )
                            })?
                        }
                        PhysicalAsrRoute::Nemotron => with_nemotron_offline_state(
                            request,
                            scheduled,
                            |predictor, acoustic| {
                                model.transcribe_nemotron_with_callback_and_prompt_physical(
                                    samples,
                                    sample_rate,
                                    language,
                                    asr_prompt,
                                    predictor,
                                    acoustic,
                                    &mut emit,
                                )
                            },
                        )?,
                    };
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut sequence,
                    )?;
                    return Ok(AsrTranscriptionStep::Complete(text, None));
                }
            }
            let details = match route {
                PhysicalAsrRoute::Qwen3 => {
                    with_single_invocation_cache(request, scheduled, |cache| {
                        model.transcribe_qwen3_with_details_and_prompt_physical(
                            samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            cache,
                        )
                    })?
                }
                PhysicalAsrRoute::VibeVoice => {
                    with_vibevoice_invocation_state(request, scheduled, |leases| {
                        model.transcribe_vibevoice_with_details_and_prompt_and_options_physical(
                            samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            segment_generation_options.clone(),
                            leases,
                        )
                    })?
                }
                PhysicalAsrRoute::GraniteSpeech => {
                    with_single_invocation_cache(request, scheduled, |cache| {
                        model
                            .transcribe_granite_speech_with_details_and_prompt_and_options_physical(
                                samples,
                                sample_rate,
                                language,
                                granite_task,
                                asr_prompt,
                                segment_generation_options.clone(),
                                cache,
                            )
                    })?
                }
                PhysicalAsrRoute::GraniteSpeech5 => {
                    with_single_invocation_cache(request, scheduled, |cache| {
                        model.transcribe_granite_speech5_physical(
                            samples,
                            sample_rate,
                            &granite_speech5_request,
                            segment_generation_options.clone(),
                            cache,
                            None,
                        )
                    })?
                }
                PhysicalAsrRoute::Whisper => {
                    with_whisper_invocation_state(request, scheduled, |self_kv, cross_kv| {
                        model.transcribe_whisper_with_details_and_prompt_physical(
                            samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            self_kv,
                            cross_kv,
                        )
                    })?
                }
                PhysicalAsrRoute::Parakeet => {
                    with_single_invocation_tensor(request, scheduled, |state| {
                        model.transcribe_parakeet_with_details_physical(
                            samples,
                            sample_rate,
                            language,
                            state,
                        )
                    })?
                }
                PhysicalAsrRoute::Nemotron => {
                    with_nemotron_offline_state(request, scheduled, |predictor, acoustic| {
                        model.transcribe_nemotron_with_details_and_prompt_physical(
                            samples,
                            sample_rate,
                            language,
                            asr_prompt,
                            predictor,
                            acoustic,
                        )
                    })?
                }
            };
            Ok(AsrTranscriptionStep::Complete(
                details.text,
                details.diagnostics,
            ))
        });
        let Some(transcription) = settle_atomic_asr_cancellation(request, transcription)? else {
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        };
        let (text, asr_diagnostics) = match transcription {
            AsrTranscriptionStep::Complete(text, diagnostics) => (text, diagnostics),
            AsrTranscriptionStep::Yielded(state) => {
                return self.yield_long_form_chunk(request, scheduled, session, *state);
            }
        };
        let asr_diagnostics = Self::with_audio_decode_timing(asr_diagnostics, audio_decode_ms);

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
            asr_diagnostics,
            error: None,
            logprobs: Vec::new(),
        }))
    }

    /// Stage one long-form chunk's run until the core commits its dispatch
    /// (with this dispatch's staged stream output), and yield: the next
    /// chunk runs in a later dispatch.
    fn yield_long_form_chunk(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        session: SessionKey,
        state: LongFormAsrState,
    ) -> Result<ModelSessionResult> {
        self.long_form_asr
            .stage(scheduled.plan_id, session, state)?;
        Ok(ModelSessionResult::yielded(
            ExecutorOutput {
                request_id: request.id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 0,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
                logprobs: Vec::new(),
            },
            crate::engine::YieldReason::QuantumExhausted,
        )
        .requiring_pending_quantum())
    }

    fn complete_chunked_asr(
        plan: &PlannedAsrChunks,
        chunked: ChunkedAsrTranscription,
    ) -> AsrTranscriptionStep {
        let (text, diagnostics) = Self::chunked_asr_result(plan, chunked);
        AsrTranscriptionStep::Complete(text, diagnostics)
    }

    /// Run the next chunk of a long-form row that dispatches its chunks one
    /// at a time, starting the run on the row's first dispatch. Yields the
    /// run until its last chunk, then completes it.
    #[allow(clippy::too_many_arguments)]
    fn run_long_form_chunk<F>(
        request: &EngineCoreRequest,
        resumed: Option<LongFormAsrState>,
        audio: &AsrExecutionAudio,
        sample_rate: u32,
        audio_decode_ms: f64,
        stream_tx: Option<&super::super::request::StreamStagingBuffer>,
        stream_policy: super::super::request::EngineStreamPolicy,
        plan: &PlannedAsrChunks,
        mut transcribe_chunk: F,
    ) -> Result<AsrTranscriptionStep>
    where
        F: FnMut(&[f32], u32, &str, Option<&mut dyn FnMut(&str)>) -> Result<AsrChunkTranscription>,
    {
        let stream = ChunkStream {
            request_id: &request.id,
            tx: stream_tx,
            policy: stream_policy,
        };
        let mut state = match resumed {
            Some(state) => state,
            None => {
                let mut sequence = 0;
                let samples = audio.shared();
                let run = ChunkedAsrRun::start(
                    &stream,
                    &mut sequence,
                    samples.len(),
                    sample_rate,
                    &plan.chunks,
                    &plan.config,
                )?;
                LongFormAsrState {
                    run,
                    sequence,
                    samples,
                    sample_rate,
                    audio_decode_ms,
                }
            }
        };
        state.run.run_chunk(
            &stream,
            &mut state.sequence,
            &|| request.is_cancelled(),
            &state.samples,
            sample_rate,
            &plan.chunks,
            Self::asr_chunk_stream_options(),
            &mut transcribe_chunk,
        )?;
        if !state.run.is_complete(&plan.chunks) {
            return Ok(AsrTranscriptionStep::Yielded(Box::new(state)));
        }
        let LongFormAsrState {
            run, mut sequence, ..
        } = state;
        let chunked = run.finish(&stream, &mut sequence, &plan.chunks, sample_rate)?;
        Ok(Self::complete_chunked_asr(plan, chunked))
    }

    /// Transcript and diagnostics of a chunked run of `plan`, with the word
    /// timings of a transcript merged by time as `timestamp_words`.
    fn chunked_asr_result(
        plan: &PlannedAsrChunks,
        chunked: ChunkedAsrTranscription,
    ) -> (String, Option<serde_json::Value>) {
        let mut diagnostics = plan.diagnostics_with_chunk_transcriptions(chunked.chunk_diagnostics);
        if let (Some(words), Some(root)) = (chunked.timestamp_words, diagnostics.as_object_mut()) {
            root.insert("timestamp_words".to_string(), timestamp_words_json(&words));
        }
        (chunked.text, Some(diagnostics))
    }

    fn with_audio_decode_timing(
        diagnostics: Option<serde_json::Value>,
        audio_decode_ms: f64,
    ) -> Option<serde_json::Value> {
        let mut payload = diagnostics.unwrap_or_else(|| serde_json::json!({}));
        if !payload.is_object() {
            payload = serde_json::json!({
                "model_diagnostics": payload
            });
        }

        if let Some(root) = payload.as_object_mut() {
            let timings = root
                .entry("timings_ms")
                .or_insert_with(|| serde_json::json!({}));
            if let Some(timings_obj) = timings.as_object_mut() {
                timings_obj.insert(
                    "audio_decode".to_string(),
                    serde_json::json!(audio_decode_ms),
                );
            } else {
                root.insert(
                    "timings_ms".to_string(),
                    serde_json::json!({ "audio_decode": audio_decode_ms }),
                );
            }
        }

        Some(payload)
    }

    fn asr_generation_options(request: &EngineCoreRequest) -> NativeAsrGenerationOptions {
        NativeAsrGenerationOptions {
            max_new_tokens: request.params.max_tokens.max(1),
            stop_token_ids: request.params.stop_token_ids.clone(),
            stop_sequences: request.params.stop_sequences.clone(),
        }
    }

    /// Effective invocation context for the loaded variant, when the executor
    /// shares the lifecycle model registry. Executors without a registry
    /// (mock workers, standalone tests) fall back to hint-only planning.
    fn asr_effective_context(&self, variant: crate::model::ModelVariant) -> Option<usize> {
        self.config
            .model_registry
            .as_ref()
            .and_then(|registry| registry.effective_context(variant))
    }

    /// `invocation_context` is the effective context and the family's audio
    /// token rate, when both are known; Granite's automatic budget is clamped
    /// to the decode room it leaves after the chunk's prompt.
    fn asr_chunk_generation_options(
        request: &EngineCoreRequest,
        family: ModelFamily,
        chunk_sample_count: usize,
        sample_rate: u32,
        base: &NativeAsrGenerationOptions,
        invocation_context: Option<(usize, f32)>,
    ) -> NativeAsrGenerationOptions {
        let mut options = base.clone();
        if request.asr_auto_max_tokens
            && matches!(family, ModelFamily::GraniteSpeechAsr)
            && sample_rate > 0
        {
            let chunk_seconds = chunk_sample_count as f32 / sample_rate as f32;
            let budget =
                granite_asr_planned_max_tokens(request, chunk_seconds).min(base.max_new_tokens);
            options.max_new_tokens = invocation_context.map_or(budget, |(context, rate)| {
                budget.min(invocation_decode_room(context, rate, chunk_seconds))
            });
        }
        if request.asr_auto_max_tokens
            && matches!(family, ModelFamily::GraniteSpeech5Asr)
            && sample_rate > 0
        {
            let chunk_seconds = chunk_sample_count as f32 / sample_rate as f32;
            let budget = granite_speech5_auto_max_tokens_for_duration(chunk_seconds)
                .min(base.max_new_tokens);
            options.max_new_tokens = invocation_context.map_or(budget, |(context, rate)| {
                budget.min(invocation_decode_room(context, rate, chunk_seconds))
            });
        }
        // VibeVoice's automatic budget is admitted at the reference ceiling;
        // each chunk decodes with its own duration budget.
        if request.asr_auto_max_tokens
            && matches!(family, ModelFamily::VibeVoiceAsr)
            && sample_rate > 0
        {
            let chunk_seconds = chunk_sample_count as f32 / sample_rate as f32;
            let budget = crate::runtime::vibevoice_asr_planned_max_tokens(request, chunk_seconds)
                .min(base.max_new_tokens);
            options.max_new_tokens = invocation_context.map_or(budget, |(context, rate)| {
                budget.min(invocation_decode_room(context, rate, chunk_seconds))
            });
        }
        options.max_new_tokens = options.max_new_tokens.max(1);
        options
    }
}

/// Decode budget the invocation solver plans an `audio_seconds` chunk with.
/// Granite plans with the budget its chunks are decoded with (GRN-6); the
/// other decoder-prompt families keep the shared reserve.
pub(super) fn asr_planned_decode_tokens(
    request: &EngineCoreRequest,
    family: ModelFamily,
    audio_seconds: f32,
) -> usize {
    match family {
        ModelFamily::GraniteSpeechAsr => granite_asr_planned_max_tokens(request, audio_seconds),
        ModelFamily::GraniteSpeech5Asr => {
            crate::runtime::granite_speech5_asr_planned_max_tokens(request, audio_seconds)
        }
        ModelFamily::VibeVoiceAsr => {
            crate::runtime::vibevoice_asr_planned_max_tokens(request, audio_seconds)
        }
        _ => MAX_ASR_NEW_TOKENS,
    }
}

#[cfg(test)]
#[path = "handler_asr_tests.rs"]
mod tests;
