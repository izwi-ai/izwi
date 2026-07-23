use std::time::Instant;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::{
    SpeakerReference, TalkerPhysicalCache, TtsGenerationParams, TtsStreamingConfig,
};
use crate::runtime::audio_io::decode_reference_audio_base64;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveQwenTtsDecode;
use super::{ExecutorOutput, ExecutorPhaseTiming, ModelSessionResult, NativeExecutor};

fn validate_qwen_tts_physical_bindings(
    has_managed_runtime: bool,
    has_talker_cache: bool,
    has_tensor_arena: bool,
    has_tensor_reservation: bool,
) -> Result<()> {
    if has_managed_runtime != has_talker_cache {
        return Err(Error::InferenceError(
            "physical Qwen TTS execution requires its exact talker reservation".to_string(),
        ));
    }
    if !has_talker_cache {
        return Err(Error::InferenceError(
            "Qwen TTS no longer supports model-owned decode caches".to_string(),
        ));
    }
    if has_tensor_arena != has_tensor_reservation {
        return Err(Error::InferenceError(
            "physical Qwen TTS tensor state requires its exact row reservation".into(),
        ));
    }
    Ok(())
}

fn qwen_tts_decode_iterations(scheduled: &ScheduledRequest) -> usize {
    if scheduled.is_prefill {
        0
    } else {
        scheduled.num_tokens.max(1)
    }
}

impl NativeExecutor {
    pub(super) fn to_tts_params(request: &EngineCoreRequest) -> TtsGenerationParams {
        request.qwen_tts_generation_params()
    }

    pub(super) fn reference_from_request(
        request: &EngineCoreRequest,
    ) -> Result<Option<SpeakerReference>> {
        if !request.has_tts_reference_for_execution() {
            return Ok(None);
        }

        let ref_audio = request.tts_reference_audio_for_execution().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        let ref_text = request.tts_reference_text_for_execution().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        if ref_text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "reference_text cannot be empty".to_string(),
            ));
        }

        // Reference conditioning has a deliberately tighter contract than
        // inference audio (32 MiB decoded and 30 seconds). Keep decoding on
        // the executor's blocking boundary so compressed input never stalls
        // the async engine worker.
        let (audio_samples, sample_rate) =
            Self::run_blocking(|| decode_reference_audio_base64(ref_audio))?;
        Ok(Some(SpeakerReference {
            audio_samples,
            text: ref_text.to_string(),
            sample_rate,
        }))
    }

    pub(super) fn qwen_tts_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.qwen_tts_request_with_managed_cache(request, scheduled, None, None)
    }

    pub(super) fn qwen_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut talker_cache: Option<TalkerPhysicalCache>,
        tensor_reservation: Option<crate::engine::ManagedTensorStateReservation>,
    ) -> Result<ModelSessionResult> {
        let tensor_arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state().cloned());
        validate_qwen_tts_physical_bindings(
            request.managed_cache_runtime().is_some(),
            talker_cache.is_some(),
            tensor_arena.is_some(),
            tensor_reservation.is_some(),
        )?;
        if scheduled.is_prefill
            && (scheduled.num_computed_tokens != 0
                || scheduled.num_tokens != request.num_prompt_tokens())
        {
            return Err(Error::InvalidInput(
                "physical Qwen TTS requires one exact full-prompt prefill quantum".to_string(),
            ));
        }
        let execution_started = Instant::now();
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let variant = request.model_variant;
        let params = Self::to_tts_params(request);
        let language = request.language.as_deref();
        let session = scheduled.session_key();

        {
            let mut active_state = {
                let mut guard = self.qwen_tts_decode_states.lock().map_err(|_| {
                    Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
                })?;
                guard.remove(&session)
            };

            if active_state
                .as_ref()
                .map(|state| state.variant != variant)
                .unwrap_or(false)
            {
                active_state = None;
            }
            if let Some(state) = active_state.as_mut() {
                state.state.install_retained_talker_reservation(
                    talker_cache.take().ok_or_else(|| {
                        Error::InferenceError(
                            "active Qwen TTS state lost its talker reservation".to_string(),
                        )
                    })?,
                )?;
                if let (Some(arena), Some(reservation)) =
                    (tensor_arena.as_ref(), tensor_reservation)
                {
                    state.state.bind_tensor_sequence(reservation.sequence)?;
                    state.state.restore_tensor_state(arena)?;
                }
            }
            let (model, new_model_lease) = if let Some(state) = active_state.as_ref() {
                (state.model.clone(), None)
            } else {
                let (model, lease) = self.qwen_model_for_request(request)?;
                (model, lease)
            };
            let model_arc = model;
            let model = model_arc.as_ref();

            let mut active_state = if let Some(state) = active_state {
                state
            } else {
                if request.is_cancelled() {
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                let normalization_started = Instant::now();
                let text = request
                    .text
                    .as_deref()
                    .ok_or_else(|| Error::InvalidInput("TTS request missing text".to_string()))?;
                let prepared = request.prepared_qwen_tts_input_for_executor()?;
                let stream_config = if stream_tx.is_some() {
                    TtsStreamingConfig::default()
                } else {
                    TtsStreamingConfig::final_only()
                };
                let normalization_ms = normalization_started.elapsed().as_secs_f64() * 1000.0;
                let prefill_started = Instant::now();
                let cache = talker_cache.take().ok_or_else(|| {
                    Error::InferenceError(
                        "Qwen TTS prefill lost its retained talker reservation".to_string(),
                    )
                })?;

                let decode_state = if let Some(reference) = prepared.reference.as_deref() {
                    Self::run_blocking(|| {
                        model.start_physical_decode_with_voice_clone_params(
                            text,
                            reference,
                            language,
                            &params,
                            stream_config,
                            cache,
                        )
                    })?
                } else if let Some(speaker) = prepared.speaker.as_deref() {
                    Self::run_blocking(|| {
                        model.start_physical_decode_with_speaker_params(
                            text,
                            speaker,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            stream_config,
                            cache,
                        )
                    })?
                } else {
                    Self::run_blocking(|| {
                        model.start_physical_decode_with_text_params(
                            text,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            stream_config,
                            cache,
                        )
                    })?
                };
                if decode_state.talker_context_len() != prepared.prefill_tokens {
                    return Err(Error::InferenceError(format!(
                        "Qwen TTS runtime prefill produced {} tokens, but admission authorized {}",
                        decode_state.talker_context_len(),
                        prepared.prefill_tokens
                    )));
                }

                let mut active = ActiveQwenTtsDecode {
                    variant,
                    model: model_arc.clone(),
                    _model_lease: new_model_lease,
                    state: decode_state,
                    last_frames_generated: 0,
                    stream_sequence: 0,
                    audio_samples_accum: Vec::new(),
                    execution_started,
                    normalization_ms,
                    prefill_ms: prefill_started.elapsed().as_secs_f64() * 1000.0,
                    sampling_ms: 0.0,
                    decode_ms: 0.0,
                    codec_ms: 0.0,
                    postprocess_ms: 0.0,
                    first_output_ms_since_start: None,
                    decode_steps: 0,
                };
                if let Some(reservation) = tensor_reservation {
                    active.state.bind_tensor_sequence(reservation.sequence)?;
                }
                active
            };

            let decode_iterations = qwen_tts_decode_iterations(scheduled);
            let mut total_tokens_generated = 0usize;
            let mut decode_steps_ran = 0usize;
            let mut finished = false;

            for _ in 0..decode_iterations {
                if request.is_cancelled() {
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                let mut predictor = super::invocation_paged_lease_for_row(request, scheduled)?;
                let step = Self::run_blocking(|| {
                    active_state
                        .model
                        .tts_decode_step_physical(&mut active_state.state, predictor.cache_mut())
                })?;
                let _predictor_completion = predictor.release()?;
                if request.is_cancelled() {
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                active_state.sampling_ms += step.sampling_ms;
                active_state.decode_ms += step.decode_ms;
                active_state.codec_ms += step.codec_ms;
                active_state.decode_steps = active_state.decode_steps.saturating_add(1);
                decode_steps_ran = decode_steps_ran.saturating_add(1);
                let step_tokens_generated = step
                    .frames_generated
                    .saturating_sub(active_state.last_frames_generated);
                active_state.last_frames_generated = step.frames_generated;
                total_tokens_generated =
                    total_tokens_generated.saturating_add(step_tokens_generated);

                if !step.samples.is_empty() {
                    if active_state.first_output_ms_since_start.is_none() {
                        active_state.first_output_ms_since_start =
                            Some(active_state.execution_started.elapsed().as_secs_f64() * 1000.0);
                    }
                    active_state
                        .audio_samples_accum
                        .extend_from_slice(&step.samples);
                    if let Some(tx) = stream_tx.as_ref() {
                        Self::stream_audio_with_policy(
                            tx,
                            stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                            step.samples.clone(),
                            24_000,
                            false,
                        )?;
                    }
                }

                if step.finished {
                    if let Some(tx) = stream_tx.as_ref() {
                        Self::stream_final_marker_with_policy(
                            tx,
                            stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                        )?;
                    }
                    finished = true;
                    break;
                }
            }

            let tokens_processed = if scheduled.is_prefill {
                request.num_prompt_tokens()
            } else {
                decode_steps_ran.max(1)
            };
            let postprocess_started = Instant::now();
            let finished_samples = if finished {
                active_state.audio_samples_accum.clone()
            } else {
                Vec::new()
            };
            active_state.postprocess_ms += postprocess_started.elapsed().as_secs_f64() * 1000.0;

            let phase_timing_override = Some(ExecutorPhaseTiming {
                normalization_ms: Some(active_state.normalization_ms),
                prefill_ms: Some(active_state.prefill_ms),
                decode_ms: Some(active_state.decode_ms),
                sampling_ms: Some(active_state.sampling_ms),
                codec_ms: Some(active_state.codec_ms),
                postprocess_ms: Some(active_state.postprocess_ms),
                first_output_ms_since_start: active_state.first_output_ms_since_start,
                prefill_steps: Some(1),
                decode_steps: Some(active_state.decode_steps),
                ..ExecutorPhaseTiming::default()
            });

            if let Some(arena) = tensor_arena.as_ref() {
                active_state
                    .state
                    .stage_tensor_state(arena, scheduled.plan_id)?;
            }
            let managed_cache_completions = active_state.state.take_managed_write_completions();
            if !finished {
                let mut guard = self.qwen_tts_decode_states.lock().map_err(|_| {
                    Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
                })?;
                guard.insert(session, active_state);
            }

            Ok(ModelSessionResult::sequence(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(finished_samples, 24_000)),
                text: None,
                input_transcription: None,
                tokens_processed,
                tokens_generated: total_tokens_generated,
                finished,
                phase_timing_override,
                asr_diagnostics: None,
                error: None,
            })
            .with_managed_cache_completions(managed_cache_completions))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn scheduled(is_prefill: bool, num_tokens: usize) -> ScheduledRequest {
        ScheduledRequest {
            plan_id: 1,
            request_id: "tts-transition".into(),
            sequence_id: 1,
            num_tokens,
            is_prefill,
            num_computed_tokens: usize::from(!is_prefill),
            work: crate::engine::WorkUnit::SequenceStep {
                phase: if is_prefill {
                    crate::engine::SequencePhase::Prefill
                } else {
                    crate::engine::SequencePhase::Decode
                },
                input: crate::engine::InputRange {
                    start: usize::from(!is_prefill),
                    end: usize::from(!is_prefill) + num_tokens,
                },
                max_output_steps: num_tokens.max(1),
            },
        }
    }

    #[test]
    fn qwen_tts_physical_binding_rejects_legacy_and_partial_ownership() {
        assert!(validate_qwen_tts_physical_bindings(true, true, true, true).is_ok());
        assert!(validate_qwen_tts_physical_bindings(false, false, false, false).is_err());
        assert!(validate_qwen_tts_physical_bindings(true, false, true, true).is_err());
        assert!(validate_qwen_tts_physical_bindings(true, true, true, false).is_err());
    }

    #[test]
    fn qwen_tts_prefill_only_hydrates_and_decode_consumes_the_row_budget() {
        assert_eq!(qwen_tts_decode_iterations(&scheduled(true, 17)), 0);
        assert_eq!(qwen_tts_decode_iterations(&scheduled(false, 0)), 1);
        assert_eq!(qwen_tts_decode_iterations(&scheduled(false, 4)), 4);
    }

    #[test]
    fn qwen_reference_audio_uses_the_thirty_second_decode_contract() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 1_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("WAV writer");
            for _ in 0..30_001 {
                writer.write_sample(0_i16).expect("WAV sample");
            }
            writer.finalize().expect("WAV finalize");
        }

        let mut request = EngineCoreRequest::tts("clone this voice");
        request.reference_audio = Some(base64::engine::general_purpose::STANDARD.encode(wav));
        request.reference_text = Some("reference transcript".to_string());

        let error = NativeExecutor::reference_from_request(&request)
            .expect_err("reference audio above thirty seconds must fail before model encoding");
        assert!(
            error.to_string().contains("production limit"),
            "unexpected reference bound error: {error}"
        );
    }
}
