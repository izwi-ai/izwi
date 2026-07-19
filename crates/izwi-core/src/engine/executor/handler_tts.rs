use std::time::Instant;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::{
    BatchedSpeakerRequest, SpeakerReference, TtsGenerationParams, TtsStreamingConfig,
};
use crate::runtime::audio_io::decode_reference_audio_base64;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::{ExecutionMode, NativeBatchMode};
use super::state::ActiveQwenTtsDecode;
use super::{ExecutorOutput, ExecutorPhaseTiming, ModelSessionResult, NativeExecutor};

pub(super) struct QwenTtsBatchResult {
    pub(super) outputs: Vec<ExecutorOutput>,
    pub(super) tensor_width: usize,
}

impl NativeExecutor {
    pub(super) fn try_qwen_tts_batch(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Option<Result<QwenTtsBatchResult>> {
        if scheduled.is_empty() || scheduled.iter().any(|item| !item.is_prefill) {
            return None;
        }
        let mut ordered = Vec::with_capacity(scheduled.len());
        for item in scheduled {
            let request = requests
                .iter()
                .copied()
                .find(|request| request.id == item.request_id)?;
            if request.streaming
                || request.is_cancelled()
                || request
                    .text
                    .as_deref()
                    .is_none_or(|text| text.trim().is_empty())
                || request.has_tts_reference_for_execution()
                || request.model_variant.map(|v| v.family())
                    != Some(crate::catalog::ModelFamily::Qwen3Tts)
                || !request
                    .model_variant
                    .and_then(|variant| variant.speech_capabilities())
                    .is_some_and(|capabilities| capabilities.supports_builtin_voices)
            {
                return None;
            }
            ordered.push(request);
        }
        let variant = ordered[0].model_variant;
        if ordered
            .iter()
            .any(|request| request.model_variant != variant)
        {
            return None;
        }
        if ordered.iter().any(|request| {
            <NativeExecutor as super::ModelExecutor>::execution_profile(self, request).is_none_or(
                |profile| {
                    profile.mode != ExecutionMode::Atomic
                        || profile.prefill_batch != NativeBatchMode::Static
                },
            )
        }) {
            // The atomic tensor API is valid only when the exact model/backend
            // rollout produced a static execution plan. In particular, an
            // ordinary singleton Qwen request must stay on its incremental
            // sequence adapter instead of generating a full utterance against
            // a one-step sequence plan.
            return None;
        }
        let model_instance_id = ordered[0].model_instance_id()?;
        let (model, model_lease) = match self.qwen_model_for_request(ordered[0]) {
            Ok(model) => model,
            Err(err) => return Some(Err(err)),
        };
        let mut model_leases = Vec::with_capacity(ordered.len());
        model_leases.extend(model_lease);
        for request in ordered.iter().skip(1) {
            if request.model_instance_id() != Some(model_instance_id) {
                // Requests admitted on opposite sides of an unload/reload
                // boundary must never share one native tensor batch.
                return None;
            }
            let (_request_model, request_lease) = match self.qwen_model_for_request(request) {
                Ok(model) => model,
                Err(err) => return Some(Err(err)),
            };
            model_leases.extend(request_lease);
        }
        let speakers = model
            .available_speakers()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        if speakers.is_empty()
            || ordered.iter().any(|request| {
                request
                    .params
                    .speaker
                    .as_deref()
                    .or(request.params.voice.as_deref())
                    .filter(|speaker| !speaker.trim().is_empty())
                    .is_some_and(|requested| {
                        !speakers
                            .iter()
                            .any(|speaker| speaker.eq_ignore_ascii_case(requested))
                    })
            })
        {
            // Let the per-request path isolate invalid speaker errors instead
            // of poisoning otherwise valid members of a tensor batch.
            return None;
        }
        let max_batch_size =
            <NativeExecutor as super::ModelExecutor>::execution_capabilities(self, ordered[0])
                .max_batch_size;
        if scheduled.len() > max_batch_size {
            return None;
        }

        Some((|| {
            let _model_leases = model_leases;
            let batch = ordered
                .iter()
                .map(|request| BatchedSpeakerRequest {
                    text: request.text.clone().unwrap_or_default(),
                    speaker: request
                        .params
                        .speaker
                        .clone()
                        .or_else(|| request.params.voice.clone())
                        .filter(|speaker| !speaker.trim().is_empty())
                        .unwrap_or_else(|| speakers[0].clone()),
                    language: request.language.clone(),
                    instruct: request.voice_description.clone(),
                    params: Self::to_tts_params(request),
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            let generated =
                Self::run_blocking(|| model.generate_with_speaker_params_batch(&batch))?;
            let per_request_ms =
                started.elapsed().as_secs_f64() * 1000.0 / generated.outputs.len().max(1) as f64;
            let outputs = ordered
                .iter()
                .zip(scheduled)
                .zip(generated.outputs)
                .map(|((request, _scheduled), output)| ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::new(output.samples, 24_000)),
                    text: None,
                    input_transcription: None,
                    tokens_processed: request.num_prompt_tokens(),
                    tokens_generated: output.frames_generated,
                    finished: true,
                    phase_timing_override: Some(ExecutorPhaseTiming {
                        decode_ms: Some(per_request_ms),
                        decode_steps: Some(output.frames_generated as u32),
                        ..Default::default()
                    }),
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(QwenTtsBatchResult {
                outputs,
                tensor_width: generated.max_tensor_batch_width,
            })
        })())
    }

    pub(super) fn to_tts_params(request: &EngineCoreRequest) -> TtsGenerationParams {
        let model_max_frames = request
            .model_variant
            .and_then(|variant| variant.tts_max_output_frames_hint())
            .unwrap_or(crate::model::ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
        TtsGenerationParams {
            temperature: request.params.temperature.max(0.0),
            top_p: request.params.top_p.clamp(0.0, 1.0),
            top_k: if request.params.top_k == 0 {
                50
            } else {
                request.params.top_k
            },
            repetition_penalty: request.params.repetition_penalty.max(1.0),
            max_frames: if request.params.max_tokens == 0 {
                model_max_frames
            } else {
                request
                    .params
                    .max_tokens
                    .clamp(16, model_max_frames.max(16))
            },
        }
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
                let available_speakers = model.available_speakers();
                let requested_speaker = request
                    .params
                    .speaker
                    .as_deref()
                    .or(request.params.voice.as_deref())
                    .filter(|s| !s.trim().is_empty());
                let reference = Self::reference_from_request(request)?;
                let stream_config = if stream_tx.is_some() {
                    TtsStreamingConfig::default()
                } else {
                    TtsStreamingConfig::final_only()
                };
                let normalization_ms = normalization_started.elapsed().as_secs_f64() * 1000.0;
                let prefill_started = Instant::now();

                let decode_state = if let Some(reference) = reference {
                    Self::run_blocking(|| {
                        model.start_decode_with_voice_clone_params(
                            text,
                            &reference,
                            language,
                            &params,
                            stream_config,
                        )
                    })?
                } else if available_speakers.is_empty() {
                    Self::run_blocking(|| {
                        model.start_decode_with_text_params(
                            text,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            stream_config,
                        )
                    })?
                } else {
                    let speaker_to_use =
                        requested_speaker.unwrap_or_else(|| available_speakers[0].as_str());
                    Self::run_blocking(|| {
                        model.start_decode_with_speaker_params(
                            text,
                            speaker_to_use,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            stream_config,
                        )
                    })?
                };

                ActiveQwenTtsDecode {
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
                }
            };

            let decode_iterations = if scheduled.is_prefill {
                1
            } else {
                scheduled.num_tokens.max(1)
            };
            let mut total_tokens_generated = 0usize;
            let mut decode_steps_ran = 0usize;
            let mut finished = false;

            for _ in 0..decode_iterations {
                if request.is_cancelled() {
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                let step = Self::run_blocking(|| {
                    active_state.model.tts_decode_step(&mut active_state.state)
                })?;
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
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{InputRange, SequencePhase, WorkUnit};
    use crate::model::ModelVariant;
    use base64::Engine;

    #[test]
    fn default_qwen_tts_profile_never_enters_atomic_batch_dispatch() {
        let executor = NativeExecutor::new(super::super::WorkerConfig::default());
        let mut request = EngineCoreRequest::tts("hello from the sequence adapter");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: request.num_prompt_tokens().max(1),
            is_prefill: true,
            block_ids: Vec::new(),
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange {
                    start: 0,
                    end: request.num_prompt_tokens(),
                },
                max_output_steps: 1,
            },
        };

        assert!(executor
            .try_qwen_tts_batch(&[&request], &[scheduled])
            .is_none());
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
