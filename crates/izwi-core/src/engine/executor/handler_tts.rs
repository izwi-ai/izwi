use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::{
    SpeakerReference, TtsGenerationParams, TtsStreamingConfig,
};

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveQwenTtsDecode;
use super::{decode_audio_base64_with_rate, ExecutorOutput, NativeExecutor};

impl NativeExecutor {
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

    fn reference_from_request(request: &EngineCoreRequest) -> Result<Option<SpeakerReference>> {
        if request.reference_audio.is_none() && request.reference_text.is_none() {
            return Ok(None);
        }

        let ref_audio = request.reference_audio.as_deref().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        let ref_text = request.reference_text.as_deref().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        if ref_text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "reference_text cannot be empty".to_string(),
            ));
        }

        let (audio_samples, sample_rate) = decode_audio_base64_with_rate(ref_audio)?;
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
    ) -> Result<ExecutorOutput> {
        let stream_tx = Self::stream_sender(request);
        let variant = request.model_variant;
        let params = Self::to_tts_params(request);
        let language = request.language.as_deref();

        self.with_qwen_model(variant, |model| {
            let mut active_state = {
                let mut guard = self.qwen_tts_decode_states.lock().map_err(|_| {
                    Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
                })?;
                guard.remove(&request.id)
            };

            if active_state
                .as_ref()
                .map(|state| state.variant != variant)
                .unwrap_or(false)
            {
                active_state = None;
            }

            let mut active_state = if let Some(state) = active_state {
                state
            } else {
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
                    state: decode_state,
                    prompt_accounted: false,
                    last_frames_generated: 0,
                    stream_sequence: 0,
                    audio_samples_accum: Vec::new(),
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
                let step = Self::run_blocking(|| model.tts_decode_step(&mut active_state.state))?;
                decode_steps_ran = decode_steps_ran.saturating_add(1);
                let step_tokens_generated = step
                    .frames_generated
                    .saturating_sub(active_state.last_frames_generated);
                active_state.last_frames_generated = step.frames_generated;
                total_tokens_generated =
                    total_tokens_generated.saturating_add(step_tokens_generated);

                if !step.samples.is_empty() {
                    active_state
                        .audio_samples_accum
                        .extend_from_slice(&step.samples);
                    if let Some(tx) = stream_tx.as_ref() {
                        Self::stream_audio(
                            tx,
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
                        Self::stream_final_marker(
                            tx,
                            &request.id,
                            &mut active_state.stream_sequence,
                        )?;
                    }
                    finished = true;
                    break;
                }
            }

            let mut tokens_processed = if scheduled.is_prefill {
                scheduled.num_tokens.max(1)
            } else {
                decode_steps_ran.max(1)
            };
            if !active_state.prompt_accounted {
                active_state.prompt_accounted = true;
                tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
            }

            let finished_samples = if finished {
                active_state.audio_samples_accum.clone()
            } else {
                Vec::new()
            };

            if !finished {
                let mut guard = self.qwen_tts_decode_states.lock().map_err(|_| {
                    Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
                })?;
                guard.insert(request.id.clone(), active_state);
            }

            Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(finished_samples, 24_000)),
                text: None,
                tokens_processed,
                tokens_generated: total_tokens_generated,
                finished,
                error: None,
            })
        })
    }

}
