use crate::error::{Error, Result};
use crate::models::architectures::lfm2::audio::LFM2_DEFAULT_S2S_PROMPT;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveSpeechToSpeechDecode;
use super::{decode_audio_base64_with_rate, ExecutorOutput, NativeExecutor};

impl NativeExecutor {
    pub(super) fn speech_to_speech_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let system_prompt = request
            .system_prompt
            .as_deref()
            .unwrap_or(LFM2_DEFAULT_S2S_PROMPT);
        let language = request.language.as_deref();
        let resolved_temperature = request.params.audio_temperature.unwrap_or(1.0);
        let resolved_top_k = request.params.audio_top_k.unwrap_or_else(|| {
            if request.params.top_k > 0 {
                request.params.top_k
            } else {
                4
            }
        });
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);

        if let Some(tx) = stream_tx.as_ref() {
            let model = self.with_registry(|registry| {
                registry.try_get_lfm2(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!("LFM2 model {variant} is not loaded"))
                })
            })?;

            let mut active_state = {
                let mut guard = self.speech_to_speech_decode_states.lock().map_err(|_| {
                    Error::InferenceError(
                        "Speech-to-speech decode state mutex poisoned".to_string(),
                    )
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
                let audio_b64 = request.audio_input.as_deref().ok_or_else(|| {
                    Error::InvalidInput("Speech-to-speech request missing audio input".to_string())
                })?;
                let (samples, sample_rate) = decode_audio_base64_with_rate(audio_b64)?;
                let decode_state = Self::run_blocking(|| {
                    model.start_speech_to_speech_decode(
                        &samples,
                        sample_rate,
                        language,
                        Some(system_prompt),
                        Some(resolved_temperature),
                        Some(resolved_top_k),
                        max_new_tokens,
                    )
                })?;
                ActiveSpeechToSpeechDecode {
                    variant,
                    state: decode_state,
                    prompt_accounted: false,
                    last_tokens_generated: 0,
                    stream_sequence: 0,
                    audio_frames_accum: Vec::new(),
                    emitted_samples: 0,
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
            let mut final_text = String::new();
            let mut finished = false;

            for _ in 0..decode_iterations {
                let step = Self::run_blocking(|| {
                    model.speech_to_speech_decode_step(&mut active_state.state)
                })?;
                decode_steps_ran = decode_steps_ran.saturating_add(1);
                let step_tokens_generated = step
                    .tokens_generated
                    .saturating_sub(active_state.last_tokens_generated);
                active_state.last_tokens_generated = step.tokens_generated;
                total_tokens_generated =
                    total_tokens_generated.saturating_add(step_tokens_generated);
                final_text = step.text.clone();

                if !step.delta.is_empty() {
                    Self::stream_text(
                        tx,
                        &request.id,
                        &mut active_state.stream_sequence,
                        step.delta.clone(),
                    )?;
                }

                if let Some(frame) = step.audio_frame.as_ref() {
                    active_state.audio_frames_accum.push(frame.clone());
                    // Decode over the accumulated frame prefix so streamed chunks stay
                    // consistent with full-utterance decode.
                    let all_samples = Self::run_blocking(|| {
                        model.decode_audio_frames(&active_state.audio_frames_accum)
                    })?;
                    let chunk_samples = Self::next_audio_delta_stable(
                        &all_samples,
                        &mut active_state.emitted_samples,
                        Self::LFM2_STREAM_TAIL_HOLDBACK_SAMPLES,
                        step.finished,
                    );
                    active_state.audio_samples_accum = all_samples;
                    if !chunk_samples.is_empty() {
                        Self::stream_audio(
                            tx,
                            &request.id,
                            &mut active_state.stream_sequence,
                            chunk_samples,
                            24_000,
                            false,
                        )?;
                    }
                }

                if step.finished {
                    if !active_state.audio_frames_accum.is_empty() {
                        let all_samples = Self::run_blocking(|| {
                            model.decode_audio_frames(&active_state.audio_frames_accum)
                        })?;
                        let final_tail = Self::next_audio_delta_stable(
                            &all_samples,
                            &mut active_state.emitted_samples,
                            Self::LFM2_STREAM_TAIL_HOLDBACK_SAMPLES,
                            true,
                        );
                        active_state.audio_samples_accum = all_samples;
                        if !final_tail.is_empty() {
                            Self::stream_audio(
                                tx,
                                &request.id,
                                &mut active_state.stream_sequence,
                                final_tail,
                                24_000,
                                false,
                            )?;
                        }
                    }
                    Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
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
                let mut guard = self.speech_to_speech_decode_states.lock().map_err(|_| {
                    Error::InferenceError(
                        "Speech-to-speech decode state mutex poisoned".to_string(),
                    )
                })?;
                guard.insert(request.id.clone(), active_state);
            }

            return Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(finished_samples, 24_000)),
                text: Some(final_text),
                tokens_processed,
                tokens_generated: total_tokens_generated,
                finished,
                error: None,
            });
        }

        let audio_b64 = request.audio_input.as_deref().ok_or_else(|| {
            Error::InvalidInput("Speech-to-speech request missing audio input".to_string())
        })?;
        let (samples, sample_rate) = decode_audio_base64_with_rate(audio_b64)?;
        let model = self.with_registry(|registry| {
            registry
                .try_get_lfm2(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("LFM2 model {variant} is not loaded")))
        })?;
        let (text, output_samples) = Self::run_blocking(|| {
            let mut sink = |_delta: &str| {};
            model.speech_to_speech_with_callback(
                &samples,
                sample_rate,
                language,
                Some(system_prompt),
                Some(resolved_temperature),
                Some(resolved_top_k),
                max_new_tokens,
                &mut sink,
            )
        })?;

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(output_samples.clone(), 24_000)),
            text: Some(text),
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (output_samples.len() / 256).max(1),
            finished: true,
            error: None,
        })
    }
}
