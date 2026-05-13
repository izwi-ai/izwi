use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use std::time::Instant;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::audio::decode_request_audio_with_rate;
use super::state::ActiveAsrDecode;
use super::{ExecutorOutput, NativeExecutor};

const MAX_ASR_NEW_TOKENS: usize = 512;

impl NativeExecutor {
    pub(super) fn transcribe_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let family = variant.family();
        let language = request.language.as_deref();
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;

        if let Some(tx) = stream_tx.as_ref() {
            if !matches!(family, ModelFamily::Voxtral) {
                let model = self.with_registry(|registry| {
                    registry.try_get_asr(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
                    })
                })?;

                if model.supports_incremental_decode() {
                    let mut active_state = {
                        let mut guard = self.asr_decode_states.lock().map_err(|_| {
                            Error::InferenceError("ASR decode state mutex poisoned".to_string())
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
                        let (samples, sample_rate) = decode_request_audio_with_rate(request)?;
                        let samples_len = samples.len();

                        let chunk_plan = Self::asr_chunk_plan(
                            &samples,
                            sample_rate,
                            model.max_audio_seconds_hint(),
                            false,
                            matches!(family, ModelFamily::WhisperAsr),
                        );
                        if chunk_plan.requires_chunk_path() {
                            let mut sequence = 0usize;
                            let text = Self::run_blocking(|| {
                                Self::transcribe_with_chunk_plan(
                                    &request.id,
                                    Some(tx),
                                    stream_policy,
                                    &mut sequence,
                                    &samples,
                                    sample_rate,
                                    &chunk_plan.chunks,
                                    &chunk_plan.config,
                                    |chunk_audio, sr| model.transcribe(chunk_audio, sr, language),
                                )
                            })?;
                            let diagnostics = Some(chunk_plan.diagnostics());

                            return Ok(ExecutorOutput {
                                request_id: request.id.clone(),
                                audio: Some(AudioOutput {
                                    samples: Vec::new(),
                                    sample_rate,
                                    duration_secs: if sample_rate > 0 {
                                        samples_len as f32 / sample_rate as f32
                                    } else {
                                        0.0
                                    },
                                }),
                                text: Some(text),
                                input_transcription: None,
                                tokens_processed: request.num_prompt_tokens(),
                                tokens_generated: (samples_len / 256).max(1),
                                finished: true,
                                phase_timing_override: None,
                                asr_diagnostics: diagnostics,
                                error: None,
                            });
                        }

                        // Keep ASR decode bounded. If EOS is missed, very high caps
                        // produce runaway gibberish and extreme latency.
                        let max_new_tokens = request.params.max_tokens.clamp(1, MAX_ASR_NEW_TOKENS);
                        let decode_state = Self::run_blocking(|| {
                            model.start_decode_state(
                                &samples,
                                sample_rate,
                                language,
                                max_new_tokens,
                            )
                        })?;
                        ActiveAsrDecode {
                            variant,
                            state: decode_state,
                            prompt_accounted: false,
                            last_tokens_generated: 0,
                            stream_sequence: 0,
                            input_sample_rate: sample_rate,
                            input_sample_count: samples_len,
                        }
                    };

                    let decode_iterations = if scheduled.is_prefill {
                        1
                    } else {
                        scheduled.num_tokens.max(1)
                    };
                    let mut decode_steps_ran = 0usize;
                    let mut total_tokens_generated = 0usize;
                    let mut final_text = String::new();
                    let mut finished = false;

                    for _ in 0..decode_iterations {
                        let step =
                            Self::run_blocking(|| model.decode_step(&mut active_state.state))?;
                        decode_steps_ran = decode_steps_ran.saturating_add(1);
                        let step_tokens_generated = step
                            .tokens_generated
                            .saturating_sub(active_state.last_tokens_generated);
                        active_state.last_tokens_generated = step.tokens_generated;
                        total_tokens_generated =
                            total_tokens_generated.saturating_add(step_tokens_generated);
                        final_text = step.text.clone();

                        if !step.delta.is_empty() {
                            Self::stream_text_per_character_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut active_state.stream_sequence,
                                &step.delta,
                            )?;
                        }
                        if step.finished {
                            Self::stream_final_marker_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut active_state.stream_sequence,
                            )?;
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
                        tokens_processed =
                            tokens_processed.saturating_add(request.num_prompt_tokens());
                    }

                    let input_sample_rate = active_state.input_sample_rate;
                    let input_sample_count = active_state.input_sample_count;

                    if !finished {
                        let mut guard = self.asr_decode_states.lock().map_err(|_| {
                            Error::InferenceError("ASR decode state mutex poisoned".to_string())
                        })?;
                        guard.insert(request.id.clone(), active_state);
                    }

                    return Ok(ExecutorOutput {
                        request_id: request.id.clone(),
                        audio: Some(AudioOutput {
                            samples: Vec::new(),
                            sample_rate: input_sample_rate,
                            duration_secs: if input_sample_rate > 0 {
                                input_sample_count as f32 / input_sample_rate as f32
                            } else {
                                0.0
                            },
                        }),
                        text: Some(final_text),
                        input_transcription: None,
                        tokens_processed,
                        tokens_generated: total_tokens_generated,
                        finished,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                    });
                }
            }
        }

        let audio_decode_started = Instant::now();
        let (samples, sample_rate) = decode_request_audio_with_rate(request)?;
        let audio_decode_ms = audio_decode_started.elapsed().as_secs_f64() * 1000.0;
        let samples_len = samples.len();

        let (text, asr_diagnostics) = Self::run_blocking(|| {
            let mut sequence = 0usize;
            if matches!(family, ModelFamily::Voxtral) {
                let model = self.with_registry(|registry| {
                    registry.try_get_voxtral(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!(
                            "Voxtral model {variant} is not loaded in registry"
                        ))
                    })
                })?;

                let chunk_plan = Self::asr_chunk_plan(&samples, sample_rate, None, false, false);
                if chunk_plan.requires_chunk_path() {
                    let text = Self::transcribe_with_chunk_plan(
                        &request.id,
                        stream_tx.as_ref(),
                        stream_policy,
                        &mut sequence,
                        &samples,
                        sample_rate,
                        &chunk_plan.chunks,
                        &chunk_plan.config,
                        |chunk_audio, sr| model.transcribe(chunk_audio, sr, language),
                    )?;
                    return Ok((text, Some(chunk_plan.diagnostics())));
                }

                if request.streaming {
                    if let Some(tx) = stream_tx.as_ref() {
                        let mut stream_err: Option<Error> = None;
                        let mut emit = |delta: &str| {
                            if stream_err.is_none() {
                                if let Err(err) = Self::stream_text_per_character_with_policy(
                                    tx,
                                    stream_policy,
                                    &request.id,
                                    &mut sequence,
                                    delta,
                                ) {
                                    stream_err = Some(err);
                                }
                            }
                        };
                        let text = model.transcribe_with_callback(
                            &samples,
                            sample_rate,
                            language,
                            &mut emit,
                        )?;
                        if let Some(err) = stream_err {
                            return Err(err);
                        }
                        Self::stream_final_marker_with_policy(
                            tx,
                            stream_policy,
                            &request.id,
                            &mut sequence,
                        )?;
                        return Ok((text, None));
                    }
                }
                return Ok((model.transcribe(&samples, sample_rate, language)?, None));
            }

            let model = self.with_registry(|registry| {
                registry.try_get_asr(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
                })
            })?;

            let chunk_plan = Self::asr_chunk_plan(
                &samples,
                sample_rate,
                model.max_audio_seconds_hint(),
                request.streaming && !model.supports_incremental_decode(),
                matches!(family, ModelFamily::WhisperAsr),
            );
            if chunk_plan.requires_chunk_path() {
                let text = Self::transcribe_with_chunk_plan(
                    &request.id,
                    stream_tx.as_ref(),
                    stream_policy,
                    &mut sequence,
                    &samples,
                    sample_rate,
                    &chunk_plan.chunks,
                    &chunk_plan.config,
                    |chunk_audio, sr| model.transcribe(chunk_audio, sr, language),
                )?;
                return Ok((text, Some(chunk_plan.diagnostics())));
            }

            if request.streaming {
                if let Some(tx) = stream_tx.as_ref() {
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) = Self::stream_text_per_character_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut sequence,
                                delta,
                            ) {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let text = model.transcribe_with_callback(
                        &samples,
                        sample_rate,
                        language,
                        &mut emit,
                    )?;
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker_with_policy(
                        tx,
                        stream_policy,
                        &request.id,
                        &mut sequence,
                    )?;
                    return Ok((text, None));
                }
            }
            let details = model.transcribe_with_details(&samples, sample_rate, language)?;
            Ok((details.text, details.diagnostics))
        })?;
        let asr_diagnostics = Self::with_audio_decode_timing(asr_diagnostics, audio_decode_ms);

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: samples_len as f32 / sample_rate as f32,
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (samples_len / 256).max(1),
            finished: true,
            phase_timing_override: None,
            asr_diagnostics,
            error: None,
        })
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
}
