use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::registry::NativeAsrGenerationOptions;
use crate::runtime::granite_auto_asr_max_tokens_for_duration;
use serde_json::json;
use std::time::Instant;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::audio::{decode_request_audio_with_rate, AsrChunkTranscription};
use super::state::ActiveAsrDecode;
use super::{ExecutorOutput, NativeExecutor};

const MAX_ASR_NEW_TOKENS: usize = 512;
const GRANITE_ASR_PREFIX_REPLAY_WORDS: usize = 0;
const GRANITE_ASR_PREFIX_REPLAY_WORDS_MAX: usize = 240;

impl NativeExecutor {
    pub(super) fn transcribe_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let family = variant.family();
        let language = request.language.as_deref();
        let asr_prompt = request.asr_prompt.as_deref();
        let generation_options = Self::asr_generation_options(request);
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
                            let chunk_stream_options = if matches!(family, ModelFamily::Qwen3Asr) {
                                Self::qwen_asr_chunk_stream_options()
                            } else {
                                Default::default()
                            };
                            let chunked = Self::run_blocking(|| {
                                Self::transcribe_with_chunk_plan_with_details_and_options(
                                    &request.id,
                                    Some(tx),
                                    stream_policy,
                                    &mut sequence,
                                    &samples,
                                    sample_rate,
                                    &chunk_plan.chunks,
                                    &chunk_plan.config,
                                    chunk_stream_options,
                                    |chunk_audio, sr, _prefix_text| {
                                        let details = model
                                            .transcribe_with_details_and_prompt_and_options(
                                                chunk_audio,
                                                sr,
                                                language,
                                                asr_prompt,
                                                generation_options.clone(),
                                            )?;
                                        Ok(AsrChunkTranscription {
                                            text: details.text,
                                            diagnostics: details.diagnostics,
                                        })
                                    },
                                )
                            })?;
                            let diagnostics =
                                Some(chunk_plan.diagnostics_with_chunk_transcriptions(
                                    chunked.chunk_diagnostics,
                                ));

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
                                text: Some(chunked.text),
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
                            model.start_decode_state_with_prompt(
                                &samples,
                                sample_rate,
                                language,
                                asr_prompt,
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
                            Self::stream_text_with_policy(
                                tx,
                                stream_policy,
                                &request.id,
                                &mut active_state.stream_sequence,
                                step.delta,
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
                    let chunked = Self::transcribe_with_chunk_plan_with_details(
                        &request.id,
                        stream_tx.as_ref(),
                        stream_policy,
                        &mut sequence,
                        &samples,
                        sample_rate,
                        &chunk_plan.chunks,
                        &chunk_plan.config,
                        |chunk_audio, sr| {
                            model
                                .transcribe_with_details(chunk_audio, sr, language)
                                .map(|details| AsrChunkTranscription {
                                    text: details.text,
                                    diagnostics: details.diagnostics,
                                })
                        },
                    )?;
                    return Ok((
                        chunked.text,
                        Some(
                            chunk_plan
                                .diagnostics_with_chunk_transcriptions(chunked.chunk_diagnostics),
                        ),
                    ));
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
                return Ok((details.text, details.diagnostics));
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
                let chunked = Self::transcribe_with_chunk_plan_with_context_and_details(
                    &request.id,
                    stream_tx.as_ref(),
                    stream_policy,
                    &mut sequence,
                    &samples,
                    sample_rate,
                    &chunk_plan.chunks,
                    &chunk_plan.config,
                    |chunk_audio, sr, prefix_text| {
                        let bounded_prefix_text = matches!(family, ModelFamily::GraniteSpeechAsr)
                            .then(|| Self::granite_asr_prefix_replay_text(prefix_text));
                        let prefix_text = bounded_prefix_text
                            .as_deref()
                            .filter(|value| !value.trim().is_empty());
                        let chunk_generation_options = Self::asr_chunk_generation_options(
                            request,
                            family,
                            chunk_audio.len(),
                            sr,
                            &generation_options,
                        );
                        let mut details = model.transcribe_with_details_prompt_prefix_and_options(
                            chunk_audio,
                            sr,
                            language,
                            asr_prompt,
                            prefix_text,
                            chunk_generation_options.clone(),
                        )?;
                        if matches!(family, ModelFamily::GraniteSpeechAsr) {
                            details = Self::recover_granite_chunk_loop(
                                &model,
                                chunk_audio,
                                sr,
                                language,
                                asr_prompt,
                                prefix_text,
                                &chunk_generation_options,
                                details,
                            )?;
                        }
                        Ok(AsrChunkTranscription {
                            text: details.text,
                            diagnostics: details.diagnostics,
                        })
                    },
                )?;
                return Ok((
                    chunked.text,
                    Some(
                        chunk_plan.diagnostics_with_chunk_transcriptions(chunked.chunk_diagnostics),
                    ),
                ));
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
                    let text = model.transcribe_with_callback_and_prompt_and_options(
                        &samples,
                        sample_rate,
                        language,
                        asr_prompt,
                        generation_options.clone(),
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
            let details = model.transcribe_with_details_and_prompt_and_options(
                &samples,
                sample_rate,
                language,
                asr_prompt,
                generation_options.clone(),
            )?;
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

    fn asr_generation_options(request: &EngineCoreRequest) -> NativeAsrGenerationOptions {
        NativeAsrGenerationOptions {
            max_new_tokens: request.params.max_tokens.max(1),
            stop_token_ids: request.params.stop_token_ids.clone(),
            stop_sequences: request.params.stop_sequences.clone(),
        }
    }

    fn asr_chunk_generation_options(
        request: &EngineCoreRequest,
        family: ModelFamily,
        chunk_sample_count: usize,
        sample_rate: u32,
        base: &NativeAsrGenerationOptions,
    ) -> NativeAsrGenerationOptions {
        let mut options = base.clone();
        if request.asr_auto_max_tokens
            && matches!(family, ModelFamily::GraniteSpeechAsr)
            && sample_rate > 0
        {
            let chunk_seconds = chunk_sample_count as f32 / sample_rate as f32;
            options.max_new_tokens =
                granite_auto_asr_max_tokens_for_duration(chunk_seconds).min(base.max_new_tokens);
        }
        options.max_new_tokens = options.max_new_tokens.max(1);
        options
    }

    fn granite_asr_prefix_replay_words() -> usize {
        Self::env_usize("IZWI_GRANITE_ASR_PREFIX_REPLAY_WORDS")
            .unwrap_or(GRANITE_ASR_PREFIX_REPLAY_WORDS)
            .min(GRANITE_ASR_PREFIX_REPLAY_WORDS_MAX)
    }

    fn granite_asr_prefix_replay_text(prefix_text: &str) -> String {
        recent_word_suffix(prefix_text, Self::granite_asr_prefix_replay_words())
    }

    #[allow(clippy::too_many_arguments)]
    fn recover_granite_chunk_loop(
        model: &crate::models::registry::NativeAsrModel,
        chunk_audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        asr_prompt: Option<&str>,
        prefix_text: Option<&str>,
        base_options: &NativeAsrGenerationOptions,
        mut details: crate::models::registry::NativeAsrTranscription,
    ) -> Result<crate::models::registry::NativeAsrTranscription> {
        let Some(original_loop) =
            granite_chunk_loop_signal(&details.text, details.diagnostics.as_ref())
        else {
            return Ok(details);
        };
        let retry_max_tokens = granite_loop_recovery_max_tokens(
            chunk_audio.len(),
            sample_rate,
            base_options.max_new_tokens,
        );
        if retry_max_tokens >= base_options.max_new_tokens {
            let trim = trim_granite_looping_chunk_text(&mut details.text, &original_loop);
            details.diagnostics = with_granite_loop_recovery_diagnostics(
                details.diagnostics,
                &original_loop,
                None,
                base_options.max_new_tokens,
                retry_max_tokens,
                false,
                if trim.is_some() {
                    "not_retried_budget_floor_trimmed_original"
                } else {
                    "not_retried_budget_floor"
                },
                trim.as_ref(),
            );
            return Ok(details);
        }

        let mut retry_options = base_options.clone();
        retry_options.max_new_tokens = retry_max_tokens;
        let mut retry = model.transcribe_with_details_prompt_prefix_and_options(
            chunk_audio,
            sample_rate,
            language,
            asr_prompt,
            prefix_text,
            retry_options,
        )?;
        let retry_loop = granite_chunk_loop_signal(&retry.text, retry.diagnostics.as_ref());
        let retry_trim = retry_loop
            .as_ref()
            .and_then(|loop_signal| trim_granite_looping_chunk_text(&mut retry.text, loop_signal));
        let use_retry = !retry.text.trim().is_empty()
            && (retry_trim.is_some()
                || retry_loop
                    .as_ref()
                    .map(|loop_signal| loop_signal.score() < original_loop.score())
                    .unwrap_or(true));

        if use_retry {
            retry.diagnostics = with_granite_loop_recovery_diagnostics(
                retry.diagnostics,
                &original_loop,
                retry_loop.as_ref(),
                base_options.max_new_tokens,
                retry_max_tokens,
                true,
                if retry_trim.is_some() {
                    "retry_selected_trimmed"
                } else {
                    "retry_selected"
                },
                retry_trim.as_ref(),
            );
            Ok(retry)
        } else {
            let original_trim = trim_granite_looping_chunk_text(&mut details.text, &original_loop);
            details.diagnostics = with_granite_loop_recovery_diagnostics(
                details.diagnostics,
                &original_loop,
                retry_loop.as_ref(),
                base_options.max_new_tokens,
                retry_max_tokens,
                false,
                if original_trim.is_some() {
                    "retry_not_better_trimmed_original"
                } else {
                    "retry_not_better"
                },
                original_trim.as_ref(),
            );
            Ok(details)
        }
    }
}

fn recent_word_suffix(text: &str, max_words: usize) -> String {
    if max_words == 0 {
        return String::new();
    }
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() <= max_words {
        return text.trim().to_string();
    }
    words[words.len() - max_words..].join(" ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GranitePhraseLoop {
    phrase: String,
    phrase_words: usize,
    repeat_count: usize,
    start_word: usize,
    trailing_words: usize,
}

impl GranitePhraseLoop {
    fn score(&self) -> usize {
        self.phrase_words.saturating_mul(self.repeat_count)
    }
}

fn granite_chunk_loop_signal(
    text: &str,
    diagnostics: Option<&serde_json::Value>,
) -> Option<GranitePhraseLoop> {
    if diagnostics
        .and_then(|value| value.pointer("/decode/stop_reason"))
        .and_then(|value| value.as_str())
        != Some("max_tokens")
    {
        return None;
    }
    repeated_phrase_loop(text)
}

fn repeated_phrase_loop(text: &str) -> Option<GranitePhraseLoop> {
    let words = normalized_words(text);
    if words.len() < 9 {
        return None;
    }
    let mut best: Option<GranitePhraseLoop> = None;
    let max_phrase_words = 12.min(words.len() / 3);
    for phrase_words in 3..=max_phrase_words {
        let mut idx = 0usize;
        while idx + phrase_words * 3 <= words.len() {
            let phrase = &words[idx..idx + phrase_words];
            let mut repeats = 1usize;
            while idx + phrase_words * (repeats + 1) <= words.len()
                && words[idx + phrase_words * repeats..idx + phrase_words * (repeats + 1)]
                    == *phrase
            {
                repeats += 1;
            }
            if repeats >= 3 {
                let repeated_words = phrase_words.saturating_mul(repeats);
                let candidate = GranitePhraseLoop {
                    phrase: phrase.join(" "),
                    phrase_words,
                    repeat_count: repeats,
                    start_word: idx,
                    trailing_words: words.len().saturating_sub(idx + repeated_words),
                };
                if best
                    .as_ref()
                    .map(|current| candidate.score() > current.score())
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
                idx += phrase_words * repeats;
            } else {
                idx += 1;
            }
        }
    }
    best
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric() || matches!(ch, '\'' | '-'))
                .flat_map(|ch| ch.to_lowercase())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

#[derive(Debug, Clone)]
struct NormalizedWordSpan {
    end_byte: usize,
}

fn normalized_word_spans(text: &str) -> Vec<NormalizedWordSpan> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut current_end = 0usize;
    for (idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() || matches!(ch, '\'' | '-') {
            current.extend(ch.to_lowercase());
            current_end = idx + ch.len_utf8();
        } else if !current.is_empty() {
            spans.push(NormalizedWordSpan {
                end_byte: current_end,
            });
            current.clear();
        }
    }
    if !current.is_empty() {
        spans.push(NormalizedWordSpan {
            end_byte: current_end,
        });
    }
    spans
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraniteLoopTrim {
    original_chars: usize,
    trimmed_chars: usize,
    original_words: usize,
    trimmed_words: usize,
}

fn trim_repeated_phrase_tail(text: &str, loop_signal: &GranitePhraseLoop) -> Option<String> {
    if loop_signal.repeat_count < 3 || loop_signal.trailing_words > loop_signal.phrase_words.max(4)
    {
        return None;
    }
    let spans = normalized_word_spans(text);
    let keep_words = loop_signal
        .start_word
        .saturating_add(loop_signal.phrase_words);
    let trim_end = spans.get(keep_words.saturating_sub(1))?.end_byte;
    let trimmed = text[..trim_end].trim_end();
    if trimmed.len() < text.trim_end().len() {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn trim_granite_looping_chunk_text(
    text: &mut String,
    loop_signal: &GranitePhraseLoop,
) -> Option<GraniteLoopTrim> {
    let original_chars = text.chars().count();
    let original_words = normalized_words(text).len();
    let trimmed = trim_repeated_phrase_tail(text, loop_signal)?;
    let trimmed_chars = trimmed.chars().count();
    let trimmed_words = normalized_words(&trimmed).len();
    *text = trimmed;
    Some(GraniteLoopTrim {
        original_chars,
        trimmed_chars,
        original_words,
        trimmed_words,
    })
}

fn granite_loop_recovery_max_tokens(
    chunk_sample_count: usize,
    sample_rate: u32,
    original_max_tokens: usize,
) -> usize {
    if original_max_tokens <= 1 {
        return original_max_tokens;
    }
    let chunk_seconds = if sample_rate > 0 {
        chunk_sample_count as f32 / sample_rate as f32
    } else {
        0.0
    };
    let auto_for_chunk = granite_auto_asr_max_tokens_for_duration(chunk_seconds);
    let reduced = original_max_tokens.saturating_mul(3) / 4;
    reduced
        .min(auto_for_chunk)
        .max(24.min(original_max_tokens.saturating_sub(1)))
        .min(original_max_tokens.saturating_sub(1))
        .max(1)
}

fn with_granite_loop_recovery_diagnostics(
    diagnostics: Option<serde_json::Value>,
    original_loop: &GranitePhraseLoop,
    retry_loop: Option<&GranitePhraseLoop>,
    original_max_tokens: usize,
    retry_max_tokens: usize,
    selected_retry: bool,
    decision: &str,
    trim: Option<&GraniteLoopTrim>,
) -> Option<serde_json::Value> {
    let mut diagnostics = diagnostics.unwrap_or_else(|| json!({}));
    if !diagnostics.is_object() {
        diagnostics = json!({ "model_diagnostics": diagnostics });
    }
    if let Some(root) = diagnostics.as_object_mut() {
        root.insert(
            "chunk_loop_recovery".to_string(),
            json!({
                "triggered": true,
                "decision": decision,
                "selected_retry": selected_retry,
                "original_max_new_tokens": original_max_tokens,
                "retry_max_new_tokens": retry_max_tokens,
                "original_loop": {
                    "phrase": original_loop.phrase,
                    "phrase_words": original_loop.phrase_words,
                    "repeat_count": original_loop.repeat_count,
                },
                "retry_loop": retry_loop.map(|loop_signal| json!({
                    "phrase": loop_signal.phrase,
                    "phrase_words": loop_signal.phrase_words,
                    "repeat_count": loop_signal.repeat_count,
                })),
                "trim": trim.map(|trim| json!({
                    "original_chars": trim.original_chars,
                    "trimmed_chars": trim.trimmed_chars,
                    "original_words": trim.original_words,
                    "trimmed_words": trim.trimmed_words,
                })),
            }),
        );
    }
    Some(diagnostics)
}

#[cfg(test)]
mod tests {
    use super::NativeExecutor;
    use crate::catalog::ModelFamily;
    use crate::engine::request::EngineCoreRequest;
    use crate::models::registry::NativeAsrGenerationOptions;

    #[test]
    fn audio_decode_timing_preserves_whisper_model_diagnostics() {
        let diagnostics = serde_json::json!({
            "model_family": "whisper_asr",
            "timings_ms": {
                "model_total": 12.5
            }
        });

        let updated = NativeExecutor::with_audio_decode_timing(Some(diagnostics), 3.25)
            .expect("diagnostics payload");

        assert_eq!(updated["model_family"], "whisper_asr");
        assert_eq!(updated["timings_ms"]["model_total"], 12.5);
        assert_eq!(updated["timings_ms"]["audio_decode"], 3.25);
    }

    #[test]
    fn audio_decode_timing_creates_diagnostics_when_missing() {
        let updated =
            NativeExecutor::with_audio_decode_timing(None, 4.0).expect("diagnostics payload");

        assert_eq!(updated["timings_ms"]["audio_decode"], 4.0);
    }

    #[test]
    fn audio_decode_timing_wraps_non_object_diagnostics() {
        let updated = NativeExecutor::with_audio_decode_timing(Some(serde_json::json!("old")), 5.0)
            .expect("diagnostics payload");

        assert_eq!(updated["model_diagnostics"], "old");
        assert_eq!(updated["timings_ms"]["audio_decode"], 5.0);
    }

    #[test]
    fn granite_auto_chunk_generation_options_use_chunk_duration() {
        let mut request = EngineCoreRequest::asr("UklGRg==");
        request.asr_auto_max_tokens = true;
        request.params.max_tokens = 2048;
        let base = NativeAsrGenerationOptions {
            max_new_tokens: 2048,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        };

        let options = NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 30,
            16_000,
            &base,
        );

        assert_eq!(options.max_new_tokens, 84);
    }

    #[test]
    fn granite_explicit_chunk_generation_options_preserve_user_budget() {
        let mut request = EngineCoreRequest::asr("UklGRg==");
        request.asr_auto_max_tokens = false;
        request.params.max_tokens = 2048;
        let base = NativeAsrGenerationOptions {
            max_new_tokens: 2048,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        };

        let options = NativeExecutor::asr_chunk_generation_options(
            &request,
            ModelFamily::GraniteSpeechAsr,
            16_000 * 30,
            16_000,
            &base,
        );

        assert_eq!(options.max_new_tokens, 2048);
    }

    #[test]
    fn granite_prefix_replay_text_is_disabled_by_default() {
        let prefix = (0..140)
            .map(|idx| format!("word{idx}"))
            .collect::<Vec<_>>()
            .join(" ");

        let replay = NativeExecutor::granite_asr_prefix_replay_text(&prefix);

        assert!(replay.is_empty());
    }

    #[test]
    fn recent_word_suffix_keeps_recent_words() {
        let prefix = (0..140)
            .map(|idx| format!("word{idx}"))
            .collect::<Vec<_>>()
            .join(" ");

        let replay = super::recent_word_suffix(&prefix, 96);
        let words = replay.split_whitespace().collect::<Vec<_>>();

        assert_eq!(words.len(), 96);
        assert_eq!(words.first(), Some(&"word44"));
        assert_eq!(words.last(), Some(&"word139"));
    }

    #[test]
    fn repeated_phrase_loop_detects_consecutive_phrase_repeats() {
        let text = "intro words if you speak in trust their nervous system contracts \
            if you speak in trust their nervous system contracts \
            if you speak in trust their nervous system contracts tail";

        let loop_signal = super::repeated_phrase_loop(text).expect("loop signal");

        assert!(loop_signal.phrase.contains("if you speak"));
        assert!(loop_signal.repeat_count >= 3);
    }

    #[test]
    fn trim_repeated_phrase_tail_keeps_first_occurrence() {
        let text = "better relationships with your friends and your family and your coworkers \
            and your boss and your boss and your boss and your boss";
        let loop_signal = super::repeated_phrase_loop(text).expect("loop signal");

        let trimmed =
            super::trim_repeated_phrase_tail(text, &loop_signal).expect("trimmed repeated tail");

        assert_eq!(
            trimmed,
            "better relationships with your friends and your family and your coworkers and your boss"
        );
    }

    #[test]
    fn trim_repeated_phrase_tail_ignores_middle_repeats() {
        let text = "alpha beta gamma alpha beta gamma alpha beta gamma then \
            a normal ending with additional words that should remain";
        let loop_signal = super::repeated_phrase_loop(text).expect("loop signal");

        assert!(super::trim_repeated_phrase_tail(text, &loop_signal).is_none());
    }

    #[test]
    fn granite_chunk_loop_signal_requires_max_token_stop() {
        let text = "alpha beta gamma alpha beta gamma alpha beta gamma";
        let stopped = serde_json::json!({
            "decode": {
                "stop_reason": "stop_token"
            }
        });
        let max_tokens = serde_json::json!({
            "decode": {
                "stop_reason": "max_tokens"
            }
        });

        assert!(super::granite_chunk_loop_signal(text, Some(&stopped)).is_none());
        assert!(super::granite_chunk_loop_signal(text, Some(&max_tokens)).is_some());
    }

    #[test]
    fn granite_loop_recovery_budget_is_shorter_than_original() {
        let retry = super::granite_loop_recovery_max_tokens(16_000 * 30, 16_000, 84);

        assert!(retry < 84);
        assert!(retry >= 24);
    }
}
