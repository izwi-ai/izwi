//! Granite Speech ASR: the retained sequence path, prefix replay and the
//! chunk loop and truncation recovery.

use super::*;

pub(super) fn recent_word_suffix(text: &str, max_words: usize) -> String {
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
pub(super) struct GranitePhraseLoop {
    pub(super) phrase: String,
    pub(super) phrase_words: usize,
    pub(super) repeat_count: usize,
    pub(super) start_word: usize,
    pub(super) trailing_words: usize,
}

impl GranitePhraseLoop {
    pub(super) fn score(&self) -> usize {
        self.phrase_words.saturating_mul(self.repeat_count)
    }
}

pub(super) fn granite_decode_stop_reason(diagnostics: Option<&serde_json::Value>) -> Option<&str> {
    diagnostics
        .and_then(|value| value.pointer("/decode/stop_reason"))
        .and_then(|value| value.as_str())
}

pub(super) fn granite_chunk_loop_signal(
    text: &str,
    diagnostics: Option<&serde_json::Value>,
) -> Option<GranitePhraseLoop> {
    if granite_decode_stop_reason(diagnostics) != Some("max_tokens") {
        return None;
    }
    repeated_phrase_loop(text)
}

/// Whether a loop-free Granite chunk was cut off by its automatic budget. A
/// caller's explicit `max_tokens` is a cap the executor must honour.
pub(super) fn granite_chunk_truncated(auto_max_tokens: bool, diagnostics: Option<&serde_json::Value>) -> bool {
    auto_max_tokens && granite_decode_stop_reason(diagnostics) == Some("max_tokens")
}

/// Budget for the one re-decode of a truncated Granite chunk: twice the
/// original, within the request ceiling and the decode room the invocation
/// context leaves. `None` when no larger budget fits.
pub(super) fn granite_truncation_retry_max_tokens(
    original_max_tokens: usize,
    ceiling: usize,
    decode_room: Option<usize>,
) -> Option<usize> {
    let doubled = original_max_tokens.saturating_mul(2).min(ceiling);
    let retry = decode_room.map_or(doubled, |room| doubled.min(room));
    (retry > original_max_tokens).then_some(retry)
}

pub(super) fn repeated_phrase_loop(text: &str) -> Option<GranitePhraseLoop> {
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

pub(super) fn normalized_words(text: &str) -> Vec<String> {
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
pub(super) struct NormalizedWordSpan {
    pub(super) end_byte: usize,
}

pub(super) fn normalized_word_spans(text: &str) -> Vec<NormalizedWordSpan> {
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
pub(super) struct GraniteLoopTrim {
    pub(super) original_chars: usize,
    pub(super) trimmed_chars: usize,
    pub(super) original_words: usize,
    pub(super) trimmed_words: usize,
}

pub(super) fn trim_repeated_phrase_tail(text: &str, loop_signal: &GranitePhraseLoop) -> Option<String> {
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

pub(super) fn trim_granite_looping_chunk_text(
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

pub(super) fn granite_loop_recovery_max_tokens(
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

pub(super) fn with_granite_loop_recovery_diagnostics(
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

pub(super) fn with_granite_truncation_retry_diagnostics(
    diagnostics: Option<serde_json::Value>,
    original_max_tokens: usize,
    retry_max_tokens: Option<usize>,
    retry_stop_reason: Option<&str>,
    selected_retry: bool,
    decision: &str,
) -> Option<serde_json::Value> {
    let mut diagnostics = diagnostics.unwrap_or_else(|| json!({}));
    if !diagnostics.is_object() {
        diagnostics = json!({ "model_diagnostics": diagnostics });
    }
    if let Some(root) = diagnostics.as_object_mut() {
        root.insert(
            "chunk_truncation_retry".to_string(),
            json!({
                "triggered": true,
                "decision": decision,
                "selected_retry": selected_retry,
                "original_max_new_tokens": original_max_tokens,
                "retry_max_new_tokens": retry_max_tokens,
                "retry_stop_reason": retry_stop_reason,
            }),
        );
    }
    Some(diagnostics)
}

impl NativeExecutor {
    pub(super) fn granite_speech_asr_sequence_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed_state: Option<super::super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        let model = request.prepared_asr_model_for_executor()?.ok_or_else(|| {
            Error::InferenceError("Granite Speech sequence lost its loaded model".into())
        })?;
        if variant.family() != ModelFamily::GraniteSpeechAsr
            || !model.supports_resumable_prefill()
            || !model.supports_incremental_decode()
        {
            return Err(Error::InvalidInput(
                "loaded Granite Speech model has no retained sequence contract".into(),
            ));
        }
        let mut retained = managed_state.take().ok_or_else(|| {
            Error::InferenceError("Granite Speech sequence lost retained paged state".into())
        })?;
        if retained.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "Granite Speech retained sequence unexpectedly received tensor state".into(),
            ));
        }
        let cache = retained.take_only_paged()?;
        let prompt_tokens = request.num_prompt_tokens();
        let resumable_span = scheduled
            .is_prefill
            .then(|| resumable_asr_prefill_span(scheduled, prompt_tokens))
            .transpose()?;
        let session = scheduled.session_key();
        let mut state_lease = ExecutorStateLease::checkout(
            &self.asr_decode_states,
            session,
            variant,
            "Granite Speech ASR decode",
        )?;
        if state_lease
            .state()
            .is_some_and(|active| active.variant != variant || !Arc::ptr_eq(&active.model, &model))
        {
            state_lease.discard_state();
        }

        let mut checkpoint = None;
        let mut outer_checkpoint = None;
        let mut fresh_state = false;
        if state_lease.state().is_some() {
            let active = state_lease.require_state_mut()?;
            outer_checkpoint = Some((active.last_tokens_generated, active.stream_sequence));
            checkpoint = Some(active.state.begin_managed_quantum(cache)?);
            state_lease.mark_dirty();
        } else {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "Granite Speech sequence lost state before initial prefill".into(),
                ));
            }
            if request.is_cancelled() {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            let artifact = request
                .prepared_granite_speech_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Granite Speech sequence lost its prepared prompt artifact".into(),
                    )
                })?;
            let (samples, sample_rate) =
                request.prepared_asr_audio_for_executor()?.ok_or_else(|| {
                    Error::InferenceError("Granite Speech sequence lost decoded audio".into())
                })?;
            let options = Self::asr_chunk_generation_options(
                request,
                ModelFamily::GraniteSpeechAsr,
                samples.len(),
                sample_rate,
                &Self::asr_generation_options(request),
                self.asr_effective_context(variant)
                    .zip(model.audio_token_rate()),
            );
            let Some(decode_state) = run_asr_model_call(request, || {
                Self::run_blocking(|| {
                    model.start_granite_speech_resumable_prefill_managed(artifact, options, cache)
                })
            })?
            else {
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            };
            if decode_state.prefill_token_count() != Some(prompt_tokens)
                || decode_state.prefill_progress() != Some(0)
                || decode_state.sequence_position() != Some(0)
            {
                return Err(Error::InferenceError(
                    "Granite Speech prompt geometry differs from scheduler admission".into(),
                ));
            }
            let model_lease = request
                .prepared_asr_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Granite Speech sequence lost model residency".into())
                })?;
            state_lease.install_state(ActiveAsrDecode {
                variant,
                model: model.clone(),
                _model_lease: model_lease,
                state: decode_state,
                last_tokens_generated: 0,
                stream_sequence: 0,
                input_sample_rate: sample_rate,
                input_sample_count: samples.len(),
            })?;
            fresh_state = true;
        }

        if request.is_cancelled() {
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        state_lease.mark_dirty();
        let execution = (|| {
            let active = state_lease.require_state_mut()?;
            let iterations = if scheduled.is_prefill {
                1
            } else {
                scheduled.num_tokens.max(1)
            };
            let mut generated = 0usize;
            let mut text = String::new();
            let mut finished = false;
            let mut cancelled = false;
            let mut events = Vec::new();
            for _ in 0..iterations {
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                let step = if let Some((start, end)) = resumable_span {
                    let Some(_complete) = run_asr_model_call(request, || {
                        Self::run_blocking(|| {
                            active
                                .model
                                .continue_resumable_prefill(&mut active.state, start, end)
                        })
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    crate::models::registry::NativeAsrDecodeStep {
                        delta: String::new(),
                        text: String::new(),
                        tokens_generated: active.last_tokens_generated,
                        finished: false,
                    }
                } else {
                    let Some(step) = run_asr_model_call(request, || {
                        Self::run_blocking(|| active.model.decode_step(&mut active.state))
                    })?
                    else {
                        cancelled = true;
                        break;
                    };
                    step
                };
                if request.is_cancelled() {
                    cancelled = true;
                    break;
                }
                generated = generated.saturating_add(
                    step.tokens_generated
                        .saturating_sub(active.last_tokens_generated),
                );
                active.last_tokens_generated = step.tokens_generated;
                text = step.text.clone();
                if !scheduled.is_prefill {
                    events.push((step.delta, step.finished));
                }
                if step.finished {
                    finished = true;
                    break;
                }
            }
            if !cancelled {
                if let Some(tx) = Self::stream_sender(request) {
                    for (delta, event_finished) in events {
                        if request.is_cancelled() {
                            cancelled = true;
                            break;
                        }
                        if !delta.is_empty() {
                            Self::stream_text_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                                delta,
                            )?;
                        }
                        if event_finished {
                            Self::stream_final_marker_with_policy(
                                &tx,
                                request.stream_policy,
                                &request.id,
                                &mut active.stream_sequence,
                            )?;
                        }
                    }
                }
                cancelled |= request.is_cancelled();
            }
            if cancelled {
                let _ = request.take_staged_stream_outputs()?;
            }
            let completions = if cancelled {
                Vec::new()
            } else {
                active.state.take_managed_write_completions()
            };
            Ok((
                generated,
                text,
                finished,
                cancelled,
                active.input_sample_rate,
                active.input_sample_count,
                completions,
            ))
        })();
        let (generated, text, finished, cancelled, sample_rate, sample_count, completions) =
            match execution {
                Ok(value) => value,
                Err(error) => {
                    let _ = request.take_staged_stream_outputs()?;
                    rollback_scalar_asr_quantum(
                        &mut state_lease,
                        &mut checkpoint,
                        outer_checkpoint,
                        fresh_state,
                    )?;
                    return Err(error);
                }
            };
        if cancelled || request.is_cancelled() {
            let _ = request.take_staged_stream_outputs()?;
            rollback_scalar_asr_quantum(
                &mut state_lease,
                &mut checkpoint,
                outer_checkpoint,
                fresh_state,
            )?;
            state_lease.release()?;
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }
        if let Some(checkpoint) = checkpoint.take() {
            state_lease
                .require_state_mut()?
                .state
                .commit_managed_quantum(checkpoint)?;
        }
        let tokens_processed = if scheduled.is_prefill {
            scheduled.num_tokens
        } else {
            scheduled.num_tokens.max(1)
        };
        if finished {
            state_lease.release()?;
        } else {
            state_lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: if sample_rate == 0 {
                    0.0
                } else {
                    sample_count as f32 / sample_rate as f32
                },
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        })
        .with_managed_cache_completions(completions))
    }

    pub(super) fn granite_asr_prefix_replay_words() -> usize {
        static WORDS: crate::env::ReadOnce<usize> = crate::env::ReadOnce::new(|| {
            NativeExecutor::env_usize("IZWI_GRANITE_ASR_PREFIX_REPLAY_WORDS")
                .unwrap_or(GRANITE_ASR_PREFIX_REPLAY_WORDS)
                .min(GRANITE_ASR_PREFIX_REPLAY_WORDS_MAX)
        });
        WORDS.get()
    }

    pub(super) fn granite_asr_prefix_replay_text(prefix_text: &str) -> String {
        recent_word_suffix(prefix_text, Self::granite_asr_prefix_replay_words())
    }

    /// Recover a Granite chunk that stopped on its decode budget: a repetition
    /// loop is re-decoded with a shorter budget and trimmed, and a plain
    /// truncation is re-decoded once with a larger one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn recover_granite_chunk(
        model: &crate::models::registry::NativeAsrModel,
        chunk_audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        asr_prompt: Option<&str>,
        prefix_text: Option<&str>,
        base_options: &NativeAsrGenerationOptions,
        mut details: crate::models::registry::NativeAsrTranscription,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        invocation_context: Option<(usize, f32)>,
    ) -> Result<crate::models::registry::NativeAsrTranscription> {
        let Some(original_loop) =
            granite_chunk_loop_signal(&details.text, details.diagnostics.as_ref())
        else {
            return Self::retry_truncated_granite_chunk(
                model,
                chunk_audio,
                sample_rate,
                language,
                task,
                asr_prompt,
                prefix_text,
                base_options,
                details,
                request,
                scheduled,
                invocation_context,
            );
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
        let mut retry = with_single_invocation_cache(request, scheduled, |cache| {
            model.transcribe_granite_speech_with_details_prompt_prefix_and_options_physical(
                chunk_audio,
                sample_rate,
                language,
                task,
                asr_prompt,
                prefix_text,
                retry_options,
                cache,
            )
        })?;
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

    /// A chunk that spent its whole automatic budget without looping was cut
    /// off mid-transcript, and the chunk overlap cannot recover a longer tail
    /// (GRN-1). Re-decode it once at twice the budget. Greedy decoding replays
    /// the same prefix, so the retry extends the original unless it
    /// degenerates into a loop that cannot be trimmed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn retry_truncated_granite_chunk(
        model: &crate::models::registry::NativeAsrModel,
        chunk_audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        task: GraniteSpeechTask,
        asr_prompt: Option<&str>,
        prefix_text: Option<&str>,
        base_options: &NativeAsrGenerationOptions,
        mut details: crate::models::registry::NativeAsrTranscription,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        invocation_context: Option<(usize, f32)>,
    ) -> Result<crate::models::registry::NativeAsrTranscription> {
        if !granite_chunk_truncated(request.asr_auto_max_tokens, details.diagnostics.as_ref()) {
            return Ok(details);
        }
        let original_max_tokens = base_options.max_new_tokens;
        let chunk_seconds = if sample_rate > 0 {
            chunk_audio.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let prompt_tokens = details
            .diagnostics
            .as_ref()
            .and_then(|value| value.get("prompt_tokens"))
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok());
        let decode_room = invocation_context.map(|(context, rate)| {
            let planned = invocation_decode_room(context, rate, chunk_seconds);
            prompt_tokens.map_or(planned, |prompt| {
                planned.min(context.saturating_sub(prompt))
            })
        });
        let Some(retry_max_tokens) = granite_truncation_retry_max_tokens(
            original_max_tokens,
            request.params.max_tokens,
            decode_room,
        ) else {
            details.diagnostics = with_granite_truncation_retry_diagnostics(
                details.diagnostics,
                original_max_tokens,
                None,
                None,
                false,
                "not_retried_budget_ceiling",
            );
            return Ok(details);
        };

        let mut retry_options = base_options.clone();
        retry_options.max_new_tokens = retry_max_tokens;
        let mut retry = with_single_invocation_cache(request, scheduled, |cache| {
            model.transcribe_granite_speech_with_details_prompt_prefix_and_options_physical(
                chunk_audio,
                sample_rate,
                language,
                task,
                asr_prompt,
                prefix_text,
                retry_options,
                cache,
            )
        })?;
        let retry_stop_reason =
            granite_decode_stop_reason(retry.diagnostics.as_ref()).map(str::to_owned);
        let retry_loop = granite_chunk_loop_signal(&retry.text, retry.diagnostics.as_ref());
        let retry_trim = retry_loop
            .as_ref()
            .and_then(|loop_signal| trim_granite_looping_chunk_text(&mut retry.text, loop_signal));
        let use_retry =
            !retry.text.trim().is_empty() && (retry_loop.is_none() || retry_trim.is_some());
        if use_retry {
            retry.diagnostics = with_granite_truncation_retry_diagnostics(
                retry.diagnostics,
                original_max_tokens,
                Some(retry_max_tokens),
                retry_stop_reason.as_deref(),
                true,
                if retry_trim.is_some() {
                    "retry_selected_trimmed"
                } else {
                    "retry_selected"
                },
            );
            Ok(retry)
        } else {
            details.diagnostics = with_granite_truncation_retry_diagnostics(
                details.diagnostics,
                original_max_tokens,
                Some(retry_max_tokens),
                retry_stop_reason.as_deref(),
                false,
                "retry_not_better",
            );
            Ok(details)
        }
    }
}
