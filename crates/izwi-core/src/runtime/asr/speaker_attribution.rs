//! Speaker-attributed ASR: Granite Speech Plus's single-pass and long-form
//! transcription, chunk recovery and turn assembly, and models that tag
//! speakers on their ASR rows (Nemotron 3.5 Transcribe).

use super::*;

pub(super) fn resolve_speaker_attributed_asr_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::GraniteSpeech412BPlus,
    };

    if !variant.supports_speaker_attributed_asr() {
        return Err(Error::InvalidInput(format!(
            "{variant} does not support speaker attributed ASR; use Granite-Speech-4.1-2B-Plus or Nemotron-3.5-Transcribe-30B-A4B-FP8, or a diarization run with {variant} as its ASR model"
        )));
    }

    Ok(variant)
}

#[derive(Debug, Clone)]
pub(super) struct GraniteSaaLongFormOutput {
    pub(super) text: String,
    pub(super) language: Option<String>,
    pub(super) warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct GraniteSaaChunkDecodeDiagnostics {
    pub(super) generated_tokens: Option<usize>,
    pub(super) max_new_tokens: Option<usize>,
    pub(super) stop_reason: Option<String>,
    pub(super) prompt_tokens: Option<usize>,
    pub(super) prompt_prefix_tokens: Option<usize>,
}

#[derive(Debug, Default)]
pub(super) struct GraniteSaaTranscriptAssembler {
    pub(super) turns: Vec<SpeakerAttributedAsrTurn>,
}

impl GraniteSaaTranscriptAssembler {
    /// Add one chunk's decode, linking its speaker labels and its overlap
    /// with the transcript so far (see [`saa_seam`]). `chunk_start_secs`
    /// dates the seam in a warning when the link is a guess.
    pub(super) fn push_chunk_text(
        &mut self,
        chunk_text: &str,
        chunk_index: usize,
        chunk_start_secs: f64,
    ) -> Vec<String> {
        let mut warnings = Vec::new();
        let incoming = speaker_attributed_asr_turns_from_text(chunk_text);
        if incoming.is_empty() {
            if !chunk_text.trim().is_empty() {
                warnings.push(format!(
                    "Granite SAA chunk {} returned text without speaker labels.",
                    chunk_index + 1
                ));
            }
            return warnings;
        }

        let has_transcript = !self.turns.is_empty();
        let link = saa_seam::link_seam(&self.turns, &incoming);
        let incoming = match link.chunk_known_through {
            Some(through) => saa_seam::drop_turn_words_through(incoming, through),
            None => incoming,
        };
        if let Some(keep_through) = link.transcript_keep_through {
            saa_seam::truncate_turns_after(&mut self.turns, keep_through);
        }
        if has_transcript {
            tracing::info!(
                chunk_index = chunk_index + 1,
                aligned_words = link.aligned_words,
                labels = ?link.labels,
                "linked Granite SAA chunk seam"
            );
            if !link.is_aligned() {
                warnings.push(format!(
                    "Speaker labels from {} on are a best guess: chunk {} shares too little text with the chunk before it to link its speakers.",
                    format_seam_clock(chunk_start_secs),
                    chunk_index + 1
                ));
            }
        }

        for mut turn in incoming {
            if let Some(global) = link.labels.get(&turn.speaker) {
                turn.speaker = global.clone();
            }
            self.push_turn(turn);
        }
        warnings
    }

    pub(super) fn push_turn(&mut self, mut turn: SpeakerAttributedAsrTurn) {
        turn.text = turn.text.trim().to_string();
        if turn.text.is_empty() {
            return;
        }
        match self.turns.last_mut() {
            Some(last) if last.speaker == turn.speaker => {
                append_with_spacing(&mut last.text, turn.text.as_str());
            }
            _ => self.turns.push(turn),
        }
    }

    pub(super) fn text(&self) -> String {
        format_saa_turns(&self.turns)
    }
}

/// `m:ss`, or `h:mm:ss` past an hour.
fn format_seam_clock(secs: f64) -> String {
    let total = if secs.is_finite() && secs > 0.0 {
        secs.round() as u64
    } else {
        0
    };
    let (hours, minutes, seconds) = (total / 3600, total / 60 % 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

pub(super) fn speaker_attributed_asr_turns_from_text(raw_text: &str) -> Vec<SpeakerAttributedAsrTurn> {
    let parsed = parse_granite_speech_output(raw_text);
    if parsed.segments.is_empty() {
        let text = parsed.text.trim();
        if text.is_empty() {
            return Vec::new();
        }
        return vec![SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: text.to_string(),
            start_secs: None,
            end_secs: None,
        }];
    }

    parsed
        .segments
        .into_iter()
        .map(|segment| SpeakerAttributedAsrTurn {
            speaker: segment
                .speaker
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(UNKNOWN_SAA_SPEAKER)
                .to_string(),
            text: segment.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        })
        .filter(|turn| !turn.text.is_empty())
        .fold(Vec::<SpeakerAttributedAsrTurn>::new(), |mut turns, turn| {
            if let Some(last) = turns.last_mut() {
                if last.speaker == turn.speaker {
                    append_with_spacing(&mut last.text, turn.text.as_str());
                    return turns;
                }
            }
            turns.push(turn);
            turns
        })
}

/// The two halves a chunk caught in a repetition loop is decoded again as,
/// sharing [`GRANITE_SAA_OVERLAP_SECS`] around its middle; `None` when a half
/// would be shorter than [`GRANITE_SAA_MIN_CHUNK_SECS`].
pub(super) fn granite_saa_split_chunk(chunk: &AudioChunk, sample_rate: u32) -> Option<[AudioChunk; 2]> {
    let seconds = |secs: f32| (secs * sample_rate as f32) as usize;
    let half_overlap = seconds(GRANITE_SAA_OVERLAP_SECS / 2.0);
    let length = chunk.end_sample.checked_sub(chunk.start_sample)?;
    if length / 2 < seconds(GRANITE_SAA_MIN_CHUNK_SECS) {
        return None;
    }
    let middle = chunk.start_sample + length / 2;
    Some([
        AudioChunk {
            start_sample: chunk.start_sample,
            end_sample: (middle + half_overlap).min(chunk.end_sample),
        },
        AudioChunk {
            start_sample: middle.saturating_sub(half_overlap).max(chunk.start_sample),
            end_sample: chunk.end_sample,
        },
    ])
}

/// Repeats of one phrase (1–12 words) at the end of `text` past the third,
/// removed: the tail of a decode that looped until its token limit. A cut
/// partial word after the loop goes too. `None` when the text does not end
/// in such a loop.
pub(super) fn trim_trailing_repetition(text: &str) -> Option<String> {
    const LOOP_REPEATS: usize = 8;
    const KEPT_REPEATS: usize = 3;
    let words = text.split_whitespace().collect::<Vec<_>>();
    let normalized = words
        .iter()
        .map(|word| {
            word.chars()
                .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    for phrase_words in 1..=12 {
        // The decode stopped mid-loop: up to a phrase's worth of trailing
        // words (at least one, a word cut mid-token) may follow the repeats.
        for cut_tail in 0..phrase_words.max(2) {
            let end = normalized.len().checked_sub(cut_tail)?;
            if end < phrase_words * LOOP_REPEATS {
                continue;
            }
            let phrase = &normalized[end - phrase_words..end];
            if phrase.iter().all(String::is_empty) {
                continue;
            }
            let mut repeats = 1;
            while end >= phrase_words * (repeats + 1)
                && normalized[end - phrase_words * (repeats + 1)..end - phrase_words * repeats]
                    == *phrase
            {
                repeats += 1;
            }
            if repeats >= LOOP_REPEATS {
                let keep = end - phrase_words * (repeats - KEPT_REPEATS);
                return Some(words[..keep].join(" "));
            }
        }
    }
    None
}

/// A warning when the memory free at load time held a shorter context than
/// the checkpoint's, so `duration_secs` of audio is cut into shorter chunks
/// (each a seam where speaker labels must be linked again).
pub(super) fn granite_saa_memory_limit_warning(
    duration_secs: f32,
    limit_secs: Option<f32>,
    nominal_limit_secs: Option<f32>,
) -> Option<String> {
    let (limit, nominal) = (limit_secs?, nominal_limit_secs?);
    if duration_secs <= limit || limit >= nominal * 0.98 {
        return None;
    }
    Some(format!(
        "Speaker attribution ran in chunks of at most {limit:.0} s instead of {nominal:.0} s because less memory was free when Granite loaded; reloading it with fewer models loaded gives longer chunks and fewer places where speaker labels can drift."
    ))
}

pub(super) fn granite_saa_long_form_config(model_limit_secs: Option<f32>) -> AsrLongFormConfig {
    // Cut in the longest pause of the window that ends at the cap, so a
    // chunk ends between phrases (often between speakers) instead of on
    // whatever quiet frame lies nearest the cap.
    let mut cfg = AsrLongFormConfig {
        target_chunk_secs: GRANITE_SAA_TARGET_CHUNK_SECS,
        hard_max_chunk_secs: GRANITE_SAA_HARD_MAX_CHUNK_SECS,
        overlap_secs: GRANITE_SAA_OVERLAP_SECS,
        min_chunk_secs: GRANITE_SAA_MIN_CHUNK_SECS,
        silence_search_secs: GRANITE_SAA_SILENCE_SEARCH_SECS,
        prefer_longest_pause: true,
        ..Default::default()
    };

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }
    let overrides = GraniteSaaChunkOverrides::get();
    if let Some(value) = overrides.target_chunk_secs {
        cfg.target_chunk_secs = value;
    }
    if let Some(value) = overrides.hard_max_chunk_secs {
        cfg.hard_max_chunk_secs = value;
    }
    if let Some(value) = overrides.overlap_secs {
        cfg.overlap_secs = value;
    }
    if let Some(value) = overrides.min_chunk_secs {
        cfg.min_chunk_secs = value;
    }
    if let Some(value) = overrides.silence_search_secs {
        cfg.silence_search_secs = value;
    }

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }

    // A context-derived limit below the family's minimum chunk size forces
    // graceful degradation: smaller chunks keep Speaker-Attributed ASR
    // functional on constrained hosts instead of failing every invocation.
    if let Some(limit) = model_limit_secs
        .filter(|value| value.is_finite() && *value > 0.0 && *value < cfg.min_chunk_secs)
    {
        tracing::warn!(
            context_limit_secs = limit,
            family_min_chunk_secs = cfg.min_chunk_secs,
            "Granite SAA invocation context limits chunks below the family minimum; speaker continuity degrades"
        );
        cfg.min_chunk_secs = limit.max(1.0);
    }

    cfg.hard_max_chunk_secs = cfg
        .hard_max_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    // The search window is centred on the target, so a target one window
    // below the cap searches the last stretch before it.
    cfg.target_chunk_secs = cfg
        .target_chunk_secs
        .min(cfg.hard_max_chunk_secs - cfg.silence_search_secs)
        .max(cfg.min_chunk_secs.max(1.0))
        .min(cfg.hard_max_chunk_secs);
    cfg.overlap_secs = cfg.overlap_secs.clamp(0.0, cfg.target_chunk_secs * 0.45);
    cfg
}

/// The `IZWI_GRANITE_SAA_CHUNK_*` overrides, read once per process.
#[derive(Debug, Clone, Copy)]
pub(super) struct GraniteSaaChunkOverrides {
    pub(super) target_chunk_secs: Option<f32>,
    pub(super) hard_max_chunk_secs: Option<f32>,
    pub(super) overlap_secs: Option<f32>,
    pub(super) min_chunk_secs: Option<f32>,
    pub(super) silence_search_secs: Option<f32>,
}

impl GraniteSaaChunkOverrides {
    pub(super) fn get() -> Self {
        static OVERRIDES: crate::env::ReadOnce<GraniteSaaChunkOverrides> =
            crate::env::ReadOnce::new(|| GraniteSaaChunkOverrides {
                target_chunk_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_TARGET_SECS"),
                hard_max_chunk_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MAX_SECS"),
                overlap_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_OVERLAP_SECS"),
                min_chunk_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MIN_SECS"),
                silence_search_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_SILENCE_SEARCH_SECS"),
            });
        OVERRIDES.get()
    }
}

pub(super) fn granite_saa_chunk_plan(
    samples: &[f32],
    sample_rate: u32,
    model_limit_secs: Option<f32>,
) -> Vec<AudioChunk> {
    let cfg = granite_saa_long_form_config(model_limit_secs);
    plan_audio_chunks(samples, sample_rate, &cfg, Some(cfg.hard_max_chunk_secs))
}

pub(super) fn granite_saa_max_new_tokens_for_duration(duration_secs: f32) -> usize {
    static OVERRIDE: crate::env::ReadOnce<Option<usize>> =
        crate::env::ReadOnce::new(|| env_positive_usize("IZWI_GRANITE_SAA_MAX_NEW_TOKENS"));
    if let Some(override_tokens) = OVERRIDE.get() {
        return override_tokens;
    }

    let duration_budget = if duration_secs.is_finite() && duration_secs > 0.0 {
        (duration_secs * GRANITE_SAA_NEW_TOKENS_PER_SECOND).ceil() as usize
    } else {
        0
    };
    GRANITE_SAA_NEW_TOKEN_RESERVE
        .saturating_add(duration_budget)
        .clamp(GRANITE_SAA_MIN_NEW_TOKENS, GRANITE_SAA_MAX_NEW_TOKENS)
}

pub(super) fn granite_saa_max_new_tokens(audio: &[f32], sample_rate: u32) -> usize {
    let duration_secs = if sample_rate > 0 {
        audio.len() as f32 / sample_rate as f32
    } else {
        0.0
    };
    granite_saa_max_new_tokens_for_duration(duration_secs)
}

pub(super) fn granite_saa_decode_diagnostics(
    diagnostics: Option<&serde_json::Value>,
) -> GraniteSaaChunkDecodeDiagnostics {
    let Some(diagnostics) = diagnostics else {
        return GraniteSaaChunkDecodeDiagnostics::default();
    };
    let decode = diagnostics.get("decode").unwrap_or(diagnostics);
    let prompt = diagnostics.get("prompt");

    GraniteSaaChunkDecodeDiagnostics {
        generated_tokens: decode
            .get("generated_tokens")
            .or_else(|| diagnostics.get("generated_tokens"))
            .and_then(json_usize),
        max_new_tokens: decode.get("max_new_tokens").and_then(json_usize),
        stop_reason: decode
            .get("stop_reason")
            .or_else(|| diagnostics.get("stop_reason"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        prompt_tokens: prompt
            .and_then(|value| value.get("prompt_tokens"))
            .or_else(|| diagnostics.get("prompt_tokens"))
            .and_then(json_usize),
        prompt_prefix_tokens: prompt
            .and_then(|value| value.get("prefix_tokens"))
            .or_else(|| diagnostics.get("prompt_prefix_tokens"))
            .and_then(json_usize),
    }
}

pub(super) fn granite_saa_max_token_warning(
    chunk_index: usize,
    diagnostics: &GraniteSaaChunkDecodeDiagnostics,
) -> Option<String> {
    if diagnostics.stop_reason.as_deref() != Some("max_tokens") {
        return None;
    }

    let generated = diagnostics
        .generated_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let max_new_tokens = diagnostics
        .max_new_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "configured".to_string());
    Some(format!(
        "Granite SAA chunk {} reached the decode token limit ({generated}/{max_new_tokens}); transcript may be incomplete.",
        chunk_index + 1
    ))
}

pub(super) fn json_usize(value: &serde_json::Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

pub(super) async fn granite_saa_long_form_transcribe<P>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    model: Arc<NativeAsrModel>,
    execution_contract: LoadedExecutionContract,
    state_binding: LoadedCapabilityBinding,
    samples: Arc<[f32]>,
    sample_rate: u32,
    language: Option<&str>,
    model_limit_secs: Option<f32>,
    residency_lease: ModelResidencyLease,
    retained_input_bytes: usize,
    on_progress: &mut P,
) -> Result<GraniteSaaLongFormOutput>
where
    P: FnMut(AsrProgress) + Send + 'static,
{
    let planning_samples = samples.clone();
    let planning_job = job.clone();
    let (mut residency_lease, chunks) = coordinator
        .run_host_blocking_stage(job, move || {
            planning_job.record_materialized_usage(decoded_audio_observation(
                retained_input_bytes,
                planning_samples.len(),
            )?)?;
            Ok((
                residency_lease,
                granite_saa_chunk_plan(&planning_samples, sample_rate, model_limit_secs),
            ))
        })
        .await?;
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Granite SAA chunk planner produced no chunks".to_string(),
        ));
    }

    on_progress(granite_saa_processing_progress(&chunks, sample_rate));

    let mut assembler = GraniteSaaTranscriptAssembler::default();
    let mut language_out = language.map(ToOwned::to_owned);
    let mut warnings = Vec::new();
    if chunks.len() > 1 {
        warnings.push(format!(
            "Granite SAA processed long audio in {} chunks; speaker label continuity across chunks is best-effort.",
            chunks.len()
        ));
    }

    // A chunk that runs out of decode budget (2.4 times what speech needs)
    // is caught in a repetition loop that ate the rest of its audio. It is
    // decoded again as two overlapping halves, joined like any other seam.
    let mut chunks = chunks;
    let mut idx = 0;
    while let Some(&chunk) = chunks.get(idx) {
        let chunk = &chunk;
        if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
            warnings.push(format!(
                "Granite SAA skipped invalid chunk {} with sample range {}..{}.",
                idx + 1,
                chunk.start_sample,
                chunk.end_sample
            ));
            idx += 1;
            continue;
        }

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkStarted,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));

        let chunk_start = chunk.start_sample;
        let chunk_end = chunk.end_sample;
        let task_model = model.clone();
        let task_samples = samples.clone();
        let observation_job = job.clone();
        let task_state_binding = state_binding.clone();
        let language_owned = language.map(ToOwned::to_owned);
        let chunk_sample_count = chunk_end - chunk_start;
        let max_new_tokens =
            granite_saa_max_new_tokens(&samples[chunk_start..chunk_end], sample_rate);
        let chunk_duration_secs = samples_to_seconds_f64(chunk_sample_count, sample_rate);
        tracing::info!(
            chunk_index = idx + 1,
            total_chunks = chunks.len(),
            chunk_start_secs = samples_to_seconds_f64(chunk.start_sample, sample_rate),
            chunk_end_secs = samples_to_seconds_f64(chunk.end_sample, sample_rate),
            chunk_duration_secs,
            max_new_tokens,
            "starting Granite SAA chunk decode"
        );
        let chunk_started = Instant::now();
        let (returned_lease, transcription) = coordinator
            .run_loaded_blocking_stage_with_invocation_paged(
                job,
                execution_contract.clone(),
                task_state_binding,
                WorkUnit::PipelineStage {
                    name: "asr.speaker_attributed.chunk".to_string(),
                    ordinal: idx,
                },
                move |leases| {
                    let chunk_audio = task_samples[chunk_start..chunk_end].to_vec();
                    observation_job.record_materialized_usage(
                        decoded_audio_with_scratch_observation(
                            retained_input_bytes,
                            task_samples.len(),
                            chunk_audio.capacity(),
                        )?,
                    )?;
                    let transcription = granite_saa_transcribe_chunk(
                        &task_model,
                        &chunk_audio,
                        sample_rate,
                        language_owned.as_deref(),
                        None,
                        max_new_tokens,
                        granite_saa_single_invocation_cache(leases)?,
                    );
                    let steady_usage =
                        decoded_audio_observation(retained_input_bytes, task_samples.len())?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(chunk_audio);
                    Ok((residency_lease, transcription?))
                },
            )
            .await?;
        residency_lease = returned_lease;
        let decode_diagnostics = granite_saa_decode_diagnostics(transcription.diagnostics.as_ref());
        tracing::info!(
            chunk_index = idx + 1,
            total_chunks = chunks.len(),
            elapsed_ms = chunk_started.elapsed().as_secs_f64() * 1000.0,
            generated_tokens = ?decode_diagnostics.generated_tokens,
            max_new_tokens = decode_diagnostics.max_new_tokens.unwrap_or(max_new_tokens),
            stop_reason = decode_diagnostics
                .stop_reason
                .as_deref()
                .unwrap_or("unknown"),
            prompt_tokens = ?decode_diagnostics.prompt_tokens,
            prompt_prefix_tokens = ?decode_diagnostics.prompt_prefix_tokens,
            "finished Granite SAA chunk decode"
        );

        // Transcript text stays out of info logs; at debug it lets a seam
        // linking change be replayed on a real recording.
        tracing::debug!(
            chunk_index = idx + 1,
            chunk_start_secs = samples_to_seconds_f64(chunk.start_sample, sample_rate),
            text = transcription.text.as_str(),
            "Granite SAA chunk text"
        );
        if language_out.is_none() {
            language_out = transcription.language.clone();
        }
        if transcription.text.trim().is_empty() {
            warnings.push(format!(
                "Granite SAA chunk {} returned empty text.",
                idx + 1
            ));
        }
        let mut text = transcription.text;
        if let Some(warning) = granite_saa_max_token_warning(idx, &decode_diagnostics) {
            if let Some(halves) = granite_saa_split_chunk(chunk, sample_rate) {
                tracing::warn!(
                    chunk_index = idx + 1,
                    chunk_start_secs = samples_to_seconds_f64(chunk.start_sample, sample_rate),
                    chunk_end_secs = samples_to_seconds_f64(chunk.end_sample, sample_rate),
                    "Granite SAA chunk ran out of decode budget; decoding it again in halves"
                );
                chunks.splice(idx..=idx, halves);
                continue;
            }
            match trim_trailing_repetition(&text) {
                Some(trimmed) => {
                    text = trimmed;
                    warnings.push(format!(
                        "Granite SAA chunk {} kept repeating a phrase until its token limit; the repeats were removed and part of its audio may be missing.",
                        idx + 1
                    ));
                }
                None => warnings.push(warning),
            }
            tracing::warn!(chunk_index = idx + 1, "Granite SAA chunk reached decode token limit");
        }
        warnings.extend(assembler.push_chunk_text(
            text.as_str(),
            idx,
            samples_to_seconds_f64(chunk.start_sample, sample_rate),
        ));

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkFinished,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));
        idx += 1;
    }

    on_progress(granite_saa_complete_progress(&chunks, sample_rate));

    Ok(GraniteSaaLongFormOutput {
        text: assembler.text(),
        language: language_out,
        warnings,
    })
}

pub(super) fn granite_saa_transcribe_chunk(
    model: &NativeAsrModel,
    audio: &[f32],
    sample_rate: u32,
    language: Option<&str>,
    prefix_text: Option<&str>,
    max_new_tokens: usize,
    cache: &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
) -> Result<NativeAsrTranscription> {
    if let Some(rate) = model.audio_token_rate() {
        // Uniform invocation admission: refuse before spending encoder work
        // on an invocation whose demand provably cannot fit. The family's
        // exact per-step check remains the authority behind this estimate.
        let duration_secs = if sample_rate > 0 {
            audio.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let audio_tokens = (duration_secs * rate).ceil() as usize;
        let prefix_tokens = prefix_text.map_or(0, |text| text.chars().count() / 4);
        let demand = ASR_TEXT_PROMPT_TOKEN_RESERVE
            .saturating_add(audio_tokens)
            .saturating_add(prefix_tokens)
            .saturating_add(max_new_tokens);
        let capacity = cache.capacity_tokens();
        if demand > capacity {
            return Err(Error::InvalidInput(format!(
                "Speaker-attributed ASR invocation needs about {demand} KV tokens (prompt reserve + {audio_tokens} audio + {prefix_tokens} prefix + {max_new_tokens} decode) but the invocation cache has capacity for {capacity}; reduce chunk length or free memory"
            )));
        }
    }
    model.transcribe_granite_speech_task_and_options_physical(
        audio,
        sample_rate,
        language,
        GraniteSpeechTask::SpeakerAttributed,
        prefix_text,
        NativeAsrGenerationOptions {
            max_new_tokens,
            ..NativeAsrGenerationOptions::default()
        },
        cache,
    )
}

pub(super) fn granite_saa_single_invocation_cache(
    leases: &mut crate::kv::v2::InvocationPagedLeaseSetV2,
) -> Result<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
    let domains = leases.domains().collect::<Vec<_>>();
    let [domain] = domains.as_slice() else {
        return Err(Error::InferenceError(format!(
            "Granite speaker-attributed ASR requires one invocation KV domain, found {}",
            domains.len()
        )));
    };
    leases.cache_mut(*domain)
}

pub(super) fn granite_saa_processing_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Processing,
        current_chunk: None,
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(0.0),
        total_audio_secs,
        percent: Some(0.0),
    }
}

pub(super) fn granite_saa_complete_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Complete,
        current_chunk: Some(chunks.len()),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: total_audio_secs,
        total_audio_secs,
        percent: Some(100.0),
    }
}

pub(super) fn granite_saa_chunk_progress(
    phase: AsrProgressPhase,
    index: usize,
    chunk: &AudioChunk,
    chunks: &[AudioChunk],
    sample_rate: u32,
) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|last| samples_to_seconds_f64(last.end_sample, sample_rate));
    let processed_audio_secs = match phase {
        AsrProgressPhase::ChunkStarted => samples_to_seconds_f64(chunk.start_sample, sample_rate),
        AsrProgressPhase::ChunkFinished => samples_to_seconds_f64(chunk.end_sample, sample_rate),
        AsrProgressPhase::Processing => 0.0,
        AsrProgressPhase::Aligning | AsrProgressPhase::Complete => {
            total_audio_secs.unwrap_or_default()
        }
    };
    let percent = total_audio_secs
        .filter(|total| *total > 0.0)
        .map(|total| ((processed_audio_secs / total) * 100.0).clamp(0.0, 100.0));

    AsrProgress {
        phase,
        current_chunk: Some(index + 1),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(processed_audio_secs),
        total_audio_secs,
        percent,
    }
}

pub(super) fn samples_to_seconds_f64(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        samples as f64 / sample_rate as f64
    }
}

pub(super) fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

pub(super) fn env_positive_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

#[cfg(test)]
pub(super) fn speaker_attributed_asr_result_from_text(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> SpeakerAttributedAsrResult {
    speaker_attributed_asr_result_from_text_with_warnings(
        raw_text,
        language,
        duration_secs,
        min_speakers,
        max_speakers,
        Vec::new(),
    )
}

pub(super) fn speaker_attributed_asr_result_from_text_with_warnings(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    mut warnings: Vec<String>,
) -> SpeakerAttributedAsrResult {
    let parsed = parse_granite_speech_output(raw_text);
    let mut speakers = HashSet::<String>::new();
    let mut turns = speaker_attributed_asr_turns_from_text(raw_text);

    for turn in &turns {
        let speaker = turn.speaker.trim();
        if speaker != UNKNOWN_SAA_SPEAKER {
            speakers.insert(speaker.to_string());
        }
    }
    if turns.is_empty() && !parsed.text.trim().is_empty() {
        turns.push(SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: parsed.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        });
    }

    let speaker_count = speakers.len();
    if let Some(min_speakers) = min_speakers {
        if speaker_count < min_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), below requested minimum {min_speakers}."
            ));
        }
    }
    if let Some(max_speakers) = max_speakers {
        if speaker_count > max_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), above requested maximum {max_speakers}."
            ));
        }
    }
    if parsed.text.trim().is_empty() {
        warnings.push("Granite SAA returned an empty transcript.".to_string());
    }

    SpeakerAttributedAsrResult {
        text: parsed.text,
        language,
        duration_secs,
        speaker_turns: turns,
        speaker_count,
        status: if warnings.is_empty() {
            SpeakerAttributedAsrStatus::Ready
        } else {
            SpeakerAttributedAsrStatus::Warning
        },
        warnings,
    }
}

pub(super) fn format_saa_turns(turns: &[SpeakerAttributedAsrTurn]) -> String {
    turns
        .iter()
        .map(format_saa_turn)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn format_saa_turn(turn: &SpeakerAttributedAsrTurn) -> String {
    format!("[{}]: {}", turn.speaker, turn.text.trim())
}

pub(super) fn append_with_spacing(target: &mut String, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if !target.trim().is_empty() && !target.ends_with(char::is_whitespace) {
        target.push(' ');
    }
    target.push_str(text);
}

/// The turns a speaker-tagging model's transcription reports
/// (`asr_diagnostics["speaker_turns"]`: `speaker` is the `<spk:N>` tag or
/// null before the first tag, `start` and `end` in seconds when its CTC
/// stage timed them), tag N as "Speaker N+1", consecutive turns of one
/// speaker merged.
pub(super) fn speaker_tagged_turns(
    diagnostics: Option<&serde_json::Value>,
) -> Vec<SpeakerAttributedAsrTurn> {
    let turns = diagnostics
        .and_then(|diagnostics| diagnostics.get("speaker_turns"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let secs = |turn: &serde_json::Value, key: &str| {
        turn.get(key)
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value as f32)
    };
    let mut out: Vec<SpeakerAttributedAsrTurn> = Vec::new();
    for turn in turns {
        let text = turn
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim();
        if text.is_empty() {
            continue;
        }
        let speaker = turn
            .get("speaker")
            .and_then(serde_json::Value::as_u64)
            .map_or_else(
                || UNKNOWN_SAA_SPEAKER.to_string(),
                |tag| format!("Speaker {}", tag + 1),
            );
        let (start_secs, end_secs) = (secs(turn, "start"), secs(turn, "end"));
        if let Some(last) = out.last_mut().filter(|last| last.speaker == speaker) {
            append_with_spacing(&mut last.text, text);
            last.end_secs = end_secs.or(last.end_secs);
            continue;
        }
        out.push(SpeakerAttributedAsrTurn {
            speaker,
            text: text.to_string(),
            start_secs,
            end_secs,
        });
    }
    out
}

impl RuntimeService {
    /// One background transcription by a speaker-tagging model with speaker
    /// tags and word timestamps: its `asr_diagnostics` carry `speaker_turns`
    /// and the CTC stage's `ctc_timestamps` (speaker-attributed ASR and
    /// one-model diarization).
    pub(in crate::runtime) async fn speaker_tagged_asr_output(
        &self,
        audio_input: AsrAudioInput<'_>,
        variant: ModelVariant,
        language: Option<&str>,
    ) -> Result<EngineOutput> {
        let admitted = self
            .build_asr_request(
                variant,
                audio_input,
                language,
                None,
                AsrTaskOptions::default().with_speaker_attribution(true),
                None,
                None,
                RuntimeRequestContext::new(crate::engine::WorkloadClass::Background),
                false,
                true,
            )
            .await?;
        self.run_admitted_request(admitted).await
    }

    /// Speaker-attributed ASR on a model that tags speakers on its own ASR
    /// rows (Nemotron 3.5 Transcribe): one transcription request with the
    /// card's speaker-tag instruction and word timestamps, whose speaker
    /// turns (timed when its CTC stage aligned them) are the result.
    async fn speaker_tagged_asr_input_with_progress<P>(
        &self,
        audio_input: AsrAudioInput<'_>,
        variant: ModelVariant,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        mut on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let progress = |phase, percent| AsrProgress {
            phase,
            current_chunk: None,
            total_chunks: Some(1),
            processed_audio_secs: None,
            total_audio_secs: None,
            percent: Some(percent),
        };
        on_progress(progress(AsrProgressPhase::Processing, 0.0));
        let output = self
            .speaker_tagged_asr_output(audio_input, variant, language)
            .await?;
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(&output.text.unwrap_or_default());
        let turns = speaker_tagged_turns(output.asr_diagnostics.as_ref());
        let speaker_count = turns
            .iter()
            .filter(|turn| turn.speaker != UNKNOWN_SAA_SPEAKER)
            .map(|turn| turn.speaker.as_str())
            .collect::<HashSet<_>>()
            .len();
        let mut warnings = Vec::new();
        if min_speakers.is_some_and(|min| speaker_count < min) {
            warnings.push(format!(
                "{variant} tagged {speaker_count} speaker(s), below the requested minimum {}.",
                min_speakers.unwrap_or_default()
            ));
        }
        if max_speakers.is_some_and(|max| speaker_count > max) {
            warnings.push(format!(
                "{variant} tagged {speaker_count} speaker(s), above the requested maximum {}.",
                max_speakers.unwrap_or_default()
            ));
        }
        if text.trim().is_empty() {
            warnings.push(format!("{variant} returned an empty transcript."));
        }
        if !turns.is_empty() && turns.iter().all(|turn| turn.start_secs.is_none()) {
            warnings.push(format!(
                "{variant} could not time its speaker turns; they are untimed."
            ));
        }
        on_progress(progress(AsrProgressPhase::Complete, 100.0));
        Ok(SpeakerAttributedAsrResult {
            text,
            language: language.map(str::to_string),
            duration_secs,
            speaker_turns: turns,
            speaker_count,
            status: if warnings.is_empty() {
                SpeakerAttributedAsrStatus::Ready
            } else {
                SpeakerAttributedAsrStatus::Warning
            },
            warnings,
        })
    }

    pub async fn speaker_attributed_asr(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        self.speaker_attributed_asr_input_with_progress(
            AsrAudioInput::Base64(audio_base64),
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        self.speaker_attributed_asr_bytes_with_progress(
            audio_bytes,
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes_with_progress<P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        self.speaker_attributed_asr_input_with_progress(
            AsrAudioInput::Bytes(audio_bytes),
            model_id,
            language,
            min_speakers,
            max_speakers,
            on_progress,
        )
        .await
    }

    pub(super) async fn speaker_attributed_asr_input_with_progress<P>(
        &self,
        audio_input: AsrAudioInput<'_>,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        mut on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_speaker_attributed_asr_variant(model_id)?;
        if variant.supports_asr_speaker_tags() {
            return self
                .speaker_tagged_asr_input_with_progress(
                    audio_input,
                    variant,
                    language,
                    min_speakers,
                    max_speakers,
                    on_progress,
                )
                .await;
        }
        validate_asr_language(variant, language)?;
        let audio_bytes = audio_input.input_bytes();
        if audio_bytes == 0 {
            return Err(Error::InvalidInput(
                "ASR request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        let input_bytes = audio_bytes
            .checked_add(language.map(str::len).unwrap_or_default())
            .ok_or_else(|| {
                Error::Overloaded("speaker-attributed ASR input overflowed".to_string())
            })?;
        self.observe_broker_capability_request(
            CapabilityKind::SpeakerAttributedAsr,
            Some(variant),
            false,
        )?;
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Background);
        let spec = self.coordinator_job_for_audio_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Pipeline,
            context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;
        let owned_audio = audio_input.into_owned_for_job(&job).await?;
        let language_owned =
            copy_optional_preparation_string(&job, language, "speaker-attributed ASR language")
                .await?;
        let language_bytes = language_owned.as_ref().map_or(0, String::capacity);
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            owned_audio.retained_bytes(),
            language_bytes,
        ])?)?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::SpeakerAttributedAsr,
                false,
                ExecutionTargetKind::PipelineRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let observation_job = job.clone();
        let (residency_lease, samples, sample_rate) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let retained_audio_bytes = owned_audio.retained_bytes();
                let (samples, sample_rate) = owned_audio.decode()?;
                let samples = Arc::<[f32]>::from(samples);
                let steady_usage = decoded_audio_observation(
                    input_bytes.checked_add(language_bytes).ok_or_else(|| {
                        Error::Overloaded(
                            "speaker-attributed ASR retained input overflowed".to_string(),
                        )
                    })?,
                    samples.len(),
                )?;
                observation_job.record_materialized_usage(add_retained_host_bytes(
                    steady_usage,
                    retained_audio_bytes,
                )?)?;
                observation_job.prepare_materialized_release(steady_usage)?;
                drop(owned_audio);
                Ok((residency_lease, samples, sample_rate))
            })
            .await?;
        let duration_secs = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let retained_input_bytes = input_bytes.checked_add(language_bytes).ok_or_else(|| {
            Error::Overloaded("speaker-attributed ASR retained input overflowed".to_string())
        })?;
        let model_limit_secs = compose_invocation_audio_limit(
            model.max_audio_seconds_hint(),
            model.audio_token_rate(),
            self.model_registry.effective_context(variant),
            granite_saa_max_new_tokens_for_duration,
        )?;
        // The context a load holds depends on the memory free when it loaded,
        // so the same file can be cut into shorter chunks (more seams) on a
        // busier machine; say so instead of leaving it silent.
        let nominal_limit_secs = compose_invocation_audio_limit(
            model.max_audio_seconds_hint(),
            model.audio_token_rate(),
            model.nominal_decoder_context(),
            granite_saa_max_new_tokens_for_duration,
        )
        .ok()
        .flatten();
        let memory_limit_warning = granite_saa_memory_limit_warning(
            duration_secs,
            model_limit_secs,
            nominal_limit_secs,
        );
        tracing::info!(
            variant = %variant,
            model_limit_secs = ?model_limit_secs,
            nominal_limit_secs = ?nominal_limit_secs,
            "resolved speaker-attributed ASR invocation audio limit"
        );
        let long_form = granite_saa_long_form_transcribe(
            &self.coordinator,
            &job,
            model,
            execution_contract,
            state_binding,
            samples,
            sample_rate,
            language_owned.as_deref(),
            model_limit_secs,
            residency_lease,
            retained_input_bytes,
            &mut on_progress,
        )
        .await?;

        let mut warnings = long_form.warnings;
        warnings.extend(memory_limit_warning);
        Ok(speaker_attributed_asr_result_from_text_with_warnings(
            long_form.text.as_str(),
            long_form.language.or(language_owned),
            duration_secs,
            min_speakers,
            max_speakers,
            warnings,
        ))
    }
}
