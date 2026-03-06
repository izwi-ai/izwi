//! Diarization runtime methods.

use crate::catalog::{
    resolve_asr_model_variant, resolve_diarization_llm_variant, resolve_diarization_model_variant,
};
use crate::error::{Error, Result};
use crate::models::registry::NativeAsrModel;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    DiarizationConfig, DiarizationResult, DiarizationSegment, DiarizationTranscriptResult,
    DiarizationUtterance, DiarizationWord,
};
use crate::ModelVariant;
use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, AudioChunk, TranscriptAssembler};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::warn;

const UNKNOWN_SPEAKER: &str = "UNKNOWN";
const MAX_UTTERANCE_GAP_SECS: f32 = 0.9;
const MIN_ALIGNMENT_COVERAGE: f32 = 0.25;
const MAX_UNATTRIBUTED_WORD_RATIO: f32 = 0.4;
const ALIGNMENT_COLLAPSE_TAIL_MS: u32 = 250;
const ALIGNMENT_PREFIX_CLUSTER_MS_CAP: u32 = 1_000;
const MAX_REASONABLE_WORD_SPAN_MS: u32 = 2_500;
const PIPELINE_SAMPLE_RATE: u32 = 16_000;
const MAX_FRAGMENT_WORDS: usize = 3;
const MAX_FRAGMENT_DURATION_SECS: f32 = 1.25;
const FRAGMENT_CONFIDENCE_MARGIN: f32 = 0.08;
const PHRASE_CAPITALIZED_GAP_SECS: f32 = 0.35;
const PHRASE_HARD_GAP_SECS: f32 = 0.75;

#[derive(Debug, Clone)]
struct PipelineAudio {
    samples: Vec<f32>,
    sample_rate: u32,
    duration_secs: f32,
}

#[derive(Debug, Clone)]
struct TranscribedChunk {
    range: AudioChunk,
    text: String,
    language: Option<String>,
}

impl RuntimeService {
    async fn diarize_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
        model_id: Option<&str>,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let variant = resolve_diarization_model_variant(model_id);
        self.load_model(variant).await?;

        let model = self
            .model_registry
            .get_diarization(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        model.diarize(samples, sample_rate, config)
    }

    /// Run speaker diarization over a single audio input.
    pub async fn diarize(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let audio_bytes = base64_decode(audio_base64)?;
        self.diarize_bytes(&audio_bytes, model_id, config).await
    }

    pub async fn diarize_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let audio = decode_pipeline_audio_bytes(audio_bytes)?;
        self.diarize_samples(&audio.samples, audio.sample_rate, model_id, config)
            .await
    }

    /// Run diarization and produce speaker-attributed transcript artifacts.
    pub async fn diarize_with_transcript(
        &self,
        audio_base64: &str,
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        llm_model_id: Option<&str>,
        config: &DiarizationConfig,
        enable_llm_refinement: bool,
    ) -> Result<DiarizationTranscriptResult> {
        let audio_bytes = base64_decode(audio_base64)?;
        self.diarize_with_transcript_bytes(
            &audio_bytes,
            diarization_model_id,
            asr_model_id,
            aligner_model_id,
            llm_model_id,
            config,
            enable_llm_refinement,
        )
        .await
    }

    pub async fn diarize_with_transcript_bytes(
        &self,
        audio_bytes: &[u8],
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        llm_model_id: Option<&str>,
        config: &DiarizationConfig,
        enable_llm_refinement: bool,
    ) -> Result<DiarizationTranscriptResult> {
        let audio = decode_pipeline_audio_bytes(audio_bytes)?;
        let diarization = self
            .diarize_samples(
                &audio.samples,
                audio.sample_rate,
                diarization_model_id,
                config,
            )
            .await?;

        let asr_variant = resolve_asr_model_variant(asr_model_id);

        let aligner_variant =
            crate::runtime::asr::resolve_forced_aligner_variant(aligner_model_id)?;
        let aligner_model = match self.load_model(aligner_variant).await {
            Ok(()) => match self.model_registry.get_asr(aligner_variant).await {
                Some(model) => Some(model),
                None => {
                    warn!(
                        "Forced aligner {} was loaded but not found in registry",
                        aligner_variant
                    );
                    None
                }
            },
            Err(err) => {
                warn!("Forced aligner load failed, using heuristic timings: {err}");
                None
            }
        };

        let aligner_limit = aligner_model
            .as_ref()
            .and_then(|model| model.max_audio_seconds_hint());
        let use_single_pass_asr = should_use_single_pass_diarization_asr(
            audio.duration_secs,
            aligner_limit,
            aligner_model.is_some(),
        );

        let (asr_text, chunk_texts, detected_language) = if use_single_pass_asr {
            self.load_model(asr_variant).await?;
            let asr_model = self
                .model_registry
                .get_asr(asr_variant)
                .await
                .ok_or_else(|| Error::ModelNotFound(asr_variant.to_string()))?;
            let audio_samples = audio.samples.clone();
            let asr_model_for_task = asr_model.clone();
            let transcription = tokio::task::spawn_blocking(move || {
                asr_model_for_task.transcribe_with_details(
                    &audio_samples,
                    PIPELINE_SAMPLE_RATE,
                    None,
                )
            })
            .await
            .map_err(|err| Error::InferenceError(format!("ASR task failed: {err}")))??;
            (transcription.text, Vec::new(), transcription.language)
        } else {
            self.load_model(asr_variant).await?;
            let asr_model = self
                .model_registry
                .get_asr(asr_variant)
                .await
                .ok_or_else(|| Error::ModelNotFound(asr_variant.to_string()))?;
            let (text, chunks) =
                transcribe_audio_chunks(asr_model, &audio, None, aligner_limit).await?;
            (text, chunks, None)
        };
        let asr_words = extract_words(&asr_text);

        let mut model_aligned_words = 0usize;
        let mut alignments = if asr_text.is_empty() {
            Vec::new()
        } else if use_single_pass_asr && aligner_model.is_some() {
            match self
                .force_align_bytes_with_model_and_language(
                    audio_bytes,
                    &asr_text,
                    detected_language.as_deref(),
                    aligner_model_id,
                )
                .await
            {
                Ok(aligned) => {
                    model_aligned_words = aligned.len();
                    aligned
                }
                Err(err) => {
                    warn!("Forced alignment failed, using heuristic timings: {err}");
                    fallback_word_timings_from_words(&asr_words, audio.duration_secs)
                }
            }
        } else if use_single_pass_asr {
            fallback_word_timings_from_words(&asr_words, audio.duration_secs)
        } else if let Some(model) = aligner_model.as_ref() {
            let (aligned, aligned_word_count) =
                force_align_audio_chunks(model.clone(), &audio, &chunk_texts).await;
            model_aligned_words = aligned_word_count;
            aligned
        } else {
            fallback_word_timings_from_chunks(&chunk_texts, audio.sample_rate)
        };

        if alignment_is_suspicious(&alignments, asr_words.len(), diarization.duration_secs) {
            warn!(
                "Forced aligner output looked invalid, using diarization-guided fallback timings"
            );
            alignments = fallback_word_timings_with_segments(
                &asr_words,
                &diarization.segments,
                diarization.duration_secs,
            );
            model_aligned_words = 0;
        }

        let (mut words, overlap_assigned_words, mut unattributed_words) =
            attribute_words_to_speakers(&alignments, &diarization.segments);
        stabilize_word_speakers(&mut words);
        normalize_phrase_speakers(&mut words, &diarization.segments);

        if attribution_requires_fallback(words.len(), overlap_assigned_words, unattributed_words) {
            let attribution_coverage = overlap_assigned_words as f32 / words.len() as f32;
            let unattributed_ratio = unattributed_words as f32 / words.len() as f32;
            warn!(
                "Speaker attribution quality too low ({:.1}% overlap-backed, {:.1}% unattributed), retrying with diarization-guided fallback timings",
                attribution_coverage * 100.0,
                unattributed_ratio * 100.0
            );
            alignments = fallback_word_timings_with_segments(
                &asr_words,
                &diarization.segments,
                diarization.duration_secs,
            );
            let (fallback_words, _fallback_overlap_assigned_words, fallback_unattributed_words) =
                attribute_words_to_speakers(&alignments, &diarization.segments);
            words = fallback_words;
            stabilize_word_speakers(&mut words);
            normalize_phrase_speakers(&mut words, &diarization.segments);
            unattributed_words = fallback_unattributed_words;
            model_aligned_words = 0;
        }

        let utterances = build_utterances(&words);
        let raw_transcript = if utterances.is_empty() {
            asr_text.clone()
        } else {
            format_utterance_transcript(&utterances)
        };

        let raw_transcript_trimmed = raw_transcript.trim();
        let mut transcript = raw_transcript.clone();
        let mut llm_refined = false;
        if enable_llm_refinement && !raw_transcript_trimmed.is_empty() {
            let llm_variant = resolve_chat_variant(llm_model_id)?;
            match self
                .polish_diarized_transcript(llm_variant, &raw_transcript)
                .await
            {
                Ok(polished) if !polished.trim().is_empty() => {
                    let polished_trimmed = polished.trim();
                    transcript = polished_trimmed.to_string();
                    llm_refined = polished_trimmed != raw_transcript_trimmed;
                }
                Ok(_) => {}
                Err(err) => {
                    warn!("Transcript refinement failed, returning raw speaker transcript: {err}");
                }
            }
        }

        let alignment_coverage = if words.is_empty() {
            0.0
        } else {
            (model_aligned_words.min(words.len())) as f32 / words.len() as f32
        };

        Ok(DiarizationTranscriptResult {
            segments: diarization.segments,
            words,
            utterances,
            asr_text,
            raw_transcript,
            transcript,
            duration_secs: diarization.duration_secs,
            speaker_count: diarization.speaker_count,
            alignment_coverage,
            unattributed_words,
            llm_refined,
        })
    }

    async fn polish_diarized_transcript(
        &self,
        llm_variant: ModelVariant,
        raw_transcript: &str,
    ) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a diarized transcript editor. Return only final transcript lines, with no analysis or hidden reasoning. Never output tags (including <think>), markdown, code fences, or commentary. Keep speaker labels and timestamps exactly unchanged. Keep line count and line order exactly unchanged. Do not invent, repeat, or omit spoken content. Only improve punctuation and readability of spoken text after the colon on each line."
                    .to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: format!(
                    "Rewrite this diarized transcript with minimal edits.\nRules:\n- Keep exactly one output line per input line.\n- Preserve each leading speaker label + timestamp prefix exactly as-is.\n- Edit only the spoken text after the colon.\n- Do not add, remove, or merge lines.\n- Do not invent new words, drop spoken words, or repeat content not present in the line.\n- Output only the final transcript lines.\n\n{}",
                    raw_transcript
                ),
            },
        ];

        let generation = self.chat_generate(llm_variant, messages, 1024).await?;
        Ok(sanitize_refined_transcript(
            &generation.text,
            raw_transcript,
        ))
    }
}

fn resolve_chat_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    resolve_diarization_llm_variant(model_id).map_err(|err| Error::InvalidInput(err.to_string()))
}

fn decode_pipeline_audio(audio_base64: &str) -> Result<PipelineAudio> {
    let audio_bytes = base64_decode(audio_base64)?;
    decode_pipeline_audio_bytes(&audio_bytes)
}

fn decode_pipeline_audio_bytes(audio_bytes: &[u8]) -> Result<PipelineAudio> {
    let (samples, sample_rate) = decode_audio_bytes(audio_bytes)?;
    let normalized = resample_linear(&samples, sample_rate, PIPELINE_SAMPLE_RATE);
    let duration_secs = if PIPELINE_SAMPLE_RATE > 0 {
        normalized.len() as f32 / PIPELINE_SAMPLE_RATE as f32
    } else {
        0.0
    };

    Ok(PipelineAudio {
        samples: normalized,
        sample_rate: PIPELINE_SAMPLE_RATE,
        duration_secs,
    })
}

fn pipeline_chunk_config() -> AsrLongFormConfig {
    let mut cfg = AsrLongFormConfig::default();
    if let Some(v) = env_positive_f32("IZWI_ASR_CHUNK_TARGET_SECS") {
        cfg.target_chunk_secs = v;
    }
    if let Some(v) = env_positive_f32("IZWI_ASR_CHUNK_MAX_SECS") {
        cfg.hard_max_chunk_secs = v;
    }
    if let Some(v) = env_positive_f32("IZWI_ASR_CHUNK_OVERLAP_SECS") {
        cfg.overlap_secs = v;
    }
    cfg
}

fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn combined_chunk_limit(asr_limit: Option<f32>, aligner_limit: Option<f32>) -> Option<f32> {
    match (asr_limit, aligner_limit) {
        (Some(asr), Some(aligner)) => Some(asr.min(aligner)),
        (Some(asr), None) => Some(asr),
        (None, Some(aligner)) => Some(aligner),
        (None, None) => None,
    }
}

fn should_use_single_pass_diarization_asr(
    duration_secs: f32,
    aligner_limit: Option<f32>,
    aligner_available: bool,
) -> bool {
    if !aligner_available {
        return true;
    }

    match aligner_limit {
        Some(limit) if limit.is_finite() && limit > 0.0 => duration_secs <= limit,
        _ => true,
    }
}

async fn transcribe_audio_chunks(
    model: Arc<NativeAsrModel>,
    audio: &PipelineAudio,
    language: Option<&str>,
    aligner_limit: Option<f32>,
) -> Result<(String, Vec<TranscribedChunk>)> {
    let cfg = pipeline_chunk_config();
    let chunk_limit = combined_chunk_limit(model.max_audio_seconds_hint(), aligner_limit);
    let chunks = plan_audio_chunks(&audio.samples, audio.sample_rate, &cfg, chunk_limit);
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "ASR chunk planner produced no chunks".to_string(),
        ));
    }

    let language = language.map(|value| value.to_string());
    let mut assembler = TranscriptAssembler::new(cfg);
    let mut transcribed = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        if chunk.end_sample <= chunk.start_sample || chunk.end_sample > audio.samples.len() {
            continue;
        }
        let chunk_audio = audio.samples[chunk.start_sample..chunk.end_sample].to_vec();
        let model = model.clone();
        let language = language.clone();
        let transcription = tokio::task::spawn_blocking(move || {
            model.transcribe_with_details(&chunk_audio, PIPELINE_SAMPLE_RATE, language.as_deref())
        })
        .await
        .map_err(|err| Error::InferenceError(format!("ASR task failed: {err}")))??;
        assembler.push_chunk_text(&transcription.text);
        transcribed.push(TranscribedChunk {
            range: chunk,
            text: transcription.text,
            language: transcription.language,
        });
    }

    Ok((assembler.finish().trim().to_string(), transcribed))
}

async fn force_align_audio_chunks(
    model: Arc<NativeAsrModel>,
    audio: &PipelineAudio,
    chunks: &[TranscribedChunk],
) -> (Vec<(String, u32, u32)>, usize) {
    let mut merged = Vec::new();
    let mut model_aligned_words = 0usize;

    for chunk in chunks {
        let words = extract_words(&chunk.text);
        if words.is_empty()
            || chunk.range.end_sample <= chunk.range.start_sample
            || chunk.range.end_sample > audio.samples.len()
        {
            continue;
        }

        let chunk_audio = audio.samples[chunk.range.start_sample..chunk.range.end_sample].to_vec();
        let chunk_duration_secs = chunk_audio.len() as f32 / audio.sample_rate.max(1) as f32;
        let chunk_start_ms = samples_to_ms(chunk.range.start_sample, audio.sample_rate);
        let text = chunk.text.clone();
        let language = chunk.language.clone();
        let model_for_task = model.clone();

        let aligned = match tokio::task::spawn_blocking(move || {
            model_for_task.force_align(
                &chunk_audio,
                PIPELINE_SAMPLE_RATE,
                &text,
                language.as_deref(),
            )
        })
        .await
        {
            Ok(Ok(aligned))
                if !aligned.is_empty()
                    && !alignment_is_suspicious(&aligned, words.len(), chunk_duration_secs) =>
            {
                model_aligned_words += aligned.len();
                aligned
            }
            Ok(Ok(_)) => fallback_word_timings_from_words(&words, chunk_duration_secs),
            Ok(Err(err)) => {
                warn!("Forced alignment failed for one chunk, using interval fallback: {err}");
                fallback_word_timings_from_words(&words, chunk_duration_secs)
            }
            Err(err) => {
                warn!("Forced alignment task failed for one chunk, using interval fallback: {err}");
                fallback_word_timings_from_words(&words, chunk_duration_secs)
            }
        };

        let rebased = aligned
            .into_iter()
            .map(|(word, start, end)| {
                (
                    word,
                    chunk_start_ms.saturating_add(start),
                    chunk_start_ms.saturating_add(end),
                )
            })
            .collect::<Vec<_>>();
        append_chunk_alignments(&mut merged, rebased);
    }

    (merged, model_aligned_words)
}

fn fallback_word_timings_from_chunks(
    chunks: &[TranscribedChunk],
    sample_rate: u32,
) -> Vec<(String, u32, u32)> {
    let mut alignments = Vec::new();

    for chunk in chunks {
        let words = extract_words(&chunk.text);
        if words.is_empty() || chunk.range.end_sample <= chunk.range.start_sample {
            continue;
        }
        let chunk_duration_secs = chunk.range.len_samples() as f32 / sample_rate.max(1) as f32;
        let chunk_start_ms = samples_to_ms(chunk.range.start_sample, sample_rate);
        let chunk_alignments = fallback_word_timings_from_words(&words, chunk_duration_secs)
            .into_iter()
            .map(|(word, start, end)| {
                (
                    word,
                    chunk_start_ms.saturating_add(start),
                    chunk_start_ms.saturating_add(end),
                )
            })
            .collect::<Vec<_>>();
        append_chunk_alignments(&mut alignments, chunk_alignments);
    }

    alignments
}

fn fallback_word_timings(text: &str, duration_secs: f32) -> Vec<(String, u32, u32)> {
    let words = extract_words(text);
    fallback_word_timings_from_words(&words, duration_secs)
}

fn fallback_word_timings_from_words(
    words: &[String],
    duration_secs: f32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let max_duration_ms = secs_to_ms(duration_secs);
    let duration_ms = if max_duration_ms > 0 {
        max_duration_ms
    } else {
        (words.len() as u32).saturating_mul(300).max(1)
    };
    let step = (duration_ms as f32 / words.len() as f32).max(1.0);

    words
        .into_iter()
        .enumerate()
        .map(|(idx, word)| {
            let start = ((idx as f32) * step).round() as u32;
            let mut end = (((idx + 1) as f32) * step).round() as u32;
            if end <= start {
                end = start.saturating_add(1);
            }
            (word.clone(), start, end)
        })
        .collect()
}

fn fallback_word_timings_with_segments(
    words: &[String],
    segments: &[DiarizationSegment],
    duration_secs: f32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut windows = segments
        .iter()
        .filter_map(|segment| {
            let start = secs_to_ms(segment.start_secs);
            let mut end = secs_to_ms(segment.end_secs);
            if end <= start {
                end = start.saturating_add(1);
            }
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
    windows.sort_by_key(|(start, _)| *start);

    if windows.is_empty() {
        return fallback_word_timings_from_words(words, duration_secs);
    }

    let durations = windows
        .iter()
        .map(|(start, end)| end.saturating_sub(*start).max(1))
        .collect::<Vec<_>>();
    let total_duration_ms = durations.iter().copied().sum::<u32>();
    if total_duration_ms == 0 {
        return fallback_word_timings_from_words(words, duration_secs);
    }

    let word_count = words.len();
    let mut allocations = vec![0usize; windows.len()];
    let mut assigned = 0usize;
    let mut remainders = Vec::with_capacity(windows.len());

    for (idx, duration_ms) in durations.iter().copied().enumerate() {
        let target = word_count as f32 * duration_ms as f32 / total_duration_ms as f32;
        let base = target.floor() as usize;
        allocations[idx] = base;
        assigned += base;
        remainders.push((target - base as f32, idx));
    }

    let mut remaining = word_count.saturating_sub(assigned);
    remainders.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, idx) in remainders {
        if remaining == 0 {
            break;
        }
        allocations[idx] += 1;
        remaining -= 1;
    }

    let allocated_total = allocations.iter().sum::<usize>();
    if allocated_total < word_count {
        if let Some(last) = allocations.last_mut() {
            *last += word_count - allocated_total;
        }
    } else if allocated_total > word_count {
        let mut excess = allocated_total - word_count;
        for allocation in allocations.iter_mut().rev() {
            if excess == 0 {
                break;
            }
            let delta = (*allocation).min(excess);
            *allocation -= delta;
            excess -= delta;
        }
    }

    let mut alignments = Vec::with_capacity(word_count);
    let mut word_idx = 0usize;
    for ((segment_start, segment_end), allocation) in windows.into_iter().zip(allocations) {
        if allocation == 0 {
            continue;
        }
        let segment_span = segment_end.saturating_sub(segment_start).max(1);
        let step = segment_span as f32 / allocation as f32;

        for local_idx in 0..allocation {
            if word_idx >= word_count {
                break;
            }
            let start = segment_start.saturating_add((local_idx as f32 * step).floor() as u32);
            let mut end = if local_idx + 1 == allocation {
                segment_end
            } else {
                segment_start.saturating_add(((local_idx + 1) as f32 * step).floor() as u32)
            };
            if end <= start {
                end = start.saturating_add(1);
            }
            alignments.push((words[word_idx].clone(), start, end));
            word_idx += 1;
        }
    }

    if word_idx < word_count {
        let remaining_words = &words[word_idx..];
        let mut carry = fallback_word_timings_from_words(remaining_words, duration_secs);
        alignments.append(&mut carry);
    }

    alignments
}

fn extract_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn alignment_is_suspicious(
    alignments: &[(String, u32, u32)],
    expected_word_count: usize,
    duration_secs: f32,
) -> bool {
    if expected_word_count == 0 {
        return false;
    }
    if alignments.is_empty() {
        return true;
    }
    if alignments.len() < expected_word_count.saturating_div(2).max(1) {
        return true;
    }

    let duration_ms = secs_to_ms(duration_secs).max(1);
    let tail_start = duration_ms.saturating_sub(ALIGNMENT_COLLAPSE_TAIL_MS);
    let prefix_window_end = ((duration_ms / 50).max(250)).min(ALIGNMENT_PREFIX_CLUSTER_MS_CAP);

    let mut min_start = u32::MAX;
    let mut max_end = 0u32;
    let mut tiny_spans = 0usize;
    let mut tail_heavy = 0usize;
    let mut prefix_clustered = 0usize;
    let mut max_word_span_ms = 0u32;

    for (_, start, end) in alignments {
        let bounded_start = (*start).min(duration_ms);
        let bounded_end = (*end).min(duration_ms).max(bounded_start);
        let span_ms = bounded_end.saturating_sub(bounded_start);

        min_start = min_start.min(bounded_start);
        max_end = max_end.max(bounded_end);
        max_word_span_ms = max_word_span_ms.max(span_ms);

        if bounded_end <= bounded_start.saturating_add(1) {
            tiny_spans += 1;
        }
        if bounded_start >= tail_start {
            tail_heavy += 1;
        }
        if bounded_end <= prefix_window_end {
            prefix_clustered += 1;
        }
    }

    let span = max_end.saturating_sub(min_start);
    let len = alignments.len();
    len >= 8
        && (tiny_spans * 10 >= len * 8
            || tail_heavy * 10 >= len * 8
            || max_word_span_ms >= MAX_REASONABLE_WORD_SPAN_MS
            || (prefix_clustered * 10 >= len * 4 && tiny_spans * 10 >= len * 3)
            || span <= (duration_ms / 20).max(1))
}

fn attribution_requires_fallback(
    word_count: usize,
    overlap_assigned_words: usize,
    unattributed_words: usize,
) -> bool {
    if word_count == 0 {
        return false;
    }

    let overlap_coverage = overlap_assigned_words as f32 / word_count as f32;
    let unattributed_ratio = unattributed_words as f32 / word_count as f32;

    overlap_coverage < MIN_ALIGNMENT_COVERAGE || unattributed_ratio > MAX_UNATTRIBUTED_WORD_RATIO
}

fn attribute_words_to_speakers(
    alignments: &[(String, u32, u32)],
    segments: &[DiarizationSegment],
) -> (Vec<DiarizationWord>, usize, usize) {
    let mut words = Vec::new();
    let mut overlap_assigned_words = 0usize;
    let mut unattributed_words = 0usize;

    for (word, start_ms, end_ms) in alignments {
        let cleaned = word.trim();
        if cleaned.is_empty() {
            continue;
        }

        let start_secs = (*start_ms as f32 / 1000.0).max(0.0);
        let mut end_secs = (*end_ms as f32 / 1000.0).max(start_secs + 0.001);
        if !end_secs.is_finite() {
            end_secs = start_secs + 0.001;
        }

        let (speaker, speaker_confidence, overlaps_segment) =
            assign_speaker_for_span(start_secs, end_secs, segments);
        if overlaps_segment {
            overlap_assigned_words += 1;
        } else {
            unattributed_words += 1;
        }

        words.push(DiarizationWord {
            word: cleaned.to_string(),
            speaker,
            start_secs,
            end_secs,
            speaker_confidence,
            overlaps_segment,
        });
    }

    words.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
    (words, overlap_assigned_words, unattributed_words)
}

fn assign_speaker_for_span(
    start_secs: f32,
    end_secs: f32,
    segments: &[DiarizationSegment],
) -> (String, Option<f32>, bool) {
    if segments.is_empty() {
        return (UNKNOWN_SPEAKER.to_string(), None, false);
    }

    let word_span = (end_secs - start_secs).max(0.001);

    let mut best_overlap = 0.0f32;
    let mut best_overlap_ratio = 0.0f32;
    let mut best_specificity = 0.0f32;
    let mut best_confidence = f32::MIN;
    let mut best_segment_span = f32::MAX;
    let mut best_segment: Option<&DiarizationSegment> = None;
    for segment in segments {
        let overlap = interval_overlap(start_secs, end_secs, segment.start_secs, segment.end_secs);
        if overlap <= 0.0 {
            continue;
        }

        let segment_span = (segment.end_secs - segment.start_secs).max(0.001);
        let overlap_ratio = (overlap / word_span).clamp(0.0, 1.0);
        let specificity = (overlap / segment_span).clamp(0.0, 1.0);
        let confidence = segment.confidence.unwrap_or(0.0);

        let overlap_ratio_tied = approx_eq_f32(overlap_ratio, best_overlap_ratio);
        let specificity_tied = approx_eq_f32(specificity, best_specificity);
        let overlap_tied = approx_eq_f32(overlap, best_overlap);
        let confidence_tied = approx_eq_f32(confidence, best_confidence);
        let replace = best_segment.is_none()
            || overlap_ratio > best_overlap_ratio
            || (overlap_ratio_tied && specificity > best_specificity)
            || (overlap_ratio_tied && specificity_tied && overlap > best_overlap)
            || (overlap_ratio_tied
                && specificity_tied
                && overlap_tied
                && confidence > best_confidence)
            || (overlap_ratio_tied
                && specificity_tied
                && overlap_tied
                && confidence_tied
                && segment_span < best_segment_span);

        if replace {
            best_overlap = overlap;
            best_overlap_ratio = overlap_ratio;
            best_specificity = specificity;
            best_confidence = confidence;
            best_segment_span = segment_span;
            best_segment = Some(segment);
        }
    }

    if let Some(segment) = best_segment.filter(|_| best_overlap > 0.0) {
        return (segment.speaker.clone(), segment.confidence, true);
    }

    let midpoint = (start_secs + end_secs) * 0.5;
    let nearest = segments
        .iter()
        .min_by(|left, right| {
            span_distance(midpoint, left.start_secs, left.end_secs).total_cmp(&span_distance(
                midpoint,
                right.start_secs,
                right.end_secs,
            ))
        })
        .expect("segments checked non-empty");

    (nearest.speaker.clone(), nearest.confidence, false)
}

fn approx_eq_f32(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1e-6
}

fn span_distance(point: f32, start: f32, end: f32) -> f32 {
    if point < start {
        start - point
    } else if point > end {
        point - end
    } else {
        0.0
    }
}

fn interval_overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

fn build_utterances(words: &[DiarizationWord]) -> Vec<DiarizationUtterance> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut utterances = Vec::new();
    let mut current = DiarizationUtterance {
        speaker: words[0].speaker.clone(),
        start_secs: words[0].start_secs,
        end_secs: words[0].end_secs,
        text: words[0].word.clone(),
        word_start: 0,
        word_end: 0,
    };

    for (idx, word) in words.iter().enumerate().skip(1) {
        let gap = (word.start_secs - current.end_secs).max(0.0);
        let same_speaker = word.speaker == current.speaker;

        if same_speaker && gap <= MAX_UTTERANCE_GAP_SECS {
            append_token(&mut current.text, &word.word);
            current.end_secs = current.end_secs.max(word.end_secs);
            current.word_end = idx;
            continue;
        }

        utterances.push(current);
        current = DiarizationUtterance {
            speaker: word.speaker.clone(),
            start_secs: word.start_secs,
            end_secs: word.end_secs,
            text: word.word.clone(),
            word_start: idx,
            word_end: idx,
        };
    }

    utterances.push(current);
    merge_adjacent_same_speaker_utterances(utterances)
}

fn merge_adjacent_same_speaker_utterances(
    utterances: Vec<DiarizationUtterance>,
) -> Vec<DiarizationUtterance> {
    if utterances.len() <= 1 {
        return utterances;
    }

    let mut merged: Vec<DiarizationUtterance> = Vec::with_capacity(utterances.len());
    for utterance in utterances {
        if let Some(previous) = merged.last_mut() {
            if previous.speaker == utterance.speaker {
                let next_text = utterance.text.trim();
                if !next_text.is_empty() {
                    if previous.text.is_empty() {
                        previous.text.push_str(next_text);
                    } else {
                        previous.text.push(' ');
                        previous.text.push_str(next_text);
                    }
                }
                previous.end_secs = previous.end_secs.max(utterance.end_secs);
                previous.word_end = utterance.word_end;
                continue;
            }
        }
        merged.push(utterance);
    }
    merged
}

fn stabilize_word_speakers(words: &mut [DiarizationWord]) {
    if words.len() < 3 {
        return;
    }

    let mut changed = true;
    while changed {
        changed = false;
        let runs = collect_speaker_runs(words);
        if runs.len() < 3 {
            break;
        }

        for run_idx in 1..runs.len() {
            let run = runs[run_idx];
            let avg_confidence = average_run_confidence(words, run.start, run.end);
            let duration = (words[run.end].end_secs - words[run.start].start_secs).max(0.0);
            let word_count = run.end + 1 - run.start;

            let previous = run_idx.checked_sub(1).map(|idx| runs[idx]);
            let next = runs.get(run_idx + 1).copied();

            let target_speaker = if let (Some(left), Some(right)) = (previous, next) {
                if left.speaker(words) == right.speaker(words)
                    && left.speaker(words) != run.speaker(words)
                    && word_count <= MAX_FRAGMENT_WORDS
                    && duration <= MAX_FRAGMENT_DURATION_SECS
                {
                    let neighbor_confidence = average_run_confidence(words, left.start, right.end);
                    if avg_confidence + FRAGMENT_CONFIDENCE_MARGIN < neighbor_confidence {
                        Some(left.speaker(words).to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else if let Some(left) = previous {
                if word_count <= 2
                    && duration <= 1.0
                    && left.speaker(words) != run.speaker(words)
                    && !run_has_overlap_backed_words(words, run.start, run.end)
                    && avg_confidence + FRAGMENT_CONFIDENCE_MARGIN
                        < average_run_confidence(words, left.start, left.end)
                {
                    Some(left.speaker(words).to_string())
                } else {
                    None
                }
            } else if let Some(right) = next {
                if word_count <= 2
                    && duration <= 1.0
                    && right.speaker(words) != run.speaker(words)
                    && !run_has_overlap_backed_words(words, run.start, run.end)
                    && avg_confidence + FRAGMENT_CONFIDENCE_MARGIN
                        < average_run_confidence(words, right.start, right.end)
                {
                    Some(right.speaker(words).to_string())
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(target_speaker) = target_speaker {
                for word in &mut words[run.start..=run.end] {
                    word.speaker = target_speaker.clone();
                }
                changed = true;
                break;
            }
        }
    }
}

fn normalize_phrase_speakers(words: &mut [DiarizationWord], segments: &[DiarizationSegment]) {
    let phrases = collect_phrase_runs(words);
    if phrases.len() < 2 {
        return;
    }

    let mut phrase_speakers = phrases
        .iter()
        .map(|phrase| choose_phrase_speaker(words, *phrase))
        .collect::<Vec<_>>();
    let phrase_normalizable = phrases
        .iter()
        .map(|phrase| phrase_can_be_normalized(words, *phrase))
        .collect::<Vec<_>>();

    for phrase_idx in 1..phrases.len().saturating_sub(1) {
        if !phrase_normalizable[phrase_idx] {
            continue;
        }
        let previous = &phrase_speakers[phrase_idx - 1];
        let current = &phrase_speakers[phrase_idx];
        let next = &phrase_speakers[phrase_idx + 1];
        if previous != current || next != current {
            continue;
        }

        if let Some(alternate) =
            fully_covering_alternate_speaker(words, phrases[phrase_idx], segments, current)
        {
            phrase_speakers[phrase_idx] = alternate;
        }
    }

    for ((phrase, speaker), normalizable) in phrases
        .iter()
        .zip(phrase_speakers.iter())
        .zip(phrase_normalizable.iter())
    {
        if !*normalizable {
            continue;
        }
        for word in &mut words[phrase.start..=phrase.end] {
            word.speaker = speaker.clone();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SpeakerRun {
    start: usize,
    end: usize,
}

impl SpeakerRun {
    fn speaker<'a>(&self, words: &'a [DiarizationWord]) -> &'a str {
        &words[self.start].speaker
    }
}

fn collect_speaker_runs(words: &[DiarizationWord]) -> Vec<SpeakerRun> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let mut start = 0usize;
    for idx in 1..words.len() {
        if words[idx].speaker != words[start].speaker {
            runs.push(SpeakerRun {
                start,
                end: idx - 1,
            });
            start = idx;
        }
    }
    runs.push(SpeakerRun {
        start,
        end: words.len() - 1,
    });
    runs
}

fn collect_phrase_runs(words: &[DiarizationWord]) -> Vec<SpeakerRun> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut phrases = Vec::new();
    let mut start = 0usize;
    for idx in 1..words.len() {
        let previous = &words[idx - 1];
        let current = &words[idx];
        let gap = (current.start_secs - previous.end_secs).max(0.0);
        let starts_capitalized = current
            .word
            .chars()
            .next()
            .map(|ch| ch.is_ascii_uppercase())
            .unwrap_or(false);

        if gap >= PHRASE_HARD_GAP_SECS || (gap >= PHRASE_CAPITALIZED_GAP_SECS && starts_capitalized)
        {
            phrases.push(SpeakerRun {
                start,
                end: idx - 1,
            });
            start = idx;
        }
    }
    phrases.push(SpeakerRun {
        start,
        end: words.len() - 1,
    });
    phrases
}

fn choose_phrase_speaker(words: &[DiarizationWord], phrase: SpeakerRun) -> String {
    let phrase_words = &words[phrase.start..=phrase.end];
    let runs = collect_speaker_runs(phrase_words);

    if runs.len() == 1 {
        return phrase_words[runs[0].start].speaker.clone();
    }

    #[derive(Debug, Clone, Copy)]
    struct PhraseSpeakerStats {
        total_duration: f32,
        confidence_weighted_duration: f32,
        total_words: usize,
        first_run_idx: usize,
    }

    let mut stats = HashMap::<&str, PhraseSpeakerStats>::new();
    for (run_idx, run) in runs.iter().copied().enumerate() {
        let duration = run_duration_secs(phrase_words, run);
        let avg_confidence = average_run_confidence(phrase_words, run.start, run.end);
        let word_count = run.end + 1 - run.start;
        stats
            .entry(run.speaker(phrase_words))
            .and_modify(|entry| {
                entry.total_duration += duration;
                entry.confidence_weighted_duration += duration * avg_confidence;
                entry.total_words += word_count;
            })
            .or_insert(PhraseSpeakerStats {
                total_duration: duration,
                confidence_weighted_duration: duration * avg_confidence,
                total_words: word_count,
                first_run_idx: run_idx,
            });
    }

    stats
        .into_iter()
        .max_by(|left, right| {
            left.1
                .total_duration
                .total_cmp(&right.1.total_duration)
                .then(
                    left.1
                        .confidence_weighted_duration
                        .total_cmp(&right.1.confidence_weighted_duration),
                )
                .then(left.1.total_words.cmp(&right.1.total_words))
                .then(right.1.first_run_idx.cmp(&left.1.first_run_idx))
        })
        .map(|(speaker, _)| speaker.to_string())
        .unwrap_or_else(|| phrase_words[runs[0].start].speaker.clone())
}

fn phrase_can_be_normalized(words: &[DiarizationWord], phrase: SpeakerRun) -> bool {
    let phrase_words = &words[phrase.start..=phrase.end];
    let mut distinct_speakers = HashMap::<&str, usize>::new();
    for word in phrase_words {
        *distinct_speakers.entry(word.speaker.as_str()).or_insert(0) += 1;
        if distinct_speakers.len() > 2 {
            return false;
        }
    }

    let runs = collect_speaker_runs(phrase_words);
    if runs.len() <= 1 {
        return true;
    }

    // A simple two-run phrase is usually a real turn boundary, not a fragment
    // that should be collapsed into one speaker.
    if runs.len() == 2 {
        return false;
    }

    if runs[0].speaker(phrase_words) != runs[runs.len() - 1].speaker(phrase_words) {
        return false;
    }

    runs.iter()
        .copied()
        .skip(1)
        .take(runs.len().saturating_sub(2))
        .all(|run| {
            let word_count = run.end + 1 - run.start;
            let duration = run_duration_secs(phrase_words, run);
            word_count <= MAX_FRAGMENT_WORDS && duration <= MAX_FRAGMENT_DURATION_SECS
        })
}

fn run_duration_secs(words: &[DiarizationWord], run: SpeakerRun) -> f32 {
    (words[run.end].end_secs - words[run.start].start_secs).max(0.0)
}

fn fully_covering_alternate_speaker(
    words: &[DiarizationWord],
    phrase: SpeakerRun,
    segments: &[DiarizationSegment],
    current_speaker: &str,
) -> Option<String> {
    let phrase_start = words[phrase.start].start_secs;
    let phrase_end = words[phrase.end].end_secs;
    let duration = (phrase_end - phrase_start).max(0.001);
    if duration <= 0.0 {
        return None;
    }

    let mut coverage_by_speaker = HashMap::<&str, f32>::new();
    for segment in segments {
        let overlap = interval_overlap(
            phrase_start,
            phrase_end,
            segment.start_secs,
            segment.end_secs,
        );
        if overlap > 0.0 {
            *coverage_by_speaker
                .entry(segment.speaker.as_str())
                .or_insert(0.0) += overlap;
        }
    }

    coverage_by_speaker
        .into_iter()
        .filter(|(speaker, _)| *speaker != current_speaker)
        .filter(|(_, coverage)| *coverage >= duration * 0.9)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(speaker, _)| speaker.to_string())
}

fn average_run_confidence(words: &[DiarizationWord], start: usize, end: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for word in &words[start..=end] {
        if let Some(confidence) = word.speaker_confidence {
            if confidence.is_finite() {
                sum += confidence.clamp(0.0, 1.0);
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn run_has_overlap_backed_words(words: &[DiarizationWord], start: usize, end: usize) -> bool {
    words[start..=end].iter().any(|word| word.overlaps_segment)
}

fn append_token(target: &mut String, token: &str) {
    if target.is_empty() {
        target.push_str(token);
        return;
    }

    let punct_only = token
        .chars()
        .all(|ch| !ch.is_alphanumeric() && ch != '\'' && ch != '-');
    if punct_only {
        target.push_str(token);
    } else {
        target.push(' ');
        target.push_str(token);
    }
}

fn format_utterance_transcript(utterances: &[DiarizationUtterance]) -> String {
    utterances
        .iter()
        .map(|utterance| {
            format!(
                "{} [{:.2}s - {:.2}s]: {}",
                utterance.speaker, utterance.start_secs, utterance.end_secs, utterance.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn sanitize_refined_transcript(candidate: &str, fallback: &str) -> String {
    let fallback_trimmed = fallback.trim();
    let stripped = strip_tagged_sections(candidate, "<think>", "</think>")
        .replace("```text", "")
        .replace("```", "");

    let candidate_lines = stripped
        .lines()
        .filter_map(extract_utterance_line)
        .collect::<Vec<_>>();
    let fallback_lines = fallback_trimmed
        .lines()
        .filter_map(extract_utterance_line)
        .collect::<Vec<_>>();

    if !candidate_lines.is_empty() {
        if fallback_lines.is_empty() {
            return candidate_lines.join("\n");
        }
        // Accept refined output only when it preserves line count and line headers.
        let structurally_consistent = candidate_lines.len() == fallback_lines.len()
            && candidate_lines.iter().zip(fallback_lines.iter()).all(
                |(candidate_line, fallback_line)| {
                    utterance_prefix(candidate_line) == utterance_prefix(fallback_line)
                        && utterance_text_similarity_ok(candidate_line, fallback_line)
                },
            );
        if structurally_consistent {
            return candidate_lines.join("\n");
        }
    }

    if !fallback_trimmed.is_empty() {
        return fallback_trimmed.to_string();
    }

    stripped.trim().to_string()
}

fn utterance_prefix(line: &str) -> Option<&str> {
    let mut trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(stripped) = trimmed.strip_prefix("- ") {
        trimmed = stripped.trim();
    } else if let Some(stripped) = trimmed.strip_prefix("* ") {
        trimmed = stripped.trim();
    }
    let header_end = trimmed.find(':')?;
    Some(trimmed[..header_end].trim())
}

fn utterance_text_similarity_ok(candidate_line: &str, fallback_line: &str) -> bool {
    let Some((_, candidate_text)) = candidate_line.split_once(':') else {
        return false;
    };
    let Some((_, fallback_text)) = fallback_line.split_once(':') else {
        return false;
    };

    let candidate_words = extract_words(candidate_text)
        .into_iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let fallback_words = extract_words(fallback_text)
        .into_iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<Vec<_>>();

    if fallback_words.is_empty() {
        return true;
    }
    if candidate_words.is_empty() {
        return false;
    }

    let (recall, precision) = bag_word_overlap(&fallback_words, &candidate_words);
    recall >= 0.75 && precision >= 0.6
}

fn bag_word_overlap(reference_words: &[String], candidate_words: &[String]) -> (f32, f32) {
    if reference_words.is_empty() || candidate_words.is_empty() {
        return (0.0, 0.0);
    }

    let mut counts = HashMap::<&str, usize>::new();
    for word in reference_words {
        *counts.entry(word.as_str()).or_insert(0) += 1;
    }

    let mut common = 0usize;
    for word in candidate_words {
        if let Some(remaining) = counts.get_mut(word.as_str()) {
            if *remaining > 0 {
                *remaining -= 1;
                common += 1;
            }
        }
    }

    let recall = common as f32 / reference_words.len() as f32;
    let precision = common as f32 / candidate_words.len() as f32;
    (recall, precision)
}

fn append_chunk_alignments(
    merged: &mut Vec<(String, u32, u32)>,
    chunk_alignments: Vec<(String, u32, u32)>,
) {
    if merged.is_empty() {
        merged.extend(chunk_alignments);
        return;
    }

    let overlap = word_overlap_prefix_len(merged, &chunk_alignments, 24);
    let skip = if overlap > 0 {
        overlap
    } else {
        trim_timing_overlap_prefix_len(merged, &chunk_alignments)
    };

    merged.extend(chunk_alignments.into_iter().skip(skip));
}

fn word_overlap_prefix_len(
    merged: &[(String, u32, u32)],
    incoming: &[(String, u32, u32)],
    max_words: usize,
) -> usize {
    let max_overlap = merged.len().min(incoming.len()).min(max_words);
    for overlap in (1..=max_overlap).rev() {
        let left = &merged[merged.len() - overlap..];
        let right = &incoming[..overlap];
        let all_match = left
            .iter()
            .zip(right.iter())
            .all(|((lw, _, _), (rw, _, _))| lw.eq_ignore_ascii_case(rw));
        if all_match {
            return overlap;
        }
    }
    0
}

fn trim_timing_overlap_prefix_len(
    merged: &[(String, u32, u32)],
    incoming: &[(String, u32, u32)],
) -> usize {
    let Some((_, _, last_end)) = merged.last() else {
        return 0;
    };
    incoming
        .iter()
        .take_while(|(_, start, end)| *end <= *last_end || *start < *last_end)
        .count()
}

fn samples_to_ms(sample_index: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_index as u64 * 1000) / sample_rate as u64) as u32
}

fn strip_tagged_sections(input: &str, start_tag: &str, end_tag: &str) -> String {
    let mut output = input.to_string();
    let start_tag = start_tag.to_ascii_lowercase();
    let end_tag = end_tag.to_ascii_lowercase();
    let start_len = start_tag.len();
    let end_len = end_tag.len();

    loop {
        let lowered = output.to_ascii_lowercase();
        let Some(start_idx) = lowered.find(&start_tag) else {
            break;
        };
        let search_from = start_idx.saturating_add(start_len);
        if let Some(end_rel) = lowered[search_from..].find(&end_tag) {
            let end_idx = search_from + end_rel + end_len;
            output.replace_range(start_idx..end_idx, "");
        } else {
            output.replace_range(start_idx..output.len(), "");
            break;
        }
    }

    output
}

fn extract_utterance_line(line: &str) -> Option<String> {
    let mut candidate = line.trim();
    if candidate.is_empty() {
        return None;
    }

    if let Some(stripped) = candidate.strip_prefix("- ") {
        candidate = stripped.trim();
    } else if let Some(stripped) = candidate.strip_prefix("* ") {
        candidate = stripped.trim();
    } else {
        let numeric_end = candidate
            .bytes()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if numeric_end > 0 && candidate[numeric_end..].starts_with(". ") {
            candidate = candidate[numeric_end + 2..].trim();
        }
    }

    if is_utterance_line(candidate) {
        Some(candidate.to_string())
    } else {
        None
    }
}

fn is_utterance_line(line: &str) -> bool {
    let trimmed = line.trim();
    let Some(header_end) = trimmed.find("]:") else {
        return false;
    };
    let header = &trimmed[..=header_end];
    let Some(bracket_start) = header.rfind('[') else {
        return false;
    };
    if header[..bracket_start].trim().is_empty() {
        return false;
    }
    let time_range = &header[bracket_start + 1..header.len() - 1];
    let Some((start, end)) = time_range.split_once(" - ") else {
        return false;
    };
    is_seconds_token(start) && is_seconds_token(end)
}

fn is_seconds_token(token: &str) -> bool {
    let Some(value) = token.trim().strip_suffix('s') else {
        return false;
    };
    value
        .parse::<f32>()
        .map(|parsed| parsed.is_finite() && parsed >= 0.0)
        .unwrap_or(false)
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if audio.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0.0f32; out_len];

    for (idx, sample) in out.iter_mut().enumerate() {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(audio.len().saturating_sub(1));
        let frac = (src_pos - left as f64) as f32;
        *sample = audio[left] * (1.0 - frac) + audio[right] * frac;
    }

    out
}

fn secs_to_ms(value: f32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else {
        (value * 1000.0).round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_word_timings_generates_monotonic_ranges() {
        let timings = fallback_word_timings("hello world from test", 2.0);
        assert_eq!(timings.len(), 4);
        for (idx, (_, start, end)) in timings.iter().enumerate() {
            assert!(end > start, "word {} should have positive duration", idx);
            if idx > 0 {
                assert!(*start >= timings[idx - 1].1);
            }
        }
    }

    #[test]
    fn append_chunk_alignments_dedupes_text_overlap() {
        let mut merged = vec![
            ("hello".to_string(), 0, 100),
            ("world".to_string(), 100, 200),
        ];
        let incoming = vec![
            ("world".to_string(), 180, 260),
            ("again".to_string(), 260, 340),
        ];

        append_chunk_alignments(&mut merged, incoming);

        assert_eq!(
            merged,
            vec![
                ("hello".to_string(), 0, 100),
                ("world".to_string(), 100, 200),
                ("again".to_string(), 260, 340),
            ]
        );
    }

    #[test]
    fn append_chunk_alignments_trims_timing_overlap_without_text_match() {
        let mut merged = vec![
            ("hello".to_string(), 0, 100),
            ("world".to_string(), 100, 220),
        ];
        let incoming = vec![
            ("there".to_string(), 180, 210),
            ("friend".to_string(), 221, 320),
        ];

        append_chunk_alignments(&mut merged, incoming);

        assert_eq!(
            merged,
            vec![
                ("hello".to_string(), 0, 100),
                ("world".to_string(), 100, 220),
                ("friend".to_string(), 221, 320),
            ]
        );
    }

    #[test]
    fn attribution_prefers_overlap_then_nearest() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 3.0,
                confidence: Some(0.8),
            },
        ];

        let aligned = vec![
            ("hello".to_string(), 100, 400),
            ("there".to_string(), 1200, 1300),
            ("friend".to_string(), 2400, 2800),
        ];

        let (words, overlap_count, unattributed) = attribute_words_to_speakers(&aligned, &segments);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].speaker, "SPEAKER_00");
        assert_eq!(words[2].speaker, "SPEAKER_01");
        assert_eq!(overlap_count, 2);
        assert_eq!(unattributed, 1);
    }

    #[test]
    fn attribution_prefers_more_specific_segment_when_overlaps_tie() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 10.0,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.2),
            },
        ];

        let aligned = vec![("hello".to_string(), 200, 800)];
        let (words, overlap_count, unattributed) = attribute_words_to_speakers(&aligned, &segments);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].speaker, "SPEAKER_01");
        assert_eq!(overlap_count, 1);
        assert_eq!(unattributed, 0);
    }

    #[test]
    fn build_utterances_merges_small_gaps_for_same_speaker() {
        let words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "world".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.5,
                end_secs: 0.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "next".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 1.2,
                end_secs: 1.6,
                speaker_confidence: None,
                overlaps_segment: true,
            },
        ];

        let utterances = build_utterances(&words);
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert!(utterances[0].text.contains("hello world"));
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn build_utterances_merges_consecutive_same_speaker_runs_after_long_pauses() {
        let words = vec![
            DiarizationWord {
                word: "first".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "block".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.4,
                end_secs: 0.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "second".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.0,
                end_secs: 2.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "block".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.4,
                end_secs: 2.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "other".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 3.2,
                end_secs: 3.6,
                speaker_confidence: None,
                overlaps_segment: true,
            },
        ];

        let utterances = build_utterances(&words);
        assert_eq!(utterances.len(), 2, "{utterances:#?}");
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert_eq!(utterances[0].start_secs, 0.0);
        assert_eq!(utterances[0].end_secs, 2.8);
        assert_eq!(utterances[0].word_start, 0);
        assert_eq!(utterances[0].word_end, 3);
        assert_eq!(utterances[0].text, "first block second block");
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn stabilize_word_speakers_absorbs_low_confidence_inner_fragment() {
        let mut words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: Some(0.82),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "there".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.5,
                end_secs: 0.7,
                speaker_confidence: Some(0.22),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "friend".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.8,
                end_secs: 1.2,
                speaker_confidence: Some(0.81),
                overlaps_segment: true,
            },
        ];

        stabilize_word_speakers(&mut words);

        assert!(words.iter().all(|word| word.speaker == "SPEAKER_00"));
    }

    #[test]
    fn stabilize_word_speakers_preserves_overlap_backed_edge_turn() {
        let mut words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: Some(0.82),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "world".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.4,
                end_secs: 0.8,
                speaker_confidence: Some(0.82),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "yeah".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.9,
                end_secs: 1.2,
                speaker_confidence: Some(0.22),
                overlaps_segment: true,
            },
        ];

        stabilize_word_speakers(&mut words);

        assert_eq!(words[2].speaker, "SPEAKER_01");
    }

    #[test]
    fn choose_phrase_speaker_prefers_total_phrase_support() {
        let words = vec![
            DiarizationWord {
                word: "So".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.0,
                end_secs: 0.3,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "Aaron".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.3,
                end_secs: 0.6,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "in".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.6,
                end_secs: 0.9,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "your".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.9,
                end_secs: 1.2,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "email".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.2,
                end_secs: 2.1,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "you".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.1,
                end_secs: 2.8,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
        ];

        let phrase = SpeakerRun {
            start: 0,
            end: words.len() - 1,
        };
        let chosen = choose_phrase_speaker(&words, phrase);

        assert_eq!(chosen, "SPEAKER_00");
    }

    #[test]
    fn normalize_phrase_speakers_preserves_two_run_turn_boundary() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.0,
                end_secs: 1.44,
                confidence: Some(0.28),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.52,
                end_secs: 4.08,
                confidence: Some(0.78),
            },
        ];
        let mut words = vec![
            test_word("So", "SPEAKER_01", 0.32, 0.64, 0.28),
            test_word("Aaron", "SPEAKER_01", 0.64, 1.04, 0.28),
            test_word("in", "SPEAKER_01", 1.20, 1.28, 0.28),
            test_word("your", "SPEAKER_01", 1.28, 1.44, 0.28),
            test_word("email", "SPEAKER_00", 1.52, 1.92, 0.78),
            test_word("you", "SPEAKER_00", 1.92, 2.08, 0.78),
            test_word("said", "SPEAKER_00", 2.08, 2.32, 0.78),
            test_word("wanted", "SPEAKER_00", 2.40, 2.72, 0.78),
        ];

        normalize_phrase_speakers(&mut words, &segments);
        let utterances = build_utterances(&words);

        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].speaker, "SPEAKER_01");
        assert_eq!(utterances[0].text, "So Aaron in your");
        assert_eq!(utterances[1].speaker, "SPEAKER_00");
        assert!(utterances[1].text.starts_with("email you said"));
    }

    #[test]
    fn normalize_phrase_speakers_can_flip_ambiguous_middle_phrase() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 6.0,
                confidence: Some(0.8),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 4.0,
                confidence: Some(0.25),
            },
        ];
        let mut words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "How".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.0,
                end_secs: 2.6,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "review".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.6,
                end_secs: 4.0,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "Yeah".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 4.5,
                end_secs: 5.4,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
        ];

        normalize_phrase_speakers(&mut words, &segments);

        assert_eq!(words[1].speaker, "SPEAKER_01");
        assert_eq!(words[2].speaker, "SPEAKER_01");
    }

    #[test]
    fn normalize_phrase_speakers_preserves_three_speaker_phrase() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.62,
                confidence: Some(0.92),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.62,
                end_secs: 1.16,
                confidence: Some(0.89),
            },
            DiarizationSegment {
                speaker: "SPEAKER_02".to_string(),
                start_secs: 1.16,
                end_secs: 1.72,
                confidence: Some(0.91),
            },
        ];
        let mut words = vec![
            test_word("Alice", "SPEAKER_00", 0.00, 0.28, 0.92),
            test_word("opens", "SPEAKER_00", 0.28, 0.58, 0.92),
            test_word("Bob", "SPEAKER_01", 0.62, 0.86, 0.89),
            test_word("answers", "SPEAKER_01", 0.86, 1.12, 0.89),
            test_word("Carol", "SPEAKER_02", 1.16, 1.40, 0.91),
            test_word("wraps", "SPEAKER_02", 1.40, 1.70, 0.91),
            test_word("Later", "SPEAKER_00", 2.20, 2.52, 0.92),
            test_word("done", "SPEAKER_00", 2.52, 2.82, 0.92),
        ];

        normalize_phrase_speakers(&mut words, &segments);
        let utterances = build_utterances(&words);

        assert_eq!(words[0].speaker, "SPEAKER_00");
        assert_eq!(words[2].speaker, "SPEAKER_01");
        assert_eq!(words[4].speaker, "SPEAKER_02");
        assert_eq!(utterances.len(), 4);
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
        assert_eq!(utterances[2].speaker, "SPEAKER_02");
    }

    #[test]
    fn phrase_normalization_matches_expected_turns_for_overlap_heavy_sample() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.32,
                end_secs: 4.0,
                confidence: Some(0.78),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.32,
                end_secs: 1.6,
                confidence: Some(0.28),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 4.88,
                end_secs: 12.32,
                confidence: Some(0.78),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 4.88,
                end_secs: 5.36,
                confidence: Some(0.25),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 11.76,
                end_secs: 12.32,
                confidence: Some(0.25),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 13.04,
                end_secs: 20.72,
                confidence: Some(0.77),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 13.04,
                end_secs: 20.72,
                confidence: Some(0.27),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 21.2,
                end_secs: 27.12,
                confidence: Some(0.78),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 21.2,
                end_secs: 21.6,
                confidence: Some(0.26),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 23.2,
                end_secs: 23.84,
                confidence: Some(0.26),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 26.08,
                end_secs: 27.12,
                confidence: Some(0.27),
            },
        ];
        let mut words = vec![
            test_word("So", "SPEAKER_01", 0.32, 0.64, 0.28),
            test_word("Aaron", "SPEAKER_01", 0.64, 1.04, 0.28),
            test_word("in", "SPEAKER_01", 1.20, 1.28, 0.28),
            test_word("your", "SPEAKER_01", 1.28, 1.44, 0.28),
            test_word("email", "SPEAKER_00", 1.52, 1.92, 0.78),
            test_word("you", "SPEAKER_00", 1.92, 2.08, 0.78),
            test_word("said", "SPEAKER_00", 2.08, 2.32, 0.78),
            test_word("you", "SPEAKER_00", 2.32, 2.40, 0.78),
            test_word("wanted", "SPEAKER_00", 2.40, 2.72, 0.78),
            test_word("to", "SPEAKER_00", 2.72, 2.80, 0.78),
            test_word("talk", "SPEAKER_00", 2.80, 3.12, 0.78),
            test_word("about", "SPEAKER_00", 3.12, 3.36, 0.78),
            test_word("the", "SPEAKER_00", 3.36, 3.52, 0.78),
            test_word("exam", "SPEAKER_00", 3.52, 4.08, 0.78),
            test_word("Yeah", "SPEAKER_01", 4.88, 5.28, 0.25),
            test_word("um", "SPEAKER_00", 5.44, 5.92, 0.78),
            test_word("I've", "SPEAKER_00", 6.24, 6.40, 0.79),
            test_word("just", "SPEAKER_00", 6.40, 6.72, 0.79),
            test_word("never", "SPEAKER_00", 6.72, 7.04, 0.79),
            test_word("taken", "SPEAKER_00", 7.04, 7.52, 0.79),
            test_word("a", "SPEAKER_00", 7.52, 7.60, 0.79),
            test_word("class", "SPEAKER_00", 7.60, 8.00, 0.79),
            test_word("with", "SPEAKER_00", 8.00, 8.16, 0.79),
            test_word("so", "SPEAKER_00", 8.16, 8.40, 0.79),
            test_word("many", "SPEAKER_00", 8.40, 8.72, 0.79),
            test_word("different", "SPEAKER_00", 8.72, 8.96, 0.79),
            test_word("readings", "SPEAKER_00", 9.04, 9.68, 0.79),
            test_word("I've", "SPEAKER_01", 10.16, 10.32, 0.26),
            test_word("managed", "SPEAKER_00", 10.32, 10.80, 0.78),
            test_word("to", "SPEAKER_00", 10.80, 10.88, 0.78),
            test_word("keep", "SPEAKER_00", 10.88, 11.12, 0.78),
            test_word("up", "SPEAKER_00", 11.12, 11.36, 0.78),
            test_word("with", "SPEAKER_00", 11.36, 11.44, 0.78),
            test_word("all", "SPEAKER_00", 11.52, 11.60, 0.78),
            test_word("the", "SPEAKER_00", 11.60, 11.76, 0.78),
            test_word("assignments", "SPEAKER_01", 11.76, 12.48, 0.25),
            test_word("but", "SPEAKER_00", 12.96, 13.12, 0.77),
            test_word("I'm", "SPEAKER_00", 13.12, 13.28, 0.77),
            test_word("not", "SPEAKER_00", 13.28, 13.44, 0.77),
            test_word("sure", "SPEAKER_00", 13.44, 13.76, 0.77),
            test_word("how", "SPEAKER_00", 13.76, 13.92, 0.77),
            test_word("to", "SPEAKER_00", 14.00, 14.24, 0.77),
            test_word("how", "SPEAKER_00", 14.96, 15.20, 0.76),
            test_word("to", "SPEAKER_00", 15.20, 15.52, 0.76),
            test_word("How", "SPEAKER_00", 16.16, 16.56, 0.76),
            test_word("to", "SPEAKER_00", 16.72, 16.72, 0.76),
            test_word("review", "SPEAKER_00", 16.88, 17.44, 0.76),
            test_word("everything", "SPEAKER_00", 17.44, 18.08, 0.76),
            test_word("Yeah", "SPEAKER_00", 18.48, 18.96, 0.76),
            test_word("in", "SPEAKER_00", 19.36, 19.52, 0.77),
            test_word("other", "SPEAKER_00", 19.60, 19.84, 0.77),
            test_word("classes", "SPEAKER_00", 19.84, 20.32, 0.77),
            test_word("I've", "SPEAKER_00", 20.32, 20.48, 0.77),
            test_word("had", "SPEAKER_00", 20.48, 20.80, 0.77),
            test_word("there's", "SPEAKER_01", 21.20, 21.44, 0.26),
            test_word("usually", "SPEAKER_00", 21.44, 21.92, 0.78),
            test_word("just", "SPEAKER_00", 21.92, 22.16, 0.78),
            test_word("one", "SPEAKER_00", 22.16, 22.48, 0.78),
            test_word("book", "SPEAKER_00", 22.48, 22.72, 0.78),
            test_word("to", "SPEAKER_00", 22.80, 22.88, 0.78),
            test_word("review", "SPEAKER_00", 22.88, 23.20, 0.78),
            test_word("not", "SPEAKER_01", 23.20, 23.36, 0.26),
            test_word("three", "SPEAKER_01", 23.60, 23.84, 0.26),
            test_word("different", "SPEAKER_00", 23.84, 24.16, 0.78),
            test_word("books", "SPEAKER_00", 24.16, 24.48, 0.78),
            test_word("plus", "SPEAKER_00", 24.48, 24.72, 0.78),
            test_word("all", "SPEAKER_00", 24.80, 24.96, 0.78),
            test_word("those", "SPEAKER_00", 24.96, 25.12, 0.78),
            test_word("other", "SPEAKER_00", 25.12, 25.44, 0.78),
            test_word("text", "SPEAKER_00", 25.44, 25.84, 0.78),
            test_word("excerpts", "SPEAKER_00", 25.84, 26.40, 0.78),
            test_word("and", "SPEAKER_01", 26.40, 26.56, 0.27),
            test_word("videos", "SPEAKER_01", 26.56, 27.28, 0.27),
        ];

        stabilize_word_speakers(&mut words);
        normalize_phrase_speakers(&mut words, &segments);
        let utterances = build_utterances(&words);

        assert!(utterances.len() >= 4, "{utterances:#?}");
        assert_eq!(utterances[0].speaker, "SPEAKER_01");
        assert_eq!(utterances[0].text, "So Aaron in your");
        assert_eq!(utterances[1].speaker, "SPEAKER_00");
        assert!(utterances[1].text.starts_with("email you said"));
        assert!(utterances
            .iter()
            .any(|utterance| utterance.speaker == "SPEAKER_01"
                && utterance.text == "How to review everything"));
        assert_eq!(utterances.last().unwrap().speaker, "SPEAKER_01");
        assert!(utterances.last().unwrap().text.ends_with("and videos"));
    }

    fn test_word(
        word: &str,
        speaker: &str,
        start_secs: f32,
        end_secs: f32,
        speaker_confidence: f32,
    ) -> DiarizationWord {
        DiarizationWord {
            word: word.to_string(),
            speaker: speaker.to_string(),
            start_secs,
            end_secs,
            speaker_confidence: Some(speaker_confidence),
            overlaps_segment: true,
        }
    }

    #[test]
    fn sanitize_refined_transcript_removes_thinking_and_keeps_lines() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there";
        let candidate = r#"
<think>
internal reasoning
</think>
Here is the refined transcript:
- SPEAKER_00 [0.00s - 1.00s]: Hello there.
"#;

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, "SPEAKER_00 [0.00s - 1.00s]: Hello there.");
    }

    #[test]
    fn sanitize_refined_transcript_falls_back_when_no_utterance_lines() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there";
        let candidate = "Here is the rewrite with cleaner punctuation.";

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, fallback);
    }

    #[test]
    fn sanitize_refined_transcript_falls_back_on_line_count_mismatch() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there";
        let candidate =
            "SPEAKER_00 [0.00s - 1.00s]: Hello there.\nSPEAKER_01 [1.00s - 2.00s]: Extra line";

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, fallback);
    }

    #[test]
    fn sanitize_refined_transcript_falls_back_on_low_similarity() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there from class";
        let candidate =
            "SPEAKER_00 [0.00s - 1.00s]: completely new invented content that was never spoken";

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, fallback);
    }

    #[test]
    fn fallback_word_timings_with_segments_places_words_in_windows() {
        let words = vec![
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
        ];
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: None,
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 3.0,
                confidence: None,
            },
        ];

        let alignments = fallback_word_timings_with_segments(&words, &segments, 3.0);
        assert_eq!(alignments.len(), 4);
        assert!(alignments[0].1 < 1_000);
        assert!(alignments[1].1 < 1_000);
        assert!(alignments[2].1 >= 2_000);
        assert!(alignments[3].1 >= 2_000);
    }

    #[test]
    fn alignment_is_suspicious_detects_tail_collapse() {
        let alignments = (0..20)
            .map(|idx| (format!("w{idx}"), 27_302u32, 27_303u32))
            .collect::<Vec<_>>();
        assert!(alignment_is_suspicious(&alignments, 20, 27.303));
    }

    #[test]
    fn alignment_is_suspicious_detects_front_loaded_micro_spans_and_bridge_word() {
        let mut alignments = (0..42)
            .map(|idx| (format!("w{idx}"), idx as u32, idx as u32 + 1))
            .collect::<Vec<_>>();
        alignments.push(("bridge".to_string(), 41, 13_600));
        alignments.extend((43..73).map(|idx| {
            let start = 18_080 + ((idx - 43) as u32 * 320);
            (format!("w{idx}"), start, start + 240)
        }));

        assert!(alignment_is_suspicious(&alignments, 73, 27.303));
    }

    #[test]
    fn attribution_requires_fallback_when_too_many_words_are_unattributed() {
        assert!(attribution_requires_fallback(73, 31, 42));
        assert!(!attribution_requires_fallback(73, 60, 13));
    }

    #[test]
    fn should_use_single_pass_diarization_asr_prefers_stable_path_when_possible() {
        assert!(should_use_single_pass_diarization_asr(
            27.3,
            Some(300.0),
            true
        ));
        assert!(should_use_single_pass_diarization_asr(27.3, None, true));
        assert!(should_use_single_pass_diarization_asr(
            600.0,
            Some(300.0),
            false
        ));
        assert!(!should_use_single_pass_diarization_asr(
            600.0,
            Some(300.0),
            true
        ));
    }
}
