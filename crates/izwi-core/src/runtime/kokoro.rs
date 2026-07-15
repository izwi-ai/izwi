//! Kokoro TTS runtime helpers (isolated from generic runtime routing).

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::kokoro::{
    kokoro_output_budget, KokoroSynthesisResult, KokoroTtsModel,
};
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::coordinator::{JobLease, JobResourceObservation};
use crate::runtime::service::RuntimeService;
use crate::runtime::tts::direct_tts_retained_input_bytes;
use crate::runtime::types::{AudioChunk, GenerationRequest, GenerationResult};

const KOKORO_STREAM_TARGET_CHARS: usize = 180;
const KOKORO_STREAM_MIN_CHARS: usize = 64;

impl RuntimeService {
    fn default_kokoro_variant() -> ModelVariant {
        ModelVariant::Kokoro82M
    }

    async fn resolve_kokoro_variant_for_request(
        &self,
        request: &GenerationRequest,
    ) -> ModelVariant {
        if let Some(variant) = request.model_variant {
            if matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts) {
                return variant;
            }
        }
        if let Some(variant) = *self.loaded_tts_variant.read().await {
            if matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts) {
                return variant;
            }
        }
        Self::default_kokoro_variant()
    }

    pub(crate) async fn kokoro_tts_generate(
        &self,
        job: &JobLease,
        request: GenerationRequest,
    ) -> Result<GenerationResult> {
        let retained_input_bytes = u64::try_from(direct_tts_retained_input_bytes(&request)?)
            .map_err(|_| Error::InvalidInput("Kokoro retained input exceeds u64".to_string()))?;
        let output_budget =
            kokoro_output_budget(&request.text, request.config.options.speed)?.max_samples;
        let variant = self.resolve_kokoro_variant_for_request(&request).await;
        self.observe_broker_capability_request(CapabilityKind::Tts, Some(variant), false)?;
        let residency_lease = self.load_model_for_job(job, variant).await?;
        let model = self
            .model_registry
            .get_kokoro(variant)
            .await
            .ok_or_else(|| Error::InferenceError("Kokoro model not loaded".to_string()))?;
        let observation_job = job.clone();

        self.coordinator
            .run_blocking_stage(job, move || {
                let _residency_lease = residency_lease;
                let opts = &request.config.options;
                let speaker = opts.speaker.as_deref().or(opts.voice.as_deref());
                let started = Instant::now();
                let result = synthesize_kokoro_with_fallback(
                    model,
                    &request.text,
                    speaker,
                    request.language.as_deref(),
                    opts.speed,
                    output_budget,
                )?;
                record_kokoro_materialized_output(
                    &observation_job,
                    retained_input_bytes,
                    result.samples.capacity(),
                    output_budget,
                )?;
                let total_time_ms = started.elapsed().as_secs_f32() * 1000.0;

                Ok(GenerationResult {
                    request_id: request.id,
                    samples: result.samples,
                    sample_rate: result.sample_rate,
                    total_tokens: result.tokens_generated,
                    total_time_ms,
                    diagnostics: None,
                })
            })
            .await
    }

    pub(crate) async fn kokoro_tts_generate_streaming(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let retained_input_bytes = u64::try_from(direct_tts_retained_input_bytes(&request)?)
            .map_err(|_| Error::InvalidInput("Kokoro retained input exceeds u64".to_string()))?;
        let output_budget = kokoro_output_budget(&request.text, request.config.options.speed)?;
        let request_id = request.id.clone();
        let variant = self.resolve_kokoro_variant_for_request(&request).await;
        self.observe_broker_capability_request(CapabilityKind::Tts, Some(variant), true)?;
        let residency_lease = self.load_model_for_job(job, variant).await?;
        let model = self
            .model_registry
            .get_kokoro(variant)
            .await
            .ok_or_else(|| Error::InferenceError("Kokoro model not loaded".to_string()))?;
        let text = request.text;
        let speaker = request
            .config
            .options
            .speaker
            .or(request.config.options.voice);
        let language = request.language;
        let speed = request.config.options.speed;
        let model_for_plan = model.clone();
        let speaker_for_plan = speaker.clone();
        let language_for_plan = language.clone();

        let (mut residency_lease, stream_chunks) = self
            .coordinator
            .run_blocking_stage(job, move || {
                let stream_chunks = plan_kokoro_streaming_chunks(
                    model_for_plan.as_ref(),
                    &text,
                    speaker_for_plan.as_deref(),
                    language_for_plan.as_deref(),
                    speed,
                )?;
                Ok((residency_lease, stream_chunks))
            })
            .await?;
        let total_chunks = stream_chunks.len();
        if total_chunks > output_budget.max_chunks {
            return Err(Error::InferenceError(format!(
                "Kokoro streaming planner exceeded its hard chunk contract: {total_chunks} > {}",
                output_budget.max_chunks
            )));
        }
        let mut expected_sample_rate: Option<u32> = None;
        let mut emitted_samples = 0usize;
        let mut materialized_output_capacity = 0usize;

        for (sequence, chunk_text) in stream_chunks.into_iter().enumerate() {
            let model_for_task = model.clone();
            let speaker_for_task = speaker.clone();
            let language_for_task = language.clone();
            let (returned_lease, synthesis) = self
                .coordinator
                .run_blocking_stage(job, move || {
                    let synthesis = synthesize_kokoro_with_fallback(
                        model_for_task,
                        &chunk_text,
                        speaker_for_task.as_deref(),
                        language_for_task.as_deref(),
                        speed,
                        output_budget.max_samples,
                    )?;
                    Ok((residency_lease, synthesis))
                })
                .await?;
            residency_lease = returned_lease;
            let current_sample_rate = synthesis.sample_rate;
            match expected_sample_rate {
                Some(expected) if expected != current_sample_rate => {
                    return Err(Error::InferenceError(format!(
                        "Kokoro streaming sample rate mismatch: expected {}, got {}",
                        expected, current_sample_rate
                    )));
                }
                None => expected_sample_rate = Some(current_sample_rate),
                _ => {}
            }

            emitted_samples = checked_kokoro_sample_growth(
                emitted_samples,
                synthesis.samples.len(),
                output_budget.max_samples,
                "streamed samples",
            )?;
            materialized_output_capacity = checked_kokoro_sample_growth(
                materialized_output_capacity,
                synthesis.samples.capacity(),
                output_budget.max_samples,
                "streamed sample capacity",
            )?;
            record_kokoro_materialized_output(
                job,
                retained_input_bytes,
                materialized_output_capacity,
                output_budget.max_samples,
            )?;

            let mut chunk = AudioChunk::new(request_id.clone(), sequence, synthesis.samples)
                .with_sample_rate(current_sample_rate);
            chunk.is_final = sequence + 1 == total_chunks;
            match job.spec.deadline {
                Some(deadline) => tokio::time::timeout_at(deadline.into(), chunk_tx.send(chunk))
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
                    .map_err(|_| {
                        Error::InferenceError("Streaming output channel closed".to_string())
                    })?,
                None => chunk_tx.send(chunk).await.map_err(|_| {
                    Error::InferenceError("Streaming output channel closed".to_string())
                })?,
            }
        }
        Ok(())
    }
}

fn synthesize_kokoro_with_fallback(
    model: Arc<KokoroTtsModel>,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
    max_samples: usize,
) -> Result<KokoroSynthesisResult> {
    match model.generate(text, speaker, language, speed) {
        Ok(result) => {
            validate_kokoro_samples(&result.samples, max_samples, "synthesis")?;
            Ok(result)
        }
        Err(err) if is_kokoro_voice_pack_limit_error(&err) => {
            info!("Kokoro phoneme limit hit for request; retrying with adaptive chunking fallback");
            generate_chunked_kokoro(model, text, speaker, language, speed, max_samples)
        }
        Err(err) => Err(err),
    }
}

fn is_kokoro_voice_pack_limit_error(err: &Error) -> bool {
    match err {
        Error::InvalidInput(msg) => {
            msg.contains("Kokoro phoneme sequence length") && msg.contains("voice-pack limit (510)")
        }
        _ => false,
    }
}

fn checked_kokoro_sample_growth(
    current: usize,
    additional: usize,
    max_samples: usize,
    label: &str,
) -> Result<usize> {
    let next = current
        .checked_add(additional)
        .ok_or_else(|| Error::Overloaded(format!("Kokoro {label} count overflowed")))?;
    if next > max_samples {
        return Err(Error::InferenceError(format!(
            "Kokoro {label} exceeded the admitted hard sample contract: {next} > {max_samples}"
        )));
    }
    Ok(next)
}

fn validate_kokoro_samples(samples: &Vec<f32>, max_samples: usize, label: &str) -> Result<()> {
    checked_kokoro_sample_growth(0, samples.len(), max_samples, label)?;
    checked_kokoro_sample_growth(
        0,
        samples.capacity(),
        max_samples,
        &format!("{label} capacity"),
    )?;
    Ok(())
}

fn append_kokoro_samples_bounded(
    combined: &mut Vec<f32>,
    chunk: Vec<f32>,
    pause_samples: usize,
    max_samples: usize,
) -> Result<()> {
    validate_kokoro_samples(&chunk, max_samples, "chunk synthesis")?;
    let additional = chunk
        .len()
        .checked_add(pause_samples)
        .ok_or_else(|| Error::Overloaded("Kokoro chunk append overflowed".to_string()))?;
    let next_len = checked_kokoro_sample_growth(
        combined.len(),
        additional,
        max_samples,
        "accumulated samples",
    )?;

    combined.try_reserve_exact(additional).map_err(|_| {
        Error::Overloaded(format!(
            "Kokoro could not reserve {additional} additional output samples"
        ))
    })?;
    if combined.capacity() > max_samples {
        return Err(Error::InferenceError(format!(
            "Kokoro accumulated sample capacity exceeded the admitted hard contract: {} > {max_samples}",
            combined.capacity()
        )));
    }
    combined.extend(chunk);
    combined.resize(next_len, 0.0);
    Ok(())
}

fn record_kokoro_materialized_output(
    job: &JobLease,
    retained_input_bytes: u64,
    output_capacity_samples: usize,
    max_samples: usize,
) -> Result<()> {
    checked_kokoro_sample_growth(
        0,
        output_capacity_samples,
        max_samples,
        "materialized sample capacity",
    )?;
    let output_bytes = u64::try_from(output_capacity_samples)
        .ok()
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| Error::Overloaded("Kokoro materialized output overflowed".to_string()))?;
    let materialized_host_bytes =
        retained_input_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| {
                Error::Overloaded("Kokoro materialized host usage overflowed".to_string())
            })?;
    // The output ceiling is part of the job's immutable pre-admission
    // reservation. This only reclassifies authorized pending bytes after the
    // bounded Vec exists; it never grows the lease.
    job.record_materialized_usage(JobResourceObservation::host(materialized_host_bytes))
}

fn generate_chunked_kokoro(
    model: Arc<KokoroTtsModel>,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
    max_samples: usize,
) -> Result<KokoroSynthesisResult> {
    let chunks = plan_kokoro_text_chunks(model.as_ref(), text, speaker, language, speed)?;
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Kokoro adaptive chunking produced no chunks".to_string(),
        ));
    }

    let mut combined_samples = Vec::new();
    let mut combined_phonemes = String::new();
    let mut total_tokens = 0usize;
    let mut sample_rate: Option<u32> = None;

    for (idx, chunk_text) in chunks.iter().enumerate() {
        let chunk = model.generate(chunk_text, speaker, language, speed)?;
        let current_sample_rate = chunk.sample_rate;
        match sample_rate {
            Some(expected) if expected != current_sample_rate => {
                return Err(Error::InferenceError(format!(
                    "Kokoro chunked synthesis sample rate mismatch: expected {}, got {}",
                    expected, current_sample_rate
                )));
            }
            None => sample_rate = Some(current_sample_rate),
            _ => {}
        }

        if !combined_phonemes.is_empty() {
            combined_phonemes.push(' ');
        }
        combined_phonemes.push_str(chunk.phonemes.trim());
        total_tokens = total_tokens
            .checked_add(chunk.tokens_generated)
            .ok_or_else(|| {
                Error::Overloaded("Kokoro generated-token count overflowed".to_string())
            })?;

        let pause_samples = if idx + 1 < chunks.len() {
            let pause_samples = ((current_sample_rate as f32) * 0.04).round() as usize;
            pause_samples
        } else {
            0
        };
        append_kokoro_samples_bounded(
            &mut combined_samples,
            chunk.samples,
            pause_samples,
            max_samples,
        )?;
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        Error::InferenceError("Kokoro adaptive chunking failed to synthesize audio".to_string())
    })?;
    info!(
        chunks = chunks.len(),
        total_tokens,
        sample_rate,
        total_samples = combined_samples.len(),
        "Kokoro adaptive chunking completed"
    );
    Ok(KokoroSynthesisResult {
        samples: combined_samples,
        sample_rate,
        tokens_generated: total_tokens,
        phonemes: combined_phonemes,
    })
}

fn plan_kokoro_streaming_chunks(
    model: &KokoroTtsModel,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
) -> Result<Vec<String>> {
    let fit_chunks = plan_kokoro_text_chunks(model, text, speaker, language, speed)?;
    let mut stream_chunks = Vec::new();

    for fit_chunk in fit_chunks {
        for candidate in split_text_for_streaming(
            fit_chunk.as_str(),
            KOKORO_STREAM_TARGET_CHARS,
            KOKORO_STREAM_MIN_CHARS,
        ) {
            let verified = plan_kokoro_text_chunks(model, &candidate, speaker, language, speed)?;
            for chunk in verified {
                if !chunk.trim().is_empty() {
                    stream_chunks.push(chunk);
                }
            }
        }
    }

    if stream_chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Kokoro streaming planner produced no chunks".to_string(),
        ));
    }

    Ok(stream_chunks)
}

fn split_text_for_streaming(text: &str, target_chars: usize, min_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut remaining = text.trim();
    if remaining.is_empty() {
        return out;
    }

    let target_chars = target_chars.max(1);
    let min_chars = min_chars.max(1).min(target_chars);

    while !remaining.is_empty() {
        let remaining_chars = remaining.chars().count();
        if remaining_chars <= target_chars {
            out.push(remaining.to_string());
            break;
        }

        let mut split_chars = pick_readable_split_point(remaining, target_chars);
        if split_chars == 0 || split_chars > remaining_chars {
            split_chars = target_chars.min(remaining_chars);
        }
        if split_chars < min_chars && remaining_chars > min_chars {
            split_chars = min_chars.min(remaining_chars);
        }

        let (candidate_head, candidate_tail) = split_at_char_index(remaining, split_chars);
        let head = candidate_head.trim_end();
        if head.is_empty() {
            let fallback_chars = target_chars.min(remaining_chars).max(1);
            let (fallback_head, fallback_tail) = split_at_char_index(remaining, fallback_chars);
            let fallback_head = fallback_head.trim_end();
            if fallback_head.is_empty() {
                out.push(remaining.to_string());
                break;
            }
            out.push(fallback_head.to_string());
            remaining = fallback_tail.trim_start();
            continue;
        }

        out.push(head.to_string());
        remaining = candidate_tail.trim_start();
    }

    out
}

fn plan_kokoro_text_chunks(
    model: &KokoroTtsModel,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    let mut remaining = text;
    let mut iterations = 0usize;

    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        iterations += 1;
        if iterations > 1024 {
            return Err(Error::InferenceError(
                "Kokoro adaptive chunking exceeded maximum chunk iterations".to_string(),
            ));
        }

        match model.prepare_request(remaining, speaker, language, speed) {
            Ok(_) => {
                chunks.push(remaining.trim_end().to_string());
                break;
            }
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => {
                let max_fit_chars =
                    find_max_fitting_prefix_chars(model, remaining, speaker, language, speed)?;
                let mut split_chars = pick_readable_split_point(remaining, max_fit_chars);
                if split_chars == 0 || split_chars > max_fit_chars {
                    split_chars = max_fit_chars;
                }

                let (candidate_head, candidate_tail) = split_at_char_index(remaining, split_chars);
                let candidate_head = candidate_head.trim_end();
                let (head, tail) = if candidate_head.is_empty() {
                    let (fallback_head, fallback_tail) =
                        split_at_char_index(remaining, max_fit_chars);
                    (fallback_head.trim_end(), fallback_tail)
                } else {
                    (candidate_head, candidate_tail)
                };

                if head.is_empty() {
                    return Err(Error::InvalidInput(
                        "Kokoro adaptive chunking could not produce a non-empty chunk".to_string(),
                    ));
                }

                chunks.push(head.to_string());
                remaining = tail;
            }
            Err(err) => return Err(err),
        }
    }

    Ok(chunks)
}

fn find_max_fitting_prefix_chars(
    model: &KokoroTtsModel,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
) -> Result<usize> {
    let total_chars = text.chars().count();
    if total_chars == 0 {
        return Err(Error::InvalidInput(
            "Kokoro adaptive chunking received empty text".to_string(),
        ));
    }

    let mut lo = 1usize;
    let mut hi = total_chars;
    let mut best = 0usize;

    while lo <= hi {
        let mid = lo + ((hi - lo) / 2);
        let (prefix, _) = split_at_char_index(text, mid);
        let prefix = prefix.trim_end();
        if prefix.is_empty() {
            lo = mid.saturating_add(1);
            continue;
        }

        match model.prepare_request(prefix, speaker, language, speed) {
            Ok(_) => {
                best = prefix.chars().count();
                lo = mid.saturating_add(1);
            }
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
            Err(Error::InvalidInput(msg))
                if msg.contains("Kokoro phonemizer produced no phonemes") =>
            {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
            Err(err) => return Err(err),
        }
    }

    if best > 0 {
        return Ok(best);
    }

    for n in (1..=total_chars).rev() {
        let (prefix, _) = split_at_char_index(text, n);
        let prefix = prefix.trim_end();
        if prefix.is_empty() {
            continue;
        }
        match model.prepare_request(prefix, speaker, language, speed) {
            Ok(_) => return Ok(prefix.chars().count()),
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => continue,
            Err(Error::InvalidInput(msg))
                if msg.contains("Kokoro phonemizer produced no phonemes") =>
            {
                continue;
            }
            Err(err) => return Err(err),
        }
    }

    Err(Error::InvalidInput(
        "Kokoro adaptive chunking could not find a chunk within the voice-pack phoneme limit"
            .to_string(),
    ))
}

fn pick_readable_split_point(text: &str, max_chars: usize) -> usize {
    if max_chars == 0 {
        return 0;
    }

    let mut last_sentence_break = None;
    let mut last_clause_break = None;
    let mut last_whitespace = None;

    for (idx, ch) in text.chars().enumerate() {
        let pos = idx + 1;
        if pos > max_chars {
            break;
        }
        if ch.is_whitespace() {
            last_whitespace = Some(pos);
            continue;
        }
        if matches!(ch, '.' | '!' | '?' | '\n') {
            last_sentence_break = Some(pos);
        } else if matches!(ch, ';' | ':' | ',') {
            last_clause_break = Some(pos);
        }
    }

    let preferred_min = (max_chars * 2) / 3;
    for candidate in [last_sentence_break, last_clause_break, last_whitespace] {
        if let Some(pos) = candidate {
            if pos >= preferred_min && pos <= max_chars {
                return pos;
            }
        }
    }
    for candidate in [last_sentence_break, last_clause_break, last_whitespace] {
        if let Some(pos) = candidate {
            if pos > 0 && pos <= max_chars {
                return pos;
            }
        }
    }
    max_chars
}

fn split_at_char_index(s: &str, n: usize) -> (&str, &str) {
    if n == 0 {
        return ("", s);
    }
    let byte_idx = s
        .char_indices()
        .nth(n)
        .map(|(idx, _)| idx)
        .unwrap_or(s.len());
    s.split_at(byte_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_text_for_streaming_keeps_short_text_single_chunk() {
        let text = "Hello from Kokoro streaming.";
        let chunks = split_text_for_streaming(text, 180, 64);
        assert_eq!(chunks, vec![text.to_string()]);
    }

    #[test]
    fn split_text_for_streaming_prefers_sentence_boundaries() {
        let text = "Sentence one ends here. Sentence two continues with additional words so the planner can split naturally.";
        let chunks = split_text_for_streaming(text, 30, 20);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].ends_with('.'));
    }

    #[test]
    fn chunk_accumulation_rejects_overflow_before_mutating_the_output() {
        let mut combined = vec![0.1, 0.2];
        let before = combined.clone();
        let before_capacity = combined.capacity();

        let result = append_kokoro_samples_bounded(&mut combined, vec![0.3, 0.4], 2, 5);

        assert!(matches!(
            result,
            Err(Error::InferenceError(message))
                if message.contains("hard sample contract")
        ));
        assert_eq!(combined, before);
        assert_eq!(combined.capacity(), before_capacity);
    }
}
