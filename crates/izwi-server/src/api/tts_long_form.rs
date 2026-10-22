use std::sync::Arc;

use izwi_core::{GenerationRequest, GenerationResult, ModelVariant, RuntimeService, TtsStopReason};

const DEFAULT_CHUNK_MAX_CHARS: usize = 480;
const CHUNK_MAX_CHARS_MIN: usize = 80;
const CHUNK_MAX_CHARS_MAX: usize = 4000;
/// Qwen3-TTS long-form segments: whole sentences packed up to about the
/// medium test paragraph (45 words, 260 characters), which ends on EOS on
/// every backend. A 460-character segment of the 250-word paragraph ran to
/// its 663-frame budget and babbled (WER 0.32 on CPU); the reference model
/// generates sentence by sentence. Text longer than this splits even when
/// one request could hold it.
const QWEN_SEGMENT_MAX_CHARS: usize = 300;

/// Silence between two long-form segments: a sentence pause, not the raw
/// back-to-back concatenation that ran sentences together (TPIPE-5).
const SEGMENT_PAUSE_SECS: f32 = 0.12;
/// A short fade on each side of the pause, so no segment ends or starts on
/// a click.
const SEGMENT_FADE_SECS: f32 = 0.01;

fn seconds_to_samples(secs: f32, sample_rate: u32) -> usize {
    (secs * sample_rate as f32).round().max(1.0) as usize
}

/// Samples in the fade on each side of a segment join.
pub(crate) fn segment_fade_samples(sample_rate: u32) -> usize {
    seconds_to_samples(SEGMENT_FADE_SECS, sample_rate)
}

/// Samples of silence between two segments.
pub(crate) fn segment_pause_samples(sample_rate: u32) -> usize {
    seconds_to_samples(SEGMENT_PAUSE_SECS, sample_rate)
}

/// Fade the first samples of a segment in from silence.
pub(crate) fn fade_in_segment(samples: &mut [f32], sample_rate: u32) {
    let fade = segment_fade_samples(sample_rate).min(samples.len());
    for (index, sample) in samples.iter_mut().take(fade).enumerate() {
        *sample *= (index + 1) as f32 / (fade + 1) as f32;
    }
}

/// Fade the last samples of a segment out to silence.
pub(crate) fn fade_out_segment(samples: &mut [f32], sample_rate: u32) {
    let fade = segment_fade_samples(sample_rate).min(samples.len());
    let tail = samples.len() - fade;
    for (index, sample) in samples[tail..].iter_mut().enumerate() {
        *sample *= (fade - index) as f32 / (fade + 1) as f32;
    }
}

/// The pause before a later segment, followed by that segment faded in.
pub(crate) fn pause_then_segment(segment: &[f32], sample_rate: u32) -> Vec<f32> {
    let pause = segment_pause_samples(sample_rate);
    let mut out = Vec::with_capacity(pause + segment.len());
    out.resize(pause, 0.0);
    out.extend_from_slice(segment);
    fade_in_segment(&mut out[pause..], sample_rate);
    out
}

/// Append one long-form segment: the first as is, a later one after a
/// faded-out tail and a short pause, faded in.
pub(crate) fn append_segment(joined: &mut Vec<f32>, segment: &[f32], sample_rate: u32) {
    if joined.is_empty() {
        joined.extend_from_slice(segment);
        return;
    }
    fade_out_segment(joined, sample_rate);
    joined.extend(pause_then_segment(segment, sample_rate));
}

fn chunk_max_chars() -> usize {
    configured_chunk_max_chars().unwrap_or(DEFAULT_CHUNK_MAX_CHARS)
}

fn configured_chunk_max_chars() -> Option<usize> {
    crate::env_config::tuning()
        .tts_long_form_chunk_max_chars
        .map(|value| value.clamp(CHUNK_MAX_CHARS_MIN, CHUNK_MAX_CHARS_MAX))
}

/// The most characters one long-form segment of `variant` packs: the
/// configured cap, else Qwen3-TTS's segment size or the default.
fn segment_max_chars(variant: ModelVariant) -> usize {
    configured_chunk_max_chars().unwrap_or(
        if variant.family() == izwi_core::ModelFamily::Qwen3Tts {
            QWEN_SEGMENT_MAX_CHARS
        } else {
            DEFAULT_CHUNK_MAX_CHARS
        },
    )
}

/// Whether text under an automatic budget splits into long-form segments:
/// when one request cannot speak it, or (Qwen3-TTS) when it is longer than
/// one segment.
fn splits_into_segments(variant: ModelVariant, text: &str) -> bool {
    crate::api::tts_policy::tts_text_exceeds_one_request(variant, text)
        || (variant.family() == izwi_core::ModelFamily::Qwen3Tts
            && text.trim().chars().count() > segment_max_chars(variant))
}

/// A stable, versioned partition of the original UTF-8 input. Audio limits are
/// assigned after partitioning, never used as a switch for complete-text planning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct SpeechTextPlan {
    pub version: u32,
    pub text_sha256: String,
    pub segments: Vec<std::ops::Range<usize>>,
    pub max_output_frames: Option<usize>,
}

impl SpeechTextPlan {
    pub fn fish(text: &str, max_output_frames: usize) -> Result<Self, izwi_core::Error> {
        use sha2::{Digest, Sha256};
        let max_bytes = crate::api::tts_policy::tts_max_text_bytes();
        if text.len() > max_bytes {
            return Err(izwi_core::Error::InvalidInput(format!(
                "Speech text exceeds configured limit of {max_bytes} bytes"
            )));
        }
        let mut segments = Vec::new();
        let mut start = 0;
        let mut chars = 0;
        let mut boundary = None;
        let mut sentence_boundary = None;
        // Keep Fish segments comfortably below its per-call context/output limit.
        let target = chunk_max_chars().min(480);
        for (offset, ch) in text.char_indices() {
            if chars >= target {
                let end = sentence_boundary
                    .filter(|end| *end > start)
                    .or_else(|| boundary.filter(|end| *end > start))
                    .unwrap_or(offset);
                segments.push(start..end);
                start = end;
                chars = text[start..offset].chars().count();
                boundary = None;
                sentence_boundary = None;
            }
            chars += 1;
            if matches!(ch, '。' | '！' | '？' | '\n' | ';' | '；') {
                sentence_boundary = Some(offset + ch.len_utf8());
            } else if matches!(ch, '.' | '!' | '?')
                && text[offset + ch.len_utf8()..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
            {
                let word = text[start..offset].split_whitespace().last().unwrap_or("");
                if ch != '.' || word.chars().count() > 3 {
                    sentence_boundary = Some(offset + ch.len_utf8());
                }
            }
            if ch.is_whitespace() || matches!(ch, '。' | '！' | '？' | ';' | '；') {
                boundary = Some(offset + ch.len_utf8());
            }
        }
        if start < text.len() && !text[start..].trim().is_empty() {
            segments.push(start..text.len());
        } else if let Some(last) = segments.last_mut() {
            last.end = text.len();
        }
        // Blank layout alone is not an inference segment. Byte ranges still
        // identify original content for durable progress and recovery.
        segments.retain(|range| !text[range.clone()].trim().is_empty());
        if segments.len() > 1 && !crate::env_config::tuning().fish_long_form_enabled {
            return Err(izwi_core::Error::InvalidInput(
                "Fish long-form generation is disabled by this deployment".into(),
            ));
        }
        let estimated_frames = segments
            .iter()
            .try_fold(0usize, |total, range| {
                total.checked_add(
                    crate::api::tts_policy::resolve_tts_output_frames(
                        ModelVariant::FishAudioS2Pro,
                        &text[range.clone()],
                        None,
                    )
                    .unwrap_or(4096),
                )
            })
            .ok_or_else(|| izwi_core::Error::InvalidInput("Speech output size overflow".into()))?;
        let estimated_frames = if max_output_frames > 0 {
            estimated_frames.min(max_output_frames)
        } else {
            estimated_frames
        };
        let max_pcm_bytes = crate::env_config::tuning()
            .tts_stream_max_pcm_bytes
            .unwrap_or(u32::MAX as usize - 44)
            .min(u32::MAX as usize - 44);
        if estimated_frames
            .checked_mul(2048 * 2)
            .is_none_or(|bytes| bytes > max_pcm_bytes)
        {
            return Err(izwi_core::Error::InvalidInput("Speech plan exceeds configured PCM/WAV output limit; shorten the text or raise the deployment limit within WAV capacity".into()));
        }
        Ok(Self {
            version: 1,
            text_sha256: hex::encode(Sha256::digest(text.as_bytes())),
            segments,
            max_output_frames: (max_output_frames > 0).then_some(max_output_frames),
        })
    }
}

fn should_enable_chunking(variant: ModelVariant, requested_max_tokens: usize) -> bool {
    let Some(model_max_frames) = variant.tts_max_output_frames_hint() else {
        return false;
    };
    requested_max_tokens == 0 || requested_max_tokens >= model_max_frames
}

pub fn split_tts_text_for_long_form(
    variant: ModelVariant,
    requested_max_tokens: usize,
    text: &str,
) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if variant == ModelVariant::FishAudioS2Pro {
        // Callers accepting work must validate SpeechTextPlan first. This legacy
        // infallible splitter is also used by Studio's text editor.
        return SpeechTextPlan::fish(trimmed, requested_max_tokens)
            .map(|plan| {
                plan.segments
                    .into_iter()
                    .map(|range| trimmed[range].to_string())
                    .collect()
            })
            .unwrap_or_else(|_| vec![trimmed.to_string()]);
    }
    if !should_enable_chunking(variant, requested_max_tokens) {
        return vec![trimmed.to_string()];
    }

    // Whole sentences, packed up to the character cap and what one request
    // of this family can speak (TPIPE-5).
    let chunks = crate::api::sentence_split::pack_sentences(
        trimmed,
        segment_max_chars(variant),
        &|segment| !crate::api::tts_policy::tts_text_exceeds_one_request(variant, segment),
    );

    if chunks.is_empty() {
        vec![trimmed.to_string()]
    } else {
        chunks
    }
}

/// Requests for one speech job. An automatic budget (`max_tokens` 0)
/// splits only text one request cannot finish (TPIPE-2), and every request
/// gets the automatic budget of its own text; an explicit budget keeps the
/// caller's split rule.
pub fn expand_generation_requests_for_long_form(
    base_request: &GenerationRequest,
    variant: ModelVariant,
) -> Vec<GenerationRequest> {
    let requested = base_request.config.options.max_tokens;
    let automatic = requested == 0 && variant != ModelVariant::FishAudioS2Pro;
    let chunks = if automatic && !splits_into_segments(variant, &base_request.text) {
        vec![base_request.text.trim().to_string()]
    } else {
        split_tts_text_for_long_form(variant, requested, base_request.text.as_str())
    };
    let budget_for = |request: &mut GenerationRequest| {
        if variant == ModelVariant::FishAudioS2Pro || automatic {
            if let Some(frames) =
                crate::api::tts_policy::resolve_tts_output_frames(variant, &request.text, None)
            {
                request.config.options.max_tokens = frames;
            } else if variant == ModelVariant::FishAudioS2Pro {
                request.config.options.max_tokens = 4096;
            }
        }
    };

    if chunks.len() <= 1 {
        let mut request = base_request.clone();
        if variant == ModelVariant::FishAudioS2Pro && request.config.options.max_tokens != 0 {
            return vec![request];
        }
        budget_for(&mut request);
        return vec![request];
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, text)| {
            let mut req = base_request.clone();
            req.id = format!("{}:{}", base_request.id, idx + 1);
            req.text = text;
            budget_for(&mut req);
            req
        })
        .collect()
}

/// Leave the budget automatic (`0`) when the client set none and the text is
/// longer than one automatic request can speak, so long form splits it
/// instead of the request being cut at its cap.
pub(crate) fn defer_automatic_budget_to_long_form(
    request: &mut GenerationRequest,
    variant: ModelVariant,
    explicit_frames: Option<usize>,
) {
    if explicit_frames.unwrap_or(0) == 0
        && variant != ModelVariant::FishAudioS2Pro
        && splits_into_segments(variant, &request.text)
    {
        request.config.options.max_tokens = 0;
    }
}

pub async fn generate_long_form_tts(
    runtime: &Arc<RuntimeService>,
    variant: ModelVariant,
    mut request: GenerationRequest,
) -> Result<GenerationResult, izwi_core::Error> {
    request.model_variant.get_or_insert(variant);
    let retry_runaways = retries_runaway_segments(variant, &request);
    let planned_requests = expand_generation_requests_for_long_form(&request, variant);
    if planned_requests.len() == 1 {
        return runtime
            .generate(planned_requests.into_iter().next().unwrap())
            .await;
    }

    let started = std::time::Instant::now();
    let split_request_count = planned_requests.len();
    let outputs = generate_segments(
        planned_requests,
        long_form_segment_concurrency(runtime),
        retry_runaways,
        &request.id,
        |segment| runtime.generate(segment),
    )
    .await?;
    let mut constituent_diagnostics = Vec::with_capacity(split_request_count);
    let mut merged_samples: Vec<f32> = Vec::new();
    let mut sample_rate: Option<u32> = None;
    let mut total_tokens = 0usize;
    let mut total_time_ms = 0f32;
    let mut stop_reasons = Vec::with_capacity(split_request_count);
    let mut retried_segments = Vec::new();
    let mut budget_segments = Vec::new();

    for (index, (output, retried)) in outputs.into_iter().enumerate() {
        if retried {
            retried_segments.push(index + 1);
        }
        if ran_to_budget(&output) {
            tracing::warn!(
                request_id = %request.id,
                segment = index + 1,
                "Long-form segment ended at its budget"
            );
            budget_segments.push(index + 1);
        }
        stop_reasons.push(output.stop_reason);
        if let Some(existing_rate) = sample_rate {
            if existing_rate != output.sample_rate {
                return Err(izwi_core::Error::InferenceError(format!(
                    "Long-form TTS sample-rate mismatch: {existing_rate} vs {}",
                    output.sample_rate
                )));
            }
        } else {
            sample_rate = Some(output.sample_rate);
        }
        append_segment(&mut merged_samples, &output.samples, output.sample_rate);
        total_tokens = total_tokens.saturating_add(output.total_tokens);
        total_time_ms += output.total_time_ms;
        constituent_diagnostics.push(output.diagnostics);
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        izwi_core::Error::InferenceError("Long-form TTS produced no chunks".to_string())
    })?;

    Ok(GenerationResult {
        request_id: request.id,
        samples: merged_samples,
        sample_rate,
        total_tokens,
        total_time_ms,
        diagnostics: Some(serde_json::json!({
            "timing_basis": "sum_of_engine_executions",
            "split_request_count": split_request_count,
            "request_wall_ms": started.elapsed().as_secs_f64() * 1000.0,
            "constituents": constituent_diagnostics,
            // 1-based segments sampled again, and those that still ended
            // at their budget (the reason for a `length` stop).
            "retried_segments": retried_segments,
            "budget_segments": budget_segments,
        })),
        stop_reason: TtsStopReason::combine(stop_reasons),
    })
}

/// Long-form segments in flight at once: the rows the engine batches TTS at
/// (`RuntimeService::tts_row_width`), so segments synthesize side by side
/// instead of one after another (SPK-F12). More would only queue.
fn long_form_segment_concurrency(runtime: &RuntimeService) -> usize {
    runtime.tts_row_width()
}

/// Generate the segments with up to `concurrency` in flight, returning each
/// output (and whether it was sampled again) in segment order. A sampled
/// Qwen3-TTS segment that ran to its budget without ending speech babbled
/// (TPIPE-27); when `retry_runaways`, it is sampled once more and the second
/// draw kept if it ended.
async fn generate_segments<F, Fut>(
    segments: Vec<GenerationRequest>,
    concurrency: usize,
    retry_runaways: bool,
    request_id: &str,
    generate: F,
) -> Result<Vec<(GenerationResult, bool)>, izwi_core::Error>
where
    F: Fn(GenerationRequest) -> Fut,
    Fut: std::future::Future<Output = Result<GenerationResult, izwi_core::Error>>,
{
    use futures::{StreamExt, TryStreamExt};
    let generate = &generate;
    futures::stream::iter(
        segments
            .into_iter()
            .enumerate()
            .map(|(index, segment)| async move {
                let retry = retry_runaways.then(|| segment.clone());
                let output = generate(segment).await?;
                let Some(retry) = retry.filter(|_| ran_to_budget(&output)) else {
                    return Ok((output, false));
                };
                tracing::warn!(
                    request_id = %request_id,
                    segment = index + 1,
                    "Long-form segment ran to its budget without end of speech; sampling it again"
                );
                let again = generate(retry).await?;
                Ok((if ran_to_budget(&again) { output } else { again }, true))
            }),
    )
    .buffered(concurrency.max(1))
    .try_collect()
    .await
}

/// Whether long form samples a Qwen3-TTS segment again when it runs to its
/// automatic budget: only for unseeded sampling, where a second draw differs
/// (greedy or a fixed seed would repeat the runaway).
fn retries_runaway_segments(variant: ModelVariant, request: &GenerationRequest) -> bool {
    let options = &request.config.options;
    variant.family() == izwi_core::ModelFamily::Qwen3Tts
        && options.max_tokens == 0
        && options.speech.seed.is_none()
        && options
            .speech
            .temperature
            .is_none_or(|temperature| temperature > 0.0)
        && options.speech.top_k != Some(1)
}

fn ran_to_budget(output: &GenerationResult) -> bool {
    output.stop_reason == Some(TtsStopReason::OutputBudget)
}

/// One live inference segment per job. Consumers receive one ordered stream and
/// one terminal total; neither PCM nor reference clones accumulate with duration.
pub(crate) async fn generate_speech_plan_stream(
    state: &crate::state::AppState,
    variant: ModelVariant,
    base: GenerationRequest,
    output: tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    workload: izwi_core::WorkloadClass,
) -> Result<(), izwi_core::Error> {
    generate_speech_plan_stream_with_progress(state, variant, base, output, workload, None).await
}

pub(crate) async fn generate_speech_plan_stream_with_progress(
    state: &crate::state::AppState,
    variant: ModelVariant,
    base: GenerationRequest,
    output: tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    workload: izwi_core::WorkloadClass,
    mut progress: Option<&mut crate::api::speech_history::DurableSpeechProgress>,
) -> Result<(), izwi_core::Error> {
    use izwi_core::{AudioChunk, ChunkStats};
    let progress_error = |error: anyhow::Error| izwi_core::Error::InferenceError(error.to_string());
    let expected_fingerprint = state
        .runtime
        .fish_s2_artifact_fingerprint()
        .await
        .ok_or_else(|| {
            izwi_core::Error::ModelLoadError("Fish model fingerprint unavailable".into())
        })?;
    if progress
        .as_deref()
        .and_then(|p| p.artifact_fingerprint())
        .is_some_and(|expected| expected != expected_fingerprint)
    {
        return Err(izwi_core::Error::InvalidInput(
            "Fish model changed during speech recovery".into(),
        ));
    }
    let plan = SpeechTextPlan::fish(&base.text, base.config.options.max_tokens)?;
    let mut ranges: std::collections::VecDeque<_> = plan.segments.into();
    let mut remaining = plan.max_output_frames.unwrap_or(usize::MAX);
    let mut sequence = progress.as_deref().map_or(0, |p| p.next_sequence());
    let mut segment = progress.as_deref().map_or(0, |p| p.completed_segments());
    let mut tokens = progress.as_deref().map_or(0, |p| p.completed_tokens());
    // A resumed plan cannot see earlier segments' stop reasons.
    let mut stop_reasons = vec![if tokens > 0 {
        None
    } else {
        Some(TtsStopReason::EndOfSpeech)
    }];
    let mut execution_ms = progress
        .as_deref()
        .map_or(0.0, |p| p.completed_execution_ms());
    let mut duration = progress
        .as_deref()
        .map_or(0.0, |p| p.completed_duration_secs());
    remaining = remaining.saturating_sub(tokens);
    let mut rate = None;
    let completed = progress.as_deref().map_or(0, |p| p.completed_text_bytes());
    if completed > base.text.len() || !base.text.is_char_boundary(completed) {
        return Err(izwi_core::Error::InvalidInput(
            "Invalid speech recovery text offset".into(),
        ));
    }
    while let Some(mut range) = ranges.pop_front() {
        if progress
            .as_deref()
            .is_some_and(|p| range.end <= p.completed_text_bytes())
        {
            continue;
        }
        // A previously completed adaptive split can end inside an original range.
        // Resume only its suffix, never replay the already committed prefix.
        if let Some(progress) = progress.as_deref() {
            range.start = range.start.max(progress.completed_text_bytes());
        }
        if let Some(progress) = progress.as_deref_mut() {
            progress
                .begin_segment(segment, range.clone(), segment + 1 + ranges.len())
                .await
                .map_err(progress_error)?;
        }
        if output.is_closed() {
            return Err(izwi_core::Error::InferenceError(
                "Speech consumer disconnected".into(),
            ));
        }
        if remaining == 0 {
            return Err(izwi_core::Error::InvalidInput(
                "Speech generation incomplete: output_limit".into(),
            ));
        }
        if state
            .runtime
            .fish_s2_artifact_fingerprint()
            .await
            .as_deref()
            != Some(expected_fingerprint.as_str())
        {
            return Err(izwi_core::Error::InvalidInput(
                "Fish model changed during speech generation".into(),
            ));
        }
        let mut request = base.clone();
        request.id = format!("{}:{}", base.id, segment);
        request.text = base.text[range.clone()].to_string();
        request.config.streaming = true;
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, &request.text, None)
                .unwrap_or(4096)
                .min(remaining);
        let permit = state.acquire_workload_permit(workload).await;
        let tenant = base.runtime_context.tenant_key;
        request.runtime_context = permit.runtime_context();
        request.runtime_context.tenant_key = tenant;
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<AudioChunk>(2);
        let runtime = state.runtime.clone();
        let task = tokio::spawn(async move { runtime.generate_streaming(request, sender).await });
        let _abort = AbortSpeechTask(task.abort_handle());
        // A progress deadline, not a wall clock: a segment fails only when
        // it publishes nothing for this long (TPIPE-6).
        let stall_timeout = crate::env_config::tuning().tts_segment_timeout;
        let mut deadline = tokio::time::Instant::now() + stall_timeout;
        let mut segment_duration = 0.0;
        let mut emitted = false;
        let mut segment_tokens = 0;
        let mut segment_ms = 0.0;
        let mut segment_stop = None;
        loop {
            // Buffered durable audio is published by its deadline even when
            // the model is between chunks.
            let flush_at = progress.as_deref().and_then(|p| p.flush_deadline());
            let flush_due = async {
                match flush_at {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            };
            let mut chunk = tokio::select! {
                chunk = receiver.recv() => match chunk {
                    Some(chunk) => chunk,
                    None => break,
                },
                _ = tokio::time::sleep_until(deadline) => return Err(izwi_core::Error::Timeout("Speech segment made no progress".into())),
                _ = output.closed() => return Err(izwi_core::Error::InferenceError("Speech consumer disconnected".into())),
                _ = flush_due => {
                    if let Some(progress) = progress.as_deref_mut() {
                        if let Some(published) = progress.flush().await.map_err(progress_error)? {
                            send_published(&output, &mut sequence, published).await?;
                        }
                    }
                    continue;
                }
            };
            deadline = tokio::time::Instant::now() + stall_timeout;
            if let Some(stats) = &chunk.stats {
                if chunk.is_final {
                    segment_tokens = stats.tokens_generated;
                    segment_ms = stats.generation_time_ms;
                    segment_stop = stats.stop_reason;
                } else {
                    segment_tokens += stats.tokens_generated;
                    segment_ms += stats.generation_time_ms;
                }
            }
            if chunk.samples.is_empty() {
                continue;
            }
            let actual = chunk.sample_rate_or(state.runtime.sample_rate().await);
            if rate.is_some_and(|expected| expected != actual) {
                drop(receiver);
                task.abort();
                let _ = task.await;
                return Err(izwi_core::Error::InferenceError(
                    "Speech segment sample rate changed".into(),
                ));
            }
            rate = Some(actual);
            segment_duration += chunk.samples.len() as f32 / actual as f32;
            chunk.request_id = base.id.clone();
            chunk.is_final = false;
            chunk.stats = None;
            emitted = true;
            // Durable audio reaches listeners only once it is published.
            let ready = match progress.as_deref_mut() {
                Some(progress) => progress
                    .publish_chunk(chunk)
                    .await
                    .map_err(progress_error)?,
                None => vec![chunk],
            };
            for chunk in ready {
                if let Err(error) = send_published(&output, &mut sequence, chunk).await {
                    drop(receiver);
                    task.abort();
                    let _ = task.await;
                    return Err(error);
                }
            }
        }
        let result = task
            .await
            .map_err(|err| izwi_core::Error::InferenceError(err.to_string()))?;
        drop(permit);
        if let Err(error) = result {
            // Exact tokenized reference+text fit is checked inside admitted model
            // preparation. Replan only before this segment has published audio.
            if !emitted && error.kind() == izwi_core::ErrorKind::ContextLengthExceeded {
                let text = &base.text[range.clone()];
                let count = text.chars().count();
                if count > 1 {
                    let middle = range.start + text.char_indices().nth(count / 2).unwrap().0;
                    ranges.push_front(middle..range.end);
                    ranges.push_front(range.start..middle);
                    continue;
                }
            }
            return Err(error);
        }
        if !emitted || segment_tokens == 0 {
            return Err(izwi_core::Error::InferenceError(
                "Speech segment completed without audio or accounting".into(),
            ));
        }
        if let Some(progress) = progress.as_deref_mut() {
            if let Some(published) = progress.flush().await.map_err(progress_error)? {
                send_published(&output, &mut sequence, published).await?;
            }
            progress
                .complete_segment(range.end, segment_tokens, segment_ms, segment_duration)
                .await
                .map_err(progress_error)?;
        }
        duration += segment_duration;
        stop_reasons.push(segment_stop);
        remaining = remaining.saturating_sub(segment_tokens);
        tokens += segment_tokens;
        execution_ms += segment_ms;
        segment += 1;
    }
    let mut terminal =
        AudioChunk::final_chunk(base.id, sequence, Vec::new()).with_sample_rate(rate.unwrap_or(0));
    terminal.stats = Some(ChunkStats {
        tokens_generated: tokens,
        generation_time_ms: execution_ms,
        rtf: if duration > 0.0 {
            execution_ms / 1000.0 / duration
        } else {
            0.0
        },
        stop_reason: TtsStopReason::combine(stop_reasons),
    });
    output
        .send(terminal)
        .await
        .map_err(|_| izwi_core::Error::InferenceError("Speech consumer disconnected".into()))
}

/// Send one published chunk under the next output sequence.
async fn send_published(
    output: &tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    sequence: &mut usize,
    mut chunk: izwi_core::AudioChunk,
) -> Result<(), izwi_core::Error> {
    chunk.sequence = *sequence;
    output
        .send(chunk)
        .await
        .map_err(|_| izwi_core::Error::InferenceError("Speech consumer disconnected".into()))?;
    *sequence += 1;
    Ok(())
}

/// Dropping a timed-out/cancelled job also cancels its current inference future.
struct AbortSpeechTask(tokio::task::AbortHandle);
impl Drop for AbortSpeechTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPK-F12: segments run `concurrency` at a time, and their outputs come
    /// back in segment order however long each takes.
    #[tokio::test]
    async fn long_form_segments_run_concurrently_in_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let in_flight = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let segments: Vec<GenerationRequest> = (0..6)
            .map(|index| GenerationRequest::new(format!("segment {index}")))
            .collect();
        let outputs = generate_segments(segments, 3, false, "request", |segment| {
            let (in_flight, peak) = (&in_flight, &peak);
            async move {
                let now = in_flight.fetch_add(1, Ordering::AcqRel) + 1;
                peak.fetch_max(now, Ordering::AcqRel);
                let index: u64 = segment.text["segment ".len()..].parse().unwrap();
                // Later segments finish first.
                tokio::time::sleep(std::time::Duration::from_millis(60 - 10 * index)).await;
                in_flight.fetch_sub(1, Ordering::AcqRel);
                Ok(GenerationResult {
                    request_id: segment.text.clone(),
                    samples: vec![index as f32],
                    sample_rate: 24_000,
                    total_tokens: 1,
                    total_time_ms: 1.0,
                    diagnostics: None,
                    stop_reason: Some(TtsStopReason::EndOfSpeech),
                })
            }
        })
        .await
        .expect("segments");
        let order: Vec<f32> = outputs
            .iter()
            .map(|(output, _)| output.samples[0])
            .collect();
        assert_eq!(order, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(peak.load(Ordering::Acquire), 3, "three segments in flight");
    }

    /// A segment that ran to its budget is sampled again, and the retry is
    /// kept only if it ended speech.
    #[tokio::test]
    async fn runaway_segments_are_sampled_again() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let outputs = generate_segments(
            vec![GenerationRequest::new("a"), GenerationRequest::new("b")],
            2,
            true,
            "request",
            |segment| {
                let call = calls.fetch_add(1, Ordering::AcqRel);
                async move {
                    // "a" runs away on its first draw only.
                    let runaway = segment.text == "a" && call == 0;
                    Ok(GenerationResult {
                        request_id: segment.text,
                        samples: vec![call as f32],
                        sample_rate: 24_000,
                        total_tokens: 1,
                        total_time_ms: 1.0,
                        diagnostics: None,
                        stop_reason: Some(if runaway {
                            TtsStopReason::OutputBudget
                        } else {
                            TtsStopReason::EndOfSpeech
                        }),
                    })
                }
            },
        )
        .await
        .expect("segments");
        assert!(outputs[0].1, "the runaway segment was retried");
        assert!(!ran_to_budget(&outputs[0].0));
        assert!(!outputs[1].1);
    }

    #[test]
    fn fish_plan_preserves_utf8_text_and_short_fast_path() {
        for text in [
            "Hello. A short paragraph!".to_string(),
            "Dr. Smith paid $3.14. 中文🎵 no spaces 日本語\n".repeat(500),
            "字".repeat(4000),
        ] {
            let plan = SpeechTextPlan::fish(&text, 0).unwrap();
            let reconstructed: String = plan
                .segments
                .iter()
                .map(|range| &text[range.clone()])
                .collect();
            assert_eq!(reconstructed, text);
            assert!(plan
                .segments
                .iter()
                .all(|range| text[range.clone()].chars().count() <= 480));
            if text.len() < 480 {
                assert_eq!(plan.segments.len(), 1);
            }
        }
    }

    #[test]
    fn fish_plan_packs_short_sentences_and_prefers_sentence_boundaries() {
        let paragraph = "First sentence. Second sentence. Third sentence.";
        assert_eq!(
            SpeechTextPlan::fish(paragraph, 0).unwrap().segments.len(),
            1
        );
        let text = format!(
            "{} {}",
            "A reasonably sized sentence. ".repeat(14),
            "word ".repeat(60)
        );
        let plan = SpeechTextPlan::fish(&text, 0).unwrap();
        assert!(text[plan.segments[0].clone()].trim_end().ends_with('.'));
    }

    #[test]
    fn fish_plan_never_generates_blank_layout_segments() {
        let text = format!(
            "{}Hello.{}World.{}",
            " ".repeat(1000),
            "\n".repeat(1000),
            " ".repeat(1000)
        );
        let plan = SpeechTextPlan::fish(&text, 0).unwrap();
        let spoken: String = plan
            .segments
            .iter()
            .map(|range| &text[range.clone()])
            .collect();
        assert_eq!(
            spoken.split_whitespace().collect::<String>(),
            "Hello.World."
        );
        assert!(plan
            .segments
            .iter()
            .all(|range| !text[range.clone()].trim().is_empty()));
    }

    #[test]
    fn fish_long_text_splits_independently_of_output_budget() {
        let text = "This sentence must all be spoken. ".repeat(1000);
        let auto = SpeechTextPlan::fish(&text, 0).unwrap();
        let limited = SpeechTextPlan::fish(&text, 2584).unwrap();
        assert!(auto.segments.len() > 1);
        assert_eq!(auto.segments, limited.segments);
        assert_eq!(limited.max_output_frames, Some(2584));
    }

    /// TPIPE-2: an automatic budget never splits long form, so text longer
    /// than one request's 120 s cap was cut at the cap with a 200. Text one
    /// automatic request cannot finish now splits, each request budgeted for
    /// its own text; shorter text stays one request with its automatic
    /// budget; an explicit budget is kept.
    #[test]
    fn automatic_budget_splits_only_text_one_request_cannot_finish() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let cap =
            izwi_core::runtime_models::architectures::qwen3::tts::qwen3_tts_auto_max_output_frames(
            );
        let sentence = "The committee reviewed every proposal in detail before the vote. ";
        let long = sentence.repeat(30);
        assert!(crate::api::tts_policy::tts_text_exceeds_one_request(
            variant, &long
        ));
        let mut request = GenerationRequest::new(long.clone()).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, &long, None).unwrap();
        assert_eq!(request.config.options.max_tokens, cap);
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(request.config.options.max_tokens, 0);
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert!(planned.len() > 1);
        for chunk in &planned {
            let budget = chunk.config.options.max_tokens;
            assert!(budget > 0 && budget < cap, "chunk budget {budget}");
        }

        let short = sentence.repeat(3);
        let mut request = GenerationRequest::new(short.clone()).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, &short, None).unwrap();
        let automatic = request.config.options.max_tokens;
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(request.config.options.max_tokens, automatic);
        assert_eq!(
            expand_generation_requests_for_long_form(&request, variant).len(),
            1
        );

        // An explicit budget is the caller's: no deferral, no split.
        let mut request = GenerationRequest::new(long).with_model_variant(variant);
        request.config.options.max_tokens = 300;
        defer_automatic_budget_to_long_form(&mut request, variant, Some(300));
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].config.options.max_tokens, 300);
        // Families without automatic long form never split.
        assert!(!crate::api::tts_policy::tts_text_exceeds_one_request(
            ModelVariant::Kokoro82M,
            &sentence.repeat(30)
        ));
    }

    /// TPIPE-5: short sentences pack into one request instead of one request
    /// per sentence; text longer than one request splits between whole
    /// sentences, and every segment fits one request.
    #[test]
    fn qwen_auto_chunking_packs_whole_sentences_up_to_one_request() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let chunks = split_tts_text_for_long_form(
            variant,
            0,
            "Hello world. This is sentence two! Final line?",
        );
        assert_eq!(
            chunks,
            vec!["Hello world. This is sentence two! Final line?"]
        );
        let sentence = "Dr. Smith paid $3.50 at 10:30 for the long and winding trip home. ";
        let text = sentence.repeat(30);
        let chunks = split_tts_text_for_long_form(variant, 0, &text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.starts_with("Dr. Smith") && chunk.ends_with("home."),
                "{chunk}"
            );
            assert!(!crate::api::tts_policy::tts_text_exceeds_one_request(
                variant, chunk
            ));
        }
    }

    /// TPIPE-27: the 250-word scoring paragraph used to split into 480-char
    /// segments only because it exceeded one request, and one 460-char
    /// segment ran to its 663-frame budget and babbled. Qwen3-TTS segments
    /// are now whole sentences packed up to 300 characters, so it splits
    /// into short segments, each with its own automatic budget, while the
    /// 45-word medium paragraph (EOS on every backend) stays one request.
    #[test]
    fn qwen_long_form_speaks_short_sentence_segments() {
        let _env = crate::test_support::env_lock();
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let long = "The old lighthouse stood at the edge of the harbor for more than a century. \
            Every evening, the keeper climbed one hundred and twelve steps to light the lamp, \
            and every morning he climbed them again to put it out. Ships from distant ports \
            used its beam to find the narrow channel between the rocks. In winter, storms \
            pushed waves high enough to wash over the lower windows, and the keeper would sit \
            by the stove, listening to the wind and writing in his logbook. He recorded the \
            weather, the ships that passed, and the small events of each day. One entry \
            describes a fishing boat that lost its engine in a gale and drifted toward the \
            cliffs until the crew saw the light and rowed to safety. Another mentions a family \
            of seals that slept on the breakwater every spring. When the lighthouse was finally \
            automated, the logbooks were sent to the town library, where students still read \
            them today. They show how a single person, doing the same careful work every night, \
            kept hundreds of strangers safe. The building is now a small museum. Visitors can \
            climb the stairs, look out over the water, and imagine what it was like to watch \
            the sea alone for so many years. On clear nights, the automated lamp still turns, \
            sweeping its light across the dark water just as it always has.";
        let mut request = GenerationRequest::new(long).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, long, None).unwrap();
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(request.config.options.max_tokens, 0);
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert!(planned.len() >= 5, "{} segments", planned.len());
        for segment in &planned {
            assert!(
                segment.text.chars().count() <= QWEN_SEGMENT_MAX_CHARS,
                "{}",
                segment.text
            );
            assert!(segment.text.ends_with('.'), "{}", segment.text);
            let budget = segment.config.options.max_tokens;
            assert!(budget > 0 && budget < 663, "{budget}: {}", segment.text);
        }
        assert_eq!(
            planned
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            long.split_whitespace().collect::<Vec<_>>().join(" ")
        );

        let medium = "Good morning. Today we are testing a speech synthesis system on a laptop. \
            The goal is to measure how fast it runs, how much memory it uses, and whether \
            every word in this paragraph comes out clearly, without skipping, repeating, \
            or adding anything at the end.";
        let mut request = GenerationRequest::new(medium).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, medium, None).unwrap();
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(
            expand_generation_requests_for_long_form(&request, variant).len(),
            1
        );
    }

    /// A segment that runs to its budget is sampled once more only when a
    /// second draw can differ: unseeded, non-greedy Qwen3-TTS sampling.
    #[test]
    fn only_unseeded_sampled_qwen_segments_are_retried() {
        let qwen = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let request = |configure: &dyn Fn(&mut GenerationRequest)| {
            let mut request = GenerationRequest::new("text").with_model_variant(qwen);
            configure(&mut request);
            request
        };
        assert!(retries_runaway_segments(qwen, &request(&|_| {})));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.speech.seed = Some(7))
        ));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.speech.temperature = Some(0.0))
        ));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.speech.top_k = Some(1))
        ));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.max_tokens = 300)
        ));
        assert!(!retries_runaway_segments(
            ModelVariant::VibeVoice15BTts,
            &request(&|_| {})
        ));
    }

    /// Segments join with a faded tail, a 120 ms pause and a faded head, not
    /// back to back.
    #[test]
    fn segments_join_with_a_faded_pause() {
        let rate = 24_000;
        let mut joined = Vec::new();
        append_segment(&mut joined, &[1.0; 480], rate);
        assert_eq!(joined, vec![1.0; 480]);
        append_segment(&mut joined, &[1.0; 480], rate);
        let pause = 2_880;
        assert_eq!(joined.len(), 480 + pause + 480);
        // The first segment fades out over its last 10 ms (240 samples).
        assert_eq!(joined[239], 1.0);
        assert!(joined[479] > 0.0 && joined[479] < 0.01);
        assert!(joined[480..480 + pause].iter().all(|sample| *sample == 0.0));
        // The second fades in over its first 10 ms.
        assert!(joined[480 + pause] > 0.0 && joined[480 + pause] < 0.01);
        assert_eq!(*joined.last().unwrap(), 1.0);
    }

    #[test]
    fn explicit_small_token_budget_disables_long_form_split() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            256,
            "Sentence one. Sentence two.",
        );
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn non_qwen_variant_stays_single_chunk() {
        let chunks =
            split_tts_text_for_long_form(ModelVariant::Kokoro82M, 0, "Sentence one. Sentence two.");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn qwen_voice_design_and_base_pack_the_same_way() {
        for variant in [
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
            ModelVariant::Qwen3Tts12Hz06BBase,
        ] {
            let chunks =
                split_tts_text_for_long_form(variant, 0, &"One sentence here. ".repeat(80));
            assert!(chunks.len() > 1, "{variant}");
            assert!(chunks.iter().all(|chunk| chunk.ends_with("here.")));
        }
    }
}
