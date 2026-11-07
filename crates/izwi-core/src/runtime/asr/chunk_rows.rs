//! A long file transcribed as concurrent chunk rows.
//!
//! A long file used to run its chunks one after another inside one request
//! (some families inside one dispatch that holds the device), so a file never
//! used the decode batch width and other users waited behind it: on an H100,
//! four concurrent 10-minute files gained 13% over one (Qwen3-ASR 32 → 35
//! audio s/s). Here the runtime decodes and plans the file once, as the
//! request route would, and submits each chunk's samples as an ordinary
//! single-pass request (a chunk row), up to the ASR row width at once. Rows
//! batch with each other and with other users' requests and may finish in
//! any order; their texts commit in plan order through the sequential chunk
//! route's merge, so the transcript, word timings and per-chunk diagnostics
//! are built alike. Text streams a committed chunk at a time, and dropping
//! the parent cancels every row in flight. Rows are internal: the parent
//! counts once in the request metrics.

use std::future::Future;
use std::time::Instant;

use futures::StreamExt;
use serde_json::json;

use super::speaker_attribution::samples_to_seconds_f64;
use super::*;
use crate::engine::{
    timestamp_words_json, AsrChunkTranscription, AsrRouteCapabilities, AudioOutput, ChunkedAsrRun,
    ChunkedAsrTranscription, ExecutorOutput, LatencyBreakdown, NativeExecutor, OutputProcessor,
    PlannedAsrChunks, WorkloadClass,
};
use crate::runtime::asr_no_speech::AsrNoSpeechGate;
use crate::runtime::audio_io::{wav_duration_seconds_fast, CANONICAL_SPEECH_SAMPLE_RATE};

/// Kill switch for chunk rows (on by default), read once per process; off,
/// a long file's chunks run one after another in one request.
pub(crate) const LONG_FORM_FANOUT_ENV: &str = "IZWI_ASR_LONG_FORM_FANOUT";

fn long_form_fanout_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(LONG_FORM_FANOUT_ENV, true));
    ENABLED.get()
}

/// Chunk rows in flight across every file at once: two batches of ASR rows
/// (one decoding while the next encodes or queues), and never more than half
/// the coordinator's queue, so files fanned out side by side cannot fill it
/// and refuse other requests. Each file keeps at most one batch of its own
/// in flight; past the shared limit its rows wait for a free place.
pub(crate) fn chunk_row_permit_count(row_width: usize, max_queued_requests: usize) -> usize {
    row_width
        .max(1)
        .saturating_mul(2)
        .min(max_queued_requests / 2)
        .max(1)
}

/// Families whose single-pass request is their ordinary short-clip route,
/// so a chunk row runs exactly as a short clip does and transcribes its
/// audio with the model call the sequential chunk route makes:
/// - Qwen3-ASR and Whisper: a batched encoder, then a retained decode row;
/// - Nemotron ASR: a batched encoder and batched RNNT rows;
/// - Voxtral: one offline invocation per row, for fairness only (other users
///   wait at most one chunk instead of a whole file).
///
/// Not the others: Parakeet, VibeVoice and Nemotron 3.5 Transcribe cover a
/// long file in one pass; Granite 4.1 can replay earlier text into a chunk
/// and Granite 5.0 re-decodes chunks, both until their continuous decode;
/// the LFM audio models run long files on their own executors.
fn chunk_rows_family(family: ModelFamily) -> bool {
    matches!(
        family,
        ModelFamily::Qwen3Asr
            | ModelFamily::WhisperAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::Voxtral
    )
}

/// Whether a long `variant` transcription for `task` runs as chunk rows with
/// the switch `enabled`. A speaker-attributed or translated transcript keeps
/// its own route.
fn chunk_rows_apply(variant: ModelVariant, task: &AsrTaskOptions, enabled: bool) -> bool {
    enabled
        && chunk_rows_family(variant.family())
        && !task.speaker_attribution
        && task.translate_to.is_none()
}

/// The parent request as the request route would build it, without its
/// audio: route planning reads its decode budget and timestamp option, and
/// the request metrics its identity and scheduling class.
#[allow(clippy::too_many_arguments)]
fn chunk_rows_parent_request(
    variant: ModelVariant,
    task: &AsrTaskOptions,
    max_tokens: Option<usize>,
    correlation_id: Option<&str>,
    runtime_context: RuntimeRequestContext,
    streaming: bool,
    native_word_timestamps: bool,
) -> EngineCoreRequest {
    let mut request = EngineCoreRequest::asr_bytes(Vec::new());
    request.model_variant = Some(variant);
    request.correlation_id = correlation_id.map(str::to_string);
    request.asr_task = task.clone();
    request.asr_native_word_timestamps = native_word_timestamps;
    apply_asr_max_tokens(&mut request, variant, max_tokens);
    request.streaming = streaming;
    request.workload_class = if streaming && runtime_context.workload_class == WorkloadClass::Online
    {
        WorkloadClass::Streaming
    } else {
        runtime_context.workload_class
    };
    request.tenant_key = runtime_context.tenant_key;
    request.priority = runtime_context.priority;
    request.admission_ms = runtime_context.admission_ms;
    request.deadline = runtime_context.deadline;
    request
}

/// What one row's engine work adds to its parent's request metrics.
#[derive(Debug, Clone, Copy, Default)]
struct ChunkRowWork {
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    /// When the engine first scheduled the row, in milliseconds since the
    /// parent started.
    scheduled_ms: Option<f64>,
}

/// One transcribed chunk row.
struct ChunkRow {
    text: String,
    diagnostics: Option<serde_json::Value>,
    /// The row's time from submission to its output.
    transcribe_ms: f64,
    work: ChunkRowWork,
}

impl ChunkRow {
    /// A row's output, submitted `submitted_ms` and finished `finished_ms`
    /// after its parent started.
    fn from_output(output: EngineOutput, submitted_ms: f64, finished_ms: f64) -> Self {
        let latency = output.latency_breakdown.as_ref();
        Self {
            text: output.text.unwrap_or_default(),
            diagnostics: output.asr_diagnostics,
            transcribe_ms: finished_ms - submitted_ms,
            work: ChunkRowWork {
                prompt_tokens: output.token_stats.prompt_tokens,
                generated_tokens: output.token_stats.generated_tokens,
                prefill_ms: latency.map_or(0.0, |latency| latency.prefill_ms),
                decode_ms: latency.map_or(0.0, |latency| latency.decode_ms),
                scheduled_ms: latency.map(|latency| {
                    submitted_ms + latency.pre_engine_ms.unwrap_or(0.0) + latency.queue_wait_ms
                }),
            },
        }
    }
}

/// A chunk plan run as rows: its transcript, each row's engine work, and
/// when the first chunk committed (milliseconds since `started`).
struct ChunkRowsRun {
    transcription: ChunkedAsrTranscription,
    work: Vec<ChunkRowWork>,
    first_commit_ms: Option<f64>,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// Run `plan`'s chunks as rows: `submit` starts chunk `index`'s row, and
/// `rows_in_flight` run at once (a finished row frees its place even while
/// an earlier one runs). Their texts commit in plan order however they
/// finish; deltas (when `stream_holdback` is set) and chunk progress follow
/// the commits. A failed row fails the run at once, and dropping the run's
/// future drops, and so cancels, every row in flight.
async fn run_chunk_rows<S, Fut, F, P>(
    plan: &PlannedAsrChunks,
    rows_in_flight: usize,
    stream_holdback: Option<usize>,
    started: Instant,
    mut submit: S,
    on_delta: &mut F,
    on_progress: &mut P,
) -> Result<ChunkRowsRun>
where
    S: FnMut(usize, AudioChunk) -> Fut,
    Fut: Future<Output = Result<ChunkRow>>,
    F: FnMut(String),
    P: FnMut(AsrProgress),
{
    let chunks = plan.chunks.as_slice();
    let (input_samples, sample_rate) = (plan.input_samples, plan.sample_rate);
    let mut run = ChunkedAsrRun::new(&plan.config);
    on_progress(NativeExecutor::asr_processing_progress(chunks, sample_rate));
    let mut rows = futures::stream::iter(chunks.iter().copied().enumerate())
        .map(|(index, chunk)| {
            // A chunk outside the audio is skipped, as the sequential run
            // skips it.
            let row =
                ChunkedAsrRun::chunk_in_bounds(&chunk, input_samples).then(|| submit(index, chunk));
            async move {
                let row = match row {
                    Some(row) => row.await.map(Some),
                    None => Ok(None),
                };
                (index, row)
            }
        })
        .buffer_unordered(rows_in_flight.max(1));
    // Rows that finished before an earlier one, waiting for their turn.
    let mut finished = std::collections::BTreeMap::new();
    let mut work = Vec::with_capacity(chunks.len());
    let mut first_commit_ms = None;
    let mut index = 0;
    while let Some((finished_index, row)) = rows.next().await {
        finished.insert(finished_index, row?);
        while let Some(row) = finished.remove(&index) {
            let chunk = &chunks[index];
            let (transcription, transcribe_ms) = match row {
                Some(row) => {
                    for phase in [
                        AsrProgressPhase::ChunkStarted,
                        AsrProgressPhase::ChunkFinished,
                    ] {
                        on_progress(NativeExecutor::asr_chunk_progress(
                            phase,
                            index,
                            chunk,
                            chunks,
                            sample_rate,
                        ));
                    }
                    work.push(row.work);
                    let transcription = AsrChunkTranscription {
                        text: row.text,
                        diagnostics: row.diagnostics,
                    };
                    (transcription, row.transcribe_ms)
                }
                None => (
                    AsrChunkTranscription {
                        text: String::new(),
                        diagnostics: None,
                    },
                    0.0,
                ),
            };
            if let Some(delta) = run.commit_chunk(
                chunks,
                input_samples,
                sample_rate,
                transcription,
                transcribe_ms,
                stream_holdback,
            ) {
                on_delta(delta);
            }
            first_commit_ms.get_or_insert_with(|| elapsed_ms(started));
            index += 1;
        }
    }
    if stream_holdback.is_some() {
        let tail = run.final_delta();
        if !tail.is_empty() {
            on_delta(tail);
        }
    }
    on_progress(NativeExecutor::asr_complete_progress(chunks, sample_rate));
    Ok(ChunkRowsRun {
        transcription: run.into_transcription(),
        work,
        first_commit_ms,
    })
}

/// The parent's terminal output for its request metrics: the whole file's
/// transcript and duration, its rows' tokens and engine time, queued until
/// its first row was scheduled and first answering when its first chunk
/// committed.
fn chunk_rows_parent_output(
    request_id: &str,
    transcription: &AsrTranscription,
    work: &[ChunkRowWork],
    first_commit_ms: Option<f64>,
    audio_decode_ms: f64,
    total_ms: f64,
) -> EngineOutput {
    let mut output = OutputProcessor::new(CANONICAL_SPEECH_SAMPLE_RATE).process(
        ExecutorOutput {
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate: CANONICAL_SPEECH_SAMPLE_RATE,
                duration_secs: transcription.duration_secs,
                streamed_samples: None,
            }),
            text: Some(transcription.text.clone()),
            ..ExecutorOutput::terminal(request_id.to_string())
        },
        0,
        std::time::Duration::from_secs_f64(total_ms.max(0.0) / 1000.0),
    );
    let prefill_ms = work.iter().map(|row| row.prefill_ms).sum::<f64>();
    let decode_ms = work.iter().map(|row| row.decode_ms).sum::<f64>();
    output.token_stats.prompt_tokens = work.iter().map(|row| row.prompt_tokens).sum();
    output.token_stats.generated_tokens = work.iter().map(|row| row.generated_tokens).sum();
    output.token_stats.prefill_time_ms = prefill_ms as f32;
    output.token_stats.decode_time_ms = decode_ms as f32;
    output.num_tokens = output.token_stats.generated_tokens;
    output.input_audio_duration_secs = Some(transcription.duration_secs);
    output.latency_breakdown = Some(LatencyBreakdown {
        pre_engine_ms: Some(0.0),
        queue_wait_ms: work
            .iter()
            .filter_map(|row| row.scheduled_ms)
            .reduce(f64::min)
            .unwrap_or(0.0),
        media_decode_ms: Some(audio_decode_ms),
        normalization_ms: None,
        prefill_ms,
        decode_ms,
        decode_wall_ms: None,
        decode_tokens: None,
        post_first_token_ms: None,
        sampling_ms: None,
        codec_ms: None,
        postprocess_ms: None,
        ttft_ms: first_commit_ms,
        total_ms,
        prefill_steps: 0,
        decode_steps: 0,
    });
    output
}

impl RuntimeService {
    /// Route capabilities of `variant` for `request`, as its route preparation
    /// plans with them, or `None` when the model is not resident.
    async fn chunk_row_capabilities(
        &self,
        variant: ModelVariant,
        request: &EngineCoreRequest,
    ) -> Option<AsrRouteCapabilities> {
        let family = variant.family();
        if family == ModelFamily::Voxtral {
            return self
                .model_registry
                .try_get_voxtral_lease(variant)
                .map(|model| model.asr_route_capabilities());
        }
        let model = self.model_registry.get_asr_lease(variant).await?;
        let effective_context = self.model_registry.effective_context(variant);
        // Qwen3-ASR and Whisper preparation refuse a model without one.
        if effective_context.is_none()
            && matches!(family, ModelFamily::Qwen3Asr | ModelFamily::WhisperAsr)
        {
            return None;
        }
        AsrRouteCapabilities::for_native_model(&model, variant, request, effective_context).ok()
    }

    /// A long transcription run as chunk rows (see the module docs), or
    /// `None` when it runs as one request: its family or task keeps its own
    /// route, the switch is off, its model is not resident, its audio fits one
    /// pass or holds no speech. Every case that returns `None` leaves the
    /// request, its errors and its metrics to the request route.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn transcribe_in_chunk_rows<F, P>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        task: &AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        streaming: bool,
        native_word_timestamps: bool,
        on_delta: &mut F,
        on_progress: &mut P,
    ) -> Result<Option<AsrTranscription>>
    where
        F: FnMut(String) + Send,
        P: FnMut(AsrProgress) + Send,
    {
        if !chunk_rows_apply(variant, task, long_form_fanout_enabled()) {
            return Ok(None);
        }
        let started = Instant::now();
        // Rows plan against the resident model; the lease keeps it resident
        // for them. A model still loading takes the request route.
        let Some(residency) = self.model_lifecycle.try_acquire_ready_lease(variant) else {
            return Ok(None);
        };
        let parent = chunk_rows_parent_request(
            variant,
            task,
            max_tokens,
            correlation_id,
            runtime_context,
            streaming,
            native_word_timestamps,
        );
        let Some(capabilities) = self.chunk_row_capabilities(variant, &parent).await else {
            return Ok(None);
        };
        // A WAV header that fits one pass needs no decode to know it.
        let Some(single_pass_secs) = capabilities.single_pass_limit_secs() else {
            return Ok(None);
        };
        if wav_duration_seconds_fast(audio_bytes).is_some_and(|secs| secs <= single_pass_secs) {
            return Ok(None);
        }
        validate_asr_language(variant, language)?;
        validate_asr_task(variant, language, prompt, task)?;
        AsrAudioInput::Bytes(audio_bytes).validate_retained_size()?;
        let source = audio_bytes.to_vec();
        let gate = variant.family().hallucinates_on_non_speech();
        let planned = tokio::task::spawn_blocking(move || {
            let decode_started = Instant::now();
            // A decode failure is the request route's to report.
            let (samples, sample_rate) = decode_audio_bytes_canonical(&source).ok()?;
            let audio_decode_ms = elapsed_ms(decode_started);
            let plan =
                crate::engine::asr_route_plan(&samples, sample_rate, &capabilities, streaming);
            // A whole input without speech is the request route's no-speech
            // gate to answer.
            if plan.chunks.len() < 2
                || (gate && AsrNoSpeechGate::evaluate(&samples, sample_rate).is_some())
            {
                return None;
            }
            Some((samples, plan, audio_decode_ms))
        })
        .await
        .map_err(|error| Error::InferenceError(format!("chunk row planning stopped: {error}")))?;
        let Some((samples, plan, audio_decode_ms)) = planned else {
            return Ok(None);
        };
        let rows_in_flight = self.adapter_registry.max_tensor_batch_size().max(1);
        let sample_rate = plan.sample_rate;
        let duration_secs = samples_to_seconds_f64(samples.len(), sample_rate);
        tracing::info!(
            %variant,
            chunks = plan.chunks.len(),
            rows_in_flight,
            audio_secs = duration_secs,
            "transcribing a long file as chunk rows"
        );
        let metrics = self
            .begin_composite_request_metrics(&parent, streaming)
            .await?;
        let samples: Arc<[f32]> = samples.into();
        let parent_id = parent.id.as_str();
        let submit = |_index: usize, chunk: AudioChunk| {
            let samples = samples.clone();
            let task = task.clone();
            async move {
                let permit = self.asr_chunk_row_permits.clone().acquire_owned();
                let _permit = match runtime_context.deadline {
                    Some(deadline) => tokio::time::timeout_at(deadline.into(), permit)
                        .await
                        .map_err(|_| Error::Timeout(parent_id.to_string()))?,
                    None => permit.await,
                }
                .map_err(|_| Error::Overloaded("ASR chunk rows are closed".to_string()))?;
                let submitted_ms = elapsed_ms(started);
                // The chunk's own decoded samples: a row neither encodes nor
                // decodes its audio.
                let chunk_samples: Arc<[f32]> =
                    samples[chunk.start_sample..chunk.end_sample].into();
                drop(samples);
                let admitted = self
                    .build_asr_request(
                        variant,
                        AsrAudioInput::Samples(&chunk_samples),
                        language,
                        prompt,
                        task,
                        max_tokens,
                        correlation_id,
                        runtime_context,
                        false,
                        native_word_timestamps,
                    )
                    .await?;
                drop(chunk_samples);
                let output = self
                    .run_admitted_request(admitted.into_asr_chunk_row())
                    .await?;
                Ok(ChunkRow::from_output(
                    output,
                    submitted_ms,
                    elapsed_ms(started),
                ))
            }
        };
        let stream_holdback = streaming
            .then(|| NativeExecutor::asr_chunk_stream_options().stable_text_holdback_chars);
        let mut streamed = String::new();
        let mut emit = |delta: String| {
            streamed.push_str(&delta);
            on_delta(delta);
        };
        let ChunkRowsRun {
            transcription: chunked,
            work,
            first_commit_ms,
        } = match run_chunk_rows(
            &plan,
            rows_in_flight,
            stream_holdback,
            started,
            submit,
            &mut emit,
            on_progress,
        )
        .await
        {
            Ok(rows) => rows,
            Err(error) => {
                metrics.fail(&error).await;
                return Err(error);
            }
        };
        // The diagnostics of the sequential chunk route, which also records
        // how many rows ran at once.
        let mut diagnostics = plan.diagnostics_with_chunk_transcriptions(chunked.chunk_diagnostics);
        if let Some(chunking) = diagnostics
            .get_mut("chunking")
            .and_then(serde_json::Value::as_object_mut)
        {
            chunking.insert("rows_in_flight".to_string(), json!(rows_in_flight));
        }
        if let Some(root) = diagnostics.as_object_mut() {
            if let Some(words) = chunked.timestamp_words.as_deref() {
                root.insert("timestamp_words".to_string(), timestamp_words_json(words));
            }
            root.insert(
                "timings_ms".to_string(),
                json!({ "audio_decode": audio_decode_ms }),
            );
        }
        let text = asr_transcript_text(&chunked.text);
        if streaming {
            if let Some(tail) = unstreamed_transcript_tail(&streamed, &text) {
                on_delta(tail.to_string());
            }
        }
        let transcription = AsrTranscription {
            text,
            language: asr_response_language(language, None, Some(&diagnostics)),
            duration_secs: duration_secs as f32,
            asr_diagnostics: Some(diagnostics),
        };
        let output = chunk_rows_parent_output(
            &parent.id,
            &transcription,
            &work,
            first_commit_ms,
            audio_decode_ms,
            elapsed_ms(started),
        );
        metrics.complete(&output).await;
        drop(residency);
        Ok(Some(transcription))
    }
}

#[cfg(test)]
#[path = "chunk_rows_tests.rs"]
mod tests;
