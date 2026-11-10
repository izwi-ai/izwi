#[path = "durable.rs"]
pub(crate) mod durable;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{Extension, Json, Path, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::audio_payload::decode_base64_audio_payload;
use crate::api::idempotency::{
    begin_durable_idempotency, extract_idempotency_key, release_durable_idempotency,
    replay_record_id, reservation_lost, DurableIdempotencyStart, DURABLE_IDEMPOTENCY_RETENTION_MS,
};
use crate::api::pagination::{encode_cursor, CursorPagination, CursorPaginationQuery};
use crate::api::request_context::RequestContext;
use crate::api::saved_voices::resolve_saved_voice_reference;
use crate::api::tts_long_form::{expand_generation_requests_for_long_form, generate_long_form_tts};
#[cfg(test)]
use crate::api::tts_policy::qwen_tts_auto_max_frames_for_text;
use crate::api::tts_policy::resolve_tts_output_frames;
use crate::batch_runtime::{
    store::{
        canonical_request_digest, sha256_hex, DurableIdempotencyReplay, DurableIdempotencyRequest,
        DurableIdempotencyReservation, DurableJobAcceptanceOutcome,
        DurableTextTtsAcceptanceOutcome, NewDurableJobAcceptance, NewDurableJobArtifact,
        NewDurableJobStage, NewDurableTextInput, NewDurableTextTtsAcceptance, NewMediaAsset,
        NewRuntimeArtifact, NewRuntimeJob, NewStageOutputArtifact, NewTextAsset,
        DURABLE_IDEMPOTENCY_DIGEST_VERSION, MAX_DURABLE_TTS_TEXT_BYTES,
    },
    types::{
        ClaimedStage, QueueClass, RuntimeArtifactKind, RuntimeArtifactRole, RuntimeJobKind,
        RuntimeJobStatus, StageResourceHints,
    },
    worker::{
        StageCancellationReason, StageCancellationSignal, StageExecutionContext,
        StageExecutionOutcome, StageExecutor,
    },
};
use crate::entity::RuntimeProjectionAttempt;
use crate::error::{core_api_error, ApiError, ApiJson};
use crate::ids::new_uuid;
use crate::media_ingest::{
    AudioIngestPolicy, CanonicalAudioProfile, ExistingAudioIngestRequest, MediaIngestError,
    MediaIngestRequest, MediaIngestResult,
};
use crate::speech_history_store::{
    CompleteSpeechHistoryRecord, NewSpeechHistoryRecord, SpeechHistoryListFilter,
    SpeechHistoryProcessingStatus, SpeechHistoryRecord, SpeechHistoryRecordListCursor,
    SpeechHistoryRecordSummary, SpeechHistoryVoiceKind, SpeechRouteKind, SpeechWavSpool,
    StoredSpeechAudio,
};
use crate::state::AppState;
use izwi_core::audio::{inspect_audio_bytes, AudioEncoder, AudioFormat};
#[cfg(test)]
use izwi_core::runtime_models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
#[cfg(test)]
use izwi_core::runtime_models::architectures::voxtral::tts::voxtral_tts_auto_max_frames_for_text;
use izwi_core::{
    parse_tts_model_variant, AudioChunk, GenerationConfig, GenerationRequest, ModelVariant,
    TtsStopReason, WorkloadClass,
};
use izwi_hooks::Principal;

const HISTORY_LIST_LIMIT: usize = 200;
const STREAM_CLIENT_DISCONNECTED_MESSAGE: &str = "Streaming client disconnected before completion";

/// A task handle that aborts its task when dropped, so a cancelled stream
/// does not leave its generation running detached.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}
pub(crate) const BATCH_TTS_STAGE_KIND: &str = "tts_synthesize";
const REFERENCE_AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const DURABLE_TTS_IDEMPOTENCY_OPERATION: &str = "speech.text_to_speech.create";
const DURABLE_TTS_IDEMPOTENCY_RESERVATION_TTL_MS: u64 = 60 * 1_000;
/// Reference audio is ingested while the key is reserved.
const DURABLE_REFERENCE_TTS_IDEMPOTENCY_RESERVATION_TTL_MS: u64 = 5 * 60 * 1_000;

#[derive(Debug, Deserialize, Default)]
pub(crate) struct RecordAudioQuery {
    #[serde(default)]
    download: bool,
    /// Transcode the stored WAV: `mp3`, `opus`, `flac`, `pcm_i16`, ... (the
    /// `/v1/audio/speech` names). Unset, the audio is sent as stored.
    #[serde(default)]
    format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpeechHistoryRecordListResponse {
    pub records: Vec<SpeechHistoryRecordSummary>,
    pub pagination: CursorPagination,
    /// Records matching the filters, ignoring paging.
    pub total: usize,
}

/// Paging plus optional filters for history lists.
#[derive(Debug, Default, Deserialize)]
pub struct SpeechHistoryListParams {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub cursor: Option<String>,
    /// Case-insensitive search over the input text and speaker name.
    #[serde(default)]
    pub q: Option<String>,
    /// `builtin` or `saved`.
    #[serde(default)]
    pub voice_kind: Option<String>,
    /// Comma-separated statuses: pending, processing, ready, failed.
    #[serde(default)]
    pub status: Option<String>,
    /// Unix milliseconds; only records created at or after this time.
    #[serde(default)]
    pub created_after: Option<u64>,
}

impl SpeechHistoryListParams {
    fn filter(&self) -> Result<SpeechHistoryListFilter, ApiError> {
        let voice_kind = match self.voice_kind.as_deref().map(str::trim) {
            None | Some("") => None,
            Some("builtin") | Some("built_in") => Some(SpeechHistoryVoiceKind::BuiltIn),
            Some("saved") => Some(SpeechHistoryVoiceKind::Saved),
            Some(other) => {
                return Err(ApiError::bad_request(format!(
                    "Invalid `voice_kind` {other:?}. Supported values: builtin, saved."
                )))
            }
        };
        let statuses = self
            .status
            .as_deref()
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|status| !status.is_empty())
            .map(|status| {
                SpeechHistoryProcessingStatus::from_db_value(status).ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "Invalid `status` {status:?}. Supported values: pending, processing, ready, failed."
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SpeechHistoryListFilter {
            search: self.q.clone(),
            voice_kind,
            statuses,
            created_after: self.created_after,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct DeleteSpeechHistoryRecordResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct CancelSpeechHistoryRecordResponse {
    pub id: String,
    pub cancelled: bool,
    pub record: SpeechHistoryRecord,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CreateSpeechHistoryRecordRequest {
    #[serde(default, alias = "model")]
    pub model_id: Option<String>,
    #[serde(default, alias = "input")]
    pub text: Option<String>,
    #[serde(default, alias = "voice")]
    pub speaker: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default, alias = "instructions")]
    pub voice_description: Option<String>,
    #[serde(default)]
    pub reference_audio: Option<String>,
    #[serde(default)]
    pub reference_text: Option<String>,
    /// The language the reference audio is spoken in, for models that clone
    /// across languages (Maya 2 Native); absent, the model infers it.
    #[serde(default)]
    pub reference_language: Option<String>,
    #[serde(default)]
    pub saved_voice_id: Option<String>,
    /// Reference voices per speaker, for models that voice several speakers
    /// (Fish Audio Drama 3); replaces `reference_audio`, `reference_text`
    /// and `saved_voice_id`. Absent from stored requests without them, so
    /// their idempotency digests are unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<crate::api::speech_dialogue::SpeechDialogueReference>,
    /// A dialogue as `{speaker, text}` lines, spoken as `<|speaker:N|>
    /// text` lines; replaces `text`, and the record stores the rendered
    /// lines as its text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<Vec<crate::api::speech_dialogue::SpeechScriptLine>>,
    /// Earlier takes this one continues (Fish Audio Drama 3 take editing):
    /// records of this route, oldest first, at most two. The prompt holds
    /// their kept chunks as completed turns before the text, and the new
    /// record's `parent_take_id` is the last. Absent from stored requests
    /// without them, so their idempotency digests are unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previous_take_ids: Vec<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// izwi speech-sampling extensions; unset keeps the family default.
    #[serde(flatten)]
    pub sampling: crate::api::tts_policy::SpeechSamplingRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchSpeechRequest {
    /// Server-generated scheduler identity; this envelope is never accepted
    /// from an HTTP client. Missing identity preserves legacy anonymous jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tenant_key: Option<[u8; 32]>,
    route_kind: SpeechRouteKind,
    model_id: String,
    input_text: String,
    request: CreateSpeechHistoryRecordRequest,
}

impl BatchSpeechRequest {
    fn for_durable_job(
        route_kind: SpeechRouteKind,
        model_id: String,
        input_text: String,
        mut request: CreateSpeechHistoryRecordRequest,
    ) -> Self {
        // The durable envelope owns these canonical execution fields. Keeping
        // copies in the public request doubles large text in storage and makes
        // the advertised text bound untruthful.
        request.model_id = None;
        request.text = None;
        request.script = None;
        request.reference_audio = None;
        for reference in &mut request.references {
            reference.audio = None;
        }
        Self {
            tenant_key: None,
            route_kind,
            model_id,
            input_text,
            request,
        }
    }
}

#[derive(Debug, Serialize)]
struct SpeechStreamEvent {
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sequence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_generated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_time_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rtf: Option<f32>,
    /// On `final`: why generation ended (`stop`, `length` or
    /// `context_length`), when a frame budget applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    record: Option<SpeechHistoryRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Per constituent request: terminal totals replace earlier deltas, even when
/// the final marker contains no audio. Long-form requests sum these totals.
#[derive(Default)]
struct StreamRequestStatistics {
    measured: bool,
    tokens: usize,
    execution_ms: f32,
    stop_reason: Option<TtsStopReason>,
}

impl StreamRequestStatistics {
    fn observe(&mut self, chunk: &AudioChunk) {
        if let Some(stats) = chunk.stats.as_ref() {
            self.measured = true;
            if chunk.is_final {
                self.tokens = stats.tokens_generated;
                self.execution_ms = stats.generation_time_ms;
                self.stop_reason = stats.stop_reason;
            } else {
                self.tokens = self.tokens.saturating_add(stats.tokens_generated);
                self.execution_ms += stats.generation_time_ms;
            }
        }
    }
}

/// Bound final provider upload memory using the existing generation allowance,
/// including every long-form constituent and the largest supported slow-down.
/// Models without a catalog duration bound use a 256 MiB fallback. Operators can
/// deliberately impose a tighter limit with IZWI_TTS_STREAM_MAX_PCM_BYTES.
fn stream_pcm_byte_allowance(
    variant: ModelVariant,
    sample_rate: u32,
    request_count: usize,
) -> usize {
    let limit = match (
        variant.tts_max_output_frames_hint(),
        variant.tts_output_frame_rate_hz_hint(),
    ) {
        (Some(frames), Some(rate)) if rate > 0.0 => {
            let frames = if variant == ModelVariant::Voxtral4BTts2603 {
                frames.max(ModelVariant::VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES)
            } else {
                frames
            };
            // One frame of rounding/codec boundary slack per constituent.
            (((frames + 1) as f64 / f64::from(rate))
                * f64::from(sample_rate)
                * 2.0
                * 4.0
                * request_count.max(1) as f64)
                .ceil() as usize
        }
        _ => 256 * 1024 * 1024,
    };
    limit.min(u32::MAX as usize - 44)
}

fn stream_pcm_byte_limit(variant: ModelVariant, sample_rate: u32, request_count: usize) -> usize {
    let limit = stream_pcm_byte_allowance(variant, sample_rate, request_count);
    crate::env_config::tuning()
        .tts_stream_max_pcm_bytes
        .map_or(limit, |configured| configured.min(limit))
        .min(u32::MAX as usize - 44)
}

fn stream_event_queue_capacity() -> usize {
    crate::env_config::tuning().audio_stream_event_queue_capacity
}

struct BudgetedStreamEvent {
    payload: String,
    _global_reservation: crate::speech_resource_budget::ByteReservation,
    _stream_reservation: crate::speech_resource_budget::ByteReservation,
}

impl AsRef<[u8]> for BudgetedStreamEvent {
    fn as_ref(&self) -> &[u8] {
        self.payload.as_bytes()
    }
}

// Model loading, reference preparation and prefill can all precede the first
// PCM chunk. Comments keep SSE intermediaries alive without inventing audio or
// progress events, and are produced on demand rather than entering the queue.
fn speech_event_stream(
    mut event_rx: mpsc::Receiver<BudgetedStreamEvent>,
) -> impl futures::Stream<Item = Result<bytes::Bytes, Infallible>> {
    async_stream::stream! {
        loop {
            match tokio::time::timeout(Duration::from_secs(10), event_rx.recv()).await {
                Ok(Some(payload)) => {
                    // Ownership follows the transport frame and every clone.
                    yield Ok(bytes::Bytes::from_owner(payload));
                }
                Ok(None) => break,
                Err(_) => yield Ok(bytes::Bytes::from_static(b": keepalive\n\n")),
            }
        }
    }
}

fn speech_stream_response(event_rx: mpsc::Receiver<BudgetedStreamEvent>) -> Response {
    let stream = speech_event_stream(event_rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .expect("static speech stream response")
}

/// What a speech stream waits on next: the model's next chunk, the end of
/// its output, or the client leaving.
enum StreamInput {
    Chunk(AudioChunk),
    Finished,
    Disconnected,
}

async fn next_stream_input(
    chunk_rx: &mut mpsc::Receiver<AudioChunk>,
    events: &mpsc::Sender<BudgetedStreamEvent>,
) -> StreamInput {
    tokio::select! {
        chunk = chunk_rx.recv() => chunk.map_or(StreamInput::Finished, StreamInput::Chunk),
        _ = events.closed() => StreamInput::Disconnected,
    }
}

#[derive(Clone)]
struct StreamEventSender {
    sender: mpsc::Sender<BudgetedStreamEvent>,
    budget: Arc<crate::speech_resource_budget::ByteBudget>,
}

async fn send_stream_event(
    event_tx: &StreamEventSender,
    event: SpeechStreamEvent,
) -> Result<(), &'static str> {
    let payload = format!("data: {}\n\n", to_stream_json(event));
    let global_reservation = crate::speech_resource_budget::event_budget()
        .reserve(payload.len())
        .map_err(|_| "Speech stream output byte capacity exhausted")?;
    let stream_reservation = event_tx
        .budget
        .reserve(payload.len())
        .map_err(|_| "Speech stream output byte capacity exhausted")?;
    let event = BudgetedStreamEvent {
        payload,
        _global_reservation: global_reservation,
        _stream_reservation: stream_reservation,
    };
    tokio::time::timeout(
        crate::env_config::tuning().audio_stream_stall_timeout,
        event_tx.sender.send(event),
    )
    .await
    .map_err(|_| "Speech stream client stalled beyond its send deadline")?
    .map_err(|_| STREAM_CLIENT_DISCONNECTED_MESSAGE)
}

// Bind playback and persisted WAV to the first actual PCM rate. The runtime's
// default codec rate may differ from the selected model (Fish emits 44.1 kHz).
async fn prepare_stream_pcm(
    event_tx: &StreamEventSender,
    request_id: &str,
    chunk: &AudioChunk,
    fallback_sample_rate: u32,
    merged_sample_rate: &mut Option<u32>,
) -> Result<Option<u32>, String> {
    if chunk.samples.is_empty() {
        return Ok(None);
    }
    let sample_rate = chunk.sample_rate_or(fallback_sample_rate).max(1);
    if let Some(expected) = *merged_sample_rate {
        if expected != sample_rate {
            return Err(format!(
                "Streaming sample rate changed from {expected} Hz to {sample_rate} Hz"
            ));
        }
    } else {
        send_stream_event(
            event_tx,
            SpeechStreamEvent {
                event: "start",
                timing: None,
                request_id: Some(request_id.to_string()),
                sequence: None,
                audio_base64: None,
                sample_count: None,
                sample_rate: Some(sample_rate),
                audio_format: Some("pcm_i16"),
                tokens_generated: None,
                generation_time_ms: None,
                audio_duration_secs: None,
                rtf: None,
                stop_reason: None,
                output_truncated: None,
                record: None,
                error: None,
            },
        )
        .await
        .map_err(str::to_string)?;
        *merged_sample_rate = Some(sample_rate);
    }
    Ok(Some(sample_rate))
}

pub async fn list_text_to_speech_records(
    State(state): State<AppState>,
    Query(params): Query<SpeechHistoryListParams>,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    list_records(state, SpeechRouteKind::TextToSpeech, params).await
}

pub async fn list_voice_design_records(
    State(state): State<AppState>,
    Query(params): Query<SpeechHistoryListParams>,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    list_records(state, SpeechRouteKind::VoiceDesign, params).await
}

pub async fn list_voice_cloning_records(
    State(state): State<AppState>,
    Query(params): Query<SpeechHistoryListParams>,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    list_records(state, SpeechRouteKind::VoiceCloning, params).await
}

pub async fn get_text_to_speech_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<SpeechHistoryRecord>, ApiError> {
    get_record(state, SpeechRouteKind::TextToSpeech, record_id).await
}

pub async fn get_voice_design_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<SpeechHistoryRecord>, ApiError> {
    get_record(state, SpeechRouteKind::VoiceDesign, record_id).await
}

pub async fn get_voice_cloning_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<SpeechHistoryRecord>, ApiError> {
    get_record(state, SpeechRouteKind::VoiceCloning, record_id).await
}

pub async fn get_text_to_speech_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<RecordAudioQuery>,
) -> Result<Response, ApiError> {
    get_record_audio(state, SpeechRouteKind::TextToSpeech, record_id, query).await
}

pub async fn get_voice_design_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<RecordAudioQuery>,
) -> Result<Response, ApiError> {
    get_record_audio(state, SpeechRouteKind::VoiceDesign, record_id, query).await
}

pub async fn get_voice_cloning_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<RecordAudioQuery>,
) -> Result<Response, ApiError> {
    get_record_audio(state, SpeechRouteKind::VoiceCloning, record_id, query).await
}

pub async fn delete_text_to_speech_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DeleteSpeechHistoryRecordResponse>, ApiError> {
    delete_record(state, SpeechRouteKind::TextToSpeech, record_id).await
}

pub async fn cancel_text_to_speech_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<CancelSpeechHistoryRecordResponse>, ApiError> {
    let record = cancel_speech_history_job(
        &state,
        SpeechRouteKind::TextToSpeech,
        &record_id,
        "Cancelled by speech history request",
    )
    .await?;
    // A stream runs in this process, not as a durable job: abort its task,
    // which stops the model, and say why the record stopped.
    let record = match record {
        None if state.speech_stream_tasks.abort(&record_id) => state
            .speech_history_store
            .update_processing_status(
                SpeechRouteKind::TextToSpeech,
                record_id.clone(),
                SpeechHistoryProcessingStatus::Failed,
                Some("Cancelled by user.".to_string()),
            )
            .await
            .map_err(map_store_error)?,
        other => other,
    };
    let record = match record {
        Some(record) => record,
        None => state
            .speech_history_store
            .get_record(SpeechRouteKind::TextToSpeech, record_id.clone())
            .await
            .map_err(map_store_error)?
            .filter(|record| {
                matches!(
                    record.processing_status,
                    SpeechHistoryProcessingStatus::Ready | SpeechHistoryProcessingStatus::Failed
                )
            })
            .ok_or_else(|| ApiError::bad_request("Speech history job is not cancellable"))?,
    };

    Ok(Json(CancelSpeechHistoryRecordResponse {
        id: record_id,
        cancelled: true,
        record,
    }))
}

pub async fn delete_voice_design_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DeleteSpeechHistoryRecordResponse>, ApiError> {
    delete_record(state, SpeechRouteKind::VoiceDesign, record_id).await
}

pub async fn delete_voice_cloning_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DeleteSpeechHistoryRecordResponse>, ApiError> {
    delete_record(state, SpeechRouteKind::VoiceCloning, record_id).await
}

pub async fn create_text_to_speech_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    headers: HeaderMap,
    ApiJson(req): ApiJson<CreateSpeechHistoryRecordRequest>,
) -> Result<Response, ApiError> {
    let idempotency_key = extract_idempotency_key(&headers)?;
    create_record(
        state,
        ctx,
        req,
        SpeechRouteKind::TextToSpeech,
        idempotency_key,
    )
    .await
}

pub async fn create_voice_design_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<CreateSpeechHistoryRecordRequest>,
) -> Result<Response, ApiError> {
    create_record(state, ctx, req, SpeechRouteKind::VoiceDesign, None).await
}

pub async fn create_voice_cloning_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<CreateSpeechHistoryRecordRequest>,
) -> Result<Response, ApiError> {
    create_record(state, ctx, req, SpeechRouteKind::VoiceCloning, None).await
}

async fn list_records(
    state: AppState,
    route_kind: SpeechRouteKind,
    params: SpeechHistoryListParams,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    let filter = params.filter()?;
    let query = CursorPaginationQuery {
        limit: params.limit,
        cursor: params.cursor,
    };
    let limit = query.resolved_limit(HISTORY_LIST_LIMIT, 500);
    let cursor = query.decode_cursor::<SpeechHistoryRecordListCursor>()?;
    let page = state
        .speech_history_store
        .list_records_filtered(route_kind, limit, cursor, &filter)
        .await
        .map_err(map_store_error)?;
    Ok(Json(SpeechHistoryRecordListResponse {
        records: page.records,
        pagination: CursorPagination {
            has_more: page.next_cursor.is_some(),
            next_cursor: page.next_cursor.map(|value| encode_cursor(&value)),
            limit,
        },
        total: page.total,
    }))
}

async fn get_record(
    state: AppState,
    route_kind: SpeechRouteKind,
    record_id: String,
) -> Result<Json<SpeechHistoryRecord>, ApiError> {
    let record = state
        .speech_history_store
        .get_record(route_kind, record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("History record not found"))?;
    Ok(Json(record))
}

async fn get_record_audio(
    state: AppState,
    route_kind: SpeechRouteKind,
    record_id: String,
    query: RecordAudioQuery,
) -> Result<Response, ApiError> {
    let as_attachment = query.download;
    let output = match query.format.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(name) => Some(crate::api::openai::audio::speech::SpeechOutput::resolve(
            Some(name),
            false,
            None,
            None,
        )?),
    };
    let audio = state
        .speech_history_store
        .get_audio_stream(route_kind, record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("History audio not found"))?;
    let mut reader = audio.audio.reader;
    if let Some(output) = output {
        use tokio::io::AsyncReadExt;
        let mut stored = Vec::new();
        reader
            .read_to_end(&mut stored)
            .await
            .map_err(|err| ApiError::internal(format!("Failed reading history audio: {err}")))?;
        let filename = audio.audio_filename.as_deref().map(|name| {
            let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
            format!("{stem}.{}", output.format.format.extension())
        });
        let encoded = tokio::task::spawn_blocking(move || {
            let (samples, rate) = decode_wav_to_f32(&stored)?;
            output
                .encode_file(&samples, rate)
                .map(|bytes| (bytes, output))
        })
        .await
        .map_err(|err| ApiError::internal(format!("Audio encoding failed: {err}")))??;
        let (bytes, output) = encoded;
        let length = bytes.len();
        let mut response = audio_response(
            StoredSpeechAudio {
                audio_bytes: bytes,
                audio_mime_type: output.content_type().to_string(),
                audio_filename: filename,
            },
            as_attachment,
        );
        response
            .headers_mut()
            .insert(header::CONTENT_LENGTH, HeaderValue::from(length));
        return Ok(response);
    }
    let mut response = audio_response(
        StoredSpeechAudio {
            audio_bytes: Vec::new(),
            audio_mime_type: audio.audio_mime_type,
            audio_filename: audio.audio_filename,
        },
        as_attachment,
    );
    let stream = async_stream::stream! {
        use tokio::io::AsyncReadExt;
        loop {
            let mut buffer = vec![0u8; 64 * 1024];
            let count = match reader.read(&mut buffer).await {
                Ok(count) => count,
                Err(error) => { yield Err::<bytes::Bytes, std::io::Error>(error); break; }
            };
            if count == 0 { break; }
            buffer.truncate(count);
            yield Ok(bytes::Bytes::from(buffer));
        }
    };
    *response.body_mut() = Body::from_stream(stream);
    if let Some(length) = audio.audio.metadata.content_length {
        response
            .headers_mut()
            .insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    }
    Ok(response)
}

async fn delete_record(
    state: AppState,
    route_kind: SpeechRouteKind,
    record_id: String,
) -> Result<Json<DeleteSpeechHistoryRecordResponse>, ApiError> {
    let _ = cancel_speech_history_job(
        &state,
        route_kind,
        &record_id,
        "Cancelled because the speech history record was deleted",
    )
    .await?;
    durable::cleanup_record_replay(&state, route_kind, &record_id)
        .await
        .map_err(map_store_error)?;
    let deleted = state
        .speech_history_store
        .delete_record(route_kind, record_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("History record not found"));
    }

    Ok(Json(DeleteSpeechHistoryRecordResponse {
        id: record_id,
        deleted: true,
    }))
}

async fn cancel_speech_history_job(
    state: &AppState,
    route_kind: SpeechRouteKind,
    record_id: &str,
    reason: &str,
) -> Result<Option<SpeechHistoryRecord>, ApiError> {
    let Some(job) = state
        .batch_runtime_store
        .get_active_job_for_route_record(
            RuntimeJobKind::TtsSpeech,
            route_kind.as_db_value(),
            record_id,
        )
        .await
        .map_err(map_store_error)?
    else {
        return Ok(None);
    };

    let _ = state
        .batch_runtime_store
        .cancel_job(&job.id, Some(reason.to_string()))
        .await
        .map_err(map_store_error)?;

    // A running executor remains authoritative until exact-attempt teardown.
    // The store returns the current projection and updates it transactionally
    // only if the durable job really became terminal.
    state
        .speech_history_store
        .get_record(route_kind, record_id.to_string())
        .await
        .map_err(map_store_error)
}

async fn create_record(
    state: AppState,
    ctx: RequestContext,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    idempotency_key: Option<String>,
) -> Result<Response, ApiError> {
    let mut req = normalize_create_request(req);
    if let Some(text) = crate::api::speech_dialogue::script_text(
        req.text.as_deref(),
        req.script.as_deref(),
        "text",
    )? {
        req.text = Some(text);
    }
    let model_id = required_trimmed(req.model_id.as_deref(), "model_id")?;
    let input_text = required_trimmed(req.text.as_deref(), "text")?;
    crate::api::tts_policy::check_tts_input_length(&input_text)?;
    let variant = parse_tts_model_variant(model_id.as_str())
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {err}")))?;
    crate::api::speech_dialogue::check_dialogue_request(variant, speech_dialogue(&req))?;
    crate::api::speech_takes::check_previous_take_ids(variant, &req.previous_take_ids)?;

    if variant.plans_own_speech_segments() {
        crate::api::tts_long_form::SpeechTextPlan::for_variant(
            variant,
            &input_text,
            req.max_output_tokens.or(req.max_tokens).unwrap_or(0),
        )
        .map_err(core_api_error)?;
    }
    validate_reference_voice_selection(&req)?;
    // The takes it continues, read now so a missing or unusable one is
    // refused before anything is queued or loaded; the stage reads them
    // again.
    let previous_codes_sha256 = crate::api::speech_takes::load_previous_takes(
        &state,
        route_kind,
        variant,
        &req.previous_take_ids,
    )
    .await?
    .into_iter()
    .map(|take| take.codes_sha256)
    .collect::<Vec<_>>();
    if route_kind == SpeechRouteKind::TextToSpeech
        && idempotency_key.is_some()
        && req.stream.unwrap_or(false)
    {
        return Err(ApiError::bad_request(
            "Idempotency-Key is not supported for streaming text-to-speech requests",
        ));
    }
    req = resolve_saved_voice_selection(&state, variant, req).await?;
    req = normalize_for_model_capabilities(route_kind, variant, req)?;
    if req.stream.unwrap_or(false) && route_kind != SpeechRouteKind::TextToSpeech {
        return Err(ApiError::bad_request(
            "Streaming is currently supported only on text-to-speech generation resources",
        ));
    }

    if route_kind == SpeechRouteKind::TextToSpeech {
        let is_text_only = req.reference_audio.is_none()
            && req.reference_text.is_none()
            && req.saved_voice_id.is_none()
            && req.references.is_empty();
        if !req.stream.unwrap_or(false) && is_text_only {
            return accept_durable_text_tts_record(
                &state,
                &ctx,
                req,
                route_kind,
                model_id,
                input_text,
                idempotency_key,
                &previous_codes_sha256,
            )
            .await;
        }

        // Streaming requests cannot carry a key (rejected above).
        let reservation = match idempotency_key {
            Some(idempotency_key) => {
                let mut keyed = req.clone();
                normalize_durable_text_tts_request(&mut keyed);
                let snapshot = BatchSpeechRequest::for_durable_job(
                    route_kind,
                    model_id.clone(),
                    input_text.clone(),
                    keyed,
                );
                let request_digest = durable_tts_request_digest(
                    &snapshot,
                    req.reference_audio.as_deref(),
                    &req.references,
                    &previous_codes_sha256,
                )?;
                match begin_durable_idempotency(
                    &state.batch_runtime_store,
                    DurableIdempotencyRequest {
                        tenant_scope: ctx.tenant_scope(),
                        operation: DURABLE_TTS_IDEMPOTENCY_OPERATION.to_string(),
                        idempotency_key,
                        digest_version: DURABLE_IDEMPOTENCY_DIGEST_VERSION,
                        request_digest,
                        reservation_ttl_ms: DURABLE_REFERENCE_TTS_IDEMPOTENCY_RESERVATION_TTL_MS,
                    },
                    "text-to-speech",
                )
                .await?
                {
                    DurableIdempotencyStart::Reserved(reservation) => Some(reservation),
                    DurableIdempotencyStart::Replay(replay) => {
                        return replay_text_to_speech_record(&state, &replay).await;
                    }
                }
            }
            None => None,
        };

        let placeholder = match create_pending_record(
            &state,
            route_kind,
            model_id.as_str(),
            input_text.as_str(),
            &req,
        )
        .await
        {
            Ok(placeholder) => placeholder,
            Err(err) => {
                release_durable_idempotency(&state.batch_runtime_store, reservation.as_ref()).await;
                return Err(err);
            }
        };

        if req.stream.unwrap_or(false) && !variant.plans_own_speech_segments() {
            return stream_record_creation(
                state,
                ctx,
                req,
                route_kind,
                variant,
                model_id,
                input_text,
                placeholder,
            )
            .await;
        }

        if let Err(err) = enqueue_batch_speech_job(
            &state,
            &placeholder,
            req.clone(),
            route_kind,
            model_id,
            input_text,
            ctx.tenant_key(),
            Some(ctx.correlation_id.clone()),
            reservation.clone(),
        )
        .await
        {
            release_durable_idempotency(&state.batch_runtime_store, reservation.as_ref()).await;
            let _ = state
                .speech_history_store
                .update_processing_status(
                    route_kind,
                    placeholder.id.clone(),
                    SpeechHistoryProcessingStatus::Failed,
                    Some(err.message.clone()),
                )
                .await;
            return Err(err);
        }

        if req.stream.unwrap_or(false) {
            return durable::replay_response(state, ctx.tenant_key(), placeholder.id, None).await;
        }
        return Ok((StatusCode::ACCEPTED, Json(placeholder)).into_response());
    }

    let (record, _, stop_reason) = synthesize_record_internal(
        &state,
        &ctx,
        ctx.tenant_key(),
        req,
        route_kind,
        variant,
        model_id,
        input_text,
        None,
        WorkloadClass::Online,
        None,
        None,
        None,
    )
    .await?;

    let mut response = Json(record).into_response();
    if let Some(stop_reason) = stop_reason {
        let headers = response.headers_mut();
        headers.insert(
            "X-Stop-Reason",
            axum::http::HeaderValue::from_static(stop_reason.as_str()),
        );
        headers.insert(
            "X-Output-Truncated",
            axum::http::HeaderValue::from_static(if stop_reason.truncated() {
                "true"
            } else {
                "false"
            }),
        );
    }
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
async fn accept_durable_text_tts_record(
    state: &AppState,
    ctx: &RequestContext,
    mut req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    model_id: String,
    input_text: String,
    idempotency_key: Option<String>,
    previous_codes_sha256: &[String],
) -> Result<Response, ApiError> {
    if input_text.len() > MAX_DURABLE_TTS_TEXT_BYTES {
        return Err(ApiError::payload_too_large(format!(
            "Text-to-speech input exceeds {MAX_DURABLE_TTS_TEXT_BYTES} bytes"
        )));
    }

    // Normalize aliases/defaults that execute identically so callers do not
    // get false idempotency conflicts for equivalent public requests.
    normalize_durable_text_tts_request(&mut req);

    let mut request_snapshot = BatchSpeechRequest::for_durable_job(
        route_kind,
        model_id.clone(),
        input_text.clone(),
        req.clone(),
    );
    request_snapshot.tenant_key = ctx.tenant_key();
    let request_json = serde_json::to_value(&request_snapshot)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;

    let reservation = if let Some(idempotency_key) = idempotency_key {
        let request_digest =
            durable_tts_request_digest(&request_snapshot, None, &[], previous_codes_sha256)?;
        match begin_durable_idempotency(
            &state.batch_runtime_store,
            DurableIdempotencyRequest {
                tenant_scope: ctx.tenant_scope(),
                operation: DURABLE_TTS_IDEMPOTENCY_OPERATION.to_string(),
                idempotency_key,
                digest_version: DURABLE_IDEMPOTENCY_DIGEST_VERSION,
                request_digest,
                reservation_ttl_ms: DURABLE_TTS_IDEMPOTENCY_RESERVATION_TTL_MS,
            },
            "text-to-speech",
        )
        .await?
        {
            DurableIdempotencyStart::Reserved(reservation) => Some(reservation),
            DurableIdempotencyStart::Replay(replay) => {
                return replay_text_to_speech_record(state, &replay).await;
            }
        }
    } else {
        None
    };

    if let Err(err) = state
        .batch_runtime_store
        .preflight_speech_admission(ctx.tenant_key())
        .await
    {
        if let Some(reservation) = reservation.as_ref() {
            let _ = state
                .batch_runtime_store
                .release_durable_idempotency(reservation)
                .await;
        }
        return Err(map_store_error(err));
    }

    let projection = NewSpeechHistoryRecord {
        stop_reason: None,
        route_kind,
        processing_status: SpeechHistoryProcessingStatus::Pending,
        processing_error: None,
        model_id: Some(model_id.clone()),
        speaker: req.speaker.clone(),
        language: req.language.clone(),
        saved_voice_id: None,
        speed: req.speed.map(f64::from),
        input_text: input_text.clone(),
        voice_description: req.voice_description.clone(),
        reference_text: None,
        generation_time_ms: 0.0,
        audio_duration_secs: None,
        rtf: None,
        tokens_generated: None,
        audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav).to_string(),
        audio_filename: Some(default_audio_filename(route_kind, "wav")),
        audio_bytes: Vec::new(),
        take: record_take_inputs(&model_id, &req),
    };
    let model_snapshot_json = serde_json::json!({
        "version": 1,
        "model_id": model_id,
        "request_sha256": sha256_hex(request_json.to_string().as_bytes()),
        "text_sha256": sha256_hex(input_text.as_bytes()),
        "reference_audio_sha256": serde_json::Value::Null,
        "reference_text_sha256": serde_json::Value::Null,
    });
    let acceptance = NewDurableTextTtsAcceptance {
        projection,
        request_json,
        model_snapshot_json,
        retry_policy_json: serde_json::json!({"max_attempts": 2}),
        priority: 0,
        max_attempts: 2,
        correlation_id: Some(ctx.correlation_id.clone()),
        stage_kind: BATCH_TTS_STAGE_KIND.to_string(),
        queue_class: QueueClass::BatchTts,
        resource_hints: StageResourceHints::default(),
        reservation: reservation.clone(),
        idempotency_retention_ms: DURABLE_IDEMPOTENCY_RETENTION_MS,
    };

    let outcome = match state
        .batch_runtime_store
        .accept_durable_text_tts(acceptance)
        .await
    {
        Ok(outcome) => outcome,
        Err(err) => {
            if let Some(reservation) = reservation.as_ref() {
                let _ = state
                    .batch_runtime_store
                    .release_durable_idempotency(reservation)
                    .await;
            }
            return Err(map_store_error(err));
        }
    };

    match outcome {
        DurableTextTtsAcceptanceOutcome::Committed(accepted) => {
            state.runtime.record_batch_tts_pipeline_job();
            Ok((StatusCode::ACCEPTED, Json(accepted.record)).into_response())
        }
        DurableTextTtsAcceptanceOutcome::ReservationLost => {
            if let Some(reservation) = reservation.as_ref() {
                let _ = state
                    .batch_runtime_store
                    .release_durable_idempotency(reservation)
                    .await;
            }
            Err(reservation_lost("Text-to-speech"))
        }
    }
}

/// Answer a keyed retry with the record its first request created.
async fn replay_text_to_speech_record(
    state: &AppState,
    replay: &DurableIdempotencyReplay,
) -> Result<Response, ApiError> {
    let record = state
        .speech_history_store
        .get_record(SpeechRouteKind::TextToSpeech, replay_record_id(replay)?)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| {
            ApiError::internal("Stored text-to-speech idempotency record no longer exists")
        })?;
    Ok((StatusCode::ACCEPTED, Json(record)).into_response())
}

fn normalize_durable_text_tts_request(req: &mut CreateSpeechHistoryRecordRequest) {
    req.stream = Some(false);
    req.max_output_tokens = req.max_output_tokens.or(req.max_tokens);
    req.max_tokens = None;
}

/// The keyed request's canonical digest. A reference voice is identified by
/// its audio payload's hash, dialogue references by theirs in order, and
/// the takes it continues by their stored codes' hashes in order; a
/// text-only digest has none of these fields, so keys stored before they
/// could be keyed still replay.
fn durable_tts_request_digest(
    request: &BatchSpeechRequest,
    reference_audio: Option<&str>,
    references: &[crate::api::speech_dialogue::SpeechDialogueReference],
    previous_codes_sha256: &[String],
) -> Result<String, ApiError> {
    let mut semantic_request = serde_json::json!({
        "version": DURABLE_IDEMPOTENCY_DIGEST_VERSION,
        "operation": DURABLE_TTS_IDEMPOTENCY_OPERATION,
        "route_kind": request.route_kind,
        "model_id": &request.model_id,
        "input_text_sha256": sha256_hex(request.input_text.as_bytes()),
        "input_text_bytes": request.input_text.len(),
        "request": &request.request,
    });
    if let Some(reference_audio) = reference_audio {
        semantic_request["reference_audio_sha256"] =
            serde_json::Value::String(sha256_hex(reference_audio.trim().as_bytes()));
    }
    if !references.is_empty() {
        semantic_request["references_audio_sha256"] = references
            .iter()
            .map(|reference| {
                serde_json::Value::String(sha256_hex(
                    reference.audio.as_deref().unwrap_or_default().as_bytes(),
                ))
            })
            .collect();
    }
    if !previous_codes_sha256.is_empty() {
        semantic_request["previous_takes_codes_sha256"] = previous_codes_sha256.into();
    }
    canonical_request_digest(&semantic_request).map_err(|err| {
        ApiError::payload_too_large(format!(
            "Text-to-speech idempotency request cannot be canonicalized: {err}"
        ))
    })
}

async fn create_pending_record(
    state: &AppState,
    route_kind: SpeechRouteKind,
    model_id: &str,
    input_text: &str,
    req: &CreateSpeechHistoryRecordRequest,
) -> Result<SpeechHistoryRecord, ApiError> {
    state
        .speech_history_store
        .create_record(NewSpeechHistoryRecord {
            stop_reason: None,
            route_kind,
            processing_status: SpeechHistoryProcessingStatus::Pending,
            processing_error: None,
            model_id: Some(model_id.to_string()),
            speaker: req.speaker.clone(),
            language: req.language.clone(),
            saved_voice_id: req.saved_voice_id.clone(),
            speed: req.speed.map(f64::from),
            input_text: input_text.to_string(),
            voice_description: req.voice_description.clone(),
            reference_text: req.reference_text.clone(),
            generation_time_ms: 0.0,
            audio_duration_secs: None,
            rtf: None,
            tokens_generated: None,
            audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav).to_string(),
            audio_filename: Some(default_audio_filename(route_kind, "wav")),
            audio_bytes: Vec::new(),
            take: record_take_inputs(model_id, req),
        })
        .await
        .map_err(map_store_error)
}

/// What a new record of `model_id` keeps of `req` for take editing.
fn record_take_inputs(
    model_id: &str,
    req: &CreateSpeechHistoryRecordRequest,
) -> crate::speech_history_store::SpeechTakeInputs {
    parse_tts_model_variant(model_id)
        .map(|variant| crate::api::speech_takes::take_inputs(variant, req))
        .unwrap_or_default()
}

/// One reference's audio as a durable input: a saved voice's stored audio
/// (`saved_voice_id`), else the request's base64 clip.
async fn ingest_batch_reference_audio(
    state: &AppState,
    placeholder: &SpeechHistoryRecord,
    route_kind: SpeechRouteKind,
    reference_audio: Option<&str>,
    saved_voice_id: Option<&str>,
) -> Result<Option<MediaIngestResult>, ApiError> {
    let Some(reference_audio) = reference_audio else {
        return Ok(None);
    };
    let policy = AudioIngestPolicy::media_upload(REFERENCE_AUDIO_UPLOAD_LIMIT_BYTES);
    let profile = CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav;

    if let Some(saved_voice_id) = saved_voice_id {
        let saved_audio = state
            .saved_voice_store
            .get_audio(saved_voice_id.to_string())
            .await
            .map_err(map_store_error)?
            .ok_or_else(|| ApiError::not_found("Saved voice audio not found"))?;
        return state
            .media_ingest
            .ingest_existing_audio(
                ExistingAudioIngestRequest {
                    bytes: saved_audio.audio_bytes,
                    storage_key: saved_audio.audio_storage_path,
                    storage_namespace: "saved_voice".to_string(),
                    content_type: saved_audio.audio_mime_type,
                    filename: saved_audio.audio_filename,
                    record_id: placeholder.id.clone(),
                    route: route_kind.as_db_value().to_string(),
                },
                policy,
                profile,
            )
            .await
            .map(Some)
            .map_err(map_media_ingest_error);
    }

    let payload = decode_base64_audio_payload(reference_audio)?;
    let content_type = payload
        .content_type_hint()
        .unwrap_or("application/octet-stream")
        .to_string();
    state
        .media_ingest
        .ingest_audio(
            MediaIngestRequest {
                bytes: payload.bytes,
                content_type,
                filename: payload.filename,
                namespace: "tts_reference".to_string(),
                record_id: placeholder.id.clone(),
                route: route_kind.as_db_value().to_string(),
            },
            policy,
            profile,
        )
        .await
        .map(Some)
        .map_err(map_media_ingest_error)
}

async fn enqueue_batch_speech_job(
    state: &AppState,
    placeholder: &SpeechHistoryRecord,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    model_id: String,
    input_text: String,
    tenant_key: Option<[u8; 32]>,
    correlation_id: Option<String>,
    reservation: Option<DurableIdempotencyReservation>,
) -> Result<(), ApiError> {
    state
        .batch_runtime_store
        .preflight_speech_admission(tenant_key)
        .await
        .map_err(map_store_error)?;
    let reference_ingest = ingest_batch_reference_audio(
        state,
        placeholder,
        route_kind,
        req.reference_audio.as_deref(),
        req.saved_voice_id.as_deref(),
    )
    .await?;
    // Each dialogue reference is its own pair of input artifacts, numbered
    // by its place in the request.
    let mut dialogue_ingests = Vec::with_capacity(req.references.len());
    for reference in &req.references {
        let ingest = ingest_batch_reference_audio(
            state,
            placeholder,
            route_kind,
            reference.audio.as_deref(),
            reference.saved_voice_id.as_deref(),
        )
        .await?
        .ok_or_else(|| ApiError::internal("A dialogue reference reached enqueue without audio"))?;
        dialogue_ingests.push(ingest);
    }
    let dialogue_references = req
        .references
        .iter()
        .map(|reference| {
            (
                crate::api::speech_dialogue::reference_speaker(reference),
                reference
                    .text
                    .as_deref()
                    .map(|text| sha256_hex(text.as_bytes())),
            )
        })
        .collect::<Vec<_>>();
    let mut request_snapshot =
        BatchSpeechRequest::for_durable_job(route_kind, model_id.clone(), input_text.clone(), req);
    request_snapshot.tenant_key = tenant_key;
    let request_json = serde_json::to_value(&request_snapshot)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;

    let text_asset_id = new_uuid();
    let text_artifact_id = new_uuid();
    let text_sha256 = sha256_hex(input_text.as_bytes());
    let mut artifacts = vec![NewDurableJobArtifact {
        id: text_artifact_id.clone(),
        artifact_kind: RuntimeArtifactKind::Text,
        artifact_role: RuntimeArtifactRole::InputOriginal,
        media_asset_id: None,
        text_asset_id: Some(text_asset_id.clone()),
        storage_key: None,
        content_type: Some("text/plain".to_string()),
        filename: Some(format!("{}.input.txt", placeholder.id)),
        size_bytes: Some(input_text.len() as u64),
        sha256: Some(text_sha256.clone()),
        metadata_json: serde_json::json!({
            "route_record_id": placeholder.id.clone(),
            "input_kind": "text"
        }),
        retention_policy: "default".to_string(),
    }];
    let mut stage_input_artifact_ids = vec![text_artifact_id];
    let mut input_media_asset_id = None;
    let mut reference_audio_sha256 = None;
    if let Some(reference_ingest) = reference_ingest {
        let reference =
            push_reference_artifacts(&mut artifacts, placeholder, reference_ingest, None)?;
        input_media_asset_id = Some(reference.source_asset_id);
        reference_audio_sha256 = reference.source_sha256;
        stage_input_artifact_ids.push(reference.canonical_artifact_id);
    }
    // The snapshot names every dialogue reference's speaker, audio and
    // transcript, so a recovered plan's identity changes with any of them.
    let mut snapshot_references = Vec::with_capacity(dialogue_ingests.len());
    for (index, (ingest, (speaker, text_sha256))) in dialogue_ingests
        .into_iter()
        .zip(dialogue_references)
        .enumerate()
    {
        let reference = push_reference_artifacts(&mut artifacts, placeholder, ingest, Some(index))?;
        input_media_asset_id.get_or_insert(reference.source_asset_id);
        stage_input_artifact_ids.push(reference.canonical_artifact_id);
        snapshot_references.push(serde_json::json!({
            "speaker": speaker,
            "audio_sha256": reference.source_sha256,
            "text_sha256": text_sha256,
        }));
    }

    let accepted = state
        .batch_runtime_store
        .accept_durable_job(NewDurableJobAcceptance {
            job: NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some(model_id),
                capability: Some("tts".to_string()),
                route_record_kind: Some(route_kind.as_db_value().to_string()),
                route_record_id: Some(placeholder.id.clone()),
                input_media_asset_id,
                input_text_asset_id: Some(text_asset_id.clone()),
                request_json: request_json.clone(),
                model_snapshot_json: serde_json::json!({
                    "version": 1,
                    "model_id": request_snapshot.model_id,
                    "request_sha256": sha256_hex(request_json.to_string().as_bytes()),
                    "text_sha256": text_sha256.clone(),
                    "reference_audio_sha256": reference_audio_sha256,
                    "reference_text_sha256": request_snapshot.request.reference_text.as_ref()
                        .map(|text| sha256_hex(text.as_bytes())),
                    "references": snapshot_references,
                }),
                retry_policy_json: serde_json::json!({"max_attempts": 2}),
                max_attempts: 2,
                idempotency_key: None,
                correlation_id,
            },
            text_input: Some(NewDurableTextInput {
                id: text_asset_id,
                asset: NewTextAsset {
                    raw_text: input_text.clone(),
                    normalized_text: Some(input_text.clone()),
                    language_hint: request_snapshot.request.language.clone(),
                    sha256: Some(text_sha256),
                    safety_status: "unchecked".to_string(),
                    retention_policy: "default".to_string(),
                    structure_json: serde_json::json!({
                        "source": "tts_input",
                        "route_record_id": placeholder.id.clone()
                    }),
                },
            }),
            artifacts,
            stages: vec![NewDurableJobStage {
                sequence: 0,
                stage_kind: BATCH_TTS_STAGE_KIND.to_string(),
                capability: Some("tts".to_string()),
                model_id: Some(request_snapshot.model_id.clone()),
                max_attempts: 2,
                input_artifact_ids: stage_input_artifact_ids,
                queue_class: QueueClass::BatchTts,
                resource_hints: StageResourceHints::default(),
            }],
            reservation,
            idempotency_response: serde_json::json!({"record_id": placeholder.id.clone()}),
            idempotency_retention_ms: DURABLE_IDEMPOTENCY_RETENTION_MS,
        })
        .await
        .map_err(map_store_error)?;
    if let DurableJobAcceptanceOutcome::ReservationLost = accepted {
        return Err(reservation_lost("Text-to-speech"));
    }

    state.runtime.record_batch_tts_pipeline_job();

    Ok(())
}

/// The input artifacts of one reference's ingest.
struct ReferenceArtifacts {
    canonical_artifact_id: String,
    source_asset_id: String,
    source_sha256: Option<String>,
}

/// Adds one reference's original and canonical audio to `artifacts`: the
/// request's one reference (`index` none) or its dialogue reference
/// `index`, which the canonical artifact's metadata records for recovery.
fn push_reference_artifacts(
    artifacts: &mut Vec<NewDurableJobArtifact>,
    placeholder: &SpeechHistoryRecord,
    ingest: MediaIngestResult,
    index: Option<usize>,
) -> Result<ReferenceArtifacts, ApiError> {
    let source_asset = ingest.source_asset.ok_or_else(|| {
        ApiError::internal("Reference audio ingest did not register its source asset")
    })?;
    let canonical = ingest.canonical.ok_or_else(|| {
        ApiError::internal("Reference audio ingest did not create a canonical asset")
    })?;
    let original_artifact_id = new_uuid();
    let canonical_artifact_id = new_uuid();
    let source_asset_id = source_asset.id.clone();
    let source_sha256 = source_asset.sha256.clone();
    let mut original_metadata = serde_json::json!({
        "route_record_id": placeholder.id.clone(),
        "input_kind": "reference_audio",
        "canonical_profile": CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav.id()
    });
    let mut canonical_metadata = serde_json::json!({
        "route_record_id": placeholder.id.clone(),
        "input_kind": "reference_audio",
        "source_artifact_id": original_artifact_id,
        "canonical_profile": CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav.id(),
        "sample_rate_hz": canonical.asset.sample_rate_hz
    });
    if let Some(index) = index {
        original_metadata["reference_index"] = index.into();
        canonical_metadata["reference_index"] = index.into();
    }
    artifacts.push(NewDurableJobArtifact {
        id: original_artifact_id,
        artifact_kind: RuntimeArtifactKind::Media,
        artifact_role: RuntimeArtifactRole::InputOriginal,
        media_asset_id: Some(source_asset.id),
        text_asset_id: None,
        storage_key: Some(source_asset.storage_key),
        content_type: Some(source_asset.content_type),
        filename: source_asset.filename,
        size_bytes: Some(source_asset.size_bytes),
        sha256: source_asset.sha256,
        metadata_json: original_metadata,
        retention_policy: "default".to_string(),
    });
    artifacts.push(NewDurableJobArtifact {
        id: canonical_artifact_id.clone(),
        artifact_kind: RuntimeArtifactKind::Media,
        artifact_role: RuntimeArtifactRole::InputCanonical,
        media_asset_id: Some(canonical.asset.id),
        text_asset_id: None,
        storage_key: Some(canonical.asset.storage_key),
        content_type: Some(canonical.asset.content_type),
        filename: canonical.asset.filename,
        size_bytes: Some(canonical.asset.size_bytes),
        sha256: canonical.asset.sha256,
        metadata_json: canonical_metadata,
        retention_policy: "default".to_string(),
    });
    Ok(ReferenceArtifacts {
        canonical_artifact_id,
        source_asset_id,
        source_sha256,
    })
}

pub(crate) fn batch_tts_stage_executor(state: AppState) -> Arc<dyn StageExecutor> {
    Arc::new(BatchTtsStageExecutor { state })
}

struct BatchTtsStageExecutor {
    state: AppState,
}

#[async_trait]
impl StageExecutor for BatchTtsStageExecutor {
    fn stage_kind(&self) -> &'static str {
        BATCH_TTS_STAGE_KIND
    }

    async fn maintenance(&self) -> anyhow::Result<()> {
        durable::cleanup_expired_replay(&self.state).await
    }

    async fn execute(&self, claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
        execute_batch_tts_stage(&self.state, claimed, None).await
    }

    async fn execute_with_context(
        &self,
        context: StageExecutionContext,
    ) -> anyhow::Result<StageExecutionOutcome> {
        let claimed = context.claimed().clone();
        execute_batch_tts_stage(&self.state, claimed, Some(context)).await
    }
}

async fn execute_batch_tts_stage(
    state: &AppState,
    claimed: ClaimedStage,
    attempt: Option<StageExecutionContext>,
) -> anyhow::Result<StageExecutionOutcome> {
    if let Some(attempt) = attempt.as_ref() {
        attempt.ensure_active().await?;
    }
    let record_id = claimed
        .job
        .route_record_id
        .clone()
        .ok_or_else(|| anyhow::anyhow!("TTS batch job is missing route_record_id"))?;
    let projection_attempt = attempt
        .as_ref()
        .map(runtime_projection_attempt)
        .transpose()?;
    let mut request: BatchSpeechRequest = serde_json::from_value(claimed.job.request_json.clone())
        .context("Failed to decode TTS batch request")?;
    hydrate_batch_reference_audio(state, &claimed, &mut request.request).await?;
    // A job queued before its model's support was removed runs on the
    // supported counterpart (stored data normalizes on read, P1.0).
    let variant = parse_tts_model_variant(&izwi_core::normalize_stored_model_id(&request.model_id))
        .map_err(|err| anyhow::anyhow!("Unsupported TTS model: {err}"))?;
    let ctx = RequestContext::new(
        claimed
            .job
            .correlation_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        Principal::local_anonymous(),
    );

    if let Some(projection_attempt) = projection_attempt.as_ref() {
        state
            .speech_history_store
            .bind_runtime_attempt(request.route_kind, record_id.clone(), projection_attempt)
            .await?
            .ok_or_else(|| anyhow::anyhow!("TTS attempt no longer owns its route projection"))?;
        state
            .speech_history_store
            .update_processing_status_for_attempt(
                request.route_kind,
                record_id.clone(),
                projection_attempt,
                SpeechHistoryProcessingStatus::Processing,
                None,
            )
            .await?
            .ok_or_else(|| anyhow::anyhow!("TTS attempt lost its route projection"))?;
    } else {
        let _ = state
            .speech_history_store
            .update_processing_status(
                request.route_kind,
                record_id.clone(),
                SpeechHistoryProcessingStatus::Processing,
                None,
            )
            .await;
    }

    let (_record, output_artifact, _) = match synthesize_record_internal(
        state,
        &ctx,
        request.tenant_key,
        request.request,
        request.route_kind,
        variant,
        request.model_id,
        request.input_text,
        Some(record_id.clone()),
        WorkloadClass::Batch,
        attempt.as_ref(),
        projection_attempt.as_ref(),
        Some(&claimed),
    )
    .await
    {
        Ok(record) => record,
        Err(err) => {
            // The yield crosses the long-form planner as an inference error's
            // text; the marker's own display identifies it.
            if err
                .message
                .contains(&crate::batch_runtime::worker::SpeechStageYield.to_string())
            {
                return Err(crate::batch_runtime::worker::SpeechStageYield.into());
            }
            match projection_attempt.as_ref() {
                Some(projection_attempt) => {
                    let _ = state
                        .speech_history_store
                        .fail_record_for_attempt(
                            request.route_kind,
                            record_id,
                            projection_attempt,
                            err.message.clone(),
                        )
                        .await;
                }
                None => {
                    let _ = state
                        .speech_history_store
                        .update_processing_status(
                            request.route_kind,
                            record_id,
                            SpeechHistoryProcessingStatus::Failed,
                            Some(err.message.clone()),
                        )
                        .await;
                }
            }
            // An invalid request fails the same way on every attempt.
            if err.status.is_client_error() {
                return Err(
                    crate::batch_runtime::worker::DeterministicStageFailure(err.message).into(),
                );
            }
            // So does a segment that published audio: a retry would refuse
            // to speak it again, and that refusal would replace this error.
            if let Some(attempt) = attempt.as_ref() {
                if durable::published_partial_segment(state, attempt).await {
                    return Err(crate::batch_runtime::worker::FinalStageFailure(err.message).into());
                }
            }
            anyhow::bail!("{}", err.message);
        }
    };

    let output_artifact = output_artifact
        .ok_or_else(|| anyhow::anyhow!("TTS stage did not publish its primary audio"))?;

    StageExecutionOutcome::try_new(vec![output_artifact.id])
}

fn runtime_projection_attempt(
    context: &StageExecutionContext,
) -> anyhow::Result<RuntimeProjectionAttempt> {
    let attempt_token = context
        .lease()
        .attempt_token
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Stage attempt is missing its publication token"))?;
    Ok(RuntimeProjectionAttempt::new(
        context.lease().stage_id.clone(),
        attempt_token,
    ))
}

async fn hydrate_batch_reference_audio(
    state: &AppState,
    claimed: &ClaimedStage,
    request: &mut CreateSpeechHistoryRecordRequest,
) -> anyhow::Result<()> {
    // Enqueueing stores every reference's audio as an input artifact and
    // drops it from the request; a reference may come without a transcript
    // (Chatterbox), so the artifact, not the transcript, says there is one.
    // A dialogue reference's artifact carries its place in the request.
    let dialogue_missing = request
        .references
        .iter()
        .any(|reference| reference.audio.is_none());
    if request.reference_audio.is_some() && !dialogue_missing {
        return Ok(());
    }

    for artifact_id in &claimed.stage.input_artifact_ids {
        let Some(artifact) = state.batch_runtime_store.get_artifact(artifact_id).await? else {
            continue;
        };
        if artifact.artifact_role != RuntimeArtifactRole::InputCanonical
            || artifact.metadata_json["input_kind"] != "reference_audio"
        {
            continue;
        }
        let slot = match artifact.metadata_json.get("reference_index") {
            Some(index) => {
                let index = index
                    .as_u64()
                    .and_then(|index| usize::try_from(index).ok())
                    .ok_or_else(|| anyhow::anyhow!("TTS reference artifact has a bad index"))?;
                &mut request
                    .references
                    .get_mut(index)
                    .ok_or_else(|| {
                        anyhow::anyhow!("TTS reference artifact {index} has no reference")
                    })?
                    .audio
            }
            None => &mut request.reference_audio,
        };
        if slot.is_some() {
            continue;
        }
        let storage_key = artifact.storage_key.as_deref().ok_or_else(|| {
            anyhow::anyhow!("Canonical TTS reference artifact is missing storage_key")
        })?;
        let stored = state
            .media_ingest
            .read_object(storage_key)
            .await
            .context("Failed to read canonical TTS reference artifact")?;
        if let Some(expected) = artifact.sha256.as_deref() {
            anyhow::ensure!(
                sha256_hex(&stored.bytes) == expected,
                "Canonical TTS reference checksum changed"
            );
        }
        if let Some(expected) = artifact.size_bytes {
            anyhow::ensure!(
                stored.bytes.len() as u64 == expected,
                "Canonical TTS reference size changed"
            );
        }
        *slot = Some(base64::engine::general_purpose::STANDARD.encode(stored.bytes));
    }

    anyhow::ensure!(
        request.reference_audio.is_some()
            || (request.reference_text.is_none() && request.saved_voice_id.is_none()),
        "TTS batch request is missing its reference-audio input artifact"
    );
    anyhow::ensure!(
        request
            .references
            .iter()
            .all(|reference| reference.audio.is_some()),
        "TTS batch request is missing a dialogue reference's audio artifact"
    );
    Ok(())
}

async fn publish_speech_audio_runtime_artifact(
    state: &AppState,
    claimed: &ClaimedStage,
    route_kind: SpeechRouteKind,
    record_id: &str,
    completion: &CompleteSpeechHistoryRecord,
    attempt: Option<&StageExecutionContext>,
) -> anyhow::Result<(crate::batch_runtime::types::RuntimeArtifact, String)> {
    let inspection = inspect_audio_bytes(completion.audio_bytes.as_slice())
        .context("Failed to inspect generated speech audio")?;
    let storage_key = state
        .media_ingest
        .persist_generated_audio(
            format!("{record_id}-output-{}", uuid::Uuid::new_v4()),
            completion.audio_filename.as_deref(),
            &completion.audio_mime_type,
            &completion.audio_bytes,
            route_kind.as_db_value(),
        )
        .await
        .context("Failed to persist generated speech output")?;
    let media_asset = match state
        .batch_runtime_store
        .create_media_asset(NewMediaAsset {
            asset_kind: "audio_generated".to_string(),
            storage_namespace: "generated_speech".to_string(),
            storage_key: storage_key.clone(),
            content_type: completion.audio_mime_type.clone(),
            filename: completion.audio_filename.clone(),
            size_bytes: completion.audio_bytes.len() as u64,
            sha256: Some(sha256_hex(completion.audio_bytes.as_slice())),
            duration_secs: Some(inspection.duration_secs as f64),
            sample_rate_hz: Some(inspection.sample_rate),
            channel_count: Some(1),
            peak_amplitude: Some(inspection.peak),
            rms_amplitude: Some(inspection.rms),
            source_asset_id: None,
            canonical_profile_version: None,
            scan_status: "not_scanned".to_string(),
            retention_policy: "default".to_string(),
            metadata_json: serde_json::json!({
                "route_record_id": record_id,
                "route_kind": route_kind.as_db_value()
            }),
        })
        .await
    {
        Ok(asset) => asset,
        Err(err) => {
            let _ = state.media_ingest.delete_object(&storage_key).await;
            return Err(err);
        }
    };

    let artifact = NewRuntimeArtifact {
        job_id: claimed.job.id.clone(),
        stage_id: Some(claimed.stage.id.clone()),
        artifact_kind: RuntimeArtifactKind::Audio,
        artifact_role: RuntimeArtifactRole::OutputPrimary,
        media_asset_id: Some(media_asset.id),
        text_asset_id: None,
        storage_key: Some(media_asset.storage_key),
        content_type: Some(media_asset.content_type),
        filename: media_asset.filename,
        size_bytes: Some(media_asset.size_bytes),
        sha256: media_asset.sha256,
        metadata_json: serde_json::json!({
            "route_record_id": record_id,
            "route_kind": route_kind.as_db_value(),
            "duration_secs": completion.audio_duration_secs,
            "tokens_generated": completion.tokens_generated
        }),
        retention_policy: "default".to_string(),
    };
    let artifact = if let Some(attempt) = attempt {
        attempt
            .publish_output_artifact(NewStageOutputArtifact {
                publication_key: "primary_audio".to_string(),
                artifact_kind: artifact.artifact_kind,
                artifact_role: artifact.artifact_role,
                media_asset_id: artifact.media_asset_id,
                text_asset_id: artifact.text_asset_id,
                storage_key: artifact.storage_key,
                content_type: artifact.content_type,
                filename: artifact.filename,
                size_bytes: artifact.size_bytes,
                sha256: artifact.sha256,
                metadata_json: artifact.metadata_json,
                retention_policy: artifact.retention_policy,
            })
            .await?
    } else {
        state.batch_runtime_store.create_artifact(artifact).await?
    };
    Ok((artifact, storage_key))
}

pub(crate) async fn synthesize_record(
    state: &AppState,
    ctx: &RequestContext,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
) -> Result<SpeechHistoryRecord, ApiError> {
    let mut req = normalize_create_request(req);
    let model_id = required_trimmed(req.model_id.as_deref(), "model_id")?;
    let input_text = required_trimmed(req.text.as_deref(), "text")?;
    let variant = parse_tts_model_variant(model_id.as_str())
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {err}")))?;

    validate_reference_voice_selection(&req)?;
    req = resolve_saved_voice_selection(state, variant, req).await?;
    req = normalize_for_model_capabilities(route_kind, variant, req)?;
    synthesize_record_internal(
        state,
        ctx,
        ctx.tenant_key(),
        req,
        route_kind,
        variant,
        model_id,
        input_text,
        None,
        WorkloadClass::Online,
        None,
        None,
        None,
    )
    .await
    .map(|(record, _, _)| record)
}

/// `work`, unless the stage is cancelled first: `work` is then dropped,
/// which aborts the engine requests it owns.
async fn unless_cancelled<T>(
    cancellation: Option<StageCancellationSignal>,
    work: impl std::future::Future<Output = T>,
) -> Result<T, StageCancellationReason> {
    match cancellation {
        Some(cancellation) => tokio::select! {
            output = work => Ok(output),
            reason = cancellation.cancelled() => Err(reason),
        },
        None => Ok(work.await),
    }
}

async fn synthesize_record_internal(
    state: &AppState,
    ctx: &RequestContext,
    tenant_key: Option<[u8; 32]>,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    variant: ModelVariant,
    model_id: String,
    input_text: String,
    target_record_id: Option<String>,
    workload_class: WorkloadClass,
    attempt: Option<&StageExecutionContext>,
    projection_attempt: Option<&RuntimeProjectionAttempt>,
    batch_publication: Option<&ClaimedStage>,
) -> Result<
    (
        SpeechHistoryRecord,
        Option<crate::batch_runtime::types::RuntimeArtifact>,
        Option<TtsStopReason>,
    ),
    ApiError,
> {
    if variant.plans_own_speech_segments() {
        return durable::synthesize_fish_record(
            state,
            ctx,
            tenant_key,
            req,
            route_kind,
            variant,
            model_id,
            input_text,
            target_record_id,
            workload_class,
            attempt,
            projection_attempt,
            batch_publication,
        )
        .await
        .map(|(record, artifact)| (record, artifact, None));
    }
    if let Some(attempt) = attempt {
        attempt
            .ensure_active()
            .await
            .map_err(|err| ApiError::internal(err.to_string()))?;
    }
    let mut generation_request = build_generation_request(
        req.clone(),
        ctx.correlation_id.clone(),
        input_text.clone(),
        false,
        variant,
    );
    crate::api::tts_long_form::defer_automatic_budget_to_long_form(
        &mut generation_request,
        variant,
        req.max_output_tokens.or(req.max_tokens),
    );
    let permit = state.acquire_workload_permit(workload_class).await;
    state
        .runtime
        .load_model(variant)
        .await
        .map_err(core_api_error)?;
    let mut runtime_context = permit.runtime_context();
    runtime_context.tenant_key = tenant_key;
    generation_request = generation_request.with_runtime_context(runtime_context);
    // No wall clock wraps active generation (TPIPE-6); a dropped request
    // cancels the engine work. A cancelled stage drops it at once, so the
    // engine stops mid-render instead of finishing the utterance first.
    let runtime = state.runtime.clone();
    let output = unless_cancelled(
        attempt.map(StageExecutionContext::cancellation),
        generate_long_form_tts(&runtime, variant, generation_request),
    )
    .await
    .map_err(|reason| {
        ApiError::internal(format!(
            "Stage execution cancelled: {}",
            reason.as_error_code()
        ))
    })?
    .map_err(core_api_error)?;
    let stop_reason = output.stop_reason;

    let encoder = AudioEncoder::new(output.sample_rate, 1);
    let output_samples = output.samples.clone();
    let encoded_audio =
        tokio::task::spawn_blocking(move || encoder.encode(&output_samples, AudioFormat::Wav))
            .await
            .map_err(|err| ApiError::internal(format!("Audio encoding failed: {err}")))?
            .map_err(core_api_error)?;

    if let Some(attempt) = attempt {
        attempt
            .ensure_active()
            .await
            .map_err(|err| ApiError::internal(err.to_string()))?;
    }

    if let Some(record_id) = target_record_id {
        let mut completion = CompleteSpeechHistoryRecord {
            stop_reason: stop_reason.map(|reason| reason.as_str().to_string()),
            model_id: Some(model_id),
            speaker: req.speaker,
            language: req.language,
            saved_voice_id: req.saved_voice_id,
            speed: req.speed.map(f64::from),
            input_text,
            voice_description: req.voice_description,
            reference_text: req.reference_text,
            generation_time_ms: output.total_time_ms as f64,
            audio_duration_secs: Some(output.duration_secs() as f64),
            rtf: Some(output.rtf() as f64),
            tokens_generated: Some(output.total_tokens),
            audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav).to_string(),
            audio_filename: Some(default_audio_filename(route_kind, "wav")),
            audio_bytes: encoded_audio,
            preexisting_audio_storage_path: None,
            generated_codes_json: None,
        };
        let output_artifact = match batch_publication {
            Some(claimed) => {
                let (artifact, storage_key) = publish_speech_audio_runtime_artifact(
                    state,
                    claimed,
                    route_kind,
                    &record_id,
                    &completion,
                    attempt,
                )
                .await
                .map_err(|err| ApiError::internal(err.to_string()))?;
                completion.adopt_published_audio(storage_key);
                Some(artifact)
            }
            None => None,
        };
        if let Some(attempt) = attempt {
            attempt
                .ensure_active()
                .await
                .map_err(|err| ApiError::internal(err.to_string()))?;
        }
        let record = match projection_attempt {
            Some(projection_attempt) => {
                state
                    .speech_history_store
                    .complete_record_for_attempt(
                        route_kind,
                        record_id,
                        projection_attempt,
                        completion,
                    )
                    .await
            }
            None => {
                state
                    .speech_history_store
                    .complete_record(route_kind, record_id, completion)
                    .await
            }
        }
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::internal("TTS attempt lost its route projection"))?;
        return Ok((record, output_artifact, stop_reason));
    }

    state
        .speech_history_store
        .create_record(NewSpeechHistoryRecord {
            stop_reason: stop_reason.map(|reason| reason.as_str().to_string()),
            route_kind,
            processing_status: SpeechHistoryProcessingStatus::Ready,
            processing_error: None,
            model_id: Some(model_id),
            speaker: req.speaker,
            language: req.language,
            saved_voice_id: req.saved_voice_id,
            speed: req.speed.map(f64::from),
            input_text,
            voice_description: req.voice_description,
            reference_text: req.reference_text,
            generation_time_ms: output.total_time_ms as f64,
            audio_duration_secs: Some(output.duration_secs() as f64),
            rtf: Some(output.rtf() as f64),
            tokens_generated: Some(output.total_tokens),
            audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav).to_string(),
            audio_filename: Some(default_audio_filename(route_kind, "wav")),
            audio_bytes: encoded_audio,
            take: Default::default(),
        })
        .await
        .map_err(map_store_error)
        .map(|record| (record, None, stop_reason))
}

async fn stream_record_creation(
    state: AppState,
    ctx: RequestContext,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    variant: ModelVariant,
    model_id: String,
    input_text: String,
    placeholder: SpeechHistoryRecord,
) -> Result<Response, ApiError> {
    let stream_entry_started = std::time::Instant::now();
    let tenant_key = ctx.tenant_key();
    let mut generation_request = build_generation_request(
        req.clone(),
        ctx.correlation_id,
        input_text.clone(),
        true,
        variant,
    );
    crate::api::tts_long_form::defer_automatic_budget_to_long_form(
        &mut generation_request,
        variant,
        req.max_output_tokens.or(req.max_tokens),
    );
    let planned_count = if variant.plans_own_speech_segments() {
        crate::api::tts_long_form::SpeechTextPlan::for_variant(
            variant,
            &generation_request.text,
            generation_request.config.options.max_tokens,
        )
        .map_err(core_api_error)?
        .segments
        .len()
    } else {
        expand_generation_requests_for_long_form(&generation_request, variant).len()
    };
    let mut planned_requests = if variant.plans_own_speech_segments() {
        vec![generation_request.clone()]
    } else {
        expand_generation_requests_for_long_form(&generation_request, variant)
    };
    let stream_request_id = generation_request.id.clone();
    let placeholder_record_id = placeholder.id.clone();

    let runtime = state.runtime.clone();
    let speech_store = state.speech_history_store.clone();
    let admission_state = state.clone();

    let (sender, event_rx) = mpsc::channel::<BudgetedStreamEvent>(stream_event_queue_capacity());
    let byte_limit = crate::env_config::tuning().audio_stream_max_event_bytes;
    let event_tx = StreamEventSender {
        sender,
        budget: crate::speech_resource_budget::ByteBudget::new(byte_limit),
    };
    let _ = send_stream_event(
        &event_tx,
        SpeechStreamEvent {
            event: "created",
            timing: None,
            request_id: Some(stream_request_id.clone()),
            sequence: None,
            audio_base64: None,
            sample_count: None,
            sample_rate: None,
            audio_format: None,
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            stop_reason: None,
            output_truncated: None,
            record: Some(placeholder),
            error: None,
        },
    )
    .await;

    let stream_tasks = state.speech_stream_tasks.clone();
    let task_record_id = placeholder_record_id.clone();
    let stream_task = tokio::spawn(async move {
        let mark_failed = |message: String| {
            let event_tx = event_tx.clone();
            let speech_store = speech_store.clone();
            let stream_request_id = stream_request_id.clone();
            let record_id = placeholder_record_id.clone();
            async move {
                let _ = speech_store
                    .update_processing_status(
                        route_kind,
                        record_id,
                        SpeechHistoryProcessingStatus::Failed,
                        Some(message.clone()),
                    )
                    .await;
                let _ = send_stream_event(
                    &event_tx,
                    SpeechStreamEvent {
                        event: "error",
                        timing: None,
                        request_id: Some(stream_request_id),
                        sequence: None,
                        audio_base64: None,
                        sample_count: None,
                        sample_rate: None,
                        audio_format: None,
                        tokens_generated: None,
                        generation_time_ms: None,
                        audio_duration_secs: None,
                        rtf: None,
                        stop_reason: None,
                        output_truncated: None,
                        record: None,
                        error: Some(message),
                    },
                )
                .await;
            }
        };

        let permit = match admission_state
            .acquire_owned_workload_permit(WorkloadClass::Streaming)
            .await
        {
            Ok(permit) => permit,
            Err(_) => {
                mark_failed("Server is shutting down".to_string()).await;
                let _ = send_stream_event(
                    &event_tx,
                    SpeechStreamEvent {
                        event: "done",
                        timing: None,
                        request_id: Some(stream_request_id),
                        sequence: None,
                        audio_base64: None,
                        sample_count: None,
                        sample_rate: None,
                        audio_format: None,
                        tokens_generated: None,
                        generation_time_ms: None,
                        audio_duration_secs: None,
                        rtf: None,
                        stop_reason: None,
                        output_truncated: None,
                        record: None,
                        error: None,
                    },
                )
                .await;
                return;
            }
        };
        if let Err(err) = runtime.load_model(variant).await {
            mark_failed(err.to_string()).await;
            return;
        }
        let mut runtime_context = permit.runtime_context();
        runtime_context.tenant_key = tenant_key;
        for request in &mut planned_requests {
            request.runtime_context = runtime_context;
        }

        let _ = speech_store
            .update_processing_status(
                route_kind,
                placeholder_record_id.clone(),
                SpeechHistoryProcessingStatus::Processing,
                None,
            )
            .await;

        let _permit = if variant.plans_own_speech_segments() {
            drop(permit);
            None
        } else {
            Some(permit)
        };
        let fallback_sample_rate = runtime.sample_rate().await;
        let mut total_samples = 0usize;
        let mut audio_duration_secs = 0.0f32;
        let mut total_tokens = 0usize;
        let mut execution_time_ms = 0.0f32;
        let mut statistics_measured = true;
        let mut stop_reasons = Vec::new();
        let split_request_count = planned_count;
        let mut first_pcm_ms = None;
        let mut last_pcm_ms = None;
        let stream_started = std::time::Instant::now();
        let mut wav_spool: Option<SpeechWavSpool> = None;
        let mut merged_sample_rate: Option<u32> = None;
        let mut global_sequence = 0usize;
        let mut failed = false;
        let mut failure_message: Option<String> = None;
        // A client that leaves (navigation, a closed window) stops the events,
        // not the take: generation runs on into the record, which the history
        // page shows when it is ready. Cancel stops it (`speech_stream_tasks`).
        let mut client_gone = false;
        for (request_index, request) in planned_requests.into_iter().enumerate() {
            // A later long-form segment opens with a short pause and fades in
            // (TPIPE-5); its predecessor's PCM is already sent.
            let mut pause_pending = request_index > 0;
            let mut request_statistics = StreamRequestStatistics::default();
            let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(32);
            let generation_engine = runtime.clone();
            let runner_state = admission_state.clone();
            // Aborting this stream task (cancel) must stop the model too.
            let mut generation_task = AbortOnDrop(tokio::spawn(async move {
                if variant.plans_own_speech_segments() {
                    crate::api::tts_long_form::generate_speech_plan_stream(
                        &runner_state,
                        variant,
                        request,
                        chunk_tx,
                        WorkloadClass::Streaming,
                    )
                    .await
                } else {
                    generation_engine
                        .generate_streaming(request, chunk_tx)
                        .await
                }
            }));

            let mut encoding_failed = false;
            loop {
                let chunk = if client_gone {
                    match chunk_rx.recv().await {
                        Some(chunk) => chunk,
                        None => break,
                    }
                } else {
                    match next_stream_input(&mut chunk_rx, &event_tx.sender).await {
                        StreamInput::Chunk(chunk) => chunk,
                        StreamInput::Finished => break,
                        StreamInput::Disconnected => {
                            client_gone = true;
                            continue;
                        }
                    }
                };
                // Terminal-only chunks carry committed frame counts and execution
                // time; metadata must be consumed even when no PCM remains.
                request_statistics.observe(&chunk);
                let chunk_sample_rate = match prepare_stream_pcm(
                    &event_tx,
                    &stream_request_id,
                    &chunk,
                    fallback_sample_rate,
                    &mut merged_sample_rate,
                )
                .await
                {
                    Ok(Some(sample_rate)) => sample_rate,
                    Ok(None) => continue,
                    Err(message) => {
                        tracing::warn!(
                            request_id = %stream_request_id,
                            record_id = %placeholder_record_id,
                            elapsed_ms = stream_entry_started.elapsed().as_millis(),
                            error = %message,
                            "Speech stream format initialization failed"
                        );
                        encoding_failed = true;
                        failure_message = Some(message);
                        break;
                    }
                };

                let chunk = if pause_pending && !chunk.samples.is_empty() {
                    pause_pending = false;
                    AudioChunk {
                        samples: crate::api::tts_long_form::pause_then_segment(
                            &chunk.samples,
                            chunk_sample_rate,
                        ),
                        ..chunk
                    }
                } else {
                    chunk
                };
                total_samples += chunk.samples.len();
                audio_duration_secs += chunk.samples.len() as f32 / chunk_sample_rate as f32;
                let pcm_ms = stream_entry_started.elapsed().as_secs_f64() * 1000.0;
                first_pcm_ms.get_or_insert(pcm_ms);
                last_pcm_ms = Some(pcm_ms);

                let stream_encoder = AudioEncoder::new(chunk_sample_rate, 1);
                let chunk_bytes = match stream_encoder.encode(&chunk.samples, AudioFormat::RawI16) {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        let _ = send_stream_event(
                            &event_tx,
                            SpeechStreamEvent {
                                event: "error",
                                timing: None,
                                request_id: Some(stream_request_id.clone()),
                                sequence: None,
                                audio_base64: None,
                                sample_count: None,
                                sample_rate: None,
                                audio_format: None,
                                tokens_generated: None,
                                generation_time_ms: None,
                                audio_duration_secs: None,
                                rtf: None,
                                stop_reason: None,
                                output_truncated: None,
                                record: None,
                                error: Some(format!("Failed to encode stream chunk: {err}")),
                            },
                        )
                        .await;
                        encoding_failed = true;
                        failure_message = Some(format!("Failed to encode stream chunk: {err}"));
                        break;
                    }
                };

                let append_result = async {
                    if wav_spool.is_none() {
                        wav_spool = Some(SpeechWavSpool::new(
                            chunk_sample_rate,
                            stream_pcm_byte_limit(variant, chunk_sample_rate, split_request_count),
                        )?);
                    }
                    wav_spool
                        .as_mut()
                        .expect("created WAV spool")
                        .append_pcm(&chunk_bytes)
                        .await
                }
                .await;
                if let Err(err) = append_result {
                    encoding_failed = true;
                    failure_message = Some(format!("Failed to persist streaming PCM: {err}"));
                    break;
                }

                if client_gone {
                    global_sequence = global_sequence.saturating_add(1);
                    continue;
                }
                if let Err(message) = send_stream_event(
                    &event_tx,
                    SpeechStreamEvent {
                        event: "chunk",
                        timing: None,
                        request_id: Some(stream_request_id.clone()),
                        sequence: Some(global_sequence),
                        audio_base64: Some(
                            base64::engine::general_purpose::STANDARD.encode(chunk_bytes),
                        ),
                        sample_count: Some(chunk.samples.len()),
                        sample_rate: Some(chunk_sample_rate),
                        audio_format: Some("pcm_i16"),
                        tokens_generated: None,
                        generation_time_ms: None,
                        audio_duration_secs: None,
                        rtf: None,
                        stop_reason: None,
                        output_truncated: None,
                        record: None,
                        error: None,
                    },
                )
                .await
                {
                    tracing::warn!(
                        request_id = %stream_request_id,
                        record_id = %placeholder_record_id,
                        sequence = global_sequence,
                        elapsed_ms = stream_entry_started.elapsed().as_millis(),
                        error = message,
                        "Speech stream PCM delivery failed; the take continues"
                    );
                    client_gone = true;
                }
                global_sequence = global_sequence.saturating_add(1);
            }

            total_tokens = total_tokens.saturating_add(request_statistics.tokens);
            execution_time_ms += request_statistics.execution_ms;
            statistics_measured &= request_statistics.measured;
            stop_reasons.push(request_statistics.stop_reason);
            drop(chunk_rx);
            if encoding_failed {
                // The take cannot be stored: stop the generation rather than
                // wait for it to finish.
                generation_task.0.abort();
            }
            let generation_outcome = (&mut generation_task.0).await;
            if encoding_failed {
                let _ = generation_outcome;
                failed = true;
                break;
            }

            match generation_outcome {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    failed = true;
                    failure_message = Some(err.to_string());
                    break;
                }
                Err(err) => {
                    failed = true;
                    failure_message = Some(format!("Streaming task failed: {err}"));
                    break;
                }
            }
        }

        if failed {
            mark_failed(stream_terminal_failure_message(failure_message)).await;
        } else {
            let stop_reason = TtsStopReason::combine(stop_reasons);
            let generation_time_ms = stream_started.elapsed().as_secs_f32() * 1000.0;
            let rtf = if audio_duration_secs > 0.0 {
                (generation_time_ms / 1000.0) / audio_duration_secs
            } else {
                0.0
            };

            let wav_result = match wav_spool {
                Some(spool) => spool.finish().await,
                None => Err(anyhow::anyhow!(
                    "Streaming generation produced no PCM audio"
                )),
            };
            match wav_result {
                Ok(wav_bytes) => {
                    let record_result = speech_store
                        .complete_record(
                            route_kind,
                            placeholder_record_id.clone(),
                            CompleteSpeechHistoryRecord {
                                stop_reason: stop_reason.map(|reason| reason.as_str().to_string()),
                                model_id: Some(model_id.clone()),
                                speaker: req.speaker.clone(),
                                language: req.language.clone(),
                                saved_voice_id: req.saved_voice_id.clone(),
                                speed: req.speed.map(f64::from),
                                input_text: input_text.clone(),
                                voice_description: req.voice_description.clone(),
                                reference_text: req.reference_text.clone(),
                                generation_time_ms: generation_time_ms as f64,
                                audio_duration_secs: Some(audio_duration_secs as f64),
                                rtf: Some(rtf as f64),
                                tokens_generated: statistics_measured.then_some(total_tokens),
                                audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav)
                                    .to_string(),
                                audio_filename: Some(default_audio_filename(route_kind, "wav")),
                                audio_bytes: wav_bytes.bytes,
                                preexisting_audio_storage_path: None,
                                generated_codes_json: None,
                            },
                        )
                        .await;

                    match record_result {
                        Ok(Some(record)) => {
                            let _ = send_stream_event(
                                &event_tx,
                                SpeechStreamEvent {
                                    event: "final",
                                    timing: Some(serde_json::json!({
                                        "generation_time_basis": "post_admission_stream_wall",
                                        "request_timing_basis": "stream_handler_entry_after_record_creation",
                                        "execution_time_ms": statistics_measured.then_some(execution_time_ms),
                                        "execution_rtf": if statistics_measured && audio_duration_secs > 0.0 {
                                            Some(execution_time_ms / 1000.0 / audio_duration_secs)
                                        } else { None },
                                        "first_pcm_ms": first_pcm_ms,
                                        "request_to_last_pcm_ms": last_pcm_ms,
                                        "request_to_last_pcm_rtf": last_pcm_ms.filter(|_| audio_duration_secs > 0.0)
                                            .map(|ms| ms / 1000.0 / f64::from(audio_duration_secs)),
                                        "split_request_count": split_request_count,
                                        "pcm_sample_count": total_samples,
                                        "token_unit": if variant.uses_fish_dac_codec() { "semantic_frames" } else { "model_tokens" },
                                    })),
                                    request_id: Some(stream_request_id.clone()),
                                    sequence: None,
                                    audio_base64: None,
                                    sample_count: None,
                                    sample_rate: None,
                                    audio_format: None,
                                    tokens_generated: statistics_measured.then_some(total_tokens),
                                    generation_time_ms: Some(generation_time_ms),
                                    audio_duration_secs: Some(audio_duration_secs),
                                    rtf: Some(rtf),
                                    stop_reason: stop_reason.map(TtsStopReason::as_str),
                                    output_truncated: stop_reason.map(TtsStopReason::truncated),
                                    record: Some(record),
                                    error: None,
                                },
                            )
                            .await;
                        }
                        Ok(None) => {
                            mark_failed("History record not found".to_string()).await;
                        }
                        Err(err) => {
                            mark_failed(format!("Failed to save speech history record: {err}"))
                                .await;
                        }
                    }
                }
                Err(err) => {
                    mark_failed(format!("Failed to finalize streaming WAV output: {err}")).await;
                }
            }
        }

        let _ = send_stream_event(
            &event_tx,
            SpeechStreamEvent {
                event: "done",
                timing: None,
                request_id: Some(stream_request_id),
                sequence: None,
                audio_base64: None,
                sample_count: None,
                sample_rate: None,
                audio_format: None,
                tokens_generated: None,
                generation_time_ms: None,
                audio_duration_secs: None,
                rtf: None,
                stop_reason: None,
                output_truncated: None,
                record: None,
                error: None,
            },
        )
        .await;
    });
    stream_tasks.insert(&task_record_id, stream_task.abort_handle());

    Ok(speech_stream_response(event_rx))
}

fn normalize_create_request(
    mut req: CreateSpeechHistoryRecordRequest,
) -> CreateSpeechHistoryRecordRequest {
    req.model_id = normalize_optional_trimmed(req.model_id);
    req.text = normalize_optional_trimmed(req.text);
    req.speaker = normalize_optional_trimmed(req.speaker);
    req.language = normalize_optional_trimmed(req.language);
    req.voice_description = normalize_optional_trimmed(req.voice_description);
    req.reference_audio = normalize_optional_trimmed(req.reference_audio);
    req.reference_text = normalize_optional_trimmed(req.reference_text);
    req.reference_language = normalize_optional_trimmed(req.reference_language);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    crate::api::speech_dialogue::normalize_references(&mut req.references);
    for id in &mut req.previous_take_ids {
        *id = id.trim().to_string();
    }
    req
}

/// The dialogue fields of `req`, as the dialogue checks read them.
fn speech_dialogue(
    req: &CreateSpeechHistoryRecordRequest,
) -> crate::api::speech_dialogue::SpeechDialogue<'_> {
    crate::api::speech_dialogue::SpeechDialogue {
        single_reference: req.reference_audio.is_some()
            || req.reference_text.is_some()
            || req.saved_voice_id.is_some(),
        references: &req.references,
        script: req.script.as_deref(),
    }
}

async fn resolve_saved_voice_selection(
    state: &AppState,
    variant: ModelVariant,
    mut req: CreateSpeechHistoryRecordRequest,
) -> Result<CreateSpeechHistoryRecordRequest, ApiError> {
    crate::api::speech_dialogue::resolve_saved_voices(state, &mut req.references).await?;
    let Some(saved_voice_id) = req.saved_voice_id.as_deref() else {
        return Ok(req);
    };

    let saved_voice = resolve_saved_voice_reference(state, saved_voice_id).await?;
    req.saved_voice_id = Some(saved_voice.voice_id);
    req.reference_audio = Some(saved_voice.reference_audio_base64);
    req.reference_text = saved_voice.reference_text;
    // The voice's language serves a model that reads one; a request's own
    // `reference_language` wins.
    if req.reference_language.is_none() && crate::api::tts_policy::takes_reference_language(variant)
    {
        req.reference_language = saved_voice.reference_language;
    }
    Ok(req)
}

fn validate_reference_voice_selection(
    req: &CreateSpeechHistoryRecordRequest,
) -> Result<(), ApiError> {
    let has_direct_reference_audio = req
        .reference_audio
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    let has_direct_reference_text = req
        .reference_text
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    // Whether the transcript is needed too depends on the model, which the
    // voice contract checks once the model is known.
    if has_direct_reference_text && !has_direct_reference_audio {
        return Err(ApiError::bad_request(
            "`reference_text` was given without `reference_audio`.",
        ));
    }

    let has_saved_voice = req
        .saved_voice_id
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    if !has_saved_voice {
        return Ok(());
    }
    if has_direct_reference_audio || has_direct_reference_text {
        return Err(ApiError::bad_request(
            "Use either `saved_voice_id` or direct `reference_audio`/`reference_text`, not both.",
        ));
    }

    Ok(())
}

fn build_generation_request(
    req: CreateSpeechHistoryRecordRequest,
    correlation_id: String,
    text: String,
    streaming: bool,
    variant: ModelVariant,
) -> GenerationRequest {
    let mut generation_config = GenerationConfig {
        streaming,
        ..GenerationConfig::default()
    };
    if let Some(temp) = req.temperature {
        generation_config.options.temperature = temp;
    }
    if let Some(speed) = req.speed {
        generation_config.options.speed = speed;
    }
    if let Some(max_tokens) = resolve_speech_history_output_frames(
        variant,
        &text,
        req.max_output_tokens.or(req.max_tokens),
    ) {
        generation_config.options.max_tokens = if variant.plans_own_speech_segments() {
            req.max_output_tokens.or(req.max_tokens).unwrap_or(0)
        } else {
            max_tokens
        };
    }
    if let Some(top_k) = req.top_k {
        generation_config.options.top_k = top_k;
    }
    generation_config.options.speech = req.sampling.options(
        req.temperature,
        crate::api::tts_policy::speech_top_k(variant, req.top_k),
    );
    generation_config.options.speaker = req.speaker.clone();

    GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        model_variant: Some(variant),
        correlation_id: Some(correlation_id),
        runtime_context: Default::default(),
        text,
        config: generation_config,
        language: req.language,
        reference_audio: req.reference_audio,
        reference_text: req.reference_text,
        reference_language: req.reference_language,
        speaker_references: crate::api::speech_dialogue::speaker_references(&req.references),
        voice_description: req.voice_description,
        chunk_history: Vec::new(),
    }
}

fn resolve_speech_history_output_frames(
    variant: ModelVariant,
    text: &str,
    requested_frames: Option<usize>,
) -> Option<usize> {
    resolve_tts_output_frames(variant, text, requested_frames)
}

fn normalize_for_model_capabilities(
    route_kind: SpeechRouteKind,
    variant: ModelVariant,
    req: CreateSpeechHistoryRecordRequest,
) -> Result<CreateSpeechHistoryRecordRequest, ApiError> {
    let capabilities = variant.speech_capabilities().ok_or_else(|| {
        ApiError::bad_request(format!(
            "{variant} does not expose speech generation capabilities."
        ))
    })?;

    let has_speaker = req
        .speaker
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    // Each dialogue reference's transcript is checked on its own; the
    // contract then reads them as the request's reference.
    let dialogue = !req.references.is_empty();
    let has_reference_audio = dialogue
        || req
            .reference_audio
            .as_deref()
            .map(has_non_empty_text)
            .unwrap_or(false);
    let has_reference_text = dialogue
        || req
            .reference_text
            .as_deref()
            .map(has_non_empty_text)
            .unwrap_or(false);
    let has_voice_description = req
        .voice_description
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    let has_saved_voice = req
        .saved_voice_id
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);

    crate::api::tts_policy::validate_speech_speed(variant, req.speed)?;
    crate::api::tts_policy::validate_speech_controls(variant, &req.sampling)?;
    crate::api::tts_policy::validate_speech_language(variant, req.language.as_deref())?;
    crate::api::tts_policy::validate_speech_reference_language(
        variant,
        req.reference_language.as_deref(),
    )?;
    crate::api::tts_policy::validate_speech_inline_tags(
        variant,
        req.text.as_deref().unwrap_or_default(),
    )?;
    // Caller text holding the model's control tokens would rewrite its
    // prompt.
    crate::api::tts_policy::validate_speech_control_tokens(
        variant,
        "text",
        req.text.as_deref().unwrap_or_default(),
    )?;
    if let Some(reference_text) = req.reference_text.as_deref() {
        crate::api::tts_policy::validate_speech_control_tokens(
            variant,
            "reference_text",
            reference_text,
        )?;
    }
    crate::api::speech_dialogue::check_dialogue_references(
        variant,
        &req.references,
        req.text.as_deref().unwrap_or_default(),
    )?;

    if req.stream.unwrap_or(false) && !capabilities.supports_streaming {
        return Err(ApiError::bad_request(format!(
            "{variant} does not support streaming synthesis. Disable `stream` or choose a model with streaming support.",
        )));
    }

    match route_kind {
        SpeechRouteKind::VoiceDesign => {
            if !capabilities.supports_voice_description {
                return Err(ApiError::bad_request(
                    "Voice design requests require a model with `voice_description` support.",
                ));
            }
            if !has_voice_description {
                return Err(ApiError::bad_request(
                    "Voice design requests require `voice_description`.",
                ));
            }
        }
        SpeechRouteKind::VoiceCloning => {
            if !capabilities.supports_reference_voice {
                return Err(ApiError::bad_request(
                    "Voice cloning requests require a model with reference-voice support.",
                ));
            }
            // A saved voice's missing transcript is the voice contract's
            // to report, below.
            if capabilities.reference_transcript_required
                && !has_saved_voice
                && (!has_reference_audio || !has_reference_text)
            {
                return Err(ApiError::bad_request(
                    "Voice cloning requests require `saved_voice_id` or both `reference_audio` and `reference_text`.",
                ));
            }
            if !has_reference_audio {
                return Err(ApiError::bad_request(
                    "Voice cloning requests require `saved_voice_id` or `reference_audio`.",
                ));
            }
        }
        SpeechRouteKind::TextToSpeech | SpeechRouteKind::Studio => {}
    }
    // The voice contract every speech route shares (TPIPE-12).
    crate::api::tts_policy::validate_speech_voice(
        variant,
        crate::api::tts_policy::SpeechVoiceRequest {
            speaker: has_speaker,
            reference_audio: has_reference_audio,
            reference_text: has_reference_text,
            saved_voice: has_saved_voice,
            voice_description: has_voice_description,
        },
    )?;

    Ok(req)
}

fn required_trimmed(raw: Option<&str>, field_name: &str) -> Result<String, ApiError> {
    let trimmed = raw.unwrap_or("").trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Missing required `{field_name}` field."
        )));
    }
    Ok(trimmed.to_string())
}

fn normalize_optional_trimmed(raw: Option<String>) -> Option<String> {
    let trimmed = raw.unwrap_or_default().trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn has_non_empty_text(raw: &str) -> bool {
    !raw.trim().is_empty()
}

fn default_audio_filename(route_kind: SpeechRouteKind, extension: &str) -> String {
    let prefix = match route_kind {
        SpeechRouteKind::TextToSpeech => "tts",
        SpeechRouteKind::VoiceDesign => "voice-design",
        SpeechRouteKind::VoiceCloning => "voice-cloning",
        SpeechRouteKind::Studio => "studio",
    };
    format!("{prefix}-{}.{}", uuid::Uuid::new_v4().simple(), extension)
}

fn stream_terminal_failure_message(message: Option<String>) -> String {
    message.unwrap_or_else(|| STREAM_CLIENT_DISCONNECTED_MESSAGE.to_string())
}

fn to_stream_json(event: SpeechStreamEvent) -> String {
    serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string())
}

/// Mono samples and rate of a stored WAV (16-bit or float; channels are
/// averaged).
fn decode_wav_to_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32), ApiError> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|err| {
        ApiError::bad_request(format!(
            "This audio is not WAV, so it can only be downloaded as stored: {err}"
        ))
    })?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()
        }
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<Result<_, _>>(),
    }
    .map_err(|err| ApiError::internal(format!("Failed reading history audio: {err}")))?;
    let samples = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((samples, spec.sample_rate))
}

fn audio_response(audio: StoredSpeechAudio, as_attachment: bool) -> Response {
    let mut response = Response::builder().status(StatusCode::OK);

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.header(header::CONTENT_TYPE, content_type);
    }

    let disposition = if as_attachment {
        audio
            .audio_filename
            .as_deref()
            .map(|filename| format!("attachment; filename=\"{}\"", filename.replace('"', "")))
            .unwrap_or_else(|| "attachment".to_string())
    } else if let Some(filename) = audio.audio_filename.as_deref() {
        format!("inline; filename=\"{}\"", filename.replace('"', ""))
    } else {
        "inline".to_string()
    };
    if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
        response = response.header(header::CONTENT_DISPOSITION, value);
    }

    response
        .body(Body::from(audio.audio_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Speech history storage error")
}

fn map_media_ingest_error(err: MediaIngestError) -> ApiError {
    if err.is_invalid_input() {
        ApiError::bad_request(err.to_string())
    } else {
        ApiError::internal(format!("Speech reference media ingest error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    /// A cancelled stage drops its synthesis at once rather than waiting for
    /// the utterance, so the engine work it owns stops mid-render.
    #[tokio::test]
    async fn cancelling_a_stage_drops_its_synthesis() {
        struct DropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = DropFlag(dropped.clone());
        let work = async move {
            let _flag = flag;
            std::future::pending::<()>().await
        };
        let cancellation = super::StageCancellationSignal::new();
        let trigger = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            trigger.cancel(super::StageCancellationReason::UserRequested);
        });
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            super::unless_cancelled(Some(cancellation), work),
        )
        .await
        .expect("cancellation ends the wait");
        assert_eq!(outcome, Err(super::StageCancellationReason::UserRequested));
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        // Without a stage there is nothing to cancel.
        assert_eq!(super::unless_cancelled(None, async { 7 }).await, Ok(7));
    }

    #[tokio::test]
    async fn stream_start_uses_first_pcm_rate_once_across_requests() {
        for explicit_rate in [Some(44_100), None] {
            let (sender, mut receiver) = mpsc::channel(4);
            let event_tx = StreamEventSender {
                sender,
                budget: crate::speech_resource_budget::ByteBudget::new(4096),
            };
            let mut merged_rate = None;
            // Metadata-only chunks must not start playback at a fallback rate.
            let empty = AudioChunk::final_chunk("first".into(), 0, Vec::new());
            assert_eq!(
                prepare_stream_pcm(&event_tx, "stream", &empty, 24_000, &mut merged_rate)
                    .await
                    .unwrap(),
                None
            );
            assert!(receiver.try_recv().is_err());
            let mut first = AudioChunk::new("first".into(), 0, vec![0.25; 2048]);
            if let Some(rate) = explicit_rate {
                first = first.with_sample_rate(rate);
            }
            let rate = prepare_stream_pcm(&event_tx, "stream", &first, 24_000, &mut merged_rate)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(rate, explicit_rate.unwrap_or(24_000));
            let event = receiver.recv().await.unwrap();
            let start: serde_json::Value =
                serde_json::from_str(event.payload.strip_prefix("data: ").unwrap().trim()).unwrap();
            assert_eq!(start["event"], "start");
            assert_eq!(start["sample_rate"], rate);
            assert_eq!(start["audio_format"], "pcm_i16");
            // Playback and the finalized WAV must interpret these same samples
            // at the same rate; otherwise pitch and duration diverge.
            let wav = AudioEncoder::new(rate, 1)
                .encode(&first.samples, AudioFormat::Wav)
                .unwrap();
            let wav_rate = u32::from_le_bytes(wav[24..28].try_into().unwrap());
            assert_eq!(start["sample_rate"], wav_rate);
            assert_eq!(merged_rate, Some(rate));
            let next = AudioChunk::new("second".into(), 0, vec![0.25; 2048]).with_sample_rate(rate);
            assert_eq!(
                prepare_stream_pcm(&event_tx, "stream", &next, 24_000, &mut merged_rate)
                    .await
                    .unwrap(),
                Some(rate)
            );
            assert!(
                receiver.try_recv().is_err(),
                "split requests must not restart playback"
            );
            let changed = next.with_sample_rate(48_000);
            assert!(
                prepare_stream_pcm(&event_tx, "stream", &changed, 24_000, &mut merged_rate)
                    .await
                    .unwrap_err()
                    .contains("Streaming sample rate changed")
            );
            assert_eq!(merged_rate, Some(rate));
            assert!(receiver.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn stream_start_delivery_failure_does_not_commit_format() {
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        let event_tx = StreamEventSender {
            sender,
            budget: crate::speech_resource_budget::ByteBudget::new(4096),
        };
        let mut merged_rate = None;
        let chunk = AudioChunk::new("request".into(), 0, vec![0.25]).with_sample_rate(44_100);
        assert_eq!(
            prepare_stream_pcm(&event_tx, "stream", &chunk, 24_000, &mut merged_rate)
                .await
                .unwrap_err(),
            STREAM_CLIENT_DISCONNECTED_MESSAGE
        );
        assert_eq!(merged_rate, None);
    }

    #[tokio::test(start_paused = true)]
    async fn sse_keepalive_survives_idle_preparation_then_delivers_data_and_eof() {
        use futures::StreamExt;
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let response = super::speech_stream_response(receiver);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        assert_eq!(
            response.headers()["cache-control"],
            "no-cache, no-transform"
        );
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let mut body = response.into_body().into_data_stream();
        let started = tokio::time::Instant::now();
        // Simulate a reader/proxy that closes an idle stream after 15 seconds.
        // Preparation takes 40 seconds, but regular comment frames keep it alive.
        for _ in 0..4 {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(15), body.next())
                .await
                .expect("idle client closed before first audio")
                .unwrap()
                .unwrap();
            assert_eq!(frame.as_ref(), b": keepalive\n\n");
        }
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(40));
        let payload = "data: {\"event\":\"chunk\"}\n\n".to_string();
        let budget = crate::speech_resource_budget::ByteBudget::new(payload.len());
        let other = crate::speech_resource_budget::ByteBudget::new(payload.len());
        sender
            .send(super::BudgetedStreamEvent {
                _global_reservation: budget.reserve(payload.len()).unwrap(),
                _stream_reservation: other.reserve(payload.len()).unwrap(),
                payload: payload.clone(),
            })
            .await
            .unwrap();
        let frame = body.next().await.unwrap().unwrap();
        assert_eq!(frame.as_ref(), payload.as_bytes());
        assert!(budget.reserve(1).is_err());
        drop(frame);
        assert!(budget.reserve(payload.len()).is_ok());
        drop(sender);
        assert!(body.next().await.is_none());
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(40));
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_sse_body_releases_queued_bytes_and_closes_sender() {
        for read_keepalive in [false, true] {
            use futures::StreamExt;
            let (sender, receiver) = tokio::sync::mpsc::channel(1);
            let mut body = super::speech_stream_response(receiver)
                .into_body()
                .into_data_stream();
            if read_keepalive {
                assert!(body.next().await.unwrap().is_ok());
            }
            let budget = crate::speech_resource_budget::ByteBudget::new(32);
            let other = crate::speech_resource_budget::ByteBudget::new(32);
            sender
                .send(super::BudgetedStreamEvent {
                    payload: "data: pending\n\n".into(),
                    _global_reservation: budget.reserve(32).unwrap(),
                    _stream_reservation: other.reserve(32).unwrap(),
                })
                .await
                .unwrap();
            assert!(budget.reserve(1).is_err());
            drop(body);
            assert!(sender.is_closed());
            assert!(budget.reserve(32).is_ok());
            assert!(other.reserve(32).is_ok());
        }
    }

    #[test]
    fn sse_byte_reservation_follows_transport_frame_and_clones() {
        let budget = crate::speech_resource_budget::ByteBudget::new(16);
        let other = crate::speech_resource_budget::ByteBudget::new(16);
        let event = super::BudgetedStreamEvent {
            payload: "data: test\n\n".to_string(),
            _global_reservation: budget.reserve(16).unwrap(),
            _stream_reservation: other.reserve(16).unwrap(),
        };
        let frame = bytes::Bytes::from_owner(event);
        let in_transport = frame.clone();
        drop(frame);
        assert!(budget.reserve(1).is_err());
        assert!(other.reserve(1).is_err());
        drop(in_transport);
        assert!(budget.reserve(16).is_ok());
        assert!(other.reserve(16).is_ok());
    }

    use super::*;

    fn base_request() -> CreateSpeechHistoryRecordRequest {
        CreateSpeechHistoryRecordRequest {
            model_id: Some("Qwen3-TTS-12Hz-1.7B-Base".to_string()),
            text: Some("Hello".to_string()),
            speaker: None,
            language: None,
            voice_description: None,
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            previous_take_ids: Vec::new(),
            temperature: None,
            speed: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: None,
            sampling: Default::default(),
        }
    }

    #[test]
    fn idempotency_headers_are_bounded_text_and_unambiguous() {
        let mut headers = HeaderMap::new();
        headers.insert("idempotency-key", HeaderValue::from_static(" request-1 "));
        headers.insert("x-idempotency-key", HeaderValue::from_static("request-1"));
        assert_eq!(
            extract_idempotency_key(&headers).unwrap().as_deref(),
            Some("request-1")
        );

        headers.insert("x-idempotency-key", HeaderValue::from_static("request-2"));
        assert_eq!(
            extract_idempotency_key(&headers).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );

        let mut headers = HeaderMap::new();
        headers.insert("idempotency-key", HeaderValue::from_static(""));
        assert_eq!(
            extract_idempotency_key(&headers).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );

        let mut headers = HeaderMap::new();
        headers.insert(
            "idempotency-key",
            HeaderValue::from_str(
                &"x".repeat(crate::batch_runtime::store::MAX_DURABLE_IDEMPOTENCY_KEY_BYTES + 1),
            )
            .unwrap(),
        );
        assert_eq!(
            extract_idempotency_key(&headers).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );

        let mut headers = HeaderMap::new();
        headers.insert(
            "idempotency-key",
            HeaderValue::from_bytes(&[0x80]).expect("opaque header value"),
        );
        assert_eq!(
            extract_idempotency_key(&headers).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn durable_tts_digest_normalizes_aliases_and_excludes_tenant_scope() {
        let mut first = base_request();
        first.max_tokens = Some(128);
        normalize_durable_text_tts_request(&mut first);
        let mut first = BatchSpeechRequest::for_durable_job(
            SpeechRouteKind::TextToSpeech,
            "Qwen3-TTS-12Hz-1.7B-Base".into(),
            "Hello".into(),
            first,
        );
        first.tenant_key = Some([1; 32]);

        let mut equivalent = base_request();
        equivalent.stream = Some(false);
        equivalent.max_output_tokens = Some(128);
        normalize_durable_text_tts_request(&mut equivalent);
        let mut equivalent = BatchSpeechRequest::for_durable_job(
            SpeechRouteKind::TextToSpeech,
            "Qwen3-TTS-12Hz-1.7B-Base".into(),
            "Hello".into(),
            equivalent,
        );
        equivalent.tenant_key = Some([2; 32]);

        assert_eq!(
            durable_tts_request_digest(&first, None, &[], &[]).unwrap(),
            durable_tts_request_digest(&equivalent, None, &[], &[]).unwrap()
        );

        // A reference voice is part of the request; its payload identifies it.
        let with_reference =
            durable_tts_request_digest(&first, Some("UklGRg=="), &[], &[]).unwrap();
        assert_ne!(
            durable_tts_request_digest(&first, None, &[], &[]).unwrap(),
            with_reference
        );
        assert_eq!(
            durable_tts_request_digest(&equivalent, Some(" UklGRg== "), &[], &[]).unwrap(),
            with_reference
        );
        assert_ne!(
            durable_tts_request_digest(&first, Some("UklGRh=="), &[], &[]).unwrap(),
            with_reference
        );

        // Dialogue references are identified by their payloads, in order.
        let reference = |audio: &str| crate::api::speech_dialogue::SpeechDialogueReference {
            speaker: Some(1),
            audio: Some(audio.into()),
            text: Some("Hi.".into()),
            saved_voice_id: None,
        };
        let two = [reference("UklGRg=="), reference("UklGRh==")];
        let dialogue = durable_tts_request_digest(&first, None, &two, &[]).unwrap();
        assert_ne!(
            dialogue,
            durable_tts_request_digest(&first, None, &[], &[]).unwrap()
        );
        assert_ne!(dialogue, with_reference);
        assert_eq!(
            durable_tts_request_digest(&equivalent, None, &two, &[]).unwrap(),
            dialogue
        );
        let swapped = [reference("UklGRh=="), reference("UklGRg==")];
        assert_ne!(
            durable_tts_request_digest(&first, None, &swapped, &[]).unwrap(),
            dialogue
        );

        // The takes a request continues are identified by their stored
        // codes, in order; a request without them digests as before.
        let takes = ["a".to_string(), "b".to_string()];
        let continued = durable_tts_request_digest(&first, None, &[], &takes).unwrap();
        assert_ne!(
            continued,
            durable_tts_request_digest(&first, None, &[], &[]).unwrap()
        );
        assert_eq!(
            durable_tts_request_digest(&equivalent, None, &[], &takes).unwrap(),
            continued
        );
        let reordered = ["b".to_string(), "a".to_string()];
        assert_ne!(
            durable_tts_request_digest(&first, None, &[], &reordered).unwrap(),
            continued
        );

        equivalent.input_text = "Different".into();
        assert_ne!(
            durable_tts_request_digest(&first, None, &[], &[]).unwrap(),
            durable_tts_request_digest(&equivalent, None, &[], &[]).unwrap()
        );
    }

    /// Drama 3's own allowance: each chunk may make its 4,096-frame budget
    /// at 44.1 kHz, from its own frame rate.
    #[test]
    fn stream_spool_allowance_covers_fish_drama3_chunks() {
        let single = stream_pcm_byte_allowance(ModelVariant::FishAudioDrama3, 44_100, 1);
        assert!(single >= ModelVariant::FISH_DRAMA3_MAX_OUTPUT_FRAMES * 2048 * 2 * 4);
        let three = stream_pcm_byte_allowance(ModelVariant::FishAudioDrama3, 44_100, 3);
        assert!(three >= single * 3 - 3 && three <= single * 3);
    }

    #[test]
    fn stream_spool_allowance_preserves_fish_full_output_and_long_form() {
        let single = stream_pcm_byte_allowance(ModelVariant::FishAudioS2Pro, 44_100, 1);
        let full_slow_pcm = ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES * 2048 * 2 * 4;
        assert!(single >= full_slow_pcm);
        let long_form = stream_pcm_byte_allowance(ModelVariant::FishAudioS2Pro, 44_100, 3);
        assert!(long_form >= full_slow_pcm * 3);
        assert!(long_form <= single * 3);
        assert!(
            stream_pcm_byte_allowance(ModelVariant::FishAudioS2Pro, 44_100, usize::MAX)
                <= u32::MAX as usize - 44
        );
    }

    #[test]
    fn terminal_only_statistics_replace_chunk_deltas_without_guessing_frames() {
        let mut statistics = StreamRequestStatistics::default();
        let mut first = AudioChunk::new("request".into(), 0, vec![0.0; 2048]);
        first.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 10.0,
            tokens_generated: 1,
            rtf: 0.2,
            stop_reason: None,
            diagnostics: None,
        });
        statistics.observe(&first);
        let mut terminal = AudioChunk::final_chunk("request".into(), 1, Vec::new());
        terminal.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 25.0,
            tokens_generated: 2,
            rtf: 0.3,
            stop_reason: None,
            diagnostics: None,
        });
        statistics.observe(&terminal);
        assert_eq!(statistics.tokens, 2);
        assert_eq!(statistics.execution_ms, 25.0);
        let mut unknown = StreamRequestStatistics::default();
        unknown.observe(&AudioChunk::final_chunk(
            "unknown".into(),
            0,
            vec![0.0; 4096],
        ));
        assert_eq!(unknown.tokens, 0);
    }

    /// TPIPE-8: a stream notices the client leaving while the model is still
    /// generating, before its next chunk arrives.
    #[tokio::test]
    async fn a_departed_client_is_noticed_before_the_next_chunk() {
        let (_chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(1);
        let (events, event_rx) = mpsc::channel::<BudgetedStreamEvent>(1);
        drop(event_rx);
        let input = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            next_stream_input(&mut chunk_rx, &events),
        )
        .await
        .expect("a closed client is noticed without waiting for audio");
        assert!(matches!(input, StreamInput::Disconnected));

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(1);
        let (events, _event_rx) = mpsc::channel::<BudgetedStreamEvent>(1);
        chunk_tx
            .send(AudioChunk::new("r".into(), 0, vec![0.1]))
            .await
            .unwrap();
        drop(chunk_tx);
        assert!(matches!(
            next_stream_input(&mut chunk_rx, &events).await,
            StreamInput::Chunk(_)
        ));
        assert!(matches!(
            next_stream_input(&mut chunk_rx, &events).await,
            StreamInput::Finished
        ));
    }

    /// The first-party stream reports why each request ended: the terminal
    /// chunk's reason is kept, and the `final` event names the combined one.
    #[test]
    fn stream_final_event_reports_the_stop_reason() {
        let mut statistics = StreamRequestStatistics::default();
        let mut terminal = AudioChunk::final_chunk("request".into(), 1, Vec::new());
        terminal.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 25.0,
            tokens_generated: 313,
            rtf: 0.3,
            stop_reason: Some(TtsStopReason::OutputBudget),
            diagnostics: None,
        });
        statistics.observe(&terminal);
        let stop_reason =
            TtsStopReason::combine([Some(TtsStopReason::EndOfSpeech), statistics.stop_reason]);
        let event = SpeechStreamEvent {
            event: "final",
            timing: None,
            request_id: None,
            sequence: None,
            audio_base64: None,
            sample_count: None,
            sample_rate: None,
            audio_format: None,
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            stop_reason: stop_reason.map(TtsStopReason::as_str),
            output_truncated: stop_reason.map(TtsStopReason::truncated),
            record: None,
            error: None,
        };
        assert_eq!(
            to_stream_json(event),
            r#"{"event":"final","stop_reason":"length","output_truncated":true}"#
        );
    }

    #[test]
    fn new_batch_request_json_omits_embedded_reference_audio() {
        let mut request = base_request();
        request.reference_audio = Some("large-base64-audio".to_string());
        request.reference_text = Some("Reference words".to_string());

        let snapshot = BatchSpeechRequest::for_durable_job(
            SpeechRouteKind::TextToSpeech,
            "Qwen3-TTS-12Hz-1.7B-Base".to_string(),
            "Hello".to_string(),
            request,
        );
        let json = serde_json::to_value(snapshot).expect("serialize batch request");

        assert!(json["request"]["model_id"].is_null());
        assert!(json["request"]["text"].is_null());
        assert!(json["request"]["reference_audio"].is_null());
        assert_eq!(json["request"]["reference_text"], "Reference words");
        // A request without dialogue stores no dialogue fields, so its
        // idempotency digest is what it was before them.
        let request = json["request"].as_object().unwrap();
        assert!(!request.contains_key("references") && !request.contains_key("script"));
    }

    /// A dialogue request's stored copy keeps each reference's speaker,
    /// transcript and saved voice, but not its audio (an input artifact
    /// each) nor the script (the record's text is the rendered script).
    #[test]
    fn dialogue_batch_request_json_omits_reference_audio_and_script() {
        use crate::api::speech_dialogue::{SpeechDialogueReference, SpeechScriptLine};
        let mut request = base_request();
        request.model_id = Some("FishAudio-Drama-3".into());
        request.references = vec![
            SpeechDialogueReference {
                speaker: Some(0),
                audio: Some("large-base64-audio".into()),
                text: Some("Hi.".into()),
                saved_voice_id: None,
            },
            SpeechDialogueReference {
                speaker: Some(1),
                audio: Some("saved-voice-audio".into()),
                text: Some("Yo.".into()),
                saved_voice_id: Some("voice-1".into()),
            },
        ];
        request.script = Some(vec![SpeechScriptLine {
            speaker: 0,
            text: "Hi.".into(),
        }]);
        let snapshot = BatchSpeechRequest::for_durable_job(
            SpeechRouteKind::TextToSpeech,
            "FishAudio-Drama-3".into(),
            "<|speaker:0|> Hi.".into(),
            request,
        );
        let json = serde_json::to_value(&snapshot).expect("serialize batch request");
        assert!(json["request"]["script"].is_null());
        assert_eq!(
            json["request"]["references"],
            serde_json::json!([
                {"speaker": 0, "text": "Hi."},
                {"speaker": 1, "text": "Yo.", "saved_voice_id": "voice-1"},
            ])
        );
        let decoded: BatchSpeechRequest = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.request.references.len(), 2);
        assert!(decoded.request.references.iter().all(|r| r.audio.is_none()));
    }

    #[test]
    fn legacy_batch_request_json_still_reads_embedded_reference_audio() {
        let mut request = base_request();
        request.reference_audio = Some("legacy-base64-audio".to_string());
        request.reference_text = Some("Reference words".to_string());
        let json = serde_json::to_value(BatchSpeechRequest {
            tenant_key: None,
            route_kind: SpeechRouteKind::TextToSpeech,
            model_id: "Qwen3-TTS-12Hz-1.7B-Base".to_string(),
            input_text: "Hello".to_string(),
            request,
        })
        .expect("serialize legacy request");

        let decoded: BatchSpeechRequest =
            serde_json::from_value(json).expect("decode legacy batch request");
        assert_eq!(
            decoded.request.reference_audio.as_deref(),
            Some("legacy-base64-audio")
        );
    }

    #[test]
    fn durable_tenant_identity_roundtrips_and_public_payload_cannot_supply_it() {
        let mut raw = serde_json::to_value(base_request()).unwrap();
        raw["tenant_key"] = serde_json::json!(vec![19; 32]);
        let public: CreateSpeechHistoryRecordRequest = serde_json::from_value(raw).unwrap();
        let mut internal = BatchSpeechRequest::for_durable_job(
            SpeechRouteKind::TextToSpeech,
            "FishAudio-S2-Pro".into(),
            "hello".into(),
            public,
        );
        assert_eq!(
            internal.tenant_key, None,
            "public fields must not become trusted scheduling identity"
        );
        internal.tenant_key = Some([7; 32]);
        let persisted = serde_json::to_value(&internal).unwrap();
        let restored: BatchSpeechRequest = serde_json::from_value(persisted.clone()).unwrap();
        assert_eq!(restored.tenant_key, Some([7; 32]));
        let mut legacy = persisted;
        legacy.as_object_mut().unwrap().remove("tenant_key");
        assert_eq!(
            serde_json::from_value::<BatchSpeechRequest>(legacy)
                .unwrap()
                .tenant_key,
            None
        );
    }

    #[test]
    fn normalize_create_request_trims_optional_fields() {
        let normalized = normalize_create_request(CreateSpeechHistoryRecordRequest {
            model_id: Some("  Qwen3-TTS-12Hz-1.7B-Base  ".to_string()),
            text: Some("  Hello there  ".to_string()),
            saved_voice_id: Some("  voice-1  ".to_string()),
            ..base_request()
        });

        assert_eq!(
            normalized.model_id.as_deref(),
            Some("Qwen3-TTS-12Hz-1.7B-Base")
        );
        assert_eq!(normalized.text.as_deref(), Some("Hello there"));
        assert_eq!(normalized.saved_voice_id.as_deref(), Some("voice-1"));
    }

    #[test]
    fn stream_terminal_failure_message_defaults_to_client_disconnect() {
        assert_eq!(
            stream_terminal_failure_message(None),
            STREAM_CLIENT_DISCONNECTED_MESSAGE
        );
        assert_eq!(
            stream_terminal_failure_message(Some("encoder failed".to_string())),
            "encoder failed"
        );
    }

    #[test]
    fn mixed_saved_voice_and_direct_reference_is_rejected() {
        let err = validate_reference_voice_selection(&CreateSpeechHistoryRecordRequest {
            saved_voice_id: Some("voice-1".to_string()),
            reference_audio: Some("abc".to_string()),
            reference_text: Some("hello".to_string()),
            ..base_request()
        })
        .expect_err("expected mixed reference inputs to fail");

        assert!(err
            .message
            .contains("Use either `saved_voice_id` or direct `reference_audio`/`reference_text`"));
    }

    #[test]
    fn reference_text_needs_reference_audio() {
        let err = validate_reference_voice_selection(&CreateSpeechHistoryRecordRequest {
            reference_text: Some("words".to_string()),
            ..base_request()
        })
        .expect_err("a transcript alone must fail");
        assert!(err.message.contains("without `reference_audio`"));

        // Audio alone is the model's call (see the voice-contract tests).
        validate_reference_voice_selection(&CreateSpeechHistoryRecordRequest {
            reference_audio: Some("abc".to_string()),
            ..base_request()
        })
        .expect("audio alone passes the model-independent check");
    }

    /// Fish Audio Drama 3 on the first-party routes: text to speech without
    /// a voice, a clone with its transcript; the cloning route still needs
    /// a reference; control tokens are refused in the text and the
    /// transcript; and `top_k: 0` reaches the engine as no cap.
    #[test]
    fn fish_drama3_first_party_requests_follow_its_voice_and_token_contract() {
        let drama3 = ModelVariant::FishAudioDrama3;
        let request = |text: &str, reference: Option<(&str, Option<&str>)>| {
            CreateSpeechHistoryRecordRequest {
                model_id: Some("FishAudio-Drama-3".to_string()),
                text: Some(text.to_string()),
                reference_audio: reference.map(|(audio, _)| audio.to_string()),
                reference_text: reference.and_then(|(_, text)| text.map(str::to_string)),
                top_k: Some(0),
                ..base_request()
            }
        };
        normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            drama3,
            request("[sighs] Hello.", None),
        )
        .expect("a seeded voice");
        normalize_for_model_capabilities(
            SpeechRouteKind::VoiceCloning,
            drama3,
            request("Hello.", Some(("UklGRg==", Some("Hi.")))),
        )
        .expect("a clone");
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::VoiceCloning,
            drama3,
            request("Hello.", None),
        )
        .unwrap_err();
        assert!(
            err.message.contains("require `saved_voice_id`"),
            "{}",
            err.message
        );
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            drama3,
            request("Hello.", Some(("UklGRg==", None))),
        )
        .unwrap_err();
        assert!(err.message.contains("Provide both"), "{}", err.message);
        for variant in [drama3, ModelVariant::FishAudioS2Pro] {
            let err = normalize_for_model_capabilities(
                SpeechRouteKind::TextToSpeech,
                variant,
                request("Hi <|semantic:7|>", Some(("UklGRg==", Some("Hi.")))),
            )
            .unwrap_err();
            assert!(
                err.message.contains("`text` contains `<|semantic:7|>`"),
                "{}",
                err.message
            );
            let err = normalize_for_model_capabilities(
                SpeechRouteKind::TextToSpeech,
                variant,
                request("Hello.", Some(("UklGRg==", Some("<|im_start|>user")))),
            )
            .unwrap_err();
            assert!(
                err.message
                    .contains("`reference_text` contains `<|im_start|>`"),
                "{}",
                err.message
            );
        }
        let generation = build_generation_request(
            request("Hello.", None),
            "c".into(),
            "Hello.".into(),
            true,
            drama3,
        );
        assert_eq!(generation.config.options.speech.top_k, None);
        // The plan's budget: raw, as S2's is (the plan resolves it per chunk).
        assert_eq!(generation.config.options.max_tokens, 0);
    }

    #[test]
    fn audio_only_cloning_is_for_models_that_take_no_transcript() {
        let audio_only = || CreateSpeechHistoryRecordRequest {
            reference_audio: Some("UklGRg==".to_string()),
            ..base_request()
        };
        normalize_for_model_capabilities(
            SpeechRouteKind::VoiceCloning,
            ModelVariant::Chatterbox500M,
            audio_only(),
        )
        .expect("Chatterbox clones from audio alone");
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::VoiceCloning,
            ModelVariant::FishAudioS2Pro,
            audio_only(),
        )
        .expect_err("Fish conditions on the transcript");
        assert!(err
            .message
            .contains("require `saved_voice_id` or both `reference_audio` and `reference_text`"));
    }

    #[test]
    fn voxtral_tts_history_requests_default_to_text_sized_auto_budget() {
        let mut req = base_request();
        req.model_id = Some("Voxtral-4B-TTS-2603".to_string());
        req.text = Some("The costs split cleanly into three buckets".to_string());
        req.speaker = Some("casual_male".to_string());

        let generation = build_generation_request(
            req,
            "test-correlation".to_string(),
            "The costs split cleanly into three buckets".to_string(),
            false,
            ModelVariant::Voxtral4BTts2603,
        );

        assert_eq!(
            generation.config.options.max_tokens,
            voxtral_tts_auto_max_frames_for_text(&generation.text)
        );
    }

    #[test]
    fn vibevoice_tts_history_requests_default_to_text_sized_auto_budget() {
        let mut req = base_request();
        req.model_id = Some("VibeVoice-1.5B".to_string());
        req.text = Some("The costs split cleanly into three buckets".to_string());
        req.reference_audio = Some("UklGRg==".to_string());
        req.reference_text = Some("hello".to_string());

        let generation = build_generation_request(
            req,
            "test-correlation".to_string(),
            "The costs split cleanly into three buckets".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );

        assert_eq!(
            generation.config.options.max_tokens,
            vibevoice_tts_auto_max_frames_for_text(&generation.text)
        );
    }

    #[test]
    fn qwen_history_auto_budget_is_text_sized_and_never_uses_family_maximum() {
        let text = "A short desktop speech-history request";
        let expected = qwen_tts_auto_max_frames_for_text(text);
        assert!(expected < ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);

        for requested in [None, Some(0)] {
            let mut req = base_request();
            req.max_output_tokens = requested;
            let generation = build_generation_request(
                req,
                "test-correlation".to_string(),
                text.to_string(),
                false,
                ModelVariant::Qwen3Tts12Hz17BBase,
            );
            assert_eq!(generation.config.options.max_tokens, expected);
        }
    }

    #[test]
    fn speech_history_output_budget_clamps_explicit_requests_to_model_limit() {
        assert_eq!(
            resolve_speech_history_output_frames(
                ModelVariant::Qwen3Tts12Hz06BCustomVoice,
                "Hello",
                Some(usize::MAX),
            ),
            Some(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        );
    }

    #[test]
    fn voice_cloning_requires_reference_source_after_resolution() {
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::VoiceCloning,
            ModelVariant::Qwen3Tts12Hz17BBase,
            base_request(),
        )
        .expect_err("expected missing reference source to fail");

        assert!(err
            .message
            .contains("Voice cloning requests require `saved_voice_id` or both"));
    }

    #[test]
    fn text_to_speech_base_models_require_reference_input() {
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            ModelVariant::Qwen3Tts12Hz17BBase,
            base_request(),
        )
        .expect_err("expected base model to require reference input");

        assert!(err
            .message
            .contains("requires `saved_voice_id` or both `reference_audio` and `reference_text`"));
    }

    #[test]
    fn text_to_speech_vibevoice_requires_reference_input() {
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            ModelVariant::VibeVoice15BTts,
            base_request(),
        )
        .expect_err("expected VibeVoice to require reference input");

        assert!(err
            .message
            .contains("requires `saved_voice_id` or both `reference_audio` and `reference_text`"));
    }

    #[test]
    fn generation_request_preserves_direct_reference_voice_fields() {
        let mut req = base_request();
        req.reference_audio = Some("UklGRg==".to_string());
        req.reference_text = Some("reference words".to_string());

        let generation = build_generation_request(
            req,
            "test-correlation".to_string(),
            "target words".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );

        assert_eq!(generation.reference_audio.as_deref(), Some("UklGRg=="));
        assert_eq!(
            generation.reference_text.as_deref(),
            Some("reference words")
        );
    }

    #[test]
    fn text_to_speech_voice_design_only_models_require_prompt() {
        let err = normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
            base_request(),
        )
        .expect_err("expected voice-design model to require voice_description");

        assert!(err.message.contains("requires a voice description"));
    }

    #[test]
    fn speed_is_refused_for_models_without_speed_control() {
        let request = |speed| CreateSpeechHistoryRecordRequest {
            model_id: Some("LFM2.5-Audio-1.5B-GGUF".to_string()),
            speed,
            speaker: Some("US Female".to_string()),
            ..base_request()
        };
        let error = normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            ModelVariant::Lfm25Audio15BGguf,
            request(Some(1.2)),
        )
        .expect_err("an unsupported speed is refused, not dropped");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        let normalized = normalize_for_model_capabilities(
            SpeechRouteKind::TextToSpeech,
            ModelVariant::Lfm25Audio15BGguf,
            request(Some(1.0)),
        )
        .expect("speed 1.0 is always honoured");
        assert_eq!(normalized.speed, Some(1.0));
    }
}
