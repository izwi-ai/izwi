use std::convert::Infallible;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{Extension, Json, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::api::tts_long_form::{expand_generation_requests_for_long_form, generate_long_form_tts};
use crate::error::ApiError;
use crate::speech_history_store::{
    NewSpeechHistoryRecord, SpeechHistoryRecord, SpeechHistoryRecordSummary, SpeechRouteKind,
    StoredSpeechAudio,
};
use crate::state::AppState;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::{
    parse_tts_model_variant, AudioChunk, GenerationConfig, GenerationRequest, ModelVariant,
};

const HISTORY_LIST_LIMIT: usize = 200;
const DEFAULT_STREAM_EVENT_QUEUE_CAPACITY: usize = 32;

#[derive(Debug, Deserialize, Default)]
pub(crate) struct RecordAudioQuery {
    #[serde(default)]
    download: bool,
}

#[derive(Debug, Serialize)]
pub struct SpeechHistoryRecordListResponse {
    pub records: Vec<SpeechHistoryRecordSummary>,
}

#[derive(Debug, Serialize)]
pub struct DeleteSpeechHistoryRecordResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize, Clone)]
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
}

#[derive(Debug, Serialize)]
struct SpeechStreamEvent {
    event: &'static str,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    record: Option<SpeechHistoryRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn stream_event_queue_capacity() -> usize {
    std::env::var("IZWI_AUDIO_STREAM_EVENT_QUEUE_CAPACITY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_STREAM_EVENT_QUEUE_CAPACITY)
}

async fn send_stream_event(
    event_tx: &mpsc::Sender<String>,
    event: SpeechStreamEvent,
) -> Result<(), ()> {
    event_tx.send(to_stream_json(event)).await.map_err(|_| ())
}

pub async fn list_text_to_speech_records(
    State(state): State<AppState>,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    list_records(state, SpeechRouteKind::TextToSpeech).await
}

pub async fn list_voice_design_records(
    State(state): State<AppState>,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    list_records(state, SpeechRouteKind::VoiceDesign).await
}

pub async fn list_voice_cloning_records(
    State(state): State<AppState>,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    list_records(state, SpeechRouteKind::VoiceCloning).await
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
    get_record_audio(
        state,
        SpeechRouteKind::TextToSpeech,
        record_id,
        query.download,
    )
    .await
}

pub async fn get_voice_design_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<RecordAudioQuery>,
) -> Result<Response, ApiError> {
    get_record_audio(
        state,
        SpeechRouteKind::VoiceDesign,
        record_id,
        query.download,
    )
    .await
}

pub async fn get_voice_cloning_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<RecordAudioQuery>,
) -> Result<Response, ApiError> {
    get_record_audio(
        state,
        SpeechRouteKind::VoiceCloning,
        record_id,
        query.download,
    )
    .await
}

pub async fn delete_text_to_speech_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DeleteSpeechHistoryRecordResponse>, ApiError> {
    delete_record(state, SpeechRouteKind::TextToSpeech, record_id).await
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
    Json(req): Json<CreateSpeechHistoryRecordRequest>,
) -> Result<Response, ApiError> {
    create_record(state, ctx, req, SpeechRouteKind::TextToSpeech).await
}

pub async fn create_voice_design_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<CreateSpeechHistoryRecordRequest>,
) -> Result<Response, ApiError> {
    create_record(state, ctx, req, SpeechRouteKind::VoiceDesign).await
}

pub async fn create_voice_cloning_record(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<CreateSpeechHistoryRecordRequest>,
) -> Result<Response, ApiError> {
    create_record(state, ctx, req, SpeechRouteKind::VoiceCloning).await
}

async fn list_records(
    state: AppState,
    route_kind: SpeechRouteKind,
) -> Result<Json<SpeechHistoryRecordListResponse>, ApiError> {
    let records = state
        .speech_history_store
        .list_records(route_kind, HISTORY_LIST_LIMIT)
        .await
        .map_err(map_store_error)?;
    Ok(Json(SpeechHistoryRecordListResponse { records }))
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
    as_attachment: bool,
) -> Result<Response, ApiError> {
    let audio = state
        .speech_history_store
        .get_audio(route_kind, record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("History audio not found"))?;
    Ok(audio_response(audio, as_attachment))
}

async fn delete_record(
    state: AppState,
    route_kind: SpeechRouteKind,
    record_id: String,
) -> Result<Json<DeleteSpeechHistoryRecordResponse>, ApiError> {
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

async fn create_record(
    state: AppState,
    ctx: RequestContext,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
) -> Result<Response, ApiError> {
    let model_id = required_trimmed(req.model_id.as_deref(), "model_id")?;
    let input_text = required_trimmed(req.text.as_deref(), "text")?;

    validate_route_requirements(route_kind, &req)?;

    let variant = parse_tts_model_variant(model_id.as_str())
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {err}")))?;
    state.runtime.load_model(variant).await?;

    if req.stream.unwrap_or(false) {
        if route_kind != SpeechRouteKind::TextToSpeech {
            return Err(ApiError::bad_request(
                "Streaming is currently supported only on /text-to-speech",
            ));
        }
        return stream_record_creation(state, ctx, req, route_kind, variant, model_id, input_text)
            .await;
    }

    let generation_request =
        build_generation_request(req.clone(), ctx.correlation_id, input_text.clone(), false);
    let planned_request_count =
        expand_generation_requests_for_long_form(&generation_request, variant).len();

    let _permit = state.acquire_permit().await;
    let timeout = Duration::from_secs(resolve_generation_timeout_secs(
        state.request_timeout_secs,
        variant,
        req.max_output_tokens.or(req.max_tokens),
        req.speed,
        planned_request_count,
    ));

    let runtime = state.runtime.clone();
    let output = tokio::time::timeout(timeout, async {
        generate_long_form_tts(&runtime, variant, generation_request).await
    })
    .await
    .map_err(|_| ApiError::internal("Speech generation timed out"))??;

    let encoder = state.runtime.audio_encoder().await;
    let output_samples = output.samples.clone();
    let encoded_audio =
        tokio::task::spawn_blocking(move || encoder.encode(&output_samples, AudioFormat::Wav))
            .await
            .map_err(|err| ApiError::internal(format!("Audio encoding failed: {err}")))??;

    let record = state
        .speech_history_store
        .create_record(NewSpeechHistoryRecord {
            route_kind,
            model_id: Some(model_id),
            speaker: req.speaker,
            language: req.language,
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
        })
        .await
        .map_err(map_store_error)?;

    Ok(Json(record).into_response())
}

async fn stream_record_creation(
    state: AppState,
    ctx: RequestContext,
    req: CreateSpeechHistoryRecordRequest,
    route_kind: SpeechRouteKind,
    variant: ModelVariant,
    model_id: String,
    input_text: String,
) -> Result<Response, ApiError> {
    let generation_request =
        build_generation_request(req.clone(), ctx.correlation_id, input_text.clone(), true);
    let planned_requests = expand_generation_requests_for_long_form(&generation_request, variant);
    let stream_request_id = generation_request.id.clone();

    let runtime = state.runtime.clone();
    let speech_store = state.speech_history_store.clone();
    let semaphore = state.request_semaphore.clone();

    let (event_tx, mut event_rx) = mpsc::channel::<String>(stream_event_queue_capacity());

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = send_stream_event(
                    &event_tx,
                    SpeechStreamEvent {
                        event: "error",
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
                        record: None,
                        error: Some("Server is shutting down".to_string()),
                    },
                )
                .await;
                let _ = send_stream_event(
                    &event_tx,
                    SpeechStreamEvent {
                        event: "done",
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
                        record: None,
                        error: None,
                    },
                )
                .await;
                return;
            }
        };

        let sample_rate = runtime.sample_rate().await;
        if send_stream_event(
            &event_tx,
            SpeechStreamEvent {
                event: "start",
                request_id: Some(stream_request_id.clone()),
                sequence: None,
                audio_base64: None,
                sample_count: None,
                sample_rate: Some(sample_rate),
                audio_format: Some("pcm_i16"),
                tokens_generated: None,
                generation_time_ms: None,
                audio_duration_secs: None,
                rtf: None,
                record: None,
                error: None,
            },
        )
        .await
        .is_err()
        {
            return;
        }

        let mut total_samples = 0usize;
        let mut total_tokens = 0usize;
        let stream_started = std::time::Instant::now();
        let stream_encoder = AudioEncoder::new(sample_rate, 1);
        let mut merged_samples: Vec<f32> = Vec::new();
        let mut global_sequence = 0usize;
        let mut failed = false;
        for request in planned_requests {
            let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(32);
            let generation_engine = runtime.clone();
            let generation_task = tokio::spawn(async move {
                generation_engine
                    .generate_streaming(request, chunk_tx)
                    .await
            });

            let mut encoding_failed = false;
            let mut stream_closed = false;
            while let Some(chunk) = chunk_rx.recv().await {
                if chunk.samples.is_empty() {
                    continue;
                }

                total_samples += chunk.samples.len();
                if let Some(stats) = chunk.stats.as_ref() {
                    total_tokens = total_tokens.saturating_add(stats.tokens_generated);
                }
                merged_samples.extend_from_slice(&chunk.samples);

                let chunk_bytes = match stream_encoder.encode(&chunk.samples, AudioFormat::RawI16) {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        let _ = send_stream_event(
                            &event_tx,
                            SpeechStreamEvent {
                                event: "error",
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
                                record: None,
                                error: Some(format!("Failed to encode stream chunk: {err}")),
                            },
                        )
                        .await;
                        encoding_failed = true;
                        break;
                    }
                };

                if send_stream_event(
                    &event_tx,
                    SpeechStreamEvent {
                        event: "chunk",
                        request_id: Some(stream_request_id.clone()),
                        sequence: Some(global_sequence),
                        audio_base64: Some(
                            base64::engine::general_purpose::STANDARD.encode(chunk_bytes),
                        ),
                        sample_count: Some(chunk.samples.len()),
                        sample_rate: None,
                        audio_format: None,
                        tokens_generated: None,
                        generation_time_ms: None,
                        audio_duration_secs: None,
                        rtf: None,
                        record: None,
                        error: None,
                    },
                )
                .await
                .is_err()
                {
                    stream_closed = true;
                    break;
                }
                global_sequence = global_sequence.saturating_add(1);
            }

            drop(chunk_rx);
            let generation_outcome = generation_task.await;
            if encoding_failed || stream_closed {
                let _ = generation_outcome;
                failed = true;
                break;
            }

            match generation_outcome {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    let _ = send_stream_event(
                        &event_tx,
                        SpeechStreamEvent {
                            event: "error",
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
                            record: None,
                            error: Some(err.to_string()),
                        },
                    )
                    .await;
                    failed = true;
                    break;
                }
                Err(err) => {
                    let _ = send_stream_event(
                        &event_tx,
                        SpeechStreamEvent {
                            event: "error",
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
                            record: None,
                            error: Some(format!("Streaming task failed: {err}")),
                        },
                    )
                    .await;
                    failed = true;
                    break;
                }
            }
        }

        if !failed {
            let generation_time_ms = stream_started.elapsed().as_secs_f32() * 1000.0;
            let audio_duration_secs = total_samples as f32 / sample_rate as f32;
            if total_tokens == 0 {
                total_tokens = total_samples / 256;
            }
            let rtf = if audio_duration_secs > 0.0 {
                (generation_time_ms / 1000.0) / audio_duration_secs
            } else {
                0.0
            };

            let wav_encoder = AudioEncoder::new(sample_rate, 1);
            match wav_encoder.encode(merged_samples.as_slice(), AudioFormat::Wav) {
                Ok(wav_bytes) => {
                    let record_result = speech_store
                        .create_record(NewSpeechHistoryRecord {
                            route_kind,
                            model_id: Some(model_id),
                            speaker: req.speaker,
                            language: req.language,
                            input_text,
                            voice_description: req.voice_description,
                            reference_text: req.reference_text,
                            generation_time_ms: generation_time_ms as f64,
                            audio_duration_secs: Some(audio_duration_secs as f64),
                            rtf: Some(rtf as f64),
                            tokens_generated: Some(total_tokens),
                            audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav)
                                .to_string(),
                            audio_filename: Some(default_audio_filename(route_kind, "wav")),
                            audio_bytes: wav_bytes,
                        })
                        .await;

                    match record_result {
                        Ok(record) => {
                            let _ = send_stream_event(
                                &event_tx,
                                SpeechStreamEvent {
                                    event: "final",
                                    request_id: Some(stream_request_id.clone()),
                                    sequence: None,
                                    audio_base64: None,
                                    sample_count: None,
                                    sample_rate: None,
                                    audio_format: None,
                                    tokens_generated: Some(total_tokens),
                                    generation_time_ms: Some(generation_time_ms),
                                    audio_duration_secs: Some(audio_duration_secs),
                                    rtf: Some(rtf),
                                    record: Some(record),
                                    error: None,
                                },
                            )
                            .await;
                        }
                        Err(err) => {
                            let _ = send_stream_event(
                                &event_tx,
                                SpeechStreamEvent {
                                    event: "error",
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
                                    record: None,
                                    error: Some(format!(
                                        "Failed to save speech history record: {err}"
                                    )),
                                },
                            )
                            .await;
                        }
                    }
                }
                Err(err) => {
                    let _ = send_stream_event(
                        &event_tx,
                        SpeechStreamEvent {
                            event: "error",
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
                            record: None,
                            error: Some(format!("Failed to encode final WAV output: {err}")),
                        },
                    )
                    .await;
                }
            }
        }

        let _ = send_stream_event(
            &event_tx,
            SpeechStreamEvent {
                event: "done",
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
                record: None,
                error: None,
            },
        )
        .await;
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| Response::new(Body::empty())))
}

fn build_generation_request(
    req: CreateSpeechHistoryRecordRequest,
    correlation_id: String,
    text: String,
    streaming: bool,
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
    if let Some(max_tokens) = req.max_output_tokens.or(req.max_tokens) {
        generation_config.options.max_tokens = max_tokens;
    }
    if let Some(top_k) = req.top_k {
        generation_config.options.top_k = top_k;
    }
    generation_config.options.speaker = req.speaker.clone();

    GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        correlation_id: Some(correlation_id),
        text,
        config: generation_config,
        language: req.language,
        reference_audio: req.reference_audio,
        reference_text: req.reference_text,
        voice_description: req.voice_description,
    }
}

fn validate_route_requirements(
    route_kind: SpeechRouteKind,
    req: &CreateSpeechHistoryRecordRequest,
) -> Result<(), ApiError> {
    match route_kind {
        SpeechRouteKind::VoiceDesign => {
            let has_description = req
                .voice_description
                .as_deref()
                .map(str::trim)
                .map(|text| !text.is_empty())
                .unwrap_or(false);
            if !has_description {
                return Err(ApiError::bad_request(
                    "Voice design requests require `voice_description`.",
                ));
            }
        }
        SpeechRouteKind::VoiceCloning => {
            let has_reference_audio = req
                .reference_audio
                .as_deref()
                .map(str::trim)
                .map(|text| !text.is_empty())
                .unwrap_or(false);
            let has_reference_text = req
                .reference_text
                .as_deref()
                .map(str::trim)
                .map(|text| !text.is_empty())
                .unwrap_or(false);
            if !has_reference_audio || !has_reference_text {
                return Err(ApiError::bad_request(
                    "Voice cloning requests require `reference_audio` and `reference_text`.",
                ));
            }
        }
        SpeechRouteKind::TextToSpeech => {}
    }

    Ok(())
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

fn resolve_generation_timeout_secs(
    default_timeout_secs: u64,
    variant: ModelVariant,
    requested_frames: Option<usize>,
    requested_speed: Option<f32>,
    planned_request_count: usize,
) -> u64 {
    let Some(model_max_frames) = variant.tts_max_output_frames_hint() else {
        return default_timeout_secs.max(1);
    };
    let Some(frame_rate_hz) = variant.tts_output_frame_rate_hz_hint() else {
        return default_timeout_secs.max(1);
    };

    let effective_frames = match requested_frames {
        Some(0) | None => model_max_frames,
        Some(value) => value.clamp(1, model_max_frames),
    };

    let speed = requested_speed.unwrap_or(1.0).clamp(0.25, 4.0) as f64;
    let estimated_audio_secs = ((effective_frames as f64) / (frame_rate_hz as f64)) / speed;

    let timeout_rtf = std::env::var("IZWI_TTS_TIMEOUT_RTF")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 1.0)
        .unwrap_or(8.0);
    let timeout_padding_secs = std::env::var("IZWI_TTS_TIMEOUT_PADDING_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30);
    let timeout_max_secs = std::env::var("IZWI_TTS_TIMEOUT_MAX_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(6 * 60 * 60);

    let adaptive_secs = (estimated_audio_secs * timeout_rtf).ceil() as u64;
    let suggested_secs = adaptive_secs
        .saturating_add(timeout_padding_secs)
        .min(timeout_max_secs.max(1));

    let per_request_timeout_secs = default_timeout_secs.max(suggested_secs).max(1);
    let request_count = planned_request_count.max(1) as u64;
    if request_count == 1 {
        return per_request_timeout_secs;
    }

    let long_form_timeout_max_secs = std::env::var("IZWI_TTS_LONG_FORM_TIMEOUT_MAX_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(24 * 60 * 60);

    per_request_timeout_secs
        .saturating_mul(request_count)
        .min(long_form_timeout_max_secs.max(per_request_timeout_secs))
}

fn default_audio_filename(route_kind: SpeechRouteKind, extension: &str) -> String {
    let prefix = match route_kind {
        SpeechRouteKind::TextToSpeech => "tts",
        SpeechRouteKind::VoiceDesign => "voice-design",
        SpeechRouteKind::VoiceCloning => "voice-cloning",
    };
    format!("{prefix}-{}.{}", uuid::Uuid::new_v4().simple(), extension)
}

fn to_stream_json(event: SpeechStreamEvent) -> String {
    serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string())
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
    ApiError::internal(format!("Speech history storage error: {err}"))
}
