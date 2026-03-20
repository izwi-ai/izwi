use axum::{
    body::Body,
    extract::{Multipart, Path, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json, RequestExt,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

use crate::diarization_store::{
    DiarizationRecord, DiarizationRecordSummary, DiarizationSegmentRecord,
    DiarizationUtteranceRecord, DiarizationWordRecord, NewDiarizationRecord,
    StoredDiarizationAudio,
};
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::DiarizationConfig;

const HISTORY_LIST_LIMIT: usize = 200;

#[derive(Debug, Serialize)]
pub struct DiarizationRecordListResponse {
    pub records: Vec<DiarizationRecordSummary>,
}

#[derive(Debug, Serialize)]
pub struct DeleteDiarizationRecordResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateDiarizationRecordRequest {
    #[serde(default)]
    pub speaker_name_overrides: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct RerunDiarizationRecordRequest {
    #[serde(default)]
    pub min_speakers: Option<usize>,
    #[serde(default)]
    pub max_speakers: Option<usize>,
    #[serde(default)]
    pub min_speech_duration_ms: Option<f64>,
    #[serde(default)]
    pub min_silence_duration_ms: Option<f64>,
    #[serde(default)]
    pub enable_llm_refinement: Option<bool>,
}

#[derive(Debug, Default)]
struct ParsedDiarizationCreateRequest {
    audio_bytes: Vec<u8>,
    audio_mime_type: Option<String>,
    audio_filename: Option<String>,
    model_id: Option<String>,
    asr_model_id: Option<String>,
    aligner_model_id: Option<String>,
    llm_model_id: Option<String>,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    min_speech_duration_ms: Option<f64>,
    min_silence_duration_ms: Option<f64>,
    enable_llm_refinement: Option<bool>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct JsonCreateRequest {
    audio_base64: String,
    #[serde(default, alias = "model", alias = "diarization_model")]
    model_id: Option<String>,
    #[serde(default, alias = "asr_model")]
    asr_model_id: Option<String>,
    #[serde(default, alias = "aligner_model")]
    aligner_model_id: Option<String>,
    #[serde(default, alias = "llm_model")]
    llm_model_id: Option<String>,
    #[serde(default)]
    min_speakers: Option<usize>,
    #[serde(default)]
    max_speakers: Option<usize>,
    #[serde(default)]
    min_speech_duration_ms: Option<f64>,
    #[serde(default)]
    min_silence_duration_ms: Option<f64>,
    #[serde(default)]
    enable_llm_refinement: Option<bool>,
    #[serde(default)]
    audio_mime_type: Option<String>,
    #[serde(default)]
    audio_filename: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
}

pub async fn list_records(
    State(state): State<AppState>,
) -> Result<Json<DiarizationRecordListResponse>, ApiError> {
    let records = state
        .diarization_store
        .list_records(HISTORY_LIST_LIMIT)
        .await
        .map_err(map_store_error)?;

    Ok(Json(DiarizationRecordListResponse { records }))
}

pub async fn get_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DiarizationRecord>, ApiError> {
    let record = state
        .diarization_store
        .get_record(record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Diarization record not found"))?;

    Ok(Json(record))
}

pub async fn get_record_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Response, ApiError> {
    let audio = state
        .diarization_store
        .get_audio(record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Diarization audio not found"))?;

    Ok(audio_response(audio))
}

pub async fn delete_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
) -> Result<Json<DeleteDiarizationRecordResponse>, ApiError> {
    let deleted = state
        .diarization_store
        .delete_record(record_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Diarization record not found"));
    }

    Ok(Json(DeleteDiarizationRecordResponse {
        id: record_id,
        deleted: true,
    }))
}

pub async fn update_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Json(req): Json<UpdateDiarizationRecordRequest>,
) -> Result<Json<DiarizationRecord>, ApiError> {
    let updated = state
        .diarization_store
        .update_speaker_name_overrides(record_id, req.speaker_name_overrides)
        .await
        .map_err(map_store_error)?;

    let record = updated.ok_or_else(|| ApiError::not_found("Diarization record not found"))?;
    Ok(Json(record))
}

pub async fn rerun_record(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Json(req): Json<RerunDiarizationRecordRequest>,
) -> Result<Json<DiarizationRecord>, ApiError> {
    let source_record = state
        .diarization_store
        .get_record(record_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Diarization record not found"))?;
    let source_audio = state
        .diarization_store
        .get_audio(record_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Diarization audio not found"))?;

    let rerun_request = build_rerun_create_request(&source_record, source_audio, req);
    let record = execute_diarization_record(&state, rerun_request).await?;

    Ok(Json(record))
}

pub async fn create_record(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response, ApiError> {
    let parsed = parse_create_request(req).await?;

    if parsed.stream {
        return Err(ApiError::bad_request(
            "Streaming diarization is not supported on persisted diarization resources",
        ));
    }

    let record = execute_diarization_record(&state, parsed).await?;

    Ok(Json(record).into_response())
}

async fn execute_diarization_record(
    state: &AppState,
    parsed: ParsedDiarizationCreateRequest,
) -> Result<DiarizationRecord, ApiError> {
    validate_speaker_bounds(parsed.min_speakers, parsed.max_speakers)?;

    let _permit = state.acquire_permit().await;
    let started = Instant::now();

    let output = state
        .runtime
        .diarize_with_transcript_bytes(
            parsed.audio_bytes.as_slice(),
            parsed.model_id.as_deref(),
            parsed.asr_model_id.as_deref(),
            parsed.aligner_model_id.as_deref(),
            parsed.llm_model_id.as_deref(),
            &DiarizationConfig {
                min_speakers: parsed.min_speakers,
                max_speakers: parsed.max_speakers,
                min_speech_duration_ms: parsed.min_speech_duration_ms.map(|v| v as f32),
                min_silence_duration_ms: parsed.min_silence_duration_ms.map(|v| v as f32),
            },
            parsed.enable_llm_refinement.unwrap_or(false),
        )
        .await?;

    let processing_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    let rtf = if output.duration_secs > 0.0 {
        Some((processing_time_ms / 1000.0) / output.duration_secs as f64)
    } else {
        None
    };

    state
        .diarization_store
        .create_record(NewDiarizationRecord {
            model_id: parsed.model_id,
            asr_model_id: parsed.asr_model_id,
            aligner_model_id: parsed.aligner_model_id,
            llm_model_id: parsed.llm_model_id,
            min_speakers: parsed.min_speakers,
            max_speakers: parsed.max_speakers,
            min_speech_duration_ms: parsed.min_speech_duration_ms,
            min_silence_duration_ms: parsed.min_silence_duration_ms,
            enable_llm_refinement: parsed.enable_llm_refinement.unwrap_or(false),
            processing_time_ms,
            duration_secs: Some(output.duration_secs as f64),
            rtf,
            speaker_count: output.speaker_count,
            alignment_coverage: Some(output.alignment_coverage as f64),
            unattributed_words: output.unattributed_words,
            llm_refined: output.llm_refined,
            asr_text: output.asr_text,
            raw_transcript: output.raw_transcript,
            transcript: output.transcript,
            segments: output
                .segments
                .into_iter()
                .map(|segment| DiarizationSegmentRecord {
                    speaker: segment.speaker,
                    start: segment.start_secs,
                    end: segment.end_secs,
                    confidence: segment.confidence,
                })
                .collect(),
            words: output
                .words
                .into_iter()
                .map(|word| DiarizationWordRecord {
                    word: word.word,
                    speaker: word.speaker,
                    start: word.start_secs,
                    end: word.end_secs,
                    speaker_confidence: word.speaker_confidence,
                    overlaps_segment: word.overlaps_segment,
                })
                .collect(),
            utterances: output
                .utterances
                .into_iter()
                .map(|utterance| DiarizationUtteranceRecord {
                    speaker: utterance.speaker,
                    start: utterance.start_secs,
                    end: utterance.end_secs,
                    text: utterance.text,
                    word_start: utterance.word_start,
                    word_end: utterance.word_end,
                })
                .collect(),
            audio_mime_type: parsed
                .audio_mime_type
                .unwrap_or_else(|| "audio/wav".to_string()),
            audio_filename: parsed.audio_filename,
            speaker_name_overrides: BTreeMap::new(),
            audio_bytes: parsed.audio_bytes,
        })
        .await
        .map_err(map_store_error)
}

fn build_rerun_create_request(
    source_record: &DiarizationRecord,
    source_audio: StoredDiarizationAudio,
    rerun: RerunDiarizationRecordRequest,
) -> ParsedDiarizationCreateRequest {
    ParsedDiarizationCreateRequest {
        audio_bytes: source_audio.audio_bytes,
        audio_mime_type: Some(source_audio.audio_mime_type),
        audio_filename: source_audio.audio_filename,
        model_id: source_record.model_id.clone(),
        asr_model_id: source_record.asr_model_id.clone(),
        aligner_model_id: source_record.aligner_model_id.clone(),
        llm_model_id: source_record.llm_model_id.clone(),
        min_speakers: rerun.min_speakers.or(source_record.min_speakers),
        max_speakers: rerun.max_speakers.or(source_record.max_speakers),
        min_speech_duration_ms: rerun
            .min_speech_duration_ms
            .or(source_record.min_speech_duration_ms),
        min_silence_duration_ms: rerun
            .min_silence_duration_ms
            .or(source_record.min_silence_duration_ms),
        enable_llm_refinement: Some(
            rerun
                .enable_llm_refinement
                .unwrap_or(source_record.enable_llm_refinement),
        ),
        stream: false,
    }
}

fn validate_speaker_bounds(
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> Result<(), ApiError> {
    if let (Some(min), Some(max)) = (min_speakers, max_speakers) {
        if min > max {
            return Err(ApiError::bad_request(
                "`min_speakers` cannot be greater than `max_speakers`.",
            ));
        }
    }

    Ok(())
}

async fn parse_create_request(req: Request) -> Result<ParsedDiarizationCreateRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<JsonCreateRequest>, _>()
            .await
            .map_err(|err| ApiError::bad_request(format!("Invalid JSON payload: {err}")))?;

        let audio_bytes = decode_audio_base64(payload.audio_base64.as_str())?;
        if audio_bytes.is_empty() {
            return Err(ApiError::bad_request("Audio payload cannot be empty."));
        }

        return Ok(ParsedDiarizationCreateRequest {
            audio_bytes,
            audio_mime_type: sanitize_optional(payload.audio_mime_type),
            audio_filename: sanitize_optional(payload.audio_filename),
            model_id: sanitize_optional(payload.model_id),
            asr_model_id: sanitize_optional(payload.asr_model_id),
            aligner_model_id: sanitize_optional(payload.aligner_model_id),
            llm_model_id: sanitize_optional(payload.llm_model_id),
            min_speakers: payload.min_speakers,
            max_speakers: payload.max_speakers,
            min_speech_duration_ms: payload.min_speech_duration_ms,
            min_silence_duration_ms: payload.min_silence_duration_ms,
            enable_llm_refinement: payload.enable_llm_refinement,
            stream: payload.stream.unwrap_or(false),
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|err| ApiError::bad_request(format!("Invalid multipart payload: {err}")))?;

        let mut out = ParsedDiarizationCreateRequest::default();

        while let Some(field) = multipart.next_field().await.map_err(|err| {
            ApiError::bad_request(format!("Failed reading multipart field: {err}"))
        })? {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let mime_type = field.content_type().map(ToString::to_string);
                    let file_name = field.file_name().map(ToString::to_string);
                    let bytes = field.bytes().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {}",
                            name, err
                        ))
                    })?;
                    if !bytes.is_empty() {
                        out.audio_bytes = bytes.to_vec();
                        out.audio_mime_type = sanitize_optional(mime_type);
                        out.audio_filename = sanitize_optional(file_name);
                    }
                }
                "audio_base64" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_base64' field: {err}"
                        ))
                    })?;
                    let decoded = decode_audio_base64(text.as_str())?;
                    if !decoded.is_empty() {
                        out.audio_bytes = decoded;
                    }
                }
                "audio_mime_type" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_mime_type' field: {err}"
                        ))
                    })?;
                    out.audio_mime_type = sanitize_optional(Some(text));
                }
                "audio_filename" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_filename' field: {err}"
                        ))
                    })?;
                    out.audio_filename = sanitize_optional(Some(text));
                }
                "model" | "diarization_model" | "model_id" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {err}",
                            name
                        ))
                    })?;
                    out.model_id = sanitize_optional(Some(text));
                }
                "asr_model" | "asr_model_id" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {err}",
                            name
                        ))
                    })?;
                    out.asr_model_id = sanitize_optional(Some(text));
                }
                "aligner_model" | "aligner_model_id" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {err}",
                            name
                        ))
                    })?;
                    out.aligner_model_id = sanitize_optional(Some(text));
                }
                "llm_model" | "llm_model_id" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {err}",
                            name
                        ))
                    })?;
                    out.llm_model_id = sanitize_optional(Some(text));
                }
                "min_speakers" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_speakers' field: {err}"
                        ))
                    })?;
                    out.min_speakers = text.trim().parse::<usize>().ok();
                }
                "max_speakers" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'max_speakers' field: {err}"
                        ))
                    })?;
                    out.max_speakers = text.trim().parse::<usize>().ok();
                }
                "min_speech_duration_ms" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_speech_duration_ms' field: {err}"
                        ))
                    })?;
                    out.min_speech_duration_ms = text.trim().parse::<f64>().ok();
                }
                "min_silence_duration_ms" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_silence_duration_ms' field: {err}"
                        ))
                    })?;
                    out.min_silence_duration_ms = text.trim().parse::<f64>().ok();
                }
                "enable_llm_refinement" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'enable_llm_refinement' field: {err}"
                        ))
                    })?;
                    out.enable_llm_refinement = Some(parse_truthy(text.as_str()));
                }
                "stream" => {
                    let text = field.text().await.map_err(|err| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'stream' field: {err}"
                        ))
                    })?;
                    out.stream = parse_truthy(text.as_str());
                }
                _ => {}
            }
        }

        if out.audio_bytes.is_empty() {
            return Err(ApiError::bad_request(
                "Missing audio input (`file` or `audio_base64`).",
            ));
        }

        return Ok(out);
    }

    Err(ApiError {
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        message: "Expected `Content-Type: application/json` or `multipart/form-data`".to_string(),
    })
}

fn decode_audio_base64(raw: &str) -> Result<Vec<u8>, ApiError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    base64::engine::general_purpose::STANDARD
        .decode(trimmed)
        .map_err(|_| ApiError::bad_request("Invalid base64 audio payload."))
}

fn parse_truthy(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sanitize_optional<T>(raw: Option<T>) -> Option<String>
where
    T: Into<String>,
{
    let value: String = raw?.into();
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn audio_response(audio: StoredDiarizationAudio) -> Response {
    let mut response = Response::builder().status(StatusCode::OK);

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.header(header::CONTENT_TYPE, content_type);
    }

    if let Some(filename) = audio.audio_filename {
        let disposition = format!("inline; filename=\"{}\"", filename.replace('"', ""));
        if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
            response = response.header(header::CONTENT_DISPOSITION, value);
        }
    }

    response
        .body(Body::from(audio.audio_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> DiarizationRecord {
        DiarizationRecord {
            id: "dir_test".to_string(),
            created_at: 1,
            model_id: Some("diar_streaming_sortformer_4spk-v2.1".to_string()),
            asr_model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            llm_model_id: Some("Qwen3-1.7B-GGUF".to_string()),
            min_speakers: Some(1),
            max_speakers: Some(4),
            min_speech_duration_ms: Some(240.0),
            min_silence_duration_ms: Some(200.0),
            enable_llm_refinement: true,
            processing_time_ms: 120.0,
            duration_secs: Some(9.5),
            rtf: Some(0.4),
            speaker_count: 2,
            corrected_speaker_count: 2,
            alignment_coverage: Some(0.94),
            unattributed_words: 0,
            llm_refined: true,
            asr_text: "hello there".to_string(),
            raw_transcript: "SPEAKER_00: hello there".to_string(),
            transcript: "SPEAKER_00: hello there".to_string(),
            segments: Vec::new(),
            words: Vec::new(),
            utterances: Vec::new(),
            speaker_name_overrides: BTreeMap::new(),
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("meeting.wav".to_string()),
        }
    }

    fn sample_audio() -> StoredDiarizationAudio {
        StoredDiarizationAudio {
            audio_bytes: vec![1_u8, 2_u8, 3_u8],
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("meeting.wav".to_string()),
        }
    }

    #[test]
    fn rerun_request_inherits_existing_record_settings() {
        let parsed = build_rerun_create_request(
            &sample_record(),
            sample_audio(),
            RerunDiarizationRecordRequest::default(),
        );

        assert_eq!(
            parsed.model_id.as_deref(),
            Some("diar_streaming_sortformer_4spk-v2.1")
        );
        assert_eq!(parsed.asr_model_id.as_deref(), Some("Parakeet-TDT-0.6B-v3"));
        assert_eq!(
            parsed.aligner_model_id.as_deref(),
            Some("Qwen3-ForcedAligner-0.6B")
        );
        assert_eq!(parsed.llm_model_id.as_deref(), Some("Qwen3-1.7B-GGUF"));
        assert_eq!(parsed.min_speakers, Some(1));
        assert_eq!(parsed.max_speakers, Some(4));
        assert_eq!(parsed.min_speech_duration_ms, Some(240.0));
        assert_eq!(parsed.min_silence_duration_ms, Some(200.0));
        assert_eq!(parsed.enable_llm_refinement, Some(true));
        assert_eq!(parsed.audio_bytes, vec![1_u8, 2_u8, 3_u8]);
    }

    #[test]
    fn rerun_request_applies_requested_overrides() {
        let parsed = build_rerun_create_request(
            &sample_record(),
            sample_audio(),
            RerunDiarizationRecordRequest {
                min_speakers: Some(2),
                max_speakers: Some(5),
                min_speech_duration_ms: Some(180.0),
                min_silence_duration_ms: Some(140.0),
                enable_llm_refinement: Some(false),
            },
        );

        assert_eq!(parsed.min_speakers, Some(2));
        assert_eq!(parsed.max_speakers, Some(5));
        assert_eq!(parsed.min_speech_duration_ms, Some(180.0));
        assert_eq!(parsed.min_silence_duration_ms, Some(140.0));
        assert_eq!(parsed.enable_llm_refinement, Some(false));
    }

    #[test]
    fn rejects_invalid_speaker_bounds() {
        let err = validate_speaker_bounds(Some(4), Some(2))
            .expect_err("min speakers should not exceed max speakers");
        assert_eq!(
            err.message,
            "`min_speakers` cannot be greater than `max_speakers`."
        );
    }
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Diarization storage error: {err}"))
}
