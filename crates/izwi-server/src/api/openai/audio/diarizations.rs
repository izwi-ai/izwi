//! OpenAI-compatible diarization endpoint.

use axum::{
    body::Body,
    extract::{Multipart, Request, State},
    http::{header, StatusCode},
    response::Response,
    Json, RequestExt,
};
use base64::Engine;
use std::time::Instant;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{
    DiarizationConfig, DiarizationResult, DiarizationSegment, DiarizationTranscriptResult,
    DiarizationUtterance, DiarizationWord,
};

#[derive(Debug, Default)]
struct DiarizationRequest {
    audio_base64: Option<String>,
    diarization_model: Option<String>,
    asr_model: Option<String>,
    aligner_model: Option<String>,
    llm_model: Option<String>,
    response_format: Option<String>,
    num_speakers: Option<usize>,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    min_speech_duration_ms: Option<f32>,
    min_silence_duration_ms: Option<f32>,
    enable_llm_refinement: Option<bool>,
    stream: bool,
}

#[derive(Debug, serde::Serialize)]
struct JsonSegment {
    speaker: String,
    start: f32,
    end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
}

#[derive(Debug, serde::Serialize)]
struct JsonWord {
    word: String,
    speaker: String,
    start: f32,
    end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_confidence: Option<f32>,
    overlaps_segment: bool,
}

#[derive(Debug, serde::Serialize)]
struct JsonUtterance {
    speaker: String,
    start: f32,
    end: f32,
    text: String,
    word_start: usize,
    word_end: usize,
}

#[derive(Debug, serde::Serialize)]
struct JsonDiarizationResponse {
    segments: Vec<JsonSegment>,
    transcript: String,
}

#[derive(Debug, serde::Serialize)]
struct VerboseJsonDiarizationResponse {
    segments: Vec<JsonSegment>,
    words: Vec<JsonWord>,
    utterances: Vec<JsonUtterance>,
    asr_text: String,
    raw_transcript: String,
    transcript: String,
    llm_refined: bool,
    alignment_coverage: f32,
    unattributed_words: usize,
    speaker_count: usize,
    duration: f32,
    processing_time_ms: f64,
    rtf: Option<f64>,
}

pub async fn diarizations(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_diarization_request(req).await?;
    if req.stream {
        return Err(ApiError::bad_request(
            "Streaming diarization is not supported for /v1/audio/diarizations",
        ));
    }

    let audio_base64 = req
        .audio_base64
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    let _permit = state.acquire_permit().await;
    let (min_speakers, max_speakers) =
        resolve_speaker_bounds(req.min_speakers, req.max_speakers, req.num_speakers)?;

    let config = DiarizationConfig {
        min_speakers,
        max_speakers,
        min_speech_duration_ms: req.min_speech_duration_ms,
        min_silence_duration_ms: req.min_silence_duration_ms,
    };

    let started = Instant::now();
    let output = state
        .runtime
        .diarize_with_transcript(
            &audio_base64,
            req.diarization_model.as_deref(),
            req.asr_model.as_deref(),
            req.aligner_model.as_deref(),
            req.llm_model.as_deref(),
            &config,
            req.enable_llm_refinement.unwrap_or(false),
        )
        .await?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let rtf = if output.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / output.duration_secs as f64)
    } else {
        None
    };

    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();

    let segments = map_segments(&output.segments);

    match response_format.as_str() {
        "json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&JsonDiarizationResponse {
                    segments,
                    transcript: output.transcript,
                })
                .unwrap(),
            ))
            .unwrap()),
        "verbose_json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&VerboseJsonDiarizationResponse {
                    segments,
                    words: map_words(&output.words),
                    utterances: map_utterances(&output.utterances),
                    asr_text: output.asr_text,
                    raw_transcript: output.raw_transcript,
                    transcript: output.transcript,
                    llm_refined: output.llm_refined,
                    alignment_coverage: output.alignment_coverage,
                    unattributed_words: output.unattributed_words,
                    speaker_count: output.speaker_count,
                    duration: output.duration_secs,
                    processing_time_ms: elapsed_ms,
                    rtf,
                })
                .unwrap(),
            ))
            .unwrap()),
        "text" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(format_diarization_text(&output)))
            .unwrap()),
        other => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported: json, verbose_json, text",
            other
        ))),
    }
}

fn map_segments(segments: &[DiarizationSegment]) -> Vec<JsonSegment> {
    segments
        .iter()
        .map(|segment| JsonSegment {
            speaker: segment.speaker.clone(),
            start: segment.start_secs,
            end: segment.end_secs,
            confidence: segment.confidence,
        })
        .collect()
}

fn map_words(words: &[DiarizationWord]) -> Vec<JsonWord> {
    words
        .iter()
        .map(|word| JsonWord {
            word: word.word.clone(),
            speaker: word.speaker.clone(),
            start: word.start_secs,
            end: word.end_secs,
            speaker_confidence: word.speaker_confidence,
            overlaps_segment: word.overlaps_segment,
        })
        .collect()
}

fn map_utterances(utterances: &[DiarizationUtterance]) -> Vec<JsonUtterance> {
    utterances
        .iter()
        .map(|utterance| JsonUtterance {
            speaker: utterance.speaker.clone(),
            start: utterance.start_secs,
            end: utterance.end_secs,
            text: utterance.text.clone(),
            word_start: utterance.word_start,
            word_end: utterance.word_end,
        })
        .collect()
}

fn format_diarization_text(output: &DiarizationTranscriptResult) -> String {
    if !output.transcript.trim().is_empty() {
        return output.transcript.clone();
    }
    format_segments_text(&DiarizationResult {
        segments: output.segments.clone(),
        duration_secs: output.duration_secs,
        speaker_count: output.speaker_count,
    })
}

fn format_segments_text(output: &DiarizationResult) -> String {
    let mut out = String::new();
    for segment in &output.segments {
        let duration = (segment.end_secs - segment.start_secs).max(0.0);
        out.push_str(&format!(
            "SPEAKER unknown 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
            segment.start_secs, duration, segment.speaker
        ));
    }
    out
}

#[derive(Debug, serde::Deserialize)]
struct JsonRequestBody {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    diarization_model: Option<String>,
    #[serde(default)]
    asr_model: Option<String>,
    #[serde(default)]
    aligner_model: Option<String>,
    #[serde(default)]
    llm_model: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    num_speakers: Option<usize>,
    #[serde(default)]
    min_speakers: Option<usize>,
    #[serde(default)]
    max_speakers: Option<usize>,
    #[serde(default)]
    min_speech_duration_ms: Option<f32>,
    #[serde(default)]
    min_silence_duration_ms: Option<f32>,
    #[serde(default)]
    enable_llm_refinement: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
}

async fn parse_diarization_request(req: Request) -> Result<DiarizationRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<JsonRequestBody>, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid JSON payload: {e}")))?;

        return Ok(DiarizationRequest {
            audio_base64: Some(payload.audio_base64),
            diarization_model: payload.model.or(payload.diarization_model),
            asr_model: payload.asr_model,
            aligner_model: payload.aligner_model,
            llm_model: payload.llm_model,
            response_format: payload.response_format,
            num_speakers: payload.num_speakers,
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
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;

        let mut out = DiarizationRequest::default();

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::bad_request(format!("Failed reading multipart field: {e}")))?
        {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let bytes = field.bytes().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {}",
                            name, e
                        ))
                    })?;
                    if !bytes.is_empty() {
                        out.audio_base64 =
                            Some(base64::engine::general_purpose::STANDARD.encode(&bytes));
                    }
                }
                "audio_base64" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_base64' field: {}",
                            e
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.audio_base64 = Some(text.trim().to_string());
                    }
                }
                "model" | "diarization_model" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {e}",
                            name
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.diarization_model = Some(text.trim().to_string());
                    }
                }
                "asr_model" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'asr_model' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.asr_model = Some(text.trim().to_string());
                    }
                }
                "aligner_model" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'aligner_model' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.aligner_model = Some(text.trim().to_string());
                    }
                }
                "llm_model" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'llm_model' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.llm_model = Some(text.trim().to_string());
                    }
                }
                "response_format" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'response_format' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.response_format = Some(text.trim().to_string());
                    }
                }
                "min_speakers" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_speakers' field: {e}"
                        ))
                    })?;
                    out.min_speakers = text.trim().parse::<usize>().ok();
                }
                "num_speakers" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'num_speakers' field: {e}"
                        ))
                    })?;
                    out.num_speakers = text.trim().parse::<usize>().ok();
                }
                "max_speakers" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'max_speakers' field: {e}"
                        ))
                    })?;
                    out.max_speakers = text.trim().parse::<usize>().ok();
                }
                "min_speech_duration_ms" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_speech_duration_ms' field: {e}"
                        ))
                    })?;
                    out.min_speech_duration_ms = text.trim().parse::<f32>().ok();
                }
                "min_silence_duration_ms" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_silence_duration_ms' field: {e}"
                        ))
                    })?;
                    out.min_silence_duration_ms = text.trim().parse::<f32>().ok();
                }
                "enable_llm_refinement" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'enable_llm_refinement' field: {e}"
                        ))
                    })?;
                    out.enable_llm_refinement = Some(matches!(
                        text.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    ));
                }
                "stream" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'stream' field: {e}"
                        ))
                    })?;
                    out.stream = matches!(
                        text.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    );
                }
                _ => {}
            }
        }

        return Ok(out);
    }

    Err(ApiError {
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        message: "Expected `Content-Type: application/json` or `multipart/form-data`".to_string(),
    })
}

fn resolve_speaker_bounds(
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    num_speakers: Option<usize>,
) -> Result<(Option<usize>, Option<usize>), ApiError> {
    let resolved_min = min_speakers.or(num_speakers);
    let resolved_max = max_speakers.or(num_speakers);

    if let (Some(min), Some(max)) = (resolved_min, resolved_max) {
        if min > max {
            return Err(ApiError::bad_request(
                "`min_speakers` cannot be greater than `max_speakers`.",
            ));
        }
    }

    Ok((resolved_min, resolved_max))
}

#[cfg(test)]
mod tests {
    use super::resolve_speaker_bounds;

    #[test]
    fn num_speakers_backfills_missing_bounds() {
        let (min, max) = resolve_speaker_bounds(None, None, Some(3)).expect("bounds should resolve");
        assert_eq!(min, Some(3));
        assert_eq!(max, Some(3));
    }

    #[test]
    fn rejects_invalid_speaker_bounds() {
        let err = resolve_speaker_bounds(Some(4), Some(2), None)
            .expect_err("min should not exceed max");
        assert_eq!(
            err.message,
            "`min_speakers` cannot be greater than `max_speakers`."
        );
    }
}
