//! Runs one batch line through its synchronous route's own code, at the
//! Batch class whatever tier the body names.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::header;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;

use super::executor::{LineJob, LineOutcome, LineRunner};
use super::BatchEndpoint;
use crate::api::audio_payload::{decode_base64_audio_payload, AudioPayload};
use crate::api::openai::audio::resolve_audio_upload_limit_bytes;
use crate::api::openai::audio::speech::{prepare_speech, run_prepared_speech, SpeechRequest};
use crate::api::openai::audio::transcriptions::{self, BatchAudio};
use crate::api::openai::service_tier::ServiceTier;
use crate::error::{store_api_error, ApiError};
use crate::file_store::{FilePurpose, FileSpool};
use crate::speech_outcomes::SpeechMode;
use crate::state::AppState;

/// The server's line runner.
pub(crate) fn audio_line_runner(state: AppState) -> LineRunner {
    Arc::new(move |job| {
        let state = state.clone();
        Box::pin(async move {
            let result = match job.endpoint {
                BatchEndpoint::Speech => speech_line(&state, &job).await,
                BatchEndpoint::Transcriptions => transcription_line(&state, &job, false).await,
                BatchEndpoint::Translations => transcription_line(&state, &job, true).await,
            };
            match result {
                Ok(body) => LineOutcome {
                    status_code: 200,
                    body,
                },
                Err(error) => LineOutcome {
                    status_code: error.status.as_u16(),
                    body: error.openai_body(),
                },
            }
        })
    })
}

/// Every line is batch work.
fn batch_tier() -> ServiceTier {
    ServiceTier::default().or_flex()
}

/// A `/v1/audio/speech` line. The audio becomes a `batch_output` file named
/// after the line, and the response body points at it.
async fn speech_line(state: &AppState, job: &LineJob) -> Result<Value, ApiError> {
    let request = SpeechRequest::deserialize(&job.body)
        .map_err(|err| ApiError::bad_request(format!("Invalid request body: {err}")))?;
    let prepared = prepare_speech(
        state,
        job.tenant.scheduling_key(),
        job.request_id.clone(),
        request,
        Some(SpeechMode::BatchItem),
        batch_tier(),
    )
    .await?;
    let output = prepared.output().clone();
    let response = run_prepared_speech(state.clone(), prepared).await?;
    let audio = to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|err| ApiError::internal(format!("Reading the line's audio failed: {err}")))?;
    let spool_error =
        |err: std::io::Error| ApiError::internal(format!("Failed to spool the audio: {err}"));
    let mut spool = FileSpool::new().map_err(spool_error)?;
    spool.write(&audio).await.map_err(spool_error)?;
    let filename = crate::api::openai::files::sanitize_filename(Some(&format!(
        "{}.{}",
        job.custom_id,
        output.label()
    )));
    let file = state
        .file_store
        .create(&job.tenant, FilePurpose::BatchOutput, filename, spool)
        .await
        .map_err(|err| store_api_error(err, "File storage error"))?;
    Ok(json!({
        "audio_file_id": file.id,
        "format": output.label(),
        "bytes": file.size_bytes,
    }))
}

/// A `/v1/audio/transcriptions` or `/v1/audio/translations` line. The
/// response body is the route's JSON; a text format (`text`, `srt`, `vtt`)
/// is wrapped as `{"text": …}`.
async fn transcription_line(
    state: &AppState,
    job: &LineJob,
    translate: bool,
) -> Result<Value, ApiError> {
    let request = transcriptions::parse_batch_body(&job.body)?;
    let audio = match request.audio()? {
        BatchAudio::Base64(encoded) => decode_base64_audio_payload(encoded)?,
        BatchAudio::FileId(file_id) => read_audio_file(state, job, file_id).await?,
    };
    let response = transcriptions::run_batch_line(
        state.clone(),
        job.request_id.clone(),
        request,
        audio,
        translate,
        batch_tier(),
    )
    .await?;
    response_json(response).await
}

/// The audio a line names by `file_id`: a `user_data` file of the batch's
/// own principal, within the audio routes' upload limit.
async fn read_audio_file(
    state: &AppState,
    job: &LineJob,
    file_id: &str,
) -> Result<AudioPayload, ApiError> {
    let opened = state
        .file_store
        .open(&job.tenant, file_id)
        .await
        .map_err(|err| store_api_error(err, "File storage error"))?;
    let Some((file, content)) = opened else {
        return Err(ApiError::invalid_param(
            "file_id",
            format!("No such File object: {file_id}"),
        ));
    };
    if file.purpose != FilePurpose::UserData {
        return Err(ApiError::invalid_param(
            "file_id",
            format!(
                "File {file_id} has purpose `{}`; audio is uploaded with purpose `user_data`",
                file.purpose.as_str()
            ),
        ));
    }
    let limit = resolve_audio_upload_limit_bytes();
    if file.size_bytes > limit as u64 {
        return Err(ApiError::payload_too_large(format!(
            "File {file_id} is larger than the audio upload limit of {} MiB",
            limit / (1024 * 1024)
        )));
    }
    let mut bytes = Vec::with_capacity(file.size_bytes as usize);
    let mut reader = content.reader;
    reader
        .read_to_end(&mut bytes)
        .await
        .map_err(|err| ApiError::internal(format!("Reading file {file_id} failed: {err}")))?;
    Ok(AudioPayload::from_bytes(bytes, None, Some(file.filename)))
}

async fn response_json(response: Response<Body>) -> Result<Value, ApiError> {
    let is_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"));
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|err| ApiError::internal(format!("Reading the line's response failed: {err}")))?;
    if is_json {
        serde_json::from_slice(&body)
            .map_err(|err| ApiError::internal(format!("The line's response is not JSON: {err}")))
    } else {
        Ok(json!({"text": String::from_utf8_lossy(&body)}))
    }
}
