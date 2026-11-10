//! `POST /v1/audio/speech/batch`: many speech requests in one call, in the
//! envelope SGLang-Omni and vLLM-Omni share:
//!
//! ```json
//! {"model": "...", "voice": "...", "response_format": "mp3",
//!  "items": [{"input": "One."}, {"input": "Two.", "speed": 1.1}]}
//! → {"id": "speech_batch_...", "object": "audio.speech.batch",
//!    "results": [{"index": 0, "status": "success", "id": "speech_...",
//!                 "audio_data": "<base64>", "format": "mp3", ...},
//!                {"index": 1, "status": "error", "error": {...}}],
//!    "total": 2, "succeeded": 1, "failed": 1}
//! ```
//!
//! Top-level speech fields are defaults each item overrides; `service_tier`
//! (or the `X-Izwi-Service-Tier` header) is the batch's alone. A batch is
//! batch work (`flex`) unless it names another tier. Items run
//! concurrently up to `IZWI_TTS_BATCH_CONCURRENCY`, each with its own
//! admission permit, so the engine batches items of one model and one batch
//! cannot hold every slot; results keep the items' order. Every item gets a
//! speech id `GET /v1/audio/speech/{id}` answers. A disconnect drops the
//! response future and with it every item still running.

use axum::{
    body::to_bytes,
    extract::{Extension, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine;
use futures::StreamExt;
use serde::Serialize;
use serde_json::{Map, Value};

use super::speech::{prepare_speech, run_prepared_speech, SpeechRequest};
use crate::api::openai::service_tier::{ServiceTier, SERVICE_TIER_HEADER};
use crate::api::request_context::RequestContext;
use crate::error::{ApiError, ApiJson};
use crate::speech_outcomes::SpeechMode;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct SpeechBatchResponse {
    id: String,
    object: &'static str,
    results: Vec<SpeechBatchResult>,
    total: usize,
    succeeded: usize,
    failed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct SpeechBatchResult {
    index: usize,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

impl SpeechBatchResult {
    fn failure(index: usize, id: Option<String>, error: &ApiError) -> Self {
        let param = error
            .detail
            .and_then(|detail| detail.param)
            .map(|param| format!("items.{index}.{param}"));
        Self {
            index,
            status: "error",
            id,
            audio_data: None,
            format: None,
            media_type: None,
            sample_rate: None,
            audio_duration_secs: None,
            finish_reason: None,
            usage: None,
            error: Some(serde_json::json!({
                "message": error.message,
                "type": if error.status.is_client_error() { "invalid_request_error" } else { "server_error" },
                "param": param,
                "code": error.detail.and_then(|detail| detail.code),
            })),
        }
    }
}

/// Fields that make no sense per item of a batch (a batch is one JSON
/// response).
const STREAM_FIELDS: [&str; 2] = ["stream", "stream_format"];

fn reject_streaming(fields: &Map<String, Value>, at: &str) -> Result<(), ApiError> {
    for field in STREAM_FIELDS {
        match fields.get(field) {
            None | Some(Value::Null) | Some(Value::Bool(false)) => {}
            Some(_) => {
                return Err(ApiError::bad_request(format!(
                    "`{at}{field}` is not allowed in a batch; each item returns a whole file"
                )))
            }
        }
    }
    Ok(())
}

pub async fn speech_batch(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    headers: HeaderMap,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    let tuning = &crate::env_config::tuning().speech_api;
    let mut defaults = body;
    let service_tier = batch_service_tier(&mut defaults, &headers)?;
    let items = match defaults.remove("items") {
        Some(Value::Array(items)) => items,
        Some(_) => return Err(ApiError::invalid_param("items", "`items` must be an array")),
        None => return Err(ApiError::invalid_param("items", "`items` is required")),
    };
    if items.is_empty() {
        return Err(ApiError::invalid_param(
            "items",
            "`items` must not be empty",
        ));
    }
    if items.len() > tuning.batch_max_items {
        return Err(ApiError::invalid_param(
            "items",
            format!(
                "A batch holds at most {} items (IZWI_TTS_BATCH_MAX_ITEMS), not {}",
                tuning.batch_max_items,
                items.len()
            ),
        ));
    }
    reject_streaming(&defaults, "")?;
    let mut merged = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let Value::Object(item) = item else {
            return Err(ApiError::bad_request(format!(
                "`items.{index}` must be an object"
            )));
        };
        reject_streaming(&item, &format!("items.{index}."))?;
        if item.contains_key("service_tier") {
            return Err(ApiError::bad_request(format!(
                "`items.{index}.service_tier` is not allowed; `service_tier` applies to the whole batch"
            )));
        }
        let mut fields = defaults.clone();
        fields.extend(item);
        merged.push(fields);
    }

    let tenant_key = ctx.tenant_key();
    let budget = tuning.batch_max_output_bytes;
    let total = merged.len();
    let jobs: Vec<_> = merged
        .into_iter()
        .enumerate()
        .map(|(index, fields)| {
            run_item(
                state.clone(),
                tenant_key,
                format!("{}-{index}", ctx.correlation_id),
                index,
                fields,
                service_tier,
            )
        })
        .collect();
    let results = futures::stream::iter(jobs)
        .buffered(tuning.batch_concurrency)
        .collect::<Vec<_>>()
        .await;

    // Keep results within the output budget, in order.
    let mut spent = 0usize;
    let mut out = Vec::with_capacity(total);
    for mut result in results {
        let size = result.audio_data.as_ref().map_or(0, String::len);
        if size > 0 && spent + size > budget {
            let error = ApiError::of(
                crate::error::ErrorClass::PayloadTooLarge,
                format!(
                    "The batch's audio passed {} MiB (IZWI_TTS_BATCH_MAX_OUTPUT_MB); split the batch",
                    budget / (1024 * 1024)
                ),
            );
            result = SpeechBatchResult::failure(result.index, result.id.take(), &error);
        } else {
            spent += size;
        }
        out.push(result);
    }
    let succeeded = out
        .iter()
        .filter(|result| result.status == "success")
        .count();
    let mut response = Json(SpeechBatchResponse {
        id: format!("speech_batch_{}", uuid::Uuid::new_v4()),
        object: "audio.speech.batch",
        total,
        succeeded,
        failed: total - succeeded,
        results: out,
        service_tier: service_tier.reported(),
    })
    .into_response();
    service_tier.stamp(&mut response);
    Ok(response)
}

/// The tier of a batch: its top-level `service_tier` (taken out of the
/// item defaults) or the request header, and `flex` when neither names one.
fn batch_service_tier(
    defaults: &mut Map<String, Value>,
    headers: &HeaderMap,
) -> Result<ServiceTier, ApiError> {
    let field = match defaults.remove("service_tier") {
        None | Some(Value::Null) => None,
        Some(Value::String(tier)) => Some(tier),
        Some(_) => {
            return Err(ApiError::invalid_param(
                "service_tier",
                "`service_tier` must be a string",
            ))
        }
    };
    ServiceTier::parse(field.as_deref(), headers.get(SERVICE_TIER_HEADER)).map(ServiceTier::or_flex)
}

async fn run_item(
    state: AppState,
    tenant_key: Option<[u8; 32]>,
    correlation_id: String,
    index: usize,
    fields: Map<String, Value>,
    service_tier: ServiceTier,
) -> SpeechBatchResult {
    let req: SpeechRequest = match serde_json::from_value(Value::Object(fields)) {
        Ok(req) => req,
        Err(err) => {
            return SpeechBatchResult::failure(
                index,
                None,
                &ApiError::bad_request(format!("Invalid item: {err}")),
            )
        }
    };
    let prepared = match prepare_speech(
        &state,
        tenant_key,
        correlation_id,
        req,
        Some(SpeechMode::BatchItem),
        service_tier,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => return SpeechBatchResult::failure(index, None, &error),
    };
    debug_assert!(!prepared.is_streamed());
    let id = prepared.speech_id().to_string();
    let output = prepared.output().clone();
    let response = match run_prepared_speech(state.clone(), prepared).await {
        Ok(response) => response,
        Err(error) => return SpeechBatchResult::failure(index, Some(id), &error),
    };
    // Read the headers before awaiting the body: a borrowed `Response` is
    // not `Send` (its body is not `Sync`).
    let (parts, body) = response.into_parts();
    let header = |name: &str| {
        parts
            .headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    let sample_rate: Option<u32> =
        header("X-Audio-Sample-Rate").and_then(|value| value.parse().ok());
    let duration: Option<f32> =
        header("X-Audio-Duration-Secs").and_then(|value| value.parse().ok());
    let tokens: usize = header("X-Tokens-Generated")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let truncated = header("X-Output-Truncated").is_some_and(|value| value == "true");
    let bytes = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return SpeechBatchResult::failure(
                index,
                Some(id),
                &ApiError::internal(format!("Reading the item's audio failed: {err}")),
            )
        }
    };
    SpeechBatchResult {
        index,
        status: "success",
        id: Some(id),
        audio_data: Some(base64::engine::general_purpose::STANDARD.encode(&bytes)),
        format: Some(output.label()),
        media_type: Some(output.content_type()),
        sample_rate,
        audio_duration_secs: duration,
        finish_reason: Some(if truncated { "length" } else { "stop" }),
        usage: Some(serde_json::json!({
            "input_tokens": 0,
            "output_tokens": tokens,
            "total_tokens": tokens,
        })),
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_fields_are_refused_at_either_level() {
        let fields = |value: Value| match value {
            Value::Object(map) => map,
            _ => unreachable!(),
        };
        assert!(reject_streaming(&fields(serde_json::json!({"stream": false})), "").is_ok());
        assert!(reject_streaming(&fields(serde_json::json!({"stream": true})), "").is_err());
        let error = reject_streaming(
            &fields(serde_json::json!({"stream_format": "sse"})),
            "items.2.",
        )
        .unwrap_err();
        assert!(error.message.contains("items.2.stream_format"));
    }

    #[test]
    fn the_batch_tier_is_its_top_level_field_or_the_header() {
        let fields = |value: Value| match value {
            Value::Object(map) => map,
            _ => unreachable!(),
        };
        let mut headers = HeaderMap::new();
        let mut defaults = fields(serde_json::json!({"model": "m", "service_tier": "flex"}));
        let tier = batch_service_tier(&mut defaults, &headers).unwrap();
        assert_eq!(tier.reported(), Some("flex"));
        // The tier is the batch's, not an item default.
        assert!(!defaults.contains_key("service_tier"));

        headers.insert(SERVICE_TIER_HEADER, "priority".parse().unwrap());
        let tier = batch_service_tier(&mut fields(serde_json::json!({})), &headers).unwrap();
        assert_eq!(tier.reported(), Some("default"));

        // A batch that names no tier is batch work, and does not report one.
        let tier =
            batch_service_tier(&mut fields(serde_json::json!({})), &HeaderMap::new()).unwrap();
        assert_eq!(tier.as_str(), "flex");
        assert_eq!(tier.reported(), None);

        for bad in [
            serde_json::json!({"service_tier": "fastest"}),
            serde_json::json!({"service_tier": 1}),
        ] {
            let error = batch_service_tier(&mut fields(bad), &HeaderMap::new()).unwrap_err();
            assert_eq!(
                error.detail.and_then(|detail| detail.param),
                Some("service_tier")
            );
        }
    }

    #[test]
    fn item_errors_name_their_item() {
        let error = ApiError::invalid_param("speed", "too fast");
        let result = SpeechBatchResult::failure(3, None, &error);
        let json = serde_json::to_value(result).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["error"]["param"], "items.3.speed");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json.get("audio_data").is_none());
    }
}
