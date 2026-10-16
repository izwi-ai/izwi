//! `Idempotency-Key` handling shared by the durable create routes.
//!
//! A keyed create reserves its tenant-scoped key before any work, commits the
//! replay result in the same transaction that accepts the job, and releases
//! the reservation when acceptance fails.

use axum::http::{HeaderMap, StatusCode};

use crate::batch_runtime::store::{
    BatchRuntimeStore, DurableIdempotencyBegin, DurableIdempotencyReplay,
    DurableIdempotencyRequest, DurableIdempotencyReservation, MAX_DURABLE_IDEMPOTENCY_KEY_BYTES,
};
use crate::error::ApiError;

/// Replay results are kept this long after acceptance.
pub(crate) const DURABLE_IDEMPOTENCY_RETENTION_MS: u64 = 24 * 60 * 60 * 1_000;

/// How a keyed create continues: with a fresh reservation, or by replaying the
/// result an earlier identical request committed.
pub(crate) enum DurableIdempotencyStart {
    Reserved(DurableIdempotencyReservation),
    Replay(DurableIdempotencyReplay),
}

/// The request's `Idempotency-Key` (or `X-Idempotency-Key`): bounded visible
/// text, and one value when both headers are sent.
pub(crate) fn extract_idempotency_key(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let mut selected: Option<String> = None;
    for name in ["idempotency-key", "x-idempotency-key"] {
        for value in headers.get_all(name).iter() {
            let value = value.to_str().map_err(|_| {
                ApiError::bad_request("Idempotency-Key must contain valid visible text")
            })?;
            let value = value.trim();
            if value.is_empty() {
                return Err(ApiError::bad_request("Idempotency-Key cannot be empty"));
            }
            if value.len() > MAX_DURABLE_IDEMPOTENCY_KEY_BYTES {
                return Err(ApiError::bad_request(format!(
                    "Idempotency-Key exceeds {MAX_DURABLE_IDEMPOTENCY_KEY_BYTES} bytes"
                )));
            }
            if value.chars().any(char::is_control) {
                return Err(ApiError::bad_request(
                    "Idempotency-Key cannot contain control characters",
                ));
            }
            match selected.as_deref() {
                Some(existing) if existing != value => {
                    return Err(ApiError::bad_request(
                        "Conflicting Idempotency-Key header values",
                    ));
                }
                Some(_) => {}
                None => selected = Some(value.to_string()),
            }
        }
    }
    Ok(selected)
}

/// Reserve a keyed create, or return the replay a committed key holds. A key
/// reused for a different request, or still being accepted, is a 409; an
/// exhausted quota is a 503. `noun` names the resource in error messages.
pub(crate) async fn begin_durable_idempotency(
    store: &BatchRuntimeStore,
    request: DurableIdempotencyRequest,
    noun: &str,
) -> Result<DurableIdempotencyStart, ApiError> {
    match store
        .reserve_durable_idempotency(request)
        .await
        .map_err(|err| crate::error::store_api_error(err, "Idempotency storage error"))?
    {
        DurableIdempotencyBegin::Acquired(reservation) => {
            Ok(DurableIdempotencyStart::Reserved(reservation))
        }
        DurableIdempotencyBegin::Replay(replay) => Ok(DurableIdempotencyStart::Replay(replay)),
        DurableIdempotencyBegin::Conflict => Err(conflict(format!(
            "Idempotency-Key was already used for a different {noun} request"
        ))),
        DurableIdempotencyBegin::InProgress { .. } => Err(conflict(format!(
            "A matching {noun} request is still being accepted"
        ))),
        DurableIdempotencyBegin::CapacityExceeded => Err(ApiError::service_unavailable(
            "Durable idempotency capacity is exhausted",
        )),
    }
}

/// Give a reservation back after its create failed, so a retry can proceed.
/// Best effort: an unreleased reservation expires on its own.
pub(crate) async fn release_durable_idempotency(
    store: &BatchRuntimeStore,
    reservation: Option<&DurableIdempotencyReservation>,
) {
    if let Some(reservation) = reservation {
        let _ = store.release_durable_idempotency(reservation).await;
    }
}

/// The reservation expired, or was taken over, before the job committed.
pub(crate) fn reservation_lost(noun: &str) -> ApiError {
    conflict(format!(
        "{noun} idempotency reservation expired before acceptance"
    ))
}

/// The `record_id` a committed replay result names.
pub(crate) fn replay_record_id(replay: &DurableIdempotencyReplay) -> Result<String, ApiError> {
    replay
        .response_json
        .get("record_id")
        .or_else(|| replay.response_json.get("id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| ApiError::internal("Stored idempotency response is invalid"))
}

fn conflict(message: String) -> ApiError {
    ApiError {
        status: StatusCode::CONFLICT,
        message,
        detail: None,
    }
}
