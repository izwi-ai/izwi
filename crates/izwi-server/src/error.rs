//! API errors. The error type, its class table and the request scope live
//! in izwi-gateway, which the gateway shares; this module maps the engine's
//! errors onto them.

use izwi_core::ErrorKind;
pub use izwi_gateway::error::*;

/// The class of an engine error kind on the local role.
pub(crate) fn core_error_class(kind: ErrorKind) -> ErrorClass {
    match kind {
        ErrorKind::InvalidInput => ErrorClass::InvalidRequest,
        ErrorKind::ContextLengthExceeded => ErrorClass::ContextLengthExceeded,
        ErrorKind::ModelNotFound => ErrorClass::ModelNotFound,
        ErrorKind::ModelBusy => ErrorClass::Conflict,
        ErrorKind::CapacityExhausted | ErrorKind::Unavailable => ErrorClass::Unavailable,
        ErrorKind::Timeout => ErrorClass::Timeout,
        ErrorKind::Cancelled => ErrorClass::Cancelled,
        ErrorKind::Internal => ErrorClass::Internal,
    }
}

/// The API error for an engine error: its kind's class, and its message.
pub(crate) fn core_api_error(err: izwi_core::Error) -> ApiError {
    let class = core_error_class(err.kind());
    match err {
        // OpenAI's message for an overflow is the bare explanation.
        izwi_core::Error::ContextLengthExceeded(message) => ApiError::of(class, message),
        err => ApiError::of(class, err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::to_bytes,
        http::StatusCode,
        response::{IntoResponse, Response},
    };
    use izwi_serving_worker::{admission_rejection_code, execution_error_code};
    use serde_json::Value;

    use super::*;

    async fn body(response: Response) -> Value {
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// One row per core error kind: the class the local role answers with,
    /// and the classes the gateway answers with when a worker reports the
    /// same kind before and after accepting the invocation. Where a row
    /// differs, the worker's view of the failure differs (see
    /// `admission_rejection_code`); the table pins every such case.
    #[test]
    fn core_kinds_map_alike_on_both_roles() {
        use ErrorClass as C;
        let rows = [
            (
                ErrorKind::InvalidInput,
                C::InvalidRequest,
                C::InvalidRequest,
                C::InvalidRequest,
            ),
            (
                ErrorKind::ContextLengthExceeded,
                C::ContextLengthExceeded,
                C::ContextLengthExceeded,
                C::InvalidRequest,
            ),
            (
                ErrorKind::ModelNotFound,
                C::ModelNotFound,
                C::Unavailable,
                C::Internal,
            ),
            (
                ErrorKind::ModelBusy,
                C::Conflict,
                C::Unavailable,
                C::Internal,
            ),
            (
                ErrorKind::CapacityExhausted,
                C::Unavailable,
                C::Unavailable,
                C::Unavailable,
            ),
            (ErrorKind::Timeout, C::Timeout, C::Unavailable, C::Timeout),
            (
                ErrorKind::Cancelled,
                C::Cancelled,
                C::Unavailable,
                C::Internal,
            ),
            (
                ErrorKind::Unavailable,
                C::Unavailable,
                C::Unavailable,
                C::Unavailable,
            ),
            (
                ErrorKind::Internal,
                C::Internal,
                C::Unavailable,
                C::Internal,
            ),
        ];
        for (kind, local, admission, execution) in rows {
            assert_eq!(core_error_class(kind), local, "{kind:?} local");
            assert_eq!(
                rejection_class(admission_rejection_code(kind)),
                admission,
                "{kind:?} rejected by a worker"
            );
            assert_eq!(
                invocation_error_class(execution_error_code(kind)),
                execution,
                "{kind:?} failed on a worker"
            );
        }
        let covered: std::collections::HashSet<_> = rows.iter().map(|row| row.0).collect();
        assert_eq!(covered, ErrorKind::ALL.into_iter().collect());
    }

    #[test]
    fn missing_dependency_maps_to_service_unavailable() {
        let api_error = core_api_error(izwi_core::Error::MissingDependency(
            "espeak-ng not found".to_string(),
        ));

        assert_eq!(api_error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(api_error.message.contains("espeak-ng not found"));
    }

    /// Statuses follow the error's kind, never its text: a load refused for
    /// capacity is a `503` because the runtime types it as overloaded, and
    /// the same advice inside a load failure stays a server fault.
    #[test]
    fn runtime_errors_map_to_client_and_capacity_statuses() {
        for (error, status) in [
            (
                izwi_core::Error::InvalidInput("audio exceeds 60 minutes".into()),
                StatusCode::BAD_REQUEST,
            ),
            (
                izwi_core::Error::ModelNotFound("Unknown-ASR".into()),
                StatusCode::NOT_FOUND,
            ),
            (
                izwi_core::Error::Overloaded(
                    "Cannot load Whisper-Large-v3-Turbo: the 2-model residency budget is full \
                     and no resident model is idle and available for eviction; unload a model \
                     to make room"
                        .into(),
                ),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                izwi_core::Error::ModelLoadError("unload a model to make room".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                izwi_core::Error::Overloaded("all ASR slots are busy".into()),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                izwi_core::Error::ModelBusy("Cannot unload model Kokoro-82M".into()),
                StatusCode::CONFLICT,
            ),
            (
                izwi_core::Error::ConfigError("invalid memory budget".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                izwi_core::Error::ModelLoadError("safetensors header is corrupt".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                izwi_core::Error::InferenceError("device lost".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ] {
            let message = error.to_string();
            let api_error = core_api_error(error);
            assert_eq!(api_error.status, status, "{message}");
            assert_eq!(api_error.message, message);
        }
        let missing = core_api_error(izwi_core::Error::ModelNotFound("Unknown-ASR".into()));
        let body = missing.openai_body();
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }

    /// D8: context overflow is OpenAI's `context_length_exceeded` 400, so
    /// clients can tell it from other bad requests; its message is the bare
    /// explanation.
    #[tokio::test]
    async fn context_overflow_is_an_openai_context_length_error() {
        let error = core_api_error(izwi_core::Error::ContextLengthExceeded(
            "This model's maximum context length is 32 tokens.".into(),
        ));
        assert_eq!(
            error.message,
            "This model's maximum context length is 32 tokens."
        );
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_eq!(
            ApiError::context_length_exceeded("x").openai_body()["error"]["param"],
            "messages"
        );
    }
}
