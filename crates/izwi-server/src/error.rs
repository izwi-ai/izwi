//! API error handling

use axum::{
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use izwi_core::ErrorKind;
use izwi_serving_protocol::{InvocationErrorCode, RejectionCode};
use serde_json::json;

/// Seconds a client should wait before retrying a `503`. Capacity refusals
/// clear when another request finishes or a model goes idle, which is
/// usually within seconds.
const SERVICE_UNAVAILABLE_RETRY_AFTER_SECS: &str = "5";

/// The `499` status nginx and izwi use for a request the client cancelled.
const CLIENT_CLOSED_REQUEST: u16 = 499;

/// The class of an API error: the one table from which both roles (local
/// and gateway) take an error's HTTP status, OpenAI `type` and `code`, and
/// whether it is retryable. Core error kinds and worker protocol codes map
/// into a class; nothing maps straight to a status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    InvalidRequest,
    ContextLengthExceeded,
    PayloadTooLarge,
    UnsupportedMediaType,
    Unauthenticated,
    PermissionDenied,
    NotFound,
    ModelNotFound,
    Conflict,
    RateLimited,
    Timeout,
    Cancelled,
    /// Out of capacity, or not ready (a model loading, a worker draining):
    /// the same request can succeed shortly.
    Unavailable,
    /// A worker answered the gateway with something it could not use.
    BadGateway,
    Internal,
}

impl ErrorClass {
    pub const ALL: [Self; 15] = [
        Self::InvalidRequest,
        Self::ContextLengthExceeded,
        Self::PayloadTooLarge,
        Self::UnsupportedMediaType,
        Self::Unauthenticated,
        Self::PermissionDenied,
        Self::NotFound,
        Self::ModelNotFound,
        Self::Conflict,
        Self::RateLimited,
        Self::Timeout,
        Self::Cancelled,
        Self::Unavailable,
        Self::BadGateway,
        Self::Internal,
    ];

    pub fn status(self) -> StatusCode {
        match self {
            Self::InvalidRequest | Self::ContextLengthExceeded => StatusCode::BAD_REQUEST,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::Unauthenticated => StatusCode::UNAUTHORIZED,
            Self::PermissionDenied => StatusCode::FORBIDDEN,
            Self::NotFound | Self::ModelNotFound => StatusCode::NOT_FOUND,
            Self::Conflict => StatusCode::CONFLICT,
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,
            Self::Cancelled => {
                StatusCode::from_u16(CLIENT_CLOSED_REQUEST).unwrap_or(StatusCode::REQUEST_TIMEOUT)
            }
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::BadGateway => StatusCode::BAD_GATEWAY,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// OpenAI's `error.type`.
    pub fn openai_type(self) -> &'static str {
        match self {
            Self::InvalidRequest
            | Self::ContextLengthExceeded
            | Self::PayloadTooLarge
            | Self::UnsupportedMediaType
            | Self::Conflict => "invalid_request_error",
            Self::Unauthenticated => "authentication_error",
            Self::PermissionDenied => "permission_denied_error",
            Self::NotFound | Self::ModelNotFound => "not_found_error",
            Self::RateLimited => "rate_limit_error",
            Self::Unavailable => "service_unavailable_error",
            Self::Timeout | Self::Cancelled | Self::BadGateway | Self::Internal => "server_error",
        }
    }

    /// OpenAI's `error.code`: what went wrong, never the HTTP status.
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_value",
            Self::ContextLengthExceeded => "context_length_exceeded",
            Self::PayloadTooLarge => "payload_too_large",
            Self::UnsupportedMediaType => "unsupported_media_type",
            Self::Unauthenticated => "invalid_api_key",
            Self::PermissionDenied => "permission_denied",
            Self::NotFound => "not_found",
            Self::ModelNotFound => "model_not_found",
            Self::Conflict => "conflict",
            Self::RateLimited => "rate_limit_exceeded",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::Unavailable => "service_unavailable",
            Self::BadGateway => "bad_gateway",
            Self::Internal => "server_error",
        }
    }

    /// OpenAI's `error.param` when the class always names one.
    pub fn param(self) -> Option<&'static str> {
        match self {
            Self::ModelNotFound => Some("model"),
            _ => None,
        }
    }

    /// Whether the same request can succeed if sent again shortly: the
    /// client may retry, and the gateway tries another worker before a
    /// worker accepts it.
    pub fn retryable(self) -> bool {
        matches!(self, Self::Unavailable | Self::RateLimited)
    }

    /// The general class of a status, for errors built from a status alone.
    pub fn from_status(status: StatusCode) -> Self {
        match status.as_u16() {
            401 => Self::Unauthenticated,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            408 => Self::Timeout,
            409 => Self::Conflict,
            413 => Self::PayloadTooLarge,
            415 => Self::UnsupportedMediaType,
            429 => Self::RateLimited,
            CLIENT_CLOSED_REQUEST => Self::Cancelled,
            502 => Self::BadGateway,
            503 => Self::Unavailable,
            code if code < 500 => Self::InvalidRequest,
            _ => Self::Internal,
        }
    }
}

impl From<ErrorKind> for ErrorClass {
    fn from(kind: ErrorKind) -> Self {
        match kind {
            ErrorKind::InvalidInput => Self::InvalidRequest,
            ErrorKind::ContextLengthExceeded => Self::ContextLengthExceeded,
            ErrorKind::ModelNotFound => Self::ModelNotFound,
            ErrorKind::ModelBusy => Self::Conflict,
            ErrorKind::CapacityExhausted | ErrorKind::Unavailable => Self::Unavailable,
            ErrorKind::Timeout => Self::Timeout,
            ErrorKind::Cancelled => Self::Cancelled,
            ErrorKind::Internal => Self::Internal,
        }
    }
}

/// The class of a worker's admission rejection. Rejections a different
/// worker might accept (busy, not ready, draining, routed with stale
/// identity) are retryable [`ErrorClass::Unavailable`], so the gateway
/// tries another worker before answering `503`.
pub(crate) fn rejection_class(code: RejectionCode) -> ErrorClass {
    match code {
        RejectionCode::InvalidRequest | RejectionCode::IncompatibleTask => {
            ErrorClass::InvalidRequest
        }
        RejectionCode::ContextLengthExceeded => ErrorClass::ContextLengthExceeded,
        RejectionCode::Unauthorized | RejectionCode::PolicyDenied => ErrorClass::PermissionDenied,
        RejectionCode::UnknownDeployment
        | RejectionCode::WrongWorkerIncarnation
        | RejectionCode::WrongModelGeneration
        | RejectionCode::ModelNotReady
        | RejectionCode::WorkerDraining
        | RejectionCode::CapacityExhausted
        | RejectionCode::QueueWaitExceeded => ErrorClass::Unavailable,
        RejectionCode::DuplicateAttemptConflict => ErrorClass::Conflict,
        RejectionCode::Unauthenticated
        | RejectionCode::UnsupportedProtocolVersion
        | RejectionCode::Unknown => ErrorClass::BadGateway,
    }
}

/// The class of a worker's terminal error after it accepted an invocation.
pub(crate) fn invocation_error_class(code: InvocationErrorCode) -> ErrorClass {
    match code {
        InvocationErrorCode::InvalidInput => ErrorClass::InvalidRequest,
        InvocationErrorCode::DeadlineExceeded => ErrorClass::Timeout,
        InvocationErrorCode::WorkerUnavailable => ErrorClass::Unavailable,
        InvocationErrorCode::ExecutionFailed
        | InvocationErrorCode::OutputLimitExceeded
        | InvocationErrorCode::Internal
        | InvocationErrorCode::Unknown => ErrorClass::Internal,
    }
}

/// API error type
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    /// The OpenAI error `param` and `code`, for errors that name them.
    pub detail: Option<ApiErrorDetail>,
}

/// The request parameter an error is about and its machine-readable code,
/// rendered as OpenAI's `error.param` and `error.code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiErrorDetail {
    pub param: Option<&'static str>,
    pub code: Option<&'static str>,
}

impl ApiError {
    /// An error of `class`. The detail is recorded only when the class's
    /// code or param differs from what its status alone would give.
    pub fn of(class: ErrorClass, msg: impl Into<String>) -> Self {
        let status = class.status();
        let detail = (class != ErrorClass::from_status(status) || class.param().is_some())
            .then_some(ApiErrorDetail {
                param: class.param(),
                code: Some(class.code()),
            });
        Self {
            status,
            message: msg.into(),
            detail,
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::InvalidRequest, msg)
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::NotFound, msg)
    }

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::PermissionDenied, msg)
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::Unauthenticated, msg)
    }

    pub fn payload_too_large(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::PayloadTooLarge, msg)
    }

    pub fn too_many_requests(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::RateLimited, msg)
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::Internal, msg)
    }

    /// A `400` about one request parameter, rendered with OpenAI's
    /// `param` and the `invalid_value` code.
    pub fn invalid_param(param: &'static str, msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            detail: Some(ApiErrorDetail {
                param: Some(param),
                code: Some("invalid_value"),
            }),
        }
    }

    /// OpenAI `unsupported_parameter`: a valid parameter this deployment
    /// cannot honour.
    pub fn unsupported_param(param: &'static str, msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            detail: Some(ApiErrorDetail {
                param: Some(param),
                code: Some("unsupported_parameter"),
            }),
        }
    }

    /// OpenAI `context_length_exceeded` on a chat request: the messages, or
    /// the messages plus the requested output, do not fit the model's
    /// context.
    pub fn context_length_exceeded(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::ContextLengthExceeded, msg).with_param("messages")
    }

    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::Unavailable, msg)
    }

    /// OpenAI `model_not_found`: the requested model is not served here.
    pub fn model_not_found(msg: impl Into<String>) -> Self {
        Self::of(ErrorClass::ModelNotFound, msg)
    }

    /// The same error about request parameter `param`.
    pub fn with_param(mut self, param: &'static str) -> Self {
        let code = self
            .detail
            .and_then(|detail| detail.code)
            .unwrap_or(self.class().code());
        self.detail = Some(ApiErrorDetail {
            param: Some(param),
            code: Some(code),
        });
        self
    }

    /// The class this error's status belongs to. Classes that share a
    /// status share their type and retryability, so the status decides
    /// both.
    pub fn class(&self) -> ErrorClass {
        ErrorClass::from_status(self.status)
    }

    /// The OpenAI error object: the HTTP error body, and the payload of an
    /// error event once a stream has started.
    pub fn openai_body(&self) -> serde_json::Value {
        let class = self.class();
        json!({
            "error": {
                "message": self.message,
                "type": class.openai_type(),
                "param": self.detail.and_then(|detail| detail.param),
                "code": self
                    .detail
                    .and_then(|detail| detail.code)
                    .unwrap_or(class.code())
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(self.openai_body());
        let mut response = (self.status, body).into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            response.headers_mut().insert(
                header::RETRY_AFTER,
                HeaderValue::from_static(SERVICE_UNAVAILABLE_RETRY_AFTER_SECS),
            );
        }
        response
    }
}

impl From<izwi_core::Error> for ApiError {
    fn from(err: izwi_core::Error) -> Self {
        let class = ErrorClass::from(err.kind());
        match err {
            // OpenAI's message for an overflow is the bare explanation.
            izwi_core::Error::ContextLengthExceeded(message) => ApiError::of(class, message),
            err => ApiError::of(class, err.to_string()),
        }
    }
}

/// A storage failure a request handler can act on. Stores return `anyhow`
/// errors; this one rides inside them so handlers map it by class instead
/// of by its text.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(crate) struct StoreError {
    class: ErrorClass,
    message: String,
}

impl StoreError {
    pub(crate) fn new(class: ErrorClass, message: impl Into<String>) -> Self {
        Self {
            class,
            message: message.into(),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::InvalidRequest, message)
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::NotFound, message)
    }

    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::PayloadTooLarge, message)
    }

    /// A per-tenant or global admission limit: `429`.
    pub(crate) fn capacity(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::RateLimited, message)
    }

    /// The typed store error anywhere in `err`'s chain.
    pub(crate) fn find(err: &anyhow::Error) -> Option<&Self> {
        err.chain().find_map(|cause| cause.downcast_ref::<Self>())
    }
}

/// The API error for a store failure: a [`StoreError`] keeps its class and
/// message; anything else is a server fault described by `context`.
pub(crate) fn store_api_error(err: anyhow::Error, context: &str) -> ApiError {
    match StoreError::find(&err) {
        Some(store) => ApiError::of(store.class, store.message.clone()),
        None => ApiError::internal(format!("{context}: {err:#}")),
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::StatusCode, response::IntoResponse};
    use izwi_core::ErrorKind;
    use izwi_serving_protocol::{InvocationErrorCode, RejectionCode};
    use izwi_serving_worker::{admission_rejection_code, execution_error_code};
    use serde_json::Value;

    use super::*;

    async fn body(response: Response) -> Value {
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// The class table: a class's type and retryability follow from its
    /// status, so an error built from a bare status renders the same way,
    /// and every code is a word, never an HTTP status.
    #[test]
    fn class_table_is_consistent_per_status() {
        let mut codes = std::collections::HashSet::new();
        for class in ErrorClass::ALL {
            let general = ErrorClass::from_status(class.status());
            assert_eq!(general.status(), class.status(), "{class:?}");
            assert_eq!(general.openai_type(), class.openai_type(), "{class:?}");
            assert_eq!(general.retryable(), class.retryable(), "{class:?}");
            assert!(codes.insert(class.code()), "{class:?} reuses a code");
            assert!(class.code().parse::<u16>().is_err(), "{class:?}");
            let error = ApiError::of(class, "x");
            let rendered = error.openai_body();
            assert_eq!(rendered["error"]["code"], class.code(), "{class:?}");
            assert_eq!(rendered["error"]["type"], class.openai_type(), "{class:?}");
        }
        assert!(ErrorClass::Unavailable.retryable());
        assert!(ErrorClass::RateLimited.retryable());
        assert!(!ErrorClass::Internal.retryable());
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
            assert_eq!(ErrorClass::from(kind), local, "{kind:?} local");
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

    /// Worker rejections the gateway retries on another worker are exactly
    /// the retryable classes, and the rest are answered at once.
    #[test]
    fn retryable_rejections_are_the_unavailable_class() {
        for (code, class) in [
            (RejectionCode::Unauthenticated, ErrorClass::BadGateway),
            (RejectionCode::Unauthorized, ErrorClass::PermissionDenied),
            (
                RejectionCode::UnsupportedProtocolVersion,
                ErrorClass::BadGateway,
            ),
            (
                RejectionCode::WrongWorkerIncarnation,
                ErrorClass::Unavailable,
            ),
            (RejectionCode::UnknownDeployment, ErrorClass::Unavailable),
            (RejectionCode::WrongModelGeneration, ErrorClass::Unavailable),
            (RejectionCode::IncompatibleTask, ErrorClass::InvalidRequest),
            (RejectionCode::ModelNotReady, ErrorClass::Unavailable),
            (RejectionCode::WorkerDraining, ErrorClass::Unavailable),
            (RejectionCode::CapacityExhausted, ErrorClass::Unavailable),
            (RejectionCode::QueueWaitExceeded, ErrorClass::Unavailable),
            (RejectionCode::InvalidRequest, ErrorClass::InvalidRequest),
            (
                RejectionCode::DuplicateAttemptConflict,
                ErrorClass::Conflict,
            ),
            (RejectionCode::PolicyDenied, ErrorClass::PermissionDenied),
            (
                RejectionCode::ContextLengthExceeded,
                ErrorClass::ContextLengthExceeded,
            ),
            (RejectionCode::Unknown, ErrorClass::BadGateway),
        ] {
            assert_eq!(rejection_class(code), class, "{code:?}");
        }
        for (code, class) in [
            (
                InvocationErrorCode::InvalidInput,
                ErrorClass::InvalidRequest,
            ),
            (InvocationErrorCode::DeadlineExceeded, ErrorClass::Timeout),
            (
                InvocationErrorCode::WorkerUnavailable,
                ErrorClass::Unavailable,
            ),
            (InvocationErrorCode::ExecutionFailed, ErrorClass::Internal),
            (
                InvocationErrorCode::OutputLimitExceeded,
                ErrorClass::Internal,
            ),
            (InvocationErrorCode::Internal, ErrorClass::Internal),
            (InvocationErrorCode::Unknown, ErrorClass::Internal),
        ] {
            assert_eq!(invocation_error_class(code), class, "{code:?}");
        }
    }

    #[test]
    fn missing_dependency_maps_to_service_unavailable() {
        let api_error = ApiError::from(izwi_core::Error::MissingDependency(
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
            let api_error = ApiError::from(error);
            assert_eq!(api_error.status, status, "{message}");
            assert_eq!(api_error.message, message);
        }
        let missing = ApiError::from(izwi_core::Error::ModelNotFound("Unknown-ASR".into()));
        let body = missing.openai_body();
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }

    #[tokio::test]
    async fn service_unavailable_uses_openai_style_error_type() {
        let response = ApiError::service_unavailable("missing espeak-ng").into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .unwrap(),
            "5"
        );

        let body = body(response).await;
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(body["error"]["code"], "service_unavailable");
        assert_eq!(body["error"]["message"], "missing espeak-ng");
    }

    #[tokio::test]
    async fn payload_too_large_uses_invalid_request_error_type() {
        let response = ApiError {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: "too large".to_string(),
            detail: None,
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let body = body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "payload_too_large");
    }

    #[tokio::test]
    async fn gateway_auth_and_rate_limit_errors_use_uniform_json_types() {
        for (error, status, error_type, code) in [
            (
                ApiError::unauthorized("api key required"),
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "invalid_api_key",
            ),
            (
                ApiError::forbidden("inference denied"),
                StatusCode::FORBIDDEN,
                "permission_denied_error",
                "permission_denied",
            ),
            (
                ApiError::too_many_requests("rate exceeded"),
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "rate_limit_exceeded",
            ),
        ] {
            let response = error.into_response();
            assert_eq!(response.status(), status);
            let body = body(response).await;
            assert_eq!(body["error"]["type"], error_type);
            assert_eq!(body["error"]["code"], code);
        }
    }

    /// D8: context overflow is OpenAI's `context_length_exceeded` 400, so
    /// clients can tell it from other bad requests; its message is the bare
    /// explanation.
    #[tokio::test]
    async fn context_overflow_is_an_openai_context_length_error() {
        let error = ApiError::from(izwi_core::Error::ContextLengthExceeded(
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

    /// A typed store error keeps its class; any other store failure is a
    /// server fault, whatever its text says.
    #[test]
    fn store_errors_map_by_class_not_text() {
        let capacity = store_api_error(
            StoreError::capacity("Speech job admission capacity exhausted").into(),
            "storage",
        );
        assert_eq!(capacity.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(capacity.message, "Speech job admission capacity exhausted");
        let wrapped = store_api_error(
            anyhow::Error::from(StoreError::too_large("input too long")).context("accepting"),
            "storage",
        );
        assert_eq!(wrapped.status, StatusCode::PAYLOAD_TOO_LARGE);
        let text = store_api_error(
            anyhow::anyhow!("Speech job admission capacity exhausted"),
            "storage",
        );
        assert_eq!(text.status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
