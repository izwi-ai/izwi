//! API errors: the error type every route returns, the class table its
//! HTTP status and OpenAI `type`/`code` come from, and the request scope that
//! decides how a server fault is shown. izwi-server maps engine errors onto
//! it; the gateway maps worker rejections and terminal errors.

use std::borrow::Cow;

use axum::{
    extract::{rejection::JsonRejection, FromRequest, Request},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use izwi_serving_protocol::{InvocationErrorCode, RejectionCode};
use serde_json::json;

/// Seconds a client should wait before retrying a `503`. Capacity refusals
/// clear when another request finishes or a model goes idle, which is
/// usually within seconds.
const SERVICE_UNAVAILABLE_RETRY_AFTER_SECS: &str = "5";

/// The message a `5xx` server fault shows the client. Its own message can
/// name paths, devices and internal state, so it is logged instead.
pub const SERVER_FAULT_MESSAGE: &str = "The server had an error while processing your request";

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

/// The class of a worker's admission rejection. Rejections a different
/// worker might accept (busy, not ready, draining, routed with stale
/// identity) are retryable [`ErrorClass::Unavailable`], so the gateway
/// tries another worker before answering `503`.
pub fn rejection_class(code: RejectionCode) -> ErrorClass {
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
pub fn invocation_error_class(code: InvocationErrorCode) -> ErrorClass {
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

tokio::task_local! {
    static REQUEST_ID: String;
    static SHOW_SERVER_FAULTS: bool;
}

/// Run `future` as the request `request_id`, so a server fault rendered
/// inside it can name the request in its redacted message.
pub async fn with_request_id<F: std::future::Future>(request_id: String, future: F) -> F::Output {
    REQUEST_ID.scope(request_id, future).await
}

/// Run `future` with server-fault messages shown in full (`show`) or
/// redacted. Only a loopback-bound local server shows them: its client is
/// this machine's user, who also owns the logs. Outside any scope (the
/// gateway, a network bind) they are redacted.
pub async fn showing_server_faults<F: std::future::Future>(show: bool, future: F) -> F::Output {
    SHOW_SERVER_FAULTS.scope(show, future).await
}

fn current_request_id() -> Option<String> {
    REQUEST_ID.try_with(Clone::clone).ok()
}

/// How the current request renders errors, captured where the request's
/// scope is set so a body polled later (an SSE stream) renders its error
/// events the way the request's own errors are rendered.
#[derive(Debug, Clone)]
pub struct ErrorScope {
    request_id: Option<String>,
    show_server_faults: bool,
}

impl ErrorScope {
    pub fn current() -> Self {
        Self {
            request_id: current_request_id(),
            show_server_faults: SHOW_SERVER_FAULTS.try_with(|show| *show).unwrap_or(false),
        }
    }

    fn render<T>(&self, render: impl FnOnce() -> T) -> T {
        let render = || SHOW_SERVER_FAULTS.sync_scope(self.show_server_faults, render);
        match &self.request_id {
            Some(request_id) => REQUEST_ID.sync_scope(request_id.clone(), render),
            None => render(),
        }
    }

    /// [`ApiError::openai_body`] as the request renders it.
    pub fn openai_body(&self, error: &ApiError) -> serde_json::Value {
        self.render(|| error.openai_body())
    }

    /// The message the request's client may see for `error`.
    pub fn client_message(&self, error: &ApiError) -> String {
        self.render(|| error.client_message().into_owned())
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

    /// A server fault: any `5xx` but `503`, whose capacity and readiness
    /// messages are written for clients.
    fn server_fault(&self) -> bool {
        self.status.is_server_error() && self.status != StatusCode::SERVICE_UNAVAILABLE
    }

    /// The message the client sees. A server fault is logged in full with
    /// the request ID and, unless the scope shows server faults, replaced
    /// by a generic message naming the request.
    fn client_message(&self) -> Cow<'_, str> {
        if !self.server_fault() {
            return Cow::Borrowed(&self.message);
        }
        let request_id = current_request_id();
        tracing::error!(
            status = self.status.as_u16(),
            request_id = request_id.as_deref().unwrap_or("-"),
            error = %self.message,
            "request failed with a server error"
        );
        if SHOW_SERVER_FAULTS.try_with(|show| *show).unwrap_or(false) {
            return Cow::Borrowed(&self.message);
        }
        Cow::Owned(match request_id {
            Some(request_id) => {
                format!(
                    "{SERVER_FAULT_MESSAGE}. Include request ID {request_id} when reporting it."
                )
            }
            None => format!("{SERVER_FAULT_MESSAGE}."),
        })
    }

    /// The OpenAI error object: the HTTP error body, and the payload of an
    /// error event once a stream has started.
    pub fn openai_body(&self) -> serde_json::Value {
        let class = self.class();
        json!({
            "error": {
                "message": self.client_message(),
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

/// A body that is not the JSON a route expects is an OpenAI-shaped `400`
/// (or `413`/`415`), never axum's plain-text rejection.
impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        let class = match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => ErrorClass::PayloadTooLarge,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => ErrorClass::UnsupportedMediaType,
            _ => ErrorClass::InvalidRequest,
        };
        let message = match &rejection {
            JsonRejection::JsonSyntaxError(_) => {
                format!(
                    "The request body is not valid JSON: {}",
                    rejection.body_text()
                )
            }
            _ => rejection.body_text(),
        };
        ApiError::of(class, message)
    }
}

/// `axum::Json` whose rejection is an [`ApiError`].
pub struct ApiJson<T>(pub T);

impl<T, S> FromRequest<S> for ApiJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state).await?;
        Ok(Self(value))
    }
}

/// A storage failure a request handler can act on. Stores return `anyhow`
/// errors; this one rides inside them so handlers map it by class instead
/// of by its text.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct StoreError {
    class: ErrorClass,
    message: String,
}

impl StoreError {
    pub fn new(class: ErrorClass, message: impl Into<String>) -> Self {
        Self {
            class,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::InvalidRequest, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::NotFound, message)
    }

    pub fn too_large(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::PayloadTooLarge, message)
    }

    /// A per-tenant or global admission limit: `429`.
    pub fn capacity(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::RateLimited, message)
    }

    /// The typed store error anywhere in `err`'s chain.
    pub fn find(err: &anyhow::Error) -> Option<&Self> {
        err.chain().find_map(|cause| cause.downcast_ref::<Self>())
    }
}

/// The API error for a store failure: a [`StoreError`] keeps its class and
/// message; anything else is a server fault described by `context`.
pub fn store_api_error(err: anyhow::Error, context: &str) -> ApiError {
    match StoreError::find(&err) {
        Some(store) => ApiError::of(store.class, store.message.clone()),
        None => ApiError::internal(format!("{context}: {err:#}")),
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::StatusCode, response::IntoResponse};
    use izwi_serving_protocol::{InvocationErrorCode, RejectionCode};
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

    /// A server fault's own text (paths, devices, internal state) never
    /// reaches a client outside a loopback-bound local server: the body names
    /// the request instead. `503` messages are written for clients and stay.
    #[tokio::test]
    async fn server_faults_are_redacted_with_the_request_id() {
        let internal = "Failed to read /var/lib/izwi/models/x.safetensors";
        let redacted = with_request_id("req-42".into(), async {
            ApiError::internal(internal).into_response()
        })
        .await;
        assert_eq!(redacted.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let message = body(redacted).await["error"]["message"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!message.contains("/var/lib"), "{message}");
        assert!(message.contains("req-42"), "{message}");

        let bad_gateway = ApiError::of(ErrorClass::BadGateway, "worker 10.0.0.7 sent junk");
        let message = bad_gateway.openai_body()["error"]["message"].to_string();
        assert!(!message.contains("10.0.0.7"), "{message}");

        let shown =
            showing_server_faults(true, async { ApiError::internal(internal).into_response() })
                .await;
        assert_eq!(body(shown).await["error"]["message"], internal);

        let unavailable = ApiError::service_unavailable("all ASR slots are busy");
        assert_eq!(
            unavailable.openai_body()["error"]["message"],
            "all ASR slots are busy"
        );
    }

    /// A stream renders its error events later, outside the request's
    /// scope; the scope captured when the stream starts still applies.
    #[tokio::test]
    async fn captured_scope_renders_stream_errors_like_the_request() {
        let error = ApiError::internal("kernel failed on /dev/gpu0");
        let shown = showing_server_faults(true, async { ErrorScope::current() }).await;
        let named = with_request_id("req-7".into(), async { ErrorScope::current() }).await;
        assert_eq!(shown.client_message(&error), "kernel failed on /dev/gpu0");
        let message = named.client_message(&error);
        assert!(
            message.contains("req-7") && !message.contains("/dev"),
            "{message}"
        );
        assert_eq!(
            named.openai_body(&error)["error"]["message"],
            message.as_str()
        );
    }

    /// A body that is not valid JSON is an OpenAI error object, not axum's
    /// plain-text rejection.
    #[tokio::test]
    async fn malformed_json_bodies_are_openai_errors() {
        #[derive(serde::Deserialize)]
        struct Body {
            #[allow(dead_code)]
            model: String,
        }
        let request = |content_type: &str, body: &'static str| {
            axum::http::Request::builder()
                .method("POST")
                .header(header::CONTENT_TYPE, content_type)
                .body(axum::body::Body::from(body))
                .unwrap()
        };
        for (request, status, code) in [
            (
                request("application/json", "{\"model\": "),
                StatusCode::BAD_REQUEST,
                "invalid_value",
            ),
            (
                request("application/json", "{\"model\": 7}"),
                StatusCode::BAD_REQUEST,
                "invalid_value",
            ),
            (
                request("text/plain", "{\"model\": \"m\"}"),
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
            ),
        ] {
            let Err(error) = ApiJson::<Body>::from_request(request, &()).await else {
                panic!("the body must be rejected");
            };
            let response = error.into_response();
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
            let body = body(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["code"], code);
        }
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
