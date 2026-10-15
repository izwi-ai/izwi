//! API error handling

use axum::{
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Seconds a client should wait before retrying a `503`. Capacity refusals
/// clear when another request finishes or a model goes idle, which is
/// usually within seconds.
const SERVICE_UNAVAILABLE_RETRY_AFTER_SECS: &str = "5";

/// A model load the runtime refused because device memory or the residency
/// budget is held by models that are busy right now. The load can succeed
/// once one of them goes idle, so it is a temporary capacity condition, not
/// a server fault. Every such refusal ends with this advice.
fn is_transient_capacity_refusal(message: &str) -> bool {
    message.contains("unload a model to make room")
}

/// API error type
#[derive(Debug)]
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
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            detail: None,
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
            detail: None,
        }
    }

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.into(),
            detail: None,
        }
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.into(),
            detail: None,
        }
    }

    pub fn payload_too_large(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: msg.into(),
            detail: None,
        }
    }

    pub fn too_many_requests(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg.into(),
            detail: None,
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
            detail: None,
        }
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

    /// OpenAI `context_length_exceeded`: the messages, or the messages plus
    /// the requested output, do not fit the model's context.
    pub fn context_length_exceeded(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            detail: Some(ApiErrorDetail {
                param: Some("messages"),
                code: Some("context_length_exceeded"),
            }),
        }
    }

    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
            detail: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": match self.status {
                    StatusCode::BAD_REQUEST => "invalid_request_error",
                    StatusCode::UNAUTHORIZED => "authentication_error",
                    StatusCode::FORBIDDEN => "permission_denied_error",
                    StatusCode::NOT_FOUND => "not_found_error",
                    StatusCode::PAYLOAD_TOO_LARGE => "invalid_request_error",
                    StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                    StatusCode::SERVICE_UNAVAILABLE => "service_unavailable_error",
                    _ => "server_error",
                },
                "param": self.detail.and_then(|detail| detail.param),
                "code": self
                    .detail
                    .and_then(|detail| detail.code)
                    .unwrap_or(self.status.as_str())
            }
        }));
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
        match &err {
            izwi_core::Error::InvalidInput(_) => ApiError::bad_request(err.to_string()),
            izwi_core::Error::ContextLengthExceeded(message) => {
                ApiError::context_length_exceeded(message.clone())
            }
            izwi_core::Error::ModelNotFound(_) => ApiError::not_found(err.to_string()),
            izwi_core::Error::ConfigError(_) => ApiError::bad_request(err.to_string()),
            izwi_core::Error::ModelLoadError(message) if is_transient_capacity_refusal(message) => {
                ApiError::service_unavailable(err.to_string())
            }
            izwi_core::Error::MissingDependency(_) => {
                ApiError::service_unavailable(err.to_string())
            }
            izwi_core::Error::Timeout(_) => ApiError {
                status: StatusCode::REQUEST_TIMEOUT,
                message: err.to_string(),
                detail: None,
            },
            izwi_core::Error::Overloaded(_) | izwi_core::Error::Backpressure(_) => {
                ApiError::service_unavailable(err.to_string())
            }
            izwi_core::Error::Cancelled(_) => ApiError {
                status: StatusCode::from_u16(499).unwrap_or(StatusCode::REQUEST_TIMEOUT),
                message: err.to_string(),
                detail: None,
            },
            _ => ApiError::internal(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::StatusCode, response::IntoResponse};
    use serde_json::Value;

    use super::ApiError;

    #[test]
    fn missing_dependency_maps_to_service_unavailable() {
        let api_error = ApiError::from(izwi_core::Error::MissingDependency(
            "espeak-ng not found".to_string(),
        ));

        assert_eq!(api_error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(api_error.message.contains("espeak-ng not found"));
    }

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
                izwi_core::Error::ModelLoadError(
                    "Cannot fit Parakeet-TDT-0.6B-v3 model tensors before state allocation: \
                     load_peak_bytes=1; no idle resident model is available for eviction; \
                     unload a model to make room"
                        .into(),
                ),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                izwi_core::Error::ModelLoadError(
                    "Cannot load Whisper-Large-v3-Turbo: the 2-model residency budget is full \
                     and no resident model is idle and available for eviction; unload a model \
                     to make room"
                        .into(),
                ),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                izwi_core::Error::Overloaded("all ASR slots are busy".into()),
                StatusCode::SERVICE_UNAVAILABLE,
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

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(
            body["error"]["code"],
            StatusCode::SERVICE_UNAVAILABLE.as_str()
        );
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

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(
            body["error"]["code"],
            StatusCode::PAYLOAD_TOO_LARGE.as_str()
        );
    }

    #[tokio::test]
    async fn gateway_auth_and_rate_limit_errors_use_uniform_json_types() {
        for (error, status, error_type) in [
            (
                ApiError::unauthorized("api key required"),
                StatusCode::UNAUTHORIZED,
                "authentication_error",
            ),
            (
                ApiError::forbidden("inference denied"),
                StatusCode::FORBIDDEN,
                "permission_denied_error",
            ),
            (
                ApiError::too_many_requests("rate exceeded"),
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
        ] {
            let response = error.into_response();
            assert_eq!(response.status(), status);
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["type"], error_type);
            assert_eq!(body["error"]["code"], status.as_str());
        }
    }

    /// D8: context overflow is OpenAI's `context_length_exceeded` 400 on the
    /// `messages` parameter, so clients can tell it from other bad requests.
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
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_eq!(body["error"]["param"], "messages");
    }
}
