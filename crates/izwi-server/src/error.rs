//! API error handling

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// API error type
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }

    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.into(),
        }
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.into(),
        }
    }

    pub fn payload_too_large(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: msg.into(),
        }
    }

    pub fn too_many_requests(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }

    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
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
                "param": null,
                "code": self.status.as_str()
            }
        }));
        (self.status, body).into_response()
    }
}

impl From<izwi_core::Error> for ApiError {
    fn from(err: izwi_core::Error) -> Self {
        match &err {
            izwi_core::Error::ModelNotFound(_) => ApiError::not_found(err.to_string()),
            izwi_core::Error::ConfigError(_) => ApiError::bad_request(err.to_string()),
            izwi_core::Error::MissingDependency(_) => {
                ApiError::service_unavailable(err.to_string())
            }
            izwi_core::Error::Timeout(_) => ApiError {
                status: StatusCode::REQUEST_TIMEOUT,
                message: err.to_string(),
            },
            izwi_core::Error::Overloaded(_) | izwi_core::Error::Backpressure(_) => {
                ApiError::service_unavailable(err.to_string())
            }
            izwi_core::Error::Cancelled(_) => ApiError {
                status: StatusCode::from_u16(499).unwrap_or(StatusCode::REQUEST_TIMEOUT),
                message: err.to_string(),
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

    #[tokio::test]
    async fn service_unavailable_uses_openai_style_error_type() {
        let response = ApiError::service_unavailable("missing espeak-ng").into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

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
}
