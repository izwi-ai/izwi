use axum::{extract::Request, http::HeaderValue, middleware, Router};
use izwi_core::ServeRuntimeConfig;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{info_span, warn};

use crate::api::request_context::attach_request_context;
use crate::state::AppState;

const DESKTOP_CORS_ORIGINS: &[&str] = &[
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
];

/// Create the main API router.
pub fn create_router(state: AppState, serve_config: &ServeRuntimeConfig) -> Router {
    let trace_layer = TraceLayer::new_for_http().make_span_with(|request: &Request| {
        let request_id = request
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-");
        info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            correlation_id = %request_id
        )
    });

    let v1_routes = Router::new()
        .merge(crate::api::internal::router())
        .merge(crate::api::agent::router())
        .merge(crate::api::chat::router())
        .merge(crate::api::media::router())
        .merge(crate::api::transcription::router())
        .merge(crate::api::diarization::router())
        .merge(crate::api::speech_history::router())
        .merge(crate::api::saved_voices::router())
        .merge(crate::api::voice_realtime::router())
        .merge(crate::api::openai::router())
        .merge(crate::api::admin::router());

    let app = Router::new().nest("/v1", v1_routes).with_state(state);

    apply_runtime_contract(app, serve_config)
        .layer(trace_layer)
        .layer(middleware::from_fn(attach_request_context))
}

fn apply_runtime_contract(mut app: Router, serve_config: &ServeRuntimeConfig) -> Router {
    if serve_config.ui_enabled {
        let index_path = serve_config.ui_dir.join("index.html");
        app = app.fallback_service(
            ServeDir::new(serve_config.ui_dir.clone()).fallback(ServeFile::new(index_path)),
        );
    }

    if let Some(cors_layer) = build_cors_layer(serve_config) {
        app = app.layer(cors_layer);
    }

    app
}

fn build_cors_layer(serve_config: &ServeRuntimeConfig) -> Option<CorsLayer> {
    let layer = CorsLayer::new().allow_methods(Any).allow_headers(Any);
    if serve_config.cors_enabled
        && (serve_config.cors_origins.is_empty()
            || serve_config.cors_origins.iter().any(|origin| origin == "*"))
    {
        return Some(layer.allow_origin(Any));
    }

    let mut allowed_origins: Vec<HeaderValue> = DESKTOP_CORS_ORIGINS
        .iter()
        .map(|origin| HeaderValue::from_static(origin))
        .collect();

    if serve_config.cors_enabled {
        for origin in &serve_config.cors_origins {
            match HeaderValue::from_str(origin) {
                Ok(value) => {
                    if !allowed_origins.iter().any(|existing| existing == &value) {
                        allowed_origins.push(value);
                    }
                }
                Err(_) => {
                    warn!("Ignoring invalid CORS origin '{}'", origin);
                }
            }
        }
    }

    if allowed_origins.is_empty() {
        warn!("Desktop CORS allowlist is empty; disabling CORS");
        None
    } else {
        Some(layer.allow_origin(allowed_origins))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
        response::Response,
        routing::get,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::Service;

    #[tokio::test]
    async fn enabled_wildcard_cors_adds_allow_origin_header() {
        let app = apply_runtime_contract(
            Router::new().route("/hello", get(|| async { "ok" })),
            &ServeRuntimeConfig {
                cors_enabled: true,
                cors_origins: vec!["*".to_string()],
                ..ServeRuntimeConfig::default()
            },
        );

        let response = send_request(
            app,
            Request::builder()
                .method(Method::GET)
                .uri("/hello")
                .header(header::ORIGIN, "http://localhost:3000")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("*"))
        );
    }

    #[tokio::test]
    async fn disabled_cors_emits_no_allow_origin_header_for_browser_origin() {
        let app = apply_runtime_contract(
            Router::new().route("/hello", get(|| async { "ok" })),
            &ServeRuntimeConfig {
                cors_enabled: false,
                ..ServeRuntimeConfig::default()
            },
        );

        let response = send_request(
            app,
            Request::builder()
                .method(Method::GET)
                .uri("/hello")
                .header(header::ORIGIN, "http://localhost:3000")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert!(response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none());
    }

    #[tokio::test]
    async fn disabled_cors_still_allows_desktop_origin() {
        let app = apply_runtime_contract(
            Router::new().route("/hello", get(|| async { "ok" })),
            &ServeRuntimeConfig {
                cors_enabled: false,
                ..ServeRuntimeConfig::default()
            },
        );

        let response = send_request(
            app,
            Request::builder()
                .method(Method::GET)
                .uri("/hello")
                .header(header::ORIGIN, "tauri://localhost")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("tauri://localhost"))
        );
    }

    #[tokio::test]
    async fn enabled_custom_cors_keeps_desktop_allowlist() {
        let app = apply_runtime_contract(
            Router::new().route("/hello", get(|| async { "ok" })),
            &ServeRuntimeConfig {
                cors_enabled: true,
                cors_origins: vec!["http://localhost:3000".to_string()],
                ..ServeRuntimeConfig::default()
            },
        );

        let browser_response = send_request(
            app.clone(),
            Request::builder()
                .method(Method::GET)
                .uri("/hello")
                .header(header::ORIGIN, "http://localhost:3000")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(
            browser_response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("http://localhost:3000"))
        );

        let desktop_response = send_request(
            app,
            Request::builder()
                .method(Method::GET)
                .uri("/hello")
                .header(header::ORIGIN, "tauri://localhost")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(
            desktop_response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("tauri://localhost"))
        );
    }

    #[tokio::test]
    async fn enabled_ui_serves_index_as_fallback() {
        let ui_dir = test_ui_dir("enabled_ui_serves_index_as_fallback");
        std::fs::create_dir_all(&ui_dir).expect("ui dir should be created");
        std::fs::write(ui_dir.join("index.html"), "<html>ui</html>")
            .expect("index should be written");

        let app = apply_runtime_contract(
            Router::new().route("/hello", get(|| async { "ok" })),
            &ServeRuntimeConfig {
                ui_enabled: true,
                ui_dir: ui_dir.clone(),
                ..ServeRuntimeConfig::default()
            },
        );

        let response = send_request(
            app,
            Request::builder()
                .uri("/missing")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        assert_eq!(body, "<html>ui</html>");

        std::fs::remove_dir_all(ui_dir).expect("ui dir should be removed");
    }

    #[tokio::test]
    async fn disabled_ui_returns_not_found_for_unknown_routes() {
        let app = apply_runtime_contract(
            Router::new().route("/hello", get(|| async { "ok" })),
            &ServeRuntimeConfig {
                ui_enabled: false,
                ..ServeRuntimeConfig::default()
            },
        );

        let response = send_request(
            app,
            Request::builder()
                .uri("/missing")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    async fn send_request(mut app: Router, request: Request<Body>) -> Response {
        app.as_service::<Body>()
            .call(request)
            .await
            .expect("request should succeed")
    }

    fn test_ui_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();

        std::env::temp_dir().join(format!("izwi-router-{name}-{nanos}"))
    }
}
