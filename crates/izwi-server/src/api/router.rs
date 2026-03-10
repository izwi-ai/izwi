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
    use crate::state::AppState;
    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
        response::Response,
        routing::get,
    };
    use izwi_core::RuntimeService;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
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

    #[tokio::test]
    async fn canonical_history_routes_still_resolve() {
        let (app, temp_dir) = test_api_app("canonical_history_routes_still_resolve");

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/transcriptions",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/transcriptions/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/transcriptions/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/diarizations",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::PATCH,
            "/v1/diarizations/missing",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/diarizations/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/diarizations/missing/reruns",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/text-to-speech-generations",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/text-to-speech-generations/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-design-generations",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-design-generations/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-clone-generations",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app,
            Method::GET,
            "/v1/voice-clone-generations/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        drop(temp_dir);
    }

    #[tokio::test]
    async fn legacy_history_routes_return_not_found() {
        let (app, temp_dir) = test_api_app("legacy_history_routes_return_not_found");

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/transcription/records",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/transcription/records/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/transcription/records/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/diarization/records",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::PATCH,
            "/v1/diarization/records/missing",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/diarization/records/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/diarization/records/missing/rerun",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/text-to-speech/records",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/text-to-speech/records/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-design/records",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-design/records/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-cloning/records",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app,
            Method::GET,
            "/v1/voice-cloning/records/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        drop(temp_dir);
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

    async fn assert_route_status(
        app: Router,
        method: Method,
        path: &str,
        body: Option<&str>,
        expected_status: StatusCode,
    ) {
        let response = send_request(app, build_request(method, path, body)).await;
        assert_eq!(
            response.status(),
            expected_status,
            "unexpected status for {path}"
        );
    }

    fn build_request(method: Method, path: &str, body: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder().method(method).uri(path);
        let payload = body.unwrap_or_default();

        if body.is_some() {
            builder = builder.header(header::CONTENT_TYPE, "application/json");
        }

        builder
            .body(if body.is_some() {
                Body::from(payload.to_owned())
            } else {
                Body::empty()
            })
            .expect("request should build")
    }

    fn test_api_app(name: &str) -> (Router, TempDirGuard) {
        let temp_dir = test_ui_dir(name);
        let models_dir = temp_dir.join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir should be created");

        let db_path = temp_dir.join("izwi.sqlite3");
        let media_dir = temp_dir.join("media");

        let _guard = env_lock();
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled: false,
            ui_dir: temp_dir.join("ui"),
            models_dir,
            ..ServeRuntimeConfig::default()
        };
        let runtime =
            with_suppressed_panic_hook(|| RuntimeService::new(serve_config.engine_config()))
                .expect("runtime should initialize");
        let state = AppState::new(runtime, &serve_config).expect("app state should initialize");

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");

        (create_router(state, &serve_config), TempDirGuard(temp_dir))
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("environment lock poisoned")
    }

    fn with_suppressed_panic_hook<T>(f: impl FnOnce() -> T) -> T {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = f();
        std::panic::set_hook(default_hook);
        result
    }

    struct TempDirGuard(PathBuf);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
