use axum::{
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware,
    response::Response,
    routing::get,
    Router,
};
use izwi_core::ServeRuntimeConfig;
use std::time::Duration;
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{field, info, info_span, warn, Span};

use crate::api::request_context::attach_request_context;
use crate::logging::{SERVICE_NAME, SERVICE_VERSION};
use crate::state::AppState;

const DESKTOP_CORS_ORIGINS: &[&str] = &[
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
];

async fn api_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

/// Create the main API router.
pub fn create_router(state: AppState, serve_config: &ServeRuntimeConfig) -> Router {
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &Request| {
            let request_id = request
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("-");
            info_span!(
                "http_request",
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                method = %request.method(),
                path = %request.uri().path(),
                uri = %request.uri(),
                correlation_id = %request_id,
                status = field::Empty,
                latency_ms = field::Empty,
                error = field::Empty
            )
        })
        .on_response(|response: &Response, latency: Duration, span: &Span| {
            let status = response.status().as_u16();
            let latency_ms = latency.as_millis() as u64;
            span.record("status", status);
            span.record("latency_ms", latency_ms);
            info!(
                parent: span,
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                status,
                latency_ms,
                "HTTP request completed"
            );
        })
        .on_failure(
            |failure: ServerErrorsFailureClass, latency: Duration, span: &Span| {
                let latency_ms = latency.as_millis() as u64;
                span.record("latency_ms", latency_ms);
                span.record("error", field::display(&failure));
                warn!(
                    parent: span,
                    service = SERVICE_NAME,
                    version = SERVICE_VERSION,
                    error = %failure,
                    latency_ms,
                    "HTTP request failed"
                );
            },
        );

    let v1_routes = Router::new()
        .merge(crate::api::internal::router())
        .merge(crate::api::onboarding::router())
        .merge(crate::api::preferences::router())
        .merge(crate::api::agent::router())
        .merge(crate::api::chat::router())
        .merge(crate::api::media::router())
        .merge(crate::api::transcription::router())
        .merge(crate::api::diarization::router())
        .merge(crate::api::speech_history::router())
        .merge(crate::api::studio::router())
        .merge(crate::api::saved_voices::router())
        .merge(crate::api::voice::router())
        .merge(crate::api::voice_realtime::router())
        .merge(crate::api::openai::router())
        .merge(crate::api::admin::router())
        .fallback(api_not_found);

    let app = Router::new()
        .route("/livez", get(crate::api::internal::probes::live_check))
        .route("/readyz", get(crate::api::internal::probes::ready_check))
        .route("/openapi.json", get(crate::api::openapi::openapi_json))
        .merge(crate::api::docs::router())
        .nest("/v1", v1_routes)
        // Compatibility mount for tooling that queries /internal/* directly.
        .nest("/internal", crate::api::internal::router())
        .with_state(state);

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
        let (app, temp_dir) = test_api_app("canonical_history_routes_still_resolve", true);

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
            Method::POST,
            "/v1/diarizations/missing/summary/regenerate",
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
    async fn unified_transcription_job_read_routes_support_both_job_kinds() {
        let (app, temp_dir) = test_api_app(
            "unified_transcription_job_read_routes_support_both_job_kinds",
            true,
        );

        let transcription_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/transcriptions",
                Some("{\"audio_base64\":\"AQ==\"}"),
            ),
        )
        .await;
        assert_eq!(transcription_create.status(), StatusCode::ACCEPTED);
        let transcription_body = read_json(transcription_create).await;
        let transcription_id = transcription_body
            .get("id")
            .and_then(|value| value.as_str())
            .expect("transcription id should exist")
            .to_string();

        let diarization_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/diarizations",
                Some("{\"audio_base64\":\"AQ==\"}"),
            ),
        )
        .await;
        assert_eq!(diarization_create.status(), StatusCode::ACCEPTED);
        let diarization_body = read_json(diarization_create).await;
        let diarization_id = diarization_body
            .get("id")
            .and_then(|value| value.as_str())
            .expect("diarization id should exist")
            .to_string();

        let unified_list = send_request(
            app.clone(),
            build_request(Method::GET, "/v1/transcriptions/jobs?job_kind=all", None),
        )
        .await;
        assert_eq!(unified_list.status(), StatusCode::OK);
        let list_payload = read_json(unified_list).await;
        let records = list_payload
            .get("records")
            .and_then(|value| value.as_array())
            .expect("records array should exist");
        let kinds = records
            .iter()
            .filter_map(|record| record.get("kind").and_then(|value| value.as_str()))
            .collect::<Vec<_>>();
        assert!(
            kinds.contains(&"transcription"),
            "unified list should include transcription records"
        );
        assert!(
            kinds.contains(&"diarization"),
            "unified list should include diarization records"
        );

        let unified_list_with_limit = send_request(
            app.clone(),
            build_request(Method::GET, "/v1/transcriptions/jobs?limit=25", None),
        )
        .await;
        assert_eq!(unified_list_with_limit.status(), StatusCode::OK);
        let list_with_limit_payload = read_json(unified_list_with_limit).await;
        assert_eq!(
            list_with_limit_payload
                .get("pagination")
                .and_then(|value| value.get("limit"))
                .and_then(|value| value.as_u64()),
            Some(25)
        );

        let unified_transcription_record = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/transcriptions/jobs/{}?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(unified_transcription_record.status(), StatusCode::OK);
        let transcription_payload = read_json(unified_transcription_record).await;
        assert_eq!(
            transcription_payload
                .get("kind")
                .and_then(|value| value.as_str()),
            Some("transcription")
        );

        let unified_diarization_record = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/transcriptions/jobs/{}?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(unified_diarization_record.status(), StatusCode::OK);
        let diarization_payload = read_json(unified_diarization_record).await;
        assert_eq!(
            diarization_payload
                .get("kind")
                .and_then(|value| value.as_str()),
            Some("diarization")
        );

        let transcription_audio = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/transcriptions/jobs/{}/audio?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(transcription_audio.status(), StatusCode::OK);

        let diarization_audio = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/transcriptions/jobs/{}/audio?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(diarization_audio.status(), StatusCode::OK);

        // Compatibility check: canonical diarization route remains available.
        let legacy_diarization_get = send_request(
            app,
            build_request(
                Method::GET,
                format!("/v1/diarizations/{}", diarization_id).as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(legacy_diarization_get.status(), StatusCode::OK);

        drop(temp_dir);
    }

    #[tokio::test]
    async fn unified_transcription_job_mutation_routes_support_both_job_kinds() {
        let (app, temp_dir) = test_api_app(
            "unified_transcription_job_mutation_routes_support_both_job_kinds",
            true,
        );

        let transcription_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/transcriptions/jobs?job_kind=transcription",
                Some("{\"audio_base64\":\"AQ==\"}"),
            ),
        )
        .await;
        assert_eq!(transcription_create.status(), StatusCode::ACCEPTED);
        let transcription_id = read_json(transcription_create)
            .await
            .get("id")
            .and_then(|value| value.as_str())
            .expect("transcription id should exist")
            .to_string();

        let diarization_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/transcriptions/jobs?job_kind=diarization",
                Some("{\"audio_base64\":\"AQ==\"}"),
            ),
        )
        .await;
        assert_eq!(diarization_create.status(), StatusCode::ACCEPTED);
        let diarization_id = read_json(diarization_create)
            .await
            .get("id")
            .and_then(|value| value.as_str())
            .expect("diarization id should exist")
            .to_string();

        let update_diarization = send_request(
            app.clone(),
            build_request(
                Method::PATCH,
                format!(
                    "/v1/transcriptions/jobs/{}?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                Some("{\"speaker_name_overrides\":{}}"),
            ),
        )
        .await;
        assert_eq!(update_diarization.status(), StatusCode::OK);

        let rerun_diarization = send_request(
            app.clone(),
            build_request(
                Method::POST,
                format!(
                    "/v1/transcriptions/jobs/{}/reruns?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(rerun_diarization.status(), StatusCode::ACCEPTED);

        let cancel_diarization = send_request(
            app.clone(),
            build_request(
                Method::POST,
                format!(
                    "/v1/transcriptions/jobs/{}/cancel?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(cancel_diarization.status(), StatusCode::OK);

        let regen_transcription_summary = send_request(
            app.clone(),
            build_request(
                Method::POST,
                format!(
                    "/v1/transcriptions/jobs/{}/summary/regenerate?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(regen_transcription_summary.status(), StatusCode::OK);

        let regen_diarization_summary = send_request(
            app.clone(),
            build_request(
                Method::POST,
                format!(
                    "/v1/transcriptions/jobs/{}/summary/regenerate?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(regen_diarization_summary.status(), StatusCode::OK);

        let delete_transcription = send_request(
            app.clone(),
            build_request(
                Method::DELETE,
                format!(
                    "/v1/transcriptions/jobs/{}?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(delete_transcription.status(), StatusCode::OK);

        let delete_diarization = send_request(
            app.clone(),
            build_request(
                Method::DELETE,
                format!(
                    "/v1/transcriptions/jobs/{}?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(delete_diarization.status(), StatusCode::OK);

        // Compatibility check: legacy mutation route still works.
        let legacy_create = send_request(
            app,
            build_request(
                Method::POST,
                "/v1/diarizations",
                Some("{\"audio_base64\":\"AQ==\"}"),
            ),
        )
        .await;
        assert_eq!(legacy_create.status(), StatusCode::ACCEPTED);

        drop(temp_dir);
    }

    #[tokio::test]
    async fn canonical_history_mutation_routes_still_resolve() {
        let (app, temp_dir) = test_api_app("canonical_history_mutation_routes_still_resolve", true);

        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/transcriptions",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/transcriptions/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/transcriptions/missing/summary/regenerate",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/diarizations",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::PUT,
            "/v1/diarizations/missing",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/diarizations/missing",
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
            Method::POST,
            "/v1/text-to-speech-generations",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/text-to-speech-generations/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/voice-design-generations",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/voice-design-generations/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/voice-clone-generations",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app,
            Method::DELETE,
            "/v1/voice-clone-generations/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        drop(temp_dir);
    }

    #[tokio::test]
    async fn api_namespace_does_not_fall_back_to_ui_when_enabled() {
        let (app, temp_dir) =
            test_api_app("api_namespace_does_not_fall_back_to_ui_when_enabled", true);

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/not-a-route",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/not-a-route",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::PUT,
            "/v1/not-a-route",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app,
            Method::DELETE,
            "/v1/not-a-route",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        drop(temp_dir);
    }

    #[tokio::test]
    async fn legacy_history_routes_return_not_found() {
        let (app, temp_dir) = test_api_app("legacy_history_routes_return_not_found", true);

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
            Method::POST,
            "/v1/transcription/records",
            Some("{}"),
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
            Method::DELETE,
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
            Method::POST,
            "/v1/diarization/records",
            Some("{}"),
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
            Method::PUT,
            "/v1/diarization/records/missing",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/diarization/records/missing",
            None,
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
            Method::POST,
            "/v1/text-to-speech/records",
            Some("{}"),
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
            Method::DELETE,
            "/v1/text-to-speech/records/missing",
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
            Method::POST,
            "/v1/voice-design/records",
            Some("{}"),
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
            Method::DELETE,
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
            app.clone(),
            Method::POST,
            "/v1/voice-cloning/records",
            Some("{}"),
            StatusCode::NOT_FOUND,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/voice-cloning/records/missing",
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

    #[tokio::test]
    async fn internal_metrics_route_is_available_for_bench_compat() {
        let (app, temp_dir) = test_api_app("internal_metrics_route_compat", false);

        assert_route_status(
            app.clone(),
            Method::GET,
            "/internal/metrics",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(app, Method::GET, "/v1/metrics", None, StatusCode::OK).await;

        drop(temp_dir);
    }

    #[tokio::test]
    async fn root_and_v1_probe_routes_are_available() {
        let (app, temp_dir) = test_api_app("probe_routes_available", false);

        assert_route_status(app.clone(), Method::GET, "/livez", None, StatusCode::OK).await;
        assert_route_status(app.clone(), Method::GET, "/readyz", None, StatusCode::OK).await;
        assert_route_status(app.clone(), Method::GET, "/v1/live", None, StatusCode::OK).await;
        assert_route_status(app, Method::GET, "/v1/ready", None, StatusCode::OK).await;

        drop(temp_dir);
    }

    #[tokio::test]
    async fn openapi_json_route_returns_valid_scaffold() {
        let (app, temp_dir) = test_api_app("openapi_json_route_returns_valid_scaffold", false);

        let response = send_request(app, build_request(Method::GET, "/openapi.json", None)).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/json"))
        );

        let json = read_json(response).await;
        assert!(json["openapi"].as_str().is_some());
        assert_eq!(json["info"]["title"], "Izwi API");
        assert_eq!(json["info"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(json["paths"].get("/livez").is_some());
        assert!(json["paths"].get("/readyz").is_some());

        drop(temp_dir);
    }

    #[tokio::test]
    async fn scalar_docs_route_returns_html_before_ui_fallback() {
        let (app, temp_dir) = test_api_app("scalar_docs_route_returns_html", true);

        let response = send_request(app, build_request(Method::GET, "/docs", None)).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/html")));

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let html = String::from_utf8(body.to_vec()).expect("html should be utf-8");
        assert!(html.contains("/openapi.json"));
        assert!(html.contains("/docs/scalar.js"));
        assert!(html.contains("\"disabled\":true"));
        assert!(!html.contains("<html>ui</html>"));

        drop(temp_dir);
    }

    #[tokio::test]
    async fn scalar_js_asset_route_returns_javascript() {
        let (app, temp_dir) = test_api_app("scalar_js_asset_route_returns_javascript", false);

        let response =
            send_request(app, build_request(Method::GET, "/docs/scalar.js", None)).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/javascript"))
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        assert!(!body.is_empty());

        drop(temp_dir);
    }

    #[tokio::test]
    async fn dynamic_routes_cover_axum_upgrade_surface() {
        let (app, temp_dir) = test_api_app("dynamic_routes_cover_axum_upgrade_surface", false);

        let cases = [
            (
                Method::GET,
                "/v1/media/nested/example.png",
                None,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                Method::GET,
                "/v1/agent/sessions/missing-session",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/agent/sessions/missing-session/turns",
                Some(r#"{"input":"hello"}"#),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::GET,
                "/v1/responses/missing-response",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::DELETE,
                "/v1/responses/missing-response",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/responses/missing-response/cancel",
                Some("{}"),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::GET,
                "/v1/responses/missing-response/input_items",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::DELETE,
                "/v1/studio/projects/missing-project/pronunciations/missing-pronunciation",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/studio/projects/missing-project/snapshots/missing-snapshot/restore",
                Some("{}"),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::PATCH,
                "/v1/studio/projects/missing-project/render-jobs/missing-job",
                Some("{}"),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/studio/projects/missing-project/segments/missing-segment/split",
                Some(r#"{"before_text":"before","after_text":"after"}"#),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/studio/projects/missing-project/segments/missing-segment/merge-next",
                Some("{}"),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/studio/projects/missing-project/segments/missing-segment/render",
                Some("{}"),
                StatusCode::NOT_FOUND,
            ),
        ];

        for (method, path, body, expected_status) in cases {
            assert_route_status(app.clone(), method, path, body, expected_status).await;
        }

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice/realtime/ws",
            None,
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app,
            Method::GET,
            "/v1/transcription/realtime/ws",
            None,
            StatusCode::BAD_REQUEST,
        )
        .await;

        drop(temp_dir);
    }

    #[tokio::test]
    async fn readyz_returns_unavailable_when_lifecycle_is_draining() {
        let (state, temp_dir) = test_state("readyz_draining", false);
        state.lifecycle.mark_ready();
        state.lifecycle.mark_draining();
        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled: false,
            ..ServeRuntimeConfig::default()
        };
        let app = create_router(state, &serve_config);

        assert_route_status(
            app,
            Method::GET,
            "/readyz",
            None,
            StatusCode::SERVICE_UNAVAILABLE,
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

    async fn read_json(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        serde_json::from_slice(bytes.as_ref()).expect("response should be valid json")
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

    fn test_api_app(name: &str, ui_enabled: bool) -> (Router, TempDirGuard) {
        let (state, temp_dir) = test_state(name, ui_enabled);
        state.lifecycle.mark_ready();
        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled,
            ..ServeRuntimeConfig::default()
        };
        (create_router(state, &serve_config), temp_dir)
    }

    fn test_state(name: &str, ui_enabled: bool) -> (AppState, TempDirGuard) {
        let temp_dir = test_ui_dir(name);
        let ui_dir = temp_dir.join("ui");
        let models_dir = temp_dir.join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir should be created");
        if ui_enabled {
            std::fs::create_dir_all(&ui_dir).expect("ui dir should be created");
            std::fs::write(ui_dir.join("index.html"), "<html>ui</html>")
                .expect("index should be written");
        }

        let db_path = temp_dir.join("izwi.sqlite3");
        let media_dir = temp_dir.join("media");

        let _guard = env_lock();
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled,
            ui_dir,
            models_dir,
            ..ServeRuntimeConfig::default()
        };
        let runtime =
            with_suppressed_panic_hook(|| RuntimeService::new(serve_config.engine_config()))
                .expect("runtime should initialize");
        let state = AppState::new(runtime, &serve_config).expect("app state should initialize");

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");

        (state, TempDirGuard(temp_dir))
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
