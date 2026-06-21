use axum::{
    Router,
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware,
    response::Response,
    routing::get,
};
use izwi_core::ServeRuntimeConfig;
use std::time::Duration;
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{Span, field, info, info_span, warn};

use crate::api::request_context::attach_enterprise_request_context;
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
    let middleware_state = state.clone();
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
        .layer(middleware::from_fn_with_state(
            middleware_state,
            attach_enterprise_request_context,
        ))
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
    use crate::test_support::env_lock;
    use async_trait::async_trait;
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode, header},
        response::Response,
        routing::get,
    };
    use base64::Engine;
    use izwi_core::{
        RuntimeService,
        audio::{AudioEncoder, AudioFormat},
    };
    use izwi_hooks::{
        AuthorizationDecision, AuthorizationRequest, EnterpriseHooks, HookResult, PolicyEngine,
    };
    use std::path::PathBuf;
    use std::sync::Arc;
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

        assert!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none()
        );
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
            "/v1/text-to-speech",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/text-to-speech/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-designs",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-designs/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-clones",
            None,
            StatusCode::OK,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::GET,
            "/v1/voice-clones/missing/audio",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        for removed_path in [
            "/v1/text-to-speech-generations",
            "/v1/voice-design-generations",
            "/v1/voice-clone-generations",
        ] {
            assert_route_status(
                app.clone(),
                Method::GET,
                removed_path,
                None,
                StatusCode::NOT_FOUND,
            )
            .await;
        }

        drop(temp_dir);
    }

    #[tokio::test]
    async fn unified_transcription_job_read_routes_support_both_job_kinds() {
        let (app, temp_dir) = test_api_app(
            "unified_transcription_job_read_routes_support_both_job_kinds",
            true,
        );
        let audio_body = tiny_audio_json_body();

        let transcription_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/speech-to-text/jobs?job_kind=transcription",
                Some(audio_body.as_str()),
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

        let saa_body = tiny_audio_json_body_with_fields(
            r#""model_id":"Granite-Speech-4.1-2B-Plus","min_speakers":2"#,
        );
        let saa_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
                Some(saa_body.as_str()),
            ),
        )
        .await;
        assert_eq!(saa_create.status(), StatusCode::ACCEPTED);
        let saa_body = read_json(saa_create).await;
        assert_eq!(
            saa_body
                .get("transcription_mode")
                .and_then(|value| value.as_str()),
            Some("speaker_attributed_asr")
        );
        assert_eq!(
            saa_body.get("model_id").and_then(|value| value.as_str()),
            Some("Granite-Speech-4.1-2B-Plus")
        );
        let saa_id = saa_body
            .get("id")
            .and_then(|value| value.as_str())
            .expect("SAA id should exist")
            .to_string();

        let diarization_create = send_request(
            app.clone(),
            build_request(Method::POST, "/v1/diarizations", Some(audio_body.as_str())),
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
            build_request(Method::GET, "/v1/speech-to-text/jobs?job_kind=all", None),
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
        assert!(
            kinds.contains(&"speaker_attributed_asr"),
            "unified list should include SAA records"
        );

        let transcription_list = send_request(
            app.clone(),
            build_request(
                Method::GET,
                "/v1/speech-to-text/jobs?job_kind=transcription",
                None,
            ),
        )
        .await;
        assert_eq!(transcription_list.status(), StatusCode::OK);
        let transcription_list_payload = read_json(transcription_list).await;
        let transcription_records = transcription_list_payload
            .get("records")
            .and_then(|value| value.as_array())
            .expect("transcription records array should exist");
        assert!(transcription_records.iter().all(|record| {
            record.get("kind").and_then(|value| value.as_str()) == Some("transcription")
        }));

        let saa_list = send_request(
            app.clone(),
            build_request(
                Method::GET,
                "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
                None,
            ),
        )
        .await;
        assert_eq!(saa_list.status(), StatusCode::OK);
        let saa_list_payload = read_json(saa_list).await;
        let saa_records = saa_list_payload
            .get("records")
            .and_then(|value| value.as_array())
            .expect("SAA records array should exist");
        assert!(saa_records.iter().all(|record| {
            record.get("kind").and_then(|value| value.as_str()) == Some("speaker_attributed_asr")
        }));

        let unified_list_with_limit = send_request(
            app.clone(),
            build_request(Method::GET, "/v1/speech-to-text/jobs?limit=25", None),
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
                    "/v1/speech-to-text/jobs/{}?job_kind=transcription",
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

        let unified_saa_record = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/speech-to-text/jobs/{}?job_kind=speaker_attributed_asr",
                    saa_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(unified_saa_record.status(), StatusCode::OK);
        let saa_payload = read_json(unified_saa_record).await;
        assert_eq!(
            saa_payload.get("kind").and_then(|value| value.as_str()),
            Some("speaker_attributed_asr")
        );

        let saa_as_transcription = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!("/v1/speech-to-text/jobs/{}?job_kind=transcription", saa_id).as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(saa_as_transcription.status(), StatusCode::NOT_FOUND);

        let unified_diarization_record = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/speech-to-text/jobs/{}?job_kind=diarization",
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
                    "/v1/speech-to-text/jobs/{}/audio?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(transcription_audio.status(), StatusCode::OK);

        let saa_audio = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/speech-to-text/jobs/{}/audio?job_kind=speaker_attributed_asr",
                    saa_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(saa_audio.status(), StatusCode::OK);

        let diarization_audio = send_request(
            app.clone(),
            build_request(
                Method::GET,
                format!(
                    "/v1/speech-to-text/jobs/{}/audio?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(diarization_audio.status(), StatusCode::OK);

        for removed_path in [
            "/v1/transcriptions/jobs?job_kind=all",
            "/v1/transcriptions/jobs/missing/audio?job_kind=transcription",
        ] {
            assert_route_status(
                app.clone(),
                Method::GET,
                removed_path,
                None,
                StatusCode::NOT_FOUND,
            )
            .await;
        }

        let canonical_diarization_get = send_request(
            app,
            build_request(
                Method::GET,
                format!("/v1/diarizations/{}", diarization_id).as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(canonical_diarization_get.status(), StatusCode::OK);

        drop(temp_dir);
    }

    #[tokio::test]
    async fn unified_transcription_job_mutation_routes_support_both_job_kinds() {
        let (app, temp_dir) = test_api_app(
            "unified_transcription_job_mutation_routes_support_both_job_kinds",
            true,
        );
        let audio_body = tiny_audio_json_body();

        let transcription_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/speech-to-text/jobs?job_kind=transcription",
                Some(audio_body.as_str()),
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

        let saa_body =
            tiny_audio_json_body_with_fields(r#""model_id":"Granite-Speech-4.1-2B-Plus""#);
        let saa_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/speech-to-text/jobs?job_kind=speaker_attributed_asr",
                Some(saa_body.as_str()),
            ),
        )
        .await;
        assert_eq!(saa_create.status(), StatusCode::ACCEPTED);
        let saa_id = read_json(saa_create)
            .await
            .get("id")
            .and_then(|value| value.as_str())
            .expect("SAA id should exist")
            .to_string();

        let diarization_create = send_request(
            app.clone(),
            build_request(
                Method::POST,
                "/v1/speech-to-text/jobs?job_kind=diarization",
                Some(audio_body.as_str()),
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
                    "/v1/speech-to-text/jobs/{}?job_kind=diarization",
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
                    "/v1/speech-to-text/jobs/{}/reruns?job_kind=diarization",
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
                    "/v1/speech-to-text/jobs/{}/cancel?job_kind=diarization",
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
                    "/v1/speech-to-text/jobs/{}/summary/regenerate?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(regen_transcription_summary.status(), StatusCode::OK);

        let regen_saa_summary = send_request(
            app.clone(),
            build_request(
                Method::POST,
                format!(
                    "/v1/speech-to-text/jobs/{}/summary/regenerate?job_kind=speaker_attributed_asr",
                    saa_id
                )
                .as_str(),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(regen_saa_summary.status(), StatusCode::OK);

        let regen_diarization_summary = send_request(
            app.clone(),
            build_request(
                Method::POST,
                format!(
                    "/v1/speech-to-text/jobs/{}/summary/regenerate?job_kind=diarization",
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
                    "/v1/speech-to-text/jobs/{}?job_kind=transcription",
                    transcription_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(delete_transcription.status(), StatusCode::OK);

        let delete_saa = send_request(
            app.clone(),
            build_request(
                Method::DELETE,
                format!(
                    "/v1/speech-to-text/jobs/{}?job_kind=speaker_attributed_asr",
                    saa_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(delete_saa.status(), StatusCode::OK);

        let delete_diarization = send_request(
            app.clone(),
            build_request(
                Method::DELETE,
                format!(
                    "/v1/speech-to-text/jobs/{}?job_kind=diarization",
                    diarization_id
                )
                .as_str(),
                None,
            ),
        )
        .await;
        assert_eq!(delete_diarization.status(), StatusCode::OK);

        for removed_path in [
            "/v1/transcriptions/jobs/missing/reruns?job_kind=diarization",
            "/v1/transcriptions/jobs/missing/cancel?job_kind=diarization",
            "/v1/transcriptions/jobs/missing/summary/regenerate?job_kind=transcription",
        ] {
            assert_route_status(
                app.clone(),
                Method::POST,
                removed_path,
                Some("{}"),
                StatusCode::NOT_FOUND,
            )
            .await;
        }

        let canonical_diarization_create = send_request(
            app,
            build_request(
                Method::POST,
                "/v1/diarizations",
                Some(tiny_audio_json_body().as_str()),
            ),
        )
        .await;
        assert_eq!(canonical_diarization_create.status(), StatusCode::ACCEPTED);

        drop(temp_dir);
    }

    #[tokio::test]
    async fn granite_speech_is_rejected_for_diarization_jobs() {
        let (app, temp_dir) = test_api_app("granite_speech_is_rejected_for_diarization_jobs", true);
        let body = tiny_audio_json_body_with_fields(r#""model_id":"Granite-Speech-4.1-2B-Plus""#);

        let response = send_request(
            app,
            build_request(
                Method::POST,
                "/v1/speech-to-text/jobs?job_kind=diarization",
                Some(body.as_str()),
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload = read_json(response).await;
        let message = payload["error"]["message"]
            .as_str()
            .expect("error message should exist");
        assert!(message.contains("speaker-attributed ASR"));
        assert!(message.contains("job_kind=speaker_attributed_asr"));

        drop(temp_dir);
    }

    #[tokio::test]
    async fn canonical_history_mutation_routes_still_resolve() {
        let (app, temp_dir) = test_api_app("canonical_history_mutation_routes_still_resolve", true);

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
            "/v1/text-to-speech",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/text-to-speech/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/voice-designs",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/voice-designs/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        assert_route_status(
            app.clone(),
            Method::POST,
            "/v1/voice-clones",
            Some("{}"),
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app.clone(),
            Method::DELETE,
            "/v1/voice-clones/missing",
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

        for removed_path in [
            "/v1/text-to-speech-generations",
            "/v1/voice-design-generations",
            "/v1/voice-clone-generations",
        ] {
            assert_route_status(
                app.clone(),
                Method::POST,
                removed_path,
                Some("{}"),
                StatusCode::NOT_FOUND,
            )
            .await;
        }

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

        for (method, path, body) in [
            (Method::POST, "/v1/audio/diarize", Some("{}")),
            (Method::POST, "/v1/audio/diarizations", Some("{}")),
            (Method::GET, "/v1/transcriptions", None),
            (Method::POST, "/v1/transcriptions", Some("{}")),
            (Method::GET, "/v1/transcriptions/missing", None),
            (Method::DELETE, "/v1/transcriptions/missing", None),
            (Method::GET, "/v1/transcriptions/missing/audio", None),
            (
                Method::POST,
                "/v1/transcriptions/missing/summary/regenerate",
                Some("{}"),
            ),
        ] {
            assert_route_status(app.clone(), method, path, body, StatusCode::NOT_FOUND).await;
        }

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
    async fn admin_models_expose_voice_app_capabilities() {
        let (app, temp_dir) = test_api_app("admin_models_expose_voice_app_capabilities", false);

        let response =
            send_request(app, build_request(Method::GET, "/v1/admin/models", None)).await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = read_json(response).await;
        let models = body["models"]
            .as_array()
            .expect("models should be an array");
        let find_model = |variant: &str| {
            models
                .iter()
                .find(|model| model["variant"].as_str() == Some(variant))
                .unwrap_or_else(|| panic!("{variant} should be listed"))
        };

        let tts_model = find_model("Kokoro-82M");
        assert_eq!(
            tts_model["route_capabilities"]["openai_audio_speech"].as_bool(),
            Some(true)
        );
        assert_eq!(
            tts_model["route_capabilities"]["voice_realtime_modular_tts"].as_bool(),
            Some(true)
        );
        assert!(
            tts_model["modalities"]
                .as_array()
                .expect("modalities should be an array")
                .iter()
                .any(|modality| modality.as_str() == Some("audio_output"))
        );

        let aligner_model = find_model("Qwen3-ForcedAligner-0.6B");
        assert_eq!(
            aligner_model["route_capabilities"]["forced_alignment"].as_bool(),
            Some(true)
        );
        assert!(
            aligner_model["modalities"]
                .as_array()
                .expect("modalities should be an array")
                .iter()
                .any(|modality| modality.as_str() == Some("timestamps"))
        );

        let diarization_model = find_model("diar_streaming_sortformer_4spk-v2.1");
        assert_eq!(
            diarization_model["route_capabilities"]["diarization_records"].as_bool(),
            Some(true)
        );
        assert_eq!(
            diarization_model["route_capabilities"]["speech_to_text_jobs"].as_bool(),
            Some(true)
        );

        let granite_model = find_model("Granite-Speech-4.1-2B-Plus");
        assert_eq!(
            granite_model["route_capabilities"]["diarization_records"].as_bool(),
            Some(false)
        );
        assert_eq!(
            granite_model["route_capabilities"]["speech_to_text_jobs"].as_bool(),
            Some(true)
        );
        let granite_modalities = granite_model["modalities"]
            .as_array()
            .expect("Granite modalities should be an array");
        assert!(
            granite_modalities
                .iter()
                .any(|modality| modality.as_str() == Some("speaker_labels"))
        );
        assert!(
            !granite_modalities
                .iter()
                .any(|modality| modality.as_str() == Some("timestamps"))
        );

        let voxtral_model = find_model("Voxtral-Mini-4B-Realtime-2602");
        assert_eq!(
            voxtral_model["route_capabilities"]["openai_audio_transcriptions"].as_bool(),
            Some(true)
        );
        assert_eq!(
            voxtral_model["route_capabilities"]["speech_to_text_jobs"].as_bool(),
            Some(true)
        );
        assert_eq!(
            voxtral_model["route_capabilities"]["speech_to_text_realtime"].as_bool(),
            Some(false)
        );
        assert_eq!(
            voxtral_model["route_capabilities"]["voice_realtime_modular_asr"].as_bool(),
            Some(false)
        );
        assert_eq!(
            voxtral_model["route_capabilities"]["voice_realtime_unified"].as_bool(),
            Some(false)
        );
        let voxtral_modalities = voxtral_model["modalities"]
            .as_array()
            .expect("Voxtral modalities should be an array");
        assert!(
            voxtral_modalities
                .iter()
                .any(|modality| modality.as_str() == Some("audio_input"))
        );
        assert!(
            voxtral_modalities
                .iter()
                .any(|modality| modality.as_str() == Some("text_output"))
        );
        assert!(
            !voxtral_modalities
                .iter()
                .any(|modality| modality.as_str() == Some("audio_output"))
        );

        drop(temp_dir);
    }

    #[tokio::test]
    async fn voice_session_rest_lifecycle_routes_work() {
        let (app, temp_dir) = test_api_app("voice_session_rest_lifecycle_routes_work", false);

        let create = send_request(
            app.clone(),
            build_request(Method::POST, "/v1/voice/sessions", Some("{}")),
        )
        .await;
        assert_eq!(create.status(), StatusCode::OK);
        let created = read_json(create).await;
        let session_id = created["session"]["id"]
            .as_str()
            .expect("created session id")
            .to_string();
        assert_eq!(created["turns"].as_array().expect("turns").len(), 0);

        let turns = send_request(
            app.clone(),
            build_request(
                Method::GET,
                &format!("/v1/voice/sessions/{session_id}/turns"),
                None,
            ),
        )
        .await;
        assert_eq!(turns.status(), StatusCode::OK);
        assert_eq!(
            read_json(turns).await.as_array().expect("turn list").len(),
            0
        );

        let patch = send_request(
            app.clone(),
            build_request(
                Method::PATCH,
                &format!("/v1/voice/sessions/{session_id}"),
                Some(r#"{"system_prompt":"Keep the exchange concise."}"#),
            ),
        )
        .await;
        assert_eq!(patch.status(), StatusCode::OK);
        let patched = read_json(patch).await;
        assert_eq!(
            patched["session"]["system_prompt"].as_str(),
            Some("Keep the exchange concise.")
        );

        let end = send_request(
            app.clone(),
            build_request(
                Method::POST,
                &format!("/v1/voice/sessions/{session_id}/end"),
                Some("{}"),
            ),
        )
        .await;
        assert_eq!(end.status(), StatusCode::OK);
        assert!(read_json(end).await["session"]["ended_at"].is_number());

        let export = send_request(
            app.clone(),
            build_request(
                Method::GET,
                &format!("/v1/voice/sessions/{session_id}/export?format=text"),
                None,
            ),
        )
        .await;
        assert_eq!(export.status(), StatusCode::OK);
        let body = axum::body::to_bytes(export.into_body(), usize::MAX)
            .await
            .expect("export body");
        assert!(
            std::str::from_utf8(&body)
                .expect("text export")
                .contains(&session_id)
        );

        let delete = send_request(
            app.clone(),
            build_request(
                Method::DELETE,
                &format!("/v1/voice/sessions/{session_id}"),
                None,
            ),
        )
        .await;
        assert_eq!(delete.status(), StatusCode::OK);

        assert_route_status(
            app,
            Method::GET,
            &format!("/v1/voice/sessions/{session_id}"),
            None,
            StatusCode::NOT_FOUND,
        )
        .await;

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
        assert!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/html"))
        );

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

        let response = send_request(app, build_request(Method::GET, "/docs/scalar.js", None)).await;

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
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/media",
                Some("{}"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Method::DELETE,
                "/v1/media/nested/example.png",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::GET,
                "/v1/voice/sessions/missing-session/turns",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::PATCH,
                "/v1/voice/sessions/missing-session",
                Some(r#"{"system_prompt":"updated"}"#),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::POST,
                "/v1/voice/sessions/missing-session/end",
                Some("{}"),
                StatusCode::NOT_FOUND,
            ),
            (
                Method::GET,
                "/v1/voice/sessions/missing-session/export",
                None,
                StatusCode::NOT_FOUND,
            ),
            (
                Method::DELETE,
                "/v1/voice/sessions/missing-session",
                None,
                StatusCode::NOT_FOUND,
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
                Method::POST,
                "/v1/audio/align",
                Some("{}"),
                StatusCode::BAD_REQUEST,
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
            app.clone(),
            Method::GET,
            "/v1/speech-to-text/realtime/ws",
            None,
            StatusCode::BAD_REQUEST,
        )
        .await;
        assert_route_status(
            app,
            Method::GET,
            "/v1/transcription/realtime/ws",
            None,
            StatusCode::NOT_FOUND,
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

    #[tokio::test]
    async fn enterprise_policy_can_deny_requests_at_http_boundary() {
        let mut hooks = EnterpriseHooks::noop();
        hooks.policy = Arc::new(DenyAllPolicy);

        let (state, temp_dir) = test_state_with_hooks("enterprise_policy_denied", false, hooks);
        state.lifecycle.mark_ready();
        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled: false,
            ..ServeRuntimeConfig::default()
        };
        let app = create_router(state, &serve_config);

        assert_route_status(app, Method::GET, "/livez", None, StatusCode::FORBIDDEN).await;

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

    fn tiny_audio_json_body() -> String {
        tiny_audio_json_body_with_fields("")
    }

    fn tiny_audio_json_body_with_fields(extra_fields: &str) -> String {
        let wav = AudioEncoder::new(16_000, 1)
            .encode(&[0.0], AudioFormat::Wav)
            .expect("tiny wav should encode");
        let extra_fields = extra_fields.trim();
        let suffix = if extra_fields.is_empty() {
            String::new()
        } else {
            format!(",{extra_fields}")
        };
        format!(
            r#"{{"audio_base64":"{}"{}}}"#,
            base64::engine::general_purpose::STANDARD.encode(wav),
            suffix
        )
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
        test_state_with_hooks(name, ui_enabled, EnterpriseHooks::noop())
    }

    fn test_state_with_hooks(
        name: &str,
        ui_enabled: bool,
        enterprise_hooks: EnterpriseHooks,
    ) -> (AppState, TempDirGuard) {
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
        let state = AppState::with_enterprise_hooks(runtime, &serve_config, enterprise_hooks)
            .expect("app state should initialize");

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");

        (state, TempDirGuard(temp_dir))
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

    struct DenyAllPolicy;

    #[async_trait]
    impl PolicyEngine for DenyAllPolicy {
        async fn authorize(
            &self,
            _request: &AuthorizationRequest,
        ) -> HookResult<AuthorizationDecision> {
            Ok(AuthorizationDecision::deny("blocked by test policy"))
        }
    }
}
