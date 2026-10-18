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

use crate::api::request_context::{attach_enterprise_request_context, RequestTrace};
use crate::local_security::{guard_local_request, LocalRequestGuard};
use crate::logging::{SERVICE_NAME, SERVICE_VERSION};
use crate::state::AppState;

pub(crate) const DESKTOP_CORS_ORIGINS: &[&str] = &[
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
            // Resolved by the request-context middleware outside this layer.
            let trace = RequestTrace::of(request).unwrap_or_else(RequestTrace::detached);
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
                trace_id = %trace.trace_id(),
                span_id = %trace.span_id(),
                parent_span_id = %trace.parent_span_id(),
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
        .merge(crate::api::jobs::router())
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

    let app = apply_ui_fallback(app, serve_config)
        .layer(trace_layer)
        .layer(middleware::from_fn_with_state(
            middleware_state,
            attach_enterprise_request_context,
        ))
        .layer(middleware::from_fn_with_state(
            LocalRequestGuard::new(serve_config),
            guard_local_request,
        ));
    // CORS is the outermost layer so browser preflights are answered before
    // authentication, which would otherwise reject them for lacking a key.
    apply_cors(app, serve_config)
}

fn apply_ui_fallback(app: Router, serve_config: &ServeRuntimeConfig) -> Router {
    if !serve_config.ui_enabled {
        return app;
    }
    let index_path = serve_config.ui_dir.join("index.html");
    app.fallback_service(
        ServeDir::new(serve_config.ui_dir.clone()).fallback(ServeFile::new(index_path)),
    )
}

fn apply_cors(app: Router, serve_config: &ServeRuntimeConfig) -> Router {
    match build_cors_layer(serve_config) {
        Some(cors_layer) => app.layer(cors_layer),
        None => app,
    }
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
                    if !allowed_origins.contains(&value) {
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
#[path = "router_tests.rs"]
mod tests;
