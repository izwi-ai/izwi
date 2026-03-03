use axum::{extract::Request, middleware, Router};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info_span;

use crate::api::request_context::attach_request_context;
use crate::state::AppState;

/// Create the main API router.
pub fn create_router(state: AppState) -> Router {
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

    // Get UI directory from environment or use default relative path
    let ui_dir = std::env::var("IZWI_UI_DIR").unwrap_or_else(|_| "ui/dist".to_string());
    let index_path = format!("{}/index.html", ui_dir);

    Router::new()
        .nest("/v1", v1_routes)
        // Serve static files for UI
        .fallback_service(
            tower_http::services::ServeDir::new(&ui_dir)
                .fallback(tower_http::services::ServeFile::new(&index_path)),
        )
        .layer(trace_layer)
        .layer(middleware::from_fn(attach_request_context))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}
