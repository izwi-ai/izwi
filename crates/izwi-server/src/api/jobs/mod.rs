//! Durable runtime job management routes.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/jobs/{job_id}", get(handlers::get_job))
        .route(
            "/jobs/{job_id}/cancel",
            axum::routing::post(handlers::cancel_job),
        )
        .route(
            "/jobs/{job_id}/retry",
            axum::routing::post(handlers::retry_job),
        )
        .route(
            "/jobs/{job_id}/artifacts",
            get(handlers::list_job_artifacts),
        )
}
