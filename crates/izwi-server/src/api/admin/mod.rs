//! Local administrative API namespace.

pub mod models;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/admin/models", get(models::list_models))
        .route(
            "/admin/models/{variant}/download",
            axum::routing::post(models::download_model),
        )
        .route(
            "/admin/models/{variant}/download/progress",
            get(models::download_progress_stream),
        )
        .route(
            "/admin/models/{variant}/download/cancel",
            axum::routing::post(models::cancel_download),
        )
        .route(
            "/admin/models/{variant}/load",
            axum::routing::post(models::load_model),
        )
        .route(
            "/admin/models/{variant}/unload",
            axum::routing::post(models::unload_model),
        )
        .route(
            "/admin/models/{variant}",
            get(models::get_model_info).delete(models::delete_model),
        )
}
