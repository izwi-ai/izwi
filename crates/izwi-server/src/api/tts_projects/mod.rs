//! First-party persisted TTS project routes.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/tts-projects",
            get(handlers::list_tts_projects).post(handlers::create_tts_project),
        )
        .route(
            "/tts-projects/:project_id",
            get(handlers::get_tts_project)
                .patch(handlers::update_tts_project)
                .delete(handlers::delete_tts_project),
        )
        .route(
            "/tts-projects/:project_id/audio",
            get(handlers::get_tts_project_audio),
        )
        .route(
            "/tts-projects/:project_id/segments/:segment_id",
            get(handlers::get_tts_project_segment)
                .patch(handlers::update_tts_project_segment)
                .delete(handlers::delete_tts_project_segment),
        )
        .route(
            "/tts-projects/:project_id/segments/:segment_id/split",
            axum::routing::post(handlers::split_tts_project_segment),
        )
        .route(
            "/tts-projects/:project_id/segments/:segment_id/render",
            axum::routing::post(handlers::render_tts_project_segment),
        )
}
