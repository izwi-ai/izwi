//! First-party persisted TTS project routes.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/tts-project-folders",
            get(handlers::list_tts_project_folders).post(handlers::create_tts_project_folder),
        )
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
            "/tts-projects/:project_id/meta",
            get(handlers::get_tts_project_meta).patch(handlers::upsert_tts_project_meta),
        )
        .route(
            "/tts-projects/:project_id/pronunciations",
            get(handlers::list_tts_project_pronunciations)
                .post(handlers::create_tts_project_pronunciation),
        )
        .route(
            "/tts-projects/:project_id/pronunciations/:pronunciation_id",
            axum::routing::delete(handlers::delete_tts_project_pronunciation),
        )
        .route(
            "/tts-projects/:project_id/snapshots",
            get(handlers::list_tts_project_snapshots).post(handlers::create_tts_project_snapshot),
        )
        .route(
            "/tts-projects/:project_id/snapshots/:snapshot_id/restore",
            axum::routing::post(handlers::restore_tts_project_snapshot),
        )
        .route(
            "/tts-projects/:project_id/render-jobs",
            get(handlers::list_tts_project_render_jobs)
                .post(handlers::create_tts_project_render_job),
        )
        .route(
            "/tts-projects/:project_id/render-jobs/:job_id",
            axum::routing::patch(handlers::update_tts_project_render_job),
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
            "/tts-projects/:project_id/segments/:segment_id/merge-next",
            axum::routing::post(handlers::merge_tts_project_segment_with_next),
        )
        .route(
            "/tts-projects/:project_id/segments/reorder",
            axum::routing::patch(handlers::reorder_tts_project_segments),
        )
        .route(
            "/tts-projects/:project_id/segments/bulk-delete",
            axum::routing::post(handlers::bulk_delete_tts_project_segments),
        )
        .route(
            "/tts-projects/:project_id/segments/:segment_id/render",
            axum::routing::post(handlers::render_tts_project_segment),
        )
}
