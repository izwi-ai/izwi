//! First-party persisted Studio project routes.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/studio/folders",
            get(handlers::list_studio_project_folders).post(handlers::create_studio_project_folder),
        )
        .route(
            "/studio/projects",
            get(handlers::list_studio_projects).post(handlers::create_studio_project),
        )
        .route(
            "/studio/projects/{project_id}",
            get(handlers::get_studio_project)
                .patch(handlers::update_studio_project)
                .delete(handlers::delete_studio_project),
        )
        .route(
            "/studio/projects/{project_id}/audio",
            get(handlers::get_studio_project_audio),
        )
        .route(
            "/studio/projects/{project_id}/meta",
            get(handlers::get_studio_project_meta).patch(handlers::upsert_studio_project_meta),
        )
        .route(
            "/studio/projects/{project_id}/pronunciations",
            get(handlers::list_studio_project_pronunciations)
                .post(handlers::create_studio_project_pronunciation),
        )
        .route(
            "/studio/projects/{project_id}/pronunciations/{pronunciation_id}",
            axum::routing::delete(handlers::delete_studio_project_pronunciation),
        )
        .route(
            "/studio/projects/{project_id}/snapshots",
            get(handlers::list_studio_project_snapshots)
                .post(handlers::create_studio_project_snapshot),
        )
        .route(
            "/studio/projects/{project_id}/snapshots/{snapshot_id}/restore",
            axum::routing::post(handlers::restore_studio_project_snapshot),
        )
        .route(
            "/studio/projects/{project_id}/render-jobs",
            get(handlers::list_studio_project_render_jobs)
                .post(handlers::create_studio_project_render_job),
        )
        .route(
            "/studio/projects/{project_id}/render-jobs/{job_id}",
            axum::routing::patch(handlers::update_studio_project_render_job),
        )
        .route(
            "/studio/projects/{project_id}/segments/{segment_id}",
            get(handlers::get_studio_project_segment)
                .patch(handlers::update_studio_project_segment)
                .delete(handlers::delete_studio_project_segment),
        )
        .route(
            "/studio/projects/{project_id}/segments",
            axum::routing::post(handlers::create_studio_project_segment),
        )
        .route(
            "/studio/projects/{project_id}/segments/{segment_id}/split",
            axum::routing::post(handlers::split_studio_project_segment),
        )
        .route(
            "/studio/projects/{project_id}/segments/{segment_id}/merge-next",
            axum::routing::post(handlers::merge_studio_project_segment_with_next),
        )
        .route(
            "/studio/projects/{project_id}/segments/reorder",
            axum::routing::patch(handlers::reorder_studio_project_segments),
        )
        .route(
            "/studio/projects/{project_id}/segments/bulk-delete",
            axum::routing::post(handlers::bulk_delete_studio_project_segments),
        )
        .route(
            "/studio/projects/{project_id}/segments/{segment_id}/render",
            axum::routing::post(handlers::render_studio_project_segment),
        )
}
