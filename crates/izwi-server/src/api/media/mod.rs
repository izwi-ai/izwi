//! Media file serving endpoints for persisted local chat attachments.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/media/*path", get(handlers::get_media))
}
