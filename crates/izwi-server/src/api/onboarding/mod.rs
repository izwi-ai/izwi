//! First-run onboarding state routes.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/onboarding", get(handlers::get_onboarding_state))
        .route("/onboarding/complete", axum::routing::post(handlers::complete_onboarding))
}
