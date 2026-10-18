//! Request context middleware. The context, request ids, tracing and the
//! enterprise and gateway middleware live in izwi-gateway, which both roles
//! share; this module binds the local role's middleware to its state.

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;

pub use izwi_gateway::api::request_context::*;

use crate::state::AppState;

pub async fn attach_enterprise_request_context(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    attach_enterprise_request_context_with_hooks(
        &state.enterprise_hooks,
        Some(state.request_admission_snapshot().global.capacity),
        crate::env_config::tuning().max_speech_requests_per_tenant,
        req,
        next,
    )
    .await
}
