//! Agent session and turn endpoints. Deprecated: `/v1/responses` runs text
//! turns with tools and stored state, and `/v1/realtime` runs voice agents.
//! Every response says so with a `Deprecation` header and a `Link` to the
//! successor.

mod handlers;

use axum::{
    http::{header, HeaderName, HeaderValue},
    middleware::map_response,
    response::Response,
    routing::{get, post},
    Router,
};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/agent/sessions", post(handlers::create_session))
        .route("/agent/sessions/{session_id}", get(handlers::get_session))
        .route(
            "/agent/sessions/{session_id}/turns",
            post(handlers::create_turn),
        )
        .layer(map_response(mark_deprecated))
}

async fn mark_deprecated(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        HeaderName::from_static("deprecation"),
        HeaderValue::from_static("true"),
    );
    headers.insert(
        header::LINK,
        HeaderValue::from_static("</v1/responses>; rel=\"successor-version\""),
    );
    response
}
