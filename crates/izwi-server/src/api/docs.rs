//! Scalar API reference routes.

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use scalar_api_reference::{get_asset_with_mime, scalar_html};
use serde_json::{json, Value};

use crate::state::AppState;

pub const DOCS_PATH: &str = "/docs";
pub const SCALAR_JS_PATH: &str = "/docs/scalar.js";
pub const OPENAPI_PATH: &str = "/openapi.json";

pub fn router() -> Router<AppState> {
    Router::new()
        .route(DOCS_PATH, get(scalar_docs))
        .route(SCALAR_JS_PATH, get(scalar_js))
}

pub fn scalar_config() -> Value {
    json!({
        "url": OPENAPI_PATH,
        "agent": {
            "disabled": true
        }
    })
}

async fn scalar_docs() -> Html<String> {
    Html(scalar_html(&scalar_config(), Some(SCALAR_JS_PATH)))
}

async fn scalar_js() -> Response<Body> {
    match get_asset_with_mime("scalar.js") {
        Some((mime_type, content)) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime_type)
            .body(Body::from(content))
            .expect("scalar asset response should build"),
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not found"))
            .expect("scalar not-found response should build"),
    }
}
