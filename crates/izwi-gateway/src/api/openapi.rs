//! The gateway's OpenAPI document: the local server's document restricted to
//! the routes the gateway serves. The schema annotations live on
//! izwi-server's handlers, so izwi-server generates this file and its
//! `gateway_openapi_document_is_current` test keeps it in step.

use axum::Json;
use serde_json::Value;
use std::sync::OnceLock;

/// The checked-in document, `gateway-openapi.json`.
const GATEWAY_OPENAPI: &str = include_str!("gateway-openapi.json");

/// The document as JSON.
pub fn gateway_document() -> &'static Value {
    static DOCUMENT: OnceLock<Value> = OnceLock::new();
    DOCUMENT.get_or_init(|| {
        serde_json::from_str(GATEWAY_OPENAPI).expect("gateway-openapi.json is valid JSON")
    })
}

pub async fn gateway_openapi_json() -> Json<Value> {
    Json(gateway_document().clone())
}
