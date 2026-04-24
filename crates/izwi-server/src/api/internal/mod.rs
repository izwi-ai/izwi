//! Internal API namespace.

pub mod health;
pub mod metrics;
pub mod probes;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health", get(health::health_check))
        .route("/live", get(probes::live_check))
        .route("/ready", get(probes::ready_check))
        .route("/metrics", get(metrics::metrics_json))
        .route("/metrics/prometheus", get(metrics::metrics_prometheus))
}
