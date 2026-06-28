//! Runtime and engine telemetry endpoints.

use axum::{body::Body, extract::State, http::header, response::Response, Json};
use izwi_core::RuntimeTelemetrySnapshot;
use serde::Serialize;
use tracing::warn;

use crate::{
    batch_runtime::{
        store::{RuntimeJobStatusCount, RuntimeStageStatusCount},
        worker::BatchWorkerSnapshot,
    },
    error::ApiError,
    state::AppState,
};

#[derive(Debug, Clone, Serialize)]
pub struct BatchRuntimeMetricsResponse {
    pub queued_stages: u64,
    pub jobs_by_status: Vec<RuntimeJobStatusCount>,
    pub stages_by_status: Vec<RuntimeStageStatusCount>,
    pub worker: BatchWorkerSnapshot,
}

pub async fn metrics_json(State(state): State<AppState>) -> Json<RuntimeTelemetrySnapshot> {
    Json(state.runtime.telemetry_snapshot().await)
}

pub async fn batch_runtime_metrics_json(
    State(state): State<AppState>,
) -> Result<Json<BatchRuntimeMetricsResponse>, ApiError> {
    collect_batch_runtime_metrics(&state)
        .await
        .map(Json)
        .map_err(|err| ApiError::internal(format!("Batch runtime metrics error: {err}")))
}

pub async fn metrics_prometheus(State(state): State<AppState>) -> Response<Body> {
    let mut payload = state.runtime.telemetry_prometheus().await;
    match collect_batch_runtime_metrics(&state).await {
        Ok(batch) => append_batch_prometheus_metrics(&mut payload, &batch),
        Err(err) => {
            warn!(error = %err, "Failed to collect batch runtime Prometheus metrics");
            payload.push_str("\n# HELP izwi_batch_runtime_metrics_collect_error Batch runtime metrics collection failure.\n");
            payload.push_str("# TYPE izwi_batch_runtime_metrics_collect_error gauge\n");
            payload.push_str("izwi_batch_runtime_metrics_collect_error 1\n");
        }
    }
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(payload))
        .unwrap()
}

async fn collect_batch_runtime_metrics(
    state: &AppState,
) -> anyhow::Result<BatchRuntimeMetricsResponse> {
    Ok(BatchRuntimeMetricsResponse {
        queued_stages: state.batch_runtime_store.queued_stage_count().await?,
        jobs_by_status: state.batch_runtime_store.job_status_counts().await?,
        stages_by_status: state.batch_runtime_store.stage_status_counts().await?,
        worker: state.batch_worker_health.snapshot(),
    })
}

fn append_batch_prometheus_metrics(payload: &mut String, batch: &BatchRuntimeMetricsResponse) {
    payload.push_str(
        "\n# HELP izwi_batch_runtime_queued_stages Queued or retrying batch runtime stages.\n",
    );
    payload.push_str("# TYPE izwi_batch_runtime_queued_stages gauge\n");
    payload.push_str(&format!(
        "izwi_batch_runtime_queued_stages {}\n",
        batch.queued_stages
    ));

    payload.push_str("# HELP izwi_batch_runtime_jobs Runtime jobs by status.\n");
    payload.push_str("# TYPE izwi_batch_runtime_jobs gauge\n");
    for count in &batch.jobs_by_status {
        payload.push_str(&format!(
            "izwi_batch_runtime_jobs{{status=\"{}\"}} {}\n",
            escape_prometheus_label(count.status.as_db_value()),
            count.count
        ));
    }

    payload.push_str("# HELP izwi_batch_runtime_stages Runtime job stages by status.\n");
    payload.push_str("# TYPE izwi_batch_runtime_stages gauge\n");
    for count in &batch.stages_by_status {
        payload.push_str(&format!(
            "izwi_batch_runtime_stages{{status=\"{}\"}} {}\n",
            escape_prometheus_label(count.status.as_db_value()),
            count.count
        ));
    }

    payload
        .push_str("# HELP izwi_batch_runtime_worker_running Local batch worker running state.\n");
    payload.push_str("# TYPE izwi_batch_runtime_worker_running gauge\n");
    payload.push_str(&format!(
        "izwi_batch_runtime_worker_running{{worker_id=\"{}\"}} {}\n",
        escape_prometheus_label(&batch.worker.worker_id),
        u8::from(batch.worker.running)
    ));
}

fn escape_prometheus_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}
