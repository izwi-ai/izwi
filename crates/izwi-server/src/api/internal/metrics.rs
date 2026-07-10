//! Runtime and engine telemetry endpoints.

use axum::{body::Body, extract::State, http::header, response::Response, Json};
use izwi_core::RuntimeTelemetrySnapshot;
use serde::Serialize;
use tracing::warn;

use crate::{
    batch_runtime::{
        store::{RuntimeJobStatusCount, RuntimeQueueHealthSnapshot, RuntimeStageStatusCount},
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
    pub queue_health: RuntimeQueueHealthSnapshot,
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
    append_server_admission_prometheus_metrics(&mut payload, &state);
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(payload))
        .unwrap()
}

fn append_server_admission_prometheus_metrics(payload: &mut String, state: &AppState) {
    let realtime = state.realtime_session_admission_snapshot();
    let media = state.media_ingest.lane_snapshot();
    payload.push_str(
        "# HELP izwi_realtime_sessions Active and available realtime websocket session slots.\n",
    );
    payload.push_str("# TYPE izwi_realtime_sessions gauge\n");
    payload.push_str(&format!(
        "izwi_realtime_sessions{{state=\"active\"}} {}\n",
        realtime.active
    ));
    payload.push_str(&format!(
        "izwi_realtime_sessions{{state=\"available\"}} {}\n",
        realtime.available
    ));
    payload.push_str(&format!(
        "izwi_realtime_sessions{{state=\"capacity\"}} {}\n",
        realtime.capacity
    ));
    payload.push_str(
        "# HELP izwi_media_ingest_decode_lanes Active and available bounded media decode lanes.\n",
    );
    payload.push_str("# TYPE izwi_media_ingest_decode_lanes gauge\n");
    payload.push_str(&format!(
        "izwi_media_ingest_decode_lanes{{state=\"active\"}} {}\n",
        media.active
    ));
    payload.push_str(&format!(
        "izwi_media_ingest_decode_lanes{{state=\"available\"}} {}\n",
        media.available
    ));
    payload.push_str(&format!(
        "izwi_media_ingest_decode_lanes{{state=\"capacity\"}} {}\n",
        media.capacity
    ));
    payload.push_str(&format!(
        "izwi_media_ingest_decode_lanes{{state=\"queued\"}} {}\n",
        media.queued
    ));
}

async fn collect_batch_runtime_metrics(
    state: &AppState,
) -> anyhow::Result<BatchRuntimeMetricsResponse> {
    Ok(BatchRuntimeMetricsResponse {
        queued_stages: state.batch_runtime_store.queued_stage_count().await?,
        jobs_by_status: state.batch_runtime_store.job_status_counts().await?,
        stages_by_status: state.batch_runtime_store.stage_status_counts().await?,
        queue_health: state
            .batch_runtime_store
            .runtime_queue_health(super::probes::resolve_batch_heartbeat_stale_after_ms())
            .await?,
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
        .push_str("# HELP izwi_batch_runtime_queue_depth Queued stages by durable queue class.\n");
    payload.push_str("# TYPE izwi_batch_runtime_queue_depth gauge\n");
    payload.push_str(
        "# HELP izwi_batch_runtime_queue_oldest_age_ms Oldest queued-stage age by durable queue class.\n",
    );
    payload.push_str("# TYPE izwi_batch_runtime_queue_oldest_age_ms gauge\n");
    payload.push_str(
        "# HELP izwi_batch_runtime_queue_uncovered Whether a queued durable class has no fresh eligible worker heartbeat.\n",
    );
    payload.push_str("# TYPE izwi_batch_runtime_queue_uncovered gauge\n");
    for queue in &batch.queue_health.queues {
        let label = escape_prometheus_label(queue.queue_class.as_db_value());
        payload.push_str(&format!(
            "izwi_batch_runtime_queue_depth{{queue_class=\"{label}\"}} {}\n",
            queue.count
        ));
        payload.push_str(&format!(
            "izwi_batch_runtime_queue_oldest_age_ms{{queue_class=\"{label}\"}} {}\n",
            queue.oldest_age_ms
        ));
        payload.push_str(&format!(
            "izwi_batch_runtime_queue_uncovered{{queue_class=\"{label}\"}} {}\n",
            u8::from(
                batch
                    .queue_health
                    .uncovered_queue_classes
                    .contains(&queue.queue_class)
            )
        ));
    }

    payload.push_str(
        "# HELP izwi_batch_runtime_workers Durable worker heartbeat state after freshness filtering.\n",
    );
    payload.push_str("# TYPE izwi_batch_runtime_workers gauge\n");
    payload.push_str(&format!(
        "izwi_batch_runtime_workers{{state=\"active\"}} {}\n",
        batch.queue_health.active_workers
    ));
    payload.push_str(&format!(
        "izwi_batch_runtime_workers{{state=\"healthy\"}} {}\n",
        batch.queue_health.healthy_workers
    ));
    payload.push_str(&format!(
        "izwi_batch_runtime_workers{{state=\"stale\"}} {}\n",
        batch.queue_health.stale_workers
    ));

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
