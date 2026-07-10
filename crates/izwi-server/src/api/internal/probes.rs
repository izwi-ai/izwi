//! Liveness and readiness probes.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use crate::batch_runtime::store::{current_timestamp_millis, RuntimeQueueHealthSnapshot};
use crate::media_ingest::MediaIngestLaneSnapshot;
use crate::state::{AppState, RealtimeSessionAdmissionSnapshot, RequestAdmissionSnapshot};

const DEFAULT_BATCH_HEARTBEAT_STALE_SECS: u64 = 5;

#[derive(Debug, Serialize)]
pub struct ProbeCheck {
    pub name: &'static str,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LiveResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub uptime_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub ready: bool,
    pub phase: String,
    pub draining: bool,
    pub uptime_secs: u64,
    pub request_admission: RequestAdmissionSnapshot,
    pub realtime_session_admission: RealtimeSessionAdmissionSnapshot,
    pub media_ingest: MediaIngestLaneSnapshot,
    pub batch_runtime: Option<RuntimeQueueHealthSnapshot>,
    pub checks: Vec<ProbeCheck>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub startup_warnings: Vec<String>,
}

pub async fn live_check(State(state): State<AppState>) -> Json<LiveResponse> {
    let lifecycle = state.lifecycle.snapshot();
    Json(LiveResponse {
        status: "alive",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: now_saturating_sub(lifecycle.started_at),
    })
}

pub async fn ready_check(State(state): State<AppState>) -> Response {
    let response = readiness_response(&state).await;
    let status = if response.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(response)).into_response()
}

async fn readiness_response(state: &AppState) -> ReadyResponse {
    let lifecycle = state.lifecycle.snapshot();
    let backend_context = state.runtime.backend_context();
    let telemetry = state.runtime.telemetry_snapshot().await;
    let startup_warnings = lifecycle.startup_warnings.clone();
    let preload_complete = startup_warnings.is_empty();
    let request_admission = state.request_admission_snapshot();
    let realtime_session_admission = state.realtime_session_admission_snapshot();
    let media_ingest = state.media_ingest.lane_snapshot();
    let heartbeat_stale_after_ms = resolve_batch_heartbeat_stale_after_ms();
    let batch_runtime = state
        .batch_runtime_store
        .runtime_queue_health(heartbeat_stale_after_ms)
        .await;

    let mut checks = vec![
        ProbeCheck {
            name: "lifecycle_ready",
            ok: lifecycle.ready,
            message: (!lifecycle.ready).then(|| format!("phase is {}", lifecycle.phase)),
        },
        ProbeCheck {
            name: "not_draining",
            ok: !lifecycle.draining,
            message: lifecycle
                .draining
                .then(|| "server is draining for shutdown".to_string()),
        },
        ProbeCheck {
            name: "preload_complete",
            ok: preload_complete,
            message: (!preload_complete).then(|| startup_warnings.join("; ")),
        },
        ProbeCheck {
            name: "backend_available",
            ok: backend_context.matches_preference(),
            message: (!backend_context.matches_preference()).then(|| {
                format!(
                    "requested backend {} selected {}",
                    backend_context.preference.as_str(),
                    backend_context.backend_kind.as_str()
                )
            }),
        },
        ProbeCheck {
            name: "stores_available",
            ok: true,
            message: None,
        },
        ProbeCheck {
            name: "request_capacity",
            ok: request_admission.global.available > 0,
            message: (request_admission.global.available == 0)
                .then(|| "all request permits are currently in use".to_string()),
        },
    ];

    let worker_healthy = telemetry.worker_panics <= telemetry.worker_restarts;
    checks.push(ProbeCheck {
        name: "worker_health",
        ok: worker_healthy,
        message: (!worker_healthy).then(|| {
            format!(
                "worker panics ({}) exceed restarts ({})",
                telemetry.worker_panics, telemetry.worker_restarts
            )
        }),
    });

    let batch_worker = state.batch_worker_health.snapshot();
    checks.push(ProbeCheck {
        name: "batch_worker_health",
        ok: batch_worker.last_error.is_none(),
        message: batch_worker.last_error,
    });

    checks.push(ProbeCheck {
        name: "batch_runtime_store",
        ok: batch_runtime.is_ok(),
        message: batch_runtime
            .as_ref()
            .err()
            .map(|err| format!("batch runtime store unavailable: {err}")),
    });
    if let Ok(queue_health) = batch_runtime.as_ref() {
        checks.push(ProbeCheck {
            name: "batch_queue_coverage",
            ok: queue_health.uncovered_queue_classes.is_empty(),
            message: (!queue_health.uncovered_queue_classes.is_empty()).then(|| {
                format!(
                    "no fresh eligible worker heartbeat covers queues: {}",
                    queue_health
                        .uncovered_queue_classes
                        .iter()
                        .map(|queue| queue.as_db_value())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }),
        });
    }

    if batch_worker.running {
        let durable_heartbeat = state
            .batch_runtime_store
            .get_worker_heartbeat(&batch_worker.worker_id)
            .await;
        let (heartbeat_ok, heartbeat_message) = match durable_heartbeat {
            Ok(Some(heartbeat)) => {
                let now = u64::try_from(current_timestamp_millis()).unwrap_or_default();
                let age_ms = now.saturating_sub(heartbeat.last_heartbeat_at);
                let fresh = age_ms <= heartbeat_stale_after_ms;
                let active = matches!(heartbeat.status.as_str(), "polling" | "idle" | "running");
                let same_instance = batch_worker.instance_id.is_empty()
                    || batch_worker.instance_id == heartbeat.instance_id;
                let ok = fresh && active && same_instance;
                let message = (!ok).then(|| {
                    format!(
                        "local worker heartbeat status={}, age_ms={}, expected_instance={}, durable_instance={}",
                        heartbeat.status,
                        age_ms,
                        batch_worker.instance_id,
                        heartbeat.instance_id
                    )
                });
                (ok, message)
            }
            Ok(None) => (
                false,
                Some("local batch worker has not published a durable heartbeat".to_string()),
            ),
            Err(err) => (
                false,
                Some(format!("local batch worker heartbeat unavailable: {err}")),
            ),
        };
        checks.push(ProbeCheck {
            name: "batch_worker_heartbeat",
            ok: heartbeat_ok,
            message: heartbeat_message,
        });
    }

    let ready = checks.iter().all(|check| check.ok);

    ReadyResponse {
        status: if ready { "ready" } else { "unready" },
        version: env!("CARGO_PKG_VERSION"),
        ready,
        phase: lifecycle.phase,
        draining: lifecycle.draining,
        uptime_secs: now_saturating_sub(lifecycle.started_at),
        request_admission,
        realtime_session_admission,
        media_ingest,
        batch_runtime: batch_runtime.ok(),
        checks,
        startup_warnings,
    }
}

pub(super) fn resolve_batch_heartbeat_stale_after_ms() -> u64 {
    std::env::var("IZWI_BATCH_WORKER_HEARTBEAT_STALE_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_BATCH_HEARTBEAT_STALE_SECS)
        .saturating_mul(1_000)
}

fn now_saturating_sub(started_at: u64) -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    now.saturating_sub(started_at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_runtime::store::WorkerHeartbeatUpdate;
    use crate::state::AppState;
    use crate::test_support::env_lock;
    use izwi_core::{backends::BackendPreference, RuntimeService, ServeRuntimeConfig};

    #[tokio::test]
    async fn readiness_reports_ready_after_lifecycle_mark_ready() {
        let (_guard, state) = test_state("readiness_ready");
        state.lifecycle.mark_ready();

        let response = readiness_response(&state).await;

        assert!(response.ready);
        assert_eq!(response.status, "ready");
        assert!(response.request_admission.global.capacity >= 1);
        assert_eq!(
            response.request_admission.global.available,
            response.request_admission.global.capacity
        );
        assert!(response.checks.iter().all(|check| check.ok));
    }

    #[tokio::test]
    async fn readiness_reports_unready_when_draining() {
        let (_guard, state) = test_state("readiness_draining");
        state.lifecycle.mark_ready();
        state.lifecycle.mark_draining();

        let response = readiness_response(&state).await;

        assert!(!response.ready);
        assert_eq!(response.status, "unready");
        assert!(response
            .checks
            .iter()
            .any(|check| check.name == "not_draining" && !check.ok));
    }

    #[tokio::test]
    async fn readiness_reports_unready_when_startup_warnings_exist() {
        let (_guard, state) = test_state("readiness_startup_warnings");
        state
            .lifecycle
            .record_startup_warnings(vec!["failed to preload model test".to_string()]);
        state.lifecycle.mark_ready();

        let response = readiness_response(&state).await;

        assert!(!response.ready);
        assert_eq!(response.status, "unready");
        assert_eq!(
            response.startup_warnings,
            vec!["failed to preload model test"]
        );
        assert!(response
            .checks
            .iter()
            .any(|check| check.name == "preload_complete" && !check.ok));
    }

    #[tokio::test]
    async fn readiness_reports_realtime_session_saturation_without_becoming_unready() {
        let (_guard, state) = test_state_with_realtime_sessions("readiness_sessions_full", Some(1));
        state.lifecycle.mark_ready();
        let _session = state
            .try_acquire_realtime_session()
            .expect("configured realtime session should be admitted");

        let response = readiness_response(&state).await;

        assert!(response.ready);
        assert_eq!(response.status, "ready");
        assert_eq!(response.realtime_session_admission.capacity, 1);
        assert_eq!(response.realtime_session_admission.active, 1);
        assert_eq!(response.realtime_session_admission.available, 0);
        assert!(response.checks.iter().all(|check| check.ok));
    }

    #[tokio::test]
    async fn readiness_requires_a_fresh_durable_heartbeat_for_a_running_local_worker() {
        let (_guard, state) = test_state("readiness_worker_heartbeat");
        state.lifecycle.mark_ready();
        state.batch_worker_health.mark_running();

        let missing = readiness_response(&state).await;
        assert!(!missing.ready);
        assert!(missing
            .checks
            .iter()
            .any(|check| check.name == "batch_worker_heartbeat" && !check.ok));

        state
            .batch_runtime_store
            .upsert_worker_heartbeat(WorkerHeartbeatUpdate {
                worker_id: "local-batch-worker".to_string(),
                status: "idle".to_string(),
                queue_names: vec!["batch".to_string()],
                current_job_id: None,
                current_stage_id: None,
                diagnostic_json: serde_json::json!({}),
            })
            .await
            .expect("durable heartbeat");

        let fresh = readiness_response(&state).await;
        assert!(fresh.ready);
        assert_eq!(
            fresh.batch_runtime.expect("batch health").healthy_workers,
            1
        );
    }

    fn test_state(name: &str) -> (TempDirGuard, AppState) {
        test_state_with_realtime_sessions(name, None)
    }

    fn test_state_with_realtime_sessions(
        name: &str,
        realtime_session_capacity: Option<usize>,
    ) -> (TempDirGuard, AppState) {
        let temp_dir = std::env::temp_dir().join(format!(
            "izwi-probes-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should be monotonic")
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).expect("temp dir should be created");
        let db_path = temp_dir.join("izwi.sqlite3");
        let media_dir = temp_dir.join("media");
        let models_dir = temp_dir.join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir should be created");

        let guard = env_lock();
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);
        if let Some(capacity) = realtime_session_capacity {
            std::env::set_var("IZWI_MAX_REALTIME_SESSIONS", capacity.to_string());
        }

        let serve_config = ServeRuntimeConfig {
            backend: BackendPreference::Cpu,
            ui_enabled: false,
            models_dir,
            ..ServeRuntimeConfig::default()
        };
        let runtime = RuntimeService::new(serve_config.engine_config()).expect("runtime");
        let state = AppState::new(runtime, &serve_config).expect("state");
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
        std::env::remove_var("IZWI_MAX_REALTIME_SESSIONS");

        (
            TempDirGuard {
                path: temp_dir,
                _env: guard,
            },
            state,
        )
    }

    struct TempDirGuard {
        path: std::path::PathBuf,
        _env: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
