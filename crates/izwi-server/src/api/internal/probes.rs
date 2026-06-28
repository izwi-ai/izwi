//! Liveness and readiness probes.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use crate::state::AppState;

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
            ok: state.request_semaphore.available_permits() > 0,
            message: (state.request_semaphore.available_permits() == 0)
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

    let batch_store_check = state.batch_runtime_store.queued_stage_count().await;
    checks.push(ProbeCheck {
        name: "batch_runtime_store",
        ok: batch_store_check.is_ok(),
        message: batch_store_check
            .err()
            .map(|err| format!("batch runtime store unavailable: {err}")),
    });

    let ready = checks.iter().all(|check| check.ok);

    ReadyResponse {
        status: if ready { "ready" } else { "unready" },
        version: env!("CARGO_PKG_VERSION"),
        ready,
        phase: lifecycle.phase,
        draining: lifecycle.draining,
        uptime_secs: now_saturating_sub(lifecycle.started_at),
        checks,
        startup_warnings,
    }
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

    fn test_state(name: &str) -> (TempDirGuard, AppState) {
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
