use izwi_core::backends::{BackendKind, BackendPreference};
use izwi_core::ServeRuntimeConfigOverrides;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

use super::*;
use crate::batch_runtime::types::{DeviceClass, ResourceTarget, RuntimeBackendClass};
use crate::cli::{resolve_serve_runtime_config_with_env, GatewayTopology};
use crate::logging::LogFormat;
use crate::roles::local::{
    asr_warmup_duration_ms, batch_stage_execution_timeout, batch_worker_concurrency,
    batch_worker_drain_timeout, configured_preload_models, http_shutdown_grace_timeout,
    local_batch_worker_queue_names, local_batch_worker_resources, serve_local,
    warmup_preloaded_models_enabled,
};
use crate::shutdown::{
    serve_with_shutdown_plan, shutdown_worker_then_cleanup, wait_for_owner_pipe_close,
    ShutdownPhases, ShutdownPlan,
};
use crate::test_support::env_lock;

fn local_test_state() -> (AppState, ServeRuntimeConfig, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let models_dir = temp_dir.path().join("models");
    std::fs::create_dir_all(&models_dir).expect("models dir");
    std::env::set_var("IZWI_DB_PATH", temp_dir.path().join("izwi.sqlite3"));
    std::env::set_var("IZWI_MEDIA_DIR", temp_dir.path().join("media"));
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
    (state, serve_config, temp_dir)
}

/// A short local shutdown: no pre-stop, a 2 s drain.
fn test_shutdown_plan() -> ShutdownPlan {
    ShutdownPlan {
        prestop: Duration::ZERO,
        budget: Duration::from_secs(2),
    }
}

async fn lifecycle_ready(client: &reqwest::Client, base: &str) -> (u16, bool) {
    let response = client
        .get(format!("{base}/readyz"))
        .send()
        .await
        .expect("readiness answers");
    let status = response.status().as_u16();
    let body: serde_json::Value = response.json().await.expect("readiness JSON");
    let lifecycle_ready = body["checks"]
        .as_array()
        .expect("readiness checks")
        .iter()
        .any(|check| check["name"] == "lifecycle_ready" && check["ok"] == true);
    (status, lifecycle_ready)
}

#[test]
fn bounded_exit_does_not_wait_for_a_blocking_drain_that_never_finishes() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = run_with_bounded_exit(
            async {
                // Stands in for an unload stuck in Device::synchronize.
                tokio::task::spawn_blocking(|| loop {
                    std::thread::park();
                });
                Ok(())
            },
            Duration::from_millis(200),
        );
        let _ = done_tx.send(result.is_ok());
    });
    let started = Instant::now();
    let ok = done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("process exit must not wait for a never-ending blocking task");
    assert!(ok);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[tokio::test]
async fn local_server_answers_liveness_while_models_preload() {
    let _env = env_lock();
    let (state, serve_config, _temp_dir) = local_test_state();
    let app = api::create_router(state.clone(), &serve_config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let base = format!("http://{}", listener.local_addr().expect("address"));
    let (release_preload, preload_released) = oneshot::channel::<()>();
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(serve_local(
        state.clone(),
        listener,
        app,
        async move {
            let _ = preload_released.await;
            Vec::new()
        },
        async move {
            let _ = stopped.await;
        },
        test_shutdown_plan(),
    ));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client");

    let live = client
        .get(format!("{base}/livez"))
        .send()
        .await
        .expect("liveness must answer while models preload");
    assert_eq!(live.status().as_u16(), 200);
    assert_eq!(lifecycle_ready(&client, &base).await, (503, false));

    release_preload.send(()).expect("preload is waiting");
    tokio::time::timeout(Duration::from_secs(10), async {
        while !lifecycle_ready(&client, &base).await.1 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("ready once preload finishes");

    stop.send(()).expect("server is running");
    tokio::time::timeout(Duration::from_secs(60), server)
        .await
        .expect("server stops")
        .expect("server task")
        .expect("server result");
}

#[tokio::test]
async fn shutdown_during_preload_abandons_startup_without_becoming_ready() {
    let _env = env_lock();
    let (state, serve_config, _temp_dir) = local_test_state();
    let app = api::create_router(state.clone(), &serve_config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");

    tokio::time::timeout(
        Duration::from_secs(60),
        serve_local(
            state.clone(),
            listener,
            app,
            std::future::pending::<Vec<String>>(),
            std::future::ready(()),
            test_shutdown_plan(),
        ),
    )
    .await
    .expect("shutdown does not wait for preload")
    .expect("server result");
    let lifecycle = state.lifecycle.snapshot();
    assert!(!lifecycle.ready);
    assert!(lifecycle.draining);
    assert!(!state.batch_worker_health.snapshot().running);
}

#[test]
fn durable_worker_concurrency_tracks_runtime_and_operator_ceiling() {
    assert_eq!(batch_worker_concurrency(64, None), 64);
    assert_eq!(batch_worker_concurrency(64, Some("17")), 17);
    assert_eq!(batch_worker_concurrency(3, Some("17")), 3);
    assert_eq!(batch_worker_concurrency(3, Some("0")), 3);
    assert_eq!(batch_worker_concurrency(0, None), 1);
}

#[test]
fn desktop_owner_pipe_monitor_returns_at_eof() {
    wait_for_owner_pipe_close(std::io::Cursor::new(Vec::<u8>::new()));
}

#[tokio::test]
async fn unconfirmed_worker_shutdown_skips_runtime_cleanup() {
    let cleaned = Arc::new(AtomicBool::new(false));
    let cleanup_flag = cleaned.clone();
    let result = shutdown_worker_then_cleanup(
        async {
            tokio::time::timeout(Duration::from_millis(1), std::future::pending::<()>())
                .await
                .map_err(|_| anyhow::anyhow!("injected worker shutdown timeout"))
        },
        async move {
            cleanup_flag.store(true, Ordering::Release);
        },
    )
    .await;

    assert!(result.is_err());
    assert!(!cleaned.load(Ordering::Acquire));
}

#[tokio::test]
async fn confirmed_worker_shutdown_runs_runtime_cleanup() {
    let cleaned = Arc::new(AtomicBool::new(false));
    let cleanup_flag = cleaned.clone();
    shutdown_worker_then_cleanup(async { Ok(()) }, async move {
        cleanup_flag.store(true, Ordering::Release);
    })
    .await
    .expect("confirmed worker shutdown");

    assert!(cleaned.load(Ordering::Acquire));
}

#[test]
fn local_batch_worker_subscribes_to_explicit_runtime_queues() {
    assert_eq!(
        local_batch_worker_queue_names(),
        vec!["batch_asr", "long_form_asr", "batch_tts"]
    );
}

#[test]
fn local_batch_worker_reports_selected_backend_resources() {
    let cpu = local_batch_worker_resources(BackendKind::Cpu, Some(1024));
    assert_eq!(cpu.targets, vec![ResourceTarget::Cpu]);
    assert_eq!(cpu.backends, vec![RuntimeBackendClass::Cpu]);
    assert_eq!(cpu.device_classes, vec![DeviceClass::Cpu]);
    assert_eq!(cpu.memory_bytes, Some(1024));

    let metal = local_batch_worker_resources(BackendKind::Metal, None);
    assert_eq!(metal.targets, vec![ResourceTarget::Gpu]);
    assert_eq!(metal.backends, vec![RuntimeBackendClass::Metal]);
    assert_eq!(metal.device_classes, vec![DeviceClass::AppleGpu]);

    let cuda = local_batch_worker_resources(BackendKind::Cuda, Some(2048));
    assert_eq!(cuda.targets, vec![ResourceTarget::Gpu]);
    assert_eq!(cuda.backends, vec![RuntimeBackendClass::Cuda]);
    assert_eq!(cuda.device_classes, vec![DeviceClass::NvidiaGpu]);
    assert_eq!(cuda.memory_bytes, Some(2048));
}

#[test]
fn local_batch_worker_timeouts_are_configurable_and_bounded() {
    let _guard = env_lock();
    std::env::remove_var("IZWI_BATCH_STAGE_TIMEOUT_SECS");
    std::env::remove_var("IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS");
    std::env::remove_var("IZWI_HTTP_SHUTDOWN_GRACE_SECS");
    std::env::remove_var("IZWI_GATEWAY_WORKER_TLS_CA_REFS");
    std::env::remove_var("IZWI_GATEWAY_WORKER_TLS_CLIENT_CERT_REF");
    std::env::remove_var("IZWI_GATEWAY_WORKER_TLS_CLIENT_KEY_REF");
    assert_eq!(batch_stage_execution_timeout(), None);
    assert_eq!(batch_worker_drain_timeout(), Duration::from_secs(20));
    assert_eq!(http_shutdown_grace_timeout(), Duration::from_secs(20));

    std::env::set_var("IZWI_BATCH_STAGE_TIMEOUT_SECS", "1");
    std::env::set_var("IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS", "999");
    std::env::set_var("IZWI_HTTP_SHUTDOWN_GRACE_SECS", "0");
    assert_eq!(
        batch_stage_execution_timeout(),
        Some(Duration::from_secs(30))
    );
    assert_eq!(batch_worker_drain_timeout(), Duration::from_secs(300));
    assert_eq!(http_shutdown_grace_timeout(), Duration::from_secs(20));

    std::env::set_var("IZWI_HTTP_SHUTDOWN_GRACE_SECS", "999");
    assert_eq!(http_shutdown_grace_timeout(), Duration::from_secs(300));

    std::env::remove_var("IZWI_BATCH_STAGE_TIMEOUT_SECS");
    std::env::remove_var("IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS");
    std::env::remove_var("IZWI_HTTP_SHUTDOWN_GRACE_SECS");
}

/// Records when each shutdown phase ran.
#[derive(Clone, Default)]
struct RecordedPhases(Arc<std::sync::Mutex<Vec<(&'static str, tokio::time::Instant)>>>);

impl RecordedPhases {
    fn record(&self, phase: &'static str) {
        self.0
            .lock()
            .unwrap()
            .push((phase, tokio::time::Instant::now()));
    }

    fn phases(&self) -> Vec<(&'static str, tokio::time::Instant)> {
        self.0.lock().unwrap().clone()
    }
}

impl ShutdownPhases for RecordedPhases {
    fn begin_prestop(&self) {
        self.record("prestop");
    }

    fn begin_drain(&self) {
        self.record("drain");
    }

    fn end_streams(&self) {
        self.record("end_streams");
    }
}

#[tokio::test]
async fn http_shutdown_finishes_without_waiting_for_a_signal_when_server_exits() {
    let phases = RecordedPhases::default();
    let result = serve_with_shutdown_plan(
        |_signal| async { "server-exited" },
        phases.clone(),
        std::future::pending::<()>(),
        test_shutdown_plan(),
    )
    .await;
    assert_eq!(result, Some("server-exited"));
    assert!(phases.phases().is_empty());
}

/// The plan's phases run in order at their times, and connections that
/// never close are dropped at the end of the budget.
#[tokio::test(start_paused = true)]
async fn http_shutdown_plan_runs_its_phases_and_bounds_stuck_connections() {
    let phases = RecordedPhases::default();
    let plan = ShutdownPlan {
        prestop: Duration::from_secs(5),
        budget: Duration::from_secs(25),
    };
    let started = tokio::time::Instant::now();
    let result = serve_with_shutdown_plan(
        |signal| async move {
            signal.await;
            std::future::pending::<()>().await
        },
        phases.clone(),
        std::future::ready(()),
        plan,
    )
    .await;
    assert_eq!(result, None);
    let at = |phase: &str| {
        phases
            .phases()
            .into_iter()
            .find(|(name, _)| *name == phase)
            .map(|(_, at)| at - started)
            .unwrap_or_else(|| panic!("{phase} did not run"))
    };
    assert_eq!(at("prestop"), Duration::ZERO);
    assert_eq!(at("drain"), plan.prestop);
    assert_eq!(at("end_streams"), plan.prestop + plan.stream_notice());
    assert_eq!(plan.stream_notice(), Duration::from_secs(18));
    assert_eq!(started.elapsed(), plan.budget);
}

/// The local role follows the gateway's plan: during the pre-stop window
/// readiness is unready while the runtime still admits work, then the
/// drain closes admission and the server stops within its budget.
#[tokio::test]
async fn local_shutdown_serves_through_prestop_then_drains() {
    let _env = env_lock();
    let (state, serve_config, _temp_dir) = local_test_state();
    let app = api::create_router(state.clone(), &serve_config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let base = format!("http://{}", listener.local_addr().expect("address"));
    let (stop, stopped) = oneshot::channel::<()>();
    let plan = ShutdownPlan {
        prestop: Duration::from_millis(600),
        budget: Duration::from_millis(1600),
    };
    let server = tokio::spawn(serve_local(
        state.clone(),
        listener,
        app,
        async { Vec::new() },
        async move {
            let _ = stopped.await;
        },
        plan,
    ));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client");
    tokio::time::timeout(Duration::from_secs(10), async {
        while !lifecycle_ready(&client, &base).await.1 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("ready once preload finishes");

    stop.send(()).expect("server is running");
    let signalled = Instant::now();
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(lifecycle_ready(&client, &base).await, (503, false));
    let live = client.get(format!("{base}/livez")).send().await.unwrap();
    assert_eq!(live.status().as_u16(), 200);
    assert!(!state.runtime.coordinator_snapshot().draining);
    assert!(!state.lifecycle.snapshot().draining);

    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("the server stops within its budget")
        .expect("server task")
        .expect("server result");
    assert!(
        signalled.elapsed() >= plan.prestop,
        "{:?}",
        signalled.elapsed()
    );
    assert!(state.runtime.coordinator_snapshot().draining);
    assert!(state.lifecycle.snapshot().draining);
}

fn clear_bind_env() {
    std::env::remove_var("IZWI_HOST");
    std::env::remove_var("IZWI_PORT");
    std::env::remove_var("IZWI_BACKEND");
    std::env::remove_var("IZWI_LOG_FORMAT");
    std::env::remove_var("IZWI_MAX_BATCH_SIZE");
    std::env::remove_var("IZWI_MAX_SCHEDULER_BATCH_SIZE");
    std::env::remove_var("IZWI_MAX_RETAINED_SEQUENCES");
    std::env::remove_var("IZWI_MAX_STAGED_TRANSACTIONS");
    std::env::remove_var("IZWI_MAX_QUEUED_REQUESTS");
    std::env::remove_var("IZWI_PHYSICAL_EXECUTION_MODE");
    std::env::remove_var("IZWI_MAX_PHYSICAL_IN_FLIGHT");
    std::env::remove_var("IZWI_NUM_THREADS");
    std::env::remove_var("IZWI_MAX_CONCURRENT");
    std::env::remove_var("IZWI_TIMEOUT");
    std::env::remove_var("IZWI_CORS");
    std::env::remove_var("IZWI_CORS_ORIGINS");
    std::env::remove_var("IZWI_NO_UI");
    std::env::remove_var("IZWI_UI_DIR");
    std::env::remove_var("MAX_CONCURRENT_REQUESTS");
    std::env::remove_var("REQUEST_TIMEOUT_SECS");
    std::env::remove_var("IZWI_PRELOAD_MODELS");
    std::env::remove_var("IZWI_WARMUP_PRELOADED_MODELS");
    std::env::remove_var("IZWI_ASR_WARMUP_DURATION_MS");
    std::env::remove_var("IZWI_GRANITE_DECODE_PROFILE");
    std::env::remove_var("IZWI_GRANITE_SPEECH_DTYPE");
    std::env::remove_var("IZWI_BATCH_STAGE_TIMEOUT_SECS");
    std::env::remove_var("IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS");
    std::env::remove_var("IZWI_HTTP_SHUTDOWN_GRACE_SECS");
    std::env::remove_var("IZWI_GATEWAY_WORKER_ADMISSION_TIMEOUT_MS");
    std::env::remove_var("IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_MS");
    std::env::remove_var("IZWI_GATEWAY_WORKER_PROGRESS_IDLE_TIMEOUT_MS");
    std::env::remove_var("IZWI_GATEWAY_SLOW_CONSUMER_TIMEOUT_MS");
}

fn parse(args: &[&str]) -> ServerArgs {
    let mut parsed = ServerArgs::try_parse_from(args).expect("arguments should parse");
    if parsed.config.is_none() {
        // Existing defaults/env tests must not depend on the developer's
        // real persisted configuration. This temporary directory is dropped
        // before resolution, selecting the normal missing-file defaults.
        parsed.config = Some(tempfile::tempdir().unwrap().path().join("absent.toml"));
    }
    parsed
}

#[test]
fn local_server_role_remains_the_default() {
    // Defaults read the environment, which other tests set under the lock.
    let args = {
        let _guard = env_lock();
        parse(&["izwi-server"])
    };
    assert_eq!(args.role, ServerRole::Local);
    assert_eq!(args.gateway.gateway_topology, GatewayTopology::Standalone);
    validate_role_topology(args.role, args.gateway.gateway_topology)
        .expect("local standalone role");
    assert!(
        validate_role_topology(ServerRole::Local, GatewayTopology::FleetOneGateway)
            .unwrap_err()
            .to_string()
            .contains("requires --role gateway")
    );
}

#[test]
fn configured_preload_models_parses_csv_env() {
    let _guard = env_lock();
    std::env::set_var(
        "IZWI_PRELOAD_MODELS",
        " Whisper-Large-v3-Turbo, Qwen3.5-4B, ,invalid ",
    );
    let models = configured_preload_models();
    assert_eq!(
        models,
        vec![
            "Whisper-Large-v3-Turbo".to_string(),
            "Qwen3.5-4B".to_string(),
            "invalid".to_string()
        ]
    );
    clear_bind_env();
}

#[test]
fn asr_warmup_duration_uses_env_and_clamps() {
    let _guard = env_lock();
    clear_bind_env();

    std::env::set_var("IZWI_ASR_WARMUP_DURATION_MS", "42");
    assert_eq!(asr_warmup_duration_ms(), 100);

    std::env::set_var("IZWI_ASR_WARMUP_DURATION_MS", "1200");
    assert_eq!(asr_warmup_duration_ms(), 1200);

    std::env::set_var("IZWI_ASR_WARMUP_DURATION_MS", "99999");
    assert_eq!(asr_warmup_duration_ms(), 5000);
    clear_bind_env();
}

#[test]
fn warmup_flag_defaults_enabled_and_honors_env() {
    let _guard = env_lock();
    clear_bind_env();
    assert!(warmup_preloaded_models_enabled());

    std::env::set_var("IZWI_WARMUP_PRELOADED_MODELS", "0");
    assert!(!warmup_preloaded_models_enabled());

    std::env::set_var("IZWI_WARMUP_PRELOADED_MODELS", "true");
    assert!(warmup_preloaded_models_enabled());
    clear_bind_env();
}

#[test]
fn backend_flag_overrides_environment() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_BACKEND", "cpu");

    let args = parse(&["izwi-server", "--backend", "cuda"]);
    let resolved = resolve_serve_runtime_config(&args).unwrap();

    assert_eq!(
        resolved.backend,
        izwi_core::backends::BackendPreference::Cuda
    );
    clear_bind_env();
}

#[test]
fn invalid_backend_value_is_rejected() {
    let result = ServerArgs::try_parse_from(["izwi-server", "--backend", "invalid"]);
    assert!(
        result.is_err(),
        "invalid backend should fail argument parsing"
    );
}

#[test]
fn physical_execution_flags_override_environment() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_PHYSICAL_EXECUTION_MODE", "concurrent");
    std::env::set_var("IZWI_MAX_PHYSICAL_IN_FLIGHT", "4");

    let args = parse(&[
        "izwi-server",
        "--physical-execution-mode",
        "shadow",
        "--max-physical-in-flight",
        "3",
    ]);
    let resolved = resolve_serve_runtime_config(&args).unwrap();

    assert_eq!(
        resolved.physical_execution_mode,
        izwi_core::PhysicalExecutionMode::Shadow
    );
    assert_eq!(resolved.max_physical_in_flight.get(), 3);
    clear_bind_env();
}

#[test]
fn granite_decode_profile_flag_defaults_off_and_parses() {
    let _guard = env_lock();
    clear_bind_env();

    assert!(!parse(&["izwi-server"]).granite_decode_profile);
    assert!(parse(&["izwi-server", "--granite-decode-profile"]).granite_decode_profile);
    clear_bind_env();
}

#[test]
fn granite_speech_dtype_flag_defaults_empty_and_parses() {
    let _guard = env_lock();
    clear_bind_env();

    assert!(parse(&["izwi-server"]).granite_speech_dtype.is_none());
    assert_eq!(
        parse(&["izwi-server", "--granite-speech-dtype", "f16"])
            .granite_speech_dtype
            .as_deref(),
        Some("f16")
    );
    clear_bind_env();
}

#[test]
fn log_format_defaults_to_text() {
    let _guard = env_lock();
    clear_bind_env();

    let args = parse(&["izwi-server"]);

    assert_eq!(args.log_format, LogFormat::Text);
    clear_bind_env();
}

#[test]
fn log_format_accepts_cli_and_environment() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_LOG_FORMAT", "json");

    let env_args = parse(&["izwi-server"]);
    let cli_args = parse(&["izwi-server", "--log-format", "text"]);

    assert_eq!(env_args.log_format, LogFormat::Json);
    assert_eq!(cli_args.log_format, LogFormat::Text);
    clear_bind_env();
}

#[test]
fn invalid_log_format_value_is_rejected() {
    let result = ServerArgs::try_parse_from(["izwi-server", "--log-format", "ndjson"]);
    assert!(
        result.is_err(),
        "invalid log format should fail argument parsing"
    );
}

#[test]
fn cli_values_override_environment() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_HOST", "0.0.0.0");
    std::env::set_var("IZWI_PORT", "8080");

    let resolved = resolve_serve_runtime_config(&parse(&[
        "izwi-server",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
    ]))
    .unwrap();

    assert_eq!(resolved.host, "127.0.0.1");
    assert_eq!(resolved.port, 9000);
    clear_bind_env();
}

#[test]
fn uses_environment_when_cli_values_missing() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_HOST", "127.0.0.1");
    std::env::set_var("IZWI_PORT", "8088");

    let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"])).unwrap();

    assert_eq!(resolved.host, "127.0.0.1");
    assert_eq!(resolved.port, 8088);
    clear_bind_env();
}

#[test]
fn falls_back_to_defaults_without_cli_or_environment() {
    let _guard = env_lock();
    clear_bind_env();

    let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"])).unwrap();

    assert_eq!(resolved.host, "127.0.0.1");
    assert_eq!(resolved.port, 8080);
    assert_eq!(
        resolved.max_batch_size,
        izwi_core::BatchSizePreference::Auto
    );
    assert_eq!(
        resolved.physical_execution_mode,
        izwi_core::PhysicalExecutionMode::Serial
    );
    assert_eq!(resolved.max_physical_in_flight.get(), 1);
    assert!(resolved.num_threads >= 1);
    clear_bind_env();
}

#[test]
fn falls_back_to_default_when_env_port_is_invalid() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_PORT", "not-a-port");

    let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"])).unwrap();

    assert_eq!(resolved.port, 8080);
    clear_bind_env();
}

#[test]
fn canonical_runtime_env_values_flow_into_serve_config() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_MAX_BATCH_SIZE", "16");
    std::env::set_var("IZWI_NUM_THREADS", "6");
    std::env::set_var("IZWI_MAX_CONCURRENT", "44");
    std::env::set_var("IZWI_TIMEOUT", "720");

    let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"])).unwrap();

    assert_eq!(resolved.max_batch_size.fixed_rows(), Some(16));
    assert_eq!(resolved.num_threads, 6);
    assert_eq!(resolved.max_concurrent_requests, 44);
    assert_eq!(resolved.request_timeout_secs, 720);
    clear_bind_env();
}

#[test]
fn legacy_runtime_env_aliases_are_still_supported() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("MAX_CONCURRENT_REQUESTS", "45");
    std::env::set_var("REQUEST_TIMEOUT_SECS", "721");

    let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"])).unwrap();

    assert_eq!(resolved.max_concurrent_requests, 45);
    assert_eq!(resolved.request_timeout_secs, 721);
    clear_bind_env();
}

#[test]
fn ui_and_cors_env_values_flow_into_serve_config() {
    let _guard = env_lock();
    clear_bind_env();
    std::env::set_var("IZWI_CORS", "1");
    std::env::set_var(
        "IZWI_CORS_ORIGINS",
        "http://localhost:3000,https://example.com",
    );
    std::env::set_var("IZWI_NO_UI", "1");
    std::env::set_var("IZWI_UI_DIR", "/tmp/izwi-ui");

    let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"])).unwrap();

    assert!(resolved.cors_enabled);
    assert_eq!(
        resolved.cors_origins,
        vec![
            "http://localhost:3000".to_string(),
            "https://example.com".to_string()
        ]
    );
    assert!(!resolved.ui_enabled);
    assert_eq!(resolved.ui_dir, std::path::PathBuf::from("/tmp/izwi-ui"));
    clear_bind_env();
}

#[test]
fn performance_startup_merges_file_inherited_environment_and_cli() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, "[runtime.performance.cuda]\nmode='off'\nmtp_draft_tokens=3\n[ runtime.performance.loading ]\ncache_max_bytes=1234\nworkers=2").unwrap();
    let args = ServerArgs::try_parse_from([
        "izwi-server",
        "--config",
        path.to_str().unwrap(),
        "--performance",
        "cuda.mode=off",
        "--performance",
        "cuda.mtp=auto",
        "--performance",
        "cuda.mtp_adaptive=false",
        "--performance",
        "loading.workers=0",
    ])
    .unwrap();
    let inherited = ServeRuntimeConfigOverrides {
        performance: izwi_core::PerformanceConfigOverrides::from_lookup(|key| match key {
            "IZWI_CUDA_MODE" => Some("auto".into()),
            "IZWI_CUDA_MTP_ADAPTIVE" => Some("true".into()),
            "IZWI_LOADING_WORKERS" => Some("6".into()),
            _ => None,
        })
        .unwrap(),
        ..Default::default()
    };
    let runtime = resolve_serve_runtime_config_with_env(&args, &inherited).unwrap();
    let config = runtime.engine_config().performance.resolve_env().unwrap();
    assert_eq!(config.cuda.mode, izwi_core::OptimizationMode::Off);
    assert!(!config.cuda.mtp_adaptive);
    assert_eq!(config.cuda.mtp_draft_tokens, 3);
    assert_eq!(config.loading.workers, 0);
    assert_eq!(config.loading.cache_max_bytes, 1234);
    assert!(!config.normalized().cuda.mtp.enabled());
}

#[test]
fn performance_startup_reads_persisted_opt_out_without_cli_flags() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, "[runtime.performance.cuda]\nmode='off'\nmtp_adaptive=false\n[ runtime.performance.loading ]\nmode='off'").unwrap();
    let args = ServerArgs::try_parse_from(["izwi-server", "--config", path.to_str().unwrap()])
        .unwrap();
    let config = resolve_serve_runtime_config_with_env(&args, &Default::default()).unwrap();
    assert!(!config.performance.cuda.enabled());
    assert!(!config.performance.cuda.mtp_adaptive);
    assert!(!config.performance.loading.enabled());
}

#[test]
fn performance_startup_rejects_malformed_config_and_cli_values() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, "[runtime.performance.cuda]\nmtp_draft_tokens=4").unwrap();
    let args = ServerArgs::try_parse_from(["izwi-server", "--config", path.to_str().unwrap()])
        .unwrap();
    assert!(resolve_serve_runtime_config_with_env(&args, &Default::default()).is_err());
    for value in [
        "cuda.mode=maybe",
        "cuda.mtp_draft_tokens=4",
        "cuda.mtp_adaptive=maybe",
        "loading.max_staging_bytes=0",
        "not_an_assignment",
    ] {
        assert!(
            ServerArgs::try_parse_from(["izwi-server", "--performance", value]).is_err(),
            "{value}"
        );
    }
    let missing = directory.path().join("not-created.toml");
    assert!(
        izwi_core::PerformanceConfigOverrides::from_user_config(Some(&missing))
            .unwrap()
            .is_empty()
    );
}

