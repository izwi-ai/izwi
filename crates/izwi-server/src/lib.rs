//! Izwi TTS Server - HTTP API for Qwen3-TTS inference

// HTTP orchestration boundaries intentionally carry the complete request/job
// context, and realtime alignment uses explicit word-coordinate indexing.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// Async tests hold a process-wide environment lock across awaits so parallel
// tests cannot observe transient environment overrides. Production code keeps
// the stricter lint enabled.
#![cfg_attr(test, allow(clippy::await_holding_lock))]

use anyhow::Context;
use clap::{Parser, ValueEnum};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::oneshot;
use tracing::{info, warn};

const DESKTOP_OWNER_PIPE_ENV: &str = "IZWI_DESKTOP_OWNER_PIPE";

mod api;
mod app;
pub use app::realtime_protocol;
pub mod batch_runtime;
mod chat_store;
mod db;
mod diarization_store;
mod entity;
mod error;
mod gateway;
mod ids;
mod logging;
pub mod media_ingest;
mod onboarding_store;
mod persistence;
mod saved_voice_store;
mod speech_history_store;
mod speech_resource_budget;
mod state;
mod storage_layout;
mod studio_project_store;
#[cfg(test)]
mod test_support;
mod transcription_store;
mod voice_defaults;
mod voice_memory;
mod voice_observation_store;
mod voice_store;

use batch_runtime::types::{
    DeviceClass, QueueClass, ResourceTarget, RuntimeBackendClass, WorkerResourceCapacity,
};
use batch_runtime::worker::{
    BatchWorkerConfig, BatchWorkerDrain, BatchWorkerRunner, BatchWorkerSupervisor,
};
use izwi_core::backends::{self, BackendKind, BackendPreference, CudaRuntimeDiagnostics};
use izwi_core::{
    parse_model_variant, RuntimeService, ServeRuntimeConfig, ServeRuntimeConfigOverrides,
};
use izwi_hooks::EnterpriseHooks;
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{
    CredentialId, DeploymentId, IncarnationId, ModelGeneration, PolicyRevision, ServiceBearerToken,
    ServiceCredentials,
};
use logging::{LogFormat, SERVICE_NAME, SERVICE_VERSION};
use persistence::PersistenceContext;
use state::AppState;

pub use app::chat::{RemoteChatExecution, RemoteChatExecutionConfig};
pub use gateway::{create_gateway_router, GatewayState};

#[derive(Debug, Parser)]
#[command(
    name = "izwi-server",
    about = "HTTP API server for Izwi local inference",
    version = env!("CARGO_PKG_VERSION")
)]
struct ServerArgs {
    /// Configuration file (defaults to the shared Izwi user config.toml).
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Process role: local inference server or hardware-independent gateway.
    #[arg(long, value_enum, env = "IZWI_SERVER_ROLE", default_value = "local")]
    role: ServerRole,

    /// Override a performance setting, e.g. cuda.mode=off; repeat for siblings.
    #[arg(long = "performance", value_name = "KEY=VALUE", value_parser = parse_performance_override)]
    performance: Vec<izwi_core::PerformanceConfigOverrides>,

    /// Host to bind to
    #[arg(short = 'H', long)]
    host: Option<String>,

    /// Port to listen on
    #[arg(short, long)]
    port: Option<u16>,

    /// Backend preference (`auto`, `cpu`, `metal`, `cuda`)
    #[arg(long, value_enum, env = "IZWI_BACKEND")]
    backend: Option<BackendArg>,

    /// Physical launch rollout mode (`serial`, `shadow`, `concurrent`)
    #[arg(long, value_name = "MODE")]
    physical_execution_mode: Option<izwi_core::PhysicalExecutionMode>,

    /// Maximum candidate physical launches in flight
    #[arg(long, value_name = "COUNT")]
    max_physical_in_flight: Option<izwi_core::PhysicalInFlightLimit>,

    /// Portable context length (`auto` or a positive token count)
    #[arg(long, value_name = "AUTO_OR_TOKENS")]
    max_sequence_length: Option<izwi_core::ContextLengthPreference>,

    /// Log output format (`text`, `json`)
    #[arg(long, value_enum, env = "IZWI_LOG_FORMAT", default_value = "text")]
    log_format: LogFormat,

    /// Enable Granite ASR decode-profile diagnostics after backend selection.
    #[arg(long)]
    granite_decode_profile: bool,

    /// Override Granite ASR dtype after backend selection (`f32`, `f16`, `bf16`).
    #[arg(long, value_name = "DTYPE")]
    granite_speech_dtype: Option<String>,

    /// Private worker base URL required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_ENDPOINT")]
    worker_endpoint: Option<String>,

    /// Rotatable private worker credential identifier required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_CREDENTIAL_ID")]
    worker_credential_id: Option<String>,

    /// Private worker bearer token required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_BEARER_TOKEN")]
    worker_bearer_token: Option<String>,

    /// Expected worker process incarnation required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_INCARNATION")]
    worker_incarnation: Option<String>,

    /// Pinned worker deployment identifier required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_DEPLOYMENT")]
    worker_deployment: Option<String>,

    /// Public model served by the pinned worker deployment.
    #[arg(long, env = "IZWI_GATEWAY_PUBLIC_MODEL")]
    public_model: Option<String>,

    /// Expected non-zero model generation required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_MODEL_GENERATION")]
    worker_model_generation: Option<u64>,

    /// Policy revision attested on private worker requests.
    #[arg(
        long,
        env = "IZWI_GATEWAY_POLICY_REVISION",
        default_value = "local-policy-v1"
    )]
    gateway_policy_revision: String,

    /// Maximum gateway requests concurrently in flight to the pinned worker.
    #[arg(long, env = "IZWI_GATEWAY_MAX_IN_FLIGHT", default_value_t = 32)]
    gateway_max_in_flight: usize,

    /// Maximum time the selected worker may spend establishing runtime ownership.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_QUEUE_WAIT_MS",
        default_value_t = 250
    )]
    gateway_worker_queue_wait_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ServerRole {
    Local,
    Gateway,
}

#[derive(Debug, Clone, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl BackendArg {
    fn as_preference(&self) -> izwi_core::backends::BackendPreference {
        match self {
            Self::Auto => izwi_core::backends::BackendPreference::Auto,
            Self::Cpu => izwi_core::backends::BackendPreference::Cpu,
            Self::Metal => izwi_core::backends::BackendPreference::Metal,
            Self::Cuda => izwi_core::backends::BackendPreference::Cuda,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BindConfig {
    host: String,
    port: u16,
}

pub async fn run_from_cli(enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    let args = ServerArgs::parse();
    run_with_args(args, enterprise_hooks).await
}

async fn run_with_args(args: ServerArgs, enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    let serve_config = resolve_serve_runtime_config(&args)?;
    if args.role == ServerRole::Gateway {
        return run_gateway(args, serve_config, enterprise_hooks).await;
    }
    maybe_delegate_to_private_cuda_runtime(&serve_config)?;

    logging::init_tracing(args.log_format);

    info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        log_format = args.log_format.as_str(),
        "Starting Izwi TTS Server"
    );

    let effective_runtime_config = serde_json::to_string(&serve_config)
        .context("failed to serialize effective server runtime configuration")?;
    info!(
        build_git_sha = option_env!("IZWI_BUILD_GIT_SHA").unwrap_or("unknown"),
        effective_runtime_config, "Resolved effective server runtime configuration"
    );
    let config = serve_config.engine_config();
    info!("Models directory: {:?}", config.models_dir);

    // Create runtime service
    let runtime = RuntimeService::new(config)?;
    if args.granite_decode_profile {
        std::env::set_var("IZWI_GRANITE_DECODE_PROFILE", "1");
        info!("Granite ASR decode profiling enabled");
    }
    if let Some(dtype) = args
        .granite_speech_dtype
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        std::env::set_var("IZWI_GRANITE_SPEECH_DTYPE", dtype);
        info!(dtype, "Granite ASR dtype override enabled");
    }
    let persistence = PersistenceContext::resolve(&enterprise_hooks).await?;
    info!(
        database_backend = ?persistence.database.backend(),
        database_migration_mode = ?persistence.database.migration_mode(),
        database_metadata_keys = persistence.database.metadata().len(),
        "Persistence database resolved"
    );
    let state = AppState::with_enterprise_hooks_and_persistence(
        runtime,
        &serve_config,
        enterprise_hooks,
        persistence,
    )?;
    let mut startup_warnings = Vec::new();
    if let Err(err) = state
        .batch_runtime_store
        .reconcile_inconsistent_states()
        .await
    {
        startup_warnings.push(format!(
            "Failed to reconcile durable runtime jobs during startup: {err}"
        ));
    }
    match state
        .speech_history_store
        .reconcile_stale_processing_records()
        .await
    {
        Ok(reconciled) if reconciled > 0 => {
            info!(reconciled, "Reconciled stale speech history records");
        }
        Ok(_) => {}
        Err(err) => startup_warnings.push(format!(
            "Failed to reconcile speech history records during startup: {err}"
        )),
    }
    startup_warnings.extend(preload_configured_models(&state).await);
    startup_warnings.extend(warmup_preloaded_asr_models(&state).await);
    if !startup_warnings.is_empty() {
        state
            .lifecycle
            .record_startup_warnings(startup_warnings.clone());
        for warning in startup_warnings {
            warn!(warning = %warning, "Startup readiness warning");
        }
    }
    state.lifecycle.mark_ready();

    info!("Runtime service initialized");
    let batch_worker_supervisor = start_batch_runtime_worker(&state);
    let batch_worker_drain = batch_worker_supervisor.drain_handle();

    // Build router
    let app = api::create_router(state.clone(), &serve_config);

    // Start server
    let bind = BindConfig {
        host: serve_config.host.clone(),
        port: serve_config.port,
    };
    let addr = format!("{}:{}", bind.host, bind.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Server listening on http://{}", addr);

    // Clone state for shutdown handler
    let shutdown_state = state.clone();
    let (shutdown_started_tx, shutdown_started_rx) = oneshot::channel();

    // Spawn server with graceful shutdown
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(
        shutdown_state,
        batch_worker_drain,
        shutdown_started_tx,
    ));

    info!("Server ready. Press Ctrl+C to stop.");
    let http_shutdown_grace = http_shutdown_grace_timeout();
    let server_result = await_http_server_shutdown(
        async move { server.await },
        shutdown_started_rx,
        http_shutdown_grace,
    )
    .await;
    if server_result.is_none() {
        warn!(
            grace_secs = http_shutdown_grace.as_secs(),
            "HTTP graceful shutdown timed out; dropping remaining connections"
        );
    }
    shutdown_worker_then_cleanup(
        batch_worker_supervisor.shutdown(),
        cleanup_runtime_for_shutdown(&state),
    )
    .await?;
    if let Some(server_result) = server_result {
        server_result?;
    }

    Ok(())
}

async fn run_gateway(
    args: ServerArgs,
    serve_config: ServeRuntimeConfig,
    enterprise_hooks: EnterpriseHooks,
) -> anyhow::Result<()> {
    logging::init_tracing(args.log_format);
    let remote = gateway_remote_execution(&args, &serve_config)?;
    let state = gateway::GatewayState::new(
        remote,
        enterprise_hooks,
        serve_config.request_timeout_secs,
        args.gateway_max_in_flight,
    );
    state.lifecycle.mark_ready();

    info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        "Starting Izwi public API gateway"
    );
    let app = gateway::create_gateway_router(state.clone(), &serve_config);
    let addr = format!("{}:{}", serve_config.host, serve_config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Gateway listening on http://{}", addr);

    let shutdown_state = state.clone();
    let (shutdown_started_tx, shutdown_started_rx) = oneshot::channel();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(gateway_shutdown_signal(shutdown_state, shutdown_started_tx));
    let http_shutdown_grace = http_shutdown_grace_timeout();
    let server_result = await_http_server_shutdown(
        async move { server.await },
        shutdown_started_rx,
        http_shutdown_grace,
    )
    .await;
    if server_result.is_none() {
        warn!(
            grace_secs = http_shutdown_grace.as_secs(),
            "Gateway HTTP graceful shutdown timed out; dropping remaining connections"
        );
    }
    if let Some(server_result) = server_result {
        server_result?;
    }
    Ok(())
}

fn gateway_remote_execution(
    args: &ServerArgs,
    serve_config: &ServeRuntimeConfig,
) -> anyhow::Result<app::chat::RemoteChatExecution> {
    fn required<'a>(value: &'a Option<String>, name: &str) -> anyhow::Result<&'a str> {
        value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("gateway mode requires {name}"))
    }

    if args.gateway_max_in_flight == 0
        || args.gateway_max_in_flight > tokio::sync::Semaphore::MAX_PERMITS
    {
        anyhow::bail!(
            "--gateway-max-in-flight must be between 1 and {}",
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }
    if args.gateway_worker_queue_wait_ms == 0 {
        anyhow::bail!("--gateway-worker-queue-wait-ms must be non-zero");
    }
    let model_generation = ModelGeneration::new(
        args.worker_model_generation
            .ok_or_else(|| anyhow::anyhow!("gateway mode requires --worker-model-generation"))?,
    )?;
    let model = parse_model_variant(required(&args.public_model, "--public-model")?)?;
    let credentials = ServiceCredentials {
        credential_id: CredentialId::new(required(
            &args.worker_credential_id,
            "--worker-credential-id",
        )?)?,
        bearer_token: ServiceBearerToken::new(required(
            &args.worker_bearer_token,
            "--worker-bearer-token or IZWI_GATEWAY_WORKER_BEARER_TOKEN",
        )?)?,
    };
    let client = WorkerClient::new(
        required(&args.worker_endpoint, "--worker-endpoint")?,
        credentials,
        WorkerClientConfig {
            max_in_flight: args.gateway_max_in_flight,
            request_timeout: Duration::from_secs(serve_config.request_timeout_secs.max(1)),
            progress_timeout: Duration::from_secs(serve_config.request_timeout_secs.max(1)),
            ..WorkerClientConfig::default()
        },
    )?;
    app::chat::RemoteChatExecution::new(
        client,
        app::chat::RemoteChatExecutionConfig {
            public_model_variant: model,
            expected_worker_incarnation: IncarnationId::new(required(
                &args.worker_incarnation,
                "--worker-incarnation",
            )?)?,
            deployment_id: DeploymentId::new(required(
                &args.worker_deployment,
                "--worker-deployment",
            )?)?,
            expected_model_generation: model_generation,
            policy_revision: PolicyRevision::new(args.gateway_policy_revision.trim())?,
            max_queue_wait: Duration::from_millis(args.gateway_worker_queue_wait_ms),
            max_output_tokens: 4096,
            // Leave room for the private event envelope within the worker's
            // default one-MiB encoded-event bound.
            max_output_bytes: 512 * 1024,
        },
    )
    .map_err(|error| anyhow::anyhow!(error.message))
}

fn start_batch_runtime_worker(state: &AppState) -> BatchWorkerSupervisor {
    let mut config =
        BatchWorkerConfig::local(format!("local-batch-worker-{}", crate::ids::new_uuid()));
    config.queue_names = local_batch_worker_queue_names();
    config.capabilities = vec!["asr".to_string(), "tts".to_string()];
    config.stage_kinds = vec![
        api::transcription::BATCH_ASR_STAGE_KIND.to_string(),
        api::speech_history::BATCH_TTS_STAGE_KIND.to_string(),
    ];
    let backend_context = state.runtime.backend_context();
    config.resources = local_batch_worker_resources(
        backend_context.backend_kind,
        backend_context.device.capabilities.available_memory_bytes,
    );
    // Durable requests feed the shared engine; this bounds admitted workers,
    // independently of the physical tensor batch chosen by that engine.
    config.resources.concurrency_slots = batch_worker_concurrency(
        state
            .runtime
            .config()
            .max_retained_sequences
            .min(state.runtime.config().max_queued_requests),
        std::env::var("IZWI_BATCH_WORKER_CONCURRENCY")
            .ok()
            .as_deref(),
    );
    config.execution_timeout = batch_stage_execution_timeout();
    config.drain_timeout = batch_worker_drain_timeout();
    BatchWorkerRunner::new(
        state.batch_runtime_store.clone(),
        vec![
            api::transcription::batch_asr_stage_executor(state.clone()),
            api::speech_history::batch_tts_stage_executor(state.clone()),
        ],
        config,
        state.batch_worker_health.clone(),
    )
    .with_runtime_observer(state.runtime.clone())
    .spawn()
}

fn local_batch_worker_queue_names() -> Vec<String> {
    [
        QueueClass::BatchAsr,
        QueueClass::LongFormAsr,
        QueueClass::BatchTts,
    ]
    .into_iter()
    .map(|queue| queue.as_db_value().to_string())
    .collect()
}

fn local_batch_worker_resources(
    backend: BackendKind,
    available_memory_bytes: Option<usize>,
) -> WorkerResourceCapacity {
    let (target, backend, device_class) = match backend {
        BackendKind::Cpu => (
            ResourceTarget::Cpu,
            RuntimeBackendClass::Cpu,
            DeviceClass::Cpu,
        ),
        BackendKind::Metal => (
            ResourceTarget::Gpu,
            RuntimeBackendClass::Metal,
            DeviceClass::AppleGpu,
        ),
        BackendKind::Cuda => (
            ResourceTarget::Gpu,
            RuntimeBackendClass::Cuda,
            DeviceClass::NvidiaGpu,
        ),
    };
    WorkerResourceCapacity {
        targets: vec![target],
        backends: vec![backend],
        device_classes: vec![device_class],
        memory_bytes: available_memory_bytes.and_then(|bytes| u64::try_from(bytes).ok()),
        concurrency_slots: 1,
        ..WorkerResourceCapacity::default()
    }
}

fn batch_worker_concurrency(runtime_capacity: usize, configured: Option<&str>) -> u32 {
    let capacity = u32::try_from(runtime_capacity).unwrap_or(u32::MAX).max(1);
    match configured {
        Some(value) => match value.parse::<u32>() {
            Ok(limit) if limit > 0 => limit.min(capacity),
            _ => {
                tracing::warn!(
                    value,
                    "Invalid IZWI_BATCH_WORKER_CONCURRENCY; using runtime capacity"
                );
                capacity
            }
        },
        None => capacity,
    }
}

fn batch_stage_execution_timeout() -> Option<Duration> {
    duration_secs_from_env("IZWI_BATCH_STAGE_TIMEOUT_SECS", 0, 30, 86_400)
}

fn batch_worker_drain_timeout() -> Duration {
    duration_secs_from_env("IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS", 20, 1, 300)
        .unwrap_or_else(|| Duration::from_secs(20))
}

fn http_shutdown_grace_timeout() -> Duration {
    duration_secs_from_env("IZWI_HTTP_SHUTDOWN_GRACE_SECS", 20, 1, 300)
        .unwrap_or_else(|| Duration::from_secs(20))
}

fn duration_secs_from_env(
    name: &str,
    default_secs: u64,
    min_secs: u64,
    max_secs: u64,
) -> Option<Duration> {
    let configured = std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default_secs);
    (configured > 0).then(|| Duration::from_secs(configured.clamp(min_secs, max_secs)))
}

fn maybe_delegate_to_private_cuda_runtime(serve_config: &ServeRuntimeConfig) -> anyhow::Result<()> {
    if cfg!(feature = "cuda") || backends::private_cuda_runtime_active() {
        return Ok(());
    }

    if !matches!(
        serve_config.backend,
        BackendPreference::Auto | BackendPreference::Cuda
    ) {
        return Ok(());
    }

    let binary_name = current_server_binary_name();
    let diagnostics = CudaRuntimeDiagnostics::detect(&binary_name);
    if diagnostics.can_start_private_runtime() {
        let runtime_path = diagnostics
            .private_runtime_path
            .as_ref()
            .expect("can_start_private_runtime requires a private runtime path");
        return exec_private_cuda_runtime(runtime_path);
    }

    if serve_config.backend == BackendPreference::Cuda {
        anyhow::bail!("{}", format_cuda_runtime_unavailable(&diagnostics));
    }

    Ok(())
}

fn exec_private_cuda_runtime(runtime_path: &Path) -> anyhow::Result<()> {
    let mut command = Command::new(runtime_path);
    command.args(std::env::args_os().skip(1));
    command.env(backends::private_cuda_runtime_env_key(), "1");
    backends::prepend_cuda_loader_paths(&mut command, runtime_path);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        let err = command.exec();
        return Err(anyhow::anyhow!(
            "failed to exec private CUDA runtime {}: {}",
            runtime_path.display(),
            err
        ));
    }

    #[cfg(windows)]
    {
        let status = command.status().map_err(|err| {
            anyhow::anyhow!(
                "failed to start private CUDA runtime {}: {}",
                runtime_path.display(),
                err
            )
        })?;
        std::process::exit(status.code().unwrap_or(1));
    }

    #[allow(unreachable_code)]
    Ok(())
}

fn current_server_binary_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| {
            if cfg!(windows) {
                "izwi-server.exe".to_string()
            } else {
                "izwi-server".to_string()
            }
        })
}

fn format_cuda_runtime_unavailable(diagnostics: &CudaRuntimeDiagnostics) -> String {
    let mut reasons = Vec::new();

    if !diagnostics.private_runtime_packaged {
        reasons.push("private CUDA runtime binary is not packaged".to_string());
    }
    if !diagnostics.runtime_libraries_available {
        if diagnostics.missing_runtime_libraries.is_empty() {
            reasons.push("CUDA runtime libraries are not available".to_string());
        } else {
            reasons.push(format!(
                "missing CUDA runtime libraries: {}",
                diagnostics.missing_runtime_libraries.join(", ")
            ));
        }
    }
    if !diagnostics.driver_available {
        reasons.push("NVIDIA driver library is not available".to_string());
    }

    if reasons.is_empty() {
        reasons.push("CUDA runtime could not be selected".to_string());
    }

    format!(
        "CUDA backend was requested, but the packaged CUDA runtime is unavailable ({})",
        reasons.join("; ")
    )
}

fn parse_performance_override(
    value: &str,
) -> izwi_core::Result<izwi_core::PerformanceConfigOverrides> {
    let (key, value) = value
        .split_once('=')
        .ok_or_else(|| izwi_core::Error::ConfigError("expected performance KEY=VALUE".into()))?;
    let mut overrides = izwi_core::PerformanceConfigOverrides::default();
    overrides.set_value(
        key.trim()
            .strip_prefix("runtime.performance.")
            .unwrap_or(key.trim()),
        value,
    )?;
    Ok(overrides)
}

fn resolve_serve_runtime_config(args: &ServerArgs) -> anyhow::Result<ServeRuntimeConfig> {
    resolve_serve_runtime_config_with_env(args, &ServeRuntimeConfigOverrides::from_env())
}

fn resolve_serve_runtime_config_with_env(
    args: &ServerArgs,
    env: &ServeRuntimeConfigOverrides,
) -> anyhow::Result<ServeRuntimeConfig> {
    let cli = ServeRuntimeConfigOverrides {
        host: args.host.clone(),
        port: args.port,
        backend: args.backend.as_ref().map(BackendArg::as_preference),
        physical_execution_mode: args.physical_execution_mode,
        max_physical_in_flight: args.max_physical_in_flight,
        max_sequence_length: args.max_sequence_length,
        ..ServeRuntimeConfigOverrides::default()
    };
    let file = ServeRuntimeConfigOverrides {
        performance: izwi_core::PerformanceConfigOverrides::from_user_config(
            args.config.as_deref(),
        )?,
        ..Default::default()
    };
    let mut runtime = ServeRuntimeConfig::from_sources(&file, env, &cli);
    for performance in &args.performance {
        runtime.performance.apply_overrides(performance);
    }
    runtime.performance.validate()?;
    Ok(runtime)
}

fn configured_preload_models() -> Vec<String> {
    std::env::var("IZWI_PRELOAD_MODELS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|raw| {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn env_u32(key: &str) -> Option<u32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
}

fn warmup_preloaded_models_enabled() -> bool {
    env_bool("IZWI_WARMUP_PRELOADED_MODELS").unwrap_or(true)
}

fn asr_warmup_duration_ms() -> u32 {
    env_u32("IZWI_ASR_WARMUP_DURATION_MS")
        .unwrap_or(800)
        .clamp(100, 5_000)
}

fn build_asr_warmup_wav(sample_rate: u32, duration_ms: u32) -> anyhow::Result<Vec<u8>> {
    let sample_rate = sample_rate.max(8_000);
    let total_samples = ((sample_rate as u64 * duration_ms as u64) / 1000).max(1) as usize;
    let freq_hz = 440.0f32;
    let amplitude = 0.12f32;

    let mut wav_bytes = Vec::new();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    {
        let cursor = Cursor::new(&mut wav_bytes);
        let mut writer = hound::WavWriter::new(cursor, spec)?;
        for idx in 0..total_samples {
            let t = idx as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * amplitude;
            let quantized = (sample * i16::MAX as f32) as i16;
            writer.write_sample(quantized)?;
        }
        writer.finalize()?;
    }

    Ok(wav_bytes)
}

async fn preload_configured_models(state: &AppState) -> Vec<String> {
    let mut warnings = Vec::new();
    let configured = configured_preload_models();
    if configured.is_empty() {
        return warnings;
    }

    info!(
        count = configured.len(),
        "Preloading models from IZWI_PRELOAD_MODELS"
    );

    for model_id in configured {
        match parse_model_variant(&model_id) {
            Ok(variant) => match state.runtime.load_model(variant).await {
                Ok(()) => {
                    info!(model = %variant, "Preloaded model");
                }
                Err(err) => {
                    warnings.push(format!("failed to preload model {model_id}: {err}"));
                    warn!(model_id = %model_id, "Failed to preload model: {err}");
                }
            },
            Err(err) => {
                warnings.push(format!("unknown preload model {model_id}: {err}"));
                warn!(model_id = %model_id, "Skipping unknown preload model id: {err}");
            }
        }
    }

    warnings
}

async fn warmup_preloaded_asr_models(state: &AppState) -> Vec<String> {
    let mut warnings = Vec::new();
    if !warmup_preloaded_models_enabled() {
        return warnings;
    }

    let configured = configured_preload_models();
    if configured.is_empty() {
        return warnings;
    }

    let duration_ms = asr_warmup_duration_ms();
    let warmup_wav = match build_asr_warmup_wav(16_000, duration_ms) {
        Ok(bytes) => bytes,
        Err(err) => {
            warnings.push(format!("failed to build ASR warmup WAV bytes: {err}"));
            warn!("Failed to build ASR warmup WAV bytes: {err}");
            return warnings;
        }
    };

    info!(
        count = configured.len(),
        duration_ms, "Running ASR warmup pass for preloaded models"
    );

    for model_id in configured {
        match parse_model_variant(&model_id) {
            Ok(variant) => {
                if !variant.is_asr() {
                    continue;
                }
                match state
                    .runtime
                    .asr_transcribe_bytes(&warmup_wav, Some(&model_id), Some("en"))
                    .await
                {
                    Ok(_) => info!(model = %model_id, "ASR warmup completed"),
                    Err(err) => {
                        warnings.push(format!("ASR warmup failed for {model_id}: {err}"));
                        warn!(model_id = %model_id, "ASR warmup failed: {err}");
                    }
                }
            }
            Err(err) => {
                warnings.push(format!("unknown warmup model {model_id}: {err}"));
                warn!(model_id = %model_id, "Skipping unknown warmup model id: {err}");
            }
        }
    }

    warnings
}

/// Wait for shutdown signal and cleanup
async fn shutdown_signal(
    state: AppState,
    batch_worker_drain: BatchWorkerDrain,
    shutdown_started: oneshot::Sender<()>,
) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    let desktop_owner_exit = desktop_owner_exit_signal();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            info!("Received SIGTERM, shutting down...");
        },
        _ = desktop_owner_exit => {
            info!("Desktop owner pipe closed, shutting down...");
        },
    }

    state.lifecycle.mark_draining();
    state.runtime.begin_drain();
    batch_worker_drain.begin();
    let _ = shutdown_started.send(());

    drop(state);
}

async fn gateway_shutdown_signal(
    state: gateway::GatewayState,
    shutdown_started: oneshot::Sender<()>,
) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, shutting down gateway..."),
        _ = terminate => info!("Received SIGTERM, shutting down gateway..."),
        _ = desktop_owner_exit_signal() => info!("Desktop owner pipe closed, shutting down gateway..."),
    }

    state.lifecycle.mark_draining();
    let _ = shutdown_started.send(());
}

async fn desktop_owner_exit_signal() {
    if std::env::var_os(DESKTOP_OWNER_PIPE_ENV).as_deref() != Some(std::ffi::OsStr::new("1")) {
        std::future::pending::<()>().await;
        return;
    }

    if let Err(err) = tokio::task::spawn_blocking(|| {
        let stdin = std::io::stdin();
        wait_for_owner_pipe_close(stdin.lock())
    })
    .await
    {
        warn!("Desktop owner-pipe monitor failed: {err}");
    }
}

fn wait_for_owner_pipe_close(mut reader: impl Read) {
    let mut buffer = [0_u8; 1];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

async fn await_http_server_shutdown<F>(
    server: F,
    shutdown_started: oneshot::Receiver<()>,
    grace: Duration,
) -> Option<F::Output>
where
    F: std::future::Future,
{
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => Some(result),
        started = shutdown_started => {
            if started.is_err() {
                return Some(server.await);
            }
            tokio::time::timeout(grace, &mut server).await.ok()
        }
    }
}

async fn cleanup_runtime_for_shutdown(state: &AppState) {
    const CLEANUP_TIMEOUT: Duration = Duration::from_secs(20);
    let started = Instant::now();
    if let Err(err) = state.runtime.wait_for_drain(CLEANUP_TIMEOUT).await {
        let snapshot = state.runtime.coordinator_snapshot();
        warn!(
            active_jobs = snapshot.active_jobs,
            active_executions = snapshot.active_executions,
            "Runtime drain failed: {err}; skipping model unload"
        );
        return;
    }
    let remaining = CLEANUP_TIMEOUT.saturating_sub(started.elapsed());
    match tokio::time::timeout(remaining, state.runtime.unload_all_models()).await {
        Ok(Ok(unloaded)) => {
            info!(
                "Runtime shutdown cleanup completed; unloaded {} model(s)",
                unloaded
            );
        }
        Ok(Err(err)) => {
            warn!("Runtime shutdown cleanup failed: {}", err);
        }
        Err(_) => {
            warn!(
                "Runtime shutdown cleanup timed out after {}s; continuing shutdown",
                CLEANUP_TIMEOUT.as_secs()
            );
        }
    }
}

async fn shutdown_worker_then_cleanup<W, C>(worker_shutdown: W, cleanup: C) -> anyhow::Result<()>
where
    W: std::future::Future<Output = anyhow::Result<()>>,
    C: std::future::Future<Output = ()>,
{
    let worker_result = worker_shutdown.await;
    cleanup.await;
    worker_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use izwi_core::ModelVariant;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

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
    async fn worker_shutdown_failure_still_runs_runtime_cleanup() {
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

    #[tokio::test]
    async fn http_shutdown_finishes_without_waiting_for_a_signal_when_server_exits() {
        let (_shutdown_tx, shutdown_rx) = oneshot::channel();
        let result = await_http_server_shutdown(
            async { "server-exited" },
            shutdown_rx,
            Duration::from_secs(1),
        )
        .await;
        assert_eq!(result, Some("server-exited"));
    }

    #[tokio::test]
    async fn http_shutdown_grace_period_bounds_stuck_connections() {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        shutdown_tx.send(()).expect("shutdown receiver is alive");
        let result = await_http_server_shutdown(
            std::future::pending::<()>(),
            shutdown_rx,
            Duration::from_millis(1),
        )
        .await;
        assert_eq!(result, None);
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
        assert_eq!(parse(&["izwi-server"]).role, ServerRole::Local);
    }

    #[test]
    fn gateway_configuration_builds_without_a_runtime_service() {
        let args = parse(&[
            "izwi-server",
            "--role",
            "gateway",
            "--worker-endpoint",
            "http://127.0.0.1:19091",
            "--worker-credential-id",
            "gateway-test-credential",
            "--worker-bearer-token",
            "gateway-test-secret",
            "--worker-incarnation",
            "worker-incarnation-1",
            "--worker-deployment",
            "qwen3-chat-v1",
            "--public-model",
            ModelVariant::Qwen34BGguf.dir_name(),
            "--worker-model-generation",
            "7",
        ]);
        let serve_config =
            resolve_serve_runtime_config_with_env(&args, &ServeRuntimeConfigOverrides::default())
                .expect("serve configuration should resolve");

        let remote = gateway_remote_execution(&args, &serve_config)
            .expect("gateway transport configuration should build");
        assert_eq!(args.role, ServerRole::Gateway);
        assert_eq!(
            remote.config().public_model_variant,
            ModelVariant::Qwen34BGguf
        );
        assert_eq!(remote.config().expected_model_generation.get(), 7);
        assert_eq!(remote.config().max_queue_wait, Duration::from_millis(250));
    }

    #[test]
    fn gateway_configuration_rejects_missing_private_credentials() {
        let args = parse(&[
            "izwi-server",
            "--role",
            "gateway",
            "--worker-endpoint",
            "http://127.0.0.1:19091",
        ]);
        let error = gateway_remote_execution(&args, &ServeRuntimeConfig::default())
            .expect_err("gateway configuration must be complete");
        assert!(error.to_string().contains("--worker-model-generation"));
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

        assert_eq!(resolved.host, "0.0.0.0");
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
}
