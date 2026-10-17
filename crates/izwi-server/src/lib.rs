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
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

const DESKTOP_OWNER_PIPE_ENV: &str = "IZWI_DESKTOP_OWNER_PIPE";

mod api;
mod app;
pub use app::realtime_protocol;
pub mod artifact_store;
pub mod batch_runtime;
mod chat_store;
mod db;
mod diarization_store;
mod entity;
mod error;
mod gateway;
mod gateway_fleet;
mod gateway_principal_keys;
mod gateway_security;
mod ids;
mod local_security;
mod logging;
pub mod media_ingest;
mod onboarding_store;
mod persistence;
mod saved_voice_store;
mod speech_history_store;
mod speech_resource_budget;
mod speech_spool;
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

use batch_runtime::store::DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT;
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
// The gateway's accelerator-free components live in izwi-gateway; these
// aliases keep their `crate::` paths for the gateway role still hosted here.
use izwi_gateway::{
    gateway_deployments, gateway_rate_quota, gateway_shared_approvals, gateway_tenant_concurrency,
    gateway_worker_tls,
};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{
    render_approvals_text, CredentialId, DeploymentId, ModelAlias, ModelGeneration, ModelReadiness,
    NdjsonLimits, NodeId, PolicyRevision, ServiceBearerToken, ServiceCredentials, TaskKind,
    WorkerDescriptor, WorkerId, WorkerStatus, MAX_REMAINING_TIME_MS,
};
use logging::{LogFormat, SERVICE_NAME, SERVICE_VERSION};
use persistence::PersistenceContext;
use state::AppState;

pub use app::remote_chat_dispatch::{RemoteChatDispatchConfig, RemoteChatDispatcher};
pub use gateway::{create_gateway_router, GatewayState};
pub use gateway_fleet::{FleetPartition, FleetPartitionError, FleetStore};
pub use gateway_rate_quota::{GatewayRateQuotaConfig, GatewayRateQuotaConfigError};
pub use gateway_security::{GatewayPerimeterConfig, GatewayPerimeterConfigError};
pub use gateway_tenant_concurrency::{
    GatewayTenantConcurrencyConfig, GatewayTenantConcurrencyConfigError,
};
pub use izwi_gateway::worker_registry;

/// The local role's HTTP API over an already started runtime, without the
/// listener, preload and persistence lifecycle of `izwi serve`. Integration
/// tests in other crates use it to run a request through the local server
/// and through a gateway and compare the answers.
#[doc(hidden)]
pub fn create_local_router(
    runtime: RuntimeService,
    serve_config: &ServeRuntimeConfig,
) -> anyhow::Result<axum::Router> {
    let state = AppState::new(runtime, serve_config)?;
    Ok(api::create_router(state, serve_config))
}

const MAX_CONFIGURED_GATEWAY_WORKERS: usize = 256;
const MAX_GATEWAY_STATUS_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_GATEWAY_ADMISSION_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_GATEWAY_STREAM_PHASE_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_GATEWAY_SHUTDOWN_BUDGET_SECS: u64 = 600;
const GATEWAY_NDJSON_MAX_LINE_BYTES: usize = 1024 * 1024;
const GATEWAY_NDJSON_MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
const GATEWAY_NDJSON_MAX_EVENTS: usize = 8192;

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

    /// Host to bind to (default 127.0.0.1). A non-loopback bind on the local
    /// role requires IZWI_API_KEY unless --allow-unauthenticated-network.
    #[arg(short = 'H', long)]
    host: Option<String>,

    /// Serve a non-loopback local-role bind without an API key. Anyone who
    /// can reach the port can then manage models and read history.
    #[arg(long, env = "IZWI_ALLOW_UNAUTHENTICATED_NETWORK")]
    allow_unauthenticated_network: bool,

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

    /// Legacy one-worker form of `--gateway-worker-endpoint`: the gateway
    /// routes to this worker through the same registry as an endpoint list.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_ENDPOINT")]
    worker_endpoint: Option<String>,

    /// Worker-network policy (`standalone` or `fleet-one-gateway`).
    #[arg(
        long,
        value_enum,
        env = "IZWI_GATEWAY_TOPOLOGY",
        default_value = "standalone"
    )]
    gateway_topology: GatewayTopology,

    /// Approved private worker URLs for registry-backed routing. Repeat this
    /// option (or use a comma-separated environment value) to add capacity.
    #[arg(
        long = "gateway-worker-endpoint",
        env = "IZWI_GATEWAY_WORKER_ENDPOINTS",
        value_delimiter = ',',
        value_name = "URL"
    )]
    gateway_worker_endpoints: Vec<String>,

    /// Statically approved worker routing entries. Repeat this option (or use
    /// a comma-separated environment value) with the standalone five-field
    /// form or v1|URL|NODE_ID|WORKER_ID|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|
    /// MODEL_GENERATION for fleet mode. This cannot be combined with the
    /// legacy gateway worker endpoint list.
    #[arg(
        long = "gateway-worker-approval",
        env = "IZWI_GATEWAY_WORKER_APPROVALS",
        value_delimiter = ',',
        value_name = "APPROVAL"
    )]
    gateway_worker_approvals: Vec<gateway_deployments::GatewayWorkerApproval>,

    /// Rotatable private worker credential identifier required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_CREDENTIAL_ID")]
    worker_credential_id: Option<String>,

    /// Private worker bearer token required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_BEARER_TOKEN")]
    worker_bearer_token: Option<String>,

    /// Accepted with `--worker-endpoint` for compatibility and ignored: the
    /// registry fences every attempt to the incarnation it observed and
    /// adopts a restarted worker.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_INCARNATION")]
    worker_incarnation: Option<String>,

    /// Chat deployment identifier of the legacy worker endpoint forms.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_DEPLOYMENT")]
    worker_deployment: Option<String>,

    /// Public model served through the gateway.
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
    #[arg(long, env = "IZWI_GATEWAY_WORKER_QUEUE_WAIT_MS", default_value_t = 250)]
    gateway_worker_queue_wait_ms: u64,

    /// Maximum wait for private invocation response headers and admission.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_ADMISSION_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    gateway_worker_admission_timeout_ms: u64,

    /// Maximum wait after admission for the first useful worker output.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_MS",
        default_value_t = 60_000
    )]
    gateway_worker_first_output_timeout_ms: u64,

    /// Extra first-output allowance per KiB of prompt, so a long prompt's
    /// prefill (slow on CPU) is not cancelled by the short-prompt deadline.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_PER_KIB_MS",
        default_value_t = 1_000
    )]
    gateway_worker_first_output_timeout_per_kib_ms: u64,

    /// Maximum idle time between useful worker output events.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_PROGRESS_IDLE_TIMEOUT_MS",
        default_value_t = 30_000
    )]
    gateway_worker_progress_idle_timeout_ms: u64,

    /// Total gateway shutdown budget after SIGTERM: the pre-stop window,
    /// then draining in-flight requests. The default fits Kubernetes'
    /// default 30 s termination grace period.
    #[arg(long, env = "IZWI_GATEWAY_SHUTDOWN_BUDGET_SECS", default_value_t = 25)]
    gateway_shutdown_budget_secs: u64,

    /// Pre-stop window at the start of the shutdown budget: readiness turns
    /// unready while requests are still served, so load balancers stop
    /// routing here before the listener closes.
    #[arg(long, env = "IZWI_GATEWAY_PRESTOP_SECS", default_value_t = 5)]
    gateway_prestop_secs: u64,

    /// Deadline for receiving a public request body. A slow upload holds a
    /// global admission permit at most this long.
    #[arg(
        long,
        env = "IZWI_GATEWAY_BODY_READ_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    gateway_body_read_timeout_ms: u64,

    /// Maximum wait while relaying an event to a connected slow consumer.
    #[arg(
        long,
        env = "IZWI_GATEWAY_SLOW_CONSUMER_TIMEOUT_MS",
        default_value_t = 5_000
    )]
    gateway_slow_consumer_timeout_ms: u64,

    /// Receiver-clock lifetime of a worker status observation.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_STATUS_TTL_MS",
        default_value_t = 10_000
    )]
    gateway_worker_status_ttl_ms: u64,

    /// Status refresh interval for each configured registry worker.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_STATUS_POLL_MS",
        default_value_t = 2_000
    )]
    gateway_worker_status_poll_ms: u64,

    /// Prefix-aware routing: prefer the worker recently routed the longest
    /// leading part of a chat prompt, within the balance threshold below.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_AFFINITY",
        default_value = "on"
    )]
    gateway_router_prefix_affinity: String,

    /// Bytes of message stream per hashed prefix block.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_BLOCK_BYTES",
        default_value_t = 256
    )]
    gateway_router_prefix_block_bytes: usize,

    /// Leading prefix blocks considered per prompt.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_MAX_BLOCKS",
        default_value_t = 128
    )]
    gateway_router_prefix_max_blocks: usize,

    /// Prefix block hashes remembered per worker (LRU).
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_ENTRIES",
        default_value_t = 16_384
    )]
    gateway_router_prefix_entries: usize,

    /// Outstanding requests a prefix-holding worker may carry beyond the
    /// least-loaded worker and still be preferred.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_BALANCE_THRESHOLD",
        default_value_t = 4
    )]
    gateway_router_balance_threshold: u32,

    /// Managed-KV utilization percent at which a worker is avoided while any
    /// eligible worker has headroom (0-100].
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_KV_PRESSURE_PCT",
        default_value_t = 90.0
    )]
    gateway_router_kv_pressure_pct: f64,

    /// Realtime relay (DS3.6): expose /v1/realtime/ws and forward
    /// izwi-realtime-v1 sessions to eligible speech_to_text workers.
    /// `off` keeps the gateway byte-identical to the pre-realtime shape.
    #[arg(long, env = "IZWI_GATEWAY_REALTIME", default_value = "off")]
    gateway_realtime: String,

    /// Bounded concurrent realtime sessions through the relay.
    #[arg(long, env = "IZWI_GATEWAY_REALTIME_MAX_SESSIONS", default_value_t = 64)]
    gateway_realtime_max_sessions: usize,

    /// End-to-end budget minted into each relayed worker session (ms).
    #[arg(
        long,
        env = "IZWI_GATEWAY_REALTIME_SESSION_BUDGET_MS",
        default_value_t = 600_000
    )]
    gateway_realtime_session_budget_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ServerRole {
    Local,
    Gateway,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GatewayTopology {
    Standalone,
    FleetOneGateway,
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
    validate_role_topology(args.role, args.gateway_topology)?;
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

    let api_key = local_security::LocalApiKey::from_env()?;
    local_security::validate_local_bind(
        &serve_config.host,
        api_key.as_ref(),
        args.allow_unauthenticated_network,
    )?;
    if api_key.is_some() {
        info!("Local API key required on /v1 and /internal routes");
    } else if !gateway_security::is_loopback_host(&serve_config.host) {
        warn!(
            host = %serve_config.host,
            "SECURITY WARNING: serving a non-loopback address WITHOUT authentication \
             (--allow-unauthenticated-network). Anyone who can reach this port can download, \
             load and delete models and read or delete history. Set IZWI_API_KEY instead."
        );
    }
    let enterprise_hooks = local_security::with_api_key(enterprise_hooks, api_key);

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
    state.runtime.spawn_idle_model_reaper();
    // Resolve precise Hugging Face download sizes off the request path so the
    // model-list endpoint stays instant on cold starts; the UI picks refined
    // sizes up on its next refresh.
    state.runtime.model_manager().spawn_expected_size_resolution();
    let mut startup_warnings = Vec::new();
    if let Err(err) = state
        .batch_runtime_store
        .reconcile_inconsistent_states(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
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
    match state
        .artifact_store
        .cleanup_due(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
        .await
    {
        Ok(report) if report.completed > 0 => {
            info!(
                completed = report.completed,
                deferred = report.deferred,
                "Reconciled orphaned provider write reservations and artifact cleanup intents"
            );
        }
        Ok(_) => {}
        Err(err) => startup_warnings.push(format!(
            "Failed to reconcile orphaned artifact cleanup state during startup: {err}"
        )),
    }
    for warning in &startup_warnings {
        warn!(warning = %warning, "Startup readiness warning");
    }
    state.lifecycle.record_startup_warnings(startup_warnings);

    // Build router
    let app = api::create_router(state.clone(), &serve_config);

    // Bind before preloading models: liveness answers during long loads, and
    // readiness reports the "initializing" phase until preload finishes.
    let bind = BindConfig {
        host: serve_config.host.clone(),
        port: serve_config.port,
    };
    let addr = format!("{}:{}", bind.host, bind.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Server listening on http://{}", listener.local_addr()?);

    let preload_state = state.clone();
    serve_local(
        state,
        listener,
        app,
        async move {
            let mut preload_failures = preload_configured_models(&preload_state).await;
            preload_failures.extend(warmup_preloaded_asr_models(&preload_state).await);
            preload_failures
        },
        shutdown_requested(),
    )
    .await
}

/// Serve the local role on an already bound listener while `preload` loads
/// the configured models. The server turns ready, and the batch worker
/// starts, only once preload finishes; a shutdown that begins first
/// abandons the preload.
async fn serve_local<P, S>(
    state: AppState,
    listener: tokio::net::TcpListener,
    app: axum::Router,
    preload: P,
    shutdown_requested: S,
) -> anyhow::Result<()>
where
    P: std::future::Future<Output = Vec<String>> + Send + 'static,
    S: std::future::Future<Output = ()> + Send + 'static,
{
    let batch_worker_drain = Arc::new(std::sync::OnceLock::new());
    let startup = tokio::spawn(complete_local_startup(
        state.clone(),
        preload,
        batch_worker_drain.clone(),
    ));

    // Clone state for shutdown handler
    let shutdown_state = state.clone();
    let (shutdown_started_tx, shutdown_started_rx) = oneshot::channel();

    // Spawn server with graceful shutdown
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(
        shutdown_requested,
        shutdown_state,
        batch_worker_drain,
        shutdown_started_tx,
    ));

    info!("Server accepting connections. Press Ctrl+C to stop.");
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
    // A preload still running when shutdown began is abandoned; a finished
    // startup hands over its batch worker.
    startup.abort();
    let batch_worker_supervisor = startup.await.ok().flatten();
    shutdown_worker_then_cleanup(
        async move {
            match batch_worker_supervisor {
                Some(supervisor) => supervisor.shutdown_for_process().await,
                None => Ok(()),
            }
        },
        cleanup_runtime_for_shutdown(&state),
    )
    .await?;
    if let Some(server_result) = server_result {
        server_result?;
    }

    Ok(())
}

/// Finish local startup after the listener is bound: record preload
/// failures (they keep readiness failed), mark the server ready and start the
/// batch worker. Nothing starts once shutdown has begun.
async fn complete_local_startup<P>(
    state: AppState,
    preload: P,
    batch_worker_drain: Arc<std::sync::OnceLock<BatchWorkerDrain>>,
) -> Option<BatchWorkerSupervisor>
where
    P: std::future::Future<Output = Vec<String>>,
{
    let preload_failures = preload.await;
    for warning in &preload_failures {
        warn!(warning = %warning, "Startup readiness warning");
    }
    state.lifecycle.record_preload_failures(preload_failures);
    if state.lifecycle.snapshot().draining {
        return None;
    }
    state.lifecycle.mark_ready();
    info!("Runtime service initialized");
    let supervisor = start_batch_runtime_worker(&state);
    let _ = batch_worker_drain.set(supervisor.drain_handle());
    // Shutdown marks the lifecycle draining before it reads the drain slot,
    // and this reads the lifecycle after filling it, so one side always sees
    // the other and the worker cannot miss the drain.
    if state.lifecycle.snapshot().draining {
        supervisor.drain_handle().begin();
    }
    Some(supervisor)
}

fn validate_role_topology(role: ServerRole, topology: GatewayTopology) -> anyhow::Result<()> {
    if role == ServerRole::Local && topology != GatewayTopology::Standalone {
        anyhow::bail!("non-standalone gateway topology requires --role gateway");
    }
    Ok(())
}

/// Apply a fleet partition to the per-gateway quota budgets.
///
/// When `partition` is `Some`, each per-tenant rate and concurrency limit is
/// divided by the fleet size (floored to 1), so N gateways each own a strict
/// non-overlapping slice of the total configured budget. No shared atomic
/// counter or coordination service is required. A crashed gateway releases
/// its partition immediately; the other gateways' partitions are unaffected.
fn apply_fleet_partition(
    rate_quota: GatewayRateQuotaConfig,
    tenant_concurrency: GatewayTenantConcurrencyConfig,
    partition: Option<crate::gateway_fleet::FleetPartition>,
) -> (GatewayRateQuotaConfig, GatewayTenantConcurrencyConfig) {
    let Some(partition) = partition else {
        return (rate_quota, tenant_concurrency);
    };
    let requests = partition.partition_limit(rate_quota.requests_per_minute());
    let burst = partition.partition_limit(rate_quota.burst_requests());
    let tracked = rate_quota.max_tracked_tenants();
    let tokens = rate_quota
        .tokens_per_minute()
        .map(|tokens| partition.partition_limit(tokens));
    let rate_quota = GatewayRateQuotaConfig::new(requests, burst, tracked)
        .and_then(|config| config.with_tokens_per_minute(tokens))
        .expect("partitioned rate-quota must remain valid");
    let per_tenant = u32::try_from(tenant_concurrency.max_active_per_tenant())
        .ok()
        .map(|value| partition.partition_concurrency(value))
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1);
    let owned = u32::try_from(tenant_concurrency.max_owned_work())
        .ok()
        .map(|value| partition.partition_concurrency(value))
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1);
    let tenant_concurrency =
        GatewayTenantConcurrencyConfig::new(per_tenant.max(1).min(owned.max(1)), owned.max(1))
            .expect("partitioned tenant concurrency must remain valid");
    (rate_quota, tenant_concurrency)
}

async fn run_gateway(
    args: ServerArgs,
    serve_config: ServeRuntimeConfig,
    enterprise_hooks: EnterpriseHooks,
) -> anyhow::Result<()> {
    logging::init_tracing(args.log_format);
    let perimeter = GatewayPerimeterConfig::from_env()?;
    let fleet_partition = crate::gateway_fleet::FleetPartition::from_env()?;
    if let Some(partition) = fleet_partition {
        info!(
            service = SERVICE_NAME,
            version = SERVICE_VERSION,
            partition = partition.index(),
            fleet_size = partition.size(),
            "Gateway fleet partition active: per-tenant budgets are divided across gateways (explicit quota fallback; the worker stays the atomic admission arbiter)"
        );
    }
    if let Ok(Some(shared)) = crate::gateway_shared_approvals::SharedApprovalsConfig::from_env() {
        match crate::gateway_shared_approvals::SharedApprovalsView::load(shared) {
            Ok(view) => info!(
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                approvals = view.approvals().len(),
                "Shared fleet approvals loaded: all gateways reading this file approve the same worker set"
            ),
            Err(error) => warn!(error = %error, "Shared fleet approvals file could not be loaded at startup"),
        }
    }
    let rate_quota = GatewayRateQuotaConfig::from_env()?;
    let tenant_concurrency = GatewayTenantConcurrencyConfig::from_env(args.gateway_max_in_flight)?;
    let (rate_quota, tenant_concurrency) =
        apply_fleet_partition(rate_quota, tenant_concurrency, fleet_partition);
    perimeter.validate_public_ingress(&serve_config)?;
    // DS0.5 scoped credentials: the manifest is the only activation switch.
    // Unset keeps the durable store untouched and authentication byte-identical.
    let principal_keys = match crate::gateway_principal_keys::manifest_path_from_env()? {
        Some(manifest_path) => {
            let store = crate::db::sqlite::StoreDatabase::from_default_path()?;
            let directory = crate::gateway_principal_keys::GatewayPrincipalDirectory::bootstrap(
                &store,
                &manifest_path,
                &perimeter,
            )
            .await?;
            info!(
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                principals = directory.len(),
                "Gateway scoped principal keys loaded from the durable store"
            );
            directory
        }
        None => crate::gateway_principal_keys::GatewayPrincipalDirectory::empty(),
    };
    let (state, _status_poller) =
        gateway_state(&args, &serve_config, enterprise_hooks, perimeter).await?;
    let state = state
        .with_rate_quota_config(rate_quota)
        .with_tenant_concurrency_config(tenant_concurrency)
        .with_principal_keys(principal_keys)
        .with_body_read_timeout(Duration::from_millis(args.gateway_body_read_timeout_ms));
    state.lifecycle.mark_ready();

    info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        "Starting Izwi public API gateway"
    );
    let app = gateway::create_gateway_router(state.clone(), &serve_config);
    let addr = format!("{}:{}", serve_config.host, serve_config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Gateway listening on http://{}", listener.local_addr()?);

    let plan = GatewayShutdownPlan {
        prestop: Duration::from_secs(args.gateway_prestop_secs),
        budget: Duration::from_secs(args.gateway_shutdown_budget_secs),
    };
    serve_gateway(listener, app, state, gateway_shutdown_requested(), plan).await
}

/// How a gateway spends its shutdown budget: a pre-stop window in which
/// readiness is unready but requests are served, then a drain of in-flight
/// requests. Streams still open near the end of the drain are ended with a
/// terminal event, so no client is cut off without one.
#[derive(Debug, Clone, Copy)]
struct GatewayShutdownPlan {
    prestop: Duration,
    budget: Duration,
}

impl GatewayShutdownPlan {
    fn drain(&self) -> Duration {
        self.budget.saturating_sub(self.prestop)
    }

    /// When, into the drain, open streams are told to end: late enough for
    /// most to finish on their own, early enough to deliver the event.
    fn stream_notice(&self) -> Duration {
        let drain = self.drain();
        drain.saturating_sub((drain / 4).min(Duration::from_secs(2)))
    }
}

async fn serve_gateway<S>(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    state: gateway::GatewayState,
    shutdown_requested: S,
    plan: GatewayShutdownPlan,
) -> anyhow::Result<()>
where
    S: std::future::Future<Output = ()> + Send + 'static,
{
    let signal_state = state.clone();
    let (drain_started_tx, drain_started_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        shutdown_requested.await;
        signal_state.begin_prestop();
        info!(
            prestop_secs = plan.prestop.as_secs_f64(),
            budget_secs = plan.budget.as_secs_f64(),
            "Gateway pre-stop: readiness is unready, requests are still served"
        );
        tokio::time::sleep(plan.prestop).await;
        signal_state.begin_drain();
        let _ = drain_started_tx.send(());
    });
    let server = async move { server.await };
    tokio::pin!(server);
    let finished = tokio::select! {
        result = &mut server => Some(result),
        started = drain_started_rx => {
            if started.is_err() {
                Some(server.await)
            } else {
                match tokio::time::timeout(plan.stream_notice(), &mut server).await {
                    Ok(result) => Some(result),
                    Err(_) => {
                        state.end_streams();
                        let rest = plan.drain().saturating_sub(plan.stream_notice());
                        tokio::time::timeout(rest, &mut server).await.ok()
                    }
                }
            }
        }
    };
    match finished {
        Some(result) => result?,
        None => warn!(
            budget_secs = plan.budget.as_secs(),
            "Gateway shutdown budget spent; dropping remaining connections"
        ),
    }
    Ok(())
}

struct GatewayWorkerStatusPoller {
    tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Boot-time poller count. Runtime adoption mutates the live poller pool;
    /// this snapshot pins the boot-sized pool for diagnostics and tests.
    poller_count: usize,
}

#[derive(Clone)]
struct GatewayWorkerExpectation {
    client: WorkerClient,
    worker_id: WorkerId,
    node_id: NodeId,
    deployment: worker_registry::ApprovedDeployment,
    validated_capacity: u32,
}

impl Drop for GatewayWorkerStatusPoller {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic per-worker poll cadence: ±10% of the configured interval,
/// drawn from the worker identity so co-configured pollers de-synchronize
/// (serving plan §5.2 "status every 2 seconds with jitter") while the exact
/// cadence stays reproducible under test.
fn jittered_poll_interval(base: Duration, seed: u64) -> Duration {
    let mut state = seed | 1;
    let draw = splitmix64(&mut state);
    let per_mille = 90 + (draw % 21);
    let millis = base.as_millis().saturating_mul(u128::from(per_mille)) / 100;
    Duration::from_millis(millis as u64).max(Duration::from_millis(1))
}

fn poll_jitter_seed(worker_id: &WorkerId, node_id: &NodeId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    worker_id.hash(&mut hasher);
    node_id.hash(&mut hasher);
    hasher.finish()
}

/// The widest jittered cadence (110% of the poll interval) must stay below the
/// freshness TTL, or a healthy worker would flap expired between observations.
fn validate_status_cadence(ttl: Duration, poll: Duration) -> anyhow::Result<()> {
    if poll.is_zero() {
        anyhow::bail!("--gateway-worker-status-poll-ms must be non-zero");
    }
    let widest = poll.as_millis().saturating_mul(11) / 10;
    if widest >= ttl.as_millis() {
        anyhow::bail!(
            "--gateway-worker-status-poll-ms must be at least 10% below the status TTL to leave room for poll jitter"
        );
    }
    Ok(())
}

async fn gateway_state(
    args: &ServerArgs,
    serve_config: &ServeRuntimeConfig,
    enterprise_hooks: EnterpriseHooks,
    perimeter: GatewayPerimeterConfig,
) -> anyhow::Result<(gateway::GatewayState, GatewayWorkerStatusPoller)> {
    validate_gateway_topology_source(args)?;
    if args.worker_endpoint.is_some()
        && !(args.gateway_worker_endpoints.is_empty() && args.gateway_worker_approvals.is_empty())
    {
        anyhow::bail!(
            "--worker-endpoint cannot be combined with registry worker endpoints or approvals; it is the one-worker form of --gateway-worker-endpoint"
        );
    }
    if args.worker_incarnation.is_some() {
        if args.worker_endpoint.is_none() {
            anyhow::bail!(
                "--worker-incarnation applies only to the legacy --worker-endpoint form; registry routing validates each discovered incarnation"
            );
        }
        warn!(
            service = SERVICE_NAME,
            version = SERVICE_VERSION,
            "--worker-incarnation is ignored: the gateway fences every attempt to the worker incarnation it observed and adopts a restarted worker"
        );
    }

    validate_gateway_limits(args)?;
    let public_model_variant = parse_model_variant(required_gateway_value(
        &args.public_model,
        "--public-model",
    )?)?;
    let public_model = ModelAlias::new(public_model_variant.dir_name())?;
    let (cli_approvals, worker_approvals, shared_approvals_view) =
        configured_gateway_worker_approvals(args, &public_model)?;
    if worker_approvals.is_empty() && shared_approvals_view.is_none() {
        anyhow::bail!(
            "gateway mode requires a worker: --gateway-worker-approval, --gateway-worker-endpoint, --worker-endpoint, or a shared approvals file"
        );
    }
    let credentials = gateway_worker_credentials(args)?;
    // With a shared approvals file, runtime view adoption (DS6) can add
    // workers beyond the boot view; the registry capacity follows the
    // approvals ceiling. Without one, the boot view is the exact capacity.
    let shared_capacity = shared_approvals_view.is_some();
    let registry_config = worker_registry::WorkerRegistryConfig {
        max_workers: if shared_capacity {
            worker_approvals
                .len()
                .max(izwi_serving_protocol::MAX_APPROVALS_FILE_ENTRIES)
        } else {
            worker_approvals.len()
        },
        max_deployments_per_worker: 32,
        max_local_dispatches: args.gateway_max_in_flight,
        status_ttl: Duration::from_millis(args.gateway_worker_status_ttl_ms),
        routing: worker_registry::RoutingConfig {
            prefix_balance_threshold: args.gateway_router_balance_threshold,
            kv_pressure_pct: args.gateway_router_kv_pressure_pct,
        },
        ..worker_registry::WorkerRegistryConfig::default()
    };
    let registry = worker_registry::WorkerRegistry::new(registry_config)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let worker_tls = gateway_worker_tls::worker_client_tls_from_env()?;
    validate_gateway_topology_policy(args.gateway_topology, &worker_approvals, &worker_tls)?;
    let client_config = gateway_worker_client_config(args, worker_tls);

    // DS6: the approval view defines the pool/generation structure before any
    // worker is validated, so a rollout window in the approvals reaches
    // selection and the replica path accepts both window generations. The
    // table is shared: the approvals refresh adopts changed views and the
    // poller completes pending cutovers at runtime.
    let rendered_view = render_approvals_text(&worker_approvals);
    let deployment_table = {
        let mut table = gateway_deployments::GatewayDeploymentTable::default();
        table
            .apply_view(&worker_approvals)
            .map_err(|error| anyhow::anyhow!("configured approvals view: {error}"))?;
        Arc::new(std::sync::Mutex::new(table))
    };
    let approvals_reporter =
        gateway_shared_approvals::GatewayApprovalsReporter::new(Arc::clone(&deployment_table));
    approvals_reporter.record(
        shared_approvals_view
            .as_ref()
            .map(gateway_shared_approvals::SharedApprovalsView::file_digest),
        &worker_approvals,
    );
    registry.set_generation_gate({
        let deployment_table = Arc::clone(&deployment_table);
        Arc::new(move |task, public_model| {
            let table = deployment_table
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            table.eligible_generation(task, public_model)
        })
    });

    let mut endpoints = BTreeSet::new();
    for approval in &worker_approvals {
        let endpoint = approval.endpoint.trim();
        if endpoint.is_empty() {
            anyhow::bail!("configured gateway worker endpoints must not be empty");
        }
        if !endpoints.insert(endpoint.to_string()) {
            anyhow::bail!("duplicate configured gateway worker endpoint: {endpoint}");
        }
    }

    // Approval errors a restart cannot fix fail boot. A worker that is merely
    // unreachable is adopted with retry once the pollers start below, so one
    // down replica never keeps the gateway from booting.
    for approval in &worker_approvals {
        gateway_worker_client(
            approval,
            &credentials,
            &client_config,
            args.gateway_topology,
        )
        .with_context(|| format!("invalid approved worker {}", approval.endpoint.trim()))?;
    }

    // Every chat model the approvals view names is served, the primary
    // `--public-model` first; requests route by their `model`.
    let chat_models = {
        let table = deployment_table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut aliases = vec![public_model.clone()];
        for approval in &worker_approvals {
            if approval.task == TaskKind::Chat && !aliases.contains(&approval.public_model) {
                aliases.push(approval.public_model.clone());
            }
        }
        aliases
            .into_iter()
            .map(|alias| {
                let deployment = table.select(TaskKind::Chat, &alias).ok_or_else(|| {
                    anyhow::anyhow!(
                        "configured gateway worker approvals do not include chat model {alias}"
                    )
                })?;
                let variant = parse_model_variant(alias.as_str()).with_context(|| {
                    format!("approved chat model {alias} is not a model this gateway knows")
                })?;
                Ok((variant, deployment.deployment_id().clone()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    let policy_revision = PolicyRevision::new(args.gateway_policy_revision.trim())?;
    let dispatchers = chat_models
        .into_iter()
        .map(|(public_model_variant, deployment_id)| {
            app::remote_chat_dispatch::RemoteChatDispatcher::new(
                registry.clone(),
                app::remote_chat_dispatch::RemoteChatDispatchConfig {
                    public_model_variant,
                    deployment_id,
                    policy_revision: policy_revision.clone(),
                    backend_policy: gateway_backend_policy(args.backend.as_ref()),
                    max_queue_wait: Duration::from_millis(args.gateway_worker_queue_wait_ms),
                    max_output_tokens: 4096,
                    max_output_bytes: 512 * 1024,
                    slow_consumer_timeout: Duration::from_millis(
                        args.gateway_slow_consumer_timeout_ms,
                    ),
                    prefix_affinity: prefix_affinity_config(args),
                },
            )
            .map_err(|error| anyhow::anyhow!(error.message))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Multi-gateway fleets share in-transit capacity claims
    // through one coordination database — a SQLite file by default, or a
    // server-backed database URL for shared fleets (DS5). Unset means
    // single-gateway operation with purely process-local state. When the
    // coordination database is set, the shared-atomic claim path is the
    // default selection posture: the worker remains the atomic admission
    // arbiter, and 1/N partitioning stays the explicitly-chosen quota
    // fallback (`IZWI_GATEWAY_FLEET_PARTITION`/`_SIZE`).
    let fleet_claim_ttl = crate::gateway_fleet::fleet_claim_ttl_from_env()?;
    let fleet_claim_timeout = crate::gateway_fleet::fleet_claim_timeout_from_env()?;
    let mut fleet_maintenance: Option<tokio::task::JoinHandle<()>> = None;
    let fleet = match crate::gateway_fleet::fleet_database_from_env()? {
        Some(fleet_database) => {
            let store_database = match fleet_database {
                crate::gateway_fleet::FleetDatabase::Path(path) => {
                    crate::db::StoreDatabase::new(path)
                }
                crate::gateway_fleet::FleetDatabase::Url(url) => {
                    crate::db::StoreDatabase::from_coordination_url(url)
                }
            };
            let store = crate::gateway_fleet::FleetStore::new(
                store_database,
            );
            // Two fleet gateways booting simultaneously race the SQLite
            // journal-mode setup before the busy timeout is in play, so the
            // first open retries briefly before failing closed.
            let mut opened = Err(anyhow::anyhow!("fleet coordination database never opened"));
            for _ in 0..10 {
                match store.connection().await {
                    Ok(_) => {
                        opened = Ok(());
                        break;
                    }
                    Err(error) => opened = Err(error),
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            opened.context("Failed to open fleet coordination database")?;
            let coordinator = Arc::new(
                app::fleet_coordinator::FleetCoordinator::new(
                    store,
                    crate::gateway_fleet::gateway_identity()?,
                )
                .with_claim_ttl(fleet_claim_ttl)
                .with_claim_timeout(fleet_claim_timeout),
            );
            let released = coordinator.release_own_claims().await;
            info!(
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                gateway_id = coordinator.gateway_id(),
                released_own_claims = released,
                claim_ttl_ms = fleet_claim_ttl.as_millis() as u64,
                claim_timeout_ms = fleet_claim_timeout.as_millis() as u64,
                selection_mode = "shared_atomic_claims",
                // Parsed once already in run_gateway; re-read here so the
                // posture log names the effective quota mode.
                quota_mode = if crate::gateway_fleet::FleetPartition::from_env()?.is_some() {
                    "partitioned_1_of_n"
                } else {
                    "worker_authoritative"
                },
                "Fleet coordination enabled: capacity claims are shared"
            );
            let maintenance_coordinator = coordinator.clone();
            fleet_maintenance = Some(tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(app::fleet_coordinator::FLEET_MAINTENANCE_INTERVAL);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                // The first tick fires immediately; the boot path just
                // released this gateway's own claims, so sweep on the cadence.
                ticker.tick().await;
                loop {
                    ticker.tick().await;
                    let reaped = maintenance_coordinator.maintenance_sweep().await;
                    if reaped > 0 {
                        debug!(
                            service = SERVICE_NAME,
                            reaped_claims = reaped,
                            "Fleet coordination maintenance sweep"
                        );
                    }
                }
            }));
            Some(coordinator)
        }
        None => None,
    };
    let mut dispatchers = dispatchers
        .into_iter()
        .map(|dispatcher| match fleet.clone() {
            Some(coordinator) => dispatcher.with_fleet_coordinator(coordinator),
            None => dispatcher,
        });
    let polling_interval = Duration::from_millis(args.gateway_worker_status_poll_ms);
    // `tasks` collects the maintenance and approvals reconcile tasks that
    // GatewayWorkerStatusPoller aborts on drop. Status pollers live in the
    // adoption map and terminate with the gateway runtime.
    let mut tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    if let Some(maintenance) = fleet_maintenance {
        tasks.push(maintenance);
    }
    let adopter = GatewayWorkerAdopter {
        registry: registry.clone(),
        deployment_table: Arc::clone(&deployment_table),
        credentials: credentials.clone(),
        client_config: client_config.clone(),
        topology: args.gateway_topology,
        fleet: fleet.clone(),
        polling_interval,
    };
    let mut adopted = BTreeMap::new();
    let mut boot_pending = 0;
    for (endpoint, error) in adopter.adopt_missing(&worker_approvals, &mut adopted).await {
        if let Some(duplicate) = error.downcast_ref::<DuplicateGatewayWorker>() {
            anyhow::bail!(
                "configured gateway endpoints must identify distinct logical workers; {duplicate}"
            );
        }
        boot_pending += 1;
        warn!(
            endpoint = %endpoint,
            error = %format!("{error:#}"),
            "Approved worker is not adoptable yet; the gateway boots without it and keeps retrying"
        );
    }
    let boot_poller_count = adopted.len();

    // Reconcile worker adoption at runtime: retry approved workers that were
    // unavailable at boot, and, with a shared approvals file, keep a locally
    // cached fresh view and adopt changed views (DS6). The rollout
    // coordinator drives generation transitions through atomic file writes;
    // adoption applies the DINV-07 eligibility rule in one table mutation,
    // and workers approved by the view are adopted (or retried) into the
    // registry and poller set. A rejected view retains the previous approval
    // state fail-closed.
    if shared_approvals_view.is_some() || boot_pending > 0 {
        let mut view = shared_approvals_view;
        let interval = view.as_ref().map_or(
            polling_interval,
            gateway_shared_approvals::SharedApprovalsView::ttl,
        );
        let refresh_table = Arc::clone(&deployment_table);
        let refresh_reporter = approvals_reporter.clone();
        let refresh_cli_approvals = cli_approvals.clone();
        let boot_approvals = worker_approvals.clone();
        tasks.push(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await;
            let mut last_applied = Some(rendered_view);
            loop {
                ticker.tick().await;
                let effective = match view.as_mut() {
                    Some(view) => {
                        if let Err(error) = view.refresh_if_due() {
                            warn!(error = %error, "Shared fleet approvals refresh failed; retaining previous view");
                            continue;
                        }
                        let mut effective = refresh_cli_approvals.clone();
                        effective.extend(view.approvals().iter().cloned());
                        let rendered = render_approvals_text(&effective);
                        if last_applied.as_deref() == Some(rendered.as_str()) {
                            refresh_reporter.record(Some(view.file_digest()), &effective);
                        } else {
                            let applied = refresh_table
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .apply_view(&effective);
                            match applied {
                                Ok(outcome) => {
                                    refresh_reporter.record(Some(view.file_digest()), &effective);
                                    last_applied = Some(rendered);
                                    info!(
                                        service = SERVICE_NAME,
                                        version = SERVICE_VERSION,
                                        approvals = effective.len(),
                                        added_generations = outcome.added.len(),
                                        removed_generations = outcome.removed.len(),
                                        "Shared fleet approvals adopted: deployment pool generations updated"
                                    );
                                }
                                Err(error) => {
                                    warn!(error = %error, "Shared fleet approvals view rejected; retaining previous approval state");
                                }
                            }
                        }
                        effective
                    }
                    None => boot_approvals.clone(),
                };
                // Stop pollers for workers the view removed before adopting,
                // so a worker that moved endpoints is not a duplicate of its
                // own stale entry. Registry records of removed workers lapse
                // naturally; the eligibility gate already excludes them.
                let view_endpoints: BTreeSet<String> = effective
                    .iter()
                    .map(|approval| approval.endpoint.trim().to_string())
                    .collect();
                adopted.retain(|endpoint, worker: &mut AdoptedGatewayWorker| {
                    let keep = view_endpoints.contains(endpoint);
                    if !keep {
                        worker.poller.abort();
                        info!(endpoint = %endpoint, "Removed approval poller stopped");
                    }
                    keep
                });
                let pending = adopter.adopt_missing(&effective, &mut adopted).await;
                for (endpoint, error) in &pending {
                    debug!(
                        endpoint = %endpoint,
                        error = %format!("{error:#}"),
                        "Approved worker adoption not complete yet; will retry"
                    );
                }
                if view.is_none() && pending.is_empty() {
                    break;
                }
            }
        }));
    }

    let primary = dispatchers
        .next()
        .expect("the primary chat model always has a dispatcher");
    let state = dispatchers.fold(
        gateway::GatewayState::new(
            primary,
            enterprise_hooks,
            perimeter,
            serve_config.request_timeout_secs,
            args.gateway_max_in_flight,
        ),
        gateway::GatewayState::with_chat_model,
    );
    let state = state.with_approvals_reporter(approvals_reporter);
    let state = if args.gateway_realtime == "on" {
        // Stage pools resolve per approved task for the public model; boot
        // fails closed only when no realtime stage can be served at all. A
        // missing individual stage refuses its admits with a policy close at
        // session time.
        let asr_deployment_id = {
            let table = deployment_table
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            table
                .select(TaskKind::SpeechToText, &public_model)
                .map(|deployment| deployment.deployment_id().clone())
        };
        let tts_deployment_id = {
            let table = deployment_table
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            table
                .select(TaskKind::TextToSpeech, &public_model)
                .map(|deployment| deployment.deployment_id().clone())
        };
        if asr_deployment_id.is_none() && tts_deployment_id.is_none() {
            return Err(anyhow::anyhow!(
                "--gateway-realtime is on but no approved worker advertises a realtime speech_to_text or text_to_speech model {public_model}"
            ));
        }
        let relay = app::realtime_relay::GatewayRealtimeRelay::new(
            registry,
            app::realtime_relay::RealtimeRelayConfig {
                deployment_id: asr_deployment_id,
                tts_deployment_id,
                public_model: public_model.clone(),
                policy_revision: policy_revision.clone(),
                backend_policy: gateway_backend_policy(args.backend.as_ref()),
                max_sessions: args.gateway_realtime_max_sessions,
                session_budget: Duration::from_millis(args.gateway_realtime_session_budget_ms),
            },
            credentials,
            client_config.tls.clone(),
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        info!(
            service = SERVICE_NAME,
            version = SERVICE_VERSION,
            asr_stage = relay.config().deployment_id.is_some(),
            tts_stage = relay.config().tts_deployment_id.is_some(),
            max_sessions = args.gateway_realtime_max_sessions,
            "Gateway realtime relay enabled for izwi-realtime-v1 sessions"
        );
        state.with_realtime_relay(relay)
    } else {
        state
    };
    let poller = GatewayWorkerStatusPoller {
        poller_count: boot_poller_count,
        tasks,
    };
    info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        pollers = poller.poller_count,
        interval_ms = polling_interval.as_millis() as u64,
        "Gateway worker status polling started"
    );
    Ok((state, poller))
}

fn initial_gateway_worker_expectation(
    client: WorkerClient,
    descriptor: &WorkerDescriptor,
    status: &WorkerStatus,
    approval: &gateway_deployments::GatewayWorkerApproval,
) -> anyhow::Result<GatewayWorkerExpectation> {
    if let Some((node_id, worker_id)) = approval.pinned_identity() {
        if descriptor.node_id != *node_id || descriptor.worker_id != *worker_id {
            anyhow::bail!(
                "worker descriptor does not match its operator-approved node and worker identity"
            );
        }
    }
    if status.worker_id != descriptor.worker_id
        || status.node_id != descriptor.node_id
        || status.incarnation_id != descriptor.incarnation_id
    {
        anyhow::bail!("initial worker descriptor and status identity do not match");
    }
    let selected_deployment = status
        .deployments
        .iter()
        .find(|deployment| deployment.deployment_id == approval.deployment_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "worker {} does not advertise configured deployment {}",
                descriptor.worker_id,
                approval.deployment_id
            )
        })?;
    if selected_deployment.public_model != approval.public_model
        || selected_deployment.model_generation != approval.model_generation
        || selected_deployment.task != approval.task
    {
        anyhow::bail!(
            "worker {} deployment {} does not match its configured task/model/generation",
            descriptor.worker_id,
            approval.deployment_id
        );
    }
    let validated_capacity = configured_worker_capacity(status)?;
    Ok(GatewayWorkerExpectation {
        client,
        worker_id: descriptor.worker_id.clone(),
        node_id: descriptor.node_id.clone(),
        deployment: worker_registry::ApprovedDeployment::from_loaded(selected_deployment),
        validated_capacity,
    })
}

fn configured_gateway_worker_approvals(
    args: &ServerArgs,
    legacy_public_model: &ModelAlias,
) -> anyhow::Result<(
    Vec<gateway_deployments::GatewayWorkerApproval>,
    Vec<gateway_deployments::GatewayWorkerApproval>,
    Option<gateway_shared_approvals::SharedApprovalsView>,
)> {
    // Returns (CLI-only approvals, effective merged view, shared file view).
    // `--worker-endpoint` is the one-worker form of the legacy endpoint list.
    let legacy_endpoints: Vec<String> = args
        .gateway_worker_endpoints
        .iter()
        .chain(args.worker_endpoint.as_ref())
        .cloned()
        .collect();
    let mut merged = if !args.gateway_worker_approvals.is_empty() {
        if !legacy_endpoints.is_empty() {
            anyhow::bail!(
                "--gateway-worker-approval cannot be combined with --gateway-worker-endpoint"
            );
        }
        if args.worker_deployment.is_some() || args.worker_model_generation.is_some() {
            anyhow::bail!(
                "--worker-deployment and --worker-model-generation apply only to legacy --gateway-worker-endpoint configuration"
            );
        }
        args.gateway_worker_approvals.clone()
    } else if !legacy_endpoints.is_empty() {
        let deployment_id = DeploymentId::new(required_gateway_value(
            &args.worker_deployment,
            "--worker-deployment",
        )?)?;
        let model_generation =
            ModelGeneration::new(args.worker_model_generation.ok_or_else(|| {
                anyhow::anyhow!("gateway mode requires --worker-model-generation")
            })?)?;
        legacy_endpoints
            .into_iter()
            .map(|endpoint| gateway_deployments::GatewayWorkerApproval {
                endpoint,
                identity: gateway_deployments::GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint,
                task: TaskKind::Chat,
                public_model: legacy_public_model.clone(),
                deployment_id: deployment_id.clone(),
                model_generation,
            })
            .collect()
    } else {
        // No CLI approvals and no legacy endpoints: the shared approvals
        // file (the DS6 rollout channel) may be the sole approval source.
        Vec::new()
    };

    // Multi-gateway fleets keep one authoritative approvals file that every
    // gateway reads, so all gateways approve the same worker set. Shared
    // entries augment (never replace) explicit CLI approvals; the existing
    // duplicate-endpoint check below rejects any conflict fail-closed.
    let cli_only = merged.clone();
    let shared_view = match gateway_shared_approvals::SharedApprovalsConfig::from_env() {
        Ok(Some(config)) => {
            let view = gateway_shared_approvals::SharedApprovalsView::load(config)
                .map_err(|error| anyhow::anyhow!("shared fleet approvals: {error}"))?;
            merged.extend(view.approvals().iter().cloned());
            Some(view)
        }
        Ok(None) => None,
        Err(error) => anyhow::bail!("shared fleet approvals: {error}"),
    };
    Ok((cli_only, merged, shared_view))
}

fn validate_gateway_topology_policy(
    topology: GatewayTopology,
    approvals: &[gateway_deployments::GatewayWorkerApproval],
    tls: &izwi_serving_client::WorkerClientTlsConfig,
) -> anyhow::Result<()> {
    let mut pinned_worker_ids = BTreeSet::new();
    for approval in approvals {
        if let Some((_node_id, worker_id)) = approval.pinned_identity() {
            if !pinned_worker_ids.insert(worker_id.clone()) {
                anyhow::bail!("duplicate operator-approved logical worker identity: {worker_id}");
            }
        }
    }
    if topology == GatewayTopology::Standalone {
        return Ok(());
    }
    if approvals.is_empty() {
        anyhow::bail!("fleet-one-gateway topology requires at least one worker approval");
    }
    if !tls.has_client_identity() {
        anyhow::bail!(
            "fleet-one-gateway topology requires a client certificate and private key for mutual TLS"
        );
    }
    if approvals
        .iter()
        .any(|approval| approval.pinned_identity().is_none())
    {
        anyhow::bail!(
            "fleet-one-gateway topology requires versioned v1 approvals with pinned node and worker identities"
        );
    }
    Ok(())
}

fn validate_gateway_topology_source(args: &ServerArgs) -> anyhow::Result<()> {
    if args.gateway_topology == GatewayTopology::FleetOneGateway
        && args.gateway_worker_approvals.is_empty()
    {
        anyhow::bail!(
            "fleet-one-gateway topology requires versioned --gateway-worker-approval entries; pinned and legacy endpoint modes are standalone-only"
        );
    }
    Ok(())
}

fn validate_gateway_worker_endpoint_policy(
    topology: GatewayTopology,
    uses_https: bool,
    uses_numeric_loopback_http: bool,
) -> anyhow::Result<()> {
    match topology {
        GatewayTopology::Standalone if !uses_numeric_loopback_http => {
            anyhow::bail!("standalone worker endpoints must use numeric-loopback HTTP")
        }
        GatewayTopology::FleetOneGateway if !uses_https => {
            anyhow::bail!("fleet-one-gateway worker endpoints must use HTTPS")
        }
        _ => {}
    }
    Ok(())
}

fn configured_worker_capacity(status: &WorkerStatus) -> anyhow::Result<u32> {
    status
        .capacity
        .max_active_invocations
        .checked_add(status.capacity.max_queued_invocations)
        .filter(|capacity| *capacity > 0)
        .ok_or_else(|| anyhow::anyhow!("worker has invalid configured capacity"))
}

fn validate_gateway_worker_observation(
    expected: &GatewayWorkerExpectation,
    descriptor: &WorkerDescriptor,
    status: WorkerStatus,
) -> anyhow::Result<WorkerStatus> {
    if descriptor.worker_id != expected.worker_id || descriptor.node_id != expected.node_id {
        anyhow::bail!("restarted worker changed its approved logical worker or node identity");
    }
    if status.incarnation_id != descriptor.incarnation_id {
        anyhow::bail!("restarted worker descriptor and status identity do not match");
    }
    validate_gateway_worker_status(expected, status)
}

fn validate_gateway_worker_status(
    expected: &GatewayWorkerExpectation,
    mut status: WorkerStatus,
) -> anyhow::Result<WorkerStatus> {
    if status.worker_id != expected.worker_id || status.node_id != expected.node_id {
        anyhow::bail!("worker status changed its approved logical worker or node identity");
    }
    if configured_worker_capacity(&status)? != expected.validated_capacity {
        anyhow::bail!("restarted worker changed its validated capacity");
    }
    let deployment = status
        .deployments
        .iter()
        .find(|deployment| deployment.deployment_id == expected.deployment.deployment_id)
        .ok_or_else(|| anyhow::anyhow!("restarted worker omitted its approved deployment"))?;
    if worker_registry::ApprovedDeployment::from_loaded(deployment) != expected.deployment {
        anyhow::bail!("restarted worker changed its approved deployment contract");
    }

    // Retain only the configured deployment in the local routing view. Other
    // worker-local deployments are neither approved nor selectable merely
    // because an authenticated endpoint advertised them.
    status
        .deployments
        .retain(|deployment| deployment.deployment_id == expected.deployment.deployment_id);
    Ok(status)
}

fn approve_gateway_worker(
    registry: &worker_registry::WorkerRegistry,
    expected: &GatewayWorkerExpectation,
    descriptor: WorkerDescriptor,
    status: WorkerStatus,
) -> anyhow::Result<()> {
    let status = validate_gateway_worker_observation(expected, &descriptor, status)?;
    registry
        .approve(worker_registry::ApprovedWorker {
            descriptor,
            client: expected.client.clone(),
            approved_deployments: BTreeMap::from([(
                expected.deployment.deployment_id.clone(),
                expected.deployment.clone(),
            )]),
            validated_capacity: expected.validated_capacity,
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    registry
        .observe_status(status)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

/// Adopts one approved worker into the running gateway: builds its
/// authenticated client, validates the live worker against the approval's
/// exact deployment contract, approves the replica into the deployment
/// table (filling the generation's learned contract), registers and
/// observes it in the registry, and completes a pending rollout cutover
/// when the adoption observes the successor Ready. Used at boot and at
/// runtime by the shared-approvals adoption path.
/// Builds the private client for one approval and applies the topology's
/// endpoint policy. Its errors are configuration errors no retry can fix.
fn gateway_worker_client(
    approval: &gateway_deployments::GatewayWorkerApproval,
    credentials: &ServiceCredentials,
    client_config: &WorkerClientConfig,
    topology: GatewayTopology,
) -> anyhow::Result<WorkerClient> {
    let client = WorkerClient::new(
        approval.endpoint.trim(),
        credentials.clone(),
        client_config.clone(),
    )?;
    validate_gateway_worker_endpoint_policy(
        topology,
        client.uses_https(),
        client.uses_numeric_loopback_http(),
    )?;
    Ok(client)
}

/// Two approved endpoints answered as the same logical worker.
#[derive(Debug, thiserror::Error)]
#[error("endpoint {endpoint} reports worker {worker_id}, already adopted at another endpoint")]
struct DuplicateGatewayWorker {
    endpoint: String,
    worker_id: WorkerId,
}

/// One adopted worker: its logical identity and its status poller.
struct AdoptedGatewayWorker {
    worker_id: WorkerId,
    poller: tokio::task::JoinHandle<()>,
}

/// Adopts approved workers into the registry and starts their pollers. Boot
/// and the runtime reconcile loop share it, so a worker that is down at boot
/// is adopted exactly as one a shared approvals view adds later.
#[derive(Clone)]
struct GatewayWorkerAdopter {
    registry: worker_registry::WorkerRegistry,
    deployment_table: Arc<std::sync::Mutex<gateway_deployments::GatewayDeploymentTable>>,
    credentials: ServiceCredentials,
    client_config: WorkerClientConfig,
    topology: GatewayTopology,
    fleet: Option<Arc<app::fleet_coordinator::FleetCoordinator>>,
    polling_interval: Duration,
}

impl GatewayWorkerAdopter {
    /// Adopts every approval whose endpoint has no adopted worker yet and
    /// returns the ones that could not be adopted, with why.
    async fn adopt_missing(
        &self,
        approvals: &[gateway_deployments::GatewayWorkerApproval],
        adopted: &mut BTreeMap<String, AdoptedGatewayWorker>,
    ) -> Vec<(String, anyhow::Error)> {
        let mut pending = Vec::new();
        for approval in approvals {
            let endpoint = approval.endpoint.trim().to_string();
            if endpoint.is_empty() || adopted.contains_key(&endpoint) {
                continue;
            }
            let adopted_ids: BTreeSet<WorkerId> = adopted
                .values()
                .map(|worker| worker.worker_id.clone())
                .collect();
            match adopt_gateway_worker(
                &self.registry,
                &self.deployment_table,
                approval,
                &self.credentials,
                &self.client_config,
                self.topology,
                &adopted_ids,
            )
            .await
            {
                Ok(expected) => {
                    info!(
                        worker_id = %expected.worker_id,
                        endpoint = %endpoint,
                        generation = expected.deployment.model_generation.get(),
                        "Approved worker adopted"
                    );
                    let worker_id = expected.worker_id.clone();
                    let poller = spawn_gateway_worker_poller(
                        expected,
                        self.registry.clone(),
                        Arc::clone(&self.deployment_table),
                        self.fleet.clone(),
                        self.polling_interval,
                    );
                    adopted.insert(endpoint, AdoptedGatewayWorker { worker_id, poller });
                }
                Err(error) => pending.push((endpoint, error)),
            }
        }
        pending
    }
}

/// Validates one approved worker against its live descriptor and status and
/// registers it. `adopted_ids` are the logical workers already adopted at
/// other endpoints; a duplicate fails before the registry changes.
async fn adopt_gateway_worker(
    registry: &worker_registry::WorkerRegistry,
    deployment_table: &Arc<std::sync::Mutex<gateway_deployments::GatewayDeploymentTable>>,
    approval: &gateway_deployments::GatewayWorkerApproval,
    credentials: &ServiceCredentials,
    client_config: &WorkerClientConfig,
    topology: GatewayTopology,
    adopted_ids: &BTreeSet<WorkerId>,
) -> anyhow::Result<GatewayWorkerExpectation> {
    let endpoint = approval.endpoint.trim();
    let client = gateway_worker_client(approval, credentials, client_config, topology)?;
    let descriptor = client
        .descriptor()
        .await
        .with_context(|| format!("failed to read approved worker descriptor from {endpoint}"))?;
    if adopted_ids.contains(&descriptor.worker_id) {
        return Err(DuplicateGatewayWorker {
            endpoint: endpoint.to_string(),
            worker_id: descriptor.worker_id,
        }
        .into());
    }
    let status = client
        .status()
        .await
        .with_context(|| format!("failed to read initial worker status from {endpoint}"))?;
    let expectation = initial_gateway_worker_expectation(client, &descriptor, &status, approval)?;
    // Register first: a failed registry approval must never mutate the
    // deployment table (a premature cutover could strand admission with no
    // eligible generation).
    approve_gateway_worker(registry, &expectation, descriptor, status.clone())?;
    let cutover_completed = {
        let mut table = deployment_table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        table
            .approve_replica(expectation.deployment.clone())
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        status
            .deployments
            .iter()
            .find(|deployment| {
                deployment.deployment_id == expectation.deployment.deployment_id
                    && deployment.readiness == ModelReadiness::Ready
            })
            .is_some_and(|deployment| {
                table.observe_generation_ready(
                    expectation.deployment.task,
                    &expectation.deployment.public_model,
                    &expectation.deployment.deployment_id,
                    deployment.model_generation,
                )
            })
    };
    if cutover_completed {
        info!(
            worker_id = %expectation.worker_id,
            deployment = %expectation.deployment.deployment_id,
            generation = expectation.deployment.model_generation.get(),
            "Rollout cutover completed: successor generation is admission-eligible"
        );
    }
    Ok(expectation)
}

/// Spawns the bounded status poller for one adopted worker expectation.
fn spawn_gateway_worker_poller(
    expected: GatewayWorkerExpectation,
    registry: worker_registry::WorkerRegistry,
    deployment_table: Arc<std::sync::Mutex<gateway_deployments::GatewayDeploymentTable>>,
    fleet: Option<std::sync::Arc<app::fleet_coordinator::FleetCoordinator>>,
    polling_interval: Duration,
) -> tokio::task::JoinHandle<()> {
    let cadence = jittered_poll_interval(
        polling_interval,
        poll_jitter_seed(&expected.worker_id, &expected.node_id),
    );
    debug!(
        worker_id = %expected.worker_id,
        configured_poll_ms = polling_interval.as_millis() as u64,
        jittered_poll_ms = cadence.as_millis() as u64,
        "Worker status poller cadence"
    );
    tokio::spawn(async move {
        // Owned by this task, so stopping the poller stops its publisher.
        let publisher = fleet
            .as_ref()
            .map(app::fleet_coordinator::FleetCoordinator::spawn_status_publisher);
        let mut ticker = tokio::time::interval(cadence);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // The adopting observation was recorded synchronously.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match refresh_gateway_worker_status(&registry, &expected, publisher.as_ref()).await {
                Ok(status) => {
                    // DS6: a Ready observation of a pending rollout
                    // successor completes the cutover atomically in
                    // the deployment table (DINV-07).
                    let cutover = status.deployments.iter().find(|deployment| {
                        deployment.deployment_id == expected.deployment.deployment_id
                            && deployment.readiness == ModelReadiness::Ready
                    });
                    let completed = cutover.is_some_and(|deployment| {
                        let mut table = deployment_table
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        table.observe_generation_ready(
                            deployment.task,
                            &deployment.public_model,
                            &deployment.deployment_id,
                            deployment.model_generation,
                        )
                    });
                    if completed {
                        info!(
                            worker_id = %expected.worker_id,
                            deployment = %expected.deployment.deployment_id,
                            generation = expected.deployment.model_generation.get(),
                            "Rollout cutover completed: successor generation is admission-eligible"
                        );
                    }
                }
                Err(error) => {
                    warn!(
                        worker_id = %expected.worker_id,
                        error = %error,
                        "Worker status refresh failed"
                    );
                }
            }
        }
    })
}

/// Records one polled status in the registry, then hands it to the fleet
/// publisher. The registry observation is what keeps the worker fresh, so it
/// never waits on the coordination store; the publisher shares it with peer
/// gateways in the background.
async fn refresh_gateway_worker_status(
    registry: &worker_registry::WorkerRegistry,
    expected: &GatewayWorkerExpectation,
    fleet: Option<&app::fleet_coordinator::FleetStatusPublisher>,
) -> anyhow::Result<WorkerStatus> {
    let status = validate_gateway_worker_status(expected, expected.client.status().await?)?;
    match registry.observe_status(status.clone()) {
        Ok(()) => {}
        Err(worker_registry::WorkerRegistryError::UnknownOrStaleIncarnation) => {
            let descriptor = expected.client.descriptor().await?;
            approve_gateway_worker(registry, expected, descriptor, status.clone())?;
        }
        Err(error) => return Err(anyhow::anyhow!(error.to_string())),
    }
    if let Some(publisher) = fleet {
        publisher.publish(status.clone());
    }
    Ok(status)
}

fn required_gateway_value<'a>(value: &'a Option<String>, name: &str) -> anyhow::Result<&'a str> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("gateway mode requires {name}"))
}

fn gateway_worker_credentials(args: &ServerArgs) -> anyhow::Result<ServiceCredentials> {
    Ok(ServiceCredentials {
        credential_id: CredentialId::new(required_gateway_value(
            &args.worker_credential_id,
            "--worker-credential-id",
        )?)?,
        bearer_token: ServiceBearerToken::new(required_gateway_value(
            &args.worker_bearer_token,
            "--worker-bearer-token or IZWI_GATEWAY_WORKER_BEARER_TOKEN",
        )?)?,
    })
}

fn gateway_backend_policy(backend: Option<&BackendArg>) -> worker_registry::BackendPolicy {
    match backend {
        Some(BackendArg::Cpu) => worker_registry::BackendPolicy::CPU_ONLY,
        Some(BackendArg::Metal) => worker_registry::BackendPolicy::METAL_ONLY,
        Some(BackendArg::Cuda) => worker_registry::BackendPolicy::CUDA_ONLY,
        Some(BackendArg::Auto) | None => worker_registry::BackendPolicy::ANY,
    }
}

fn validate_gateway_limits(args: &ServerArgs) -> anyhow::Result<()> {
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
    let admission_timeout = Duration::from_millis(args.gateway_worker_admission_timeout_ms);
    if admission_timeout.is_zero() || admission_timeout > MAX_GATEWAY_ADMISSION_TIMEOUT {
        anyhow::bail!(
            "--gateway-worker-admission-timeout-ms must be between 1 and {}",
            MAX_GATEWAY_ADMISSION_TIMEOUT.as_millis()
        );
    }
    if Duration::from_millis(args.gateway_worker_queue_wait_ms) > admission_timeout {
        anyhow::bail!(
            "--gateway-worker-queue-wait-ms must not exceed the worker admission timeout"
        );
    }
    for (name, value) in [
        (
            "--gateway-worker-first-output-timeout-ms",
            args.gateway_worker_first_output_timeout_ms,
        ),
        (
            "--gateway-worker-progress-idle-timeout-ms",
            args.gateway_worker_progress_idle_timeout_ms,
        ),
    ] {
        let timeout = Duration::from_millis(value);
        if timeout.is_zero() || timeout > MAX_GATEWAY_STREAM_PHASE_TIMEOUT {
            anyhow::bail!(
                "{name} must be between 1 and {}",
                MAX_GATEWAY_STREAM_PHASE_TIMEOUT.as_millis()
            );
        }
    }
    if args.gateway_shutdown_budget_secs == 0
        || args.gateway_shutdown_budget_secs > MAX_GATEWAY_SHUTDOWN_BUDGET_SECS
        || args.gateway_prestop_secs >= args.gateway_shutdown_budget_secs
    {
        anyhow::bail!(
            "--gateway-shutdown-budget-secs must be between 1 and {MAX_GATEWAY_SHUTDOWN_BUDGET_SECS} and longer than --gateway-prestop-secs"
        );
    }
    let body_read_timeout = Duration::from_millis(args.gateway_body_read_timeout_ms);
    if body_read_timeout.is_zero() || body_read_timeout > MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT {
        anyhow::bail!(
            "--gateway-body-read-timeout-ms must be between 1 and {}",
            MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT.as_millis()
        );
    }
    if Duration::from_millis(args.gateway_worker_first_output_timeout_per_kib_ms)
        > MAX_GATEWAY_STREAM_PHASE_TIMEOUT
    {
        anyhow::bail!(
            "--gateway-worker-first-output-timeout-per-kib-ms must be at most {}",
            MAX_GATEWAY_STREAM_PHASE_TIMEOUT.as_millis()
        );
    }
    let slow_consumer_timeout = Duration::from_millis(args.gateway_slow_consumer_timeout_ms);
    if slow_consumer_timeout.is_zero() || slow_consumer_timeout > MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT
    {
        anyhow::bail!(
            "--gateway-slow-consumer-timeout-ms must be between 1 and {}",
            MAX_GATEWAY_SLOW_CONSUMER_TIMEOUT.as_millis()
        );
    }
    let configured_workers = args
        .gateway_worker_endpoints
        .len()
        .checked_add(args.gateway_worker_approvals.len())
        .ok_or_else(|| anyhow::anyhow!("configured gateway worker count overflowed"))?;
    if configured_workers > MAX_CONFIGURED_GATEWAY_WORKERS {
        anyhow::bail!(
            "at most {MAX_CONFIGURED_GATEWAY_WORKERS} gateway worker endpoints or approvals are supported"
        );
    }
    let ttl = Duration::from_millis(args.gateway_worker_status_ttl_ms);
    let poll = Duration::from_millis(args.gateway_worker_status_poll_ms);
    if ttl.is_zero() || ttl > MAX_GATEWAY_STATUS_TTL {
        anyhow::bail!("--gateway-worker-status-ttl-ms is outside the supported range");
    }
    validate_status_cadence(ttl, poll)?;
    validate_router_args(args)?;
    validate_realtime_relay_args(args)?;
    Ok(())
}

/// Parses and bounds-checks the routing knobs. Fail-closed on any
/// out-of-range or non-`on|off` value so typos never silently change routing.
fn validate_router_args(args: &ServerArgs) -> anyhow::Result<()> {
    match args.gateway_router_prefix_affinity.as_str() {
        "on" | "off" => {}
        other => {
            anyhow::bail!("--gateway-router-prefix-affinity must be `on` or `off`, got `{other}`")
        }
    }
    let usage = args.gateway_router_kv_pressure_pct;
    if usage.is_nan() || usage <= 0.0 || usage > 100.0 {
        anyhow::bail!("--gateway-router-kv-pressure-pct must be greater than 0 and at most 100");
    }
    worker_registry::RoutingConfig {
        prefix_balance_threshold: args.gateway_router_balance_threshold,
        kv_pressure_pct: usage,
    }
    .validate()
    .map_err(|error| anyhow::anyhow!("--gateway-router-balance-threshold: {error}"))?;
    app::prefix_router::PrefixAffinityConfig {
        block_bytes: args.gateway_router_prefix_block_bytes,
        max_blocks: args.gateway_router_prefix_max_blocks,
        max_entries_per_worker: args.gateway_router_prefix_entries,
    }
    .validate()
    .map_err(|error| anyhow::anyhow!(error.message))?;
    Ok(())
}

/// Parses and bounds-checks the realtime relay knobs (DS3.6).
fn validate_realtime_relay_args(args: &ServerArgs) -> anyhow::Result<()> {
    match args.gateway_realtime.as_str() {
        "on" | "off" => {}
        other => anyhow::bail!("--gateway-realtime must be `on` or `off`, got `{other}`"),
    }
    if args.gateway_realtime == "off" {
        return Ok(());
    }
    if args.gateway_realtime_max_sessions == 0
        || args.gateway_realtime_max_sessions > app::realtime_relay::MAX_RELAY_SESSIONS
    {
        anyhow::bail!(
            "--gateway-realtime-max-sessions must be between 1 and {}",
            app::realtime_relay::MAX_RELAY_SESSIONS
        );
    }
    let budget = Duration::from_millis(args.gateway_realtime_session_budget_ms);
    if budget.is_zero() || budget > Duration::from_millis(MAX_REMAINING_TIME_MS) {
        anyhow::bail!(
            "--gateway-realtime-session-budget-ms must be between 1 and {MAX_REMAINING_TIME_MS}"
        );
    }
    Ok(())
}

fn prefix_affinity_config(args: &ServerArgs) -> Option<app::prefix_router::PrefixAffinityConfig> {
    (args.gateway_router_prefix_affinity == "on").then_some(
        app::prefix_router::PrefixAffinityConfig {
            block_bytes: args.gateway_router_prefix_block_bytes,
            max_blocks: args.gateway_router_prefix_max_blocks,
            max_entries_per_worker: args.gateway_router_prefix_entries,
        },
    )
}

fn gateway_worker_client_config(
    args: &ServerArgs,
    tls: izwi_serving_client::WorkerClientTlsConfig,
) -> WorkerClientConfig {
    WorkerClientConfig {
        max_in_flight: args.gateway_max_in_flight,
        request_timeout: Duration::from_millis(args.gateway_worker_admission_timeout_ms),
        first_output_timeout: Duration::from_millis(args.gateway_worker_first_output_timeout_ms),
        first_output_timeout_per_kib: Duration::from_millis(
            args.gateway_worker_first_output_timeout_per_kib_ms,
        ),
        progress_timeout: Duration::from_millis(args.gateway_worker_progress_idle_timeout_ms),
        ndjson_limits: NdjsonLimits {
            max_line_bytes: GATEWAY_NDJSON_MAX_LINE_BYTES,
            max_total_bytes: GATEWAY_NDJSON_MAX_TOTAL_BYTES,
            max_events: GATEWAY_NDJSON_MAX_EVENTS,
            ..NdjsonLimits::default()
        },
        tls,
        ..WorkerClientConfig::default()
    }
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
    .with_artifact_store(state.artifact_store.clone())
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
    let file = ServeRuntimeConfigOverrides::from_user_config(args.config.as_deref())?;
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

/// Resolve on Ctrl+C, SIGTERM or the desktop owner pipe closing.
async fn shutdown_requested() {
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
}

/// Wait for shutdown and begin draining the server, runtime and batch worker.
async fn shutdown_signal<S>(
    shutdown_requested: S,
    state: AppState,
    batch_worker_drain: Arc<std::sync::OnceLock<BatchWorkerDrain>>,
    shutdown_started: oneshot::Sender<()>,
) where
    S: std::future::Future<Output = ()>,
{
    shutdown_requested.await;

    state.lifecycle.mark_draining();
    state.runtime.begin_drain();
    if let Some(drain) = batch_worker_drain.get() {
        drain.begin();
    }
    let _ = shutdown_started.send(());

    drop(state);
}

async fn gateway_shutdown_requested() {
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
    worker_shutdown.await?;
    cleanup.await;
    Ok(())
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
