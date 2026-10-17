//! The `izwi-server` command line: arguments, roles and how they resolve to the runtime
//! configuration.

use clap::{Parser, ValueEnum};
use izwi_core::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};
use std::path::PathBuf;

use crate::gateway_deployments;
use crate::logging::LogFormat;

#[derive(Debug, Parser)]
#[command(
    name = "izwi-server",
    about = "HTTP API server for Izwi local inference",
    version = env!("CARGO_PKG_VERSION")
)]
pub(crate) struct ServerArgs {
    /// Configuration file (defaults to the shared Izwi user config.toml).
    #[arg(long, value_name = "PATH")]
    pub(crate) config: Option<PathBuf>,

    /// Process role: local inference server or hardware-independent gateway.
    #[arg(long, value_enum, env = "IZWI_SERVER_ROLE", default_value = "local")]
    pub(crate) role: ServerRole,

    /// Override a performance setting, e.g. cuda.mode=off; repeat for siblings.
    #[arg(long = "performance", value_name = "KEY=VALUE", value_parser = parse_performance_override)]
    pub(crate) performance: Vec<izwi_core::PerformanceConfigOverrides>,

    /// Host to bind to (default 127.0.0.1). A non-loopback bind on the local
    /// role requires IZWI_API_KEY unless --allow-unauthenticated-network.
    #[arg(short = 'H', long)]
    pub(crate) host: Option<String>,

    /// Serve a non-loopback local-role bind without an API key. Anyone who
    /// can reach the port can then manage models and read history. Also
    /// IZWI_ALLOW_UNAUTHENTICATED_NETWORK, read with the shared boolean
    /// vocabulary so a mistyped value is an error rather than `true`.
    #[arg(long)]
    pub(crate) allow_unauthenticated_network: bool,

    /// Port to listen on
    #[arg(short, long)]
    pub(crate) port: Option<u16>,

    /// Backend preference (`auto`, `cpu`, `metal`, `cuda`)
    #[arg(long, value_enum, env = "IZWI_BACKEND")]
    pub(crate) backend: Option<BackendArg>,

    /// Physical launch rollout mode (`serial`, `shadow`, `concurrent`)
    #[arg(long, value_name = "MODE")]
    pub(crate) physical_execution_mode: Option<izwi_core::PhysicalExecutionMode>,

    /// Maximum candidate physical launches in flight
    #[arg(long, value_name = "COUNT")]
    pub(crate) max_physical_in_flight: Option<izwi_core::PhysicalInFlightLimit>,

    /// Portable context length (`auto` or a positive token count)
    #[arg(long, value_name = "AUTO_OR_TOKENS")]
    pub(crate) max_sequence_length: Option<izwi_core::ContextLengthPreference>,

    /// Log output format (`text`, `json`)
    #[arg(long, value_enum, env = "IZWI_LOG_FORMAT", default_value = "text")]
    pub(crate) log_format: LogFormat,

    /// Enable Granite ASR decode-profile diagnostics after backend selection.
    #[arg(long)]
    pub(crate) granite_decode_profile: bool,

    /// Override Granite ASR dtype after backend selection (`f32`, `f16`, `bf16`).
    #[arg(long, value_name = "DTYPE")]
    pub(crate) granite_speech_dtype: Option<String>,

    /// Legacy one-worker form of `--gateway-worker-endpoint`: the gateway
    /// routes to this worker through the same registry as an endpoint list.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_ENDPOINT")]
    pub(crate) worker_endpoint: Option<String>,

    /// Worker-network policy (`standalone` or `fleet-one-gateway`).
    #[arg(
        long,
        value_enum,
        env = "IZWI_GATEWAY_TOPOLOGY",
        default_value = "standalone"
    )]
    pub(crate) gateway_topology: GatewayTopology,

    /// Approved private worker URLs for registry-backed routing. Repeat this
    /// option (or use a comma-separated environment value) to add capacity.
    #[arg(
        long = "gateway-worker-endpoint",
        env = "IZWI_GATEWAY_WORKER_ENDPOINTS",
        value_delimiter = ',',
        value_name = "URL"
    )]
    pub(crate) gateway_worker_endpoints: Vec<String>,

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
    pub(crate) gateway_worker_approvals: Vec<gateway_deployments::GatewayWorkerApproval>,

    /// Rotatable private worker credential identifier required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_CREDENTIAL_ID")]
    pub(crate) worker_credential_id: Option<String>,

    /// Private worker bearer token required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_BEARER_TOKEN")]
    pub(crate) worker_bearer_token: Option<String>,

    /// Accepted with `--worker-endpoint` for compatibility and ignored: the
    /// registry fences every attempt to the incarnation it observed and
    /// adopts a restarted worker.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_INCARNATION")]
    pub(crate) worker_incarnation: Option<String>,

    /// Chat deployment identifier of the legacy worker endpoint forms.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_DEPLOYMENT")]
    pub(crate) worker_deployment: Option<String>,

    /// Public model served through the gateway.
    #[arg(long, env = "IZWI_GATEWAY_PUBLIC_MODEL")]
    pub(crate) public_model: Option<String>,

    /// Expected non-zero model generation required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_MODEL_GENERATION")]
    pub(crate) worker_model_generation: Option<u64>,

    /// Policy revision attested on private worker requests.
    #[arg(
        long,
        env = "IZWI_GATEWAY_POLICY_REVISION",
        default_value = "local-policy-v1"
    )]
    pub(crate) gateway_policy_revision: String,

    /// Maximum gateway requests concurrently in flight to the pinned worker.
    #[arg(long, env = "IZWI_GATEWAY_MAX_IN_FLIGHT", default_value_t = 32)]
    pub(crate) gateway_max_in_flight: usize,

    /// Maximum time the selected worker may spend establishing runtime ownership.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_QUEUE_WAIT_MS", default_value_t = 250)]
    pub(crate) gateway_worker_queue_wait_ms: u64,

    /// Maximum wait for private invocation response headers and admission.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_ADMISSION_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    pub(crate) gateway_worker_admission_timeout_ms: u64,

    /// Maximum wait after admission for the first useful worker output.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_MS",
        default_value_t = 60_000
    )]
    pub(crate) gateway_worker_first_output_timeout_ms: u64,

    /// Extra first-output allowance per KiB of prompt, so a long prompt's
    /// prefill (slow on CPU) is not cancelled by the short-prompt deadline.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_PER_KIB_MS",
        default_value_t = 1_000
    )]
    pub(crate) gateway_worker_first_output_timeout_per_kib_ms: u64,

    /// Maximum idle time between useful worker output events.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_PROGRESS_IDLE_TIMEOUT_MS",
        default_value_t = 30_000
    )]
    pub(crate) gateway_worker_progress_idle_timeout_ms: u64,

    /// Total gateway shutdown budget after SIGTERM: the pre-stop window,
    /// then draining in-flight requests. The default fits Kubernetes'
    /// default 30 s termination grace period.
    #[arg(long, env = "IZWI_GATEWAY_SHUTDOWN_BUDGET_SECS", default_value_t = 25)]
    pub(crate) gateway_shutdown_budget_secs: u64,

    /// Pre-stop window at the start of the shutdown budget: readiness turns
    /// unready while requests are still served, so load balancers stop
    /// routing here before the listener closes.
    #[arg(long, env = "IZWI_GATEWAY_PRESTOP_SECS", default_value_t = 5)]
    pub(crate) gateway_prestop_secs: u64,

    /// Deadline for receiving a public request body. A slow upload holds a
    /// global admission permit at most this long.
    #[arg(
        long,
        env = "IZWI_GATEWAY_BODY_READ_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    pub(crate) gateway_body_read_timeout_ms: u64,

    /// Maximum wait while relaying an event to a connected slow consumer.
    #[arg(
        long,
        env = "IZWI_GATEWAY_SLOW_CONSUMER_TIMEOUT_MS",
        default_value_t = 5_000
    )]
    pub(crate) gateway_slow_consumer_timeout_ms: u64,

    /// Receiver-clock lifetime of a worker status observation.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_STATUS_TTL_MS",
        default_value_t = 10_000
    )]
    pub(crate) gateway_worker_status_ttl_ms: u64,

    /// Status refresh interval for each configured registry worker.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_STATUS_POLL_MS",
        default_value_t = 2_000
    )]
    pub(crate) gateway_worker_status_poll_ms: u64,

    /// Prefix-aware routing: prefer the worker recently routed the longest
    /// leading part of a chat prompt, within the balance threshold below.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_AFFINITY",
        default_value = "on"
    )]
    pub(crate) gateway_router_prefix_affinity: String,

    /// Bytes of message stream per hashed prefix block.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_BLOCK_BYTES",
        default_value_t = 256
    )]
    pub(crate) gateway_router_prefix_block_bytes: usize,

    /// Leading prefix blocks considered per prompt.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_MAX_BLOCKS",
        default_value_t = 128
    )]
    pub(crate) gateway_router_prefix_max_blocks: usize,

    /// Prefix block hashes remembered per worker (LRU).
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_ENTRIES",
        default_value_t = 16_384
    )]
    pub(crate) gateway_router_prefix_entries: usize,

    /// Outstanding requests a prefix-holding worker may carry beyond the
    /// least-loaded worker and still be preferred.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_BALANCE_THRESHOLD",
        default_value_t = 4
    )]
    pub(crate) gateway_router_balance_threshold: u32,

    /// Managed-KV utilization percent at which a worker is avoided while any
    /// eligible worker has headroom (0-100].
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_KV_PRESSURE_PCT",
        default_value_t = 90.0
    )]
    pub(crate) gateway_router_kv_pressure_pct: f64,

    /// Realtime relay (DS3.6): expose /v1/realtime/ws and forward
    /// izwi-realtime-v1 sessions to eligible speech_to_text workers.
    /// `off` keeps the gateway byte-identical to the pre-realtime shape.
    #[arg(long, env = "IZWI_GATEWAY_REALTIME", default_value = "off")]
    pub(crate) gateway_realtime: String,

    /// Bounded concurrent realtime sessions through the relay.
    #[arg(long, env = "IZWI_GATEWAY_REALTIME_MAX_SESSIONS", default_value_t = 64)]
    pub(crate) gateway_realtime_max_sessions: usize,

    /// End-to-end budget minted into each relayed worker session (ms).
    #[arg(
        long,
        env = "IZWI_GATEWAY_REALTIME_SESSION_BUDGET_MS",
        default_value_t = 600_000
    )]
    pub(crate) gateway_realtime_session_budget_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServerRole {
    Local,
    Gateway,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum GatewayTopology {
    Standalone,
    FleetOneGateway,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum BackendArg {
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
pub(crate) struct BindConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
}

pub(crate) fn validate_role_topology(
    role: ServerRole,
    topology: GatewayTopology,
) -> anyhow::Result<()> {
    if role == ServerRole::Local && topology != GatewayTopology::Standalone {
        anyhow::bail!("non-standalone gateway topology requires --role gateway");
    }
    Ok(())
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

pub(crate) fn resolve_serve_runtime_config(
    args: &ServerArgs,
) -> anyhow::Result<ServeRuntimeConfig> {
    resolve_serve_runtime_config_with_env(args, &ServeRuntimeConfigOverrides::from_env())
}

pub(crate) fn resolve_serve_runtime_config_with_env(
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
