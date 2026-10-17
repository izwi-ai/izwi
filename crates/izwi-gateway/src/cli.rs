//! The gateway's command line: the role's flags, which `izwi-server --role
//! gateway` also takes, and the standalone `izwi-gateway` binary's own
//! listener and logging flags.

use clap::{Args, Parser, ValueEnum};

use crate::gateway_deployments;
use crate::logging::LogFormat;

/// The `izwi-gateway` binary's command line.
#[derive(Debug, Parser)]
#[command(
    name = "izwi-gateway",
    about = "Izwi public API gateway: routes OpenAI-compatible requests to approved inference workers",
    version = env!("CARGO_PKG_VERSION")
)]
pub struct GatewayCli {
    /// Host to bind to (default 127.0.0.1). A non-loopback plaintext bind
    /// requires IZWI_GATEWAY_TRUSTED_INGRESS_TLS.
    #[arg(short = 'H', long, env = "IZWI_HOST")]
    pub host: Option<String>,

    /// Port to listen on (default 8080).
    #[arg(short, long, env = "IZWI_PORT")]
    pub port: Option<u16>,

    /// Worker backends chat may route to (`auto`, `cpu`, `metal`, `cuda`).
    #[arg(long, value_enum, env = "IZWI_BACKEND")]
    pub backend: Option<BackendArg>,

    /// Log output format (`text`, `json`)
    #[arg(long, value_enum, env = "IZWI_LOG_FORMAT", default_value = "text")]
    pub log_format: LogFormat,

    #[command(flatten)]
    pub gateway: GatewayArgs,
}

/// The gateway role's flags.
#[derive(Debug, Clone, Args)]
pub struct GatewayArgs {
    /// Legacy one-worker form of `--gateway-worker-endpoint`: the gateway
    /// routes to this worker through the same registry as an endpoint list.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_ENDPOINT")]
    pub worker_endpoint: Option<String>,

    /// Worker-network policy (`standalone` or `fleet-one-gateway`).
    #[arg(
        long,
        value_enum,
        env = "IZWI_GATEWAY_TOPOLOGY",
        default_value = "standalone"
    )]
    pub gateway_topology: GatewayTopology,

    /// Approved private worker URLs for registry-backed routing. Repeat this
    /// option (or use a comma-separated environment value) to add capacity.
    #[arg(
        long = "gateway-worker-endpoint",
        env = "IZWI_GATEWAY_WORKER_ENDPOINTS",
        value_delimiter = ',',
        value_name = "URL"
    )]
    pub gateway_worker_endpoints: Vec<String>,

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
    pub gateway_worker_approvals: Vec<gateway_deployments::GatewayWorkerApproval>,

    /// Rotatable private worker credential identifier required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_CREDENTIAL_ID")]
    pub worker_credential_id: Option<String>,

    /// Private worker bearer token required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_BEARER_TOKEN", hide_env_values = true)]
    pub worker_bearer_token: Option<String>,

    /// Accepted with `--worker-endpoint` for compatibility and ignored: the
    /// registry fences every attempt to the incarnation it observed and
    /// adopts a restarted worker.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_INCARNATION")]
    pub worker_incarnation: Option<String>,

    /// Chat deployment identifier of the legacy worker endpoint forms.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_DEPLOYMENT")]
    pub worker_deployment: Option<String>,

    /// Public model served through the gateway.
    #[arg(long, env = "IZWI_GATEWAY_PUBLIC_MODEL")]
    pub public_model: Option<String>,

    /// Expected non-zero model generation required by gateway mode.
    #[arg(long, env = "IZWI_GATEWAY_MODEL_GENERATION")]
    pub worker_model_generation: Option<u64>,

    /// Policy revision attested on private worker requests.
    #[arg(
        long,
        env = "IZWI_GATEWAY_POLICY_REVISION",
        default_value = "local-policy-v1"
    )]
    pub gateway_policy_revision: String,

    /// Maximum gateway requests concurrently in flight to the pinned worker.
    #[arg(long, env = "IZWI_GATEWAY_MAX_IN_FLIGHT", default_value_t = 32)]
    pub gateway_max_in_flight: usize,

    /// Maximum time the selected worker may spend establishing runtime ownership.
    #[arg(long, env = "IZWI_GATEWAY_WORKER_QUEUE_WAIT_MS", default_value_t = 250)]
    pub gateway_worker_queue_wait_ms: u64,

    /// Maximum wait for private invocation response headers and admission.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_ADMISSION_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    pub gateway_worker_admission_timeout_ms: u64,

    /// Maximum wait after admission for the first useful worker output.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_MS",
        default_value_t = 60_000
    )]
    pub gateway_worker_first_output_timeout_ms: u64,

    /// Extra first-output allowance per KiB of prompt, so a long prompt's
    /// prefill (slow on CPU) is not cancelled by the short-prompt deadline.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_PER_KIB_MS",
        default_value_t = 1_000
    )]
    pub gateway_worker_first_output_timeout_per_kib_ms: u64,

    /// Maximum idle time between useful worker output events.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_PROGRESS_IDLE_TIMEOUT_MS",
        default_value_t = 30_000
    )]
    pub gateway_worker_progress_idle_timeout_ms: u64,

    /// Total gateway shutdown budget after SIGTERM: the pre-stop window,
    /// then draining in-flight requests. The default fits Kubernetes'
    /// default 30 s termination grace period.
    #[arg(long, env = "IZWI_GATEWAY_SHUTDOWN_BUDGET_SECS", default_value_t = 25)]
    pub gateway_shutdown_budget_secs: u64,

    /// Pre-stop window at the start of the shutdown budget: readiness turns
    /// unready while requests are still served, so load balancers stop
    /// routing here before the listener closes.
    #[arg(long, env = "IZWI_GATEWAY_PRESTOP_SECS", default_value_t = 5)]
    pub gateway_prestop_secs: u64,

    /// Deadline for receiving a public request body. A slow upload holds a
    /// global admission permit at most this long.
    #[arg(
        long,
        env = "IZWI_GATEWAY_BODY_READ_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    pub gateway_body_read_timeout_ms: u64,

    /// Maximum wait while relaying an event to a connected slow consumer.
    #[arg(
        long,
        env = "IZWI_GATEWAY_SLOW_CONSUMER_TIMEOUT_MS",
        default_value_t = 5_000
    )]
    pub gateway_slow_consumer_timeout_ms: u64,

    /// Receiver-clock lifetime of a worker status observation.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_STATUS_TTL_MS",
        default_value_t = 10_000
    )]
    pub gateway_worker_status_ttl_ms: u64,

    /// Status refresh interval for each configured registry worker.
    #[arg(
        long,
        env = "IZWI_GATEWAY_WORKER_STATUS_POLL_MS",
        default_value_t = 2_000
    )]
    pub gateway_worker_status_poll_ms: u64,

    /// Prefix-aware routing: prefer the worker recently routed the longest
    /// leading part of a chat prompt, within the balance threshold below.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_AFFINITY",
        default_value = "on"
    )]
    pub gateway_router_prefix_affinity: String,

    /// Bytes of message stream per hashed prefix block.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_BLOCK_BYTES",
        default_value_t = 256
    )]
    pub gateway_router_prefix_block_bytes: usize,

    /// Leading prefix blocks considered per prompt.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_MAX_BLOCKS",
        default_value_t = 128
    )]
    pub gateway_router_prefix_max_blocks: usize,

    /// Prefix block hashes remembered per worker (LRU).
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_PREFIX_ENTRIES",
        default_value_t = 16_384
    )]
    pub gateway_router_prefix_entries: usize,

    /// Outstanding requests a prefix-holding worker may carry beyond the
    /// least-loaded worker and still be preferred.
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_BALANCE_THRESHOLD",
        default_value_t = 4
    )]
    pub gateway_router_balance_threshold: u32,

    /// Managed-KV utilization percent at which a worker is avoided while any
    /// eligible worker has headroom (0-100].
    #[arg(
        long,
        env = "IZWI_GATEWAY_ROUTER_KV_PRESSURE_PCT",
        default_value_t = 90.0
    )]
    pub gateway_router_kv_pressure_pct: f64,

    /// Realtime relay (DS3.6): expose /v1/realtime/ws and forward
    /// izwi-realtime-v1 sessions to eligible speech_to_text workers.
    /// `off` keeps the gateway byte-identical to the pre-realtime shape.
    #[arg(long, env = "IZWI_GATEWAY_REALTIME", default_value = "off")]
    pub gateway_realtime: String,

    /// Bounded concurrent realtime sessions through the relay.
    #[arg(long, env = "IZWI_GATEWAY_REALTIME_MAX_SESSIONS", default_value_t = 64)]
    pub gateway_realtime_max_sessions: usize,

    /// End-to-end budget minted into each relayed worker session (ms).
    #[arg(
        long,
        env = "IZWI_GATEWAY_REALTIME_SESSION_BUDGET_MS",
        default_value_t = 600_000
    )]
    pub gateway_realtime_session_budget_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GatewayTopology {
    Standalone,
    FleetOneGateway,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum BackendArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
}
