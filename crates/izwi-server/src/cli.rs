//! The `izwi-server` command line: arguments, roles and how they resolve to the runtime
//! configuration.

use clap::{Parser, Subcommand, ValueEnum};
use izwi_core::backends::BackendPreference;
use izwi_core::{ServeRuntimeConfig, ServeRuntimeConfigOverrides};
pub(crate) use izwi_gateway::cli::{BackendArg, GatewayArgs, GatewayTopology};
use izwi_gateway::{GatewayConfig, GatewayServeConfig};
use std::path::PathBuf;

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
    /// role requires IZWI_API_KEY unless IZWI_LOOPBACK_PUBLISHED or
    /// --allow-unauthenticated-network.
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

    /// The gateway role's flags (`--role gateway`); the standalone
    /// `izwi-gateway` binary takes the same ones.
    #[command(flatten)]
    pub(crate) gateway: GatewayArgs,

    #[command(subcommand)]
    pub(crate) command: Option<ServerCommand>,
}

/// One-shot commands run instead of serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Subcommand)]
pub(crate) enum ServerCommand {
    /// Exit 0 when this server's listener answers /readyz with 2xx. The
    /// container HEALTHCHECK; it reads the same flags and environment as
    /// the server to find the port.
    Healthcheck,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ServerRole {
    Local,
    Gateway,
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
        backend: args.backend.as_ref().map(backend_preference),
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

fn backend_preference(backend: &BackendArg) -> BackendPreference {
    match backend {
        BackendArg::Auto => BackendPreference::Auto,
        BackendArg::Cpu => BackendPreference::Cpu,
        BackendArg::Metal => BackendPreference::Metal,
        BackendArg::Cuda => BackendPreference::Cuda,
    }
}

/// The gateway listener's settings in the server's runtime configuration.
pub(crate) fn gateway_serve_config(serve_config: &ServeRuntimeConfig) -> GatewayServeConfig {
    GatewayServeConfig {
        host: serve_config.host.clone(),
        port: serve_config.port,
        request_timeout_secs: serve_config.request_timeout_secs,
        cors_enabled: serve_config.cors_enabled,
        cors_origins: serve_config.cors_origins.clone(),
    }
}

/// What `--role gateway` runs from: the gateway flags, the listener settings
/// of the server's runtime configuration and the server's OpenAI
/// compatibility profile, exactly as the role read them before it moved to
/// izwi-gateway.
pub(crate) fn gateway_config(args: ServerArgs, serve_config: &ServeRuntimeConfig) -> GatewayConfig {
    GatewayConfig {
        args: args.gateway,
        serve: gateway_serve_config(serve_config),
        backend: args.backend,
        compat_profile: crate::env_config::tuning().openai_compat_profile,
    }
}
