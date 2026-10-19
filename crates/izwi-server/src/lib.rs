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

use anyhow::Context as _;
use clap::Parser;
use std::time::Duration;

mod api;
mod app;
pub use app::realtime_protocol;
pub mod artifact_store;
pub mod batch_runtime;
mod chat_store;
mod cli;
mod diarization_store;
mod entity;
mod env_config;
mod error;
mod local_security;
mod logging;
pub mod media_ingest;
mod node_identity;
mod onboarding_store;
mod persistence;
mod roles;
mod saved_voice_store;
mod shutdown;
mod speech_history_store;
mod speech_resource_budget;
mod speech_spool;
mod speech_text_list;
mod state;
mod studio_project_store;
#[cfg(test)]
mod test_support;
mod text_preview;
mod transcription_store;
mod voice_memory;
mod voice_observation_store;
mod voice_store;

use cli::{resolve_serve_runtime_config, validate_role_topology, ServerArgs, ServerRole};
use izwi_core::{RuntimeService, ServeRuntimeConfig};
// The durable store lives in izwi-db, shared with the gateway; these aliases
// keep its `crate::db`, `crate::storage_layout` and `crate::voice_defaults`
// paths.
use izwi_db as db;
use izwi_db::{storage_layout, voice_defaults};
use izwi_hooks::EnterpriseHooks;
// The gateway role and the HTTP foundations both roles share live in
// izwi-gateway; these aliases keep their `crate::` paths.
use izwi_gateway::{gateway_security, ids};
use roles::local::run_local;
use state::AppState;

pub use izwi_gateway::worker_registry;
pub use izwi_gateway::{
    FleetPartition, FleetPartitionError, FleetStore, GatewayPerimeterConfig,
    GatewayPerimeterConfigError, GatewayRateQuotaConfig, GatewayRateQuotaConfigError, GatewayState,
    GatewayTenantConcurrencyConfig, GatewayTenantConcurrencyConfigError, RemoteChatDispatchConfig,
    RemoteChatDispatcher,
};

/// The gateway's router with the listener settings of `serve_config`.
///
/// Kept for callers that hold the server's runtime configuration; the
/// gateway itself takes [`izwi_gateway::GatewayServeConfig`].
pub fn create_gateway_router(
    state: GatewayState,
    serve_config: &ServeRuntimeConfig,
) -> axum::Router {
    izwi_gateway::create_gateway_router(state, &cli::gateway_serve_config(serve_config))
}

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

/// How long blocking work still running after shutdown cleanup (for example a
/// model unload waiting in `Device::synchronize` on queued GPU work) may delay
/// process exit. Cleanup itself is bounded by its own timeout first.
pub const PROCESS_EXIT_GRACE: Duration = Duration::from_secs(2);

/// Runs the server from the command line on a Tokio runtime of its own:
/// the entry point for a synchronous `main`.
///
/// The configuration is resolved first, while the process is still
/// single-threaded, so the local role sizes the engine's CPU compute
/// libraries through their environment variables
/// ([`izwi_core::env::configure_process_threads`]) before any thread that
/// might read the environment exists. The runtime's shutdown is bounded by
/// [`PROCESS_EXIT_GRACE`], so SIGTERM exits after the cleanup budget instead
/// of waiting on a stuck GPU drain.
pub fn run_from_cli_blocking(enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    let (args, serve_config) = resolve_cli()?;
    if matches!(args.role, ServerRole::Local) {
        // SAFETY: no runtime or other thread has started yet.
        unsafe { izwi_core::env::configure_process_threads(serve_config.num_threads) };
    }
    run_with_bounded_exit(
        run(args, serve_config, enterprise_hooks),
        PROCESS_EXIT_GRACE,
    )
}

/// The name an earlier community binary's `main` called; the same as
/// [`run_from_cli_blocking`].
pub fn run_server_process(enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    run_from_cli_blocking(enterprise_hooks)
}

fn run_with_bounded_exit<T>(
    future: impl std::future::Future<Output = anyhow::Result<T>>,
    exit_grace: Duration,
) -> anyhow::Result<T> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the Tokio runtime")?;
    let result = runtime.block_on(future);
    // Dropping a runtime waits for every blocking task without limit.
    runtime.shutdown_timeout(exit_grace);
    result
}

/// Runs the server from the command line inside the caller's Tokio runtime
/// (`#[tokio::main] async fn main`).
///
/// The legacy async entry, kept for binaries that already own a runtime.
/// It sets the same CPU thread hints as [`run_from_cli_blocking`], but by
/// then the runtime's worker threads exist, so the environment write can
/// race a concurrent read, as it always has on this path. New binaries
/// should call [`run_from_cli_blocking`] from a synchronous `main`.
pub async fn run_from_cli(enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    let (args, serve_config) = resolve_cli()?;
    if matches!(args.role, ServerRole::Local) {
        // SAFETY: not guaranteed (see above); this path has always written
        // the hints after the runtime started, so it is no worse than before.
        unsafe { izwi_core::env::configure_process_threads(serve_config.num_threads) };
    }
    run(args, serve_config, enterprise_hooks).await
}

fn resolve_cli() -> anyhow::Result<(ServerArgs, ServeRuntimeConfig)> {
    let args = ServerArgs::parse();
    validate_role_topology(args.role, args.gateway.gateway_topology)?;
    env_config::init()?;
    let serve_config = resolve_serve_runtime_config(&args)?;
    Ok((args, serve_config))
}

async fn run(
    args: ServerArgs,
    serve_config: ServeRuntimeConfig,
    enterprise_hooks: EnterpriseHooks,
) -> anyhow::Result<()> {
    match args.role {
        // The role izwi-gateway runs, kept for deployments that start the
        // server with `--role gateway`.
        ServerRole::Gateway => {
            logging::init_tracing(args.log_format);
            let config = cli::gateway_config(args, &serve_config);
            izwi_gateway::run_gateway(config, enterprise_hooks).await
        }
        ServerRole::Local => run_local(args, serve_config, enterprise_hooks).await,
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
