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

use clap::Parser;

mod api;
mod app;
pub use app::realtime_protocol;
pub mod artifact_store;
pub mod batch_runtime;
mod chat_store;
mod cli;
mod db;
mod diarization_store;
mod entity;
mod env_config;
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
mod roles;
mod saved_voice_store;
mod shutdown;
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

use cli::{resolve_serve_runtime_config, validate_role_topology, ServerArgs, ServerRole};
use izwi_core::{RuntimeService, ServeRuntimeConfig};
use izwi_hooks::EnterpriseHooks;
// The gateway's accelerator-free components live in izwi-gateway; these
// aliases keep their `crate::` paths for the gateway role still hosted here.
use izwi_gateway::{
    gateway_deployments, gateway_rate_quota, gateway_shared_approvals, gateway_tenant_concurrency,
    gateway_worker_tls,
};
use roles::{gateway::run_gateway, local::run_local};
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

pub async fn run_from_cli(enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    let args = ServerArgs::parse();
    run_with_args(args, enterprise_hooks).await
}

async fn run_with_args(args: ServerArgs, enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    validate_role_topology(args.role, args.gateway_topology)?;
    env_config::init()?;
    let serve_config = resolve_serve_runtime_config(&args)?;
    match args.role {
        ServerRole::Gateway => run_gateway(args, serve_config, enterprise_hooks).await,
        ServerRole::Local => run_local(args, serve_config, enterprise_hooks).await,
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
