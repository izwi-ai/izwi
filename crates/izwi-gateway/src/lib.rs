//! The Izwi public API gateway: an OpenAI-compatible HTTP front end that
//! authenticates, admits and routes requests to approved inference workers.
//! It runs no model, so it never links the inference engine:
//! check-serving-boundary.sh keeps this crate free of izwi-core and every
//! accelerator.
//!
//! The `izwi-gateway` binary runs it ([`run_from_cli_blocking`]);
//! `izwi-server --role gateway` delegates to the same [`run_gateway`].
//! izwi-server's local role also reuses the HTTP foundations here (errors,
//! request context, the OpenAI chat pipeline, the realtime wire formats),
//! which is why their module paths mirror izwi-server's.

// The lint policy these modules had in izwi-server: request paths carry the
// complete request context, and realtime alignment indexes explicitly.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// Async tests hold a process-wide environment lock across awaits so parallel
// tests cannot observe transient environment overrides.
#![cfg_attr(test, allow(clippy::await_holding_lock))]

use anyhow::Context as _;
use clap::Parser;
use izwi_hooks::EnterpriseHooks;

pub mod api;
pub mod app;
pub mod cli;
pub mod config;
pub mod error;
pub mod gateway;
pub mod gateway_deployments;
pub mod gateway_fleet;
pub mod gateway_principal_keys;
pub mod gateway_rate_quota;
pub mod gateway_security;
pub mod gateway_shared_approvals;
pub mod gateway_tenant_concurrency;
pub mod gateway_worker_tls;
pub mod ids;
pub mod lifecycle;
pub mod logging;
pub mod roles;
pub mod shutdown;
#[cfg(test)]
mod test_support;
pub mod worker_registry;

// The durable store, shared with izwi-server.
use izwi_db as db;
use izwi_db::storage_layout;
#[cfg(all(test, feature = "db-postgres"))]
use izwi_db::voice_defaults;

pub use app::remote_chat_dispatch::{RemoteChatDispatchConfig, RemoteChatDispatcher};
pub use config::{GatewayConfig, GatewayServeConfig};
pub use gateway::{create_gateway_router, GatewayState};
pub use gateway_fleet::{FleetPartition, FleetPartitionError, FleetStore};
pub use gateway_rate_quota::{GatewayRateQuotaConfig, GatewayRateQuotaConfigError};
pub use gateway_security::{GatewayPerimeterConfig, GatewayPerimeterConfigError};
pub use gateway_tenant_concurrency::{
    GatewayTenantConcurrencyConfig, GatewayTenantConcurrencyConfigError,
};
pub use roles::gateway::run_gateway;

/// Runs the gateway from the command line on a Tokio runtime of its own:
/// the entry point for a synchronous `main`.
pub fn run_from_cli_blocking(enterprise_hooks: EnterpriseHooks) -> anyhow::Result<()> {
    let cli = cli::GatewayCli::parse();
    let log_format = cli.log_format;
    let config = GatewayConfig::from_cli(cli)?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the Tokio runtime")?
        .block_on(async move {
            logging::init_tracing(log_format, logging::GATEWAY_SERVICE_NAME);
            run_gateway(config, enterprise_hooks).await
        })
}
