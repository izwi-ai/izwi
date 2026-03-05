//! Izwi TTS Server - HTTP API for Qwen3-TTS inference

use clap::{Parser, ValueEnum};
use std::time::Duration;
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod app;
mod chat_store;
mod diarization_store;
mod error;
mod saved_voice_store;
mod speech_history_store;
mod state;
mod storage_layout;
mod transcription_store;

use izwi_core::{EngineConfig, RuntimeService};
use state::AppState;

#[derive(Debug, Parser)]
#[command(
    name = "izwi-server",
    about = "HTTP API server for Izwi local inference",
    version = env!("CARGO_PKG_VERSION")
)]
struct ServerArgs {
    /// Host to bind to
    #[arg(short = 'H', long)]
    host: Option<String>,

    /// Port to listen on
    #[arg(short, long)]
    port: Option<u16>,

    /// Backend preference (`auto`, `cpu`, `metal`, `cuda`)
    #[arg(long, value_enum, env = "IZWI_BACKEND")]
    backend: Option<BackendArg>,
}

#[derive(Debug, Clone, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl BackendArg {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BindConfig {
    host: String,
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();

    apply_backend_override(&args);

    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "izwi_server=warn,izwi_core=warn,tower_http=warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Izwi TTS Server");

    // Load configuration
    let config = EngineConfig::default();
    info!("Models directory: {:?}", config.models_dir);

    // Create runtime service
    let runtime = RuntimeService::new(config)?;
    let state = AppState::new(runtime)?;

    info!("Runtime service initialized");

    // Build router
    let app = api::create_router(state.clone());

    // Start server
    let bind = resolve_bind_config(args);
    let addr = format!("{}:{}", bind.host, bind.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Server listening on http://{}", addr);

    // Clone state for shutdown handler
    let shutdown_state = state.clone();

    // Spawn server with graceful shutdown
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(shutdown_state));

    info!("Server ready. Press Ctrl+C to stop.");
    server.await?;

    Ok(())
}

fn apply_backend_override(args: &ServerArgs) {
    if let Some(backend) = args.backend.as_ref() {
        std::env::set_var("IZWI_BACKEND", backend.as_str());
    }
}

fn resolve_bind_config(args: ServerArgs) -> BindConfig {
    BindConfig {
        host: args.host.unwrap_or_else(host_from_env_or_default),
        port: args.port.unwrap_or_else(port_from_env_or_default),
    }
}

fn host_from_env_or_default() -> String {
    match std::env::var("IZWI_HOST") {
        Ok(raw) => {
            let host = raw.trim();
            if host.is_empty() {
                warn!("Empty IZWI_HOST, falling back to 0.0.0.0");
                "0.0.0.0".to_string()
            } else {
                host.to_string()
            }
        }
        Err(_) => "0.0.0.0".to_string(),
    }
}

fn port_from_env_or_default() -> u16 {
    match std::env::var("IZWI_PORT") {
        Ok(raw) => match raw.trim().parse::<u16>() {
            Ok(parsed) => parsed,
            Err(_) => {
                warn!("Invalid IZWI_PORT='{}', falling back to 8080", raw);
                8080
            }
        },
        Err(_) => 8080,
    }
}

/// Wait for shutdown signal and cleanup
async fn shutdown_signal(state: AppState) {
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
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            info!("Received SIGTERM, shutting down...");
        },
    }

    const CLEANUP_TIMEOUT: Duration = Duration::from_secs(20);
    match tokio::time::timeout(CLEANUP_TIMEOUT, state.runtime.unload_all_models()).await {
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

    drop(state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("environment lock poisoned")
    }

    fn clear_bind_env() {
        std::env::remove_var("IZWI_HOST");
        std::env::remove_var("IZWI_PORT");
        std::env::remove_var("IZWI_BACKEND");
    }

    fn parse(args: &[&str]) -> ServerArgs {
        ServerArgs::try_parse_from(args).expect("arguments should parse")
    }

    #[test]
    fn backend_flag_overrides_environment() {
        let _guard = env_lock();
        clear_bind_env();
        std::env::set_var("IZWI_BACKEND", "cpu");

        let args = parse(&["izwi-server", "--backend", "cuda"]);
        apply_backend_override(&args);

        assert_eq!(std::env::var("IZWI_BACKEND").unwrap(), "cuda");
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
    fn cli_values_override_environment() {
        let _guard = env_lock();
        clear_bind_env();
        std::env::set_var("IZWI_HOST", "0.0.0.0");
        std::env::set_var("IZWI_PORT", "8080");

        let bind = resolve_bind_config(parse(&[
            "izwi-server",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]));

        assert_eq!(bind.host, "127.0.0.1");
        assert_eq!(bind.port, 9000);
        clear_bind_env();
    }

    #[test]
    fn uses_environment_when_cli_values_missing() {
        let _guard = env_lock();
        clear_bind_env();
        std::env::set_var("IZWI_HOST", "127.0.0.1");
        std::env::set_var("IZWI_PORT", "8088");

        let bind = resolve_bind_config(parse(&["izwi-server"]));

        assert_eq!(bind.host, "127.0.0.1");
        assert_eq!(bind.port, 8088);
        clear_bind_env();
    }

    #[test]
    fn falls_back_to_defaults_without_cli_or_environment() {
        let _guard = env_lock();
        clear_bind_env();

        let bind = resolve_bind_config(parse(&["izwi-server"]));

        assert_eq!(bind.host, "0.0.0.0");
        assert_eq!(bind.port, 8080);
    }

    #[test]
    fn falls_back_to_default_when_env_port_is_invalid() {
        let _guard = env_lock();
        clear_bind_env();
        std::env::set_var("IZWI_PORT", "not-a-port");

        let bind = resolve_bind_config(parse(&["izwi-server"]));

        assert_eq!(bind.port, 8080);
        clear_bind_env();
    }
}
