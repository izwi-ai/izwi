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

use izwi_core::{RuntimeService, ServeRuntimeConfig, ServeRuntimeConfigOverrides};
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();

    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "izwi_server=warn,izwi_core=warn,tower_http=warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Izwi TTS Server");

    let serve_config = resolve_serve_runtime_config(&args);
    let config = serve_config.engine_config();
    info!("Models directory: {:?}", config.models_dir);

    // Create runtime service
    let runtime = RuntimeService::new(config)?;
    let state = AppState::new(runtime, &serve_config)?;

    info!("Runtime service initialized");

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

    // Spawn server with graceful shutdown
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(shutdown_state));

    info!("Server ready. Press Ctrl+C to stop.");
    server.await?;

    Ok(())
}

fn resolve_serve_runtime_config(args: &ServerArgs) -> ServeRuntimeConfig {
    let cli = ServeRuntimeConfigOverrides {
        host: args.host.clone(),
        port: args.port,
        backend: args.backend.as_ref().map(BackendArg::as_preference),
        ..ServeRuntimeConfigOverrides::default()
    };
    let env = ServeRuntimeConfigOverrides::from_env();
    ServeRuntimeConfig::from_sources(&ServeRuntimeConfigOverrides::default(), &env, &cli)
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
        std::env::remove_var("IZWI_MAX_BATCH_SIZE");
        std::env::remove_var("IZWI_NUM_THREADS");
        std::env::remove_var("IZWI_MAX_CONCURRENT");
        std::env::remove_var("IZWI_TIMEOUT");
        std::env::remove_var("IZWI_CORS");
        std::env::remove_var("IZWI_CORS_ORIGINS");
        std::env::remove_var("IZWI_NO_UI");
        std::env::remove_var("IZWI_UI_DIR");
        std::env::remove_var("MAX_CONCURRENT_REQUESTS");
        std::env::remove_var("REQUEST_TIMEOUT_SECS");
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
        let resolved = resolve_serve_runtime_config(&args);

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
        ]));

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

        let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"]));

        assert_eq!(resolved.host, "127.0.0.1");
        assert_eq!(resolved.port, 8088);
        clear_bind_env();
    }

    #[test]
    fn falls_back_to_defaults_without_cli_or_environment() {
        let _guard = env_lock();
        clear_bind_env();

        let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"]));

        assert_eq!(resolved.host, "0.0.0.0");
        assert_eq!(resolved.port, 8080);
        assert_eq!(resolved.max_batch_size, 8);
        assert!(resolved.num_threads >= 1);
        clear_bind_env();
    }

    #[test]
    fn falls_back_to_default_when_env_port_is_invalid() {
        let _guard = env_lock();
        clear_bind_env();
        std::env::set_var("IZWI_PORT", "not-a-port");

        let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"]));

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

        let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"]));

        assert_eq!(resolved.max_batch_size, 16);
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

        let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"]));

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

        let resolved = resolve_serve_runtime_config(&parse(&["izwi-server"]));

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
}
