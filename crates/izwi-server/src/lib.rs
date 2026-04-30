//! Izwi TTS Server - HTTP API for Qwen3-TTS inference

use clap::{Parser, ValueEnum};
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use tokio::signal;
use tracing::{info, warn};

mod api;
mod app;
mod chat_store;
mod diarization_store;
mod error;
mod ids;
mod logging;
mod onboarding_store;
mod saved_voice_store;
mod speech_history_store;
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

use izwi_core::backends::{self, BackendPreference, CudaRuntimeDiagnostics};
use izwi_core::{
    parse_model_variant, RuntimeService, ServeRuntimeConfig, ServeRuntimeConfigOverrides,
};
use izwi_hooks::EnterpriseHooks;
use logging::{LogFormat, SERVICE_NAME, SERVICE_VERSION};
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

    /// Log output format (`text`, `json`)
    #[arg(long, value_enum, env = "IZWI_LOG_FORMAT", default_value = "text")]
    log_format: LogFormat,
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
    maybe_delegate_to_private_cuda_runtime(&args)?;

    logging::init_tracing(args.log_format);

    info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        log_format = args.log_format.as_str(),
        "Starting Izwi TTS Server"
    );

    let serve_config = resolve_serve_runtime_config(&args);
    let config = serve_config.engine_config();
    info!("Models directory: {:?}", config.models_dir);

    // Create runtime service
    let runtime = RuntimeService::new(config)?;
    let state = AppState::with_enterprise_hooks(runtime, &serve_config, enterprise_hooks)?;
    let mut startup_warnings = preload_configured_models(&state).await;
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

fn maybe_delegate_to_private_cuda_runtime(args: &ServerArgs) -> anyhow::Result<()> {
    if cfg!(feature = "cuda") || backends::private_cuda_runtime_active() {
        return Ok(());
    }

    let serve_config = resolve_serve_runtime_config(args);
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

    state.lifecycle.mark_draining();

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
    use crate::test_support::env_lock;

    fn clear_bind_env() {
        std::env::remove_var("IZWI_HOST");
        std::env::remove_var("IZWI_PORT");
        std::env::remove_var("IZWI_BACKEND");
        std::env::remove_var("IZWI_LOG_FORMAT");
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
        std::env::remove_var("IZWI_PRELOAD_MODELS");
        std::env::remove_var("IZWI_WARMUP_PRELOADED_MODELS");
        std::env::remove_var("IZWI_ASR_WARMUP_DURATION_MS");
    }

    fn parse(args: &[&str]) -> ServerArgs {
        ServerArgs::try_parse_from(args).expect("arguments should parse")
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
