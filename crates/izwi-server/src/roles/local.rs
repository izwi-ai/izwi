//! The local role: one process that owns the runtime, the HTTP API, the durable stores and the
//! batch worker.

use anyhow::Context;
use izwi_core::backends::{self, BackendKind, BackendPreference, CudaRuntimeDiagnostics};
use izwi_core::{parse_model_variant, RuntimeService, ServeRuntimeConfig};
use izwi_hooks::EnterpriseHooks;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::batch_runtime::store::DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT;
use crate::batch_runtime::types::{
    DeviceClass, QueueClass, ResourceTarget, RuntimeBackendClass, WorkerResourceCapacity,
};
use crate::batch_runtime::worker::{
    BatchWorkerConfig, BatchWorkerDrain, BatchWorkerRunner, BatchWorkerSupervisor,
};
use crate::cli::{BindConfig, ServerArgs};
use crate::logging::{SERVICE_NAME, SERVICE_VERSION};
use crate::persistence::PersistenceContext;
use crate::shutdown::{
    cleanup_runtime_for_shutdown, serve_with_shutdown_plan, shutdown_requested,
    shutdown_worker_then_cleanup, ShutdownPhases, ShutdownPlan,
};
use crate::state::AppState;
use crate::{api, gateway_security, local_security, logging};

/// The local role: one process that owns the runtime, the HTTP API and the
/// durable stores.
pub(crate) async fn run_local(
    args: ServerArgs,
    serve_config: ServeRuntimeConfig,
    enterprise_hooks: EnterpriseHooks,
) -> anyhow::Result<()> {
    maybe_delegate_to_private_cuda_runtime(&serve_config)?;

    logging::init_tracing(args.log_format);

    info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        log_format = args.log_format.as_str(),
        "Starting Izwi TTS Server"
    );

    let api_key = local_security::LocalApiKey::from_env()?;
    let tuning = crate::env_config::tuning();
    local_security::validate_local_bind(
        &serve_config.host,
        &tuning.local_exposure,
        api_key.as_ref(),
        args.allow_unauthenticated_network || tuning.allow_unauthenticated_network,
    )?;
    local_security::validate_local_cors(&serve_config, api_key.as_ref())?;
    let loopback_bind = gateway_security::is_loopback_host(&serve_config.host);
    if api_key.is_some() {
        info!("Local API key required on /v1 and /internal routes");
    } else if !loopback_bind && tuning.local_exposure.loopback_published {
        info!(
            host = %serve_config.host,
            allowed_hosts = ?tuning.local_exposure.allowed_hosts,
            "IZWI_LOOPBACK_PUBLISHED: serving without an API key as a loopback-only server; \
             only loopback and IZWI_ALLOWED_HOSTS Host names are answered. Publish the port \
             on 127.0.0.1 only, or set IZWI_API_KEY."
        );
    } else if !loopback_bind {
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
    let mut config = serve_config.engine_config();
    info!("Models directory: {:?}", config.models_dir);
    // The Granite diagnostics flags reach the engine through its config; the
    // engine falls back to IZWI_GRANITE_DECODE_PROFILE / _SPEECH_DTYPE.
    if args.granite_decode_profile {
        config.granite_speech.decode_profile = true;
        info!("Granite ASR decode profiling enabled");
    }
    if let Some(dtype) = args
        .granite_speech_dtype
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        config.granite_speech.speech_dtype = Some(dtype.to_string());
        info!(dtype, "Granite ASR dtype override enabled");
    }

    // Create runtime service
    let runtime = RuntimeService::new(config)?;
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
    // Streamed speech, diarization jobs and streamed transcriptions run as
    // tasks of the node that accepted them; fail the ones this node left
    // running. On a shared database only its own (P1V-1).
    let mut unscoped_recovery = false;
    match state
        .speech_history_store
        .reconcile_stale_processing_records()
        .await
    {
        Ok(Some(reconciled)) if reconciled > 0 => {
            info!(reconciled, "Reconciled stale speech history records");
        }
        Ok(Some(_)) => {}
        Ok(None) => unscoped_recovery = true,
        Err(err) => startup_warnings.push(format!(
            "Failed to reconcile speech history records during startup: {err}"
        )),
    }
    match state.diarization_store.fail_interrupted_records().await {
        Ok(Some(failed)) if failed > 0 => {
            info!(failed, "Failed diarization jobs interrupted by a shutdown");
        }
        Ok(Some(_)) => {}
        Ok(None) => unscoped_recovery = true,
        Err(err) => startup_warnings.push(format!(
            "Failed to recover interrupted diarization jobs during startup: {err}"
        )),
    }
    match state.transcription_store.fail_interrupted_records().await {
        Ok(Some(failed)) if failed > 0 => {
            info!(failed, "Failed transcriptions interrupted by a shutdown");
        }
        Ok(Some(_)) => {}
        Ok(None) => unscoped_recovery = true,
        Err(err) => startup_warnings.push(format!(
            "Failed to recover interrupted transcriptions during startup: {err}"
        )),
    }
    if unscoped_recovery {
        // A configuration gap, not a readiness failure: the node serves.
        warn!(
            "Skipped failing interrupted jobs: this database may be shared by several nodes. \
             Set {} to a stable id unique to this node to recover the jobs it left running.",
            crate::node_identity::NODE_ID_ENV
        );
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
        local_shutdown_plan(),
    )
    .await
}

/// Serve the local role on an already bound listener while `preload` loads
/// the configured models. The server turns ready, and the batch worker
/// starts, only once preload finishes; a shutdown that begins first
/// abandons the preload. Shutdown follows `plan` like the gateway's, then
/// stops the batch worker and unloads the runtime.
pub(crate) async fn serve_local<P, S>(
    state: AppState,
    listener: tokio::net::TcpListener,
    app: axum::Router,
    preload: P,
    shutdown_requested: S,
    plan: ShutdownPlan,
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

    info!("Server accepting connections. Press Ctrl+C to stop.");
    let server_result = serve_with_shutdown_plan(
        |signal| izwi_gateway::http_serve::serve_router(listener, app, signal),
        LocalShutdown {
            state: state.clone(),
            batch_worker_drain,
        },
        shutdown_requested,
        plan,
    )
    .await;
    if server_result.is_none() {
        warn!(
            budget_secs = plan.budget.as_secs(),
            "HTTP shutdown budget spent; dropping remaining connections"
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

/// The local role's shutdown steps.
#[derive(Clone)]
struct LocalShutdown {
    state: AppState,
    batch_worker_drain: Arc<std::sync::OnceLock<BatchWorkerDrain>>,
}

impl ShutdownPhases for LocalShutdown {
    fn begin_prestop(&self) {
        self.state.lifecycle.mark_stopping();
    }

    fn begin_drain(&self) {
        self.state.lifecycle.mark_draining();
        self.state.runtime.begin_drain();
        if let Some(drain) = self.batch_worker_drain.get() {
            drain.begin();
        }
    }

    fn end_streams(&self) {
        self.state.stream_shutdown.send_replace(true);
    }
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
    if state.lifecycle.shutdown_started() {
        return None;
    }
    state.lifecycle.mark_ready();
    info!("Runtime service initialized");
    crate::api::studio::spawn_render_worker(state.clone());
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

pub(crate) fn local_batch_worker_queue_names() -> Vec<String> {
    [
        QueueClass::BatchAsr,
        QueueClass::LongFormAsr,
        QueueClass::BatchTts,
    ]
    .into_iter()
    .map(|queue| queue.as_db_value().to_string())
    .collect()
}

pub(crate) fn local_batch_worker_resources(
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

pub(crate) fn batch_worker_concurrency(runtime_capacity: usize, configured: Option<&str>) -> u32 {
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

pub(crate) fn batch_stage_execution_timeout() -> Option<Duration> {
    duration_secs_from_env("IZWI_BATCH_STAGE_TIMEOUT_SECS", 0, 30, 86_400)
}

pub(crate) fn batch_worker_drain_timeout() -> Duration {
    duration_secs_from_env("IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS", 20, 1, 300)
        .unwrap_or_else(|| Duration::from_secs(20))
}

pub(crate) fn http_shutdown_grace_timeout() -> Duration {
    duration_secs_from_env("IZWI_HTTP_SHUTDOWN_GRACE_SECS", 20, 1, 300)
        .unwrap_or_else(|| Duration::from_secs(20))
}

/// The pre-stop window at the start of shutdown
/// (`IZWI_HTTP_SHUTDOWN_PRESTOP_SECS`, 0-60, default 0: a desktop or
/// command-line server stops at once; set it behind a load balancer).
pub(crate) fn http_shutdown_prestop() -> Duration {
    Duration::from_secs(
        std::env::var("IZWI_HTTP_SHUTDOWN_PRESTOP_SECS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(0)
            .min(60),
    )
}

/// The local role's shutdown budget: the pre-stop window, then the HTTP
/// drain grace.
pub(crate) fn local_shutdown_plan() -> ShutdownPlan {
    let prestop = http_shutdown_prestop();
    ShutdownPlan {
        prestop,
        budget: prestop + http_shutdown_grace_timeout(),
    }
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

pub(crate) fn configured_preload_models() -> Vec<String> {
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

pub(crate) fn warmup_preloaded_models_enabled() -> bool {
    env_bool("IZWI_WARMUP_PRELOADED_MODELS").unwrap_or(true)
}

pub(crate) fn asr_warmup_duration_ms() -> u32 {
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
