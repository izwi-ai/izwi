use anyhow::{bail, Context};
use env_config::{WORKER_TLS_CERT_REF_ENV, WORKER_TLS_CLIENT_CA_REF_ENV, WORKER_TLS_KEY_REF_ENV};
use izwi_core::{
    backends::RuntimeDeviceAssignment, EngineConfig, ModelVariant, RealtimeAsrSessionConfig,
    RuntimeHealth, RuntimeService,
};
use izwi_serving_fence::{
    try_acquire_worker_fences, LockLease, LockNamespace, WorkerFenceLeases, WorkerLockPaths,
    MANAGED_WORKER_ENV, WORKER_EXECUTION_REPRESENTATION_ENV, WORKER_GENERATION_FENCE_ENV,
    WORKER_INCARNATION_ENV, WORKER_MAX_CONTEXT_TOKENS_ENV, WORKER_MAX_OUTPUT_TOKENS_ENV,
    WORKER_MODEL_LOAD_LOCK_ENV, WORKER_MODEL_LOAD_SLOTS_ENV, WORKER_OWNERSHIP_LOCK_ENV,
    WORKER_PRECISION_ENV, WORKER_TASK_ENV,
};
use izwi_serving_protocol::env::EnvSnapshot;
use izwi_serving_protocol::secret_file::{file_reference_path, read_secret_file, SecretFile};
use izwi_serving_protocol::*;
use izwi_serving_worker::{
    warm_up_asr_runtime, warm_up_chat_runtime, warm_up_tts_runtime, RuntimeChatExecutor,
    RuntimeRealtimeAsrExecutor, RuntimeRealtimeTtsExecutor, WorkerConfig, WorkerService,
    DEFAULT_ATTEMPT_RETENTION, DEFAULT_EVENT_CHANNEL_CAPACITY, DEFAULT_MAX_EVENT_BYTES,
    DEFAULT_MAX_REQUEST_BYTES, DEFAULT_MAX_RETAINED_ATTEMPTS, REALTIME_SESSION_DEFAULT_LIMITS,
};
use std::{
    collections::BTreeSet, net::SocketAddr, path::Path, path::PathBuf, sync::Arc, time::Duration,
};
use tokio::io::AsyncReadExt;
use tokio_rustls::rustls::pki_types::pem::{self, PemObject};
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};

mod env_config;

const MAX_TLS_PEM_FILE_BYTES: u64 = 256 * 1024;

/// Text logs by default; `IZWI_LOG_FORMAT=json` for one JSON object per line.
fn init_tracing(env: &EnvSnapshot) -> anyhow::Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "izwi_serving_worker=info,izwi_core=info".into());
    // Request spans go to an OTLP collector only when one is configured.
    let registry = tracing_subscriber::registry().with(filter).with(
        izwi_serving_client::otlp::otlp_layer_from_env("izwi-serving-worker"),
    );
    match env.get("IZWI_LOG_FORMAT")? {
        None | Some("text") => registry.with(tracing_subscriber::fmt::layer()).init(),
        Some("json") => registry
            .with(izwi_serving_worker::json_log_layer(std::io::stdout))
            .init(),
        Some(other) => anyhow::bail!("IZWI_LOG_FORMAT must be text or json, not {other:?}"),
    }
    Ok(())
}

/// Synchronous so the process-wide startup that writes the environment runs
/// before any other thread exists: the Tokio runtime and the OTLP exporter
/// start after it.
fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        return healthcheck(&env_config::capture());
    }
    // A worker started without the serving allocator settings (macOS: not
    // by the supervisor) restarts itself with them; see izwi_env::allocator.
    if let Err(error) = izwi_env::reexec_with_serving_allocator_env() {
        eprintln!(
            "izwi-serving-worker: could not restart with the serving allocator settings \
             ({error}); continuing with the default allocator"
        );
    }
    // Every IZWI_* variable this worker reads, read once.
    let env = env_config::capture();
    let process = WorkerProcessConfig::from_env(&env)?;
    // SAFETY: no runtime or other thread has started yet.
    unsafe { izwi_core::env::configure_process_threads(process.thread_budget()) };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("start the Tokio runtime")?
        .block_on(run(env, process))
}

/// `izwi-serving-worker healthcheck`, the container HEALTHCHECK: GET the
/// status route from this worker's own plaintext listener with the
/// credentials the gateway presents. They come from the environment, so the
/// bearer token never appears in argv.
fn healthcheck(env: &EnvSnapshot) -> anyhow::Result<()> {
    use izwi_serving_protocol::health_probe::{probe_address, probe_http, HEALTH_PROBE_TIMEOUT};
    if env
        .get(WORKER_TLS_CERT_REF_ENV)?
        .is_some_and(|reference| !reference.trim().is_empty())
    {
        bail!("healthcheck probes plaintext workers only; this worker serves TLS");
    }
    let bind = env
        .get_or("IZWI_WORKER_BIND", "127.0.0.1:9470")?
        .parse::<SocketAddr>()
        .context("parse IZWI_WORKER_BIND")?;
    let addr = probe_address(&bind.ip().to_string(), bind.port())?;
    let authorization = format!("Bearer {}", env.required("IZWI_WORKER_BEARER_TOKEN")?);
    let status = probe_http(
        addr,
        WORKER_STATUS_PATH,
        &[
            ("authorization", &authorization),
            (
                SERVICE_CREDENTIAL_ID_HEADER,
                env.required("IZWI_WORKER_CREDENTIAL_ID")?,
            ),
        ],
        HEALTH_PROBE_TIMEOUT,
    )
    .with_context(|| format!("GET http://{addr}{WORKER_STATUS_PATH} failed"))?;
    anyhow::ensure!(
        (200..300).contains(&status),
        "GET http://{addr}{WORKER_STATUS_PATH} returned {status}"
    );
    Ok(())
}

async fn run(env: EnvSnapshot, process: WorkerProcessConfig) -> anyhow::Result<()> {
    init_tracing(&env)?;
    if process.insecure_plaintext_bind() {
        tracing::warn!(
            address = %process.bind,
            "binding plaintext worker to non-loopback address via IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND"
        );
    }
    let managed_locks = managed_lock_context(&env, &process)?;
    let _fences = acquire_managed_worker_fences(&process, managed_locks.as_ref())?;
    if let DeviceAssignment::Cpu { affinity, .. } = &process.assignment {
        if !affinity.is_empty() {
            tracing::warn!(?affinity, "CPU affinity is advisory on this worker build");
        }
    }
    verify_artifact_manifest(&process)?;
    // Serialize the allocation-heavy startup section across workers on this
    // node. Static supervisor validation bounds resident budgets; the runtime's
    // resource authority performs the exact load-peak reservation below.
    let model_load_stage = acquire_model_load_stage(&process, managed_locks.as_ref())?;
    // Serving-policy env (mirrors the single-process server's resolution):
    // an explicit prefix-caching choice replaces the catalog-auto default
    // (true = explicit enablement requiring a namespace, false = kill
    // switch); with no explicit choice, catalog-auto engages reuse per
    // loaded model where the catalog cell has backend-lane evidence.
    // Chunked prefill remains opt-in.
    let explicit_prefix_caching = env.flag("IZWI_ENABLE_PREFIX_CACHING")?;
    let enable_prefix_caching = explicit_prefix_caching.unwrap_or(false);
    let prefix_reuse_catalog_auto = match explicit_prefix_caching {
        Some(_) => false,
        None => env.flag_or(
            izwi_core::serve_runtime::ENV_PREFIX_REUSE_AUTO,
            izwi_core::serve_runtime::PREFIX_REUSE_CATALOG_AUTO_DEFAULT,
        )?,
    };
    let managed_prefix_cache_salt = env
        .get("IZWI_MANAGED_PREFIX_CACHE_SALT")?
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    if enable_prefix_caching && managed_prefix_cache_salt.is_none() {
        bail!("IZWI_ENABLE_PREFIX_CACHING=1 requires IZWI_MANAGED_PREFIX_CACHE_SALT");
    }
    // The worker's concurrency: `IZWI_MAX_NUM_SEQS`, else an explicit
    // `IZWI_WORKER_MAX_ACTIVE` (the engine then serves that many sequences),
    // else the runtime's automatic sizing for the loaded model and backend.
    let requested_seqs = process.max_num_seqs.or(process.max_active_invocations);
    let enable_chunked_prefill = env
        .flag("IZWI_ENABLE_CHUNKED_PREFILL")?
        .unwrap_or_else(|| izwi_core::config::default_chunked_prefill_for(requested_seqs));
    let engine_defaults = izwi_core::config::EngineConfig::default();
    let max_sequence_length =
        match env.get("IZWI_MAX_SEQUENCE_LENGTH")? {
            Some(raw) if !raw.trim().is_empty() => Some(
                izwi_core::config::ContextLengthPreference::explicit(raw.trim().parse().map_err(
                    |error| anyhow::anyhow!("invalid IZWI_MAX_SEQUENCE_LENGTH: {error}"),
                )?)?,
            ),
            _ => None,
        };
    let engine = EngineConfig {
        models_dir: process.models_dir.clone(),
        max_loaded_models: Some(1),
        // Live runtime jobs must cover every admitted invocation.
        max_queued_requests: engine_defaults
            .max_queued_requests
            .max(requested_seqs.unwrap_or(0)),
        // Rows follow `max_num_seqs`, or the backend's automatic sizing.
        max_num_seqs: requested_seqs,
        max_scheduler_batch_size: 0,
        max_retained_sequences: 0,
        max_staged_transactions: 0,
        num_threads: process.thread_budget(),
        enable_prefix_caching,
        prefix_reuse_catalog_auto,
        managed_prefix_cache_salt,
        // Unset stays automatic: each model sizes its prefix budget at load.
        max_prefix_cache_pages: env.parse("IZWI_MAX_PREFIX_CACHE_PAGES")?,
        enable_chunked_prefill,
        chunked_prefill_threshold: env.parse_or(
            "IZWI_CHUNKED_PREFILL_THRESHOLD",
            engine_defaults.chunked_prefill_threshold,
        )?,
        max_tokens_per_step: env.parse_or(
            izwi_core::serve_runtime::ENV_MAX_TOKENS_PER_STEP,
            engine_defaults.max_tokens_per_step,
        )?,
        // DS4 hierarchical offload: explicit opt-in via a positive host pool
        // budget; the kill switch forces it off even when a budget is set.
        kv_host_pool_budget_bytes: if env.flag(izwi_core::serve_runtime::ENV_KV_HOST_OFFLOAD)?
            == Some(false)
        {
            0
        } else {
            env.parse_or(
                izwi_core::serve_runtime::ENV_KV_HOST_POOL_BUDGET_BYTES,
                engine_defaults.kv_host_pool_budget_bytes,
            )?
        },
        kv_offload_high_watermark: env.parse_or(
            izwi_core::serve_runtime::ENV_KV_OFFLOAD_HIGH_WATERMARK,
            engine_defaults.kv_offload_high_watermark,
        )?,
        kv_offload_low_watermark: env.parse_or(
            izwi_core::serve_runtime::ENV_KV_OFFLOAD_LOW_WATERMARK,
            engine_defaults.kv_offload_low_watermark,
        )?,
        kv_offload_max_in_flight_pages: env.parse_or(
            izwi_core::serve_runtime::ENV_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES,
            engine_defaults.kv_offload_max_in_flight_pages,
        )?,
        kv_offload_max_promotion_pages: env.parse_or(
            izwi_core::serve_runtime::ENV_KV_OFFLOAD_MAX_PROMOTION_PAGES,
            engine_defaults.kv_offload_max_promotion_pages,
        )?,
        max_sequence_length: max_sequence_length.unwrap_or(engine_defaults.max_sequence_length),
        asr_realtime: realtime_asr_limits(&env)?,
        ..EngineConfig::default()
    };
    // A worker process owns one explicitly assigned backend for its entire life.
    // The strict constructor rejects unavailable or mismatched assignments.
    let runtime = Arc::new(
        RuntimeService::new_assigned(engine, runtime_assignment(&process.assignment)?)
            .context("initialize exactly assigned worker runtime")?,
    );
    runtime
        .load_model(process.variant)
        .await
        .with_context(|| format!("load selected model {}", process.public_model))?;
    match process.task {
        TaskKind::Chat => {
            warm_up_chat_runtime(
                &runtime,
                process.variant,
                std::time::Duration::from_secs(30),
            )
            .await
            .context("warm selected model before binding")?;
        }
        TaskKind::SpeechToText => {
            warm_up_asr_runtime(
                &runtime,
                process.variant,
                std::time::Duration::from_secs(60),
            )
            .await
            .context("warm selected realtime ASR model before binding")?;
        }
        TaskKind::TextToSpeech => {
            warm_up_tts_runtime(
                &runtime,
                process.variant,
                std::time::Duration::from_secs(120),
            )
            .await
            .context("warm selected realtime TTS model before binding")?;
        }
        TaskKind::Unknown => bail!("parse_worker_task yields only known tasks"),
    }
    // Resident model memory stays charged to this worker's runtime and resource
    // lease. Only the transient node-wide load stage is released here.
    drop(model_load_stage);

    let credentials = ServiceCredentials {
        credential_id: process.credential_id.clone(),
        bearer_token: process.bearer_token.clone(),
    };
    let mut features = BTreeSet::from([WorkerFeature::Cancellation, WorkerFeature::AttemptQuery]);
    if process.streaming {
        features.insert(WorkerFeature::Streaming);
    }
    if matches!(
        process.task,
        TaskKind::SpeechToText | TaskKind::TextToSpeech
    ) {
        features.insert(WorkerFeature::RealtimeSocket);
    }
    if process.task == TaskKind::TextToSpeech {
        // The TTS stage honours the text stream's voice and incremental text.
        features.insert(WorkerFeature::RealtimeTtsTextStream);
    }
    if process.task == TaskKind::Chat {
        // `RuntimeChatExecutor` maps every minor-4 chat control the way the
        // local server does.
        features.extend([
            WorkerFeature::ChatSeed,
            WorkerFeature::ChatStopSequences,
            WorkerFeature::ChatTopK,
            WorkerFeature::ChatPenalties,
            WorkerFeature::ChatLogprobs,
            WorkerFeature::ChatContextPolicy,
            WorkerFeature::ChatSkipSpecialTokens,
        ]);
        if izwi_core::chat_supports_json_object(process.variant) {
            features.insert(WorkerFeature::ChatJsonObject);
        }
        if izwi_core::chat_supports_tools(process.variant) {
            features.insert(WorkerFeature::ChatTools);
        }
    }
    let descriptor = WorkerDescriptor {
        schema_version: PROTOCOL_V1,
        supported_protocol_versions: vec![PROTOCOL_V1],
        worker_id: process.worker_id.clone(),
        node_id: process.node_id.clone(),
        incarnation_id: process.incarnation_id.clone(),
        build_version: env!("CARGO_PKG_VERSION").into(),
        assignment: process.assignment.clone(),
        features,
    };
    let fitted_context_tokens = runtime
        .loaded_model_diagnostics()
        .await
        .into_iter()
        .find(|loaded| loaded.variant_id == process.variant.dir_name())
        .and_then(|loaded| loaded.effective_context_tokens);
    let deployment = loaded_deployment(&process, fitted_context_tokens)?;
    let capacity = WorkerCapacity::resolve(
        process.max_active_invocations,
        requested_seqs,
        runtime.resolved_concurrency().scheduler_rows,
        process.max_retained_attempts,
    )?;
    tracing::info!(
        max_active_invocations = capacity.max_active_invocations,
        max_retained_attempts = capacity.max_retained_attempts,
        profile = runtime.resolved_concurrency().profile.as_str(),
        "worker capacity"
    );
    let worker_config = WorkerConfig {
        descriptor,
        deployment,
        credentials,
        max_active_invocations: capacity.max_active_invocations,
        max_request_bytes: process.max_request_bytes,
        max_retained_attempts: capacity.max_retained_attempts,
        attempt_retention: process.attempt_retention,
        event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
        max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
        slow_consumer_timeout: izwi_core::stream_slow_consumer_timeout(),
        realtime_session_limits: REALTIME_SESSION_DEFAULT_LIMITS,
    };
    let result = match process.task {
        TaskKind::Chat => {
            let worker = WorkerService::new(
                worker_config,
                RuntimeChatExecutor::new(runtime, process.variant, process.streaming),
            )?;
            serve_worker(worker, &process).await
        }
        TaskKind::SpeechToText => {
            let worker = WorkerService::new(
                worker_config,
                RuntimeRealtimeAsrExecutor::new(runtime, process.variant),
            )?;
            serve_worker(worker, &process).await
        }
        TaskKind::TextToSpeech => {
            let worker = WorkerService::new(
                worker_config,
                RuntimeRealtimeTtsExecutor::new(runtime, process.variant).await,
            )?;
            serve_worker(worker, &process).await
        }
        TaskKind::Unknown => bail!("parse_worker_task yields only known tasks"),
    };
    result
}

/// Binds the constructed worker service and serves it with the same
/// TLS/plain rules for every executor type. Once the shutdown sequence has
/// drained and cancelled, open connections get the termination grace to
/// close; a worker still held open after it exits on its own, since tearing
/// the runtime down would wait on whatever work kept them open.
async fn serve_worker<E: izwi_serving_worker::InvocationExecutor>(
    worker: WorkerService<E>,
    process: &WorkerProcessConfig,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(process.bind)
        .await
        .with_context(|| format!("bind private worker at {}", process.bind))?;
    let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
    let grace = ShutdownGrace {
        drain: process.drain_grace,
        cancellation: process.cancellation_grace,
    };
    let shutdown = {
        let worker = worker.clone();
        let managed = process.managed;
        async move {
            shutdown_sequence(
                &worker,
                cooperative_stop_requested(managed),
                escalated_stop_requested(managed),
                grace,
            )
            .await;
            let _ = stopped_tx.send(());
        }
    };
    if process.exit_on_runtime_failure {
        tokio::spawn(exit_on_runtime_failure(worker.clone()));
    }
    tracing::info!(
        address = %process.bind,
        model = %process.public_model,
        task = ?process.task,
        backend = ?process.assignment.backend(),
        tls = process.tls.is_some(),
        "worker ready"
    );
    let served = if let Some(tls) = &process.tls {
        let tls_listener = TlsListener::new(listener, tls.acceptor.clone(), TLS_HANDSHAKE_TIMEOUT)
            .context("start private TLS worker listener")?;
        let server = axum::serve(tls_listener, worker.router()).with_graceful_shutdown(shutdown);
        serve_within_termination_grace(
            async move { server.await.context("serve private TLS worker") },
            stopped_rx,
            process.termination_grace,
        )
        .await
    } else {
        let server = axum::serve(listener, worker.router()).with_graceful_shutdown(shutdown);
        serve_within_termination_grace(
            async move { server.await.context("serve private worker") },
            stopped_rx,
            process.termination_grace,
        )
        .await
    };
    match served {
        Some(result) => result,
        None => {
            tracing::warn!(
                termination_grace_ms = process.termination_grace.as_millis() as u64,
                active_invocations = worker.active_invocations(),
                "connections outlived the termination grace; exiting without teardown"
            );
            std::process::exit(TERMINATION_GRACE_EXIT_CODE);
        }
    }
}

/// Exit status for a worker that stopped because open connections
/// outlived its termination grace.
const TERMINATION_GRACE_EXIT_CODE: i32 = 1;

/// Runs `server` to completion, or, once `stopped` fires, for at most
/// `termination_grace` more. `None` means the grace ran out.
async fn serve_within_termination_grace<F: std::future::Future>(
    server: F,
    stopped: tokio::sync::oneshot::Receiver<()>,
    termination_grace: Duration,
) -> Option<F::Output> {
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => Some(result),
        stopped = stopped => {
            if stopped.is_err() {
                return Some(server.await);
            }
            tokio::time::timeout(termination_grace, &mut server).await.ok()
        }
    }
}

/// Exit status for a worker whose runtime failed permanently (EX_SOFTWARE).
const RUNTIME_FAILURE_EXIT_CODE: i32 = 70;
const RUNTIME_HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Watch runtime health and exit once it fails permanently. Every invocation
/// is already rejected, so in-flight work cannot complete; exiting directly
/// (not through runtime teardown) avoids waiting on a blocking task stuck in
/// a faulted device call. The supervisor treats any exit as a failure and
/// restarts the worker with its bounded backoff.
async fn exit_on_runtime_failure<E: izwi_serving_worker::InvocationExecutor>(
    worker: WorkerService<E>,
) {
    let mut poll = tokio::time::interval(RUNTIME_HEALTH_POLL_INTERVAL);
    loop {
        poll.tick().await;
        if let RuntimeHealth::Failed { reason } = worker.runtime_health() {
            tracing::error!(
                %reason,
                exit_code = RUNTIME_FAILURE_EXIT_CODE,
                "worker runtime failed permanently; exiting so the supervisor restarts it"
            );
            std::process::exit(RUNTIME_FAILURE_EXIT_CODE);
        }
    }
}

struct ManagedLockContext {
    namespace: LockNamespace,
    paths: WorkerLockPaths,
    model_load_slots: u32,
}

/// Parses the supervisor-provided model-load slot count. Absent means the
/// historical single-slot node; values outside the supervisor's validated
/// range are rejected rather than clamped so a misconfigured node fails loudly.
fn model_load_slots_from(raw: Option<&str>) -> anyhow::Result<u32> {
    const MAX_SLOTS: u32 = izwi_serving_fence::MAX_MODEL_LOAD_SLOTS;
    let Some(raw) = raw else {
        return Ok(izwi_serving_fence::DEFAULT_MODEL_LOAD_SLOTS);
    };
    let slots: u32 = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("IZWI_WORKER_MODEL_LOAD_SLOTS must be an unsigned integer"))?;
    if slots == 0 || slots > MAX_SLOTS {
        bail!("IZWI_WORKER_MODEL_LOAD_SLOTS must be between 1 and {MAX_SLOTS}");
    }
    Ok(slots)
}

fn managed_lock_context(
    env: &EnvSnapshot,
    process: &WorkerProcessConfig,
) -> anyhow::Result<Option<ManagedLockContext>> {
    if !process.managed {
        return Ok(None);
    }
    let ownership = PathBuf::from(env.required(WORKER_OWNERSHIP_LOCK_ENV)?);
    let generation = PathBuf::from(env.required(WORKER_GENERATION_FENCE_ENV)?);
    let model_load = PathBuf::from(env.required(WORKER_MODEL_LOAD_LOCK_ENV)?);
    let directory = ownership
        .parent()
        .context("worker ownership lock must have a parent directory")?;
    if generation.parent() != Some(directory) || model_load.parent() != Some(directory) {
        bail!("worker ownership, generation, and model-load locks must share one namespace");
    }
    let namespace = LockNamespace::open(directory).context("open worker lock namespace")?;
    let paths = WorkerLockPaths::for_worker(&namespace, &process.worker_id, &process.assignment);
    if ownership.as_path() != paths.ownership()
        || generation.as_path() != paths.generation_fence()
        || model_load.as_path() != paths.model_load()
    {
        bail!("managed worker lock paths do not match the assigned resource and node namespace");
    }
    Ok(Some(ManagedLockContext {
        namespace,
        paths,
        model_load_slots: model_load_slots_from(env.get(WORKER_MODEL_LOAD_SLOTS_ENV)?)?,
    }))
}

fn acquire_managed_worker_fences(
    process: &WorkerProcessConfig,
    locks: Option<&ManagedLockContext>,
) -> anyhow::Result<Option<WorkerFenceLeases>> {
    let Some(locks) = locks else {
        return Ok(None);
    };
    let metadata = format!(
        "worker={} incarnation={}",
        process.worker_id, process.incarnation_id
    );
    let leases = try_acquire_worker_fences(
        &locks.namespace,
        locks.paths.ownership(),
        locks.paths.generation_fence(),
        metadata.as_bytes(),
    )
    .context("acquire assigned resource and generation fences")?;
    Ok(Some(leases))
}

fn acquire_model_load_stage(
    process: &WorkerProcessConfig,
    locks: Option<&ManagedLockContext>,
) -> anyhow::Result<Option<LockLease>> {
    let Some(locks) = locks else {
        return Ok(None);
    };
    let metadata = format!(
        "worker={} incarnation={} deployment={} generation={}",
        process.worker_id,
        process.incarnation_id,
        process.deployment_id,
        process.model_generation.get()
    );
    locks
        .namespace
        .acquire_model_load_slot(locks.model_load_slots, metadata.as_bytes())
        .map(Some)
        .context("wait for a node model-load stage slot")
}

/// The supervisor's shutdown graces the worker spends before its
/// termination grace (`[shutdown]` in the supervisor's node config).
#[derive(Debug, Clone, Copy)]
struct ShutdownGrace {
    drain: Duration,
    cancellation: Duration,
}

/// What the shutdown sequence asks of a worker service.
trait Stoppable {
    async fn begin_draining(&self);
    fn request_cancel_all(&self);
    fn active_invocations(&self) -> usize;
}

impl<E: izwi_serving_worker::InvocationExecutor> Stoppable for WorkerService<E> {
    async fn begin_draining(&self) {
        WorkerService::begin_draining(self).await;
    }

    fn request_cancel_all(&self) {
        WorkerService::request_cancel_all(self);
    }

    fn active_invocations(&self) -> usize {
        WorkerService::active_invocations(self)
    }
}

/// Ctrl+C, the supervisor closing its control pipe, or, for a worker
/// nothing supervises, SIGTERM: stop admitting, drain, then cancel.
async fn cooperative_stop_requested(managed: bool) {
    if managed {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = parent_control_closed() => {}
        }
    } else {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            () = terminate_signal() => {}
        }
    }
}

/// The supervisor's SIGTERM: its cooperative window is over and it kills the
/// worker once its termination grace runs out. Never resolves for a worker
/// nothing supervises, where SIGTERM is the cooperative request instead.
async fn escalated_stop_requested(managed: bool) {
    if managed {
        terminate_signal().await;
    } else {
        std::future::pending::<()>().await;
    }
}

#[cfg(unix)]
async fn terminate_signal() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut terminate) => {
            terminate.recv().await;
        }
        Err(error) => {
            tracing::warn!(%error, "cannot listen for SIGTERM");
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(not(unix))]
async fn terminate_signal() {
    std::future::pending::<()>().await;
}

/// Stops the worker once `cooperative` or `escalated` resolves: admission
/// closes, in-flight work drains for the drain grace, what is left is
/// cancelled and given the cancellation grace. An escalation, before or
/// during those waits, cancels at once and skips the remaining waits, so
/// the termination grace that follows is all the time left.
async fn shutdown_sequence<W: Stoppable>(
    worker: &W,
    cooperative: impl std::future::Future<Output = ()>,
    escalated: impl std::future::Future<Output = ()>,
    grace: ShutdownGrace,
) {
    tokio::pin!(escalated);
    let mut escalated_now = tokio::select! {
        () = cooperative => false,
        () = &mut escalated => true,
    };
    worker.begin_draining().await;
    if !escalated_now {
        tokio::select! {
            idle = wait_until_idle(worker, grace.drain) => {
                if idle {
                    return;
                }
            }
            () = &mut escalated => escalated_now = true,
        }
    }
    worker.request_cancel_all();
    if !escalated_now {
        tokio::select! {
            _ = wait_until_idle(worker, grace.cancellation) => {}
            () = &mut escalated => {}
        }
    }
}

async fn parent_control_closed() {
    let mut input = tokio::io::stdin();
    let mut heartbeat = [0_u8; 1];
    loop {
        match input.read(&mut heartbeat).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

async fn wait_until_idle<W: Stoppable>(worker: &W, timeout: Duration) -> bool {
    tokio::time::timeout(timeout, async {
        while worker.active_invocations() != 0 {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .is_ok()
}

struct WorkerProcessConfig {
    bind: SocketAddr,
    tls: Option<WorkerTlsConfig>,
    models_dir: PathBuf,
    task: TaskKind,
    variant: ModelVariant,
    public_model: ModelAlias,
    worker_id: WorkerId,
    node_id: NodeId,
    incarnation_id: IncarnationId,
    deployment_id: DeploymentId,
    artifact_revision: ArtifactRevision,
    model_generation: ModelGeneration,
    credential_id: CredentialId,
    bearer_token: ServiceBearerToken,
    assignment: DeviceAssignment,
    /// `IZWI_WORKER_MAX_ACTIVE`; unset derives it from the loaded model.
    max_active_invocations: Option<usize>,
    /// `IZWI_MAX_NUM_SEQS`: sequences the engine serves concurrently.
    max_num_seqs: Option<usize>,
    max_request_bytes: usize,
    /// `IZWI_WORKER_MAX_RETAINED_ATTEMPTS`; unset keeps the default, raised
    /// to cover a derived capacity.
    max_retained_attempts: Option<usize>,
    attempt_retention: Duration,
    streaming: bool,
    managed: bool,
    drain_grace: Duration,
    cancellation_grace: Duration,
    /// How long open connections may outlast the drain and cancellation
    /// graces, or a supervisor's SIGTERM, before the worker exits anyway.
    termination_grace: Duration,
    /// Exit after the runtime fails permanently (a post-entry panic or a
    /// latched fatal device fault) so the supervisor restarts the worker
    /// with its bounded backoff instead of leaving a Failed process up.
    exit_on_runtime_failure: bool,
    declared: DeclaredDeployment,
}

/// How many invocations the worker admits at once, and how many terminal
/// attempts it keeps (never fewer than it admits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerCapacity {
    max_active_invocations: usize,
    max_retained_attempts: usize,
}

impl WorkerCapacity {
    /// An explicit `IZWI_WORKER_MAX_ACTIVE` wins; else `IZWI_MAX_NUM_SEQS`;
    /// else the rows the runtime serves the loaded model at, so a default
    /// worker admits as many users as its engine batches instead of one. A
    /// capacity that is not pinned stays within the attempts an operator
    /// keeps; a pinned one below them is a configuration error.
    fn resolve(
        explicit_active: Option<usize>,
        max_num_seqs: Option<usize>,
        runtime_rows: usize,
        explicit_retained: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut max_active_invocations = explicit_active
            .or(max_num_seqs)
            .unwrap_or(runtime_rows)
            .max(1);
        let max_retained_attempts = match explicit_retained {
            Some(retained) if retained < max_active_invocations => {
                if explicit_active.is_some() {
                    bail!(
                        "IZWI_WORKER_MAX_RETAINED_ATTEMPTS ({retained}) is below \
                         IZWI_WORKER_MAX_ACTIVE ({max_active_invocations})"
                    );
                }
                tracing::warn!(
                    derived = max_active_invocations,
                    retained,
                    "worker admission capped at the retained attempt records"
                );
                max_active_invocations = retained.max(1);
                retained.max(1)
            }
            Some(retained) => retained,
            None => DEFAULT_MAX_RETAINED_ATTEMPTS.max(max_active_invocations),
        };
        Ok(Self {
            max_active_invocations,
            max_retained_attempts,
        })
    }
}

/// The execution profile and token limits a supervisor approved for this
/// worker. Unset values are derived from the loaded model.
struct DeclaredDeployment {
    precision: Option<String>,
    execution_representation: Option<String>,
    max_context_tokens: Option<u32>,
    max_output_tokens: Option<u32>,
}

impl DeclaredDeployment {
    fn from_env(env: &EnvSnapshot) -> anyhow::Result<Self> {
        let label = |name: &str| -> anyhow::Result<Option<String>> {
            Ok(env
                .get(name)?
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string))
        };
        let limit = |name: &str| -> anyhow::Result<Option<u32>> {
            Ok(env
                .parse::<std::num::NonZeroU32>(name)?
                .map(std::num::NonZeroU32::get))
        };
        Ok(Self {
            precision: label(WORKER_PRECISION_ENV)?,
            execution_representation: label(WORKER_EXECUTION_REPRESENTATION_ENV)?,
            max_context_tokens: limit(WORKER_MAX_CONTEXT_TOKENS_ENV)?,
            max_output_tokens: limit(WORKER_MAX_OUTPUT_TOKENS_ENV)?,
        })
    }
}

impl WorkerProcessConfig {
    fn from_env(env: &EnvSnapshot) -> anyhow::Result<Self> {
        let task = parse_worker_task(env)?;
        let model = env
            .get_or(
                "IZWI_WORKER_MODEL",
                match task {
                    TaskKind::Chat => "LFM2.5-1.2B-Instruct-GGUF",
                    TaskKind::SpeechToText => "Nemotron-3.5-ASR-Streaming-0.6B",
                    TaskKind::TextToSpeech => "Kokoro-82M",
                    TaskKind::Unknown => bail!("parse_worker_task yields only known tasks"),
                },
            )?
            .to_string();
        let variant = match task {
            TaskKind::Chat => {
                let variant = izwi_core::parse_chat_model_variant(Some(&model))
                    .with_context(|| format!("parse IZWI_WORKER_MODEL={model}"))?;
                if variant != ModelVariant::Lfm2512BInstructGguf
                    && variant != ModelVariant::Qwen3827BFp8
                    && variant != ModelVariant::Qwen3Moe30bA3bGguf
                    && variant != ModelVariant::Qwen36Moe35BA3BFp8
                {
                    bail!(
                        "serving worker supports only LFM2.5-1.2B-Instruct-GGUF, Qwen3.8-27B-FP8, Qwen3-30B-A3B-GGUF, and Qwen3.6-35B-A3B-FP8 for chat"
                    );
                }
                variant
            }
            TaskKind::SpeechToText => {
                // Only families with proven realtime stream decode can serve
                // realtime ASR sessions; anything else fails closed at boot.
                let variant = izwi_core::parse_model_variant(&model)
                    .with_context(|| format!("parse IZWI_WORKER_MODEL={model}"))?;
                match variant.family() {
                    izwi_core::catalog::ModelFamily::NemotronAsr
                    | izwi_core::catalog::ModelFamily::Voxtral => variant,
                    other => bail!(
                        "realtime speech_to_text workers support only Nemotron ASR and Voxtral families; {model} is {other:?}"
                    ),
                }
            }
            TaskKind::TextToSpeech => {
                // Only families with a streaming synthesis surface can serve
                // realtime TTS sessions; anything else fails closed at boot.
                let variant = izwi_core::parse_model_variant(&model)
                    .with_context(|| format!("parse IZWI_WORKER_MODEL={model}"))?;
                match variant.family() {
                    izwi_core::catalog::ModelFamily::Qwen3Tts
                    | izwi_core::catalog::ModelFamily::KokoroTts
                    | izwi_core::catalog::ModelFamily::VoxtralTts
                    | izwi_core::catalog::ModelFamily::VibeVoiceTts
                    | izwi_core::catalog::ModelFamily::FishS2Tts
                    | izwi_core::catalog::ModelFamily::Lfm25Audio => variant,
                    other => bail!(
                        "realtime text_to_speech workers support only TTS synthesis families; {model} is {other:?}"
                    ),
                }
            }
            TaskKind::Unknown => bail!("parse_worker_task yields only known tasks"),
        };
        let assignment = parse_assignment(env)?;
        let max_active_invocations = env.parse::<usize>("IZWI_WORKER_MAX_ACTIVE")?;
        if max_active_invocations == Some(0) {
            bail!("IZWI_WORKER_MAX_ACTIVE must be non-zero");
        }
        let max_num_seqs = env
            .parse::<usize>("IZWI_MAX_NUM_SEQS")?
            .filter(|seqs| *seqs > 0);
        let bind: SocketAddr = env
            .get_or("IZWI_WORKER_BIND", "127.0.0.1:9470")?
            .parse()
            .context("parse IZWI_WORKER_BIND")?;
        let tls = WorkerTlsConfig::from_env(env)?;
        let allow_insecure_plaintext =
            env.flag_or("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND", false)?;
        if tls.is_none() && !bind.ip().is_loopback() && !allow_insecure_plaintext {
            bail!(
                "plaintext worker transport may bind only to a loopback address; \
                 configure {WORKER_TLS_CERT_REF_ENV} and {WORKER_TLS_KEY_REF_ENV} for \
                 non-loopback TLS, or explicitly acknowledge container networking with \
                 IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND=1"
            );
        }
        Ok(Self {
            bind,
            tls,
            models_dir: env
                .get_os("IZWI_MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_models_dir),
            task,
            variant,
            public_model: ModelAlias::new(model)?,
            worker_id: WorkerId::new(env.get_or("IZWI_WORKER_ID", "local-cpu-1")?)?,
            node_id: NodeId::new(env.get_or("IZWI_WORKER_NODE_ID", "local-node")?)?,
            incarnation_id: IncarnationId::new(
                env.get(WORKER_INCARNATION_ENV)?
                    .map(str::to_string)
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            )?,
            deployment_id: DeploymentId::new(
                env.get_or("IZWI_WORKER_DEPLOYMENT_ID", "lfm25-cpu-v1")?,
            )?,
            artifact_revision: ArtifactRevision::new(
                env.required("IZWI_WORKER_ARTIFACT_REVISION")?,
            )?,
            model_generation: ModelGeneration::new(
                env.parse_or("IZWI_WORKER_MODEL_GENERATION", 1u64)?,
            )?,
            credential_id: CredentialId::new(env.required("IZWI_WORKER_CREDENTIAL_ID")?)?,
            bearer_token: ServiceBearerToken::new(env.required("IZWI_WORKER_BEARER_TOKEN")?)?,
            assignment,
            max_active_invocations,
            max_num_seqs,
            max_request_bytes: env
                .parse_or("IZWI_WORKER_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES)?,
            max_retained_attempts: env.parse::<usize>("IZWI_WORKER_MAX_RETAINED_ATTEMPTS")?,
            attempt_retention: Duration::from_secs(env.parse_or(
                "IZWI_WORKER_ATTEMPT_RETENTION_SECS",
                DEFAULT_ATTEMPT_RETENTION.as_secs(),
            )?),
            streaming: env.flag_or("IZWI_WORKER_STREAMING", true)?,
            managed: managed_worker(env)?,
            drain_grace: Duration::from_millis(parse_bounded_duration_env(
                env,
                "IZWI_WORKER_DRAIN_GRACE_MS",
                30_000,
            )?),
            cancellation_grace: Duration::from_millis(parse_bounded_duration_env(
                env,
                "IZWI_WORKER_CANCELLATION_GRACE_MS",
                10_000,
            )?),
            termination_grace: Duration::from_millis(parse_bounded_duration_env(
                env,
                "IZWI_WORKER_TERMINATION_GRACE_MS",
                5_000,
            )?),
            exit_on_runtime_failure: env.flag_or("IZWI_WORKER_EXIT_ON_RUNTIME_FAILURE", true)?,
            declared: DeclaredDeployment::from_env(env)?,
        })
    }

    /// Plaintext on a non-loopback address, which `from_env` accepts only
    /// with IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND.
    fn insecure_plaintext_bind(&self) -> bool {
        self.tls.is_none() && !self.bind.ip().is_loopback()
    }

    fn thread_budget(&self) -> usize {
        match &self.assignment {
            DeviceAssignment::Cpu { thread_budget, .. } => usize::from(*thread_budget),
            DeviceAssignment::Metal { .. }
            | DeviceAssignment::Cuda { .. }
            | DeviceAssignment::Unknown => 1,
        }
    }
}

struct WorkerTlsConfig {
    acceptor: tokio_rustls::TlsAcceptor,
}

impl std::fmt::Debug for WorkerTlsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerTlsConfig")
            .field("acceptor", &"[REDACTED]")
            .finish()
    }
}

impl WorkerTlsConfig {
    fn from_env(env: &EnvSnapshot) -> anyhow::Result<Option<Self>> {
        let cert_ref = env.get(WORKER_TLS_CERT_REF_ENV)?;
        let key_ref = env.get(WORKER_TLS_KEY_REF_ENV)?;
        let client_ca_ref = env.get(WORKER_TLS_CLIENT_CA_REF_ENV)?;

        match (cert_ref, key_ref) {
            (Some(cert_ref), Some(key_ref)) => {
                let cert_bytes = load_pem_file(cert_ref, "server certificate")?.bytes;
                let key_file = load_pem_file(key_ref, "server private key")?;
                if let Some(mode) = key_file.shared_mode {
                    tracing::warn!(
                        mode = format_args!("{mode:o}"),
                        "the worker TLS private key file is readable by other users"
                    );
                }
                let key_bytes = key_file.bytes;
                let certs = CertificateDer::pem_slice_iter(&cert_bytes)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| anyhow::anyhow!("server certificate PEM is invalid"))?;
                if certs.is_empty() {
                    bail!("server certificate file contains no certificate PEM blocks");
                }
                // The first PKCS#8, PKCS#1 or SEC1 key block.
                let key = PrivateKeyDer::from_pem_slice(&key_bytes).map_err(|error| match error {
                    pem::Error::NoItemsFound => {
                        anyhow::anyhow!("server private key file contains no key PEM blocks")
                    }
                    _ => anyhow::anyhow!("server private key PEM is invalid"),
                })?;

                let mut config = if let Some(ca_ref) = client_ca_ref {
                    let ca_bytes = load_pem_file(ca_ref, "client CA")?.bytes;
                    let client_certs = CertificateDer::pem_slice_iter(&ca_bytes)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| anyhow::anyhow!("client CA PEM is invalid"))?;
                    if client_certs.is_empty() {
                        bail!("client CA file contains no certificate PEM blocks");
                    }
                    let mut roots = tokio_rustls::rustls::RootCertStore::empty();
                    for cert in client_certs {
                        roots
                            .add(cert)
                            .map_err(|e| anyhow::anyhow!("invalid client CA certificate: {e}"))?;
                    }
                    let verifier = tokio_rustls::rustls::server::WebPkiClientVerifier::builder(
                        Arc::new(roots),
                    )
                    .build()
                    .map_err(|e| anyhow::anyhow!("invalid client CA verifier: {e}"))?;
                    tokio_rustls::rustls::ServerConfig::builder()
                        .with_client_cert_verifier(verifier)
                        .with_single_cert(certs, key)
                        .map_err(|e| anyhow::anyhow!("invalid mTLS configuration: {e}"))?
                } else {
                    tokio_rustls::rustls::ServerConfig::builder()
                        .with_no_client_auth()
                        .with_single_cert(certs, key)
                        .map_err(|e| anyhow::anyhow!("invalid server TLS configuration: {e}"))?
                };

                // HTTP/2 lets a gateway multiplex its invocations and status
                // polls over one connection per worker; HTTP/1.1 stays
                // available for clients that do not offer h2 (WebSocket
                // upgrades included).
                config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
                Ok(Some(Self {
                    acceptor: tokio_rustls::TlsAcceptor::from(Arc::new(config)),
                }))
            }
            (Some(_), None) => bail!(
                "{WORKER_TLS_CERT_REF_ENV} is set but {WORKER_TLS_KEY_REF_ENV} is missing; \
                 both are required for TLS"
            ),
            (None, Some(_)) => bail!(
                "{WORKER_TLS_KEY_REF_ENV} is set but {WORKER_TLS_CERT_REF_ENV} is missing; \
                 both are required for TLS"
            ),
            (None, None) => Ok(None),
        }
    }
}

fn load_pem_file(reference: &str, label: &str) -> anyhow::Result<SecretFile> {
    let path = file_reference_path(reference).map_err(|error| {
        anyhow::anyhow!("{label} reference must be an absolute file: path ({error})")
    })?;
    read_secret_file(path, MAX_TLS_PEM_FILE_BYTES)
        .map_err(|error| anyhow::anyhow!("read {label} at {}: {error}", path.display()))
}

/// How long a connecting client may take to complete its TLS handshake.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// TLS listener for `axum::serve`, which accepts connections serially.
/// Handshakes run in their own tasks, each bounded by a timeout, and only
/// completed ones reach the serve loop, so a client that connects and never
/// finishes its handshake cannot block other connections (status polls
/// included).
struct TlsListener {
    local_addr: SocketAddr,
    handshaken: tokio::sync::mpsc::Receiver<(
        tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
        SocketAddr,
    )>,
    acceptor_task: tokio::task::JoinHandle<()>,
}

impl TlsListener {
    fn new(
        inner: tokio::net::TcpListener,
        acceptor: tokio_rustls::TlsAcceptor,
        handshake_timeout: Duration,
    ) -> std::io::Result<Self> {
        let local_addr = inner.local_addr()?;
        let (handshaken_tx, handshaken) = tokio::sync::mpsc::channel(64);
        let acceptor_task = tokio::spawn(async move {
            loop {
                let (stream, addr) = match inner.accept().await {
                    Ok(accepted) => accepted,
                    Err(error) => {
                        tracing::warn!(%error, "worker TCP accept failed");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let acceptor = acceptor.clone();
                let handshaken_tx = handshaken_tx.clone();
                tokio::spawn(async move {
                    match tokio::time::timeout(handshake_timeout, acceptor.accept(stream)).await {
                        Ok(Ok(tls_stream)) => {
                            let _ = handshaken_tx.send((tls_stream, addr)).await;
                        }
                        Ok(Err(error)) => {
                            tracing::warn!(%error, "worker TLS handshake failed");
                        }
                        Err(_) => tracing::warn!(%addr, "worker TLS handshake timed out"),
                    }
                });
            }
        });
        Ok(Self {
            local_addr,
            handshaken,
            acceptor_task,
        })
    }
}

impl Drop for TlsListener {
    fn drop(&mut self) {
        self.acceptor_task.abort();
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.handshaken.recv().await {
            Some(connection) => connection,
            // The acceptor task owns a sender until it is aborted on drop.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        Ok(self.local_addr)
    }
}

fn parse_assignment(env: &EnvSnapshot) -> anyhow::Result<DeviceAssignment> {
    match env
        .get_or("IZWI_BACKEND", "cpu")?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => {
            let thread_budget = env.parse_or("IZWI_WORKER_CPU_THREADS", 1u16)?;
            if thread_budget == 0 {
                bail!("IZWI_WORKER_CPU_THREADS must be non-zero");
            }
            let host_memory_limit_bytes = parse_positive_env(
                env,
                "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget(env, "IZWI_CPU_MEMORY_BUDGET_BYTES", host_memory_limit_bytes)?;
            Ok(DeviceAssignment::Cpu {
                thread_budget,
                affinity: parse_cpu_affinity(env)?,
                host_memory_limit_bytes,
            })
        }
        "metal" => {
            let shared_memory_limit_bytes = parse_positive_env(
                env,
                "IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget(
                env,
                "IZWI_METAL_MEMORY_BUDGET_BYTES",
                shared_memory_limit_bytes,
            )?;
            Ok(DeviceAssignment::Metal {
                device_id: DeviceId::new(env.required("IZWI_WORKER_EXPECTED_DEVICE_ID")?)?,
                process_local_device_index: env.parse_or("IZWI_METAL_DEVICE_ORDINAL", 0u32)?,
                shared_memory_limit_bytes,
            })
        }
        "cuda" => {
            let device_memory_limit_bytes = parse_positive_env(
                env,
                "IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget(
                env,
                "IZWI_CUDA_MEMORY_BUDGET_BYTES",
                device_memory_limit_bytes,
            )?;
            let host_memory_limit_bytes = parse_positive_env(
                env,
                "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget(
                env,
                "IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES",
                host_memory_limit_bytes,
            )?;
            Ok(DeviceAssignment::Cuda {
                device_uuid: DeviceId::new(env.required("IZWI_WORKER_EXPECTED_DEVICE_UUID")?)?,
                process_local_device_index: env.parse_or("IZWI_CUDA_DEVICE_ORDINAL", 0u32)?,
                device_memory_limit_bytes,
                host_memory_limit_bytes,
            })
        }
        backend => bail!("unsupported IZWI_BACKEND={backend}; expected cpu, metal, or cuda"),
    }
}

fn runtime_assignment(assignment: &DeviceAssignment) -> anyhow::Result<RuntimeDeviceAssignment> {
    Ok(match assignment {
        DeviceAssignment::Cpu { .. } => RuntimeDeviceAssignment::Cpu,
        DeviceAssignment::Metal {
            device_id,
            process_local_device_index,
            ..
        } => RuntimeDeviceAssignment::Metal {
            process_local_device_index: usize::try_from(*process_local_device_index)
                .context("Metal device index does not fit this platform")?,
            expected_device_id: device_id.as_str().into(),
        },
        DeviceAssignment::Cuda {
            device_uuid,
            process_local_device_index,
            ..
        } => RuntimeDeviceAssignment::Cuda {
            process_local_device_index: usize::try_from(*process_local_device_index)
                .context("CUDA device index does not fit this platform")?,
            expected_device_uuid: device_uuid.as_str().into(),
        },
        DeviceAssignment::Unknown => bail!("parse_assignment yields only known backends"),
    })
}

fn parse_cpu_affinity(env: &EnvSnapshot) -> anyhow::Result<Vec<u16>> {
    let Some(raw) = env.get("IZWI_WORKER_CPU_AFFINITY")? else {
        return Ok(Vec::new());
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    let affinity = raw
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<u16>()
                .map_err(|error| anyhow::anyhow!("invalid IZWI_WORKER_CPU_AFFINITY: {error}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if affinity.len() > 1024 {
        bail!("IZWI_WORKER_CPU_AFFINITY contains more than 1024 entries");
    }
    let unique = affinity.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != affinity.len() {
        bail!("IZWI_WORKER_CPU_AFFINITY contains duplicate CPU IDs");
    }
    Ok(affinity)
}

/// The realtime ASR session limits set in the environment; an unset limit
/// takes the engine's per-backend default, and the engine rejects zero.
fn realtime_asr_limits(env: &EnvSnapshot) -> anyhow::Result<RealtimeAsrSessionConfig> {
    use izwi_core::serve_runtime::{
        ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS, ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
        ENV_ASR_REALTIME_MAX_SESSIONS,
    };
    Ok(RealtimeAsrSessionConfig {
        max_sessions: env.parse(ENV_ASR_REALTIME_MAX_SESSIONS)?,
        max_lifetime_secs: env.parse(ENV_ASR_REALTIME_MAX_LIFETIME_SECS)?,
        idle_timeout_secs: env.parse(ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS)?,
    })
}

fn parse_positive_env<T>(env: &EnvSnapshot, name: &str, fallback: T) -> anyhow::Result<T>
where
    T: std::str::FromStr + Default + PartialEq,
    T::Err: std::fmt::Display,
{
    let value = env.parse_or(name, fallback)?;
    if value == T::default() {
        bail!("{name} must be non-zero");
    }
    Ok(value)
}

fn parse_bounded_duration_env(
    env: &EnvSnapshot,
    name: &str,
    fallback_ms: u64,
) -> anyhow::Result<u64> {
    let value = parse_positive_env(env, name, fallback_ms)?;
    if value > 24 * 60 * 60 * 1000 {
        bail!("{name} must not exceed one day");
    }
    Ok(value)
}

fn verify_matching_budget(env: &EnvSnapshot, name: &str, advertised: u64) -> anyhow::Result<()> {
    match env.parse::<u64>(name)? {
        Some(enforced) => {
            if enforced != advertised {
                bail!("{name} must match the worker assignment budget");
            }
        }
        None if managed_worker(env)? => {
            bail!("managed worker requires {name} to enforce its assignment budget");
        }
        None => {
            tracing::warn!(budget = advertised, variable = name, "local worker memory assignment is advisory because no runtime budget was configured");
        }
    }
    Ok(())
}

fn managed_worker(env: &EnvSnapshot) -> anyhow::Result<bool> {
    Ok(env.flag_or(MANAGED_WORKER_ENV, false)?)
}

fn verify_artifact_manifest(process: &WorkerProcessConfig) -> anyhow::Result<()> {
    let model_dir = process.models_dir.join(process.variant.dir_name());
    let manifest = izwi_core::artifacts::read_artifact_manifest(&model_dir)
        .with_context(|| format!("read artifact manifest in {}", model_dir.display()))?
        .with_context(|| format!("artifact manifest is required in {}", model_dir.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported artifact manifest schema {}",
            manifest.schema_version
        );
    }
    if manifest.variant != process.variant
        || manifest.repo_id != process.variant.repo_id()
        || manifest.revision != process.artifact_revision.as_str()
    {
        bail!("artifact manifest identity does not match the assigned deployment");
    }
    if manifest.files.is_empty() {
        bail!("artifact manifest must list the selected model files");
    }
    for relative in &manifest.files {
        let path = Path::new(relative);
        if path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            || !model_dir.join(path).is_file()
        {
            bail!("artifact manifest references an unavailable file: {relative}");
        }
    }
    Ok(())
}

/// How this worker serves `variant`: (weight precision, execution
/// representation).
fn execution_profile(
    task: TaskKind,
    variant: ModelVariant,
) -> anyhow::Result<(&'static str, &'static str)> {
    Ok(match task {
        TaskKind::Chat => match variant {
            ModelVariant::Lfm2512BInstructGguf => ("gguf-q4_k_m", "native-lfm2"),
            ModelVariant::Qwen3Moe30bA3bGguf => ("gguf-q4_k_m", "native-qwen3-moe"),
            ModelVariant::Qwen3827BFp8 => ("fp8-block", "native-qwen38"),
            ModelVariant::Qwen36Moe35BA3BFp8 => ("fp8-block", "native-qwen35-moe"),
            other => bail!("no serving execution profile for chat model {other}"),
        },
        TaskKind::SpeechToText => (
            "fp16",
            match variant.family() {
                izwi_core::catalog::ModelFamily::NemotronAsr => "native-nemotron-realtime",
                _ => "native-voxtral-realtime",
            },
        ),
        TaskKind::TextToSpeech => ("fp32", "native-tts-realtime"),
        TaskKind::Unknown => bail!("parse_worker_task yields only known tasks"),
    })
}

/// The deployment this worker advertises: identity from the process
/// configuration, execution profile and token limits from the loaded model.
/// A supervisor-declared profile or limit is advertised exactly once the
/// loaded model proves it; one the model cannot honour stops startup, so the
/// worker never reports a capability it does not have and its status equals
/// the approved deployment.
fn loaded_deployment(
    process: &WorkerProcessConfig,
    fitted_context_tokens: Option<usize>,
) -> anyhow::Result<LoadedDeployment> {
    let (precision, execution_representation) = execution_profile(process.task, process.variant)?;
    let declared = &process.declared;
    for (name, declared, served) in [
        (WORKER_PRECISION_ENV, &declared.precision, precision),
        (
            WORKER_EXECUTION_REPRESENTATION_ENV,
            &declared.execution_representation,
            execution_representation,
        ),
    ] {
        if let Some(declared) = declared.as_deref().filter(|declared| *declared != served) {
            bail!(
                "{name}={declared} does not match {}, which this worker serves as {served}",
                process.public_model
            );
        }
    }
    let (input_format, output_format, realtime) = match process.task {
        TaskKind::Chat => (InputFormat::ChatMessages, OutputFormat::Text, false),
        TaskKind::SpeechToText => (InputFormat::PcmAudio, OutputFormat::Text, true),
        TaskKind::TextToSpeech => (InputFormat::Text, OutputFormat::PcmAudio, true),
        TaskKind::Unknown => bail!("parse_worker_task yields only known tasks"),
    };
    let (max_context_tokens, max_output_tokens) = if realtime {
        // Realtime sessions carry audio or bounded text, not chat context;
        // the session-level byte budget in the subprotocol bounds input.
        if declared.max_context_tokens.is_some() || declared.max_output_tokens.is_some() {
            bail!("realtime speech deployments do not advertise token limits");
        }
        (None, None)
    } else {
        let fitted = fitted_context_tokens
            .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX))
            .context("the loaded chat model reports no fitted context")?;
        let context = declared.max_context_tokens.unwrap_or(fitted);
        if context > fitted {
            bail!(
                "{WORKER_MAX_CONTEXT_TOKENS_ENV}={context} exceeds the loaded model's fitted \
                 context of {fitted} tokens"
            );
        }
        let output = declared.max_output_tokens.unwrap_or(context);
        if output > context {
            bail!(
                "{WORKER_MAX_OUTPUT_TOKENS_ENV}={output} exceeds the advertised context of \
                 {context} tokens"
            );
        }
        (Some(context), Some(output))
    };
    Ok(LoadedDeployment {
        deployment_id: process.deployment_id.clone(),
        public_model: process.public_model.clone(),
        artifact_revision: process.artifact_revision.clone(),
        model_generation: process.model_generation,
        task: process.task,
        backend: process.assignment.backend(),
        precision: precision.into(),
        execution_representation: execution_representation.into(),
        tokenizer_revision: None,
        readiness: ModelReadiness::Ready,
        capability: Capability {
            task: process.task,
            streaming: realtime || process.streaming,
            realtime,
            cancellation: CancellationBehavior::Cooperative,
            accepted_input_formats: BTreeSet::from([input_format]),
            output_formats: BTreeSet::from([output_format]),
            max_input_bytes: process.max_request_bytes as u64,
            max_context_tokens,
            max_output_tokens,
        },
        // Routing signals are per-status observations populated from engine
        // telemetry at status time, never static startup metadata.
        kv_cache_usage_pct: None,
        prefix_hits_total: None,
        prefix_queries_total: None,
        prefix_evictions_total: None,
        kv_host_pages: None,
        kv_demotions_total: None,
        kv_promotions_total: None,
        kv_promotion_latency_avg_seconds: None,
        tokens_out_per_s_ema: None,
        observation_cost_units: None,
    })
}

/// Parses `IZWI_WORKER_TASK`. Chat remains the default so existing
/// deployments are byte-identical.
fn parse_worker_task(env: &EnvSnapshot) -> anyhow::Result<TaskKind> {
    match env
        .get_or(WORKER_TASK_ENV, "chat")?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "chat" => Ok(TaskKind::Chat),
        "speech_to_text" => Ok(TaskKind::SpeechToText),
        "text_to_speech" => Ok(TaskKind::TextToSpeech),
        other => Err(anyhow::anyhow!(
            "invalid IZWI_WORKER_TASK={other}; expected chat, speech_to_text, or text_to_speech"
        )),
    }
}

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker whose one invocation never finishes on its own.
    #[derive(Default)]
    struct StuckWorker {
        draining: std::sync::atomic::AtomicBool,
        cancelled: std::sync::atomic::AtomicBool,
    }

    impl Stoppable for StuckWorker {
        async fn begin_draining(&self) {
            self.draining
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        fn request_cancel_all(&self) {
            self.cancelled
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        fn active_invocations(&self) -> usize {
            1
        }
    }

    const GRACE: ShutdownGrace = ShutdownGrace {
        drain: Duration::from_secs(30),
        cancellation: Duration::from_secs(10),
    };

    /// The supervisor's SIGTERM in the middle of the drain cancels at once
    /// instead of waiting out the drain and cancellation graces it already
    /// spent: what is left is its termination grace.
    #[tokio::test(start_paused = true)]
    async fn escalation_during_the_drain_cancels_at_once() {
        let worker = StuckWorker::default();
        let started = tokio::time::Instant::now();
        shutdown_sequence(
            &worker,
            std::future::ready(()),
            tokio::time::sleep(Duration::from_secs(1)),
            GRACE,
        )
        .await;
        assert!(worker.draining.load(std::sync::atomic::Ordering::SeqCst));
        assert!(worker.cancelled.load(std::sync::atomic::Ordering::SeqCst));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    /// Without an escalation the sequence spends both graces on a worker
    /// that never goes idle.
    #[tokio::test(start_paused = true)]
    async fn cooperative_shutdown_spends_both_graces() {
        let worker = StuckWorker::default();
        let started = tokio::time::Instant::now();
        shutdown_sequence(
            &worker,
            std::future::ready(()),
            std::future::pending(),
            GRACE,
        )
        .await;
        assert!(worker.cancelled.load(std::sync::atomic::Ordering::SeqCst));
        assert!(started.elapsed() >= GRACE.drain + GRACE.cancellation);
    }

    /// A connection that never closes holds the server open only for the
    /// termination grace after the shutdown sequence finished.
    #[tokio::test(start_paused = true)]
    async fn open_connections_get_only_the_termination_grace() {
        let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
        stopped_tx.send(()).unwrap();
        let started = tokio::time::Instant::now();
        let served = tokio::time::timeout(
            Duration::from_secs(60),
            serve_within_termination_grace(
                std::future::pending::<()>(),
                stopped_rx,
                Duration::from_secs(5),
            ),
        )
        .await
        .expect("the termination grace bounds the server");
        assert!(served.is_none());
        assert_eq!(started.elapsed(), Duration::from_secs(5));
    }

    #[test]
    fn termination_grace_is_read_and_bounded() {
        let env = |value: &str| {
            EnvSnapshot::from_pairs(
                env_config::WORKER_ENV,
                [("IZWI_WORKER_TERMINATION_GRACE_MS", value)],
            )
        };
        assert_eq!(
            parse_bounded_duration_env(&env("2500"), "IZWI_WORKER_TERMINATION_GRACE_MS", 5_000)
                .unwrap(),
            2_500
        );
        assert!(
            parse_bounded_duration_env(&env("0"), "IZWI_WORKER_TERMINATION_GRACE_MS", 5_000)
                .is_err()
        );
    }

    #[test]
    fn worker_capacity_defaults_to_the_engine_rows_not_one_request() {
        // A default worker admits as many invocations as its engine batches.
        let auto = WorkerCapacity::resolve(None, None, 8, None).unwrap();
        assert_eq!(auto.max_active_invocations, 8);
        assert_eq!(auto.max_retained_attempts, DEFAULT_MAX_RETAINED_ATTEMPTS);
        // IZWI_MAX_NUM_SEQS sizes it; an explicit IZWI_WORKER_MAX_ACTIVE wins.
        let seqs = WorkerCapacity::resolve(None, Some(3), 8, None).unwrap();
        assert_eq!(seqs.max_active_invocations, 3);
        let explicit = WorkerCapacity::resolve(Some(1), Some(3), 8, None).unwrap();
        assert_eq!(explicit.max_active_invocations, 1);
        // A large derived capacity raises the default retained attempts...
        let wide = WorkerCapacity::resolve(None, None, 2_000, None).unwrap();
        assert_eq!(wide.max_retained_attempts, 2_000);
        // ...and an explicit value below a derived capacity caps it, so a
        // worker on a large GPU still starts under a supervisor's 128.
        let capped = WorkerCapacity::resolve(None, None, 320, Some(128)).unwrap();
        assert_eq!(capped.max_active_invocations, 128);
        assert_eq!(capped.max_retained_attempts, 128);
        assert_eq!(
            WorkerCapacity::resolve(None, Some(16), 8, Some(4))
                .unwrap()
                .max_active_invocations,
            4
        );
        // A pinned capacity above the retained records is an error.
        assert!(WorkerCapacity::resolve(Some(16), None, 8, Some(4)).is_err());
        assert_eq!(
            WorkerCapacity::resolve(None, Some(16), 8, Some(32))
                .unwrap()
                .max_retained_attempts,
            32
        );
    }

    #[test]
    fn model_load_slots_parse_is_bounded_and_defaults_to_one() {
        assert_eq!(model_load_slots_from(None).unwrap(), 1);
        assert_eq!(model_load_slots_from(Some("1")).unwrap(), 1);
        assert_eq!(model_load_slots_from(Some(" 4 ")).unwrap(), 4);
        assert_eq!(
            model_load_slots_from(Some("64")).unwrap(),
            izwi_serving_fence::MAX_MODEL_LOAD_SLOTS
        );
        assert!(model_load_slots_from(Some("0")).is_err());
        assert!(model_load_slots_from(Some("65")).is_err());
        assert!(model_load_slots_from(Some("two")).is_err());
        assert!(model_load_slots_from(Some("-1")).is_err());
    }

    #[test]
    fn protocol_assignments_map_to_strict_runtime_assignments() {
        assert_eq!(
            runtime_assignment(&DeviceAssignment::Cpu {
                thread_budget: 2,
                affinity: vec![0, 1],
                host_memory_limit_bytes: 1024,
            })
            .unwrap(),
            RuntimeDeviceAssignment::Cpu
        );
        assert_eq!(
            runtime_assignment(&DeviceAssignment::Metal {
                device_id: DeviceId::new("metal:1234").unwrap(),
                process_local_device_index: 1,
                shared_memory_limit_bytes: 2048,
            })
            .unwrap(),
            RuntimeDeviceAssignment::Metal {
                process_local_device_index: 1,
                expected_device_id: "metal:1234".into(),
            }
        );
        assert_eq!(
            runtime_assignment(&DeviceAssignment::Cuda {
                device_uuid: DeviceId::new("GPU-0123").unwrap(),
                process_local_device_index: 0,
                device_memory_limit_bytes: 2048,
                host_memory_limit_bytes: 1024,
            })
            .unwrap(),
            RuntimeDeviceAssignment::Cuda {
                process_local_device_index: 0,
                expected_device_uuid: "GPU-0123".into(),
            }
        );
    }

    /// Parses the worker configuration a supervisor would launch for one
    /// deployment, and returns it with the supervisor's approval.
    fn supervised_worker(
        deployment: &str,
    ) -> (
        WorkerProcessConfig,
        izwi_serving_supervisor::ExpectedWorkerIdentity,
    ) {
        use izwi_serving_supervisor::{
            build_child_launch_spec, BinaryCatalog, BinaryRecord, ExpectedWorkerIdentity,
            HostInventory, NodeConfig, ResolvedWorkerSecret, WorkerBinaryFlavor,
        };
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let binary = root.path().join("worker");
        std::fs::write(&binary, b"worker").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let config = format!(
            r#"
schema_version = 2
node_id = "node-a"
working_directory = "{root}"
runtime_directory = "{root}/run"
host_memory_budget_bytes = 4096

[[workers]]
worker_id = "worker-a"
bind = "127.0.0.1:9470"
binary = "cpu"
credential_id = "credential-a"
bearer_token_env = "WORKER_A_TOKEN"
max_active_invocations = 1
max_request_bytes = 4096
streaming = true

[workers.assignment]
backend = "cpu"
thread_budget = 1
affinity = [0]
host_memory_limit_bytes = 4096

[workers.deployment]
deployment_id = "deployment-a"
artifact_revision = "revision-a"
model_generation = 3
backend = "cpu"
models_directory = "{root}"
{deployment}
"#,
            root = root.path().display()
        );
        let node = NodeConfig::parse_bounded(config.as_bytes())
            .unwrap()
            .validate(
                &HostInventory {
                    effective_cpu_ids: vec![0],
                    allocatable_host_memory_bytes: 8192,
                    metal_devices: Vec::new(),
                    cuda_devices: Vec::new(),
                },
                &BinaryCatalog::new([(
                    WorkerBinaryFlavor::Cpu,
                    BinaryRecord {
                        path: binary,
                        supported_backends: vec![BackendKind::Cpu],
                    },
                )]),
            )
            .unwrap();
        let worker_id = WorkerId::new("worker-a").unwrap();
        let incarnation = IncarnationId::new("incarnation-a").unwrap();
        let namespace = LockNamespace::open(root.path().join("locks")).unwrap();
        let assignment = node.worker(&worker_id).unwrap().assignment.clone();
        let spec = build_child_launch_spec(
            &node,
            &worker_id,
            &incarnation,
            "WORKER_A_TOKEN",
            &ResolvedWorkerSecret {
                bearer_token: ServiceBearerToken::new("worker-a-secret").unwrap(),
            },
            &Default::default(),
            &WorkerLockPaths::for_worker(&namespace, &worker_id, &assignment),
        )
        .unwrap();

        // The worker captures only the variables it registers.
        let env = env_config::snapshot(spec.environment().iter().filter_map(|(name, value)| {
            let name = name.to_str()?;
            env_config::WORKER_ENV
                .iter()
                .any(|var| var.name == name)
                .then(|| (name, value.to_str().unwrap()))
        }));
        let process = WorkerProcessConfig::from_env(&env).unwrap();
        let expected = ExpectedWorkerIdentity::from_config(&node, &worker_id, incarnation).unwrap();
        (process, expected)
    }

    const SUPERVISED_CHAT: &str = r#"
public_model = "LFM2.5-1.2B-Instruct-GGUF"
task = "chat"
precision = "gguf-q4_k_m"
execution_representation = "native-lfm2"

[workers.deployment.capability]
streaming = true
realtime = false
cancellation = "cooperative"
accepted_input_formats = ["chat_messages"]
output_formats = ["text"]
max_input_bytes = 4096
max_context_tokens = 8192
max_output_tokens = 4096
"#;

    #[test]
    fn worker_deployment_equals_the_supervisor_approval() {
        let (process, expected) = supervised_worker(SUPERVISED_CHAT);
        let deployment = loaded_deployment(&process, Some(32_768)).unwrap();
        assert_eq!(expected.verify_deployment(&deployment), Ok(()));
        assert_eq!(deployment.capability.max_context_tokens, Some(8192));
        assert_eq!(deployment.capability.max_output_tokens, Some(4096));

        let (process, expected) = supervised_worker(
            r#"
public_model = "Nemotron-3.5-ASR-Streaming-0.6B"
task = "speech_to_text"
precision = "fp16"
execution_representation = "native-nemotron-realtime"

[workers.deployment.capability]
streaming = true
realtime = true
cancellation = "cooperative"
accepted_input_formats = ["pcm_audio"]
output_formats = ["text"]
max_input_bytes = 4096
"#,
        );
        assert_eq!(process.task, TaskKind::SpeechToText);
        let deployment = loaded_deployment(&process, None).unwrap();
        assert_eq!(expected.verify_deployment(&deployment), Ok(()));
    }

    #[test]
    fn worker_refuses_a_declaration_its_loaded_model_cannot_honour() {
        let (process, _) = supervised_worker(SUPERVISED_CHAT);
        let error = loaded_deployment(&process, Some(4096)).unwrap_err();
        assert!(error.to_string().contains("fitted context"), "{error}");
        assert!(loaded_deployment(&process, None).is_err());

        let (process, _) =
            supervised_worker(&SUPERVISED_CHAT.replace("native-lfm2", "native-qwen38"));
        let error = loaded_deployment(&process, Some(32_768)).unwrap_err();
        assert!(error.to_string().contains("native-lfm2"), "{error}");
    }

    #[test]
    fn standalone_chat_worker_advertises_its_fitted_context() {
        let (mut process, _) = supervised_worker(SUPERVISED_CHAT);
        process.declared = DeclaredDeployment {
            precision: None,
            execution_representation: None,
            max_context_tokens: None,
            max_output_tokens: None,
        };
        let deployment = loaded_deployment(&process, Some(32_768)).unwrap();
        assert_eq!(deployment.precision, "gguf-q4_k_m");
        assert_eq!(deployment.execution_representation, "native-lfm2");
        assert_eq!(deployment.capability.max_context_tokens, Some(32_768));
        assert_eq!(deployment.capability.max_output_tokens, Some(32_768));

        process.declared.max_output_tokens = Some(1024);
        let deployment = loaded_deployment(&process, Some(32_768)).unwrap();
        assert_eq!(deployment.capability.max_output_tokens, Some(1024));
    }

    // A test CA and a 127.0.0.1/localhost server certificate it signed,
    // both valid until 2126.
    const TEST_TLS_CA_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBojCCAUmgAwIBAgIUdBX0RhdufHDBNF8a5FM9v08YGpkwCgYIKoZIzj0EAwIw
HjEcMBoGA1UEAwwTaXp3aS13b3JrZXItdGVzdC1jYTAgFw0yNjEwMTUwMDQ2NDVa
GA8yMTI2MDkyMTAwNDY0NVowHjEcMBoGA1UEAwwTaXp3aS13b3JrZXItdGVzdC1j
YTBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABH6nOtgS54Cm3yRQLlEuAyOjJ3EF
FbehOd+roB/cz5zV22Vca4829Tjt78IKqMjYMgAM83vfGO0BjYH1P98Y4GujYzBh
MB0GA1UdDgQWBBSHCmIoNmcszJw5SuI+kD2tGf63NDAfBgNVHSMEGDAWgBSHCmIo
NmcszJw5SuI+kD2tGf63NDAPBgNVHRMBAf8EBTADAQH/MA4GA1UdDwEB/wQEAwIC
BDAKBggqhkjOPQQDAgNHADBEAiBsSEjXq630d3WuZ1A2SfNeTgVAGjY/UiI1ihwt
ncCWmAIgF/OmCK7S2tW5gSC0RruJ+HTbQceCUW2RJWDR3UCVzMg=
-----END CERTIFICATE-----\n";
    const TEST_TLS_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIByjCCAW+gAwIBAgIUHdiP51T8U6u0cmd1WdbO0gq5ZCowCgYIKoZIzj0EAwIw
HjEcMBoGA1UEAwwTaXp3aS13b3JrZXItdGVzdC1jYTAgFw0yNjEwMTUwMDQ2NDVa
GA8yMTI2MDkyMTAwNDY0NVowFDESMBAGA1UEAwwJbG9jYWxob3N0MFkwEwYHKoZI
zj0CAQYIKoZIzj0DAQcDQgAEawKV8g783ImEAkgeHXTmlt5bAUsCTdhZb8PBRWvA
U24lyrzBrR/Vz5uIg4TeK0q7ipxtc/yjMkpeeZdx1+5/CaOBkjCBjzAaBgNVHREE
EzARhwR/AAABgglsb2NhbGhvc3QwDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8EBAMC
B4AwEwYDVR0lBAwwCgYIKwYBBQUHAwEwHQYDVR0OBBYEFF/nl2HJSUkKIf++pjqo
OxYwfaHTMB8GA1UdIwQYMBaAFIcKYig2ZyzMnDlK4j6QPa0Z/rc0MAoGCCqGSM49
BAMCA0kAMEYCIQCDcIS9dgHuzrVJZ7h8voeSmzHbnAPDP5NkkMw3bEhReQIhALcZ
+gix6hpATwk3SOgWPxw/u6+6bzgIQevkjzg2Z/zd
-----END CERTIFICATE-----\n";
    const TEST_TLS_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg5EWocTLY0FLh8QHA
n4hqQcj6Ww1lUPQR8+rJ24RBjCuhRANCAARrApXyDvzciYQCSB4ddOaW3lsBSwJN
2Flvw8FFa8BTbiXKvMGtH9XPm4iDhN4rSruKnG1z/KMySl55l3HX7n8J
-----END PRIVATE KEY-----\n";

    #[tokio::test]
    async fn stalled_tls_handshake_does_not_block_other_connections() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio_rustls::rustls;

        let tls = test_tls_config();
        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = tcp.local_addr().unwrap();
        let handshake_timeout = Duration::from_secs(1);
        let listener = TlsListener::new(tcp, tls.acceptor, handshake_timeout).unwrap();
        let app = axum::Router::new().route("/", axum::routing::get(|| async { "ok" }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await });

        // A client that connects and never sends its ClientHello.
        let mut silent = tokio::net::TcpStream::connect(address).await.unwrap();

        let mut roots = rustls::RootCertStore::empty();
        for der in CertificateDer::pem_slice_iter(TEST_TLS_CA_PEM.as_bytes()) {
            roots.add(der.unwrap()).unwrap();
        }
        let connector = tokio_rustls::TlsConnector::from(Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        ));
        let response = tokio::time::timeout(handshake_timeout / 2, async {
            let stream = tokio::net::TcpStream::connect(address).await.unwrap();
            let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
            let mut stream = connector.connect(server_name, stream).await.unwrap();
            stream
                .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).await.unwrap();
            response
        })
        .await
        .expect("a stalled handshake must not delay other connections");
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.ends_with("ok"), "{response}");

        // The stalled client is disconnected once its handshake times out.
        let mut byte = [0_u8; 1];
        let closed = tokio::time::timeout(handshake_timeout * 3, silent.read(&mut byte))
            .await
            .expect("the server closes a stalled handshake");
        assert!(matches!(closed, Ok(0) | Err(_)), "{closed:?}");
        server.abort();
    }

    #[test]
    fn realtime_asr_limits_come_from_the_environment() {
        use izwi_core::serve_runtime::{
            ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS, ENV_ASR_REALTIME_MAX_SESSIONS,
        };
        let env = env_config::snapshot([
            (ENV_ASR_REALTIME_MAX_SESSIONS, "8"),
            (ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS, "300"),
        ]);
        assert_eq!(
            realtime_asr_limits(&env).unwrap(),
            RealtimeAsrSessionConfig {
                max_sessions: Some(8),
                max_lifetime_secs: None,
                idle_timeout_secs: Some(300),
            }
        );
        let invalid = env_config::snapshot([(ENV_ASR_REALTIME_MAX_SESSIONS, "many")]);
        assert!(realtime_asr_limits(&invalid).is_err());
    }

    fn test_tls_config() -> WorkerTlsConfig {
        let files = tempfile::tempdir().unwrap();
        let (cert, key) = (files.path().join("cert.pem"), files.path().join("key.pem"));
        std::fs::write(&cert, TEST_TLS_CERT_PEM).unwrap();
        std::fs::write(&key, TEST_TLS_KEY_PEM).unwrap();
        let (cert, key) = (
            format!("file:{}", cert.display()),
            format!("file:{}", key.display()),
        );
        let env = env_config::snapshot([
            (WORKER_TLS_CERT_REF_ENV, cert.as_str()),
            (WORKER_TLS_KEY_REF_ENV, key.as_str()),
        ]);
        WorkerTlsConfig::from_env(&env).unwrap().unwrap()
    }

    #[tokio::test]
    async fn gateway_client_speaks_http2_to_a_tls_worker() {
        use izwi_serving_client::{WorkerClient, WorkerClientConfig, WorkerClientTlsConfig};

        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = tcp.local_addr().unwrap();
        let listener =
            TlsListener::new(tcp, test_tls_config().acceptor, Duration::from_secs(1)).unwrap();
        let versions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&versions);
        let app = axum::Router::new().route(
            izwi_serving_protocol::WORKER_STATUS_PATH,
            axum::routing::get(move |request: axum::extract::Request| async move {
                seen.lock().unwrap().push(request.version());
                axum::http::StatusCode::SERVICE_UNAVAILABLE
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await });

        let client = WorkerClient::new(
            &format!("https://localhost:{}/", address.port()),
            izwi_serving_protocol::ServiceCredentials {
                credential_id: izwi_serving_protocol::CredentialId::new("gateway").unwrap(),
                bearer_token: izwi_serving_protocol::ServiceBearerToken::new("secret").unwrap(),
            },
            WorkerClientConfig {
                tls: WorkerClientTlsConfig::from_pem(
                    vec![TEST_TLS_CA_PEM.as_bytes().to_vec()],
                    None,
                    None,
                )
                .unwrap(),
                ..WorkerClientConfig::default()
            },
        )
        .unwrap();
        for _ in 0..2 {
            let _ = client.status().await;
        }
        assert_eq!(
            *versions.lock().unwrap(),
            vec![axum::http::Version::HTTP_2; 2]
        );
        server.abort();
    }

    #[test]
    fn tls_config_rejects_partial_cert_and_key_pair() {
        let tls = |pairs: &[(&str, &str)]| {
            WorkerTlsConfig::from_env(&env_config::snapshot(pairs.iter().copied()))
        };
        assert!(tls(&[]).unwrap().is_none());
        assert!(tls(&[(WORKER_TLS_CERT_REF_ENV, "file:/tmp/cert.pem")]).is_err());
        assert!(tls(&[(WORKER_TLS_KEY_REF_ENV, "file:/tmp/key.pem")]).is_err());
    }

    #[test]
    fn tls_config_rejects_non_file_references_and_relative_paths() {
        let result = WorkerTlsConfig::from_env(&env_config::snapshot([
            (WORKER_TLS_CERT_REF_ENV, "http://example.com/cert.pem"),
            (WORKER_TLS_KEY_REF_ENV, "file:relative/key.pem"),
        ]));
        assert!(result.is_err());
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("file: path"),
            "error should mention file: path requirement: {error}"
        );
    }

    #[test]
    fn non_loopback_plaintext_bind_fails_closed_without_acknowledgement() {
        let required = [
            ("IZWI_WORKER_BIND", "0.0.0.0:9470"),
            ("IZWI_WORKER_ARTIFACT_REVISION", "test-rev"),
            ("IZWI_WORKER_CREDENTIAL_ID", "test-cred"),
            ("IZWI_WORKER_BEARER_TOKEN", "test-token-12345678"),
        ];
        let result = WorkerProcessConfig::from_env(&env_config::snapshot(required));
        assert!(result.is_err());
        let error = result.err().unwrap().to_string();
        assert!(error.contains("plaintext worker transport may bind only to a loopback address"));

        let acknowledged = required
            .into_iter()
            .chain([("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND", "1")]);
        let result = WorkerProcessConfig::from_env(&env_config::snapshot(acknowledged));
        assert!(result.is_ok());

        // A value the shared boolean policy does not recognise fails closed.
        let garbled = required
            .into_iter()
            .chain([("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND", "sure")]);
        let error = WorkerProcessConfig::from_env(&env_config::snapshot(garbled))
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND must be one of"),
            "{error}"
        );
    }
}
