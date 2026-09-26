use anyhow::{bail, Context};
use izwi_core::{backends::RuntimeDeviceAssignment, EngineConfig, ModelVariant, RuntimeService};
use izwi_serving_protocol::*;
use izwi_serving_supervisor::{
    try_acquire_worker_fences, LockLease, LockNamespace, WorkerFenceLeases, WorkerLockPaths,
    WORKER_GENERATION_FENCE_ENV, WORKER_MODEL_LOAD_LOCK_ENV, WORKER_MODEL_LOAD_SLOTS_ENV,
    WORKER_OWNERSHIP_LOCK_ENV,
};
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

const WORKER_TLS_CERT_REF_ENV: &str = "IZWI_WORKER_TLS_CERT_REF";
const WORKER_TLS_KEY_REF_ENV: &str = "IZWI_WORKER_TLS_KEY_REF";
const WORKER_TLS_CLIENT_CA_REF_ENV: &str = "IZWI_WORKER_TLS_CLIENT_CA_REF";
const MAX_TLS_PEM_FILE_BYTES: u64 = 256 * 1024;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "izwi_serving_worker=info,izwi_core=info".into()),
        )
        .init();

    let process = WorkerProcessConfig::from_env()?;
    let managed_locks = managed_lock_context(&process)?;
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
    let explicit_prefix_caching = env_flag_value("IZWI_ENABLE_PREFIX_CACHING");
    let enable_prefix_caching = explicit_prefix_caching.unwrap_or(false);
    let prefix_reuse_catalog_auto = match explicit_prefix_caching {
        Some(_) => false,
        None => env_flag_value(izwi_core::serve_runtime::ENV_PREFIX_REUSE_AUTO)
            .unwrap_or(izwi_core::serve_runtime::PREFIX_REUSE_CATALOG_AUTO_DEFAULT),
    };
    let managed_prefix_cache_salt = std::env::var("IZWI_MANAGED_PREFIX_CACHE_SALT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if enable_prefix_caching && managed_prefix_cache_salt.is_none() {
        bail!("IZWI_ENABLE_PREFIX_CACHING=1 requires IZWI_MANAGED_PREFIX_CACHE_SALT");
    }
    let enable_chunked_prefill = parse_env_flag("IZWI_ENABLE_CHUNKED_PREFILL");
    let engine_defaults = izwi_core::config::EngineConfig::default();
    let max_sequence_length =
        match std::env::var("IZWI_MAX_SEQUENCE_LENGTH") {
            Ok(raw) if !raw.trim().is_empty() => Some(
                izwi_core::config::ContextLengthPreference::explicit(raw.trim().parse().map_err(
                    |error| anyhow::anyhow!("invalid IZWI_MAX_SEQUENCE_LENGTH: {error}"),
                )?)?,
            ),
            _ => None,
        };
    let engine = EngineConfig {
        models_dir: process.models_dir.clone(),
        max_loaded_models: Some(1),
        max_queued_requests: process.max_active_invocations,
        max_scheduler_batch_size: process.max_active_invocations,
        max_retained_sequences: process.max_active_invocations,
        max_staged_transactions: process.max_active_invocations,
        num_threads: process.thread_budget(),
        enable_prefix_caching,
        prefix_reuse_catalog_auto,
        managed_prefix_cache_salt,
        max_prefix_cache_pages: parse_env(
            "IZWI_MAX_PREFIX_CACHE_PAGES",
            engine_defaults.max_prefix_cache_pages,
        )?,
        enable_chunked_prefill,
        chunked_prefill_threshold: parse_env(
            "IZWI_CHUNKED_PREFILL_THRESHOLD",
            engine_defaults.chunked_prefill_threshold,
        )?,
        max_sequence_length: max_sequence_length.unwrap_or(engine_defaults.max_sequence_length),
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
    let deployment = match process.task {
        TaskKind::Chat => LoadedDeployment {
            deployment_id: process.deployment_id.clone(),
            public_model: process.public_model.clone(),
            artifact_revision: process.artifact_revision.clone(),
            model_generation: process.model_generation,
            task: TaskKind::Chat,
            backend: process.assignment.backend(),
            precision: "gguf-q4_k_m".into(),
            execution_representation: "native-lfm2".into(),
            tokenizer_revision: None,
            readiness: ModelReadiness::Ready,
            capability: Capability {
                task: TaskKind::Chat,
                streaming: process.streaming,
                realtime: false,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                output_formats: BTreeSet::from([OutputFormat::Text]),
                max_input_bytes: process.max_request_bytes as u64,
                max_context_tokens: Some(32),
                max_output_tokens: Some(32),
            },
            // Routing signals are per-status observations populated from engine
            // telemetry at status time, never static startup metadata.
            kv_cache_usage_pct: None,
            prefix_hits_total: None,
            prefix_queries_total: None,
            prefix_evictions_total: None,
            tokens_out_per_s_ema: None,
            observation_cost_units: None,
        },
        TaskKind::SpeechToText => LoadedDeployment {
            deployment_id: process.deployment_id.clone(),
            public_model: process.public_model.clone(),
            artifact_revision: process.artifact_revision.clone(),
            model_generation: process.model_generation,
            task: TaskKind::SpeechToText,
            backend: process.assignment.backend(),
            precision: "fp16".into(),
            execution_representation: match variant_family_name(&process.variant) {
                "nemotron_asr" => "native-nemotron-realtime".into(),
                _ => "native-voxtral-realtime".into(),
            },
            tokenizer_revision: None,
            readiness: ModelReadiness::Ready,
            capability: Capability {
                task: TaskKind::SpeechToText,
                streaming: true,
                realtime: true,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::PcmAudio]),
                output_formats: BTreeSet::from([OutputFormat::Text]),
                max_input_bytes: process.max_request_bytes as u64,
                // Realtime sessions carry audio, not chat context; the
                // session-level byte budget in the subprotocol bounds input.
                max_context_tokens: None,
                max_output_tokens: None,
            },
            kv_cache_usage_pct: None,
            prefix_hits_total: None,
            prefix_queries_total: None,
            prefix_evictions_total: None,
            tokens_out_per_s_ema: None,
            observation_cost_units: None,
        },
        TaskKind::TextToSpeech => LoadedDeployment {
            deployment_id: process.deployment_id.clone(),
            public_model: process.public_model.clone(),
            artifact_revision: process.artifact_revision.clone(),
            model_generation: process.model_generation,
            task: TaskKind::TextToSpeech,
            backend: process.assignment.backend(),
            precision: "fp32".into(),
            execution_representation: "native-tts-realtime".into(),
            tokenizer_revision: None,
            readiness: ModelReadiness::Ready,
            capability: Capability {
                task: TaskKind::TextToSpeech,
                streaming: true,
                realtime: true,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::Text]),
                output_formats: BTreeSet::from([OutputFormat::PcmAudio]),
                max_input_bytes: process.max_request_bytes as u64,
                // Realtime sessions carry bounded text, not chat context; the
                // session-level byte budget in the subprotocol bounds input.
                max_context_tokens: None,
                max_output_tokens: None,
            },
            kv_cache_usage_pct: None,
            prefix_hits_total: None,
            prefix_queries_total: None,
            prefix_evictions_total: None,
            tokens_out_per_s_ema: None,
            observation_cost_units: None,
        },
    };
    let worker_config = WorkerConfig {
        descriptor,
        deployment,
        credentials,
        max_active_invocations: process.max_active_invocations,
        max_request_bytes: process.max_request_bytes,
        max_retained_attempts: process.max_retained_attempts,
        attempt_retention: process.attempt_retention,
        event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
        max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
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
    };
    result
}

/// Binds the constructed worker service and serves it with the same
/// TLS/plain rules for every executor type.
async fn serve_worker<E: izwi_serving_worker::InvocationExecutor>(
    worker: WorkerService<E>,
    process: &WorkerProcessConfig,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(process.bind)
        .await
        .with_context(|| format!("bind private worker at {}", process.bind))?;
    let shutdown = shutdown_signal(
        worker.clone(),
        process.managed,
        process.drain_grace,
        process.cancellation_grace,
    );
    tracing::info!(
        address = %process.bind,
        model = %process.public_model,
        task = ?process.task,
        backend = ?process.assignment.backend(),
        tls = process.tls.is_some(),
        "worker ready"
    );
    if let Some(tls) = &process.tls {
        let tls_listener = TlsListener {
            inner: listener,
            acceptor: tls.acceptor.clone(),
        };
        axum::serve(tls_listener, worker.router())
            .with_graceful_shutdown(shutdown)
            .await
            .context("serve private TLS worker")
    } else {
        axum::serve(listener, worker.router())
            .with_graceful_shutdown(shutdown)
            .await
            .context("serve private worker")
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
    const MAX_SLOTS: u32 = izwi_serving_supervisor::MAX_MODEL_LOAD_SLOTS;
    let Some(raw) = raw else {
        return Ok(izwi_serving_supervisor::DEFAULT_MODEL_LOAD_SLOTS);
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
    process: &WorkerProcessConfig,
) -> anyhow::Result<Option<ManagedLockContext>> {
    if !process.managed {
        return Ok(None);
    }
    let ownership = PathBuf::from(required_env(WORKER_OWNERSHIP_LOCK_ENV)?);
    let generation = PathBuf::from(required_env(WORKER_GENERATION_FENCE_ENV)?);
    let model_load = PathBuf::from(required_env(WORKER_MODEL_LOAD_LOCK_ENV)?);
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
        model_load_slots: model_load_slots_from(
            std::env::var(WORKER_MODEL_LOAD_SLOTS_ENV).ok().as_deref(),
        )?,
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

async fn shutdown_signal<E: izwi_serving_worker::InvocationExecutor>(
    worker: WorkerService<E>,
    managed: bool,
    drain_grace: Duration,
    cancellation_grace: Duration,
) {
    if managed {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = parent_control_closed() => {}
        }
    } else {
        let _ = tokio::signal::ctrl_c().await;
    }

    worker.begin_draining().await;
    if wait_until_idle(&worker, drain_grace).await {
        return;
    }
    worker.request_cancel_all();
    let _ = wait_until_idle(&worker, cancellation_grace).await;
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

async fn wait_until_idle<E: izwi_serving_worker::InvocationExecutor>(
    worker: &WorkerService<E>,
    timeout: Duration,
) -> bool {
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
    max_active_invocations: usize,
    max_request_bytes: usize,
    max_retained_attempts: usize,
    attempt_retention: Duration,
    streaming: bool,
    managed: bool,
    drain_grace: Duration,
    cancellation_grace: Duration,
}

impl WorkerProcessConfig {
    fn from_env() -> anyhow::Result<Self> {
        let task = parse_worker_task()?;
        let model = env_or(
            "IZWI_WORKER_MODEL",
            match task {
                TaskKind::Chat => "LFM2.5-1.2B-Instruct-GGUF",
                TaskKind::SpeechToText => "Nemotron-3.5-ASR-Streaming-0.6B",
                TaskKind::TextToSpeech => "Kokoro-82M",
            },
        );
        let variant = match task {
            TaskKind::Chat => {
                let variant = izwi_core::parse_chat_model_variant(Some(&model))
                    .with_context(|| format!("parse IZWI_WORKER_MODEL={model}"))?;
                if variant != ModelVariant::Lfm2512BInstructGguf
                    && variant != ModelVariant::Qwen3827BFp8
                {
                    bail!("serving worker supports only LFM2.5-1.2B-Instruct-GGUF and Qwen3.8-27B-FP8 for chat");
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
        };
        let assignment = parse_assignment()?;
        let max_active_invocations = parse_env("IZWI_WORKER_MAX_ACTIVE", 1usize)?;
        if max_active_invocations == 0 {
            bail!("IZWI_WORKER_MAX_ACTIVE must be non-zero");
        }
        let bind: SocketAddr = env_or("IZWI_WORKER_BIND", "127.0.0.1:9470")
            .parse()
            .context("parse IZWI_WORKER_BIND")?;
        let tls = WorkerTlsConfig::from_env()?;
        let allow_insecure_plaintext = std::env::var("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND")
            .ok()
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE"));
        if tls.is_none() && !bind.ip().is_loopback() && !allow_insecure_plaintext {
            bail!(
                "plaintext worker transport may bind only to a loopback address; \
                 configure {WORKER_TLS_CERT_REF_ENV} and {WORKER_TLS_KEY_REF_ENV} for \
                 non-loopback TLS, or explicitly acknowledge container networking with \
                 IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND=1"
            );
        }
        if allow_insecure_plaintext && !bind.ip().is_loopback() {
            tracing::warn!(
                address = %bind,
                "binding plaintext worker to non-loopback address via IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND"
            );
        }
        Ok(Self {
            bind,
            tls,
            models_dir: std::env::var_os("IZWI_MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_models_dir),
            task,
            variant,
            public_model: ModelAlias::new(model)?,
            worker_id: WorkerId::new(env_or("IZWI_WORKER_ID", "local-cpu-1"))?,
            node_id: NodeId::new(env_or("IZWI_WORKER_NODE_ID", "local-node"))?,
            incarnation_id: IncarnationId::new(
                std::env::var("IZWI_WORKER_INCARNATION_ID")
                    .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string()),
            )?,
            deployment_id: DeploymentId::new(env_or("IZWI_WORKER_DEPLOYMENT_ID", "lfm25-cpu-v1"))?,
            artifact_revision: ArtifactRevision::new(required_env(
                "IZWI_WORKER_ARTIFACT_REVISION",
            )?)?,
            model_generation: ModelGeneration::new(parse_env(
                "IZWI_WORKER_MODEL_GENERATION",
                1u64,
            )?)?,
            credential_id: CredentialId::new(required_env("IZWI_WORKER_CREDENTIAL_ID")?)?,
            bearer_token: ServiceBearerToken::new(required_env("IZWI_WORKER_BEARER_TOKEN")?)?,
            assignment,
            max_active_invocations,
            max_request_bytes: parse_env(
                "IZWI_WORKER_MAX_REQUEST_BYTES",
                DEFAULT_MAX_REQUEST_BYTES,
            )?,
            max_retained_attempts: parse_env(
                "IZWI_WORKER_MAX_RETAINED_ATTEMPTS",
                DEFAULT_MAX_RETAINED_ATTEMPTS,
            )?,
            attempt_retention: Duration::from_secs(parse_env(
                "IZWI_WORKER_ATTEMPT_RETENTION_SECS",
                DEFAULT_ATTEMPT_RETENTION.as_secs(),
            )?),
            streaming: parse_env("IZWI_WORKER_STREAMING", true)?,
            managed: managed_worker(),
            drain_grace: Duration::from_millis(parse_bounded_duration_env(
                "IZWI_WORKER_DRAIN_GRACE_MS",
                30_000,
            )?),
            cancellation_grace: Duration::from_millis(parse_bounded_duration_env(
                "IZWI_WORKER_CANCELLATION_GRACE_MS",
                10_000,
            )?),
        })
    }

    fn thread_budget(&self) -> usize {
        match &self.assignment {
            DeviceAssignment::Cpu { thread_budget, .. } => usize::from(*thread_budget),
            DeviceAssignment::Metal { .. } | DeviceAssignment::Cuda { .. } => 1,
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
    fn from_env() -> anyhow::Result<Option<Self>> {
        let cert_ref = std::env::var(WORKER_TLS_CERT_REF_ENV).ok();
        let key_ref = std::env::var(WORKER_TLS_KEY_REF_ENV).ok();
        let client_ca_ref = std::env::var(WORKER_TLS_CLIENT_CA_REF_ENV).ok();

        match (cert_ref, key_ref) {
            (Some(cert_ref), Some(key_ref)) => {
                let cert_bytes = load_pem_file(&cert_ref, "server certificate")?;
                let key_bytes = load_pem_file(&key_ref, "server private key")?;
                let cert_items = rustls_pemfile::read_all(&mut cert_bytes.as_slice())
                    .map_err(|_| anyhow::anyhow!("server certificate PEM is invalid"))?;
                let key_items = rustls_pemfile::read_all(&mut key_bytes.as_slice())
                    .map_err(|_| anyhow::anyhow!("server private key PEM is invalid"))?;
                let certs: Vec<_> = cert_items
                    .into_iter()
                    .filter_map(|item| match item {
                        rustls_pemfile::Item::X509Certificate(der) => {
                            Some(tokio_rustls::rustls::pki_types::CertificateDer::from(der))
                        }
                        _ => None,
                    })
                    .collect();
                if certs.is_empty() {
                    bail!("server certificate file contains no certificate PEM blocks");
                }
                let key = key_items
                    .into_iter()
                    .filter_map(|item| match item {
                        rustls_pemfile::Item::PKCS8Key(der) => {
                            Some(tokio_rustls::rustls::pki_types::PrivateKeyDer::Pkcs8(
                                tokio_rustls::rustls::pki_types::PrivatePkcs8KeyDer::from(der),
                            ))
                        }
                        rustls_pemfile::Item::RSAKey(der) => {
                            Some(tokio_rustls::rustls::pki_types::PrivateKeyDer::Pkcs1(
                                tokio_rustls::rustls::pki_types::PrivatePkcs1KeyDer::from(der),
                            ))
                        }
                        rustls_pemfile::Item::ECKey(der) => {
                            Some(tokio_rustls::rustls::pki_types::PrivateKeyDer::Sec1(
                                tokio_rustls::rustls::pki_types::PrivateSec1KeyDer::from(der),
                            ))
                        }
                        _ => None,
                    })
                    .next()
                    .context("server private key file contains no key PEM blocks")?;

                let config = if let Some(ca_ref) = client_ca_ref {
                    let ca_bytes = load_pem_file(&ca_ref, "client CA")?;
                    let ca_items = rustls_pemfile::read_all(&mut ca_bytes.as_slice())
                        .map_err(|_| anyhow::anyhow!("client CA PEM is invalid"))?;
                    let client_certs: Vec<_> = ca_items
                        .into_iter()
                        .filter_map(|item| match item {
                            rustls_pemfile::Item::X509Certificate(der) => Some(der),
                            _ => None,
                        })
                        .collect();
                    if client_certs.is_empty() {
                        bail!("client CA file contains no certificate PEM blocks");
                    }
                    let mut roots = tokio_rustls::rustls::RootCertStore::empty();
                    for cert in client_certs {
                        roots
                            .add(tokio_rustls::rustls::pki_types::CertificateDer::from(cert))
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

fn load_pem_file(reference: &str, label: &str) -> anyhow::Result<Vec<u8>> {
    let path = reference
        .strip_prefix("file:")
        .with_context(|| format!("{label} reference must be a file: path"))?;
    let path = Path::new(path);
    if !path.is_absolute() {
        bail!("{label} reference must be an absolute file: path");
    }
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("read {label} metadata at {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{label} reference must point to a regular file");
    }
    if metadata.len() > MAX_TLS_PEM_FILE_BYTES {
        bail!("{label} file exceeds {MAX_TLS_PEM_FILE_BYTES} bytes");
    }
    let bytes =
        std::fs::read(path).with_context(|| format!("read {label} at {}", path.display()))?;
    Ok(bytes)
}

struct TlsListener {
    inner: tokio::net::TcpListener,
    acceptor: tokio_rustls::TlsAcceptor,
}

impl axum::serve::Listener for TlsListener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.inner.accept().await {
                Ok((stream, addr)) => match self.acceptor.accept(stream).await {
                    Ok(tls_stream) => return (tls_stream, addr),
                    Err(error) => {
                        tracing::warn!(%error, "worker TLS handshake failed");
                    }
                },
                Err(error) => {
                    tracing::warn!(%error, "worker TCP accept failed");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

fn parse_assignment() -> anyhow::Result<DeviceAssignment> {
    match env_or("IZWI_BACKEND", "cpu")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => {
            let thread_budget = parse_env("IZWI_WORKER_CPU_THREADS", 1u16)?;
            if thread_budget == 0 {
                bail!("IZWI_WORKER_CPU_THREADS must be non-zero");
            }
            let host_memory_limit_bytes = parse_positive_env(
                "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget("IZWI_CPU_MEMORY_BUDGET_BYTES", host_memory_limit_bytes)?;
            Ok(DeviceAssignment::Cpu {
                thread_budget,
                affinity: parse_cpu_affinity()?,
                host_memory_limit_bytes,
            })
        }
        "metal" => {
            let shared_memory_limit_bytes = parse_positive_env(
                "IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget("IZWI_METAL_MEMORY_BUDGET_BYTES", shared_memory_limit_bytes)?;
            Ok(DeviceAssignment::Metal {
                device_id: DeviceId::new(required_env("IZWI_WORKER_EXPECTED_DEVICE_ID")?)?,
                process_local_device_index: parse_env("IZWI_METAL_DEVICE_ORDINAL", 0u32)?,
                shared_memory_limit_bytes,
            })
        }
        "cuda" => {
            let device_memory_limit_bytes = parse_positive_env(
                "IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget("IZWI_CUDA_MEMORY_BUDGET_BYTES", device_memory_limit_bytes)?;
            let host_memory_limit_bytes = parse_positive_env(
                "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?;
            verify_matching_budget(
                "IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES",
                host_memory_limit_bytes,
            )?;
            Ok(DeviceAssignment::Cuda {
                device_uuid: DeviceId::new(required_env("IZWI_WORKER_EXPECTED_DEVICE_UUID")?)?,
                process_local_device_index: parse_env("IZWI_CUDA_DEVICE_ORDINAL", 0u32)?,
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
    })
}

fn parse_cpu_affinity() -> anyhow::Result<Vec<u16>> {
    let Some(raw) = std::env::var_os("IZWI_WORKER_CPU_AFFINITY") else {
        return Ok(Vec::new());
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("IZWI_WORKER_CPU_AFFINITY must be UTF-8"))?;
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

fn parse_positive_env<T>(name: &str, fallback: T) -> anyhow::Result<T>
where
    T: std::str::FromStr + Default + PartialEq,
    T::Err: std::fmt::Display,
{
    let value = parse_env(name, fallback)?;
    if value == T::default() {
        bail!("{name} must be non-zero");
    }
    Ok(value)
}

fn parse_bounded_duration_env(name: &str, fallback_ms: u64) -> anyhow::Result<u64> {
    let value = parse_positive_env(name, fallback_ms)?;
    if value > 24 * 60 * 60 * 1000 {
        bail!("{name} must not exceed one day");
    }
    Ok(value)
}

fn verify_matching_budget(name: &str, advertised: u64) -> anyhow::Result<()> {
    match std::env::var(name) {
        Ok(raw) => {
            let enforced = raw
                .parse::<u64>()
                .with_context(|| format!("parse {name}"))?;
            if enforced != advertised {
                bail!("{name} must match the worker assignment budget");
            }
        }
        Err(std::env::VarError::NotPresent) if managed_worker() => {
            bail!("managed worker requires {name} to enforce its assignment budget");
        }
        Err(std::env::VarError::NotPresent) => {
            tracing::warn!(budget = advertised, variable = name, "local worker memory assignment is advisory because no runtime budget was configured");
        }
        Err(error) => bail!("failed to read {name}: {error}"),
    }
    Ok(())
}

fn managed_worker() -> bool {
    std::env::var("IZWI_WORKER_MANAGED")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
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

/// Stable family label for the deployment's execution representation.
fn variant_family_name(variant: &ModelVariant) -> &'static str {
    match variant.family() {
        izwi_core::catalog::ModelFamily::NemotronAsr => "nemotron_asr",
        izwi_core::catalog::ModelFamily::Voxtral => "voxtral",
        _ => "other",
    }
}

fn env_or(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.into())
}

/// Parses `IZWI_WORKER_TASK`. Chat remains the default so existing
/// deployments are byte-identical.
fn parse_worker_task() -> anyhow::Result<TaskKind> {
    match env_or("IZWI_WORKER_TASK", "chat")
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

fn required_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        bail!("{name} must not be empty");
    }
    Ok(value)
}

fn parse_env<T>(name: &str, fallback: T) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid {name}: {error}")),
        Err(_) => Ok(fallback),
    }
}

/// Read a tri-state env flag: `None` when unset or unrecognized, so an
/// absent variable is distinguishable from an explicit `0`/`false`.
fn env_flag_value(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn parse_env_flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
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

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn model_load_slots_parse_is_bounded_and_defaults_to_one() {
        assert_eq!(model_load_slots_from(None).unwrap(), 1);
        assert_eq!(model_load_slots_from(Some("1")).unwrap(), 1);
        assert_eq!(model_load_slots_from(Some(" 4 ")).unwrap(), 4);
        assert_eq!(
            model_load_slots_from(Some("64")).unwrap(),
            izwi_serving_supervisor::MAX_MODEL_LOAD_SLOTS
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

    #[test]
    fn tls_config_rejects_partial_cert_and_key_pair() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var(WORKER_TLS_CERT_REF_ENV);
        std::env::remove_var(WORKER_TLS_KEY_REF_ENV);
        std::env::remove_var(WORKER_TLS_CLIENT_CA_REF_ENV);
        assert!(WorkerTlsConfig::from_env().unwrap().is_none());

        std::env::set_var(WORKER_TLS_CERT_REF_ENV, "file:/tmp/cert.pem");
        assert!(WorkerTlsConfig::from_env().is_err());

        std::env::remove_var(WORKER_TLS_CERT_REF_ENV);
        std::env::set_var(WORKER_TLS_KEY_REF_ENV, "file:/tmp/key.pem");
        assert!(WorkerTlsConfig::from_env().is_err());

        std::env::remove_var(WORKER_TLS_KEY_REF_ENV);
        assert!(WorkerTlsConfig::from_env().unwrap().is_none());
    }

    #[test]
    fn tls_config_rejects_non_file_references_and_relative_paths() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var(WORKER_TLS_CERT_REF_ENV, "http://example.com/cert.pem");
        std::env::set_var(WORKER_TLS_KEY_REF_ENV, "file:relative/key.pem");
        let result = WorkerTlsConfig::from_env();
        assert!(result.is_err());
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("file: path"),
            "error should mention file: path requirement: {error}"
        );
        std::env::remove_var(WORKER_TLS_CERT_REF_ENV);
        std::env::remove_var(WORKER_TLS_KEY_REF_ENV);
    }

    #[test]
    fn non_loopback_plaintext_bind_fails_closed_without_acknowledgement() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND");
        std::env::remove_var(WORKER_TLS_CERT_REF_ENV);
        std::env::remove_var(WORKER_TLS_KEY_REF_ENV);
        std::env::set_var("IZWI_WORKER_BIND", "0.0.0.0:9470");
        std::env::set_var("IZWI_WORKER_ARTIFACT_REVISION", "test-rev");
        std::env::set_var("IZWI_WORKER_CREDENTIAL_ID", "test-cred");
        std::env::set_var("IZWI_WORKER_BEARER_TOKEN", "test-token-12345678");

        let result = WorkerProcessConfig::from_env();
        assert!(result.is_err());
        let error = result.err().unwrap().to_string();
        assert!(error.contains("plaintext worker transport may bind only to a loopback address"));

        std::env::set_var("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND", "1");
        let result = WorkerProcessConfig::from_env();
        assert!(result.is_ok());

        std::env::remove_var("IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND");
        std::env::remove_var("IZWI_WORKER_BIND");
        std::env::remove_var("IZWI_WORKER_ARTIFACT_REVISION");
        std::env::remove_var("IZWI_WORKER_CREDENTIAL_ID");
        std::env::remove_var("IZWI_WORKER_BEARER_TOKEN");
    }
}
