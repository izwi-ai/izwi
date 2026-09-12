use anyhow::{bail, Context};
use izwi_core::{backends::BackendPreference, EngineConfig, ModelVariant, RuntimeService};
use izwi_serving_protocol::*;
use izwi_serving_worker::{
    warm_up_chat_runtime, RuntimeChatExecutor, WorkerConfig, WorkerService,
    DEFAULT_ATTEMPT_RETENTION, DEFAULT_EVENT_CHANNEL_CAPACITY, DEFAULT_MAX_EVENT_BYTES,
    DEFAULT_MAX_REQUEST_BYTES, DEFAULT_MAX_RETAINED_ATTEMPTS,
};
use std::{
    collections::BTreeSet, net::SocketAddr, path::Path, path::PathBuf, sync::Arc, time::Duration,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "izwi_serving_worker=info,izwi_core=info".into()),
        )
        .init();

    let process = CpuWorkerProcessConfig::from_env()?;
    verify_artifact_manifest(&process)?;
    let mut engine = EngineConfig {
        backend: BackendPreference::Cpu,
        models_dir: process.models_dir.clone(),
        max_loaded_models: Some(1),
        max_queued_requests: process.max_active_invocations,
        max_scheduler_batch_size: process.max_active_invocations,
        max_retained_sequences: process.max_active_invocations,
        max_staged_transactions: process.max_active_invocations,
        num_threads: process.thread_budget as usize,
        ..EngineConfig::default()
    };
    // A worker process owns one explicitly assigned backend for its entire life.
    // RuntimeService rejects unavailable explicit backends instead of falling back.
    engine.backend = BackendPreference::Cpu;
    let runtime = Arc::new(RuntimeService::new(engine).context("initialize CPU runtime")?);
    runtime
        .load_model(process.variant)
        .await
        .with_context(|| format!("load selected model {}", process.public_model))?;
    warm_up_chat_runtime(
        &runtime,
        process.variant,
        std::time::Duration::from_secs(30),
    )
    .await
    .context("warm selected CPU model before binding")?;

    let credentials = ServiceCredentials {
        credential_id: CredentialId::new(process.credential_id)?,
        bearer_token: ServiceBearerToken::new(process.bearer_token)?,
    };
    let descriptor = WorkerDescriptor {
        schema_version: PROTOCOL_V1,
        supported_protocol_versions: vec![PROTOCOL_V1],
        worker_id: WorkerId::new(process.worker_id)?,
        node_id: NodeId::new(process.node_id)?,
        incarnation_id: IncarnationId::new(uuid::Uuid::new_v4().to_string())?,
        build_version: env!("CARGO_PKG_VERSION").into(),
        assignment: DeviceAssignment::Cpu {
            thread_budget: process.thread_budget,
            affinity: Vec::new(),
            host_memory_limit_bytes: process.host_memory_limit_bytes,
        },
        features: BTreeSet::from([
            WorkerFeature::Streaming,
            WorkerFeature::Cancellation,
            WorkerFeature::AttemptQuery,
        ]),
    };
    let deployment = LoadedDeployment {
        deployment_id: DeploymentId::new(process.deployment_id)?,
        public_model: ModelAlias::new(process.public_model.clone())?,
        artifact_revision: ArtifactRevision::new(process.artifact_revision)?,
        model_generation: ModelGeneration::new(1)?,
        task: TaskKind::Chat,
        backend: BackendKind::Cpu,
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
    };
    let worker = WorkerService::new(
        WorkerConfig {
            descriptor,
            deployment,
            credentials,
            max_active_invocations: process.max_active_invocations,
            max_request_bytes: process.max_request_bytes,
            max_retained_attempts: process.max_retained_attempts,
            attempt_retention: process.attempt_retention,
            event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
            max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
        },
        RuntimeChatExecutor::new(runtime, process.variant, process.streaming),
    )?;

    let listener = tokio::net::TcpListener::bind(process.bind)
        .await
        .with_context(|| format!("bind private worker at {}", process.bind))?;
    tracing::info!(address = %process.bind, model = %process.public_model, "CPU worker ready");
    axum::serve(listener, worker.router())
        .with_graceful_shutdown(shutdown_signal(worker))
        .await
        .context("serve private worker")
}

async fn shutdown_signal<E: izwi_serving_worker::InvocationExecutor>(worker: WorkerService<E>) {
    if tokio::signal::ctrl_c().await.is_ok() {
        worker.begin_draining().await;
        while worker.active_invocations() != 0 {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    }
}

struct CpuWorkerProcessConfig {
    bind: SocketAddr,
    models_dir: PathBuf,
    variant: ModelVariant,
    public_model: String,
    worker_id: String,
    node_id: String,
    deployment_id: String,
    artifact_revision: String,
    credential_id: String,
    bearer_token: String,
    thread_budget: u16,
    host_memory_limit_bytes: u64,
    max_active_invocations: usize,
    max_request_bytes: usize,
    max_retained_attempts: usize,
    attempt_retention: Duration,
    streaming: bool,
}

impl CpuWorkerProcessConfig {
    fn from_env() -> anyhow::Result<Self> {
        let model = env_or("IZWI_WORKER_MODEL", "LFM2.5-1.2B-Instruct-GGUF");
        let variant = izwi_core::parse_chat_model_variant(Some(&model))
            .with_context(|| format!("parse IZWI_WORKER_MODEL={model}"))?;
        if variant != ModelVariant::Lfm2512BInstructGguf {
            bail!("Phase-2 CPU worker supports only LFM2.5-1.2B-Instruct-GGUF");
        }
        let thread_budget = parse_env("IZWI_WORKER_CPU_THREADS", 1u16)?;
        if thread_budget == 0 {
            bail!("IZWI_WORKER_CPU_THREADS must be non-zero");
        }
        let max_active_invocations = parse_env("IZWI_WORKER_MAX_ACTIVE", 1usize)?;
        if max_active_invocations == 0 {
            bail!("IZWI_WORKER_MAX_ACTIVE must be non-zero");
        }
        let bind: SocketAddr = env_or("IZWI_WORKER_BIND", "127.0.0.1:9470")
            .parse()
            .context("parse IZWI_WORKER_BIND")?;
        if !bind.ip().is_loopback() {
            bail!("Phase-2 plaintext worker transport may bind only to a loopback address");
        }
        Ok(Self {
            bind,
            models_dir: std::env::var_os("IZWI_MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_models_dir),
            variant,
            public_model: model,
            worker_id: env_or("IZWI_WORKER_ID", "local-cpu-1"),
            node_id: env_or("IZWI_WORKER_NODE_ID", "local-node"),
            deployment_id: env_or("IZWI_WORKER_DEPLOYMENT_ID", "lfm25-cpu-v1"),
            artifact_revision: required_env("IZWI_WORKER_ARTIFACT_REVISION")?,
            credential_id: required_env("IZWI_WORKER_CREDENTIAL_ID")?,
            bearer_token: required_env("IZWI_WORKER_BEARER_TOKEN")?,
            thread_budget,
            host_memory_limit_bytes: parse_env(
                "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
                2 * 1024 * 1024 * 1024u64,
            )?,
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
        })
    }
}

fn verify_artifact_manifest(process: &CpuWorkerProcessConfig) -> anyhow::Result<()> {
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
        || manifest.revision != process.artifact_revision
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

fn env_or(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.into())
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

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}
