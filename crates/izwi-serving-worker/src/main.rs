use anyhow::{bail, Context};
use izwi_core::{backends::RuntimeDeviceAssignment, EngineConfig, ModelVariant, RuntimeService};
use izwi_serving_protocol::*;
use izwi_serving_supervisor::{
    try_acquire_worker_fences, LockNamespace, WorkerFenceLeases, WORKER_GENERATION_FENCE_ENV,
    WORKER_OWNERSHIP_LOCK_ENV,
};
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

    let process = WorkerProcessConfig::from_env()?;
    let _fences = acquire_managed_worker_fences(&process)?;
    if let DeviceAssignment::Cpu { affinity, .. } = &process.assignment {
        if !affinity.is_empty() {
            tracing::warn!(?affinity, "CPU affinity is advisory on this worker build");
        }
    }
    verify_artifact_manifest(&process)?;
    let engine = EngineConfig {
        models_dir: process.models_dir.clone(),
        max_loaded_models: Some(1),
        max_queued_requests: process.max_active_invocations,
        max_scheduler_batch_size: process.max_active_invocations,
        max_retained_sequences: process.max_active_invocations,
        max_staged_transactions: process.max_active_invocations,
        num_threads: process.thread_budget(),
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
    warm_up_chat_runtime(
        &runtime,
        process.variant,
        std::time::Duration::from_secs(30),
    )
    .await
    .context("warm selected model before binding")?;

    let credentials = ServiceCredentials {
        credential_id: process.credential_id.clone(),
        bearer_token: process.bearer_token.clone(),
    };
    let mut features = BTreeSet::from([WorkerFeature::Cancellation, WorkerFeature::AttemptQuery]);
    if process.streaming {
        features.insert(WorkerFeature::Streaming);
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
    let deployment = LoadedDeployment {
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
    tracing::info!(address = %process.bind, model = %process.public_model, backend = ?process.assignment.backend(), "worker ready");
    axum::serve(listener, worker.router())
        .with_graceful_shutdown(shutdown_signal(worker))
        .await
        .context("serve private worker")
}

fn acquire_managed_worker_fences(
    process: &WorkerProcessConfig,
) -> anyhow::Result<Option<WorkerFenceLeases>> {
    if !managed_worker() {
        return Ok(None);
    }
    let ownership = PathBuf::from(required_env(WORKER_OWNERSHIP_LOCK_ENV)?);
    let generation = PathBuf::from(required_env(WORKER_GENERATION_FENCE_ENV)?);
    let directory = ownership
        .parent()
        .context("worker ownership lock must have a parent directory")?;
    if generation.parent() != Some(directory) {
        bail!("worker ownership and generation locks must share one namespace");
    }
    let namespace = LockNamespace::open(directory).context("open worker lock namespace")?;
    let metadata = format!(
        "worker={} incarnation={}",
        process.worker_id, process.incarnation_id
    );
    let leases =
        try_acquire_worker_fences(&namespace, &ownership, &generation, metadata.as_bytes())
            .context("acquire assigned resource and generation fences")?;
    Ok(Some(leases))
}

async fn shutdown_signal<E: izwi_serving_worker::InvocationExecutor>(worker: WorkerService<E>) {
    if tokio::signal::ctrl_c().await.is_ok() {
        worker.begin_draining().await;
        while worker.active_invocations() != 0 {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    }
}

struct WorkerProcessConfig {
    bind: SocketAddr,
    models_dir: PathBuf,
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
}

impl WorkerProcessConfig {
    fn from_env() -> anyhow::Result<Self> {
        let model = env_or("IZWI_WORKER_MODEL", "LFM2.5-1.2B-Instruct-GGUF");
        let variant = izwi_core::parse_chat_model_variant(Some(&model))
            .with_context(|| format!("parse IZWI_WORKER_MODEL={model}"))?;
        if variant != ModelVariant::Lfm2512BInstructGguf {
            bail!("serving worker currently supports only LFM2.5-1.2B-Instruct-GGUF");
        }
        let assignment = parse_assignment()?;
        let max_active_invocations = parse_env("IZWI_WORKER_MAX_ACTIVE", 1usize)?;
        if max_active_invocations == 0 {
            bail!("IZWI_WORKER_MAX_ACTIVE must be non-zero");
        }
        let bind: SocketAddr = env_or("IZWI_WORKER_BIND", "127.0.0.1:9470")
            .parse()
            .context("parse IZWI_WORKER_BIND")?;
        if !bind.ip().is_loopback() {
            bail!("plaintext worker transport may bind only to a loopback address");
        }
        Ok(Self {
            bind,
            models_dir: std::env::var_os("IZWI_MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_models_dir),
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
        })
    }

    fn thread_budget(&self) -> usize {
        match &self.assignment {
            DeviceAssignment::Cpu { thread_budget, .. } => usize::from(*thread_budget),
            DeviceAssignment::Metal { .. } | DeviceAssignment::Cuda { .. } => 1,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
