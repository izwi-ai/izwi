use izwi_serving_protocol::{
    ArtifactRevision, BackendKind, CancellationBehavior, Capability, CredentialId, DeploymentId,
    DeviceAssignment, DeviceId, InputFormat, ModelAlias, ModelGeneration, NodeId, OutputFormat,
    TaskKind, WorkerId,
};
use serde::{Deserialize, Deserializer};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

pub const NODE_CONFIG_SCHEMA_VERSION: u16 = 2;
pub const MAX_NODE_CONFIG_BYTES: usize = 1024 * 1024;
pub const MAX_WORKERS_PER_NODE: usize = 64;
pub const MAX_ENV_NAME_BYTES: usize = 128;
pub const MAX_PATH_BYTES: usize = 4096;
pub const MAX_RESTARTS_PER_WINDOW: u16 = 100;
pub const MAX_POLICY_DURATION_MS: u64 = 24 * 60 * 60 * 1000;
pub const MAX_ACTIVE_INVOCATIONS_PER_WORKER: u32 = 1024;
pub const MAX_REQUEST_BYTES_PER_WORKER: usize = 64 * 1024 * 1024;
pub const MAX_RETAINED_ATTEMPTS_PER_WORKER: usize = 65_536;
pub const MAX_EXECUTION_PROFILE_LABEL_BYTES: usize = 128;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    pub schema_version: u16,
    pub node_id: NodeId,
    pub working_directory: PathBuf,
    pub runtime_directory: PathBuf,
    pub host_memory_budget_bytes: u64,
    pub workers: Vec<WorkerConfig>,
    #[serde(default)]
    pub readiness: ReadinessPolicy,
    #[serde(default)]
    pub restart: RestartPolicy,
    #[serde(default)]
    pub shutdown: ShutdownPolicy,
}

impl NodeConfig {
    pub fn parse_bounded(bytes: &[u8]) -> Result<Self, ConfigError> {
        if bytes.len() > MAX_NODE_CONFIG_BYTES {
            return Err(ConfigError::ConfigTooLarge {
                actual: bytes.len(),
                maximum: MAX_NODE_CONFIG_BYTES,
            });
        }
        let text = std::str::from_utf8(bytes).map_err(ConfigError::Utf8)?;
        toml::from_str(text).map_err(ConfigError::Toml)
    }

    pub fn validate(
        self,
        inventory: &HostInventory,
        binaries: &BinaryCatalog,
    ) -> Result<ValidatedNodeConfig, ConfigError> {
        validate_path_length("working_directory", &self.working_directory)?;
        validate_path_length("runtime_directory", &self.runtime_directory)?;
        if self.schema_version != NODE_CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::UnsupportedSchema(self.schema_version));
        }
        if self.workers.is_empty() || self.workers.len() > MAX_WORKERS_PER_NODE {
            return Err(ConfigError::InvalidWorkerCount(self.workers.len()));
        }
        if self.host_memory_budget_bytes == 0 {
            return Err(ConfigError::ZeroBudget("host_memory_budget_bytes"));
        }
        if self.host_memory_budget_bytes > inventory.allocatable_host_memory_bytes {
            return Err(ConfigError::HostMemoryOvercommit {
                requested: self.host_memory_budget_bytes,
                available: inventory.allocatable_host_memory_bytes,
            });
        }
        validate_directory("working_directory", &self.working_directory)?;
        self.readiness.validate()?;
        self.restart.validate()?;
        self.shutdown.validate()?;
        validate_inventory(inventory)?;

        let effective_cpus = inventory
            .effective_cpu_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();

        let mut worker_ids = BTreeSet::new();
        let mut endpoints = BTreeSet::new();
        let mut exclusive_devices = BTreeSet::new();
        let mut total_threads = 0_u64;
        let mut total_host_memory = 0_u64;
        let mut resolved_binaries = BTreeMap::new();

        for worker in &self.workers {
            if !worker_ids.insert(worker.worker_id.clone()) {
                return Err(ConfigError::DuplicateWorker(worker.worker_id.clone()));
            }
            if worker.bind.port() == 0 || !worker.bind.ip().is_loopback() {
                return Err(ConfigError::InvalidEndpoint {
                    worker: worker.worker_id.clone(),
                    endpoint: worker.bind,
                });
            }
            if !endpoints.insert(worker.bind) {
                return Err(ConfigError::DuplicateEndpoint(worker.bind));
            }
            worker.validate_common()?;

            let binary = binaries
                .resolve(worker.binary)
                .ok_or(ConfigError::UnknownWorkerBinary(worker.binary))?;
            if !binary
                .supported_backends
                .contains(&worker.assignment.backend())
            {
                return Err(ConfigError::UnsupportedBinaryBackend {
                    worker: worker.worker_id.clone(),
                    binary: worker.binary,
                    backend: worker.assignment.backend(),
                });
            }
            let binary_path = validate_executable(&binary.path)?;
            resolved_binaries.insert(worker.worker_id.clone(), binary_path);

            match &worker.assignment {
                DeviceAssignment::Cpu {
                    thread_budget,
                    affinity,
                    host_memory_limit_bytes,
                } => {
                    if *thread_budget == 0 {
                        return Err(ConfigError::ZeroWorkerBudget {
                            worker: worker.worker_id.clone(),
                            field: "thread_budget",
                        });
                    }
                    if *host_memory_limit_bytes == 0 {
                        return Err(ConfigError::ZeroWorkerBudget {
                            worker: worker.worker_id.clone(),
                            field: "host_memory_limit_bytes",
                        });
                    }
                    let mut seen_affinity = BTreeSet::new();
                    for cpu in affinity {
                        if !seen_affinity.insert(*cpu) {
                            return Err(ConfigError::DuplicateCpuAffinity {
                                worker: worker.worker_id.clone(),
                                cpu: *cpu,
                            });
                        }
                        if !effective_cpus.contains(cpu) {
                            return Err(ConfigError::UnavailableCpuAffinity {
                                worker: worker.worker_id.clone(),
                                cpu: *cpu,
                            });
                        }
                    }
                    if !affinity.is_empty() && usize::from(*thread_budget) > affinity.len() {
                        return Err(ConfigError::AffinityBelowThreadBudget {
                            worker: worker.worker_id.clone(),
                            threads: *thread_budget,
                            cpus: affinity.len(),
                        });
                    }
                    total_threads = checked_add(
                        total_threads,
                        u64::from(*thread_budget),
                        "aggregate CPU thread budget",
                    )?;
                    total_host_memory = checked_add(
                        total_host_memory,
                        *host_memory_limit_bytes,
                        "aggregate host memory budget",
                    )?;
                }
                DeviceAssignment::Metal {
                    device_id,
                    process_local_device_index,
                    shared_memory_limit_bytes,
                } => {
                    if *shared_memory_limit_bytes == 0 {
                        return Err(ConfigError::ZeroWorkerBudget {
                            worker: worker.worker_id.clone(),
                            field: "shared_memory_limit_bytes",
                        });
                    }
                    reserve_exclusive(&mut exclusive_devices, BackendKind::Metal, device_id)?;
                    let device = inventory
                        .metal_devices
                        .iter()
                        .find(|candidate| candidate.device_id == *device_id)
                        .ok_or_else(|| ConfigError::UnknownDevice {
                            backend: BackendKind::Metal,
                            device: device_id.clone(),
                        })?;
                    if device.process_local_device_index != *process_local_device_index {
                        return Err(ConfigError::DeviceIndexMismatch {
                            backend: BackendKind::Metal,
                            device: device_id.clone(),
                            configured: *process_local_device_index,
                            discovered: device.process_local_device_index,
                        });
                    }
                    if !device.unified_memory {
                        return Err(ConfigError::MetalIsNotUnified(device_id.clone()));
                    }
                    total_host_memory = checked_add(
                        total_host_memory,
                        *shared_memory_limit_bytes,
                        "aggregate host/unified memory budget",
                    )?;
                }
                DeviceAssignment::Cuda {
                    device_uuid,
                    process_local_device_index,
                    device_memory_limit_bytes,
                    host_memory_limit_bytes,
                } => {
                    if *process_local_device_index != 0 {
                        return Err(ConfigError::CudaLocalIndexMustBeZero {
                            worker: worker.worker_id.clone(),
                            configured: *process_local_device_index,
                        });
                    }
                    if *device_memory_limit_bytes == 0 || *host_memory_limit_bytes == 0 {
                        return Err(ConfigError::ZeroWorkerBudget {
                            worker: worker.worker_id.clone(),
                            field: "CUDA device and host memory limits",
                        });
                    }
                    reserve_exclusive(&mut exclusive_devices, BackendKind::Cuda, device_uuid)?;
                    let device = inventory
                        .cuda_devices
                        .iter()
                        .find(|candidate| candidate.device_uuid == *device_uuid)
                        .ok_or_else(|| ConfigError::UnknownDevice {
                            backend: BackendKind::Cuda,
                            device: device_uuid.clone(),
                        })?;
                    if *device_memory_limit_bytes > device.total_memory_bytes {
                        return Err(ConfigError::DeviceMemoryOvercommit {
                            device: device_uuid.clone(),
                            requested: *device_memory_limit_bytes,
                            available: device.total_memory_bytes,
                        });
                    }
                    total_host_memory = checked_add(
                        total_host_memory,
                        *host_memory_limit_bytes,
                        "aggregate host memory budget",
                    )?;
                }
            }
        }

        if total_threads != 0 && effective_cpus.is_empty() {
            return Err(ConfigError::EmptyCpuInventory);
        }
        if total_threads > effective_cpus.len() as u64 {
            return Err(ConfigError::CpuOvercommit {
                requested: total_threads,
                available: effective_cpus.len(),
            });
        }
        if total_host_memory > self.host_memory_budget_bytes {
            return Err(ConfigError::HostMemoryOvercommit {
                requested: total_host_memory,
                available: self.host_memory_budget_bytes,
            });
        }

        Ok(ValidatedNodeConfig {
            config: self,
            resolved_binaries,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    pub worker_id: WorkerId,
    pub bind: SocketAddr,
    pub binary: WorkerBinaryFlavor,
    pub credential_id: CredentialId,
    pub bearer_token_env: String,
    #[serde(deserialize_with = "deserialize_assignment_strict")]
    pub assignment: DeviceAssignment,
    pub deployment: DeploymentConfig,
    #[serde(default = "default_max_active_invocations")]
    pub max_active_invocations: u32,
    #[serde(default = "default_max_request_bytes")]
    pub max_request_bytes: usize,
    #[serde(default = "default_max_retained_attempts")]
    pub max_retained_attempts: usize,
    #[serde(default = "default_attempt_retention_secs")]
    pub attempt_retention_secs: u64,
    #[serde(default = "default_streaming")]
    pub streaming: bool,
}

impl WorkerConfig {
    fn validate_common(&self) -> Result<(), ConfigError> {
        validate_env_name(&self.bearer_token_env)?;
        validate_path_length("models_directory", &self.deployment.models_directory)?;
        if self.max_active_invocations == 0 {
            return Err(ConfigError::ZeroWorkerBudget {
                worker: self.worker_id.clone(),
                field: "max_active_invocations",
            });
        }
        if self.max_active_invocations > MAX_ACTIVE_INVOCATIONS_PER_WORKER {
            return Err(ConfigError::WorkerLimitTooLarge {
                worker: self.worker_id.clone(),
                field: "max_active_invocations",
                maximum: u64::from(MAX_ACTIVE_INVOCATIONS_PER_WORKER),
            });
        }
        if self.max_request_bytes == 0 {
            return Err(ConfigError::ZeroWorkerBudget {
                worker: self.worker_id.clone(),
                field: "max_request_bytes",
            });
        }
        if self.max_request_bytes > MAX_REQUEST_BYTES_PER_WORKER {
            return Err(ConfigError::WorkerLimitTooLarge {
                worker: self.worker_id.clone(),
                field: "max_request_bytes",
                maximum: MAX_REQUEST_BYTES_PER_WORKER as u64,
            });
        }
        if self.max_retained_attempts > MAX_RETAINED_ATTEMPTS_PER_WORKER {
            return Err(ConfigError::WorkerLimitTooLarge {
                worker: self.worker_id.clone(),
                field: "max_retained_attempts",
                maximum: MAX_RETAINED_ATTEMPTS_PER_WORKER as u64,
            });
        }
        if self.max_retained_attempts < self.max_active_invocations as usize {
            return Err(ConfigError::AttemptRetentionBelowCapacity {
                worker: self.worker_id.clone(),
            });
        }
        if !(1..=86_400).contains(&self.attempt_retention_secs) {
            return Err(ConfigError::InvalidAttemptRetention {
                worker: self.worker_id.clone(),
            });
        }
        if self.deployment.backend != self.assignment.backend() {
            return Err(ConfigError::DeploymentBackendMismatch {
                worker: self.worker_id.clone(),
                assignment: self.assignment.backend(),
                deployment: self.deployment.backend,
            });
        }
        self.deployment.validate_capability(self)?;
        if !self.deployment.models_directory.is_dir() {
            return Err(ConfigError::InvalidDirectory {
                field: "models_directory",
                path: self.deployment.models_directory.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentConfig {
    pub deployment_id: DeploymentId,
    pub public_model: ModelAlias,
    pub artifact_revision: ArtifactRevision,
    pub model_generation: ModelGeneration,
    pub task: TaskKind,
    pub backend: BackendKind,
    pub precision: String,
    pub execution_representation: String,
    pub tokenizer_revision: Option<ArtifactRevision>,
    pub capability: CapabilityProfileConfig,
    pub models_directory: PathBuf,
}

impl DeploymentConfig {
    fn validate_capability(&self, worker: &WorkerConfig) -> Result<(), ConfigError> {
        validate_execution_label("precision", &self.precision)?;
        validate_execution_label("execution_representation", &self.execution_representation)?;
        if self.capability.accepted_input_formats.is_empty() {
            return Err(ConfigError::EmptyCapabilityFormats {
                worker: worker.worker_id.clone(),
                field: "accepted_input_formats",
            });
        }
        if self.capability.output_formats.is_empty() {
            return Err(ConfigError::EmptyCapabilityFormats {
                worker: worker.worker_id.clone(),
                field: "output_formats",
            });
        }
        let request_bytes = u64::try_from(worker.max_request_bytes)
            .map_err(|_| ConfigError::BudgetOverflow("worker request byte budget"))?;
        if self.capability.max_input_bytes != request_bytes {
            return Err(ConfigError::CapabilityInputLimitMismatch {
                worker: worker.worker_id.clone(),
                configured: self.capability.max_input_bytes,
                worker_limit: request_bytes,
            });
        }
        if self.capability.streaming != worker.streaming {
            return Err(ConfigError::CapabilityStreamingMismatch {
                worker: worker.worker_id.clone(),
            });
        }
        if self.capability.max_context_tokens == Some(0)
            || self.capability.max_output_tokens == Some(0)
        {
            return Err(ConfigError::ZeroCapabilityTokenLimit {
                worker: worker.worker_id.clone(),
            });
        }
        Ok(())
    }

    pub(crate) fn expected_capability(&self) -> Capability {
        Capability {
            task: self.task,
            streaming: self.capability.streaming,
            realtime: self.capability.realtime,
            cancellation: self.capability.cancellation,
            accepted_input_formats: self.capability.accepted_input_formats.clone(),
            output_formats: self.capability.output_formats.clone(),
            max_input_bytes: self.capability.max_input_bytes,
            max_context_tokens: self.capability.max_context_tokens,
            max_output_tokens: self.capability.max_output_tokens,
        }
    }
}

/// Exact invocation-shape contract that a deployment must advertise before it is ready.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityProfileConfig {
    pub streaming: bool,
    pub realtime: bool,
    pub cancellation: CancellationBehavior,
    pub accepted_input_formats: BTreeSet<InputFormat>,
    pub output_formats: BTreeSet<OutputFormat>,
    pub max_input_bytes: u64,
    pub max_context_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerBinaryFlavor {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct BinaryRecord {
    pub path: PathBuf,
    pub supported_backends: Vec<BackendKind>,
}

#[derive(Debug, Clone, Default)]
pub struct BinaryCatalog {
    records: BTreeMap<WorkerBinaryFlavor, BinaryRecord>,
}

impl BinaryCatalog {
    pub fn new(records: impl IntoIterator<Item = (WorkerBinaryFlavor, BinaryRecord)>) -> Self {
        Self {
            records: records.into_iter().collect(),
        }
    }

    pub fn resolve(&self, flavor: WorkerBinaryFlavor) -> Option<&BinaryRecord> {
        self.records.get(&flavor)
    }
}

#[derive(Debug, Clone)]
pub struct HostInventory {
    pub effective_cpu_ids: Vec<u16>,
    /// Memory available to Izwi after reserving space for the OS and other software.
    pub allocatable_host_memory_bytes: u64,
    pub metal_devices: Vec<MetalDeviceInventory>,
    pub cuda_devices: Vec<CudaDeviceInventory>,
}

#[derive(Debug, Clone)]
pub struct MetalDeviceInventory {
    pub device_id: DeviceId,
    pub process_local_device_index: u32,
    pub unified_memory: bool,
}

#[derive(Debug, Clone)]
pub struct CudaDeviceInventory {
    pub device_uuid: DeviceId,
    pub host_device_index: u32,
    pub total_memory_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReadinessPolicy {
    pub startup_timeout_ms: u64,
    pub poll_interval_ms: u64,
}

impl Default for ReadinessPolicy {
    fn default() -> Self {
        Self {
            startup_timeout_ms: 120_000,
            poll_interval_ms: 200,
        }
    }
}

impl ReadinessPolicy {
    fn validate(&self) -> Result<(), ConfigError> {
        validate_duration("readiness.startup_timeout_ms", self.startup_timeout_ms)?;
        validate_duration("readiness.poll_interval_ms", self.poll_interval_ms)?;
        if self.poll_interval_ms > self.startup_timeout_ms {
            return Err(ConfigError::InvalidPolicy(
                "readiness poll interval exceeds startup timeout",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RestartPolicy {
    pub initial_backoff_ms: u64,
    pub maximum_backoff_ms: u64,
    pub restart_window_ms: u64,
    pub stable_reset_ms: u64,
    pub max_restarts_per_window: u16,
    pub jitter_percent: u8,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            initial_backoff_ms: 250,
            maximum_backoff_ms: 30_000,
            restart_window_ms: 300_000,
            stable_reset_ms: 600_000,
            max_restarts_per_window: 5,
            jitter_percent: 20,
        }
    }
}

impl RestartPolicy {
    fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("restart.initial_backoff_ms", self.initial_backoff_ms),
            ("restart.maximum_backoff_ms", self.maximum_backoff_ms),
            ("restart.restart_window_ms", self.restart_window_ms),
            ("restart.stable_reset_ms", self.stable_reset_ms),
        ] {
            validate_duration(name, value)?;
        }
        if self.maximum_backoff_ms < self.initial_backoff_ms {
            return Err(ConfigError::InvalidPolicy(
                "maximum restart backoff is below initial backoff",
            ));
        }
        if self.max_restarts_per_window == 0
            || self.max_restarts_per_window > MAX_RESTARTS_PER_WINDOW
        {
            return Err(ConfigError::InvalidPolicy(
                "restart budget must be between one and 100",
            ));
        }
        if self.jitter_percent > 100 {
            return Err(ConfigError::InvalidPolicy(
                "restart jitter percentage exceeds 100",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShutdownPolicy {
    pub drain_grace_ms: u64,
    pub cancellation_grace_ms: u64,
    pub termination_grace_ms: u64,
}

impl Default for ShutdownPolicy {
    fn default() -> Self {
        Self {
            drain_grace_ms: 30_000,
            cancellation_grace_ms: 10_000,
            termination_grace_ms: 5_000,
        }
    }
}

impl ShutdownPolicy {
    fn validate(&self) -> Result<(), ConfigError> {
        validate_duration("shutdown.drain_grace_ms", self.drain_grace_ms)?;
        validate_duration("shutdown.cancellation_grace_ms", self.cancellation_grace_ms)?;
        validate_duration("shutdown.termination_grace_ms", self.termination_grace_ms)
    }
}

#[derive(Debug, Clone)]
pub struct ValidatedNodeConfig {
    config: NodeConfig,
    resolved_binaries: BTreeMap<WorkerId, PathBuf>,
}

impl ValidatedNodeConfig {
    pub fn config(&self) -> &NodeConfig {
        &self.config
    }

    pub fn worker(&self, worker_id: &WorkerId) -> Option<&WorkerConfig> {
        self.config
            .workers
            .iter()
            .find(|worker| worker.worker_id == *worker_id)
    }

    pub fn binary_path(&self, worker_id: &WorkerId) -> Option<&Path> {
        self.resolved_binaries.get(worker_id).map(PathBuf::as_path)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("node configuration is {actual} bytes; maximum is {maximum}")]
    ConfigTooLarge { actual: usize, maximum: usize },
    #[error("node configuration is not UTF-8: {0}")]
    Utf8(#[source] std::str::Utf8Error),
    #[error("invalid node TOML: {0}")]
    Toml(#[source] toml::de::Error),
    #[error("unsupported node configuration schema {0}")]
    UnsupportedSchema(u16),
    #[error("node must configure between one and {MAX_WORKERS_PER_NODE} workers; got {0}")]
    InvalidWorkerCount(usize),
    #[error("{0} must be non-zero")]
    ZeroBudget(&'static str),
    #[error("host memory budget requests {requested} bytes; available budget is {available}")]
    HostMemoryOvercommit { requested: u64, available: u64 },
    #[error("effective CPU inventory is empty")]
    EmptyCpuInventory,
    #[error("host inventory repeats {backend:?} device {device}")]
    DuplicateInventoryDevice {
        backend: BackendKind,
        device: DeviceId,
    },
    #[error("duplicate worker id {0}")]
    DuplicateWorker(WorkerId),
    #[error("worker {worker} has invalid private endpoint {endpoint}; it must be loopback with a non-zero port")]
    InvalidEndpoint {
        worker: WorkerId,
        endpoint: SocketAddr,
    },
    #[error("duplicate private worker endpoint {0}")]
    DuplicateEndpoint(SocketAddr),
    #[error("unknown worker binary flavor {0:?}")]
    UnknownWorkerBinary(WorkerBinaryFlavor),
    #[error("worker {worker} binary {binary:?} does not support {backend:?}")]
    UnsupportedBinaryBackend {
        worker: WorkerId,
        binary: WorkerBinaryFlavor,
        backend: BackendKind,
    },
    #[error("invalid executable {path}: {reason}")]
    InvalidExecutable { path: PathBuf, reason: String },
    #[error("{field} is not an existing directory: {path}")]
    InvalidDirectory { field: &'static str, path: PathBuf },
    #[error("{field} path is longer than {MAX_PATH_BYTES} bytes")]
    PathTooLong { field: &'static str },
    #[error("{field} path must be absolute: {path}")]
    PathNotAbsolute { field: &'static str, path: PathBuf },
    #[error("worker {worker} {field} must be non-zero")]
    ZeroWorkerBudget {
        worker: WorkerId,
        field: &'static str,
    },
    #[error("worker {worker} {field} exceeds the hard maximum {maximum}")]
    WorkerLimitTooLarge {
        worker: WorkerId,
        field: &'static str,
        maximum: u64,
    },
    #[error("worker {worker} repeats CPU {cpu} in its affinity")]
    DuplicateCpuAffinity { worker: WorkerId, cpu: u16 },
    #[error("worker {worker} affinity CPU {cpu} is not in the effective CPU set")]
    UnavailableCpuAffinity { worker: WorkerId, cpu: u16 },
    #[error("worker {worker} requests {threads} threads but affinity contains only {cpus} CPUs")]
    AffinityBelowThreadBudget {
        worker: WorkerId,
        threads: u16,
        cpus: usize,
    },
    #[error("aggregate CPU budget requests {requested} threads; {available} are effective")]
    CpuOvercommit { requested: u64, available: usize },
    #[error("duplicate exclusive {backend:?} device assignment {device}")]
    DuplicateExclusiveDevice {
        backend: BackendKind,
        device: DeviceId,
    },
    #[error("configured {backend:?} device {device} was not discovered")]
    UnknownDevice {
        backend: BackendKind,
        device: DeviceId,
    },
    #[error(
        "{backend:?} device {device} configured local index {configured}, discovered {discovered}"
    )]
    DeviceIndexMismatch {
        backend: BackendKind,
        device: DeviceId,
        configured: u32,
        discovered: u32,
    },
    #[error("Metal device {0} did not report unified memory")]
    MetalIsNotUnified(DeviceId),
    #[error("CUDA worker {worker} local index must be zero after UUID visibility mapping; got {configured}")]
    CudaLocalIndexMustBeZero { worker: WorkerId, configured: u32 },
    #[error("device {device} memory budget requests {requested} bytes; device has {available}")]
    DeviceMemoryOvercommit {
        device: DeviceId,
        requested: u64,
        available: u64,
    },
    #[error(
        "worker {worker} deployment backend {deployment:?} differs from assignment {assignment:?}"
    )]
    DeploymentBackendMismatch {
        worker: WorkerId,
        assignment: BackendKind,
        deployment: BackendKind,
    },
    #[error(
        "{field} must be a non-empty label of at most {MAX_EXECUTION_PROFILE_LABEL_BYTES} bytes"
    )]
    InvalidExecutionProfileLabel { field: &'static str },
    #[error("worker {worker} capability {field} must not be empty")]
    EmptyCapabilityFormats {
        worker: WorkerId,
        field: &'static str,
    },
    #[error(
        "worker {worker} capability max_input_bytes {configured} differs from request limit {worker_limit}"
    )]
    CapabilityInputLimitMismatch {
        worker: WorkerId,
        configured: u64,
        worker_limit: u64,
    },
    #[error("worker {worker} capability streaming differs from the worker streaming setting")]
    CapabilityStreamingMismatch { worker: WorkerId },
    #[error("worker {worker} capability token limits must be non-zero when specified")]
    ZeroCapabilityTokenLimit { worker: WorkerId },
    #[error("worker {worker} retained attempt capacity is below active invocation capacity")]
    AttemptRetentionBelowCapacity { worker: WorkerId },
    #[error("worker {worker} attempt retention must be between one second and one day")]
    InvalidAttemptRetention { worker: WorkerId },
    #[error("invalid bearer-token environment variable name {0:?}")]
    InvalidEnvironmentName(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("integer overflow while calculating {0}")]
    BudgetOverflow(&'static str),
}

fn default_max_active_invocations() -> u32 {
    1
}

fn default_max_request_bytes() -> usize {
    1024 * 1024
}

fn default_max_retained_attempts() -> usize {
    64
}

fn default_attempt_retention_secs() -> u64 {
    300
}

fn default_streaming() -> bool {
    true
}

fn validate_execution_label(field: &'static str, value: &str) -> Result<(), ConfigError> {
    if value.is_empty()
        || value.len() > MAX_EXECUTION_PROFILE_LABEL_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ConfigError::InvalidExecutionProfileLabel { field });
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
enum StrictDeviceAssignment {
    Cpu {
        thread_budget: u16,
        #[serde(default)]
        affinity: Vec<u16>,
        host_memory_limit_bytes: u64,
    },
    Metal {
        device_id: DeviceId,
        process_local_device_index: u32,
        shared_memory_limit_bytes: u64,
    },
    Cuda {
        device_uuid: DeviceId,
        process_local_device_index: u32,
        device_memory_limit_bytes: u64,
        host_memory_limit_bytes: u64,
    },
}

fn deserialize_assignment_strict<'de, D>(deserializer: D) -> Result<DeviceAssignment, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(match StrictDeviceAssignment::deserialize(deserializer)? {
        StrictDeviceAssignment::Cpu {
            thread_budget,
            affinity,
            host_memory_limit_bytes,
        } => DeviceAssignment::Cpu {
            thread_budget,
            affinity,
            host_memory_limit_bytes,
        },
        StrictDeviceAssignment::Metal {
            device_id,
            process_local_device_index,
            shared_memory_limit_bytes,
        } => DeviceAssignment::Metal {
            device_id,
            process_local_device_index,
            shared_memory_limit_bytes,
        },
        StrictDeviceAssignment::Cuda {
            device_uuid,
            process_local_device_index,
            device_memory_limit_bytes,
            host_memory_limit_bytes,
        } => DeviceAssignment::Cuda {
            device_uuid,
            process_local_device_index,
            device_memory_limit_bytes,
            host_memory_limit_bytes,
        },
    })
}

fn validate_duration(name: &'static str, value: u64) -> Result<(), ConfigError> {
    if value == 0 || value > MAX_POLICY_DURATION_MS {
        return Err(ConfigError::InvalidPolicy(match name {
            "readiness.startup_timeout_ms" => "readiness startup timeout is out of range",
            "readiness.poll_interval_ms" => "readiness poll interval is out of range",
            "restart.initial_backoff_ms" => "initial restart backoff is out of range",
            "restart.maximum_backoff_ms" => "maximum restart backoff is out of range",
            "restart.restart_window_ms" => "restart window is out of range",
            "restart.stable_reset_ms" => "stable restart reset is out of range",
            "shutdown.drain_grace_ms" => "shutdown drain grace is out of range",
            "shutdown.cancellation_grace_ms" => "shutdown cancellation grace is out of range",
            _ => "shutdown termination grace is out of range",
        }));
    }
    Ok(())
}

fn validate_env_name(name: &str) -> Result<(), ConfigError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_ENV_NAME_BYTES
        && name
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_uppercase() || byte == b'_')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_');
    if valid {
        Ok(())
    } else {
        Err(ConfigError::InvalidEnvironmentName(name.to_string()))
    }
}

fn validate_path_length(field: &'static str, path: &Path) -> Result<(), ConfigError> {
    if !path.is_absolute() {
        return Err(ConfigError::PathNotAbsolute {
            field,
            path: path.to_path_buf(),
        });
    }
    if path.as_os_str().to_string_lossy().len() > MAX_PATH_BYTES {
        Err(ConfigError::PathTooLong { field })
    } else {
        Ok(())
    }
}

fn validate_directory(field: &'static str, path: &Path) -> Result<(), ConfigError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(ConfigError::InvalidDirectory {
            field,
            path: path.to_path_buf(),
        })
    }
}

fn validate_executable(path: &Path) -> Result<PathBuf, ConfigError> {
    validate_path_length("worker executable", path)?;
    let metadata = fs::symlink_metadata(path).map_err(|error| ConfigError::InvalidExecutable {
        path: path.to_path_buf(),
        reason: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ConfigError::InvalidExecutable {
            path: path.to_path_buf(),
            reason: "not a regular non-symlink file".into(),
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(ConfigError::InvalidExecutable {
                path: path.to_path_buf(),
                reason: "no executable permission bit is set".into(),
            });
        }
    }
    fs::canonicalize(path).map_err(|error| ConfigError::InvalidExecutable {
        path: path.to_path_buf(),
        reason: error.to_string(),
    })
}

fn checked_add(left: u64, right: u64, name: &'static str) -> Result<u64, ConfigError> {
    left.checked_add(right)
        .ok_or(ConfigError::BudgetOverflow(name))
}

fn validate_inventory(inventory: &HostInventory) -> Result<(), ConfigError> {
    let mut metal = BTreeSet::new();
    for device in &inventory.metal_devices {
        if !metal.insert(device.device_id.clone()) {
            return Err(ConfigError::DuplicateInventoryDevice {
                backend: BackendKind::Metal,
                device: device.device_id.clone(),
            });
        }
    }
    let mut cuda = BTreeSet::new();
    for device in &inventory.cuda_devices {
        if !cuda.insert(device.device_uuid.clone()) {
            return Err(ConfigError::DuplicateInventoryDevice {
                backend: BackendKind::Cuda,
                device: device.device_uuid.clone(),
            });
        }
    }
    Ok(())
}

fn reserve_exclusive(
    devices: &mut BTreeSet<(u8, DeviceId)>,
    backend: BackendKind,
    device: &DeviceId,
) -> Result<(), ConfigError> {
    let backend_key = match backend {
        BackendKind::Cpu => 0,
        BackendKind::Metal => 1,
        BackendKind::Cuda => 2,
    };
    if devices.insert((backend_key, device.clone())) {
        Ok(())
    } else {
        Err(ConfigError::DuplicateExclusiveDevice {
            backend,
            device: device.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn executable(directory: &Path) -> PathBuf {
        let path = directory.join("worker");
        fs::File::create(&path)
            .unwrap()
            .write_all(b"worker")
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        path
    }

    fn cpu_config(directory: &Path) -> NodeConfig {
        NodeConfig {
            schema_version: NODE_CONFIG_SCHEMA_VERSION,
            node_id: id("node-a"),
            working_directory: directory.to_path_buf(),
            runtime_directory: directory.join("run"),
            host_memory_budget_bytes: 1024,
            workers: vec![WorkerConfig {
                worker_id: id("cpu-1"),
                bind: "127.0.0.1:9470".parse().unwrap(),
                binary: WorkerBinaryFlavor::Cpu,
                credential_id: id("credential-1"),
                bearer_token_env: "TEST_WORKER_TOKEN".into(),
                assignment: DeviceAssignment::Cpu {
                    thread_budget: 2,
                    affinity: vec![0, 1],
                    host_memory_limit_bytes: 512,
                },
                deployment: DeploymentConfig {
                    deployment_id: id("deployment-1"),
                    public_model: id("model-1"),
                    artifact_revision: id("revision-1"),
                    model_generation: ModelGeneration::new(1).unwrap(),
                    task: TaskKind::Chat,
                    backend: BackendKind::Cpu,
                    precision: "gguf-q4_k_m".into(),
                    execution_representation: "native-lfm2".into(),
                    tokenizer_revision: None,
                    capability: CapabilityProfileConfig {
                        streaming: true,
                        realtime: false,
                        cancellation: CancellationBehavior::Cooperative,
                        accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                        output_formats: BTreeSet::from([OutputFormat::Text]),
                        max_input_bytes: 4096,
                        max_context_tokens: Some(32),
                        max_output_tokens: Some(32),
                    },
                    models_directory: directory.to_path_buf(),
                },
                max_active_invocations: 1,
                max_request_bytes: 4096,
                max_retained_attempts: 2,
                attempt_retention_secs: 30,
                streaming: true,
            }],
            readiness: ReadinessPolicy::default(),
            restart: RestartPolicy::default(),
            shutdown: ShutdownPolicy::default(),
        }
    }

    fn inventory() -> HostInventory {
        HostInventory {
            effective_cpu_ids: vec![0, 1, 2, 3],
            allocatable_host_memory_bytes: 2048,
            metal_devices: Vec::new(),
            cuda_devices: Vec::new(),
        }
    }

    fn catalog(path: PathBuf) -> BinaryCatalog {
        BinaryCatalog::new([(
            WorkerBinaryFlavor::Cpu,
            BinaryRecord {
                path,
                supported_backends: vec![BackendKind::Cpu],
            },
        )])
    }

    #[test]
    fn bounded_parser_rejects_oversize_and_unknown_fields() {
        let error = NodeConfig::parse_bounded(&vec![b'x'; MAX_NODE_CONFIG_BYTES + 1]).unwrap_err();
        assert!(matches!(error, ConfigError::ConfigTooLarge { .. }));
        let error = NodeConfig::parse_bounded(
            b"schema_version=1\nnode_id='n'\nworking_directory='.'\nruntime_directory='.'\nhost_memory_budget_bytes=1\nworkers=[]\nunknown=true",
        )
        .unwrap_err();
        assert!(matches!(error, ConfigError::Toml(_)));

        let nested = br#"
schema_version = 2
node_id = "node-a"
working_directory = "/tmp"
runtime_directory = "/tmp/izwi-run"
host_memory_budget_bytes = 1024

[[workers]]
worker_id = "cpu-1"
bind = "127.0.0.1:9470"
binary = "cpu"
credential_id = "credential-1"
bearer_token_env = "IZWI_SUPERVISOR_SECRET_CPU_1"

[workers.assignment]
backend = "cpu"
thread_budget = 1
host_memory_limit_bytes = 512
unexpected_assignment_field = true

[workers.deployment]
deployment_id = "deployment-1"
public_model = "model-1"
artifact_revision = "revision-1"
model_generation = 1
task = "chat"
backend = "cpu"
precision = "gguf-q4_k_m"
execution_representation = "native-lfm2"
models_directory = "/tmp"

[workers.deployment.capability]
streaming = true
realtime = false
cancellation = "cooperative"
accepted_input_formats = ["chat_messages"]
output_formats = ["text"]
max_input_bytes = 512
max_context_tokens = 32
max_output_tokens = 32
"#;
        assert!(matches!(
            NodeConfig::parse_bounded(nested),
            Err(ConfigError::Toml(_))
        ));
    }

    #[test]
    fn valid_cpu_assignment_resolves_binary() {
        let directory = tempfile::tempdir().unwrap();
        let binary = executable(directory.path());
        let config = cpu_config(directory.path());
        let validated = config
            .validate(&inventory(), &catalog(binary.clone()))
            .unwrap();
        let binary = fs::canonicalize(binary).unwrap();
        assert_eq!(validated.binary_path(&id("cpu-1")), Some(binary.as_path()));
    }

    #[test]
    fn valid_toml_parses_the_strict_nested_assignment() {
        let directory = tempfile::tempdir().unwrap();
        let path = format!("{:?}", directory.path().to_string_lossy());
        let toml = format!(
            r#"
schema_version = 2
node_id = "node-a"
working_directory = {path}
runtime_directory = {path}
host_memory_budget_bytes = 1024

[[workers]]
worker_id = "cpu-1"
bind = "127.0.0.1:9470"
binary = "cpu"
credential_id = "credential-1"
bearer_token_env = "IZWI_SUPERVISOR_SECRET_CPU_1"

[workers.assignment]
backend = "cpu"
thread_budget = 1
affinity = [0]
host_memory_limit_bytes = 512

[workers.deployment]
deployment_id = "deployment-1"
public_model = "model-1"
artifact_revision = "revision-1"
model_generation = 1
task = "chat"
backend = "cpu"
precision = "gguf-q4_k_m"
execution_representation = "native-lfm2"
models_directory = {path}

[workers.deployment.capability]
streaming = true
realtime = false
cancellation = "cooperative"
accepted_input_formats = ["chat_messages"]
output_formats = ["text"]
max_input_bytes = 1048576
max_context_tokens = 32
max_output_tokens = 32
"#
        );
        let parsed = NodeConfig::parse_bounded(toml.as_bytes()).unwrap();
        assert_eq!(parsed.workers.len(), 1);
        assert_eq!(parsed.workers[0].deployment.task, TaskKind::Chat);
        assert_eq!(
            parsed.workers[0].deployment.capability.output_formats,
            BTreeSet::from([OutputFormat::Text])
        );
        assert!(matches!(
            parsed.workers[0].assignment,
            DeviceAssignment::Cpu {
                thread_budget: 1,
                host_memory_limit_bytes: 512,
                ..
            }
        ));

        let unknown_capability_field = toml.replace(
            "max_output_tokens = 32",
            "max_output_tokens = 32\nunknown_capability_field = true",
        );
        assert!(matches!(
            NodeConfig::parse_bounded(unknown_capability_field.as_bytes()),
            Err(ConfigError::Toml(_))
        ));
    }

    #[test]
    fn rejects_capability_profiles_that_disagree_with_worker_limits() {
        let directory = tempfile::tempdir().unwrap();
        let binaries = catalog(executable(directory.path()));

        let mut input_limit = cpu_config(directory.path());
        input_limit.workers[0].deployment.capability.max_input_bytes = 4095;
        assert!(matches!(
            input_limit.validate(&inventory(), &binaries),
            Err(ConfigError::CapabilityInputLimitMismatch { .. })
        ));

        let mut streaming = cpu_config(directory.path());
        streaming.workers[0].deployment.capability.streaming = false;
        assert!(matches!(
            streaming.validate(&inventory(), &binaries),
            Err(ConfigError::CapabilityStreamingMismatch { .. })
        ));
    }

    #[test]
    fn rejects_cpu_and_host_memory_overcommit() {
        let directory = tempfile::tempdir().unwrap();
        let binaries = catalog(executable(directory.path()));
        let mut config = cpu_config(directory.path());
        if let DeviceAssignment::Cpu { thread_budget, .. } = &mut config.workers[0].assignment {
            *thread_budget = 5;
        }
        assert!(matches!(
            config.validate(&inventory(), &binaries),
            Err(ConfigError::AffinityBelowThreadBudget { .. })
                | Err(ConfigError::CpuOvercommit { .. })
        ));

        let mut config = cpu_config(directory.path());
        if let DeviceAssignment::Cpu {
            host_memory_limit_bytes,
            ..
        } = &mut config.workers[0].assignment
        {
            *host_memory_limit_bytes = 1025;
        }
        assert!(matches!(
            config.validate(&inventory(), &binaries),
            Err(ConfigError::HostMemoryOvercommit { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_exclusive_device_assignment() {
        let directory = tempfile::tempdir().unwrap();
        let binary = executable(directory.path());
        let cuda_id: DeviceId = id("GPU-1234");
        let mut config = cpu_config(directory.path());
        let mut second = config.workers[0].clone();
        for (index, worker) in [&mut config.workers[0], &mut second]
            .into_iter()
            .enumerate()
        {
            worker.worker_id = id(if index == 0 { "cuda-1" } else { "cuda-2" });
            worker.bind = format!("127.0.0.1:{}", 9470 + index).parse().unwrap();
            worker.binary = WorkerBinaryFlavor::Cuda;
            worker.assignment = DeviceAssignment::Cuda {
                device_uuid: cuda_id.clone(),
                process_local_device_index: 0,
                device_memory_limit_bytes: 512,
                host_memory_limit_bytes: 128,
            };
            worker.deployment.backend = BackendKind::Cuda;
        }
        config.workers.push(second);
        let binaries = BinaryCatalog::new([(
            WorkerBinaryFlavor::Cuda,
            BinaryRecord {
                path: binary,
                supported_backends: vec![BackendKind::Cuda],
            },
        )]);
        let mut inventory = inventory();
        inventory.cuda_devices.push(CudaDeviceInventory {
            device_uuid: cuda_id,
            host_device_index: 3,
            total_memory_bytes: 1024,
        });
        assert!(matches!(
            config.validate(&inventory, &binaries),
            Err(ConfigError::DuplicateExclusiveDevice { .. })
        ));
    }

    #[test]
    fn rejects_non_loopback_and_bad_secret_environment_name() {
        let directory = tempfile::tempdir().unwrap();
        let binaries = catalog(executable(directory.path()));
        let mut config = cpu_config(directory.path());
        config.workers[0].bind = "0.0.0.0:9470".parse().unwrap();
        assert!(matches!(
            config.validate(&inventory(), &binaries),
            Err(ConfigError::InvalidEndpoint { .. })
        ));

        let mut config = cpu_config(directory.path());
        config.workers[0].bearer_token_env = "bad-name".into();
        assert!(matches!(
            config.validate(&inventory(), &binaries),
            Err(ConfigError::InvalidEnvironmentName(_))
        ));
    }
}
