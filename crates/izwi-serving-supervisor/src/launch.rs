use crate::{ConfigError, ValidatedNodeConfig};
use izwi_serving_protocol::{DeviceAssignment, IncarnationId, ServiceBearerToken, WorkerId};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::PathBuf,
    process::{Command, Stdio},
};

/// Environment variables that may be deliberately inherited by a worker.
/// Everything else is removed before these and the assignment variables are set.
pub const INHERITED_ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "RUST_LOG",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
];

pub const MANAGED_WORKER_ENV: &str = "IZWI_WORKER_MANAGED";
pub const WORKER_INCARNATION_ENV: &str = "IZWI_WORKER_INCARNATION_ID";
pub const WORKER_OWNERSHIP_LOCK_ENV: &str = "IZWI_WORKER_OWNERSHIP_LOCK";
pub const WORKER_GENERATION_FENCE_ENV: &str = "IZWI_WORKER_GENERATION_FENCE_LOCK";
pub const WORKER_MODEL_LOAD_LOCK_ENV: &str = "IZWI_WORKER_MODEL_LOAD_LOCK";
pub const MAX_INHERITED_ENV_VALUE_BYTES: usize = 64 * 1024;
pub const MAX_INHERITED_ENV_TOTAL_BYTES: usize = 256 * 1024;

#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedWorkerSecret {
    pub bearer_token: ServiceBearerToken,
}

impl std::fmt::Debug for ResolvedWorkerSecret {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedWorkerSecret")
            .field("bearer_token", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLockPaths {
    ownership: PathBuf,
    generation_fence: PathBuf,
    model_load: PathBuf,
}

impl WorkerLockPaths {
    pub fn for_worker(
        namespace: &crate::LockNamespace,
        worker_id: &WorkerId,
        assignment: &DeviceAssignment,
    ) -> Self {
        let resource_identity = match assignment {
            DeviceAssignment::Cpu { .. } => format!("cpu-worker:{worker_id}"),
            DeviceAssignment::Metal { device_id, .. } => format!("metal:{device_id}"),
            DeviceAssignment::Cuda { device_uuid, .. } => format!("cuda:{device_uuid}"),
        };
        Self {
            ownership: namespace.resource_path(resource_identity.as_bytes()),
            generation_fence: namespace.generation_fence_path(),
            model_load: namespace.model_load_path(),
        }
    }

    pub fn ownership(&self) -> &std::path::Path {
        &self.ownership
    }

    pub fn generation_fence(&self) -> &std::path::Path {
        &self.generation_fence
    }

    pub fn model_load(&self) -> &std::path::Path {
        &self.model_load
    }

    pub fn try_acquire(
        &self,
        namespace: &crate::LockNamespace,
        metadata: &[u8],
    ) -> Result<crate::WorkerFenceLeases, crate::LockError> {
        crate::try_acquire_worker_fences(
            namespace,
            &self.ownership,
            &self.generation_fence,
            metadata,
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct ChildLaunchSpec {
    program: PathBuf,
    arguments: Vec<OsString>,
    working_directory: PathBuf,
    clear_environment: bool,
    environment: BTreeMap<OsString, OsString>,
    pipe_stdin_for_ownership: bool,
}

impl std::fmt::Debug for ChildLaunchSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let environment = self
            .environment
            .keys()
            .map(|key| key.to_string_lossy())
            .collect::<Vec<_>>();
        formatter
            .debug_struct("ChildLaunchSpec")
            .field("program", &self.program)
            .field("arguments", &self.arguments)
            .field("working_directory", &self.working_directory)
            .field("clear_environment", &self.clear_environment)
            .field("environment_keys", &environment)
            .field("pipe_stdin_for_ownership", &self.pipe_stdin_for_ownership)
            .finish()
    }
}

impl ChildLaunchSpec {
    pub fn program(&self) -> &std::path::Path {
        &self.program
    }

    pub fn arguments(&self) -> &[OsString] {
        &self.arguments
    }

    pub fn working_directory(&self) -> &std::path::Path {
        &self.working_directory
    }

    pub fn clears_environment(&self) -> bool {
        self.clear_environment
    }

    pub fn environment(&self) -> &BTreeMap<OsString, OsString> {
        &self.environment
    }

    pub fn pipes_stdin_for_ownership(&self) -> bool {
        self.pipe_stdin_for_ownership
    }

    /// Constructs, but does not spawn, a command with the launch invariants applied.
    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        self.apply_to(&mut command);
        command
    }

    fn apply_to(&self, command: &mut Command) {
        if self.clear_environment {
            command.env_clear();
        }
        command
            .args(&self.arguments)
            .current_dir(&self.working_directory)
            .envs(&self.environment);
        if self.pipe_stdin_for_ownership {
            command.stdin(Stdio::piped());
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LaunchSpecError {
    #[error("unknown worker {0}")]
    UnknownWorker(WorkerId),
    #[error("resolved worker binary is missing for {0}")]
    MissingBinary(WorkerId),
    #[error("worker {worker} secret source {environment} was not resolved")]
    SecretSourceMismatch {
        worker: WorkerId,
        environment: String,
    },
    #[error("inherited environment variable {name:?} is {actual} bytes; maximum is {maximum}")]
    InheritedEnvironmentValueTooLarge {
        name: OsString,
        actual: usize,
        maximum: usize,
    },
    #[error("allowlisted inherited environment is {actual} bytes; maximum is {maximum}")]
    InheritedEnvironmentTooLarge { actual: usize, maximum: usize },
    #[error("invalid validated configuration: {0}")]
    InvalidConfig(#[from] ConfigError),
}

pub fn build_child_launch_spec(
    node: &ValidatedNodeConfig,
    worker_id: &WorkerId,
    incarnation: &IncarnationId,
    secret_environment_name: &str,
    secret: &ResolvedWorkerSecret,
    inherited_environment: &BTreeMap<OsString, OsString>,
    locks: &WorkerLockPaths,
) -> Result<ChildLaunchSpec, LaunchSpecError> {
    let worker = node
        .worker(worker_id)
        .ok_or_else(|| LaunchSpecError::UnknownWorker(worker_id.clone()))?;
    if worker.bearer_token_env != secret_environment_name {
        return Err(LaunchSpecError::SecretSourceMismatch {
            worker: worker_id.clone(),
            environment: worker.bearer_token_env.clone(),
        });
    }
    let program = node
        .binary_path(worker_id)
        .ok_or_else(|| LaunchSpecError::MissingBinary(worker_id.clone()))?
        .to_path_buf();

    let mut environment = BTreeMap::new();
    let mut inherited_bytes = 0_usize;
    for (key, value) in inherited_environment
        .iter()
        .filter(|(key, _)| allowed_inherited_key(key))
    {
        let value_bytes = os_string_bytes(value);
        if value_bytes > MAX_INHERITED_ENV_VALUE_BYTES {
            return Err(LaunchSpecError::InheritedEnvironmentValueTooLarge {
                name: key.clone(),
                actual: value_bytes,
                maximum: MAX_INHERITED_ENV_VALUE_BYTES,
            });
        }
        inherited_bytes = inherited_bytes
            .checked_add(os_string_bytes(key))
            .and_then(|total| total.checked_add(value_bytes))
            .ok_or(LaunchSpecError::InheritedEnvironmentTooLarge {
                actual: usize::MAX,
                maximum: MAX_INHERITED_ENV_TOTAL_BYTES,
            })?;
        if inherited_bytes > MAX_INHERITED_ENV_TOTAL_BYTES {
            return Err(LaunchSpecError::InheritedEnvironmentTooLarge {
                actual: inherited_bytes,
                maximum: MAX_INHERITED_ENV_TOTAL_BYTES,
            });
        }
        environment.insert(key.clone(), value.clone());
    }
    let mut set = |name: &str, value: OsString| {
        environment.insert(OsString::from(name), value);
    };
    set(MANAGED_WORKER_ENV, "1".into());
    set(WORKER_INCARNATION_ENV, incarnation.as_str().into());
    set("IZWI_WORKER_BIND", worker.bind.to_string().into());
    set("IZWI_WORKER_ID", worker.worker_id.as_str().into());
    set("IZWI_WORKER_NODE_ID", node.config().node_id.as_str().into());
    set(
        "IZWI_WORKER_DEPLOYMENT_ID",
        worker.deployment.deployment_id.as_str().into(),
    );
    set(
        "IZWI_WORKER_MODEL",
        worker.deployment.public_model.as_str().into(),
    );
    set(
        "IZWI_WORKER_ARTIFACT_REVISION",
        worker.deployment.artifact_revision.as_str().into(),
    );
    set(
        "IZWI_WORKER_MODEL_GENERATION",
        worker.deployment.model_generation.get().to_string().into(),
    );
    set(
        "IZWI_MODELS_DIR",
        worker.deployment.models_directory.as_os_str().to_owned(),
    );
    set(
        "IZWI_WORKER_CREDENTIAL_ID",
        worker.credential_id.as_str().into(),
    );
    set(
        "IZWI_WORKER_BEARER_TOKEN",
        secret.bearer_token.expose_secret().into(),
    );
    set(
        "IZWI_WORKER_MAX_ACTIVE",
        worker.max_active_invocations.to_string().into(),
    );
    set(
        "IZWI_WORKER_MAX_REQUEST_BYTES",
        worker.max_request_bytes.to_string().into(),
    );
    set(
        "IZWI_WORKER_MAX_RETAINED_ATTEMPTS",
        worker.max_retained_attempts.to_string().into(),
    );
    set(
        "IZWI_WORKER_ATTEMPT_RETENTION_SECS",
        worker.attempt_retention_secs.to_string().into(),
    );
    set("IZWI_WORKER_STREAMING", worker.streaming.to_string().into());
    set(
        "IZWI_WORKER_DRAIN_GRACE_MS",
        node.config().shutdown.drain_grace_ms.to_string().into(),
    );
    set(
        "IZWI_WORKER_CANCELLATION_GRACE_MS",
        node.config()
            .shutdown
            .cancellation_grace_ms
            .to_string()
            .into(),
    );
    set(
        "IZWI_WORKER_TERMINATION_GRACE_MS",
        node.config()
            .shutdown
            .termination_grace_ms
            .to_string()
            .into(),
    );
    set(
        WORKER_OWNERSHIP_LOCK_ENV,
        locks.ownership().as_os_str().to_owned(),
    );
    set(
        WORKER_GENERATION_FENCE_ENV,
        locks.generation_fence().as_os_str().to_owned(),
    );
    set(
        WORKER_MODEL_LOAD_LOCK_ENV,
        locks.model_load().as_os_str().to_owned(),
    );

    match &worker.assignment {
        DeviceAssignment::Cpu {
            thread_budget,
            affinity,
            host_memory_limit_bytes,
        } => {
            set("IZWI_BACKEND", "cpu".into());
            set("IZWI_WORKER_CPU_THREADS", thread_budget.to_string().into());
            set(
                "IZWI_WORKER_CPU_AFFINITY",
                affinity
                    .iter()
                    .map(u16::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
                    .into(),
            );
            let budget = host_memory_limit_bytes.to_string();
            set("IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES", budget.clone().into());
            // RuntimeService's worker-local ResourceAuthority reads this variable.
            set("IZWI_CPU_MEMORY_BUDGET_BYTES", budget.into());
        }
        DeviceAssignment::Metal {
            device_id,
            process_local_device_index,
            shared_memory_limit_bytes,
        } => {
            set("IZWI_BACKEND", "metal".into());
            set("IZWI_WORKER_EXPECTED_DEVICE_ID", device_id.as_str().into());
            set(
                "IZWI_METAL_DEVICE_ORDINAL",
                process_local_device_index.to_string().into(),
            );
            let budget = shared_memory_limit_bytes.to_string();
            set(
                "IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES",
                budget.clone().into(),
            );
            set("IZWI_METAL_MEMORY_BUDGET_BYTES", budget.into());
        }
        DeviceAssignment::Cuda {
            device_uuid,
            process_local_device_index,
            device_memory_limit_bytes,
            host_memory_limit_bytes,
        } => {
            debug_assert_eq!(*process_local_device_index, 0);
            set("IZWI_BACKEND", "cuda".into());
            // Stable UUID visibility is established before any CUDA runtime is initialized.
            set("CUDA_VISIBLE_DEVICES", device_uuid.as_str().into());
            set(
                "IZWI_WORKER_EXPECTED_DEVICE_UUID",
                device_uuid.as_str().into(),
            );
            set("IZWI_CUDA_DEVICE_ORDINAL", "0".into());
            set(
                "IZWI_CUDA_MEMORY_BUDGET_BYTES",
                device_memory_limit_bytes.to_string().into(),
            );
            set(
                "IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES",
                device_memory_limit_bytes.to_string().into(),
            );
            set(
                "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
                host_memory_limit_bytes.to_string().into(),
            );
            set(
                "IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES",
                host_memory_limit_bytes.to_string().into(),
            );
        }
    }

    Ok(ChildLaunchSpec {
        program,
        arguments: Vec::new(),
        working_directory: node.config().working_directory.clone(),
        clear_environment: true,
        environment,
        pipe_stdin_for_ownership: true,
    })
}

fn allowed_inherited_key(key: &OsStr) -> bool {
    INHERITED_ENV_ALLOWLIST
        .iter()
        .any(|allowed| key == OsStr::new(allowed))
}

fn os_string_bytes(value: &OsStr) -> usize {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        return value.as_bytes().len();
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        return value.encode_wide().count().saturating_mul(2);
    }
    #[allow(unreachable_code)]
    value.to_string_lossy().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BinaryCatalog, BinaryRecord, DeploymentConfig, HostInventory, NodeConfig, ReadinessPolicy,
        RestartPolicy, ShutdownPolicy, WorkerBinaryFlavor, WorkerConfig,
        NODE_CONFIG_SCHEMA_VERSION,
    };
    use izwi_serving_protocol::{
        ArtifactRevision, BackendKind, CredentialId, DeploymentId, ModelAlias, ModelGeneration,
        NodeId,
    };
    use std::{fs, io::Write, path::Path};

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

    fn validated(directory: &Path, assignment: DeviceAssignment) -> ValidatedNodeConfig {
        let backend = assignment.backend();
        let flavor = match backend {
            BackendKind::Cpu => WorkerBinaryFlavor::Cpu,
            BackendKind::Metal => WorkerBinaryFlavor::Metal,
            BackendKind::Cuda => WorkerBinaryFlavor::Cuda,
        };
        let config = NodeConfig {
            schema_version: NODE_CONFIG_SCHEMA_VERSION,
            node_id: id::<NodeId>("node-a"),
            working_directory: directory.into(),
            runtime_directory: directory.join("run"),
            host_memory_budget_bytes: 4096,
            workers: vec![WorkerConfig {
                worker_id: id("worker-a"),
                bind: "127.0.0.1:9470".parse().unwrap(),
                binary: flavor,
                credential_id: id::<CredentialId>("credential-a"),
                bearer_token_env: "WORKER_A_TOKEN".into(),
                assignment: assignment.clone(),
                deployment: DeploymentConfig {
                    deployment_id: id::<DeploymentId>("deployment-a"),
                    public_model: id::<ModelAlias>("model-a"),
                    artifact_revision: id::<ArtifactRevision>("revision-a"),
                    model_generation: ModelGeneration::new(7).unwrap(),
                    backend,
                    models_directory: directory.into(),
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
        };
        let mut inventory = HostInventory {
            effective_cpu_ids: vec![0, 1],
            allocatable_host_memory_bytes: 8192,
            metal_devices: Vec::new(),
            cuda_devices: Vec::new(),
        };
        match &assignment {
            DeviceAssignment::Metal {
                device_id,
                process_local_device_index,
                ..
            } => inventory.metal_devices.push(crate::MetalDeviceInventory {
                device_id: device_id.clone(),
                process_local_device_index: *process_local_device_index,
                unified_memory: true,
            }),
            DeviceAssignment::Cuda {
                device_uuid,
                device_memory_limit_bytes,
                ..
            } => inventory.cuda_devices.push(crate::CudaDeviceInventory {
                device_uuid: device_uuid.clone(),
                host_device_index: 3,
                total_memory_bytes: *device_memory_limit_bytes + 1,
            }),
            DeviceAssignment::Cpu { .. } => {}
        }
        let catalog = BinaryCatalog::new([(
            flavor,
            BinaryRecord {
                path: executable(directory),
                supported_backends: vec![backend],
            },
        )]);
        config.validate(&inventory, &catalog).unwrap()
    }

    fn env_value<'a>(spec: &'a ChildLaunchSpec, name: &str) -> &'a OsStr {
        spec.environment
            .get(OsStr::new(name))
            .map(OsString::as_os_str)
            .unwrap()
    }

    fn build(directory: &Path, assignment: DeviceAssignment) -> ChildLaunchSpec {
        let validated = validated(directory, assignment.clone());
        let inherited = BTreeMap::from([
            (OsString::from("PATH"), OsString::from("/bin")),
            (OsString::from("UNSAFE_PARENT_SECRET"), OsString::from("no")),
            (
                OsString::from("CUDA_VISIBLE_DEVICES"),
                OsString::from("stale"),
            ),
        ]);
        let namespace = crate::LockNamespace::open(directory.join("locks")).unwrap();
        let worker_id = id("worker-a");
        let locks = WorkerLockPaths::for_worker(&namespace, &worker_id, &assignment);
        build_child_launch_spec(
            &validated,
            &worker_id,
            &id("incarnation-a"),
            "WORKER_A_TOKEN",
            &ResolvedWorkerSecret {
                bearer_token: ServiceBearerToken::new("secret-value").unwrap(),
            },
            &inherited,
            &locks,
        )
        .unwrap()
    }

    #[test]
    fn cpu_plan_is_allowlisted_and_sets_both_memory_budgets() {
        let directory = tempfile::tempdir().unwrap();
        let spec = build(
            directory.path(),
            DeviceAssignment::Cpu {
                thread_budget: 2,
                affinity: vec![0, 1],
                host_memory_limit_bytes: 1024,
            },
        );
        assert!(spec.clear_environment);
        assert!(spec.pipe_stdin_for_ownership);
        assert_eq!(env_value(&spec, "PATH"), "/bin");
        assert!(!spec
            .environment
            .contains_key(OsStr::new("UNSAFE_PARENT_SECRET")));
        assert!(!spec
            .environment
            .contains_key(OsStr::new("CUDA_VISIBLE_DEVICES")));
        assert_eq!(env_value(&spec, "IZWI_BACKEND"), "cpu");
        assert_eq!(env_value(&spec, "IZWI_WORKER_CPU_AFFINITY"), "0,1");
        assert_eq!(
            env_value(&spec, "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES"),
            env_value(&spec, "IZWI_CPU_MEMORY_BUDGET_BYTES")
        );
        assert_eq!(env_value(&spec, "IZWI_WORKER_DRAIN_GRACE_MS"), "30000");
        assert_eq!(
            env_value(&spec, "IZWI_WORKER_CANCELLATION_GRACE_MS"),
            "10000"
        );
        assert_eq!(env_value(&spec, "IZWI_WORKER_TERMINATION_GRACE_MS"), "5000");
        let model_load_lock = Path::new(env_value(&spec, WORKER_MODEL_LOAD_LOCK_ENV));
        assert!(model_load_lock.is_absolute());
        assert!(model_load_lock
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("model-load-"));
        let command = spec.command();
        assert_eq!(command.get_program(), spec.program());
        assert_eq!(command.get_current_dir(), Some(spec.working_directory()));
        assert!(command.get_args().next().is_none());
        assert!(command
            .get_envs()
            .any(|(name, value)| name == OsStr::new("IZWI_BACKEND")
                && value == Some(OsStr::new("cpu"))));
    }

    #[test]
    fn cuda_plan_maps_stable_uuid_to_process_ordinal_zero() {
        let directory = tempfile::tempdir().unwrap();
        let spec = build(
            directory.path(),
            DeviceAssignment::Cuda {
                device_uuid: id("GPU-1234"),
                process_local_device_index: 0,
                device_memory_limit_bytes: 2048,
                host_memory_limit_bytes: 512,
            },
        );
        assert_eq!(env_value(&spec, "CUDA_VISIBLE_DEVICES"), "GPU-1234");
        assert_eq!(
            env_value(&spec, "IZWI_WORKER_EXPECTED_DEVICE_UUID"),
            "GPU-1234"
        );
        assert_eq!(env_value(&spec, "IZWI_CUDA_DEVICE_ORDINAL"), "0");
        assert_eq!(env_value(&spec, "IZWI_CUDA_MEMORY_BUDGET_BYTES"), "2048");
        assert_eq!(
            env_value(&spec, "IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES"),
            "512"
        );
    }

    #[test]
    fn metal_plan_maps_identity_ordinal_and_shared_budget() {
        let directory = tempfile::tempdir().unwrap();
        let spec = build(
            directory.path(),
            DeviceAssignment::Metal {
                device_id: id("metal-registry-1"),
                process_local_device_index: 1,
                shared_memory_limit_bytes: 2048,
            },
        );
        assert_eq!(
            env_value(&spec, "IZWI_WORKER_EXPECTED_DEVICE_ID"),
            "metal-registry-1"
        );
        assert_eq!(env_value(&spec, "IZWI_METAL_DEVICE_ORDINAL"), "1");
        assert_eq!(env_value(&spec, "IZWI_METAL_MEMORY_BUDGET_BYTES"), "2048");
    }

    #[test]
    fn launch_debug_never_contains_secret_values() {
        let directory = tempfile::tempdir().unwrap();
        let spec = build(
            directory.path(),
            DeviceAssignment::Cpu {
                thread_budget: 1,
                affinity: vec![0],
                host_memory_limit_bytes: 128,
            },
        );
        let debug = format!("{spec:?}");
        assert!(!debug.contains("secret-value"));
        assert!(debug.contains("IZWI_WORKER_BEARER_TOKEN"));
    }

    #[test]
    fn allowlisted_inherited_environment_values_are_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let assignment = DeviceAssignment::Cpu {
            thread_budget: 1,
            affinity: vec![0],
            host_memory_limit_bytes: 128,
        };
        let validated = validated(directory.path(), assignment.clone());
        let namespace = crate::LockNamespace::open(directory.path().join("locks")).unwrap();
        let worker_id = id("worker-a");
        let locks = WorkerLockPaths::for_worker(&namespace, &worker_id, &assignment);
        let inherited = BTreeMap::from([(
            OsString::from("PATH"),
            OsString::from("x".repeat(MAX_INHERITED_ENV_VALUE_BYTES + 1)),
        )]);
        assert!(matches!(
            build_child_launch_spec(
                &validated,
                &worker_id,
                &id("incarnation-a"),
                "WORKER_A_TOKEN",
                &ResolvedWorkerSecret {
                    bearer_token: ServiceBearerToken::new("secret-value").unwrap(),
                },
                &inherited,
                &locks,
            ),
            Err(LaunchSpecError::InheritedEnvironmentValueTooLarge { .. })
        ));
    }

    #[test]
    fn accelerator_ownership_and_model_load_paths_are_assignment_scoped() {
        let directory = tempfile::tempdir().unwrap();
        let namespace = crate::LockNamespace::open(directory.path().join("locks")).unwrap();
        let assignment = DeviceAssignment::Cuda {
            device_uuid: id("GPU-1234"),
            process_local_device_index: 0,
            device_memory_limit_bytes: 2048,
            host_memory_limit_bytes: 512,
        };
        let first = WorkerLockPaths::for_worker(&namespace, &id("worker-a"), &assignment);
        let second = WorkerLockPaths::for_worker(&namespace, &id("worker-b"), &assignment);
        assert_eq!(first.ownership(), second.ownership());
        assert_eq!(first.model_load(), second.model_load());
        assert_eq!(first.model_load(), namespace.model_load_path().as_path());
    }
}
