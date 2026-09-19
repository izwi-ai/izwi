//! Bounded, receiver-clock worker discovery and deterministic request routing.
//!
//! Registration is deliberately an approved control-plane operation: callers
//! must supply an already configured authenticated [`WorkerClient`] together
//! with the descriptor, allowed deployments, and validated capacity. Worker
//! status can update only that exact worker incarnation. The registry is an
//! efficiency mechanism; the selected worker remains the authoritative source
//! of admission. The registry exposes only one exact-worker exclusion for a
//! caller-owned, provably pre-acceptance alternate attempt; it never retries.

use izwi_serving_client::WorkerClient;
use izwi_serving_protocol::{
    ArtifactRevision, BackendKind, CancellationBehavior, Capability, DeploymentId, IncarnationId,
    InputFormat, LoadedDeployment, ModelAlias, ModelGeneration, ModelReadiness, NodeId,
    OutputFormat, SchemaVersion, TaskKind, WorkerDescriptor, WorkerFeature, WorkerId,
    WorkerProcessState, WorkerStatus, PROTOCOL_V1,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

const MAX_PROTOCOL_VERSIONS: usize = 8;
const MAX_BUILD_VERSION_BYTES: usize = 256;
const MAX_EXECUTION_LABEL_BYTES: usize = 128;
const MAX_REGISTRY_WORKERS: usize = 4096;
const MAX_DEPLOYMENTS_PER_WORKER: usize = 256;
const MAX_LOCAL_DISPATCHES: usize = 65_536;
const MAX_STATUS_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_CIRCUIT_FAILURE_THRESHOLD: u32 = 1024;
const MAX_CIRCUIT_OPEN_DURATION: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRegistryConfig {
    pub max_workers: usize,
    pub max_deployments_per_worker: usize,
    pub max_local_dispatches: usize,
    pub status_ttl: Duration,
    pub circuit_failure_threshold: u32,
    pub circuit_open_duration: Duration,
    pub randomized_tie_breaking: bool,
}

impl WorkerRegistryConfig {
    pub fn validate(&self) -> Result<(), WorkerRegistryError> {
        if self.max_workers == 0 || self.max_workers > MAX_REGISTRY_WORKERS {
            return Err(WorkerRegistryError::InvalidConfig(
                "max_workers is outside the supported range",
            ));
        }
        if self.max_deployments_per_worker == 0
            || self.max_deployments_per_worker > MAX_DEPLOYMENTS_PER_WORKER
        {
            return Err(WorkerRegistryError::InvalidConfig(
                "max_deployments_per_worker is outside the supported range",
            ));
        }
        if self.max_local_dispatches == 0 || self.max_local_dispatches > MAX_LOCAL_DISPATCHES {
            return Err(WorkerRegistryError::InvalidConfig(
                "max_local_dispatches is outside the supported range",
            ));
        }
        if self.status_ttl.is_zero() || self.status_ttl > MAX_STATUS_TTL {
            return Err(WorkerRegistryError::InvalidConfig(
                "status_ttl is outside the supported range",
            ));
        }
        if self.circuit_failure_threshold == 0
            || self.circuit_failure_threshold > MAX_CIRCUIT_FAILURE_THRESHOLD
        {
            return Err(WorkerRegistryError::InvalidConfig(
                "circuit_failure_threshold is outside the supported range",
            ));
        }
        if self.circuit_open_duration.is_zero()
            || self.circuit_open_duration > MAX_CIRCUIT_OPEN_DURATION
        {
            return Err(WorkerRegistryError::InvalidConfig(
                "circuit_open_duration is outside the supported range",
            ));
        }
        Ok(())
    }
}

impl Default for WorkerRegistryConfig {
    fn default() -> Self {
        Self {
            max_workers: 256,
            max_deployments_per_worker: 32,
            max_local_dispatches: 1024,
            status_ttl: Duration::from_secs(10),
            circuit_failure_threshold: 3,
            circuit_open_duration: Duration::from_secs(30),
            randomized_tie_breaking: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct WorkerInstanceKey {
    pub worker_id: WorkerId,
    pub incarnation_id: IncarnationId,
}

impl WorkerInstanceKey {
    fn from_descriptor(descriptor: &WorkerDescriptor) -> Self {
        Self {
            worker_id: descriptor.worker_id.clone(),
            incarnation_id: descriptor.incarnation_id.clone(),
        }
    }

    fn from_status(status: &WorkerStatus) -> Self {
        Self {
            worker_id: status.worker_id.clone(),
            incarnation_id: status.incarnation_id.clone(),
        }
    }
}

/// Cluster-wide capacity claims for one worker incarnation, as observed by
/// peer gateways sharing a fleet coordination store.
///
/// Selection consults this synchronously, so implementations must serve
/// locally cached views (populated on the status-poller cadence), never
/// live I/O under the registry lock. The worker remains the atomic
/// admission arbiter; cluster claims only steer selection away from
/// workers that peers have already filled.
pub trait FleetCapacityView: Send + Sync {
    fn cluster_claims(&self, worker: &WorkerInstanceKey) -> u64;
}

/// Single-gateway behavior: only local dispatches count against capacity.
pub struct NoFleetCapacity;

impl FleetCapacityView for NoFleetCapacity {
    fn cluster_claims(&self, _worker: &WorkerInstanceKey) -> u64 {
        0
    }
}

#[derive(Debug, Clone)]
pub struct ApprovedWorker {
    pub descriptor: WorkerDescriptor,
    pub client: WorkerClient,
    pub approved_deployments: BTreeMap<DeploymentId, ApprovedDeployment>,
    /// Capacity validated from the worker's deployment/resource configuration.
    pub validated_capacity: u32,
}

/// Static deployment identity and capability approved by the control plane.
/// Readiness is deliberately excluded because it is an observed lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedDeployment {
    pub deployment_id: DeploymentId,
    pub public_model: ModelAlias,
    pub artifact_revision: ArtifactRevision,
    pub model_generation: ModelGeneration,
    pub task: TaskKind,
    pub backend: BackendKind,
    pub precision: String,
    pub execution_representation: String,
    pub tokenizer_revision: Option<ArtifactRevision>,
    pub capability: Capability,
}

impl ApprovedDeployment {
    pub fn from_loaded(deployment: &LoadedDeployment) -> Self {
        Self {
            deployment_id: deployment.deployment_id.clone(),
            public_model: deployment.public_model.clone(),
            artifact_revision: deployment.artifact_revision.clone(),
            model_generation: deployment.model_generation,
            task: deployment.task,
            backend: deployment.backend,
            precision: deployment.precision.clone(),
            execution_representation: deployment.execution_representation.clone(),
            tokenizer_revision: deployment.tokenizer_revision.clone(),
            capability: deployment.capability.clone(),
        }
    }

    fn matches(&self, deployment: &LoadedDeployment) -> bool {
        self.deployment_id == deployment.deployment_id
            && self.public_model == deployment.public_model
            && self.artifact_revision == deployment.artifact_revision
            && self.model_generation == deployment.model_generation
            && self.task == deployment.task
            && self.backend == deployment.backend
            && self.precision == deployment.precision
            && self.execution_representation == deployment.execution_representation
            && self.tokenizer_revision == deployment.tokenizer_revision
            && self.capability == deployment.capability
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPolicy {
    pub cpu: bool,
    pub metal: bool,
    pub cuda: bool,
}

impl BackendPolicy {
    pub const ANY: Self = Self {
        cpu: true,
        metal: true,
        cuda: true,
    };

    pub const CPU_ONLY: Self = Self {
        cpu: true,
        metal: false,
        cuda: false,
    };

    pub const METAL_ONLY: Self = Self {
        cpu: false,
        metal: true,
        cuda: false,
    };

    pub const CUDA_ONLY: Self = Self {
        cpu: false,
        metal: false,
        cuda: true,
    };

    pub const fn allows(self, backend: BackendKind) -> bool {
        match backend {
            BackendKind::Cpu => self.cpu,
            BackendKind::Metal => self.metal,
            BackendKind::Cuda => self.cuda,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerSelectionRequest {
    pub protocol_version: SchemaVersion,
    pub deployment_id: DeploymentId,
    pub public_model: ModelAlias,
    pub task: TaskKind,
    pub input_format: InputFormat,
    pub output_format: OutputFormat,
    pub streaming: bool,
    pub realtime: bool,
    /// `Some` requires the worker to advertise exactly this cancellation contract.
    pub cancellation: Option<CancellationBehavior>,
    pub backend_policy: BackendPolicy,
    pub input_bytes: u64,
    pub context_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentReadinessSnapshot {
    pub approved_workers: usize,
    pub fresh_running_workers: usize,
    pub ready_workers: usize,
    pub workers_with_observed_credit: usize,
}

#[derive(Debug)]
pub struct SelectedWorker {
    pub key: WorkerInstanceKey,
    pub node_id: NodeId,
    pub deployment_id: DeploymentId,
    pub model_generation: ModelGeneration,
    pub backend: BackendKind,
    pub client: WorkerClient,
    pub dispatch: LocalDispatchGuard,
    /// Observed admission credits at select time, after local and cluster
    /// claims are accounted. Fleet coordinators use this as the atomic cap
    /// for a cluster capacity claim; it is advisory, never authority.
    pub available_credits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerRegistryError {
    #[error("invalid worker registry configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("worker registry has reached its configured worker limit")]
    WorkerLimitReached,
    #[error("worker registration has too many approved deployments")]
    DeploymentLimitExceeded,
    #[error("worker registration must have non-zero validated capacity")]
    InvalidValidatedCapacity,
    #[error("worker descriptor does not advertise the required protocol")]
    IncompatibleProtocol,
    #[error("worker metadata exceeds a registry retention limit")]
    MetadataLimitExceeded,
    #[error("worker incarnation is already registered")]
    AlreadyRegistered,
    #[error("status does not identify the currently approved worker incarnation")]
    UnknownOrStaleIncarnation,
    #[error("status identity does not match its approved descriptor")]
    IdentityMismatch,
    #[error("status sequence must be non-zero and strictly increasing")]
    StaleStatusSequence,
    #[error("receiver observation time moved backwards")]
    ObservationTimeRegressed,
    #[error("worker status has too many deployments")]
    DeploymentStatusLimitExceeded,
    #[error("worker status contains a duplicate deployment")]
    DuplicateDeployment,
    #[error("worker status advertises an unapproved deployment")]
    UnapprovedDeployment,
    #[error("worker status deployment differs from its approved static identity or capability")]
    ApprovedDeploymentMismatch,
    #[error("worker status deployment backend differs from its process assignment")]
    BackendMismatch,
    #[error("worker capacity snapshot is internally inconsistent")]
    InvalidCapacity,
    #[error("worker configured capacity differs from its approved capacity")]
    CapacityMismatch,
    #[error("no fresh, ready worker satisfies the exact request requirements")]
    NoEligibleWorker,
    #[error("gateway local dispatch tracking is full")]
    LocalDispatchLimitReached,
    #[error("local dispatch reservation is no longer tracked")]
    UnknownLocalDispatch,
}

#[derive(Clone)]
pub struct WorkerRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

impl std::fmt::Debug for WorkerRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = lock_recover(&self.inner);
        formatter
            .debug_struct("WorkerRegistry")
            .field("config", &inner.config)
            .field("workers", &inner.workers.len())
            .field("local_dispatches", &inner.dispatches.len())
            .finish()
    }
}

struct RegistryInner {
    config: WorkerRegistryConfig,
    workers: BTreeMap<WorkerInstanceKey, WorkerRecord>,
    active_incarnations: BTreeMap<WorkerId, IncarnationId>,
    dispatches: BTreeMap<u64, LocalDispatchRecord>,
    next_dispatch_id: u64,
}

struct WorkerRecord {
    registration: ApprovedWorker,
    observation: Option<StatusObservation>,
    circuit: WorkerCircuitState,
}

struct StatusObservation {
    status: WorkerStatus,
    received_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerCircuitState {
    Closed { consecutive_transport_failures: u32 },
    Open { opened_at: Instant },
    HalfOpenProbe,
}

impl Default for WorkerCircuitState {
    fn default() -> Self {
        Self::Closed {
            consecutive_transport_failures: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalDispatchState {
    Dispatched,
    AcceptedAfter { status_sequence: u64 },
    Reconciled,
}

struct LocalDispatchRecord {
    worker: WorkerInstanceKey,
    state: LocalDispatchState,
}

impl WorkerRegistry {
    pub fn new(config: WorkerRegistryConfig) -> Result<Self, WorkerRegistryError> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                config,
                workers: BTreeMap::new(),
                active_incarnations: BTreeMap::new(),
                dispatches: BTreeMap::new(),
                next_dispatch_id: 1,
            })),
        })
    }

    /// Adds an explicitly approved worker incarnation. Re-registering a logical
    /// worker replaces its previous incarnation and makes all later status from
    /// the old process ineligible.
    pub fn approve(&self, registration: ApprovedWorker) -> Result<(), WorkerRegistryError> {
        let mut inner = lock_recover(&self.inner);
        validate_registration(&inner.config, &registration)?;
        let key = WorkerInstanceKey::from_descriptor(&registration.descriptor);
        if inner.workers.contains_key(&key) {
            return Err(WorkerRegistryError::AlreadyRegistered);
        }

        let previous_key =
            inner
                .active_incarnations
                .get(&key.worker_id)
                .cloned()
                .map(|incarnation_id| WorkerInstanceKey {
                    worker_id: key.worker_id.clone(),
                    incarnation_id,
                });
        let is_replacement = previous_key.is_some();
        if !is_replacement && inner.workers.len() >= inner.config.max_workers {
            return Err(WorkerRegistryError::WorkerLimitReached);
        }
        if let Some(previous_key) = previous_key {
            inner.workers.remove(&previous_key);
        }
        inner
            .active_incarnations
            .insert(key.worker_id.clone(), key.incarnation_id.clone());
        inner.workers.insert(
            key,
            WorkerRecord {
                registration,
                observation: None,
                circuit: WorkerCircuitState::default(),
            },
        );
        Ok(())
    }

    pub fn revoke(&self, key: &WorkerInstanceKey) -> bool {
        let mut inner = lock_recover(&self.inner);
        let removed = inner.workers.remove(key).is_some();
        if inner.active_incarnations.get(&key.worker_id) == Some(&key.incarnation_id) {
            inner.active_incarnations.remove(&key.worker_id);
        }
        removed
    }

    pub fn observe_status(&self, status: WorkerStatus) -> Result<(), WorkerRegistryError> {
        self.observe_status_at(status, Instant::now())
    }

    /// Records a status already authenticated by the approved private client,
    /// using only a receiver-local monotonic freshness timestamp. The explicit
    /// timestamp is exposed for deterministic boundary tests.
    pub fn observe_status_at(
        &self,
        status: WorkerStatus,
        received_at: Instant,
    ) -> Result<(), WorkerRegistryError> {
        let mut inner = lock_recover(&self.inner);
        let key = WorkerInstanceKey::from_status(&status);
        if inner.active_incarnations.get(&key.worker_id) != Some(&key.incarnation_id) {
            return Err(WorkerRegistryError::UnknownOrStaleIncarnation);
        }

        let config = inner.config.clone();
        let record = inner
            .workers
            .get_mut(&key)
            .ok_or(WorkerRegistryError::UnknownOrStaleIncarnation)?;
        validate_status(
            &config,
            &record.registration,
            record.observation.as_ref(),
            &status,
            received_at,
        )?;
        let sequence = status.status_sequence;
        record.observation = Some(StatusObservation {
            status,
            received_at,
        });

        for dispatch in inner.dispatches.values_mut() {
            if dispatch.worker != key {
                continue;
            }
            if let LocalDispatchState::AcceptedAfter { status_sequence } = dispatch.state {
                if sequence > status_sequence {
                    dispatch.state = LocalDispatchState::Reconciled;
                }
            }
        }
        Ok(())
    }

    pub fn deployment_readiness(
        &self,
        deployment_id: &DeploymentId,
    ) -> DeploymentReadinessSnapshot {
        self.deployment_readiness_at(deployment_id, Instant::now())
    }

    pub fn deployment_readiness_at(
        &self,
        deployment_id: &DeploymentId,
        now: Instant,
    ) -> DeploymentReadinessSnapshot {
        let inner = lock_recover(&self.inner);
        let mut snapshot = DeploymentReadinessSnapshot {
            approved_workers: 0,
            fresh_running_workers: 0,
            ready_workers: 0,
            workers_with_observed_credit: 0,
        };
        for (key, record) in &inner.workers {
            if !record
                .registration
                .approved_deployments
                .contains_key(deployment_id)
            {
                continue;
            }
            snapshot.approved_workers += 1;
            let Some(observation) = fresh_observation(record, now, inner.config.status_ttl) else {
                continue;
            };
            if observation.status.process_state != WorkerProcessState::Running {
                continue;
            }
            snapshot.fresh_running_workers += 1;
            let Some(deployment) = ready_deployment(&observation.status, deployment_id) else {
                continue;
            };
            snapshot.ready_workers += 1;
            if observed_credit_available(&inner, key, &observation.status.capacity, 0) {
                snapshot.workers_with_observed_credit += 1;
            }
            let _ = deployment;
        }
        snapshot
    }

    /// Reports whether a receiver-clock-fresh running worker is ready for the
    /// exact request contract. Capacity is intentionally excluded: an
    /// otherwise healthy but currently full deployment remains service-ready.
    pub fn has_fresh_compatible_worker(&self, request: &WorkerSelectionRequest) -> bool {
        let now = Instant::now();
        let inner = lock_recover(&self.inner);
        inner.workers.values().any(|record| {
            fresh_observation(record, now, inner.config.status_ttl)
                .filter(|observation| {
                    circuit_allows_selection(
                        record,
                        observation,
                        now,
                        inner.config.circuit_open_duration,
                    )
                })
                .and_then(|observation| eligible_deployment(record, observation, request))
                .is_some()
        })
    }

    pub fn select_and_reserve(
        &self,
        request: &WorkerSelectionRequest,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        self.select_and_reserve_excluding(request, None)
    }

    /// Selects a worker while excluding at most one exact worker incarnation.
    /// This supports a bounded, provably pre-acceptance alternate selection;
    /// the registry intentionally does not retain an unbounded attempted set.
    pub fn select_and_reserve_excluding(
        &self,
        request: &WorkerSelectionRequest,
        excluded: Option<&WorkerInstanceKey>,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        self.select_and_reserve_excluding_at(request, excluded, Instant::now())
    }

    /// Selects and immediately accounts for one local dispatch while holding
    /// the registry lock. Ties are stable by worker then incarnation identity.
    pub fn select_and_reserve_at(
        &self,
        request: &WorkerSelectionRequest,
        now: Instant,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        self.select_and_reserve_excluding_at(request, None, now)
    }

    /// Deterministic-time variant used by receiver-clock circuit tests.
    pub fn select_and_reserve_excluding_at(
        &self,
        request: &WorkerSelectionRequest,
        excluded: Option<&WorkerInstanceKey>,
        now: Instant,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        self.select_and_reserve_with_fleet_at(request, excluded, &NoFleetCapacity, now)
    }

    /// Fleet-aware selection: peer gateways' cluster claims count against a
    /// worker's observed credits alongside local dispatches. A worker with no
    /// observable cluster capacity left is skipped even when its last direct
    /// status still advertises credits, so two gateways racing for the last
    /// credit do not both dispatch at it. The selected worker still admits
    /// atomically; a fleet-level overestimate only costs one alternate
    /// dispatch, never duplicate execution.
    pub fn select_and_reserve_with_fleet_at(
        &self,
        request: &WorkerSelectionRequest,
        excluded: Option<&WorkerInstanceKey>,
        fleet: &dyn FleetCapacityView,
        now: Instant,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        let mut inner = lock_recover(&self.inner);
        if inner.dispatches.len() >= inner.config.max_local_dispatches {
            return Err(WorkerRegistryError::LocalDispatchLimitReached);
        }

        let mut selected: Option<(WorkerInstanceKey, LoadedDeployment, u64, u32, u32)> = None;
        for (key, record) in &inner.workers {
            if excluded == Some(key) {
                continue;
            }
            let Some(observation) = fresh_observation(record, now, inner.config.status_ttl) else {
                continue;
            };
            if !circuit_allows_selection(
                record,
                observation,
                now,
                inner.config.circuit_open_duration,
            ) {
                continue;
            }
            let Some(deployment) = eligible_deployment(record, observation, request) else {
                continue;
            };
            let cluster_claims = fleet.cluster_claims(key);
            if !observed_credit_available(&inner, key, &observation.status.capacity, cluster_claims)
            {
                continue;
            }
            let local = unreconciled_dispatches(&inner, key) as u64;
            let outstanding = u64::from(observation.status.capacity.active_invocations)
                .saturating_add(u64::from(observation.status.capacity.queued_invocations))
                .saturating_add(local)
                .saturating_add(cluster_claims);
            let capacity = record.registration.validated_capacity;
            let available = observation.status.capacity.available_admission_credits;

            let replace = selected.as_ref().is_none_or(
                |(selected_key, _, selected_outstanding, selected_capacity, _)| {
                    let candidate_score = u128::from(outstanding) * u128::from(*selected_capacity);
                    let selected_score = u128::from(*selected_outstanding) * u128::from(capacity);
                    if candidate_score < selected_score {
                        true
                    } else if candidate_score > selected_score {
                        false
                    } else if inner.config.randomized_tie_breaking {
                        uuid::Uuid::new_v4().as_bytes()[0] % 2 == 0
                    } else {
                        key < selected_key
                    }
                },
            );
            if replace {
                selected = Some((
                    key.clone(),
                    deployment.clone(),
                    outstanding,
                    capacity,
                    available,
                ));
            }
        }

        let (key, deployment, _, _, available_credits) =
            selected.ok_or(WorkerRegistryError::NoEligibleWorker)?;
        let (selected_client, node_id, circuit_probe) = {
            let record = inner
                .workers
                .get_mut(&key)
                .expect("selected worker remains registered under the registry lock");
            let circuit_probe = matches!(record.circuit, WorkerCircuitState::Open { .. });
            if circuit_probe {
                record.circuit = WorkerCircuitState::HalfOpenProbe;
            }
            (
                record.registration.client.clone(),
                record.registration.descriptor.node_id.clone(),
                circuit_probe,
            )
        };
        let backend = deployment.backend;
        let deployment_id = deployment.deployment_id.clone();
        let model_generation = deployment.model_generation;
        let dispatch_id = next_dispatch_id(&mut inner);
        inner.dispatches.insert(
            dispatch_id,
            LocalDispatchRecord {
                worker: key.clone(),
                state: LocalDispatchState::Dispatched,
            },
        );

        Ok(SelectedWorker {
            key: key.clone(),
            node_id,
            deployment_id,
            model_generation,
            backend,
            client: selected_client,
            dispatch: LocalDispatchGuard {
                registry: Arc::downgrade(&self.inner),
                dispatch_id,
                worker: key,
                circuit_probe,
            },
            available_credits,
        })
    }

    /// Records any authenticated response from this exact worker incarnation.
    /// Accepted requests and deterministic rejections both prove reachability
    /// and close the circuit.
    pub fn report_worker_reachable(
        &self,
        key: &WorkerInstanceKey,
    ) -> Result<(), WorkerRegistryError> {
        let mut inner = lock_recover(&self.inner);
        let record = active_worker_mut(&mut inner, key)?;
        record.circuit = WorkerCircuitState::default();
        Ok(())
    }

    /// Records an outcome for which the gateway cannot authenticate a worker
    /// response. Closed circuits accumulate bounded consecutive strikes;
    /// a failed half-open probe immediately reopens the circuit.
    pub fn report_worker_transport_failure(
        &self,
        key: &WorkerInstanceKey,
    ) -> Result<(), WorkerRegistryError> {
        self.report_worker_transport_failure_at(key, Instant::now())
    }

    /// Deterministic-time variant used by receiver-clock circuit tests.
    pub fn report_worker_transport_failure_at(
        &self,
        key: &WorkerInstanceKey,
        now: Instant,
    ) -> Result<(), WorkerRegistryError> {
        let mut inner = lock_recover(&self.inner);
        let failure_threshold = inner.config.circuit_failure_threshold;
        let record = active_worker_mut(&mut inner, key)?;
        record.circuit = match record.circuit {
            WorkerCircuitState::Closed {
                consecutive_transport_failures,
            } => {
                let strikes = consecutive_transport_failures.saturating_add(1);
                if strikes >= failure_threshold {
                    WorkerCircuitState::Open { opened_at: now }
                } else {
                    WorkerCircuitState::Closed {
                        consecutive_transport_failures: strikes,
                    }
                }
            }
            WorkerCircuitState::Open { .. } | WorkerCircuitState::HalfOpenProbe => {
                WorkerCircuitState::Open { opened_at: now }
            }
        };
        Ok(())
    }
}

#[derive(Debug)]
pub struct LocalDispatchGuard {
    registry: Weak<Mutex<RegistryInner>>,
    dispatch_id: u64,
    worker: WorkerInstanceKey,
    circuit_probe: bool,
}

impl LocalDispatchGuard {
    pub fn worker(&self) -> &WorkerInstanceKey {
        &self.worker
    }

    /// Call only after the private client has returned its authenticated,
    /// validated accepted event. Acceptance proves reachability and closes a
    /// half-open circuit. A subsequent status sequence can then reconcile the
    /// local capacity estimate.
    pub fn mark_accepted(&mut self) -> Result<(), WorkerRegistryError> {
        let registry = self
            .registry
            .upgrade()
            .ok_or(WorkerRegistryError::UnknownLocalDispatch)?;
        let mut inner = lock_recover(&registry);
        if !inner.dispatches.contains_key(&self.dispatch_id) {
            return Err(WorkerRegistryError::UnknownLocalDispatch);
        }
        let sequence = inner
            .workers
            .get(&self.worker)
            .and_then(|worker| worker.observation.as_ref())
            .map(|observation| observation.status.status_sequence)
            .unwrap_or(0);
        if inner.active_incarnations.get(&self.worker.worker_id)
            == Some(&self.worker.incarnation_id)
        {
            inner
                .workers
                .get_mut(&self.worker)
                .expect("active worker incarnation remains registered")
                .circuit = WorkerCircuitState::default();
        }
        let dispatch = inner
            .dispatches
            .get_mut(&self.dispatch_id)
            .expect("local dispatch was verified under the registry lock");
        if dispatch.state == LocalDispatchState::Dispatched {
            dispatch.state = LocalDispatchState::AcceptedAfter {
                status_sequence: sequence,
            };
        }
        Ok(())
    }
}

impl Drop for LocalDispatchGuard {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut inner = lock_recover(&registry);
        inner.dispatches.remove(&self.dispatch_id);
        if self.circuit_probe
            && inner.active_incarnations.get(&self.worker.worker_id)
                == Some(&self.worker.incarnation_id)
        {
            let record = inner
                .workers
                .get_mut(&self.worker)
                .expect("active worker incarnation remains registered");
            if record.circuit == WorkerCircuitState::HalfOpenProbe {
                record.circuit = WorkerCircuitState::Open {
                    opened_at: Instant::now(),
                };
            }
        }
    }
}

fn validate_registration(
    config: &WorkerRegistryConfig,
    registration: &ApprovedWorker,
) -> Result<(), WorkerRegistryError> {
    if registration.approved_deployments.len() > config.max_deployments_per_worker {
        return Err(WorkerRegistryError::DeploymentLimitExceeded);
    }
    if registration.validated_capacity == 0 {
        return Err(WorkerRegistryError::InvalidValidatedCapacity);
    }
    let descriptor = &registration.descriptor;
    if registration
        .approved_deployments
        .iter()
        .any(|(id, deployment)| {
            id != &deployment.deployment_id || deployment.backend != descriptor.assignment.backend()
        })
    {
        return Err(WorkerRegistryError::ApprovedDeploymentMismatch);
    }
    if registration
        .approved_deployments
        .values()
        .any(|deployment| {
            deployment.precision.is_empty()
                || deployment.precision.len() > MAX_EXECUTION_LABEL_BYTES
                || deployment.execution_representation.is_empty()
                || deployment.execution_representation.len() > MAX_EXECUTION_LABEL_BYTES
        })
    {
        return Err(WorkerRegistryError::MetadataLimitExceeded);
    }
    if descriptor.supported_protocol_versions.len() > MAX_PROTOCOL_VERSIONS
        || descriptor.build_version.is_empty()
        || descriptor.build_version.len() > MAX_BUILD_VERSION_BYTES
    {
        return Err(WorkerRegistryError::MetadataLimitExceeded);
    }
    if !PROTOCOL_V1.is_supported_by(descriptor.schema_version)
        || !descriptor
            .supported_protocol_versions
            .iter()
            .any(|version| PROTOCOL_V1.is_supported_by(*version))
    {
        return Err(WorkerRegistryError::IncompatibleProtocol);
    }
    Ok(())
}

fn validate_status(
    config: &WorkerRegistryConfig,
    registration: &ApprovedWorker,
    previous: Option<&StatusObservation>,
    status: &WorkerStatus,
    received_at: Instant,
) -> Result<(), WorkerRegistryError> {
    let descriptor = &registration.descriptor;
    if status.worker_id != descriptor.worker_id
        || status.node_id != descriptor.node_id
        || status.incarnation_id != descriptor.incarnation_id
    {
        return Err(WorkerRegistryError::IdentityMismatch);
    }
    if !PROTOCOL_V1.is_supported_by(status.schema_version) {
        return Err(WorkerRegistryError::IncompatibleProtocol);
    }
    if status.status_sequence == 0
        || previous
            .is_some_and(|observation| status.status_sequence <= observation.status.status_sequence)
    {
        return Err(WorkerRegistryError::StaleStatusSequence);
    }
    if previous.is_some_and(|observation| received_at < observation.received_at) {
        return Err(WorkerRegistryError::ObservationTimeRegressed);
    }
    if status.deployments.len() > config.max_deployments_per_worker {
        return Err(WorkerRegistryError::DeploymentStatusLimitExceeded);
    }

    let mut seen = BTreeSet::new();
    for deployment in &status.deployments {
        if !seen.insert(deployment.deployment_id.clone()) {
            return Err(WorkerRegistryError::DuplicateDeployment);
        }
        if deployment.backend != descriptor.assignment.backend() {
            return Err(WorkerRegistryError::BackendMismatch);
        }
        let Some(approved) = registration
            .approved_deployments
            .get(&deployment.deployment_id)
        else {
            return Err(WorkerRegistryError::UnapprovedDeployment);
        };
        if !approved.matches(deployment) {
            return Err(WorkerRegistryError::ApprovedDeploymentMismatch);
        }
        if deployment.precision.is_empty()
            || deployment.precision.len() > MAX_EXECUTION_LABEL_BYTES
            || deployment.execution_representation.is_empty()
            || deployment.execution_representation.len() > MAX_EXECUTION_LABEL_BYTES
        {
            return Err(WorkerRegistryError::MetadataLimitExceeded);
        }
    }

    let capacity = &status.capacity;
    let total_capacity =
        u64::from(capacity.max_active_invocations) + u64::from(capacity.max_queued_invocations);
    let outstanding =
        u64::from(capacity.active_invocations) + u64::from(capacity.queued_invocations);
    if capacity.active_invocations > capacity.max_active_invocations
        || capacity.queued_invocations > capacity.max_queued_invocations
        || capacity.reserved_sessions > capacity.max_sessions
        || u64::from(capacity.available_admission_credits)
            > total_capacity.saturating_sub(outstanding)
    {
        return Err(WorkerRegistryError::InvalidCapacity);
    }
    if total_capacity != u64::from(registration.validated_capacity) {
        return Err(WorkerRegistryError::CapacityMismatch);
    }
    Ok(())
}

fn active_worker_mut<'a>(
    inner: &'a mut RegistryInner,
    key: &WorkerInstanceKey,
) -> Result<&'a mut WorkerRecord, WorkerRegistryError> {
    if inner.active_incarnations.get(&key.worker_id) != Some(&key.incarnation_id) {
        return Err(WorkerRegistryError::UnknownOrStaleIncarnation);
    }
    inner
        .workers
        .get_mut(key)
        .ok_or(WorkerRegistryError::UnknownOrStaleIncarnation)
}

fn circuit_allows_selection(
    record: &WorkerRecord,
    observation: &StatusObservation,
    now: Instant,
    open_duration: Duration,
) -> bool {
    match record.circuit {
        WorkerCircuitState::Closed { .. } => true,
        WorkerCircuitState::HalfOpenProbe => false,
        WorkerCircuitState::Open { opened_at } => {
            observation.received_at > opened_at
                && now
                    .checked_duration_since(opened_at)
                    .is_some_and(|elapsed| elapsed >= open_duration)
        }
    }
}

fn fresh_observation<'a>(
    record: &'a WorkerRecord,
    now: Instant,
    ttl: Duration,
) -> Option<&'a StatusObservation> {
    let observation = record.observation.as_ref()?;
    let age = now.checked_duration_since(observation.received_at)?;
    (age < ttl).then_some(observation)
}

fn ready_deployment<'a>(
    status: &'a WorkerStatus,
    deployment_id: &DeploymentId,
) -> Option<&'a LoadedDeployment> {
    status.deployments.iter().find(|deployment| {
        deployment.deployment_id == *deployment_id && deployment.readiness == ModelReadiness::Ready
    })
}

fn eligible_deployment<'a>(
    record: &'a WorkerRecord,
    observation: &'a StatusObservation,
    request: &WorkerSelectionRequest,
) -> Option<&'a LoadedDeployment> {
    if observation.status.process_state != WorkerProcessState::Running
        || !request
            .protocol_version
            .is_supported_by(observation.status.schema_version)
        || !record
            .registration
            .descriptor
            .supported_protocol_versions
            .iter()
            .any(|version| request.protocol_version.is_supported_by(*version))
        || !record
            .registration
            .approved_deployments
            .contains_key(&request.deployment_id)
    {
        return None;
    }

    let deployment = ready_deployment(&observation.status, &request.deployment_id)?;
    let capability = &deployment.capability;
    if deployment.public_model != request.public_model
        || deployment.task != request.task
        || capability.task != request.task
        || deployment.backend != record.registration.descriptor.assignment.backend()
        || !request.backend_policy.allows(deployment.backend)
        || !capability
            .accepted_input_formats
            .contains(&request.input_format)
        || !capability.output_formats.contains(&request.output_format)
        || request.streaming
            && (!capability.streaming
                || !record
                    .registration
                    .descriptor
                    .features
                    .contains(&WorkerFeature::Streaming))
        || request.realtime && !capability.realtime
        || request.cancellation.is_some_and(|required| {
            capability.cancellation != required
                || (required != CancellationBehavior::NotSupported
                    && !record
                        .registration
                        .descriptor
                        .features
                        .contains(&WorkerFeature::Cancellation))
        })
        || request.input_bytes > capability.max_input_bytes
        || request.context_tokens.is_some_and(|requested| {
            capability
                .max_context_tokens
                .is_none_or(|maximum| requested > maximum)
        })
        || request.output_tokens.is_some_and(|requested| {
            capability
                .max_output_tokens
                .is_none_or(|maximum| requested > maximum)
        })
    {
        return None;
    }
    Some(deployment)
}

fn observed_credit_available(
    inner: &RegistryInner,
    worker: &WorkerInstanceKey,
    capacity: &izwi_serving_protocol::CapacitySnapshot,
    cluster_claims: u64,
) -> bool {
    let local = unreconciled_dispatches(inner, worker) as u64;
    capacity.available_admission_credits > 0
        && local.saturating_add(cluster_claims) < u64::from(capacity.available_admission_credits)
}

fn unreconciled_dispatches(inner: &RegistryInner, worker: &WorkerInstanceKey) -> usize {
    inner
        .dispatches
        .values()
        .filter(|dispatch| {
            dispatch.worker == *worker && dispatch.state != LocalDispatchState::Reconciled
        })
        .count()
}

fn next_dispatch_id(inner: &mut RegistryInner) -> u64 {
    loop {
        let candidate = inner.next_dispatch_id;
        inner.next_dispatch_id = inner.next_dispatch_id.wrapping_add(1).max(1);
        if !inner.dispatches.contains_key(&candidate) {
            return candidate;
        }
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_client::WorkerClientConfig;
    use izwi_serving_protocol::{
        ArtifactRevision, Capability, CapacitySnapshot, CredentialId, DeviceAssignment, DeviceId,
        ServiceBearerToken, ServiceCredentials,
    };

    fn id<T>(value: &str) -> T
    where
        T: TryFrom<String>,
        T::Error: std::fmt::Debug,
    {
        T::try_from(value.to_string()).unwrap()
    }

    fn client(port: u16) -> WorkerClient {
        WorkerClient::new(
            &format!("http://127.0.0.1:{port}"),
            ServiceCredentials {
                credential_id: id::<CredentialId>("gateway-key"),
                bearer_token: ServiceBearerToken::new("secret").unwrap(),
            },
            WorkerClientConfig::default(),
        )
        .unwrap()
    }

    fn descriptor(worker: &str, incarnation: &str, backend: BackendKind) -> WorkerDescriptor {
        let assignment = match backend {
            BackendKind::Cpu => DeviceAssignment::Cpu {
                thread_budget: 2,
                affinity: vec![],
                host_memory_limit_bytes: 1024,
            },
            BackendKind::Metal => DeviceAssignment::Metal {
                device_id: id::<DeviceId>("metal:1"),
                process_local_device_index: 0,
                shared_memory_limit_bytes: 1024,
            },
            BackendKind::Cuda => DeviceAssignment::Cuda {
                device_uuid: id::<DeviceId>("GPU-a"),
                process_local_device_index: 0,
                device_memory_limit_bytes: 1024,
                host_memory_limit_bytes: 1024,
            },
        };
        WorkerDescriptor {
            schema_version: PROTOCOL_V1,
            supported_protocol_versions: vec![PROTOCOL_V1],
            worker_id: id::<WorkerId>(worker),
            node_id: id::<NodeId>("node-a"),
            incarnation_id: id::<IncarnationId>(incarnation),
            build_version: "test".into(),
            assignment,
            features: BTreeSet::from([WorkerFeature::Streaming, WorkerFeature::Cancellation]),
        }
    }

    fn deployment(name: &str, alias: &str, backend: BackendKind) -> LoadedDeployment {
        LoadedDeployment {
            deployment_id: id::<DeploymentId>(name),
            public_model: id::<ModelAlias>(alias),
            artifact_revision: id::<ArtifactRevision>("sha256:artifact"),
            model_generation: ModelGeneration::new(1).unwrap(),
            task: TaskKind::Chat,
            backend,
            precision: "f32".into(),
            execution_representation: "gguf".into(),
            tokenizer_revision: None,
            readiness: ModelReadiness::Ready,
            capability: Capability {
                task: TaskKind::Chat,
                streaming: true,
                realtime: false,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                output_formats: BTreeSet::from([OutputFormat::Text]),
                max_input_bytes: 4096,
                max_context_tokens: Some(1024),
                max_output_tokens: Some(128),
            },
        }
    }

    fn capacity(max: u32, active: u32, credits: u32) -> CapacitySnapshot {
        CapacitySnapshot {
            max_active_invocations: max,
            active_invocations: active,
            max_queued_invocations: 0,
            queued_invocations: 0,
            max_sessions: 0,
            reserved_sessions: 0,
            available_admission_credits: credits,
            outstanding_cost_units: 0,
        }
    }

    fn status(
        descriptor: &WorkerDescriptor,
        sequence: u64,
        deployments: Vec<LoadedDeployment>,
        capacity: CapacitySnapshot,
    ) -> WorkerStatus {
        WorkerStatus {
            schema_version: PROTOCOL_V1,
            worker_id: descriptor.worker_id.clone(),
            node_id: descriptor.node_id.clone(),
            incarnation_id: descriptor.incarnation_id.clone(),
            status_sequence: sequence,
            process_state: WorkerProcessState::Running,
            deployments,
            capacity,
        }
    }

    fn registration(
        worker: &str,
        incarnation: &str,
        backend: BackendKind,
        port: u16,
        validated_capacity: u32,
    ) -> ApprovedWorker {
        ApprovedWorker {
            descriptor: descriptor(worker, incarnation, backend),
            client: client(port),
            approved_deployments: {
                let deployment = deployment("chat-prod", "lfm2", backend);
                BTreeMap::from([(
                    deployment.deployment_id.clone(),
                    ApprovedDeployment::from_loaded(&deployment),
                )])
            },
            validated_capacity,
        }
    }

    fn selection() -> WorkerSelectionRequest {
        WorkerSelectionRequest {
            protocol_version: PROTOCOL_V1,
            deployment_id: id::<DeploymentId>("chat-prod"),
            public_model: id::<ModelAlias>("lfm2"),
            task: TaskKind::Chat,
            input_format: InputFormat::ChatMessages,
            output_format: OutputFormat::Text,
            streaming: true,
            realtime: false,
            cancellation: Some(CancellationBehavior::Cooperative),
            backend_policy: BackendPolicy::ANY,
            input_bytes: 128,
            context_tokens: Some(64),
            output_tokens: Some(32),
        }
    }

    #[test]
    fn config_and_registration_limits_are_fail_closed() {
        let mut config = WorkerRegistryConfig::default();
        config.max_workers = 0;
        assert_eq!(
            WorkerRegistry::new(config).unwrap_err(),
            WorkerRegistryError::InvalidConfig("max_workers is outside the supported range")
        );

        let mut config = WorkerRegistryConfig::default();
        config.circuit_failure_threshold = 0;
        assert_eq!(
            WorkerRegistry::new(config).unwrap_err(),
            WorkerRegistryError::InvalidConfig(
                "circuit_failure_threshold is outside the supported range"
            )
        );

        let mut config = WorkerRegistryConfig::default();
        config.circuit_open_duration = MAX_CIRCUIT_OPEN_DURATION + Duration::from_secs(1);
        assert_eq!(
            WorkerRegistry::new(config).unwrap_err(),
            WorkerRegistryError::InvalidConfig(
                "circuit_open_duration is outside the supported range"
            )
        );

        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            max_workers: 1,
            max_deployments_per_worker: 1,
            max_local_dispatches: 1,
            status_ttl: Duration::from_secs(10),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        registry
            .approve(registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1))
            .unwrap();
        assert_eq!(
            registry
                .approve(registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 1))
                .unwrap_err(),
            WorkerRegistryError::WorkerLimitReached
        );
    }

    #[test]
    fn replacement_incarnation_starts_unready_and_rejects_late_status() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let first = registration("worker-a", "inc-old", BackendKind::Cpu, 9101, 1);
        let old_descriptor = first.descriptor.clone();
        registry.approve(first).unwrap();
        registry
            .observe_status(status(
                &old_descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ))
            .unwrap();

        let replacement = registration("worker-a", "inc-new", BackendKind::Cpu, 9102, 1);
        registry.approve(replacement).unwrap();
        assert_eq!(
            registry.select_and_reserve(&selection()).unwrap_err(),
            WorkerRegistryError::NoEligibleWorker
        );
        assert_eq!(
            registry
                .observe_status(status(
                    &old_descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ))
                .unwrap_err(),
            WorkerRegistryError::UnknownOrStaleIncarnation
        );
    }

    #[test]
    fn status_sequence_and_receiver_time_must_increase() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        let now = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now,
            )
            .unwrap();
        assert_eq!(
            registry
                .observe_status_at(
                    status(
                        &descriptor,
                        2,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(1, 0, 1),
                    ),
                    now + Duration::from_secs(1),
                )
                .unwrap_err(),
            WorkerRegistryError::StaleStatusSequence
        );
        assert_eq!(
            registry
                .observe_status_at(
                    status(
                        &descriptor,
                        3,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(1, 0, 1),
                    ),
                    now - Duration::from_millis(1),
                )
                .unwrap_err(),
            WorkerRegistryError::ObservationTimeRegressed
        );
    }

    #[test]
    fn freshness_expires_at_the_receiver_clock_boundary() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            status_ttl: Duration::from_secs(10),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        let observed = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                observed,
            )
            .unwrap();
        assert!(registry
            .select_and_reserve_at(&selection(), observed + Duration::from_secs(10))
            .is_err());
        let selected = registry
            .select_and_reserve_at(&selection(), observed + Duration::from_millis(9_999))
            .unwrap();
        drop(selected);
    }

    #[test]
    fn partition_expiry_blocks_new_work_until_a_newer_authenticated_status() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            status_ttl: Duration::from_secs(10),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        let observed = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 0, 2),
                ),
                observed,
            )
            .unwrap();

        let mut active = registry
            .select_and_reserve_at(&selection(), observed + Duration::from_secs(1))
            .unwrap();
        active.dispatch.mark_accepted().unwrap();
        assert_eq!(
            registry
                .select_and_reserve_at(&selection(), observed + Duration::from_secs(10))
                .unwrap_err(),
            WorkerRegistryError::NoEligibleWorker,
            "status expiry must stop only new selection"
        );
        {
            let inner = lock_recover(&registry.inner);
            assert_eq!(inner.dispatches.len(), 1, "active ownership is retained");
        }

        registry
            .observe_status_at(
                status(
                    &descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 1, 1),
                ),
                observed + Duration::from_secs(11),
            )
            .unwrap();
        let after_reconnect = registry
            .select_and_reserve_at(&selection(), observed + Duration::from_secs(11))
            .unwrap();
        assert_eq!(after_reconnect.key.incarnation_id.as_str(), "inc-a");
        drop(after_reconnect);
        drop(active);
    }

    #[test]
    fn readiness_is_per_deployment_and_separate_from_saturation() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        let mut loading = deployment("chat-prod", "lfm2", BackendKind::Cpu);
        loading.readiness = ModelReadiness::Loading;
        let now = Instant::now();
        registry
            .observe_status_at(
                status(&descriptor, 1, vec![loading], capacity(1, 1, 0)),
                now,
            )
            .unwrap();
        let snapshot = registry.deployment_readiness_at(&id("chat-prod"), now);
        assert_eq!(snapshot.approved_workers, 1);
        assert_eq!(snapshot.fresh_running_workers, 1);
        assert_eq!(snapshot.ready_workers, 0);
        assert_eq!(snapshot.workers_with_observed_credit, 0);
    }

    #[test]
    fn status_rejects_unapproved_duplicate_backend_and_capacity_claims() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();

        assert_eq!(
            registry
                .observe_status(status(
                    &descriptor,
                    1,
                    vec![deployment("other", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ))
                .unwrap_err(),
            WorkerRegistryError::UnapprovedDeployment
        );
        let duplicate = deployment("chat-prod", "lfm2", BackendKind::Cpu);
        assert_eq!(
            registry
                .observe_status(status(
                    &descriptor,
                    1,
                    vec![duplicate.clone(), duplicate],
                    capacity(1, 0, 1),
                ))
                .unwrap_err(),
            WorkerRegistryError::DuplicateDeployment
        );
        assert_eq!(
            registry
                .observe_status(status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cuda)],
                    capacity(1, 0, 1),
                ))
                .unwrap_err(),
            WorkerRegistryError::BackendMismatch
        );
        let mut wrong_generation = deployment("chat-prod", "lfm2", BackendKind::Cpu);
        wrong_generation.model_generation = ModelGeneration::new(2).unwrap();
        assert_eq!(
            registry
                .observe_status(status(
                    &descriptor,
                    1,
                    vec![wrong_generation],
                    capacity(1, 0, 1),
                ))
                .unwrap_err(),
            WorkerRegistryError::ApprovedDeploymentMismatch
        );
        assert_eq!(
            registry
                .observe_status(status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 0, 2),
                ))
                .unwrap_err(),
            WorkerRegistryError::CapacityMismatch
        );
    }

    #[test]
    fn exact_capability_and_backend_filters_fail_closed() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ))
            .unwrap();

        let mutations: Vec<Box<dyn Fn(&mut WorkerSelectionRequest)>> = vec![
            Box::new(|request| request.public_model = id("other-model")),
            Box::new(|request| request.task = TaskKind::TextToSpeech),
            Box::new(|request| request.input_format = InputFormat::EncodedAudio),
            Box::new(|request| request.output_format = OutputFormat::Json),
            Box::new(|request| request.realtime = true),
            Box::new(|request| request.cancellation = Some(CancellationBehavior::TeardownRequired)),
            Box::new(|request| request.backend_policy = BackendPolicy::CUDA_ONLY),
            Box::new(|request| request.input_bytes = 4097),
            Box::new(|request| request.context_tokens = Some(1025)),
            Box::new(|request| request.output_tokens = Some(129)),
            Box::new(|request| request.protocol_version = SchemaVersion::new(2, 0)),
        ];
        for mutate in mutations {
            let mut request = selection();
            mutate(&mut request);
            assert_eq!(
                registry.select_and_reserve(&request).unwrap_err(),
                WorkerRegistryError::NoEligibleWorker
            );
        }
    }

    #[test]
    fn mixed_backends_route_by_declared_policy_without_a_fixed_preference() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let workers = [
            ("worker-a-metal", "inc-metal", BackendKind::Metal, 9101),
            ("worker-b-cuda", "inc-cuda", BackendKind::Cuda, 9102),
            ("worker-c-cpu", "inc-cpu", BackendKind::Cpu, 9103),
        ];
        for (worker, incarnation, backend, port) in workers {
            let approved = registration(worker, incarnation, backend, port, 1);
            let descriptor = approved.descriptor.clone();
            registry.approve(approved).unwrap();
            registry
                .observe_status(status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", backend)],
                    capacity(1, 0, 1),
                ))
                .unwrap();
        }

        // ANY uses the normal load/stable-identity ordering. It does not
        // silently prefer CUDA, Metal, or CPU.
        let selected = registry.select_and_reserve(&selection()).unwrap();
        assert_eq!(selected.backend, BackendKind::Metal);
        drop(selected);

        for (policy, expected) in [
            (BackendPolicy::CPU_ONLY, BackendKind::Cpu),
            (BackendPolicy::METAL_ONLY, BackendKind::Metal),
            (BackendPolicy::CUDA_ONLY, BackendKind::Cuda),
        ] {
            let mut request = selection();
            request.backend_policy = policy;
            let selected = registry.select_and_reserve(&request).unwrap();
            assert_eq!(selected.backend, expected);
            drop(selected);
        }
    }

    #[test]
    fn selection_uses_capacity_weighted_score_then_stable_identity() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 4);
        let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 2);
        let descriptor_a = worker_a.descriptor.clone();
        let descriptor_b = worker_b.descriptor.clone();
        registry.approve(worker_b).unwrap();
        registry.approve(worker_a).unwrap();
        let now = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor_a,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(4, 2, 2),
                ),
                now,
            )
            .unwrap();
        registry
            .observe_status_at(
                status(
                    &descriptor_b,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 1, 1),
                ),
                now,
            )
            .unwrap();
        let selected = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(selected.key.worker_id.as_str(), "worker-a");
    }

    #[test]
    fn randomized_tie_breaking_distributes_choices_across_equal_workers() {
        let config = WorkerRegistryConfig {
            randomized_tie_breaking: true,
            ..WorkerRegistryConfig::default()
        };
        let registry = WorkerRegistry::new(config).unwrap();
        let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 10);
        let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 10);
        let descriptor_a = worker_a.descriptor.clone();
        let descriptor_b = worker_b.descriptor.clone();
        registry.approve(worker_b).unwrap();
        registry.approve(worker_a).unwrap();
        let now = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor_a,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(10, 0, 10),
                ),
                now,
            )
            .unwrap();
        registry
            .observe_status_at(
                status(
                    &descriptor_b,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(10, 0, 10),
                ),
                now,
            )
            .unwrap();

        let mut a_selected = 0;
        let mut b_selected = 0;
        for _ in 0..100 {
            let selected = registry.select_and_reserve_at(&selection(), now).unwrap();
            if selected.key.worker_id.as_str() == "worker-a" {
                a_selected += 1;
            } else if selected.key.worker_id.as_str() == "worker-b" {
                b_selected += 1;
            }
            drop(selected);
        }
        assert!(a_selected > 0, "worker-a was never selected");
        assert!(b_selected > 0, "worker-b was never selected");
    }

    struct StubFleetCapacity {
        claims: BTreeMap<(String, String), u64>,
    }

    impl FleetCapacityView for StubFleetCapacity {
        fn cluster_claims(&self, worker: &WorkerInstanceKey) -> u64 {
            self.claims
                .get(&(
                    worker.worker_id.as_str().to_string(),
                    worker.incarnation_id.as_str().to_string(),
                ))
                .copied()
                .unwrap_or(0)
        }
    }

    #[test]
    fn fleet_cluster_claims_steer_selection_away_from_peer_filled_workers() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 4);
        let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 4);
        let descriptor_a = worker_a.descriptor.clone();
        let descriptor_b = worker_b.descriptor.clone();
        registry.approve(worker_a).unwrap();
        registry.approve(worker_b).unwrap();
        let now = Instant::now();
        for descriptor in [&descriptor_a, &descriptor_b] {
            registry
                .observe_status_at(
                    status(
                        descriptor,
                        1,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(4, 0, 2),
                    ),
                    now,
                )
                .unwrap();
        }
        let peers_filled_a = StubFleetCapacity {
            claims: BTreeMap::from([(("worker-a".to_string(), "inc-a".to_string()), 2)]),
        };
        let selected = registry
            .select_and_reserve_with_fleet_at(&selection(), None, &peers_filled_a, now)
            .unwrap();
        assert_eq!(
            selected.key.worker_id.as_str(),
            "worker-b",
            "peer claims consume worker-a's observed credits"
        );
        drop(selected);
        let peers_filled_both = StubFleetCapacity {
            claims: BTreeMap::from([
                (("worker-a".to_string(), "inc-a".to_string()), 2),
                (("worker-b".to_string(), "inc-b".to_string()), 5),
            ]),
        };
        assert!(
            matches!(
                registry.select_and_reserve_with_fleet_at(
                    &selection(),
                    None,
                    &peers_filled_both,
                    now
                ),
                Err(WorkerRegistryError::NoEligibleWorker)
            ),
            "no worker is eligible when cluster claims exhaust all observed credits"
        );
    }

    #[tokio::test]
    async fn circuit_opens_on_partition_and_recovers_through_half_open_probe() {
        // End-to-end partition behavior with real time: repeated transport
        // failures (a severed gateway/worker link) trip the breaker and stop
        // selection; cooldown alone does not restore eligibility; a newer
        // authenticated status admits exactly one half-open probe, whose
        // acceptance closes the circuit and resumes normal selection.
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            circuit_failure_threshold: 3,
            circuit_open_duration: Duration::from_millis(50),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let worker = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
        let descriptor = worker.descriptor.clone();
        registry.approve(worker).unwrap();
        let observe = |registry: &WorkerRegistry, sequence: u64| {
            registry
                .observe_status_at(
                    status(
                        &descriptor,
                        sequence,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(2, 0, 2),
                    ),
                    Instant::now(),
                )
                .unwrap()
        };
        observe(&registry, 1);
        let selected = registry
            .select_and_reserve_at(&selection(), Instant::now())
            .unwrap();
        let key = selected.key.clone();
        drop(selected);

        for _ in 0..3 {
            registry.report_worker_transport_failure(&key).unwrap();
        }
        assert!(
            matches!(
                registry.select_and_reserve_at(&selection(), Instant::now()),
                Err(WorkerRegistryError::NoEligibleWorker)
            ),
            "an open circuit must stop selection even with a fresh observation"
        );
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(
            matches!(
                registry.select_and_reserve_at(&selection(), Instant::now()),
                Err(WorkerRegistryError::NoEligibleWorker)
            ),
            "cooldown alone must not restore eligibility without a newer status"
        );

        observe(&registry, 2);
        let mut probe = registry
            .select_and_reserve_at(&selection(), Instant::now())
            .expect("a newer status must admit exactly one half-open probe");
        probe.dispatch.mark_accepted().unwrap();
        drop(probe);
        registry
            .select_and_reserve_at(&selection(), Instant::now())
            .expect("acceptance must close the circuit and resume selection");
    }

    #[test]
    fn local_dispatches_are_bounded_and_affect_selection() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            max_local_dispatches: 2,
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
        let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 2);
        let descriptor_a = worker_a.descriptor.clone();
        let descriptor_b = worker_b.descriptor.clone();
        registry.approve(worker_a).unwrap();
        registry.approve(worker_b).unwrap();
        let now = Instant::now();
        for descriptor in [&descriptor_a, &descriptor_b] {
            registry
                .observe_status_at(
                    status(
                        descriptor,
                        1,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(2, 0, 2),
                    ),
                    now,
                )
                .unwrap();
        }

        let first = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(first.key.worker_id.as_str(), "worker-a");
        let second = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(second.key.worker_id.as_str(), "worker-b");
        assert_eq!(
            registry
                .select_and_reserve_at(&selection(), now)
                .unwrap_err(),
            WorkerRegistryError::LocalDispatchLimitReached
        );
        drop(first);
        let replacement = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(replacement.key.worker_id.as_str(), "worker-a");
    }

    #[test]
    fn accepted_dispatch_reconciles_only_after_a_newer_status() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
        let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 2);
        let descriptor_a = worker_a.descriptor.clone();
        let descriptor_b = worker_b.descriptor.clone();
        registry.approve(worker_a).unwrap();
        registry.approve(worker_b).unwrap();
        let now = Instant::now();
        for descriptor in [&descriptor_a, &descriptor_b] {
            registry
                .observe_status_at(
                    status(
                        descriptor,
                        1,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(2, 0, 2),
                    ),
                    now,
                )
                .unwrap();
        }

        let mut first = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(first.key.worker_id.as_str(), "worker-a");
        first.dispatch.mark_accepted().unwrap();
        {
            let inner = lock_recover(&registry.inner);
            assert_eq!(unreconciled_dispatches(&inner, &first.key), 1);
        }
        let before_update = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(before_update.key.worker_id.as_str(), "worker-b");
        drop(before_update);

        registry
            .observe_status_at(
                status(
                    &descriptor_a,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 1, 1),
                ),
                now + Duration::from_millis(1),
            )
            .unwrap();
        {
            let inner = lock_recover(&registry.inner);
            assert_eq!(unreconciled_dispatches(&inner, &first.key), 0);
        }
        let reconciled = registry
            .select_and_reserve_at(&selection(), now + Duration::from_millis(1))
            .unwrap();
        assert_eq!(reconciled.key.worker_id.as_str(), "worker-b");
    }

    #[test]
    fn exact_worker_exclusion_selects_one_bounded_alternate() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 1);
        let descriptor_a = worker_a.descriptor.clone();
        let descriptor_b = worker_b.descriptor.clone();
        registry.approve(worker_a).unwrap();
        registry.approve(worker_b).unwrap();
        let now = Instant::now();
        for descriptor in [&descriptor_a, &descriptor_b] {
            registry
                .observe_status_at(
                    status(
                        descriptor,
                        1,
                        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                        capacity(1, 0, 1),
                    ),
                    now,
                )
                .unwrap();
        }

        let first = registry.select_and_reserve_at(&selection(), now).unwrap();
        assert_eq!(first.key.worker_id.as_str(), "worker-a");
        let excluded = first.key.clone();
        drop(first);

        let alternate = registry
            .select_and_reserve_excluding_at(&selection(), Some(&excluded), now)
            .unwrap();
        assert_eq!(alternate.key.worker_id.as_str(), "worker-b");
    }

    #[test]
    fn circuit_requires_cooldown_and_post_open_status_then_grants_one_probe() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            status_ttl: Duration::from_secs(30),
            circuit_failure_threshold: 2,
            circuit_open_duration: Duration::from_secs(3),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        let key = WorkerInstanceKey::from_descriptor(&descriptor);
        registry.approve(approved).unwrap();
        let now = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now,
            )
            .unwrap();

        registry
            .report_worker_transport_failure_at(&key, now + Duration::from_secs(1))
            .unwrap();
        let after_one_strike = registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(1))
            .unwrap();
        drop(after_one_strike);
        registry
            .report_worker_transport_failure_at(&key, now + Duration::from_secs(2))
            .unwrap();

        assert_eq!(
            registry
                .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
                .unwrap_err(),
            WorkerRegistryError::NoEligibleWorker
        );
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now + Duration::from_secs(3),
            )
            .unwrap();
        assert_eq!(
            registry
                .select_and_reserve_at(&selection(), now + Duration::from_secs(4))
                .unwrap_err(),
            WorkerRegistryError::NoEligibleWorker
        );

        let mut probe = registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
            .unwrap();
        assert_eq!(
            registry
                .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
                .unwrap_err(),
            WorkerRegistryError::NoEligibleWorker
        );
        probe.dispatch.mark_accepted().unwrap();
        drop(probe);
        let closed = registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
            .unwrap();
        drop(closed);
    }

    #[test]
    fn stale_circuit_outcomes_cannot_affect_replacement_incarnation() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            circuit_failure_threshold: 1,
            circuit_open_duration: Duration::from_secs(1),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let old = registration("worker-a", "inc-old", BackendKind::Cpu, 9101, 1);
        let old_key = WorkerInstanceKey::from_descriptor(&old.descriptor);
        registry.approve(old).unwrap();
        registry.report_worker_transport_failure(&old_key).unwrap();

        let replacement = registration("worker-a", "inc-new", BackendKind::Cpu, 9102, 1);
        let replacement_descriptor = replacement.descriptor.clone();
        registry.approve(replacement).unwrap();
        registry
            .observe_status(status(
                &replacement_descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ))
            .unwrap();

        assert_eq!(
            registry.report_worker_reachable(&old_key).unwrap_err(),
            WorkerRegistryError::UnknownOrStaleIncarnation
        );
        assert_eq!(
            registry
                .report_worker_transport_failure(&old_key)
                .unwrap_err(),
            WorkerRegistryError::UnknownOrStaleIncarnation
        );
        let selected = registry.select_and_reserve(&selection()).unwrap();
        assert_eq!(selected.key.incarnation_id.as_str(), "inc-new");
    }

    #[test]
    fn authenticated_reachable_outcome_resets_consecutive_strikes() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            circuit_failure_threshold: 2,
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        let key = WorkerInstanceKey::from_descriptor(&descriptor);
        registry.approve(approved).unwrap();
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ))
            .unwrap();

        registry.report_worker_transport_failure(&key).unwrap();
        registry.report_worker_reachable(&key).unwrap();
        registry.report_worker_transport_failure(&key).unwrap();

        let selected = registry.select_and_reserve(&selection()).unwrap();
        assert_eq!(selected.key, key);
    }

    #[test]
    fn abandoned_half_open_probe_reopens_the_circuit() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig {
            circuit_failure_threshold: 1,
            circuit_open_duration: Duration::from_secs(1),
            ..WorkerRegistryConfig::default()
        })
        .unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        let key = WorkerInstanceKey::from_descriptor(&descriptor);
        registry.approve(approved).unwrap();
        let now = Instant::now();
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now,
            )
            .unwrap();
        registry
            .report_worker_transport_failure_at(&key, now + Duration::from_secs(1))
            .unwrap();
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now + Duration::from_secs(2),
            )
            .unwrap();

        let probe = registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(3))
            .unwrap();
        drop(probe);

        let inner = lock_recover(&registry.inner);
        assert!(matches!(
            inner.workers.get(&key).expect("registered worker").circuit,
            WorkerCircuitState::Open { .. }
        ));
    }

    #[test]
    fn draining_saturated_or_future_observations_are_ineligible() {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        let now = Instant::now();
        let mut draining = status(
            &descriptor,
            1,
            vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
            capacity(1, 0, 1),
        );
        draining.process_state = WorkerProcessState::Draining;
        registry.observe_status_at(draining, now).unwrap();
        assert!(registry.select_and_reserve_at(&selection(), now).is_err());

        registry
            .observe_status_at(
                status(
                    &descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 1, 0),
                ),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert!(registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(1))
            .is_err());
        assert!(registry.select_and_reserve_at(&selection(), now).is_err());
    }
}
