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
const MAX_PREFIX_BALANCE_THRESHOLD: u32 = 65_536;

/// Load-aware routing policy shared by every selection.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingConfig {
    /// How many more outstanding requests than the least-loaded worker a
    /// worker holding the request's longest prefix may carry and still win.
    /// Beyond it, load decides.
    pub prefix_balance_threshold: u32,
    /// Managed-KV utilization percent at or above which a worker is under
    /// pressure: it is chosen only when every eligible worker is.
    pub kv_pressure_pct: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            prefix_balance_threshold: 4,
            kv_pressure_pct: 90.0,
        }
    }
}

impl RoutingConfig {
    pub fn validate(&self) -> Result<(), WorkerRegistryError> {
        if self.kv_pressure_pct.is_nan()
            || self.kv_pressure_pct <= 0.0
            || self.kv_pressure_pct > 100.0
        {
            return Err(WorkerRegistryError::InvalidConfig(
                "kv_pressure_pct is outside the supported range",
            ));
        }
        if self.prefix_balance_threshold > MAX_PREFIX_BALANCE_THRESHOLD {
            return Err(WorkerRegistryError::InvalidConfig(
                "prefix_balance_threshold is outside the supported range",
            ));
        }
        Ok(())
    }
}

/// Request-specific cache affinity consulted during selection: how many
/// leading prompt blocks a worker is believed to hold.
pub trait PrefixAffinity {
    fn matched_blocks(&self, worker: &WorkerInstanceKey) -> usize;
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkerRegistryConfig {
    pub max_workers: usize,
    pub max_deployments_per_worker: usize,
    pub max_local_dispatches: usize,
    pub status_ttl: Duration,
    pub circuit_failure_threshold: u32,
    pub circuit_open_duration: Duration,
    pub randomized_tie_breaking: bool,
    pub routing: RoutingConfig,
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
        self.routing.validate()?;
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
            routing: RoutingConfig::default(),
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
            BackendKind::Unknown => false,
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
    /// Features the worker must advertise, one per request control that
    /// needs one (minor 4).
    pub required_features: BTreeSet<WorkerFeature>,
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
    /// The worker's advertised features and the protocol version
    /// negotiated with it.
    pub features: BTreeSet<WorkerFeature>,
    pub protocol_version: SchemaVersion,
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

/// Optional selection gate consulted for every candidate: when it reports
/// the rollout-eligible generation for a (task, model) pool, only workers of
/// that generation are selectable. `None` (or an absent gate) keeps every
/// approved generation selectable — the single-generation behavior.
pub type EligibleGenerationGate =
    Arc<dyn Fn(TaskKind, &ModelAlias) -> Option<ModelGeneration> + Send + Sync>;

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
    /// Per-worker index of `dispatches`, so selection and status
    /// reconciliation touch only one worker's dispatches.
    worker_dispatches: BTreeMap<WorkerInstanceKey, WorkerDispatches>,
    next_dispatch_id: u64,
    generation_gate: Option<EligibleGenerationGate>,
}

#[derive(Default)]
struct WorkerDispatches {
    ids: BTreeSet<u64>,
    /// Dispatches not yet `Reconciled` by a newer status.
    unreconciled: usize,
}

impl RegistryInner {
    fn insert_dispatch(&mut self, dispatch_id: u64, worker: WorkerInstanceKey) {
        let index = self.worker_dispatches.entry(worker.clone()).or_default();
        index.ids.insert(dispatch_id);
        index.unreconciled += 1;
        self.dispatches.insert(
            dispatch_id,
            LocalDispatchRecord {
                worker,
                state: LocalDispatchState::Dispatched,
            },
        );
    }

    fn remove_dispatch(&mut self, dispatch_id: u64) {
        let Some(record) = self.dispatches.remove(&dispatch_id) else {
            return;
        };
        let Some(index) = self.worker_dispatches.get_mut(&record.worker) else {
            return;
        };
        index.ids.remove(&dispatch_id);
        if record.state != LocalDispatchState::Reconciled {
            index.unreconciled -= 1;
        }
        if index.ids.is_empty() {
            self.worker_dispatches.remove(&record.worker);
        }
    }

    /// Marks this worker's accepted dispatches reconciled once a status newer
    /// than their acceptance arrives.
    fn reconcile_dispatches(&mut self, worker: &WorkerInstanceKey, sequence: u64) {
        let Some(index) = self.worker_dispatches.get_mut(worker) else {
            return;
        };
        for dispatch_id in &index.ids {
            let dispatch = self
                .dispatches
                .get_mut(dispatch_id)
                .expect("indexed dispatch is tracked");
            if let LocalDispatchState::AcceptedAfter { status_sequence } = dispatch.state {
                if sequence > status_sequence {
                    dispatch.state = LocalDispatchState::Reconciled;
                    index.unreconciled -= 1;
                }
            }
        }
    }
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
                worker_dispatches: BTreeMap::new(),
                next_dispatch_id: 1,
                generation_gate: None,
            })),
        })
    }

    /// Installs the rollout-eligible-generation gate (DS6). Selection skips
    /// workers whose deployment generation is not currently eligible for new
    /// admissions; the deployment table is the authority.
    pub fn set_generation_gate(&self, gate: EligibleGenerationGate) {
        let mut inner = lock_recover(&self.inner);
        inner.generation_gate = Some(gate);
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

        inner.reconcile_dispatches(&key, sequence);
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
                .is_some_and(|deployment| {
                    inner.generation_gate.as_ref().is_none_or(|gate| {
                        gate(request.task, &request.public_model)
                            .is_none_or(|eligible| deployment.model_generation == eligible)
                    })
                })
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

    /// Single-gateway selection steered by request prefix affinity. See
    /// `select_and_reserve_routed` for the contract.
    pub fn select_and_reserve_with_affinity(
        &self,
        request: &WorkerSelectionRequest,
        affinity: &dyn PrefixAffinity,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        self.select_and_reserve_routed(
            request,
            None,
            Some(affinity),
            &NoFleetCapacity,
            Instant::now(),
        )
    }
}

/// One eligible worker considered by selection, carrying the exact scoring
/// inputs the comparisons consume.
#[derive(Debug, Clone)]
struct Candidate {
    key: WorkerInstanceKey,
    deployment: LoadedDeployment,
    outstanding: u64,
    capacity: u32,
    available: u32,
    /// Leading prompt blocks this worker probably holds (0 without affinity).
    matched_blocks: usize,
}

impl Candidate {
    /// Cross-multiplied capacity-weighted load. Equal load prefers the worker
    /// that observed the higher output rate, then the stable (or randomized)
    /// identity tie-break.
    fn beats(&self, incumbent: &Candidate, randomized: bool) -> bool {
        let candidate_score = u128::from(self.outstanding) * u128::from(incumbent.capacity);
        let incumbent_score = u128::from(incumbent.outstanding) * u128::from(self.capacity);
        if candidate_score != incumbent_score {
            return candidate_score < incumbent_score;
        }
        let rate = |candidate: &Candidate| {
            candidate
                .deployment
                .tokens_out_per_s_ema
                .filter(|rate| rate.is_finite() && *rate > 0.0)
                .unwrap_or(0.0)
        };
        let (own_rate, incumbent_rate) = (rate(self), rate(incumbent));
        if own_rate != incumbent_rate {
            return own_rate > incumbent_rate;
        }
        if randomized {
            uuid::Uuid::new_v4().as_bytes()[0].is_multiple_of(2)
        } else {
            self.key < incumbent.key
        }
    }

    /// The longer matched prefix wins; equal matches compare on load.
    fn beats_on_prefix(&self, incumbent: &Candidate, randomized: bool) -> bool {
        if self.matched_blocks != incumbent.matched_blocks {
            return self.matched_blocks > incumbent.matched_blocks;
        }
        self.beats(incumbent, randomized)
    }
}

fn consider_candidate(track: &mut Option<Candidate>, candidate: Candidate, randomized: bool) {
    match track {
        Some(incumbent) if !candidate.beats(incumbent, randomized) => {}
        _ => *track = Some(candidate),
    }
}

/// A worker whose managed-KV arena is at or above the pressure threshold.
/// Workers that report no KV signal are never under pressure.
fn kv_pressured(config: &RoutingConfig, deployment: &LoadedDeployment) -> bool {
    deployment
        .kv_cache_usage_pct
        .is_some_and(|usage| usage >= config.kv_pressure_pct)
}

impl WorkerRegistry {
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
        self.select_and_reserve_routed(request, excluded, None, fleet, now)
    }

    /// Selection with every routing input. Among eligible workers:
    ///
    /// 1. Workers under KV pressure are chosen only when every eligible
    ///    worker is.
    /// 2. With `affinity`, the worker holding the request's longest prompt
    ///    prefix wins, unless it carries more than
    ///    `prefix_balance_threshold` outstanding requests beyond the
    ///    least-loaded worker; then load decides.
    /// 3. Otherwise the capacity-weighted least-loaded worker wins.
    ///
    /// Affinity is advisory: a lost or wrong match costs one prefill, never
    /// a failed request.
    pub fn select_and_reserve_routed(
        &self,
        request: &WorkerSelectionRequest,
        excluded: Option<&WorkerInstanceKey>,
        affinity: Option<&dyn PrefixAffinity>,
        fleet: &dyn FleetCapacityView,
        now: Instant,
    ) -> Result<SelectedWorker, WorkerRegistryError> {
        let mut inner = lock_recover(&self.inner);
        if inner.dispatches.len() >= inner.config.max_local_dispatches {
            return Err(WorkerRegistryError::LocalDispatchLimitReached);
        }
        let randomized = inner.config.randomized_tie_breaking;

        let mut least: Option<Candidate> = None;
        let mut least_pressured: Option<Candidate> = None;
        let mut longest_prefix: Option<Candidate> = None;
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
            if let Some(gate) = &inner.generation_gate {
                if let Some(eligible) = gate(request.task, &request.public_model) {
                    if deployment.model_generation != eligible {
                        continue;
                    }
                }
            }
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

            let mut candidate = Candidate {
                key: key.clone(),
                deployment: deployment.clone(),
                outstanding,
                capacity: record.registration.validated_capacity,
                available: observation.status.capacity.available_admission_credits,
                matched_blocks: 0,
            };
            if kv_pressured(&inner.config.routing, deployment) {
                consider_candidate(&mut least_pressured, candidate, randomized);
                continue;
            }
            candidate.matched_blocks = affinity.map_or(0, |affinity| affinity.matched_blocks(key));
            if candidate.matched_blocks > 0 {
                match &longest_prefix {
                    Some(incumbent) if !candidate.beats_on_prefix(incumbent, randomized) => {}
                    _ => longest_prefix = Some(candidate.clone()),
                }
            }
            consider_candidate(&mut least, candidate, randomized);
        }

        let least = least
            .or(least_pressured)
            .ok_or(WorkerRegistryError::NoEligibleWorker)?;
        let balance = u64::from(inner.config.routing.prefix_balance_threshold);
        let chosen = match longest_prefix {
            Some(prefix) if prefix.outstanding <= least.outstanding.saturating_add(balance) => {
                prefix
            }
            _ => least,
        };
        let Candidate {
            key,
            deployment,
            available: available_credits,
            ..
        } = chosen;
        let (selected_client, node_id, circuit_probe, features, protocol_version) = {
            let record = inner
                .workers
                .get_mut(&key)
                .expect("selected worker remains registered under the registry lock");
            let descriptor = &record.registration.descriptor;
            // Eligibility already required a shared major.
            let protocol_version = request
                .protocol_version
                .negotiate(&descriptor.supported_protocol_versions)
                .ok_or(WorkerRegistryError::IncompatibleProtocol)?;
            let features = descriptor.features.clone();
            let node_id = descriptor.node_id.clone();
            let circuit_probe = matches!(record.circuit, WorkerCircuitState::Open { .. });
            if circuit_probe {
                record.circuit = WorkerCircuitState::HalfOpenProbe;
            }
            (
                record.registration.client.clone(),
                node_id,
                circuit_probe,
                features,
                protocol_version,
            )
        };
        let backend = deployment.backend;
        let deployment_id = deployment.deployment_id.clone();
        let model_generation = deployment.model_generation;
        let dispatch_id = next_dispatch_id(&mut inner);
        inner.insert_dispatch(dispatch_id, key.clone());

        Ok(SelectedWorker {
            key: key.clone(),
            node_id,
            deployment_id,
            model_generation,
            backend,
            client: selected_client,
            features,
            protocol_version,
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
        inner.remove_dispatch(self.dispatch_id);
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
    // Additive-minor tolerance: any worker speaking major version 1 can
    // register, whatever minor it implements. Optional routing-signal fields
    // decode as absent on older workers and are ignored where unsupported.
    if !PROTOCOL_V1.shares_major_with(descriptor.schema_version)
        || !descriptor
            .supported_protocol_versions
            .iter()
            .any(|version| PROTOCOL_V1.shares_major_with(*version))
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
    if !PROTOCOL_V1.shares_major_with(status.schema_version) {
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

fn fresh_observation(
    record: &WorkerRecord,
    now: Instant,
    ttl: Duration,
) -> Option<&StatusObservation> {
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
            .shares_major_with(observation.status.schema_version)
        || !record
            .registration
            .descriptor
            .supported_protocol_versions
            .iter()
            .any(|version| request.protocol_version.shares_major_with(*version))
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
        || !request
            .required_features
            .is_subset(&record.registration.descriptor.features)
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
        .worker_dispatches
        .get(worker)
        .map_or(0, |index| index.unreconciled)
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
#[path = "worker_registry_tests.rs"]
mod tests;
