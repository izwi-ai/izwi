//! Global inference admission and device-execution coordination.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use candle_core::DeviceLocation;
use tokio::sync::Notify;
use tokio::sync::{OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

use serde::Serialize;

use crate::backends::{BackendKind, DeviceKind, DeviceProfile};
use crate::engine::{
    BatchId, BatchWorkspaceLease, CapacitySource, ExecutionDomain, ExecutionGroupId,
    NativeBatchMode, PhysicalCapacityProvider, PhysicalCapacitySnapshot, Priority,
    ReservationClass, ReservationOwner, ResourceAmount, ResourceAuthority, ResourceEstimate,
    ResourceLease, ResourceVector, WorkUnit, WorkloadClass,
};
use crate::error::{Error, Result};
use crate::runtime::adapters::{LoadedCapabilityBinding, LoadedExecutionContract};

static NEXT_EXECUTION_GROUP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordinatorLane {
    Realtime,
    Resumable,
    Atomic,
    Pipeline,
}

#[derive(Debug, Clone)]
pub struct JobSpec {
    pub request_id: String,
    pub lane: CoordinatorLane,
    pub priority: Priority,
    pub workload_class: WorkloadClass,
    pub deadline: Option<Instant>,
    pub resources: ResourceEstimate,
}

/// Physical memory currently owned by an admitted direct job. Host and
/// accelerator bytes are mapped to the backend's actual memory domains before
/// reconciliation, while the original reservation remains immutable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JobResourceObservation {
    pub host_bytes: u64,
    pub accelerator_bytes: u64,
}

impl JobResourceObservation {
    pub const fn new(host_bytes: u64, accelerator_bytes: u64) -> Self {
        Self {
            host_bytes,
            accelerator_bytes,
        }
    }

    pub const fn host(host_bytes: u64) -> Self {
        Self::new(host_bytes, 0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CoordinatorSnapshot {
    pub capacity: usize,
    pub active_jobs: usize,
    pub active_model_loads: usize,
    pub active_executions: usize,
    /// Total physical memory reserved across every backend memory domain.
    pub reserved_memory_bytes: u64,
    pub reserved_host_memory_bytes: u64,
    pub reserved_device_memory_bytes: u64,
    pub reserved_unified_memory_bytes: u64,
    pub admitted_total: u64,
    pub rejected_total: u64,
    pub expired_total: u64,
    pub draining: bool,
}

#[derive(Debug)]
pub struct InferenceCoordinator {
    execution_group_id: ExecutionGroupId,
    capacity: usize,
    backend: BackendKind,
    jobs: Arc<Semaphore>,
    host_work: Arc<Semaphore>,
    execution: Arc<Semaphore>,
    resources: Arc<ResourceAuthority>,
    admission_gate: Mutex<()>,
    idle: Notify,
    active_jobs: AtomicUsize,
    active_model_loads: AtomicUsize,
    active_executions: AtomicUsize,
    admitted_total: AtomicU64,
    rejected_total: AtomicU64,
    expired_total: AtomicU64,
    draining: AtomicBool,
}

impl InferenceCoordinator {
    #[cfg(test)]
    pub fn new(backend: BackendKind, execution_parallelism: usize, max_queued_jobs: usize) -> Self {
        let provider = Arc::new(DeviceCapacityProvider::for_tests(backend));
        Self::with_resource_authority(
            backend,
            execution_parallelism,
            max_queued_jobs,
            Arc::new(ResourceAuthority::new(provider)),
        )
    }

    pub fn new_with_device(
        backend: BackendKind,
        device: DeviceProfile,
        execution_parallelism: usize,
        max_queued_jobs: usize,
    ) -> Result<Self> {
        let location = validate_device_identity(backend, &device)?;
        let provider = Arc::new(DeviceCapacityProvider::new(backend, device)?);
        let resources = shared_resource_authority(location, provider)?;
        Ok(Self::with_resource_authority(
            backend,
            execution_parallelism,
            max_queued_jobs,
            resources,
        ))
    }

    fn with_resource_authority(
        backend: BackendKind,
        execution_parallelism: usize,
        max_queued_jobs: usize,
        resources: Arc<ResourceAuthority>,
    ) -> Self {
        let capacity = match backend {
            BackendKind::Metal => 1,
            BackendKind::Cpu | BackendKind::Cuda => execution_parallelism.max(1),
        };
        Self {
            execution_group_id: ExecutionGroupId::new(
                NEXT_EXECUTION_GROUP_ID.fetch_add(1, Ordering::Relaxed),
            ),
            capacity,
            backend,
            jobs: Arc::new(Semaphore::new(max_queued_jobs.max(capacity).max(1))),
            host_work: Arc::new(Semaphore::new(capacity)),
            execution: Arc::new(Semaphore::new(capacity)),
            resources,
            admission_gate: Mutex::new(()),
            idle: Notify::new(),
            active_jobs: AtomicUsize::new(0),
            active_model_loads: AtomicUsize::new(0),
            active_executions: AtomicUsize::new(0),
            admitted_total: AtomicU64::new(0),
            rejected_total: AtomicU64::new(0),
            expired_total: AtomicU64::new(0),
            draining: AtomicBool::new(false),
        }
    }

    pub fn resource_authority(&self) -> Arc<ResourceAuthority> {
        self.resources.clone()
    }

    pub(crate) fn execution_group_id(&self) -> ExecutionGroupId {
        self.execution_group_id
    }

    pub fn snapshot(&self) -> CoordinatorSnapshot {
        let reserved = self.resources.snapshot().reserved;
        let shared_host_unified = known_memory_bytes(reserved.host_bytes)
            .saturating_add(known_memory_bytes(reserved.unified_bytes));
        let reserved_host_memory_bytes = match self.backend {
            BackendKind::Cpu => shared_host_unified,
            BackendKind::Metal => 0,
            BackendKind::Cuda => known_memory_bytes(reserved.host_bytes),
        };
        let reserved_device_memory_bytes = known_memory_bytes(reserved.device_bytes);
        let reserved_unified_memory_bytes = match self.backend {
            BackendKind::Metal => shared_host_unified,
            BackendKind::Cpu => 0,
            BackendKind::Cuda => known_memory_bytes(reserved.unified_bytes),
        };
        let reserved_memory_bytes = reserved_host_memory_bytes
            .saturating_add(reserved_device_memory_bytes)
            .saturating_add(reserved_unified_memory_bytes);
        CoordinatorSnapshot {
            capacity: self.capacity,
            active_jobs: self.active_jobs.load(Ordering::Relaxed),
            active_model_loads: self.active_model_loads.load(Ordering::Relaxed),
            active_executions: self.active_executions.load(Ordering::Relaxed),
            reserved_memory_bytes,
            reserved_host_memory_bytes,
            reserved_device_memory_bytes,
            reserved_unified_memory_bytes,
            admitted_total: self.admitted_total.load(Ordering::Relaxed),
            rejected_total: self.rejected_total.load(Ordering::Relaxed),
            expired_total: self.expired_total.load(Ordering::Relaxed),
            draining: self.draining.load(Ordering::Acquire),
        }
    }

    pub fn begin_drain(&self) {
        let _gate = self
            .admission_gate
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.draining.store(true, Ordering::Release);
    }

    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    pub async fn wait_for_idle(&self, deadline: Instant) -> Result<()> {
        loop {
            let notified = self.idle.notified();
            if self.active_jobs.load(Ordering::Acquire) == 0
                && self.active_model_loads.load(Ordering::Acquire) == 0
                && self.active_executions.load(Ordering::Acquire) == 0
            {
                return Ok(());
            }
            tokio::time::timeout_at(deadline.into(), notified)
                .await
                .map_err(|_| Error::Timeout("inference coordinator drain".to_string()))?;
        }
    }

    pub async fn admit(self: &Arc<Self>, spec: JobSpec) -> Result<JobLease> {
        self.admit_with_initial_observation(spec, None)
    }

    /// Admit a job whose retained input allocation already exists and is
    /// reflected in the physical provider. Reservation and observation are
    /// committed atomically so existing memory is not charged as pending.
    pub async fn admit_observed(
        self: &Arc<Self>,
        spec: JobSpec,
        observation: JobResourceObservation,
    ) -> Result<JobLease> {
        self.admit_with_initial_observation(spec, Some(observation))
    }

    fn admit_with_initial_observation(
        self: &Arc<Self>,
        spec: JobSpec,
        observation: Option<JobResourceObservation>,
    ) -> Result<JobLease> {
        let _gate = self
            .admission_gate
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if self.draining.load(Ordering::Acquire) {
            self.rejected_total.fetch_add(1, Ordering::Relaxed);
            return Err(Error::Overloaded("runtime is draining".to_string()));
        }
        if spec
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            self.expired_total.fetch_add(1, Ordering::Relaxed);
            return Err(Error::Timeout(spec.request_id));
        }
        let permit = self.jobs.clone().try_acquire_owned().map_err(|_| {
            self.rejected_total.fetch_add(1, Ordering::Relaxed);
            Error::Overloaded("global inference queue is full".to_string())
        })?;
        let effective_resources = effective_resources(spec.resources, self.backend)?;
        let owner = ReservationOwner::new(ReservationClass::Request, spec.request_id.clone());
        let reservation = observation
            .map(|observation| observed_resources(observation, self.backend))
            .transpose()
            .and_then(|materialized| match materialized {
                Some(materialized) => self.resources.reserve_with_initial_materialized(
                    owner,
                    effective_resources,
                    materialized,
                ),
                None => self.resources.reserve(owner, effective_resources),
            })
            .inspect_err(|_err| {
                self.rejected_total.fetch_add(1, Ordering::Relaxed);
            })?;
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
        Ok(JobLease {
            _inner: Arc::new(JobLeaseInner {
                coordinator: self.clone(),
                _permit: permit,
                reservation,
            }),
            spec,
        })
    }

    /// Admit one cold model-load lifecycle operation. Model loads do not take
    /// an execution permit, but drain must wait for artifact acquisition,
    /// instantiation, and publication to finish before unloading residency.
    pub fn begin_model_load(
        self: &Arc<Self>,
        model_key: impl Into<String>,
    ) -> Result<ModelLoadLease> {
        let _gate = self
            .admission_gate
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if self.draining.load(Ordering::Acquire) {
            self.rejected_total.fetch_add(1, Ordering::Relaxed);
            return Err(Error::Overloaded("runtime is draining".to_string()));
        }
        self.active_model_loads.fetch_add(1, Ordering::AcqRel);
        Ok(ModelLoadLease {
            coordinator: self.clone(),
            _model_key: model_key.into(),
        })
    }

    async fn acquire_execution(
        self: &Arc<Self>,
        deadline: Option<Instant>,
    ) -> Result<ExecutionLease> {
        self.acquire_execution_units(1, deadline).await
    }

    /// Reserve the complete backend execution budget for one scheduler step.
    /// A CPU or CUDA step may fan out across `request_parallelism` worker threads, so
    /// holding a single permit would allow unrelated direct work to exceed the
    /// configured backend concurrency. Metal remains strictly singleton.
    async fn acquire_engine_step(self: &Arc<Self>) -> Result<ExecutionLease> {
        self.acquire_execution_units(self.capacity, None).await
    }

    /// Run one scheduler step while exclusively owning this backend's complete
    /// execution budget. Callers provide work, but never receive or retain a
    /// raw device permit; all model-forward routes therefore share this one
    /// coordinator-owned execution boundary.
    pub(crate) async fn run_engine_step<T, F>(self: &Arc<Self>, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let _execution = self.acquire_engine_step().await?;
        future.await
    }

    /// Reserve scratch memory for one physical tensor dispatch. This does not
    /// acquire execution capacity; the coordinator-owned engine or direct
    /// runner retains execution capacity independently for the device work.
    pub(crate) fn reserve_batch_workspace(
        &self,
        execution_group: ExecutionGroupId,
        batch_id: BatchId,
        resources: ResourceVector,
    ) -> Result<BatchWorkspaceLease> {
        if !matches!(resources.kv_bytes, ResourceAmount::Known(0)) {
            return Err(Error::InvalidInput(
                "batch workspace estimate contains persistent KV resources".to_string(),
            ));
        }
        let resources = effective_resources(resources, self.backend)?;
        self.resources
            .reserve_batch_workspace(execution_group, batch_id, resources)
    }

    async fn acquire_execution_units(
        self: &Arc<Self>,
        units: usize,
        deadline: Option<Instant>,
    ) -> Result<ExecutionLease> {
        if units == 0 {
            return Err(Error::InvalidInput(
                "execution units must be greater than zero".to_string(),
            ));
        }
        if units > self.capacity {
            return Err(Error::InvalidInput(format!(
                "requested {units} execution units exceeds coordinator capacity {}",
                self.capacity
            )));
        }
        let units = u32::try_from(units).map_err(|_| {
            Error::InvalidInput("execution unit request exceeds supported range".to_string())
        })?;
        let acquire = self.execution.clone().acquire_many_owned(units);
        let permit = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire)
                .await
                .map_err(|_| {
                    self.expired_total.fetch_add(1, Ordering::Relaxed);
                    Error::Timeout("device execution capacity".to_string())
                })?
                .map_err(|_| Error::Overloaded("coordinator closed".to_string()))?,
            None => acquire
                .await
                .map_err(|_| Error::Overloaded("coordinator closed".to_string()))?,
        };
        self.active_executions.fetch_add(1, Ordering::Relaxed);
        Ok(ExecutionLease {
            coordinator: self.clone(),
            _permit: permit,
        })
    }

    /// Admit a request before invoking its potentially expensive preparation
    /// closure. The caller deadline covers model loading and preprocessing as
    /// well as later execution, and a rejected request performs no preparation.
    pub async fn admit_then_prepare<T, P, F>(
        self: &Arc<Self>,
        spec: JobSpec,
        prepare: P,
    ) -> Result<(JobLease, T)>
    where
        P: FnOnce() -> F,
        F: Future<Output = Result<T>>,
    {
        let job = self.admit(spec).await?;
        self.prepare_admitted_job(job, prepare).await
    }

    /// Admit an input that is already physically allocated, then perform
    /// asynchronous preparation under the same end-to-end deadline.
    pub async fn admit_then_prepare_observed<T, P, F>(
        self: &Arc<Self>,
        spec: JobSpec,
        observation: JobResourceObservation,
        prepare: P,
    ) -> Result<(JobLease, T)>
    where
        P: FnOnce() -> F,
        F: Future<Output = Result<T>>,
    {
        let job = self.admit_observed(spec, observation).await?;
        self.prepare_admitted_job(job, prepare).await
    }

    async fn prepare_admitted_job<T, P, F>(
        self: &Arc<Self>,
        job: JobLease,
        prepare: P,
    ) -> Result<(JobLease, T)>
    where
        P: FnOnce() -> F,
        F: Future<Output = Result<T>>,
    {
        let request_id = job.spec.request_id.clone();
        let prepared = match job.spec.deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), prepare())
                .await
                .map_err(|_| {
                    self.expired_total.fetch_add(1, Ordering::Relaxed);
                    Error::Timeout(request_id.clone())
                })??,
            None => prepare().await?,
        };
        if job
            .spec
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            self.expired_total.fetch_add(1, Ordering::Relaxed);
            return Err(Error::Timeout(request_id));
        }
        Ok((job, prepared))
    }

    #[cfg(test)]
    pub(crate) async fn run_stage<T, F>(self: &Arc<Self>, job: &JobLease, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let deadline = job.spec.deadline;
        let _execution = self.acquire_execution(deadline).await?;
        let result = future.await;
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            self.expired_total.fetch_add(1, Ordering::Relaxed);
            return Err(Error::Timeout("direct inference job".to_string()));
        }
        result
    }

    /// Acquire an operation-ordering guard under the job's deadline. This is
    /// used before allocating owned inputs for serialized blocking stages, so
    /// cancelled or timed-out callers cannot queue unbounded input copies.
    pub(crate) async fn acquire_job_ordering(
        &self,
        job: &JobLease,
        ordering: Arc<tokio::sync::Mutex<()>>,
    ) -> Result<OwnedMutexGuard<()>> {
        let acquire = ordering.lock_owned();
        match job.spec.deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire)
                .await
                .map_err(|_| {
                    self.expired_total.fetch_add(1, Ordering::Relaxed);
                    Error::Timeout(job.spec.request_id.clone())
                }),
            None => Ok(acquire.await),
        }
    }

    /// Execute synchronous model work off Tokio workers while retaining the
    /// execution permit and job reservation until physical work really ends.
    /// Deadline expiry detaches the blocking task instead of cancelling it.
    #[cfg(test)]
    pub(crate) async fn run_blocking_stage<T, F>(
        self: &Arc<Self>,
        job: &JobLease,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        self.run_blocking_stage_inner(job, operation).await
    }

    /// Execute admitted CPU-side preparation without spending backend device
    /// capacity. Host work has its own bounded lane and retains the job lease
    /// until a detached blocking task physically exits.
    pub(crate) async fn run_host_blocking_stage<T, F>(
        self: &Arc<Self>,
        job: &JobLease,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let acquire = self.host_work.clone().acquire_owned();
        let permit = match job.spec.deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire)
                .await
                .map_err(|_| {
                    self.expired_total.fetch_add(1, Ordering::Relaxed);
                    Error::Timeout(job.spec.request_id.clone())
                })?
                .map_err(|_| Error::Overloaded("coordinator closed".to_string()))?,
            None => acquire
                .await
                .map_err(|_| Error::Overloaded("coordinator closed".to_string()))?,
        };
        self.run_blocking_task(job, permit, "host preparation", operation)
            .await
    }

    /// Execute one independently scheduled scalar row through an exact loaded
    /// adapter. `max_batch_size` bounds concurrent rows for scalar adapters; it
    /// does not imply that this call receives a native tensor batch.
    /// Direct, realtime, and pipeline runners use this compatibility boundary
    /// until their model stage gains a proven native tensor adapter.
    pub(crate) async fn run_loaded_blocking_stage<T, F>(
        self: &Arc<Self>,
        job: &JobLease,
        contract: LoadedExecutionContract,
        work: WorkUnit,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        if !contract.execution_profile.resolved_from_loaded_model {
            return Err(Error::InferenceError(
                "runtime execution contract was not resolved from a loaded model".to_string(),
            ));
        }
        if contract.execution_group_id != self.execution_group_id {
            return Err(Error::InferenceError(
                "runtime execution contract belongs to a different execution group".to_string(),
            ));
        }
        if contract.execution_profile.backend != self.backend {
            return Err(Error::InferenceError(
                "runtime execution contract belongs to a different backend".to_string(),
            ));
        }
        let binding = contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.domain != ExecutionDomain::ExecutionGroup {
            return Err(Error::InvalidInput(
                "host-only adapter stage cannot enter device execution".to_string(),
            ));
        }
        if stage.batch_mode != NativeBatchMode::None {
            return Err(Error::InvalidInput(
                "runtime scalar runner cannot execute a native tensor stage".to_string(),
            ));
        }

        self.run_blocking_stage_inner(job, move || {
            let _contract = contract;
            let _work = work;
            operation()
        })
        .await
    }

    /// Execute a scalar stage with the complete typed invocation workspace
    /// sealed into its loaded capability binding. Paged attention, recurrent,
    /// append, ring, and static state are acquired and completed atomically.
    pub(crate) async fn run_loaded_blocking_stage_with_invocation_workspace<T, F>(
        self: &Arc<Self>,
        job: &JobLease,
        contract: LoadedExecutionContract,
        capability: LoadedCapabilityBinding,
        work: WorkUnit,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut crate::kv::v2::InvocationWorkspaceLeaseSetV2) -> Result<T> + Send + 'static,
    {
        let binding = contract.adapter_binding()?;
        if capability.execution != binding {
            return Err(Error::InferenceError(
                "direct invocation state belongs to a different loaded execution adapter"
                    .to_string(),
            ));
        }
        capability
            .state
            .validate_against(self.backend, &capability.execution)?;
        let stage = binding.stage_for_work(&work)?;
        if stage.domain != ExecutionDomain::ExecutionGroup
            || stage.batch_mode != NativeBatchMode::None
        {
            return Err(Error::InvalidInput(
                "direct invocation state requires a scalar execution-group stage".to_string(),
            ));
        }
        let stage_id = stage.id;
        self.run_blocking_stage_inner(job, move || {
            let _contract = contract;
            let _work = work;
            let mut leases = capability
                .state
                .lease_complete_invocation_workspace_set(stage_id)?;
            let output = operation(&mut leases)?;
            let _completions = leases.release()?;
            Ok(output)
        })
        .await
    }

    /// Execute a scalar stage with the complete invocation-paged domain set
    /// sealed into its loaded capability binding.
    pub(crate) async fn run_loaded_blocking_stage_with_invocation_paged<T, F>(
        self: &Arc<Self>,
        job: &JobLease,
        contract: LoadedExecutionContract,
        capability: LoadedCapabilityBinding,
        work: WorkUnit,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut crate::kv::v2::InvocationPagedLeaseSetV2) -> Result<T> + Send + 'static,
    {
        let binding = contract.adapter_binding()?;
        if capability.execution != binding {
            return Err(Error::InferenceError(
                "direct invocation state belongs to a different loaded execution adapter"
                    .to_string(),
            ));
        }
        capability
            .state
            .validate_against(self.backend, &capability.execution)?;
        let stage = binding.stage_for_work(&work)?;
        if stage.domain != ExecutionDomain::ExecutionGroup
            || stage.batch_mode != NativeBatchMode::None
        {
            return Err(Error::InvalidInput(
                "direct invocation paging requires a scalar execution-group stage".to_string(),
            ));
        }
        let stage_id = stage.id;
        self.run_blocking_stage_inner(job, move || {
            let _contract = contract;
            let _work = work;
            let mut leases = capability
                .state
                .lease_complete_invocation_paged_set(stage_id)?;
            let output = operation(&mut leases)?;
            let _completions = leases.release()?;
            Ok(output)
        })
        .await
    }

    async fn run_blocking_stage_inner<T, F>(
        self: &Arc<Self>,
        job: &JobLease,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let deadline = job.spec.deadline;
        let execution = self.acquire_execution(deadline).await?;
        self.run_blocking_task(job, execution, "inference", operation)
            .await
    }

    async fn run_blocking_task<T, F, G>(
        self: &Arc<Self>,
        job: &JobLease,
        guard: G,
        task_kind: &'static str,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
        G: Send + 'static,
    {
        let deadline = job.spec.deadline;
        let request_id = job.spec.request_id.clone();
        let retained_job = job.clone();
        let blocking_request_id = request_id.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let _guard = guard;
            let _job = retained_job;
            if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
                return Err(Error::Timeout(blocking_request_id));
            }
            operation()
        });
        let joined = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), handle)
                .await
                .map_err(|_| {
                    self.expired_total.fetch_add(1, Ordering::Relaxed);
                    Error::Timeout(request_id.clone())
                })?,
            None => handle.await,
        };
        let result = joined.map_err(|err| {
            Error::InferenceError(format!("blocking {task_kind} task failed: {err}"))
        })?;
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            self.expired_total.fetch_add(1, Ordering::Relaxed);
            return Err(Error::Timeout(request_id));
        }
        result
    }
}

fn validate_device_identity(
    backend: BackendKind,
    device: &DeviceProfile,
) -> Result<DeviceLocation> {
    let location = device.device.location();
    validate_device_identity_parts(backend, device.kind, location)?;
    Ok(location)
}

fn validate_device_identity_parts(
    backend: BackendKind,
    kind: DeviceKind,
    location: DeviceLocation,
) -> Result<()> {
    let kind_backend = BackendKind::from(kind);
    let location_backend = match location {
        DeviceLocation::Cpu => BackendKind::Cpu,
        DeviceLocation::Metal { .. } => BackendKind::Metal,
        DeviceLocation::Cuda { .. } => BackendKind::Cuda,
    };
    if backend != kind_backend || kind_backend != location_backend {
        return Err(Error::ConfigError(format!(
            "inconsistent inference device identity: backend={backend:?}, kind={kind:?}, location={location:?}"
        )));
    }
    Ok(())
}

#[derive(Debug, Default)]
struct ResourceAuthorityRegistry {
    state: Mutex<ResourceAuthorityRegistryState>,
}

#[derive(Debug, Default)]
struct ResourceAuthorityRegistryState {
    host_unified: Option<SharedAuthorityRegistration>,
    exclusive_devices: HashMap<DeviceLocation, Weak<ResourceAuthority>>,
}

#[derive(Debug)]
struct SharedAuthorityRegistration {
    authority: Weak<ResourceAuthority>,
    provider: Arc<SharedHostUnifiedCapacityProvider>,
}

#[derive(Debug, Default)]
struct SharedHostUnifiedCapacityProvider {
    providers: Mutex<HashMap<DeviceLocation, Arc<dyn PhysicalCapacityProvider>>>,
}

impl SharedHostUnifiedCapacityProvider {
    fn register(&self, location: DeviceLocation, provider: Arc<dyn PhysicalCapacityProvider>) {
        self.providers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .entry(location)
            .or_insert(provider);
    }
}

impl PhysicalCapacityProvider for SharedHostUnifiedCapacityProvider {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        self.combined_snapshot(false)
    }

    fn refresh_after_release(&self) -> PhysicalCapacitySnapshot {
        self.combined_snapshot(true)
    }
}

impl SharedHostUnifiedCapacityProvider {
    fn combined_snapshot(&self, refresh_after_release: bool) -> PhysicalCapacitySnapshot {
        let providers = self
            .providers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut snapshots = providers.values().map(|provider| {
            if refresh_after_release {
                provider.refresh_after_release()
            } else {
                provider.snapshot()
            }
        });
        let Some(first) = snapshots.next() else {
            return PhysicalCapacitySnapshot {
                capacity: shared_host_unified_vector(ResourceAmount::Unknown),
                available: shared_host_unified_vector(ResourceAmount::Unknown),
                source: CapacitySource::Unavailable,
            };
        };
        let mut capacity = shared_host_unified_amount(first.capacity);
        let mut available = shared_host_unified_amount(first.available);
        for snapshot in snapshots {
            capacity =
                minimum_known_amount(capacity, shared_host_unified_amount(snapshot.capacity));
            available =
                minimum_known_amount(available, shared_host_unified_amount(snapshot.available));
        }
        PhysicalCapacitySnapshot {
            capacity: shared_host_unified_vector(capacity),
            available: shared_host_unified_vector(available),
            source: first.source,
        }
    }
}

fn shared_host_unified_amount(resources: ResourceVector) -> ResourceAmount {
    match (resources.host_bytes, resources.unified_bytes) {
        (ResourceAmount::Known(host), ResourceAmount::Known(unified)) => host
            .checked_add(unified)
            .map(ResourceAmount::Known)
            .unwrap_or(ResourceAmount::Unknown),
        _ => ResourceAmount::Unknown,
    }
}

fn shared_host_unified_vector(amount: ResourceAmount) -> ResourceVector {
    ResourceVector {
        unified_bytes: amount,
        ..ResourceVector::zero()
    }
}

fn minimum_known_amount(left: ResourceAmount, right: ResourceAmount) -> ResourceAmount {
    match (left, right) {
        (ResourceAmount::Known(left), ResourceAmount::Known(right)) => {
            ResourceAmount::Known(left.min(right))
        }
        _ => ResourceAmount::Unknown,
    }
}

impl ResourceAuthorityRegistry {
    fn authority_for(
        &self,
        location: DeviceLocation,
        provider: Arc<dyn PhysicalCapacityProvider>,
    ) -> Result<Arc<ResourceAuthority>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state
            .exclusive_devices
            .retain(|_, authority| authority.strong_count() > 0);
        if state
            .host_unified
            .as_ref()
            .is_some_and(|registration| registration.authority.strong_count() == 0)
        {
            state.host_unified = None;
        }

        match location {
            DeviceLocation::Cpu | DeviceLocation::Metal { .. } => {
                if let Some(conflicting_location) = state.exclusive_devices.keys().next().copied() {
                    return Err(resource_authority_conflict(conflicting_location, location));
                }
                if let Some(registration) = state.host_unified.as_ref() {
                    if let Some(authority) = registration.authority.upgrade() {
                        registration.provider.register(location, provider);
                        return Ok(authority);
                    }
                }
                let shared_provider = Arc::new(SharedHostUnifiedCapacityProvider::default());
                shared_provider.register(location, provider);
                let authority = Arc::new(ResourceAuthority::new_shared_host_unified(
                    shared_provider.clone(),
                ));
                state.host_unified = Some(SharedAuthorityRegistration {
                    authority: Arc::downgrade(&authority),
                    provider: shared_provider,
                });
                Ok(authority)
            }
            DeviceLocation::Cuda { .. } => {
                if state
                    .host_unified
                    .as_ref()
                    .and_then(|registration| registration.authority.upgrade())
                    .is_some()
                {
                    return Err(resource_authority_conflict(DeviceLocation::Cpu, location));
                }
                if let Some(authority) = state
                    .exclusive_devices
                    .get(&location)
                    .and_then(Weak::upgrade)
                {
                    return Ok(authority);
                }
                if let Some(conflicting_location) = state.exclusive_devices.keys().next().copied() {
                    return Err(resource_authority_conflict(conflicting_location, location));
                }
                let authority = Arc::new(ResourceAuthority::new(provider));
                state
                    .exclusive_devices
                    .insert(location, Arc::downgrade(&authority));
                Ok(authority)
            }
        }
    }
}

fn resource_authority_conflict(active: DeviceLocation, requested: DeviceLocation) -> Error {
    Error::ConfigError(format!(
        "simultaneous physical device locations are unsupported: {active:?} is already active and {requested:?} cannot be registered because the shared host-memory domain for CUDA is not yet split from per-device accounting"
    ))
}

fn shared_resource_authority(
    location: DeviceLocation,
    provider: Arc<dyn PhysicalCapacityProvider>,
) -> Result<Arc<ResourceAuthority>> {
    // CPU host allocations and Metal unified allocations spend one physical
    // pool on Apple Silicon, so they share a canonical authority even when
    // workers use distinct DeviceLocation identities. CUDA remains exclusive
    // until its shared host claim and per-device claim become composite.
    static AUTHORITIES: OnceLock<ResourceAuthorityRegistry> = OnceLock::new();
    AUTHORITIES
        .get_or_init(ResourceAuthorityRegistry::default)
        .authority_for(location, provider)
}

#[derive(Debug, Clone)]
pub struct JobLease {
    _inner: Arc<JobLeaseInner>,
    pub spec: JobSpec,
}

impl JobLease {
    /// Reconcile observed physical usage against this job's immutable
    /// authorization. Future authorized growth remains pending. Observations
    /// are monotonic; the runtime restores pending claims before dropping or
    /// replacing temporary physical allocations.
    pub fn record_materialized_usage(&self, observation: JobResourceObservation) -> Result<()> {
        let resources = observed_resources(observation, self._inner.coordinator.backend)?;
        self._inner.reservation.record_materialized_usage(resources)
    }

    /// Restore a pending claim before releasing or replacing temporary
    /// physical storage. Delaying the physical release is conservative; doing
    /// it before this transition is not allowed.
    pub(crate) fn prepare_materialized_release(
        &self,
        observation: JobResourceObservation,
    ) -> Result<()> {
        let resources = observed_resources(observation, self._inner.coordinator.backend)?;
        self._inner
            .reservation
            .prepare_materialized_release(resources)
    }
}

#[derive(Debug)]
pub struct ModelLoadLease {
    coordinator: Arc<InferenceCoordinator>,
    _model_key: String,
}

impl Drop for ModelLoadLease {
    fn drop(&mut self) {
        if self
            .coordinator
            .active_model_loads
            .fetch_sub(1, Ordering::AcqRel)
            == 1
        {
            self.coordinator.idle.notify_waiters();
        }
    }
}

#[derive(Debug)]
struct JobLeaseInner {
    coordinator: Arc<InferenceCoordinator>,
    _permit: OwnedSemaphorePermit,
    reservation: ResourceLease,
}

impl Drop for JobLeaseInner {
    fn drop(&mut self) {
        if self.coordinator.active_jobs.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.coordinator.idle.notify_waiters();
        }
    }
}

fn effective_resources(requested: ResourceVector, backend: BackendKind) -> Result<ResourceVector> {
    let reject_off_domain = |amount: ResourceAmount, domain: &str| {
        match amount {
        ResourceAmount::Known(0) => Ok(()),
        ResourceAmount::Known(_) => Err(Error::InvalidInput(format!(
            "{backend:?} request resource estimate contains nonzero {domain} memory outside its physical backend domain"
        ))),
        ResourceAmount::Unknown => Err(Error::InvalidInput(format!(
            "{backend:?} request resource estimate contains unresolved {domain} memory outside its physical backend domain"
        ))),
    }
    };
    match requested.compute_slots {
        ResourceAmount::Known(0) => {}
        ResourceAmount::Known(_) => {
            return Err(Error::InvalidInput(
                "request resource estimate contains nonzero compute_slots, but runtime concurrency is governed by coordinator permits"
                    .to_string(),
            ));
        }
        ResourceAmount::Unknown => {
            return Err(Error::InvalidInput(
                "request resource estimate contains unresolved compute_slots, but runtime concurrency is governed by coordinator permits"
                    .to_string(),
            ));
        }
    }
    match backend {
        BackendKind::Cpu => {
            reject_off_domain(requested.unified_bytes, "unified")?;
            reject_off_domain(requested.device_bytes, "device")?;
        }
        BackendKind::Metal => {
            reject_off_domain(requested.host_bytes, "host")?;
            reject_off_domain(requested.device_bytes, "device")?;
        }
        BackendKind::Cuda => {
            reject_off_domain(requested.unified_bytes, "unified")?;
        }
    }
    let requested_memory = match backend {
        BackendKind::Cpu => requested.host_bytes,
        BackendKind::Metal => requested.unified_bytes,
        BackendKind::Cuda => requested.device_bytes,
    };
    let ResourceAmount::Known(memory) = requested_memory else {
        return Err(Error::InvalidInput(
            "request memory estimate is unresolved".to_string(),
        ));
    };
    let kv = known_or_zero(requested.kv_bytes)?;
    let temporary = known_or_zero(requested.temporary_bytes)?;
    let effective_memory = ResourceAmount::Known(
        memory
            .checked_add(kv)
            .and_then(|value| value.checked_add(temporary))
            .ok_or_else(|| Error::Overloaded("request memory estimate overflow".to_string()))?,
    );
    let mut effective = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => effective.host_bytes = effective_memory,
        BackendKind::Metal => effective.unified_bytes = effective_memory,
        BackendKind::Cuda => {
            effective.host_bytes = ResourceAmount::Known(known_or_zero(requested.host_bytes)?);
            effective.device_bytes = effective_memory;
        }
    }
    Ok(effective)
}

fn observed_resources(
    observation: JobResourceObservation,
    backend: BackendKind,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes = ResourceAmount::Known(
                observation
                    .host_bytes
                    .checked_add(observation.accelerator_bytes)
                    .ok_or_else(|| {
                        Error::InferenceError("observed job memory overflowed".to_string())
                    })?,
            );
        }
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(
                observation
                    .host_bytes
                    .checked_add(observation.accelerator_bytes)
                    .ok_or_else(|| {
                        Error::InferenceError("observed job memory overflowed".to_string())
                    })?,
            );
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(observation.host_bytes);
            resources.device_bytes = ResourceAmount::Known(observation.accelerator_bytes);
        }
    }
    Ok(resources)
}

fn known_or_zero(amount: ResourceAmount) -> Result<u64> {
    match amount {
        ResourceAmount::Known(value) => Ok(value),
        ResourceAmount::Unknown => Err(Error::InvalidInput(
            "request resource estimate contains an unresolved quantity".to_string(),
        )),
    }
}

fn known_memory_bytes(amount: ResourceAmount) -> u64 {
    match amount {
        ResourceAmount::Known(value) => value,
        ResourceAmount::Unknown => 0,
    }
}

// Admission calls the provider while the shared resource-authority ledger is
// locked. Keep fresh and bounded-stale reads to a cache lookup; slow or wedged
// OS/device probes are isolated to the single sampler worker. After hard-stale
// expiry, admission waits once for that worker under a strict bound and fails
// closed if no successful refresh arrives.
const CAPACITY_SAMPLE_FRESH_FOR: Duration = Duration::from_millis(250);
const CAPACITY_SAMPLE_MAX_STALE: Duration = Duration::from_secs(1);
const CAPACITY_SAMPLE_RETRY_AFTER: Duration = Duration::from_millis(250);
// macOS capacity sampling runs two commands with individual 250 ms deadlines.
// Keep admission bounded beyond their combined worst case while CUDA/CPU probes
// remain isolated on the same sampler thread.
const CAPACITY_REFRESH_WAIT: Duration = Duration::from_millis(550);

#[cfg(target_os = "macos")]
const CAPACITY_COMMAND_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy)]
struct CachedCapacitySample {
    snapshot: PhysicalCapacitySnapshot,
    sampled_at: Instant,
}

#[derive(Debug)]
struct CapacitySampleState {
    sample: Option<CachedCapacitySample>,
    last_attempt: Option<Instant>,
    refresh_in_flight: bool,
}

#[derive(Debug)]
struct CapacitySampleCache {
    state: Mutex<CapacitySampleState>,
    refreshed: Condvar,
}

#[derive(Debug, Clone, Copy)]
struct CapacityCacheDecision {
    snapshot: Option<PhysicalCapacitySnapshot>,
    request_refresh: bool,
}

impl CapacitySampleCache {
    fn new(initial: Option<PhysicalCapacitySnapshot>, now: Instant) -> Self {
        Self {
            state: Mutex::new(CapacitySampleState {
                sample: initial.map(|snapshot| CachedCapacitySample {
                    snapshot,
                    sampled_at: now,
                }),
                last_attempt: Some(now),
                refresh_in_flight: false,
            }),
            refreshed: Condvar::new(),
        }
    }

    fn decision(&self, now: Instant) -> CapacityCacheDecision {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let sample_age = state
            .sample
            .and_then(|sample| now.checked_duration_since(sample.sampled_at));
        let fresh = sample_age.is_some_and(|age| age <= CAPACITY_SAMPLE_FRESH_FOR);
        let retry_ready = state
            .last_attempt
            .and_then(|last_attempt| now.checked_duration_since(last_attempt))
            .is_none_or(|elapsed| elapsed >= CAPACITY_SAMPLE_RETRY_AFTER);
        let request_refresh = !fresh && !state.refresh_in_flight && retry_ready;
        if request_refresh {
            state.refresh_in_flight = true;
            state.last_attempt = Some(now);
        }
        let snapshot = sample_age
            .filter(|age| *age <= CAPACITY_SAMPLE_MAX_STALE)
            .and_then(|_| state.sample.map(|sample| sample.snapshot));
        CapacityCacheDecision {
            snapshot,
            request_refresh,
        }
    }

    fn finish_refresh(&self, snapshot: Option<PhysicalCapacitySnapshot>, now: Instant) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(snapshot) =
            snapshot.filter(|_| state.sample.is_none_or(|current| now >= current.sampled_at))
        {
            state.sample = Some(CachedCapacitySample {
                snapshot,
                sampled_at: now,
            });
        }
        state.refresh_in_flight = false;
        drop(state);
        self.refreshed.notify_all();
    }

    fn publish_sample(&self, snapshot: PhysicalCapacitySnapshot, sampled_at: Instant) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let should_publish = state
            .sample
            .is_none_or(|current| sampled_at >= current.sampled_at);
        if should_publish {
            state.sample = Some(CachedCapacitySample {
                snapshot,
                sampled_at,
            });
        }
        drop(state);
        self.refreshed.notify_all();
    }

    /// Wait for an already-isolated refresh only when hard-stale expiry left no
    /// usable snapshot. The caller remains fail-closed if the probe fails,
    /// disconnects, or does not finish within the bounded wait.
    fn wait_for_refresh(&self, timeout: Duration) -> Option<PhysicalCapacitySnapshot> {
        let deadline = Instant::now().checked_add(timeout)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        loop {
            let now = Instant::now();
            let snapshot = state
                .sample
                .filter(|sample| {
                    now.checked_duration_since(sample.sampled_at)
                        .is_some_and(|age| age <= CAPACITY_SAMPLE_MAX_STALE)
                })
                .map(|sample| sample.snapshot);
            if snapshot.is_some() || !state.refresh_in_flight {
                return snapshot;
            }

            let remaining = deadline.saturating_duration_since(now);
            if remaining.is_zero() {
                return None;
            }
            let (next_state, _wait) = self
                .refreshed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poison| poison.into_inner());
            state = next_state;
        }
    }
}

fn guarded_capacity_sample<F>(sample: F) -> Option<PhysicalCapacitySnapshot>
where
    F: FnOnce() -> Option<PhysicalCapacitySnapshot>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(sample))
        .ok()
        .flatten()
}

#[derive(Debug, Clone)]
struct DeviceCapacityProbe {
    backend: BackendKind,
    device: Option<DeviceProfile>,
    configured_cap: Option<u64>,
    test_capacity: Option<u64>,
}

impl DeviceCapacityProbe {
    fn apply_cap(&self, total: u64, available: u64) -> (u64, u64) {
        match self.configured_cap {
            Some(cap) => {
                let effective_total = total.min(cap);
                let used = total.saturating_sub(available.min(total));
                (effective_total, effective_total.saturating_sub(used))
            }
            None => (total, available),
        }
    }

    fn vector(&self, amount: ResourceAmount) -> ResourceVector {
        let mut vector = ResourceVector::zero();
        match self.backend {
            BackendKind::Cpu => vector.host_bytes = amount,
            BackendKind::Metal => vector.unified_bytes = amount,
            BackendKind::Cuda => vector.device_bytes = amount,
        }
        vector
    }

    fn unavailable_snapshot(&self) -> PhysicalCapacitySnapshot {
        PhysicalCapacitySnapshot {
            capacity: self.vector(ResourceAmount::Unknown),
            available: self.vector(ResourceAmount::Unknown),
            source: CapacitySource::Unavailable,
        }
    }

    fn observed_capacity(&self) -> Option<(u64, u64, CapacitySource)> {
        if let Some(capacity) = self.test_capacity {
            return Some((capacity, capacity, CapacitySource::Test));
        }
        let device = self.device.as_ref()?;
        match self.backend {
            BackendKind::Cpu => host_memory_snapshot()
                .map(|(total, available)| (total, available, CapacitySource::OperatingSystem)),
            BackendKind::Metal => metal_memory_snapshot(device),
            BackendKind::Cuda => cuda_memory_snapshot(device),
        }
    }

    fn sample(&self) -> Option<PhysicalCapacitySnapshot> {
        let (total, available, source) = self.observed_capacity()?;
        let (capacity, available) = self.apply_cap(total, available);
        let mut capacity_vector = self.vector(ResourceAmount::Known(capacity));
        let mut available_vector = self.vector(ResourceAmount::Known(available));
        if self.backend == BackendKind::Cuda {
            let (host_total, host_available) = match self.test_capacity {
                Some(capacity) => (capacity, capacity),
                None => host_memory_snapshot()?,
            };
            capacity_vector.host_bytes = ResourceAmount::Known(host_total);
            available_vector.host_bytes = ResourceAmount::Known(host_available);
        }
        Some(PhysicalCapacitySnapshot {
            capacity: capacity_vector,
            available: available_vector,
            source,
        })
    }
}

#[derive(Debug)]
struct DeviceCapacityProvider {
    probe: DeviceCapacityProbe,
    cache: Arc<CapacitySampleCache>,
    refresh: Option<std::sync::mpsc::SyncSender<()>>,
}

impl DeviceCapacityProvider {
    fn new(backend: BackendKind, device: DeviceProfile) -> Result<Self> {
        let env_name = match backend {
            BackendKind::Cpu => "IZWI_CPU_MEMORY_BUDGET_BYTES",
            BackendKind::Metal => "IZWI_METAL_MEMORY_BUDGET_BYTES",
            BackendKind::Cuda => "IZWI_CUDA_MEMORY_BUDGET_BYTES",
        };
        let configured_cap = match std::env::var(env_name) {
            Ok(raw) => Some(
                raw.parse::<u64>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        Error::ConfigError(format!("{env_name} must be a positive integer"))
                    })?,
            ),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => {
                return Err(Error::ConfigError(format!(
                    "failed to read {env_name}: {err}"
                )))
            }
        };
        let probe = DeviceCapacityProbe {
            backend,
            device: Some(device),
            configured_cap,
            test_capacity: cfg!(test).then_some(1024 * 1024 * 1024 * 1024),
        };
        Self::from_probe(probe)
    }

    #[cfg(test)]
    fn for_tests(backend: BackendKind) -> Self {
        let probe = DeviceCapacityProbe {
            backend,
            device: None,
            configured_cap: None,
            test_capacity: Some(64 * 1024 * 1024 * 1024),
        };
        Self::from_probe(probe).expect("fixed test capacity must initialize")
    }

    fn from_probe(probe: DeviceCapacityProbe) -> Result<Self> {
        let now = Instant::now();
        let initial = probe.sample().ok_or_else(|| {
            Error::ConfigError(format!(
                "failed to query physical capacity for {:?} inference device",
                probe.backend
            ))
        })?;
        let cache = Arc::new(CapacitySampleCache::new(Some(initial), now));
        if probe.test_capacity.is_some() {
            return Ok(Self {
                probe,
                cache,
                refresh: None,
            });
        }
        let (refresh, requests) = std::sync::mpsc::sync_channel(1);
        let refresh_probe = probe.clone();
        let refresh_cache = cache.clone();
        std::thread::Builder::new()
            .name(format!("izwi-capacity-{:?}", probe.backend).to_lowercase())
            .spawn(move || {
                while requests.recv().is_ok() {
                    let sampled_at = Instant::now();
                    let sample = guarded_capacity_sample(|| refresh_probe.sample());
                    refresh_cache.finish_refresh(sample, sampled_at);
                }
            })
            .map_err(|err| {
                Error::ConfigError(format!(
                    "failed to start physical-capacity sampler for {:?}: {err}",
                    probe.backend
                ))
            })?;
        Ok(Self {
            probe,
            cache,
            refresh: Some(refresh),
        })
    }
}

impl PhysicalCapacityProvider for DeviceCapacityProvider {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        if self.probe.test_capacity.is_some() {
            return self
                .probe
                .sample()
                .unwrap_or_else(|| self.probe.unavailable_snapshot());
        }
        let decision = self.cache.decision(Instant::now());
        if decision.request_refresh {
            match self.refresh.as_ref().map(|refresh| refresh.try_send(())) {
                Some(Ok(())) => {}
                Some(Err(std::sync::mpsc::TrySendError::Full(()))) => {}
                Some(Err(std::sync::mpsc::TrySendError::Disconnected(()))) | None => {
                    self.cache.finish_refresh(None, Instant::now());
                }
            }
        }
        decision
            .snapshot
            .or_else(|| self.cache.wait_for_refresh(CAPACITY_REFRESH_WAIT))
            .unwrap_or_else(|| self.probe.unavailable_snapshot())
    }

    fn refresh_after_release(&self) -> PhysicalCapacitySnapshot {
        let sampled_at = Instant::now();
        let snapshot = guarded_capacity_sample(|| self.probe.sample())
            .unwrap_or_else(|| self.probe.unavailable_snapshot());
        self.cache.publish_sample(snapshot, sampled_at);
        snapshot
    }
}

fn host_memory_snapshot() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        let value = |name: &str| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key == name).then(|| {
                    value
                        .split_whitespace()
                        .next()
                        .and_then(|raw| raw.parse::<u64>().ok())
                        .map(|kb| kb.saturating_mul(1024))
                })?
            })
        };
        let mut total = value("MemTotal")?;
        let mut available = value("MemAvailable")?;
        if let (Ok(max), Ok(current)) = (
            std::fs::read_to_string("/sys/fs/cgroup/memory.max"),
            std::fs::read_to_string("/sys/fs/cgroup/memory.current"),
        ) {
            if let (Ok(max), Ok(current)) =
                (max.trim().parse::<u64>(), current.trim().parse::<u64>())
            {
                total = total.min(max);
                available = available.min(max.saturating_sub(current));
            }
        }
        return Some((total, available));
    }
    #[cfg(target_os = "macos")]
    {
        let total_output = command_stdout_with_timeout(
            "/usr/sbin/sysctl",
            &["-n", "hw.memsize"],
            CAPACITY_COMMAND_TIMEOUT,
        )?;
        let total = std::str::from_utf8(&total_output)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        let vm_output =
            command_stdout_with_timeout("/usr/bin/vm_stat", &[], CAPACITY_COMMAND_TIMEOUT)?;
        let available = parse_macos_vm_stat_available(std::str::from_utf8(&vm_output).ok()?)?;
        return Some((total, available.min(total)));
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "macos")]
fn command_stdout_with_timeout(program: &str, args: &[&str], timeout: Duration) -> Option<Vec<u8>> {
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now().checked_add(timeout)?;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut output = Vec::new();
                let mut stdout = child.stdout.take()?;
                std::io::Read::read_to_end(&mut stdout, &mut output).ok()?;
                return Some(output);
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

#[cfg(any(target_os = "macos", test))]
fn parse_macos_vm_stat_available(contents: &str) -> Option<u64> {
    let page_size = contents
        .lines()
        .next()?
        .split_once("page size of ")?
        .1
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    let pages = |label: &str| {
        contents.lines().find_map(|line| {
            let (key, raw) = line.split_once(':')?;
            (key.trim() == label).then(|| raw.trim().trim_end_matches('.').parse::<u64>().ok())?
        })
    };

    // Free, inactive, and speculative pages can be reclaimed without paging
    // anonymous active memory. Purgeable pages are deliberately excluded
    // because they may already be represented on one of those VM queues.
    let reclaimable_pages = pages("Pages free")?
        .checked_add(pages("Pages inactive")?)?
        .checked_add(pages("Pages speculative").unwrap_or(0))?;
    reclaimable_pages.checked_mul(page_size)
}

#[cfg(feature = "metal")]
fn metal_memory_snapshot(device: &DeviceProfile) -> Option<(u64, u64, CapacitySource)> {
    let metal = device.device.as_metal_device().ok()?.metal_device();
    let metal_total = u64::try_from(metal.recommended_max_working_set_size()).ok()?;
    let metal_allocated = u64::try_from(metal.current_allocated_size()).ok()?;
    let (host_total, host_available) = host_memory_snapshot()?;
    let (total, available) =
        combine_metal_memory_snapshot(metal_total, metal_allocated, host_total, host_available);
    Some((total, available, CapacitySource::MetalWorkingSet))
}

#[cfg(not(feature = "metal"))]
fn metal_memory_snapshot(_device: &DeviceProfile) -> Option<(u64, u64, CapacitySource)> {
    None
}

fn combine_metal_memory_snapshot(
    metal_total: u64,
    metal_allocated: u64,
    host_total: u64,
    host_available: u64,
) -> (u64, u64) {
    let total = metal_total.min(host_total);
    let metal_available = metal_total.saturating_sub(metal_allocated);
    let available = metal_available.min(host_available).min(total);
    (total, available)
}

#[cfg(feature = "cuda")]
fn cuda_memory_snapshot(device: &DeviceProfile) -> Option<(u64, u64, CapacitySource)> {
    let stream = device.device.as_cuda_device().ok()?.cuda_stream();
    let (available, total) = stream.context().mem_get_info().ok()?;
    Some((
        u64::try_from(total).ok()?,
        u64::try_from(available).ok()?,
        CapacitySource::CudaDriver,
    ))
}

#[cfg(not(feature = "cuda"))]
fn cuda_memory_snapshot(_device: &DeviceProfile) -> Option<(u64, u64, CapacitySource)> {
    None
}

#[derive(Debug)]
struct ExecutionLease {
    coordinator: Arc<InferenceCoordinator>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        if self
            .coordinator
            .active_executions
            .fetch_sub(1, Ordering::AcqRel)
            == 1
        {
            self.coordinator.idle.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ResourceVector;
    use crate::model::ModelVariant;
    use crate::runtime::adapters::{CapabilityKind, LoadedModelBundle, RuntimeAdapterRegistry};

    #[derive(Debug)]
    struct MutableCapacityProvider {
        capacity: ResourceVector,
        available: Mutex<ResourceVector>,
    }

    impl MutableCapacityProvider {
        fn new(capacity: ResourceVector) -> Self {
            Self {
                capacity,
                available: Mutex::new(capacity),
            }
        }

        fn set_available(&self, available: ResourceVector) {
            *self
                .available
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = available;
        }
    }

    impl PhysicalCapacityProvider for MutableCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: *self
                    .available
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()),
                source: CapacitySource::Test,
            }
        }
    }

    fn resources_for_location(location: DeviceLocation, bytes: u64) -> ResourceVector {
        let mut resources = ResourceVector::zero();
        match location {
            DeviceLocation::Cpu => resources.host_bytes = ResourceAmount::Known(bytes),
            DeviceLocation::Metal { .. } => resources.unified_bytes = ResourceAmount::Known(bytes),
            DeviceLocation::Cuda { .. } => resources.device_bytes = ResourceAmount::Known(bytes),
        }
        resources
    }

    fn provider_for_location(location: DeviceLocation, bytes: u64) -> Arc<MutableCapacityProvider> {
        Arc::new(MutableCapacityProvider::new(resources_for_location(
            location, bytes,
        )))
    }

    fn authority_resources_for_location(location: DeviceLocation, bytes: u64) -> ResourceVector {
        match location {
            DeviceLocation::Cpu | DeviceLocation::Metal { .. } => {
                shared_host_unified_vector(ResourceAmount::Known(bytes))
            }
            DeviceLocation::Cuda { .. } => resources_for_location(location, bytes),
        }
    }

    fn cuda_resources(host_bytes: u64, device_bytes: u64) -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(host_bytes),
            device_bytes: ResourceAmount::Known(device_bytes),
            ..ResourceVector::zero()
        }
    }

    fn cuda_provider(host_bytes: u64, device_bytes: u64) -> Arc<MutableCapacityProvider> {
        Arc::new(MutableCapacityProvider::new(cuda_resources(
            host_bytes,
            device_bytes,
        )))
    }

    #[test]
    fn fixed_cuda_capacity_populates_both_physical_memory_domains() {
        let provider = DeviceCapacityProvider::for_tests(BackendKind::Cuda);
        let snapshot = provider.snapshot();
        let fixed = ResourceAmount::Known(64 * 1024 * 1024 * 1024);

        assert_eq!(snapshot.source, CapacitySource::Test);
        assert_eq!(snapshot.capacity.host_bytes, fixed);
        assert_eq!(snapshot.capacity.device_bytes, fixed);
        assert_eq!(snapshot.available.host_bytes, fixed);
        assert_eq!(snapshot.available.device_bytes, fixed);
        assert_eq!(snapshot.capacity.unified_bytes, ResourceAmount::Known(0));
        assert_eq!(snapshot.available.unified_bytes, ResourceAmount::Known(0));
    }

    fn host_capacity_snapshot(bytes: u64) -> PhysicalCapacitySnapshot {
        PhysicalCapacitySnapshot {
            capacity: resources_for_location(DeviceLocation::Cpu, bytes),
            available: resources_for_location(DeviceLocation::Cpu, bytes),
            source: CapacitySource::Test,
        }
    }

    #[test]
    fn device_identity_validation_rejects_backend_kind_or_location_mismatch() {
        for (backend, kind, location) in [
            (BackendKind::Cpu, DeviceKind::Cpu, DeviceLocation::Cpu),
            (
                BackendKind::Metal,
                DeviceKind::Metal,
                DeviceLocation::Metal { gpu_id: 3 },
            ),
            (
                BackendKind::Cuda,
                DeviceKind::Cuda,
                DeviceLocation::Cuda { gpu_id: 5 },
            ),
        ] {
            validate_device_identity_parts(backend, kind, location)
                .expect("consistent device identity must be accepted");
        }

        for (backend, kind, location) in [
            (
                BackendKind::Cpu,
                DeviceKind::Metal,
                DeviceLocation::Metal { gpu_id: 0 },
            ),
            (
                BackendKind::Metal,
                DeviceKind::Cpu,
                DeviceLocation::Metal { gpu_id: 0 },
            ),
            (
                BackendKind::Cuda,
                DeviceKind::Cuda,
                DeviceLocation::Metal { gpu_id: 0 },
            ),
        ] {
            assert!(matches!(
                validate_device_identity_parts(backend, kind, location),
                Err(Error::ConfigError(message))
                    if message.contains("inconsistent inference device identity")
            ));
        }
    }

    #[test]
    fn cpu_resource_estimates_reject_accelerator_memory_domains() {
        let mut resources = ResourceVector::zero();
        resources.host_bytes = ResourceAmount::Known(1);
        resources.device_bytes = ResourceAmount::Known(1);
        assert!(matches!(
            effective_resources(resources, BackendKind::Cpu),
            Err(Error::InvalidInput(message)) if message.contains("nonzero device")
        ));
    }

    #[test]
    fn metal_resource_estimates_reject_separate_host_memory_domains() {
        let mut resources = ResourceVector::zero();
        resources.unified_bytes = ResourceAmount::Known(1);
        resources.host_bytes = ResourceAmount::Known(1);
        assert!(matches!(
            effective_resources(resources, BackendKind::Metal),
            Err(Error::InvalidInput(message)) if message.contains("nonzero host")
        ));
    }

    #[test]
    fn cuda_resource_estimates_reject_unified_memory_domains() {
        let mut resources = ResourceVector::zero();
        resources.device_bytes = ResourceAmount::Known(1);
        resources.unified_bytes = ResourceAmount::Known(1);
        assert!(matches!(
            effective_resources(resources, BackendKind::Cuda),
            Err(Error::InvalidInput(message)) if message.contains("nonzero unified")
        ));
    }

    #[test]
    fn resource_estimates_reject_unaccounted_compute_slots() {
        for compute_slots in [ResourceAmount::Known(1), ResourceAmount::Unknown] {
            let mut resources = ResourceVector::zero();
            resources.host_bytes = ResourceAmount::Known(1);
            resources.compute_slots = compute_slots;
            assert!(matches!(
                effective_resources(resources, BackendKind::Cpu),
                Err(Error::InvalidInput(message)) if message.contains("compute_slots")
            ));
        }
    }

    #[test]
    fn same_physical_device_identity_shares_authority_and_reservations() {
        for (index, location) in [
            DeviceLocation::Cpu,
            DeviceLocation::Metal { gpu_id: 7 },
            DeviceLocation::Cuda { gpu_id: 7 },
        ]
        .into_iter()
        .enumerate()
        {
            let registry = ResourceAuthorityRegistry::default();
            let first = registry
                .authority_for(location, provider_for_location(location, 100))
                .unwrap();
            let second = registry
                .authority_for(location, provider_for_location(location, 200))
                .unwrap();

            assert!(Arc::ptr_eq(&first, &second));
            let _held = first
                .reserve(
                    ReservationOwner::new(
                        ReservationClass::Request,
                        format!("shared-device-first-{index}"),
                    ),
                    resources_for_location(location, 60),
                )
                .unwrap();
            assert!(matches!(
                second.reserve(
                    ReservationOwner::new(
                        ReservationClass::Request,
                        format!("shared-device-second-{index}"),
                    ),
                    resources_for_location(location, 50),
                ),
                Err(Error::Overloaded(_))
            ));
            assert_eq!(
                second.snapshot().physical.capacity,
                authority_resources_for_location(location, 100),
                "the first provider remains authoritative for a shared identity"
            );
        }
    }

    #[test]
    fn cpu_and_metal_locations_share_one_host_unified_authority() {
        let registry = ResourceAuthorityRegistry::default();
        let cpu = registry
            .authority_for(
                DeviceLocation::Cpu,
                provider_for_location(DeviceLocation::Cpu, 100),
            )
            .unwrap();
        let metal = registry
            .authority_for(
                DeviceLocation::Metal { gpu_id: 0 },
                provider_for_location(DeviceLocation::Metal { gpu_id: 0 }, 80),
            )
            .unwrap();
        let other_metal = registry
            .authority_for(
                DeviceLocation::Metal { gpu_id: 1 },
                provider_for_location(DeviceLocation::Metal { gpu_id: 1 }, 90),
            )
            .unwrap();

        assert!(Arc::ptr_eq(&cpu, &metal));
        assert!(Arc::ptr_eq(&cpu, &other_metal));
        assert_eq!(
            cpu.snapshot().physical.capacity,
            shared_host_unified_vector(ResourceAmount::Known(80)),
            "the most conservative live provider controls the shared pool"
        );
        let _cpu_lease = cpu
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "shared-cpu"),
                resources_for_location(DeviceLocation::Cpu, 60),
            )
            .unwrap();
        assert!(matches!(
            metal.reserve(
                ReservationOwner::new(ReservationClass::Request, "shared-metal"),
                resources_for_location(DeviceLocation::Metal { gpu_id: 0 }, 30),
            ),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(
            metal.snapshot().reserved,
            shared_host_unified_vector(ResourceAmount::Known(60))
        );
    }

    #[test]
    fn shared_authority_telemetry_projects_usage_into_each_backend_domain() {
        let registry = ResourceAuthorityRegistry::default();
        let authority = registry
            .authority_for(
                DeviceLocation::Cpu,
                provider_for_location(DeviceLocation::Cpu, 100),
            )
            .unwrap();
        registry
            .authority_for(
                DeviceLocation::Metal { gpu_id: 0 },
                provider_for_location(DeviceLocation::Metal { gpu_id: 0 }, 100),
            )
            .unwrap();
        let cpu = InferenceCoordinator::with_resource_authority(
            BackendKind::Cpu,
            1,
            1,
            authority.clone(),
        );
        let metal = InferenceCoordinator::with_resource_authority(
            BackendKind::Metal,
            1,
            1,
            authority.clone(),
        );
        let _lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "telemetry-alias"),
                resources_for_location(DeviceLocation::Cpu, 40),
            )
            .unwrap();

        let cpu_snapshot = cpu.snapshot();
        assert_eq!(cpu_snapshot.reserved_host_memory_bytes, 40);
        assert_eq!(cpu_snapshot.reserved_unified_memory_bytes, 0);
        let metal_snapshot = metal.snapshot();
        assert_eq!(metal_snapshot.reserved_host_memory_bytes, 0);
        assert_eq!(metal_snapshot.reserved_unified_memory_bytes, 40);
    }

    #[test]
    fn cuda_and_shared_host_unified_locations_remain_exclusive() {
        for shared_location in [DeviceLocation::Cpu, DeviceLocation::Metal { gpu_id: 0 }] {
            for (first_location, second_location) in [
                (shared_location, DeviceLocation::Cuda { gpu_id: 0 }),
                (DeviceLocation::Cuda { gpu_id: 0 }, shared_location),
            ] {
                let registry = ResourceAuthorityRegistry::default();
                let first = registry
                    .authority_for(first_location, provider_for_location(first_location, 100))
                    .unwrap();
                assert!(matches!(
                    registry.authority_for(
                        second_location,
                        provider_for_location(second_location, 100),
                    ),
                    Err(Error::ConfigError(message))
                        if message.contains("shared host-memory domain")
                ));
                drop(first);
                registry
                    .authority_for(second_location, provider_for_location(second_location, 100))
                    .expect("a new domain may register after the active authority expires");
            }
        }
    }

    #[test]
    fn cross_backend_coordinators_cannot_duplicate_host_headroom() {
        let registry = ResourceAuthorityRegistry::default();
        let cpu = registry
            .authority_for(
                DeviceLocation::Cpu,
                provider_for_location(DeviceLocation::Cpu, 100),
            )
            .unwrap();
        let _cpu_lease = cpu
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "cpu-host-claim"),
                resources_for_location(DeviceLocation::Cpu, 60),
            )
            .unwrap();

        assert!(matches!(
            registry.authority_for(
                DeviceLocation::Cuda { gpu_id: 0 },
                cuda_provider(100, 100),
            ),
            Err(Error::ConfigError(message))
                if message.contains("shared host-memory domain")
        ));
        assert_eq!(
            cpu.snapshot().reserved,
            shared_host_unified_vector(ResourceAmount::Known(60))
        );
    }

    #[test]
    fn second_cuda_ordinal_cannot_spend_the_same_host_headroom() {
        let registry = ResourceAuthorityRegistry::default();
        let first_location = DeviceLocation::Cuda { gpu_id: 0 };
        let second_location = DeviceLocation::Cuda { gpu_id: 1 };
        let first = registry
            .authority_for(first_location, cuda_provider(100, 100))
            .unwrap();
        let _first_lease = first
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "cuda-zero-host-claim"),
                cuda_resources(60, 10),
            )
            .unwrap();

        assert!(matches!(
            registry.authority_for(second_location, cuda_provider(100, 100)),
            Err(Error::ConfigError(message))
                if message.contains("shared host-memory domain")
        ));
        assert_eq!(first.snapshot().reserved, cuda_resources(60, 10));
    }

    #[test]
    fn expired_device_authority_is_recreated_with_its_new_provider() {
        for location in [
            DeviceLocation::Cpu,
            DeviceLocation::Metal { gpu_id: 11 },
            DeviceLocation::Cuda { gpu_id: 11 },
        ] {
            let registry = ResourceAuthorityRegistry::default();
            let first = registry
                .authority_for(location, provider_for_location(location, 100))
                .unwrap();
            let weak = Arc::downgrade(&first);
            drop(first);
            assert!(weak.upgrade().is_none());

            let recreated = registry
                .authority_for(location, provider_for_location(location, 200))
                .unwrap();
            assert_eq!(
                recreated.snapshot().physical.capacity,
                authority_resources_for_location(location, 200)
            );
        }
    }

    fn job(id: &str) -> JobSpec {
        let mut resources = ResourceVector::zero();
        resources.host_bytes = ResourceAmount::Known(64 * 1024 * 1024);
        JobSpec {
            request_id: id.to_string(),
            lane: CoordinatorLane::Atomic,
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            deadline: None,
            resources,
        }
    }

    #[tokio::test]
    async fn cuda_snapshot_reports_host_and_device_reservations() {
        let coordinator = Arc::new(InferenceCoordinator::with_resource_authority(
            BackendKind::Cuda,
            1,
            1,
            Arc::new(ResourceAuthority::new(cuda_provider(100, 100))),
        ));
        let spec = JobSpec {
            resources: cuda_resources(7, 11),
            ..job("cuda-domain-telemetry")
        };

        let lease = coordinator.admit(spec).await.expect("CUDA admission");
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.reserved_host_memory_bytes, 7);
        assert_eq!(snapshot.reserved_device_memory_bytes, 11);
        assert_eq!(snapshot.reserved_unified_memory_bytes, 0);
        assert_eq!(snapshot.reserved_memory_bytes, 18);

        drop(lease);
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.reserved_host_memory_bytes, 0);
        assert_eq!(snapshot.reserved_device_memory_bytes, 0);
        assert_eq!(snapshot.reserved_memory_bytes, 0);
    }

    #[test]
    fn macos_vm_stat_parser_uses_reclaimable_pages() {
        let snapshot = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free: 10.\n\
Pages active: 50.\n\
Pages inactive: 20.\n\
Pages speculative: 5.\n\
Pages purgeable: 7.\n";

        assert_eq!(
            parse_macos_vm_stat_available(snapshot),
            Some((10 + 20 + 5) * 16_384)
        );
    }

    #[test]
    fn macos_vm_stat_parser_fails_closed_without_required_fields() {
        let snapshot = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n\
Pages free: 10.\n";

        assert_eq!(parse_macos_vm_stat_available(snapshot), None);
    }

    #[test]
    fn capacity_cache_coalesces_refreshes_while_serving_bounded_stale_data() {
        let started = Instant::now();
        let expected = host_capacity_snapshot(100);
        let cache = CapacitySampleCache::new(Some(expected), started);
        let stale_at = started + CAPACITY_SAMPLE_FRESH_FOR + Duration::from_millis(1);

        let first = cache.decision(stale_at);
        assert_eq!(first.snapshot, Some(expected));
        assert!(first.request_refresh);

        let concurrent = cache.decision(stale_at + Duration::from_millis(1));
        assert_eq!(concurrent.snapshot, Some(expected));
        assert!(!concurrent.request_refresh);
    }

    #[test]
    fn capacity_cache_fails_closed_after_stale_expiry_and_retries_later() {
        let started = Instant::now();
        let expected = host_capacity_snapshot(100);
        let cache = CapacitySampleCache::new(Some(expected), started);
        let first_attempt = started + CAPACITY_SAMPLE_FRESH_FOR + Duration::from_millis(1);
        assert!(cache.decision(first_attempt).request_refresh);
        cache.finish_refresh(None, first_attempt + Duration::from_millis(1));

        let expired_at = started + CAPACITY_SAMPLE_MAX_STALE + Duration::from_millis(1);
        let expired = cache.decision(expired_at);
        assert_eq!(expired.snapshot, None);
        assert!(expired.request_refresh);

        let concurrent = cache.decision(expired_at + Duration::from_millis(1));
        assert_eq!(concurrent.snapshot, None);
        assert!(!concurrent.request_refresh);
    }

    #[test]
    fn older_async_refresh_cannot_overwrite_post_release_sample() {
        let started = Instant::now();
        let initial = host_capacity_snapshot(10);
        let stale_worker_sample = host_capacity_snapshot(20);
        let released = host_capacity_snapshot(100);
        let cache = CapacitySampleCache::new(Some(initial), started);
        let stale_worker_started = started + Duration::from_millis(1);
        let release_sampled_at = started + Duration::from_millis(2);

        cache.publish_sample(released, release_sampled_at);
        cache.finish_refresh(Some(stale_worker_sample), stale_worker_started);

        assert_eq!(
            cache
                .decision(release_sampled_at + Duration::from_millis(1))
                .snapshot,
            Some(released)
        );
    }

    #[test]
    fn capacity_cache_waits_for_successful_refresh_after_hard_stale_expiry() {
        let now = Instant::now();
        let started = now
            .checked_sub(CAPACITY_SAMPLE_MAX_STALE + Duration::from_millis(1))
            .unwrap();
        let initial = host_capacity_snapshot(100);
        let refreshed = host_capacity_snapshot(40);
        let cache = Arc::new(CapacitySampleCache::new(Some(initial), started));
        let decision = cache.decision(now);
        assert_eq!(decision.snapshot, None);
        assert!(decision.request_refresh);

        let worker_cache = cache.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            worker_cache.finish_refresh(Some(refreshed), Instant::now());
        });
        let snapshot = cache.wait_for_refresh(Duration::from_secs(1));
        worker.join().unwrap();

        assert_eq!(snapshot, Some(refreshed));
    }

    #[test]
    fn capacity_cache_failed_refresh_wakes_waiters_and_remains_fail_closed() {
        let now = Instant::now();
        let started = now
            .checked_sub(CAPACITY_SAMPLE_MAX_STALE + Duration::from_millis(1))
            .unwrap();
        let initial = host_capacity_snapshot(100);
        let cache = Arc::new(CapacitySampleCache::new(Some(initial), started));
        let decision = cache.decision(now);
        assert_eq!(decision.snapshot, None);
        assert!(decision.request_refresh);

        let worker_cache = cache.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            worker_cache.finish_refresh(None, Instant::now());
        });
        let snapshot = cache.wait_for_refresh(Duration::from_secs(1));
        worker.join().unwrap();

        assert_eq!(snapshot, None);
    }

    #[test]
    fn capacity_provider_waits_for_queued_refresh_instead_of_false_unavailable() {
        let now = Instant::now();
        let started = now
            .checked_sub(CAPACITY_SAMPLE_MAX_STALE + Duration::from_millis(1))
            .unwrap();
        let initial = host_capacity_snapshot(100);
        let refreshed = host_capacity_snapshot(40);
        let cache = Arc::new(CapacitySampleCache::new(Some(initial), started));
        let (refresh, requests) = std::sync::mpsc::sync_channel(1);
        let worker_cache = cache.clone();
        let worker = std::thread::spawn(move || {
            requests.recv().unwrap();
            std::thread::sleep(Duration::from_millis(10));
            worker_cache.finish_refresh(Some(refreshed), Instant::now());
        });
        let provider = DeviceCapacityProvider {
            probe: DeviceCapacityProbe {
                backend: BackendKind::Cpu,
                device: None,
                configured_cap: None,
                test_capacity: None,
            },
            cache,
            refresh: Some(refresh),
        };

        let snapshot = provider.snapshot();
        worker.join().unwrap();

        assert_eq!(snapshot, refreshed);
    }

    #[test]
    fn successful_capacity_refresh_replaces_the_cached_sample() {
        let started = Instant::now();
        let initial = host_capacity_snapshot(100);
        let refreshed = host_capacity_snapshot(40);
        let cache = CapacitySampleCache::new(Some(initial), started);
        let refresh_at = started + CAPACITY_SAMPLE_FRESH_FOR + Duration::from_millis(1);
        assert!(cache.decision(refresh_at).request_refresh);
        cache.finish_refresh(Some(refreshed), refresh_at + Duration::from_millis(1));

        let decision = cache.decision(refresh_at + Duration::from_millis(2));
        assert_eq!(decision.snapshot, Some(refreshed));
        assert!(!decision.request_refresh);
    }

    #[test]
    fn configured_device_cap_subtracts_memory_already_in_use() {
        let probe = DeviceCapacityProbe {
            backend: BackendKind::Cuda,
            device: None,
            configured_cap: Some(80),
            test_capacity: None,
        };

        assert_eq!(probe.apply_cap(100, 70), (80, 50));
        assert_eq!(probe.apply_cap(100, 10), (80, 0));
        assert_eq!(probe.apply_cap(64, 40), (64, 40));
    }

    #[test]
    fn panicking_capacity_probe_fails_closed_without_escaping_sampler() {
        let sample = guarded_capacity_sample(|| -> Option<PhysicalCapacitySnapshot> {
            panic!("simulated backend probe failure")
        });
        assert_eq!(sample, None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn capacity_probe_command_timeout_kills_slow_child() {
        let output = command_stdout_with_timeout("/bin/sleep", &["1"], Duration::from_millis(10));
        assert_eq!(output, None);
    }

    #[test]
    fn metal_capacity_is_bounded_by_host_memory_pressure() {
        assert_eq!(combine_metal_memory_snapshot(100, 20, 200, 30), (100, 30));
        assert_eq!(combine_metal_memory_snapshot(200, 10, 80, 70), (80, 70));
    }

    #[test]
    fn metal_capacity_is_bounded_by_working_set_pressure() {
        assert_eq!(combine_metal_memory_snapshot(100, 90, 200, 150), (100, 10));
        assert_eq!(combine_metal_memory_snapshot(100, 120, 200, 150), (100, 0));
    }

    #[test]
    fn batch_workspace_is_transient_and_does_not_own_execution_capacity() {
        let coordinator = InferenceCoordinator::new(BackendKind::Cpu, 8, 8);
        let mut resources = ResourceVector::zero();
        resources.temporary_bytes = ResourceAmount::Known(8);

        let workspace = coordinator
            .reserve_batch_workspace(ExecutionGroupId::new(4), BatchId::new(12), resources)
            .expect("batch workspace");

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.reserved_host_memory_bytes, 8);
        assert_eq!(snapshot.active_jobs, 0);
        assert_eq!(snapshot.active_executions, 0);
        drop(workspace);
        assert_eq!(coordinator.snapshot().reserved_memory_bytes, 0);
    }

    #[test]
    fn batch_workspace_rejects_persistent_cache_accounting() {
        let coordinator = InferenceCoordinator::new(BackendKind::Cpu, 1, 1);
        let mut resources = ResourceVector::zero();
        resources.kv_bytes = ResourceAmount::Known(1);

        assert!(matches!(
            coordinator.reserve_batch_workspace(
                ExecutionGroupId::new(1),
                BatchId::new(1),
                resources,
            ),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(coordinator.snapshot().reserved_memory_bytes, 0);
    }

    #[tokio::test]
    async fn queue_is_bounded_and_raii_reconciles_counts() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 1));
        let lease = coordinator.admit(job("first")).await.unwrap();
        assert_eq!(
            coordinator.snapshot().reserved_memory_bytes,
            64 * 1024 * 1024
        );
        assert!(matches!(
            coordinator.admit(job("second")).await,
            Err(Error::Overloaded(_))
        ));
        drop(lease);
        let second = coordinator.admit(job("second")).await.unwrap();
        assert_eq!(coordinator.snapshot().active_jobs, 1);
        drop(second);
        assert_eq!(coordinator.snapshot().active_jobs, 0);
        assert_eq!(coordinator.snapshot().reserved_memory_bytes, 0);

        let parallel = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 2, 1));
        let first = parallel.admit(job("parallel-first")).await.unwrap();
        let second = parallel.admit(job("parallel-second")).await.unwrap();
        assert!(matches!(
            parallel.admit(job("parallel-third")).await,
            Err(Error::Overloaded(_))
        ));
        drop((first, second));
        assert_eq!(parallel.snapshot().active_jobs, 0);
    }

    #[tokio::test]
    async fn rejected_admission_never_starts_expensive_preparation() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 1));
        let active = coordinator.admit(job("active")).await.unwrap();
        let preparation_calls = Arc::new(AtomicUsize::new(0));
        let calls = preparation_calls.clone();

        let result = coordinator
            .admit_then_prepare(job("rejected"), move || async move {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .await;

        assert!(matches!(result, Err(Error::Overloaded(_))));
        assert_eq!(preparation_calls.load(Ordering::Relaxed), 0);
        assert_eq!(coordinator.snapshot().admitted_total, 1);
        drop(active);
    }

    #[tokio::test]
    async fn caller_deadline_covers_preparation_after_admission() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 1));
        let preparation_calls = Arc::new(AtomicUsize::new(0));
        let calls = preparation_calls.clone();
        let mut spec = job("deadline-preparation");
        spec.deadline = Some(Instant::now() + std::time::Duration::from_millis(5));

        let result = coordinator
            .admit_then_prepare(spec, move || async move {
                calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(())
            })
            .await;

        assert!(matches!(result, Err(Error::Timeout(_))));
        assert_eq!(preparation_calls.load(Ordering::Relaxed), 1);
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.active_jobs, 0);
        assert_eq!(snapshot.expired_total, 1);
    }

    #[tokio::test]
    async fn cpu_and_cuda_use_configured_capacity_while_metal_serializes() {
        assert_eq!(
            InferenceCoordinator::new(BackendKind::Cpu, 8, 8).capacity,
            8
        );
        assert_eq!(
            InferenceCoordinator::new(BackendKind::Metal, 8, 8).capacity,
            1
        );
        assert_eq!(
            InferenceCoordinator::new(BackendKind::Cuda, 4, 8).capacity,
            4
        );
    }

    #[tokio::test]
    async fn drain_rejects_new_jobs() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        coordinator.begin_drain();
        assert!(matches!(
            coordinator.admit(job("late")).await,
            Err(Error::Overloaded(_))
        ));
    }

    #[tokio::test]
    async fn drain_waits_for_active_jobs_to_release() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let lease = coordinator.admit(job("active")).await.unwrap();
        coordinator.begin_drain();
        let waiting = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .wait_for_idle(Instant::now() + std::time::Duration::from_secs(1))
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());

        drop(lease);

        waiting.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn drain_waits_for_cold_model_load_and_rejects_new_loads() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let load = coordinator
            .begin_model_load("model-load:test")
            .expect("model load admitted before drain");
        assert_eq!(coordinator.snapshot().active_model_loads, 1);

        coordinator.begin_drain();
        let short_deadline = Instant::now() + std::time::Duration::from_millis(5);
        assert!(matches!(
            coordinator.wait_for_idle(short_deadline).await,
            Err(Error::Timeout(_))
        ));
        assert!(matches!(
            coordinator.begin_model_load("model-load:late"),
            Err(Error::Overloaded(_))
        ));

        drop(load);
        coordinator
            .wait_for_idle(Instant::now() + std::time::Duration::from_secs(1))
            .await
            .expect("load release must unblock drain");
        assert_eq!(coordinator.snapshot().active_model_loads, 0);
    }

    #[tokio::test]
    async fn execution_deadline_is_counted() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let _held = coordinator.acquire_execution(None).await.unwrap();

        let result = coordinator
            .acquire_execution(Some(Instant::now() + std::time::Duration::from_millis(5)))
            .await;

        assert!(matches!(result, Err(Error::Timeout(_))));
        assert_eq!(coordinator.snapshot().expired_total, 1);
    }

    #[tokio::test]
    async fn cuda_execution_uses_configured_concurrency() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cuda, 3, 4));
        let first = coordinator.acquire_execution(None).await.unwrap();
        let second = coordinator.acquire_execution(None).await.unwrap();
        let third = coordinator.acquire_execution(None).await.unwrap();

        assert_eq!(coordinator.snapshot().active_executions, 3);
        let blocked = coordinator
            .acquire_execution(Some(Instant::now() + std::time::Duration::from_millis(5)))
            .await;
        assert!(matches!(blocked, Err(Error::Timeout(_))));

        drop((first, second, third));
        assert_eq!(coordinator.snapshot().active_executions, 0);
    }

    #[tokio::test]
    async fn execution_unit_requests_are_bounded_by_backend_capacity() {
        let cpu = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 8, 8));
        assert!(matches!(
            cpu.acquire_execution_units(0, None).await,
            Err(Error::InvalidInput(_))
        ));
        let cpu_lease = cpu.acquire_execution_units(2, None).await.unwrap();
        assert_eq!(cpu.snapshot().active_executions, 1);
        drop(cpu_lease);
        assert!(matches!(
            cpu.acquire_execution_units(9, None).await,
            Err(Error::InvalidInput(_))
        ));

        let metal = Arc::new(InferenceCoordinator::new(BackendKind::Metal, 8, 8));
        assert!(matches!(
            metal.acquire_execution_units(2, None).await,
            Err(Error::InvalidInput(_))
        ));

        let cuda = Arc::new(InferenceCoordinator::new(BackendKind::Cuda, 4, 8));
        let lease = cuda.acquire_execution_units(4, None).await.unwrap();
        assert_eq!(cuda.snapshot().active_executions, 1);
        drop(lease);
        assert_eq!(cuda.snapshot().active_executions, 0);
    }

    #[tokio::test]
    async fn engine_step_reserves_all_cuda_execution_units() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cuda, 4, 8));
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let runner_coordinator = coordinator.clone();
        let runner = tokio::spawn(async move {
            runner_coordinator
                .run_engine_step(async move {
                    release_rx.await.unwrap();
                    Ok(())
                })
                .await
        });
        while coordinator.snapshot().active_executions == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(coordinator.snapshot().active_executions, 1);

        let competing = coordinator
            .acquire_execution(Some(Instant::now() + std::time::Duration::from_millis(5)))
            .await;
        assert!(matches!(competing, Err(Error::Timeout(_))));

        release_tx.send(()).unwrap();
        runner.await.unwrap().unwrap();
        coordinator.acquire_execution(None).await.unwrap();
    }

    #[tokio::test]
    async fn loaded_scalar_stage_accepts_parallel_row_capacity_and_rejects_cross_group_work() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let job = coordinator.admit(job("loaded-scalar")).await.unwrap();
        let adapters = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 4).unwrap();
        let bundle = LoadedModelBundle::bind(
            &adapters,
            coordinator.execution_group_id(),
            crate::engine::ModelInstanceId::new(1),
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .unwrap();
        let contract = bundle.contract(CapabilityKind::Tts, false).unwrap();
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
        assert_eq!(contract.stages[0].max_batch_size, 4);
        let calls = Arc::new(AtomicUsize::new(0));
        let task_calls = calls.clone();

        let output = coordinator
            .run_loaded_blocking_stage(
                &job,
                contract,
                WorkUnit::AtomicJob {
                    kind: "tts".to_string(),
                },
                move || {
                    task_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(7usize)
                },
            )
            .await
            .unwrap();
        assert_eq!(output, 7);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let wrong_group = LoadedModelBundle::bind(
            &adapters,
            ExecutionGroupId::new(coordinator.execution_group_id().get() + 1),
            crate::engine::ModelInstanceId::new(1),
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .unwrap();
        let contract = wrong_group.contract(CapabilityKind::Tts, false).unwrap();
        let task_calls = calls.clone();
        let error = coordinator
            .run_loaded_blocking_stage(
                &job,
                contract,
                WorkUnit::AtomicJob {
                    kind: "tts".to_string(),
                },
                move || {
                    task_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                },
            )
            .await
            .expect_err("cross-group contract must fail closed");

        assert!(error.to_string().contains("different execution group"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(coordinator.snapshot().active_executions, 0);
    }

    #[tokio::test]
    async fn cloned_job_scope_keeps_one_admission_until_the_last_stage_releases() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let parent = coordinator.admit(job("pipeline")).await.unwrap();
        let stage = parent.clone();
        assert_eq!(coordinator.snapshot().active_jobs, 1);
        assert_eq!(
            coordinator.snapshot().reserved_memory_bytes,
            64 * 1024 * 1024
        );

        drop(parent);
        assert_eq!(coordinator.snapshot().active_jobs, 1);
        drop(stage);
        assert_eq!(coordinator.snapshot().active_jobs, 0);
        assert_eq!(coordinator.snapshot().reserved_memory_bytes, 0);
    }

    #[tokio::test]
    async fn materialized_job_usage_is_backend_aware_and_not_counted_twice() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut capacity = ResourceVector::zero();
            let mut available = ResourceVector::zero();
            let mut first_resources = ResourceVector::zero();
            let mut second_resources = ResourceVector::zero();
            let observation = match backend {
                BackendKind::Cpu => {
                    capacity.host_bytes = ResourceAmount::Known(100);
                    available.host_bytes = ResourceAmount::Known(85);
                    first_resources.host_bytes = ResourceAmount::Known(10);
                    second_resources.host_bytes = ResourceAmount::Known(85);
                    JobResourceObservation::new(10, 5)
                }
                BackendKind::Metal => {
                    capacity.unified_bytes = ResourceAmount::Known(100);
                    available.unified_bytes = ResourceAmount::Known(85);
                    first_resources.unified_bytes = ResourceAmount::Known(10);
                    second_resources.unified_bytes = ResourceAmount::Known(85);
                    JobResourceObservation::new(10, 5)
                }
                BackendKind::Cuda => {
                    capacity.host_bytes = ResourceAmount::Known(100);
                    capacity.device_bytes = ResourceAmount::Known(100);
                    available.host_bytes = ResourceAmount::Known(96);
                    available.device_bytes = ResourceAmount::Known(85);
                    first_resources.host_bytes = ResourceAmount::Known(4);
                    first_resources.device_bytes = ResourceAmount::Known(10);
                    second_resources.host_bytes = ResourceAmount::Known(96);
                    second_resources.device_bytes = ResourceAmount::Known(85);
                    JobResourceObservation::new(4, 15)
                }
            };
            first_resources.kv_bytes = ResourceAmount::Known(2);
            first_resources.temporary_bytes = ResourceAmount::Known(3);
            let first_spec = JobSpec {
                resources: first_resources,
                ..job("materialized")
            };
            let second_spec = JobSpec {
                resources: second_resources,
                ..job("fitting")
            };

            let pending_provider = Arc::new(MutableCapacityProvider::new(capacity));
            pending_provider.set_available(available);
            let pending_coordinator = Arc::new(InferenceCoordinator::with_resource_authority(
                backend,
                2,
                4,
                Arc::new(ResourceAuthority::new(pending_provider)),
            ));
            let pending = pending_coordinator.admit(first_spec.clone()).await.unwrap();
            assert!(matches!(
                pending_coordinator.admit(second_spec.clone()).await,
                Err(Error::Overloaded(_))
            ));
            drop(pending);

            let provider = Arc::new(MutableCapacityProvider::new(capacity));
            provider.set_available(available);
            let coordinator = Arc::new(InferenceCoordinator::with_resource_authority(
                backend,
                2,
                4,
                Arc::new(ResourceAuthority::new(provider)),
            ));
            let first = coordinator
                .admit_observed(first_spec, observation)
                .await
                .unwrap();
            let second = coordinator.admit(second_spec).await.unwrap();
            drop((second, first));
        }
    }

    #[tokio::test]
    async fn initial_materialization_is_atomic_with_live_headroom_admission() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut capacity = ResourceVector::zero();
            let mut available = ResourceVector::zero();
            let mut resources = ResourceVector::zero();
            let observation = match backend {
                BackendKind::Cpu => {
                    capacity.host_bytes = ResourceAmount::Known(100);
                    available.host_bytes = ResourceAmount::Known(85);
                    resources.host_bytes = ResourceAmount::Known(100);
                    JobResourceObservation::new(10, 5)
                }
                BackendKind::Metal => {
                    capacity.unified_bytes = ResourceAmount::Known(100);
                    available.unified_bytes = ResourceAmount::Known(85);
                    resources.unified_bytes = ResourceAmount::Known(100);
                    JobResourceObservation::new(10, 5)
                }
                BackendKind::Cuda => {
                    capacity.host_bytes = ResourceAmount::Known(100);
                    capacity.device_bytes = ResourceAmount::Known(100);
                    available.host_bytes = ResourceAmount::Known(96);
                    available.device_bytes = ResourceAmount::Known(85);
                    resources.host_bytes = ResourceAmount::Known(100);
                    resources.device_bytes = ResourceAmount::Known(100);
                    JobResourceObservation::new(4, 15)
                }
            };
            let spec = JobSpec {
                resources,
                ..job("preexisting-input")
            };

            let pending_provider = Arc::new(MutableCapacityProvider::new(capacity));
            pending_provider.set_available(available);
            let pending_coordinator = Arc::new(InferenceCoordinator::with_resource_authority(
                backend,
                2,
                4,
                Arc::new(ResourceAuthority::new(pending_provider)),
            ));
            assert!(matches!(
                pending_coordinator.admit(spec.clone()).await,
                Err(Error::Overloaded(_))
            ));

            let observed_provider = Arc::new(MutableCapacityProvider::new(capacity));
            observed_provider.set_available(available);
            let observed_coordinator = Arc::new(InferenceCoordinator::with_resource_authority(
                backend,
                2,
                4,
                Arc::new(ResourceAuthority::new(observed_provider)),
            ));
            let lease = observed_coordinator
                .admit_observed(spec, observation)
                .await
                .unwrap();
            drop(lease);
        }
    }

    #[tokio::test]
    async fn job_observation_cannot_expand_immutable_authorization() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let invalid = coordinator
            .admit_observed(
                job("invalid-initial"),
                JobResourceObservation::host(65 * 1024 * 1024),
            )
            .await;
        assert!(matches!(invalid, Err(Error::InvalidInput(_))));
        assert_eq!(coordinator.snapshot().active_jobs, 0);
        assert_eq!(coordinator.resource_authority().snapshot().reservations, 0);

        let job = coordinator.admit(job("immutable-update")).await.unwrap();

        assert!(matches!(
            job.record_materialized_usage(JobResourceObservation::host(65 * 1024 * 1024)),
            Err(Error::InferenceError(_))
        ));
        assert_eq!(
            job.spec.resources.host_bytes,
            ResourceAmount::Known(64 * 1024 * 1024)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_stage_deadline_keeps_tokio_responsive_and_retains_leases() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let mut spec = job("blocking-deadline");
        spec.deadline = Some(Instant::now() + std::time::Duration::from_millis(250));
        let job = coordinator.admit(spec).await.unwrap();
        let runner_job = job.clone();
        drop(job);
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let task_release = release.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task_coordinator = coordinator.clone();
        let runner = tokio::spawn(async move {
            task_coordinator
                .run_blocking_stage(&runner_job, move || {
                    let _ = started_tx.send(());
                    let (lock, wake) = &*task_release;
                    let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|poison| poison.into_inner());
                    }
                    Ok(())
                })
                .await
        });

        started_rx.await.unwrap();
        let heartbeat = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            tokio::time::sleep(std::time::Duration::from_millis(5)),
        )
        .await;
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), runner).await;
        let while_blocked = coordinator.snapshot();

        {
            let (lock, wake) = &*release;
            *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
            wake.notify_all();
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while coordinator.snapshot().active_executions != 0
                || coordinator.snapshot().active_jobs != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert!(heartbeat.is_ok());
        assert!(matches!(result, Ok(Ok(Err(Error::Timeout(_))))));
        assert_eq!(while_blocked.active_executions, 1);
        assert_eq!(while_blocked.active_jobs, 1);
        assert_eq!(coordinator.snapshot().expired_total, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn host_blocking_work_does_not_consume_device_execution_capacity() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let host_job = coordinator.admit(job("host-work")).await.unwrap();
        let device_job = coordinator.admit(job("device-work")).await.unwrap();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let task_release = release.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task_coordinator = coordinator.clone();
        let task_job = host_job.clone();
        let host = tokio::spawn(async move {
            task_coordinator
                .run_host_blocking_stage(&task_job, move || {
                    let _ = started_tx.send(());
                    let (lock, wake) = &*task_release;
                    let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|poison| poison.into_inner());
                    }
                    Ok(())
                })
                .await
        });

        started_rx.await.unwrap();
        assert_eq!(coordinator.snapshot().active_executions, 0);
        tokio::time::timeout(
            Duration::from_millis(100),
            coordinator.run_blocking_stage(&device_job, || Ok(())),
        )
        .await
        .expect("device work should not wait for host preparation")
        .unwrap();
        assert_eq!(coordinator.snapshot().active_executions, 0);

        let (lock, wake) = &*release;
        *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
        host.await.unwrap().unwrap();
        drop((host_job, device_job));
        assert_eq!(coordinator.snapshot().active_jobs, 0);
    }

    #[test]
    fn queued_blocking_stage_rechecks_deadline_before_calling_model_work() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let blocker_release = Arc::new((Mutex::new(false), Condvar::new()));
            let task_release = blocker_release.clone();
            let (blocker_started_tx, blocker_started_rx) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                let _ = blocker_started_tx.send(());
                let (lock, wake) = &*task_release;
                let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                while !*released {
                    released = wake
                        .wait(released)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
            });
            blocker_started_rx.await.unwrap();

            let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
            let mut spec = job("queued-blocking-deadline");
            spec.deadline = Some(Instant::now() + std::time::Duration::from_millis(50));
            let job = coordinator.admit(spec).await.unwrap();
            let runner_job = job.clone();
            drop(job);
            let called = Arc::new(AtomicUsize::new(0));
            let task_called = called.clone();
            let task_coordinator = coordinator.clone();
            let runner = tokio::spawn(async move {
                task_coordinator
                    .run_blocking_stage(&runner_job, move || {
                        task_called.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    })
                    .await
            });

            let result = tokio::time::timeout(std::time::Duration::from_secs(1), runner)
                .await
                .unwrap()
                .unwrap();
            let while_queued = coordinator.snapshot();
            {
                let (lock, wake) = &*blocker_release;
                *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
                wake.notify_all();
            }
            blocker.await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while coordinator.snapshot().active_executions != 0
                    || coordinator.snapshot().active_jobs != 0
                {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();

            assert!(matches!(result, Err(Error::Timeout(id)) if id == "queued-blocking-deadline"));
            assert_eq!(while_queued.active_executions, 1);
            assert_eq!(while_queued.active_jobs, 1);
            assert_eq!(called.load(Ordering::Relaxed), 0);
            assert_eq!(coordinator.snapshot().expired_total, 1);
        });
    }

    #[tokio::test]
    async fn blocking_stage_maps_worker_panics_and_releases_execution() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let job = coordinator.admit(job("blocking-panic")).await.unwrap();

        let result = coordinator
            .run_blocking_stage::<(), _>(&job, || panic!("test blocking panic"))
            .await;

        assert!(matches!(
            result,
            Err(Error::InferenceError(message))
                if message.contains("blocking inference task failed")
        ));
        assert_eq!(coordinator.snapshot().active_executions, 0);
        drop(job);
        assert_eq!(coordinator.snapshot().active_jobs, 0);
    }

    #[tokio::test]
    async fn ordering_wait_timeout_is_counted_once_with_request_identity() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let held = gate.clone().lock_owned().await;
        let mut spec = job("ordering-deadline");
        spec.deadline = Some(Instant::now() + std::time::Duration::from_millis(20));
        let job = coordinator.admit(spec).await.unwrap();

        let result = coordinator.acquire_job_ordering(&job, gate).await;

        assert!(matches!(result, Err(Error::Timeout(id)) if id == "ordering-deadline"));
        assert_eq!(coordinator.snapshot().expired_total, 1);
        drop(held);
    }

    #[tokio::test]
    async fn expired_atomic_stage_retains_execution_until_physical_work_finishes() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let mut spec = job("atomic");
        spec.deadline = Some(Instant::now() + std::time::Duration::from_millis(5));
        let job = coordinator.admit(spec).await.unwrap();
        let running = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .run_stage(&job, async {
                        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                        Ok(())
                    })
                    .await
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert_eq!(coordinator.snapshot().active_executions, 1);
        assert!(matches!(running.await.unwrap(), Err(Error::Timeout(_))));
        assert_eq!(coordinator.snapshot().active_executions, 0);
    }

    #[tokio::test]
    async fn resource_rejection_is_counted() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let mut oversized = job("oversized");
        oversized.resources.host_bytes = ResourceAmount::Known(u64::MAX);

        assert!(coordinator.admit(oversized).await.is_err());
        assert_eq!(coordinator.snapshot().rejected_total, 1);
    }
}
