//! Global inference admission and device-execution coordination.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

use tokio::sync::Notify;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use serde::Serialize;

use crate::backends::{BackendKind, DeviceProfile};
use crate::engine::{
    CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot, Priority, ReservationClass,
    ReservationOwner, ResourceAmount, ResourceAuthority, ResourceEstimate, ResourceLease,
    ResourceVector, WorkloadClass,
};
use crate::error::{Error, Result};

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CoordinatorSnapshot {
    pub capacity: usize,
    pub active_jobs: usize,
    pub active_model_loads: usize,
    pub active_executions: usize,
    pub reserved_memory_bytes: u64,
    pub admitted_total: u64,
    pub rejected_total: u64,
    pub expired_total: u64,
    pub draining: bool,
}

#[derive(Debug)]
pub struct InferenceCoordinator {
    capacity: usize,
    backend: BackendKind,
    jobs: Arc<Semaphore>,
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
        let provider = Arc::new(DeviceCapacityProvider::new(backend, device)?);
        let resources = shared_resource_authority(backend, provider);
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
            BackendKind::Cpu | BackendKind::Metal => 1,
            BackendKind::Cuda => execution_parallelism.max(1),
        };
        Self {
            capacity,
            backend,
            jobs: Arc::new(Semaphore::new(max_queued_jobs.max(capacity).max(1))),
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

    pub fn snapshot(&self) -> CoordinatorSnapshot {
        let reserved_memory_bytes = memory_bytes(self.resources.snapshot().reserved, self.backend);
        CoordinatorSnapshot {
            capacity: self.capacity,
            active_jobs: self.active_jobs.load(Ordering::Relaxed),
            active_model_loads: self.active_model_loads.load(Ordering::Relaxed),
            active_executions: self.active_executions.load(Ordering::Relaxed),
            reserved_memory_bytes,
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
        let reservation = self
            .resources
            .reserve(
                ReservationOwner::new(ReservationClass::Request, spec.request_id.clone()),
                effective_resources,
            )
            .map_err(|err| {
                self.rejected_total.fetch_add(1, Ordering::Relaxed);
                err
            })?;
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
        Ok(JobLease {
            _inner: Arc::new(JobLeaseInner {
                coordinator: self.clone(),
                _permit: permit,
                _reservation: reservation,
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

    pub async fn acquire_execution(
        self: &Arc<Self>,
        deadline: Option<Instant>,
    ) -> Result<ExecutionLease> {
        self.acquire_execution_units(1, deadline).await
    }

    /// Reserve the complete backend execution budget for one scheduler step.
    /// A CUDA step may fan out across `request_parallelism` worker threads, so
    /// holding a single permit would allow unrelated direct work to exceed the
    /// configured device concurrency. CPU and Metal have capacity one.
    pub async fn acquire_engine_step(self: &Arc<Self>) -> Result<ExecutionLease> {
        self.acquire_execution_units(self.capacity, None).await
    }

    pub async fn acquire_execution_units(
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

    pub async fn run_direct<T, F>(self: &Arc<Self>, spec: JobSpec, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let job = self.admit(spec).await?;
        self.run_stage(&job, future).await
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

    pub async fn run_stage<T, F>(self: &Arc<Self>, job: &JobLease, future: F) -> Result<T>
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
}

fn shared_resource_authority(
    backend: BackendKind,
    provider: Arc<dyn PhysicalCapacityProvider>,
) -> Arc<ResourceAuthority> {
    static AUTHORITIES: OnceLock<Mutex<HashMap<BackendKind, Weak<ResourceAuthority>>>> =
        OnceLock::new();
    let mut authorities = AUTHORITIES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some(authority) = authorities.get(&backend).and_then(Weak::upgrade) {
        return authority;
    }
    let authority = Arc::new(ResourceAuthority::new(provider));
    authorities.insert(backend, Arc::downgrade(&authority));
    authority
}

#[derive(Debug, Clone)]
pub struct JobLease {
    _inner: Arc<JobLeaseInner>,
    pub spec: JobSpec,
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
    _reservation: ResourceLease,
}

impl Drop for JobLeaseInner {
    fn drop(&mut self) {
        if self.coordinator.active_jobs.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.coordinator.idle.notify_waiters();
        }
    }
}

fn effective_resources(requested: ResourceVector, backend: BackendKind) -> Result<ResourceVector> {
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

fn known_or_zero(amount: ResourceAmount) -> Result<u64> {
    match amount {
        ResourceAmount::Known(value) => Ok(value),
        ResourceAmount::Unknown => Err(Error::InvalidInput(
            "request resource estimate contains an unresolved quantity".to_string(),
        )),
    }
}

fn memory_bytes(resources: ResourceVector, backend: BackendKind) -> u64 {
    let amount = match backend {
        BackendKind::Cpu => resources.host_bytes,
        BackendKind::Metal => resources.unified_bytes,
        BackendKind::Cuda => resources.device_bytes,
    };
    match amount {
        ResourceAmount::Known(value) => value,
        ResourceAmount::Unknown => 0,
    }
}

#[derive(Debug)]
struct DeviceCapacityProvider {
    backend: BackendKind,
    device: Option<DeviceProfile>,
    configured_cap: Option<u64>,
    test_capacity: Option<u64>,
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
        Ok(Self {
            backend,
            device: Some(device),
            configured_cap,
            test_capacity: cfg!(test).then_some(1024 * 1024 * 1024 * 1024),
        })
    }

    #[cfg(test)]
    fn for_tests(backend: BackendKind) -> Self {
        Self {
            backend,
            device: None,
            configured_cap: None,
            test_capacity: Some(64 * 1024 * 1024 * 1024),
        }
    }

    fn apply_cap(&self, total: u64, available: u64) -> (u64, u64) {
        match self.configured_cap {
            Some(cap) => (total.min(cap), available.min(cap)),
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
}

impl PhysicalCapacityProvider for DeviceCapacityProvider {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        let Some((total, available, source)) = self.observed_capacity() else {
            return PhysicalCapacitySnapshot {
                capacity: self.vector(ResourceAmount::Unknown),
                available: self.vector(ResourceAmount::Unknown),
                source: CapacitySource::Unavailable,
            };
        };
        let (capacity, available) = self.apply_cap(total, available);
        let mut capacity_vector = self.vector(ResourceAmount::Known(capacity));
        let mut available_vector = self.vector(ResourceAmount::Known(available));
        if self.backend == BackendKind::Cuda {
            let Some((host_total, host_available)) = host_memory_snapshot() else {
                return PhysicalCapacitySnapshot {
                    capacity: self.vector(ResourceAmount::Unknown),
                    available: self.vector(ResourceAmount::Unknown),
                    source: CapacitySource::Unavailable,
                };
            };
            capacity_vector.host_bytes = ResourceAmount::Known(host_total);
            available_vector.host_bytes = ResourceAmount::Known(host_available);
        }
        PhysicalCapacitySnapshot {
            capacity: capacity_vector,
            available: available_vector,
            source,
        }
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
        let total_output = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !total_output.status.success() {
            return None;
        }
        let total = std::str::from_utf8(&total_output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        let vm_output = std::process::Command::new("/usr/bin/vm_stat")
            .output()
            .ok()?;
        if !vm_output.status.success() {
            return None;
        }
        let available =
            parse_macos_vm_stat_available(std::str::from_utf8(&vm_output.stdout).ok()?)?;
        return Some((total, available.min(total)));
    }
    #[allow(unreachable_code)]
    None
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
pub struct ExecutionLease {
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
    fn metal_capacity_is_bounded_by_host_memory_pressure() {
        assert_eq!(combine_metal_memory_snapshot(100, 20, 200, 30), (100, 30));
        assert_eq!(combine_metal_memory_snapshot(200, 10, 80, 70), (80, 70));
    }

    #[test]
    fn metal_capacity_is_bounded_by_working_set_pressure() {
        assert_eq!(combine_metal_memory_snapshot(100, 90, 200, 150), (100, 10));
        assert_eq!(combine_metal_memory_snapshot(100, 120, 200, 150), (100, 0));
    }

    #[tokio::test]
    async fn queue_is_bounded_and_raii_reconciles_counts() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 8, 1));
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
    async fn cpu_and_metal_serialize_while_cuda_uses_configured_capacity() {
        assert_eq!(
            InferenceCoordinator::new(BackendKind::Cpu, 8, 8).capacity,
            1
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
        assert!(matches!(
            cpu.acquire_execution_units(2, None).await,
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
        let engine_step = coordinator.acquire_engine_step().await.unwrap();
        assert_eq!(coordinator.snapshot().active_executions, 1);

        let competing = coordinator
            .acquire_execution(Some(Instant::now() + std::time::Duration::from_millis(5)))
            .await;
        assert!(matches!(competing, Err(Error::Timeout(_))));

        drop(engine_step);
        coordinator.acquire_execution(None).await.unwrap();
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
