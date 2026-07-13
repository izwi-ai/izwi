//! Global inference admission and device-execution coordination.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::Notify;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use serde::Serialize;

use crate::backends::BackendKind;
use crate::engine::{
    Priority, ReservationId, ResourceAmount, ResourceEstimate, ResourceLedger, ResourceVector,
    WorkloadClass,
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
    resources: Mutex<ResourceLedger>,
    admission_gate: Mutex<()>,
    idle: Notify,
    active_jobs: AtomicUsize,
    active_executions: AtomicUsize,
    admitted_total: AtomicU64,
    rejected_total: AtomicU64,
    expired_total: AtomicU64,
    draining: AtomicBool,
}

impl InferenceCoordinator {
    pub fn new(backend: BackendKind, execution_parallelism: usize, max_queued_jobs: usize) -> Self {
        let capacity = match backend {
            BackendKind::Cpu | BackendKind::Metal => 1,
            BackendKind::Cuda => execution_parallelism.max(1),
        };
        Self {
            capacity,
            backend,
            jobs: Arc::new(Semaphore::new(max_queued_jobs.max(capacity).max(1))),
            execution: Arc::new(Semaphore::new(capacity)),
            resources: Mutex::new(ResourceLedger::new(resource_capacity(backend))),
            admission_gate: Mutex::new(()),
            idle: Notify::new(),
            active_jobs: AtomicUsize::new(0),
            active_executions: AtomicUsize::new(0),
            admitted_total: AtomicU64::new(0),
            rejected_total: AtomicU64::new(0),
            expired_total: AtomicU64::new(0),
            draining: AtomicBool::new(false),
        }
    }

    pub fn snapshot(&self) -> CoordinatorSnapshot {
        let reserved_memory_bytes = self
            .resources
            .lock()
            .ok()
            .map(|ledger| memory_bytes(ledger.used(), self.backend))
            .unwrap_or_default();
        CoordinatorSnapshot {
            capacity: self.capacity,
            active_jobs: self.active_jobs.load(Ordering::Relaxed),
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
        let effective_resources = effective_resources(spec.resources, self.backend);
        let reservation = self
            .resources
            .lock()
            .map_err(|_| Error::InferenceError("resource ledger mutex poisoned".to_string()))?
            .reserve(effective_resources)
            .map_err(|err| {
                self.rejected_total.fetch_add(1, Ordering::Relaxed);
                err
            })?;
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
        Ok(JobLease {
            coordinator: self.clone(),
            _permit: permit,
            reservation: reservation.id,
            spec,
        })
    }

    pub async fn acquire_execution(
        self: &Arc<Self>,
        deadline: Option<Instant>,
    ) -> Result<ExecutionLease> {
        self.acquire_execution_units(1, deadline).await
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
        let deadline = spec.deadline;
        let _job = self.admit(spec).await?;
        let _execution = self.acquire_execution(deadline).await?;
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), future)
                .await
                .map_err(|_| {
                    self.expired_total.fetch_add(1, Ordering::Relaxed);
                    Error::Timeout("direct inference job".to_string())
                })?,
            None => future.await,
        }
    }
}

#[derive(Debug)]
pub struct JobLease {
    coordinator: Arc<InferenceCoordinator>,
    _permit: OwnedSemaphorePermit,
    reservation: ReservationId,
    pub spec: JobSpec,
}

impl Drop for JobLease {
    fn drop(&mut self) {
        if let Ok(mut resources) = self.coordinator.resources.lock() {
            let _ = resources.release(self.reservation);
        }
        if self.coordinator.active_jobs.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.coordinator.idle.notify_waiters();
        }
    }
}

const FALLBACK_JOB_MEMORY_BYTES: u64 = 256 * 1024 * 1024;

fn env_budget(name: &str, fallback: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

fn resource_capacity(backend: BackendKind) -> ResourceVector {
    let mut capacity = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            capacity.host_bytes = ResourceAmount::Known(env_budget(
                "IZWI_CPU_MEMORY_BUDGET_BYTES",
                8 * 1024 * 1024 * 1024,
            ));
        }
        BackendKind::Metal => {
            capacity.unified_bytes = ResourceAmount::Known(env_budget(
                "IZWI_METAL_MEMORY_BUDGET_BYTES",
                6 * 1024 * 1024 * 1024,
            ));
        }
        BackendKind::Cuda => {
            capacity.device_bytes = ResourceAmount::Known(env_budget(
                "IZWI_CUDA_MEMORY_BUDGET_BYTES",
                8 * 1024 * 1024 * 1024,
            ));
        }
    }
    capacity
}

fn effective_resources(requested: ResourceVector, backend: BackendKind) -> ResourceVector {
    let requested_memory = match backend {
        BackendKind::Cpu => requested.host_bytes,
        BackendKind::Metal => requested.unified_bytes,
        BackendKind::Cuda => requested.device_bytes,
    };
    let effective_memory = match requested_memory {
        ResourceAmount::Known(value) => ResourceAmount::Known(value),
        ResourceAmount::Unknown => ResourceAmount::Known(FALLBACK_JOB_MEMORY_BYTES),
    };
    let mut effective = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => effective.host_bytes = effective_memory,
        BackendKind::Metal => effective.unified_bytes = effective_memory,
        BackendKind::Cuda => effective.device_bytes = effective_memory,
    }
    effective
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
        JobSpec {
            request_id: id.to_string(),
            lane: CoordinatorLane::Atomic,
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            deadline: None,
            resources: ResourceVector::default(),
        }
    }

    #[tokio::test]
    async fn queue_is_bounded_and_raii_reconciles_counts() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 8, 1));
        let lease = coordinator.admit(job("first")).await.unwrap();
        assert_eq!(
            coordinator.snapshot().reserved_memory_bytes,
            FALLBACK_JOB_MEMORY_BYTES
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
    async fn resource_rejection_is_counted() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let mut oversized = job("oversized");
        oversized.resources.host_bytes = ResourceAmount::Known(u64::MAX);

        assert!(coordinator.admit(oversized).await.is_err());
        assert_eq!(coordinator.snapshot().rejected_total, 1);
    }
}
