//! Global inference admission and device-execution coordination.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::backends::BackendKind;
use crate::engine::{Priority, ResourceEstimate, WorkloadClass};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorSnapshot {
    pub capacity: usize,
    pub active_jobs: usize,
    pub active_executions: usize,
    pub admitted_total: u64,
    pub rejected_total: u64,
    pub expired_total: u64,
    pub draining: bool,
}

#[derive(Debug)]
pub struct InferenceCoordinator {
    capacity: usize,
    jobs: Arc<Semaphore>,
    execution: Arc<Semaphore>,
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
            jobs: Arc::new(Semaphore::new(max_queued_jobs.max(capacity).max(1))),
            execution: Arc::new(Semaphore::new(capacity)),
            active_jobs: AtomicUsize::new(0),
            active_executions: AtomicUsize::new(0),
            admitted_total: AtomicU64::new(0),
            rejected_total: AtomicU64::new(0),
            expired_total: AtomicU64::new(0),
            draining: AtomicBool::new(false),
        }
    }

    pub fn snapshot(&self) -> CoordinatorSnapshot {
        CoordinatorSnapshot {
            capacity: self.capacity,
            active_jobs: self.active_jobs.load(Ordering::Relaxed),
            active_executions: self.active_executions.load(Ordering::Relaxed),
            admitted_total: self.admitted_total.load(Ordering::Relaxed),
            rejected_total: self.rejected_total.load(Ordering::Relaxed),
            expired_total: self.expired_total.load(Ordering::Relaxed),
            draining: self.draining.load(Ordering::Acquire),
        }
    }

    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Release);
    }

    pub async fn admit(self: &Arc<Self>, spec: JobSpec) -> Result<JobLease> {
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
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
        Ok(JobLease {
            coordinator: self.clone(),
            _permit: permit,
            spec,
        })
    }

    pub async fn acquire_execution(
        self: &Arc<Self>,
        deadline: Option<Instant>,
    ) -> Result<ExecutionLease> {
        let acquire = self
            .execution
            .clone()
            .acquire_many_owned(self.capacity as u32);
        let permit = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), acquire)
                .await
                .map_err(|_| Error::Timeout("device execution capacity".to_string()))?
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
                .map_err(|_| Error::Timeout("direct inference job".to_string()))?,
            None => future.await,
        }
    }
}

#[derive(Debug)]
pub struct JobLease {
    coordinator: Arc<InferenceCoordinator>,
    _permit: OwnedSemaphorePermit,
    pub spec: JobSpec,
}

impl Drop for JobLease {
    fn drop(&mut self) {
        self.coordinator.active_jobs.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct ExecutionLease {
    coordinator: Arc<InferenceCoordinator>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        self.coordinator
            .active_executions
            .fetch_sub(1, Ordering::Relaxed);
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
        assert!(matches!(
            coordinator.admit(job("second")).await,
            Err(Error::Overloaded(_))
        ));
        drop(lease);
        let second = coordinator.admit(job("second")).await.unwrap();
        assert_eq!(coordinator.snapshot().active_jobs, 1);
        drop(second);
        assert_eq!(coordinator.snapshot().active_jobs, 0);
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
}
