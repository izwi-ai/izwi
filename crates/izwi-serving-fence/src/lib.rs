//! The launch contract between the serving supervisor and the workers it
//! starts: the lock and fence primitives both sides take, the per-worker lock
//! paths, and the environment that hands them over. Kept apart from the
//! supervisor so a worker links neither the supervisor nor its transport
//! client.

mod locks;

use izwi_serving_protocol::{DeviceAssignment, WorkerId};
use std::path::PathBuf;

pub use locks::*;

pub const MANAGED_WORKER_ENV: &str = "IZWI_WORKER_MANAGED";
pub const WORKER_INCARNATION_ENV: &str = "IZWI_WORKER_INCARNATION_ID";
pub const WORKER_OWNERSHIP_LOCK_ENV: &str = "IZWI_WORKER_OWNERSHIP_LOCK";
pub const WORKER_GENERATION_FENCE_ENV: &str = "IZWI_WORKER_GENERATION_FENCE_LOCK";
pub const WORKER_MODEL_LOAD_LOCK_ENV: &str = "IZWI_WORKER_MODEL_LOAD_LOCK";
pub const WORKER_MODEL_LOAD_SLOTS_ENV: &str = "IZWI_WORKER_MODEL_LOAD_SLOTS";
pub const WORKER_TASK_ENV: &str = "IZWI_WORKER_TASK";
/// The deployment's declared execution profile and token limits. The worker
/// advertises exactly these once its loaded model proves them and refuses to
/// start otherwise, so its status equals the approved deployment.
pub const WORKER_PRECISION_ENV: &str = "IZWI_WORKER_PRECISION";
pub const WORKER_EXECUTION_REPRESENTATION_ENV: &str = "IZWI_WORKER_EXECUTION_REPRESENTATION";
pub const WORKER_MAX_CONTEXT_TOKENS_ENV: &str = "IZWI_WORKER_MAX_CONTEXT_TOKENS";
pub const WORKER_MAX_OUTPUT_TOKENS_ENV: &str = "IZWI_WORKER_MAX_OUTPUT_TOKENS";
/// Bounds of a node's parallel model-load stage slots.
pub const DEFAULT_MODEL_LOAD_SLOTS: u32 = 1;
pub const MAX_MODEL_LOAD_SLOTS: u32 = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLockPaths {
    ownership: PathBuf,
    generation_fence: PathBuf,
    model_load: PathBuf,
}

impl WorkerLockPaths {
    pub fn for_worker(
        namespace: &LockNamespace,
        worker_id: &WorkerId,
        assignment: &DeviceAssignment,
    ) -> Self {
        let resource_identity = match assignment {
            DeviceAssignment::Cpu { .. } => format!("cpu-worker:{worker_id}"),
            DeviceAssignment::Metal { device_id, .. } => format!("metal:{device_id}"),
            DeviceAssignment::Cuda { device_uuid, .. } => format!("cuda:{device_uuid}"),
            DeviceAssignment::Unknown => format!("unknown-worker:{worker_id}"),
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
        namespace: &LockNamespace,
        metadata: &[u8],
    ) -> Result<WorkerFenceLeases, LockError> {
        try_acquire_worker_fences(namespace, &self.ownership, &self.generation_fence, metadata)
    }
}
