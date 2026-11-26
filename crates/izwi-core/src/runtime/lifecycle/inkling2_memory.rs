//! Load admission for Inkling 2, priced from the checkpoint header before a
//! weight is read.
//!
//! | Part | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | Experts (NVFP4, MXFP8), as stored | mapped file | mapped file | mapped file (host) |
//! | Dense text tensors, F32 | host | device | device |
//! | Routers, F32 | host | host | host |
//!
//! The load's transient is the largest dense tensor's host buffer and, on
//! the backends where host and device share memory, its transposed copy.
//! The vision tower and the MTP head are not loaded, so they are not priced.

use std::path::Path;

use super::{model_resource_plan, ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::{ResourceAmount, ResourceVector};
use crate::error::Result;
use crate::models::architectures::inkling2::memory::Inkling2Footprint;

pub(super) fn resource_plan(model_path: &Path, backend: BackendKind) -> Result<ModelResourcePlan> {
    let footprint = Inkling2Footprint::load(model_path)?;
    plan_for(footprint, backend)
}

fn plan_for(footprint: Inkling2Footprint, backend: BackendKind) -> Result<ModelResourcePlan> {
    tracing::info!(
        backend = ?backend,
        expert_bytes = footprint.expert_bytes,
        dense_bytes = footprint.dense_bytes,
        resident_bytes = footprint.resident_bytes(backend),
        load_peak_bytes = footprint.load_peak_bytes(backend),
        "Priced Inkling 2 from its checkpoint header"
    );
    let mut plan = model_resource_plan(
        backend,
        ModelMemoryEstimate {
            load_peak_bytes: footprint.load_peak_bytes(backend),
            resident_bytes: footprint.resident_bytes(backend),
        },
    );
    if backend == BackendKind::Cuda {
        // The device holds the dense tensors; the host keeps the mapped
        // experts and the routers for the model's lifetime.
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(footprint.cuda_host_load_peak_bytes());
        plan.resident_authorization = plan.resident_authorization.checked_add(ResourceVector {
            host_bytes: ResourceAmount::Known(footprint.host_resident_bytes()),
            ..ResourceVector::zero()
        })?;
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(amount: ResourceAmount) -> u64 {
        match amount {
            ResourceAmount::Known(bytes) => bytes,
            other => panic!("unpriced amount {other:?}"),
        }
    }

    #[test]
    fn each_backend_is_charged_where_it_holds_the_weights() {
        let footprint = Inkling2Footprint {
            expert_bytes: 1_000,
            dense_bytes: 400,
            router_bytes: 10,
            largest_dense_bytes: 100,
        };
        let cpu = plan_for(footprint, BackendKind::Cpu).unwrap();
        assert_eq!(known(cpu.resident_authorization.host_bytes), 1_410);
        assert_eq!(known(cpu.load_authorization.host_bytes), 1_610);
        let metal = plan_for(footprint, BackendKind::Metal).unwrap();
        assert_eq!(known(metal.resident_authorization.unified_bytes), 1_410);
        assert_eq!(known(metal.load_authorization.unified_bytes), 1_610);
        let cuda = plan_for(footprint, BackendKind::Cuda).unwrap();
        assert_eq!(known(cuda.resident_authorization.device_bytes), 400);
        assert_eq!(known(cuda.resident_authorization.host_bytes), 1_010);
        assert_eq!(known(cuda.load_authorization.device_bytes), 500);
        assert_eq!(known(cuda.load_authorization.host_bytes), 1_110);
    }
}
