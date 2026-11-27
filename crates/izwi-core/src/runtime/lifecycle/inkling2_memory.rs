//! Load admission for Inkling 2, priced from the checkpoint header before a
//! weight is read.
//!
//! | Part | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | Experts (NVFP4, MXFP8), as stored | mapped file | mapped file | device (the mapped file when they run on the host) |
//! | Large dense matrices, text and vision | host, F32 | device, F16 | device, F32 (BF16 on request) |
//! | Other dense tensors, F32 | host | device | device |
//! | Routers, F32 | host | host | host |
//!
//! The dense tensors are priced at the dtype the backend holds them in (the
//! same choice the loader makes, its environment switch included). The
//! load's transient is one dense tensor in flight: the largest one's held
//! size on the host and again on the device. The block drafter, when the
//! load serves it, is priced as F32 dense weights; its keys and values are
//! paged state the engine prices per token.

use std::path::Path;

use super::{model_resource_plan, ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::{ResourceAmount, ResourceVector};
use crate::error::Result;
use crate::models::architectures::inkling2::chat::{
    decode_graph_device_bytes, ExpertBackendRequest,
};
use crate::models::architectures::inkling2::memory::Inkling2Footprint;
use crate::models::architectures::inkling2::precision::projection_dtype;

pub(super) fn resource_plan(model_path: &Path, backend: BackendKind) -> Result<ModelResourcePlan> {
    let footprint = Inkling2Footprint::load(model_path, backend)?;
    // The same request the loader reads.
    let experts_on_device = ExpertBackendRequest::from_env().wants_device(backend);
    plan_for(
        footprint,
        backend,
        experts_on_device,
        decode_graph_device_bytes(backend),
    )
}

fn plan_for(
    footprint: Inkling2Footprint,
    backend: BackendKind,
    experts_on_device: bool,
    graph_bytes: u64,
) -> Result<ModelResourcePlan> {
    tracing::info!(
        backend = ?backend,
        experts_on_device,
        expert_bytes = footprint.expert_bytes,
        dense_bytes = footprint.dense_bytes,
        projection_dtype = ?projection_dtype(backend),
        resident_bytes = footprint.resident_bytes(backend, experts_on_device),
        load_peak_bytes = footprint.load_peak_bytes(backend, experts_on_device),
        "Priced Inkling 2 from its checkpoint header"
    );
    let mut plan = model_resource_plan(
        backend,
        ModelMemoryEstimate {
            load_peak_bytes: footprint.load_peak_bytes(backend, experts_on_device),
            resident_bytes: footprint.resident_bytes(backend, experts_on_device),
        },
    );
    // The decode graphs allocate on the first decode steps: charged from the
    // load, pending until then, so state fitting cannot take the room first.
    let graphs = ResourceVector {
        device_bytes: ResourceAmount::Known(graph_bytes),
        ..ResourceVector::zero()
    };
    plan.load_authorization = plan.load_authorization.checked_add(graphs)?;
    plan.resident_authorization = plan.resident_authorization.checked_add(graphs)?;
    plan.deferred_resident_authorization = graphs;
    if backend == BackendKind::Cuda {
        // The device holds the dense tensors, and the experts when they run
        // there; the host keeps the routers and, otherwise, the mapped
        // experts for the model's lifetime.
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(footprint.cuda_host_load_peak_bytes(experts_on_device));
        plan.resident_authorization = plan.resident_authorization.checked_add(ResourceVector {
            host_bytes: ResourceAmount::Known(footprint.host_resident_bytes(experts_on_device)),
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
            vision_bytes: 0,
            router_bytes: 10,
            largest_dense_bytes: 100,
        };
        let cpu = plan_for(footprint, BackendKind::Cpu, false, 0).unwrap();
        assert_eq!(known(cpu.resident_authorization.host_bytes), 1_410);
        assert_eq!(known(cpu.load_authorization.host_bytes), 1_610);
        let metal = plan_for(footprint, BackendKind::Metal, false, 0).unwrap();
        assert_eq!(known(metal.resident_authorization.unified_bytes), 1_410);
        assert_eq!(known(metal.load_authorization.unified_bytes), 1_610);
        // CUDA with the experts on the host.
        let cuda = plan_for(footprint, BackendKind::Cuda, false, 0).unwrap();
        assert_eq!(known(cuda.resident_authorization.device_bytes), 400);
        assert_eq!(known(cuda.resident_authorization.host_bytes), 1_010);
        assert_eq!(known(cuda.load_authorization.device_bytes), 500);
        assert_eq!(known(cuda.load_authorization.host_bytes), 1_110);
        // CUDA as served: the experts and a router copy on the device.
        let cuda = plan_for(footprint, BackendKind::Cuda, true, 0).unwrap();
        assert_eq!(known(cuda.resident_authorization.device_bytes), 1_410);
        assert_eq!(known(cuda.resident_authorization.host_bytes), 10);
        assert_eq!(known(cuda.load_authorization.device_bytes), 1_510);
        assert_eq!(known(cuda.load_authorization.host_bytes), 110);
        // The decode graphs are charged to the device and kept pending.
        let cuda = plan_for(footprint, BackendKind::Cuda, true, 64).unwrap();
        assert_eq!(known(cuda.resident_authorization.device_bytes), 1_474);
        assert_eq!(known(cuda.load_authorization.device_bytes), 1_574);
        assert_eq!(known(cuda.deferred_resident_authorization.device_bytes), 64);
    }

    /// The tiny checkpoint through the header: a backend that holds 16-bit
    /// projections is charged for them, the others for F32 ones.
    #[test]
    fn the_dense_tensors_are_charged_at_the_dtype_the_backend_holds() {
        use crate::models::architectures::inkling2::config::Inkling2Config;
        use crate::models::architectures::inkling2::precision::DENSE_DTYPE_ENV;
        use crate::models::architectures::inkling2::Inkling2Checkpoint;
        use candle_core::DType;

        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_golden");
        let config = Inkling2Config::load(&dir).unwrap();
        let checkpoint = Inkling2Checkpoint::open(&dir, &config).unwrap();
        let wide = Inkling2Footprint::of(&config, &checkpoint, DType::F32);
        let narrow = Inkling2Footprint::of(&config, &checkpoint, DType::F16);
        assert!(narrow.dense_bytes < wide.dense_bytes);
        let saved = wide.dense_bytes - narrow.dense_bytes;
        let resident = |footprint, backend| {
            let plan = plan_for(footprint, backend, false, 0).unwrap();
            match backend {
                BackendKind::Cpu => known(plan.resident_authorization.host_bytes),
                BackendKind::Metal => known(plan.resident_authorization.unified_bytes),
                BackendKind::Cuda => known(plan.resident_authorization.device_bytes),
            }
        };
        for backend in [BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(resident(wide, backend) - resident(narrow, backend), saved);
        }
        // The loader's own choice: sixteen bits are a request everywhere.
        if crate::env::value(DENSE_DTYPE_ENV).is_none() {
            for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
                assert_eq!(Inkling2Footprint::load(&dir, backend).unwrap(), wide);
            }
        }
    }
}
