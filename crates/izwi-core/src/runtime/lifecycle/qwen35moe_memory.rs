//! Admission inventory for the qwen3_5_moe persistent representation
//! (published checkpoint: Qwen3.6-35B-A3B-FP8).
//!
//! The numbers derive from the loader's own pinned tensor plan, so admission
//! can never drift from what the checkpoint actually materializes: CPU packs
//! projections as Q8_0 and keeps dense tensors in F32, Metal expands F16,
//! CUDA expands BF16 (`projection_residency_policy`). Scale companions and
//! MTP/vision tensors never become resident text-trunk state.
use super::{ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::ResourceAmount;
use crate::error::{Error, Result};
use crate::models::architectures::qwen35moe::native::pinned_representation_inventory;
use std::path::Path;

const PORTABLE_CONVERSION_SCRATCH_BYTES: u64 = 1024 * 1024 * 1024;
const CUDA_DEVICE_CONVERSION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const CUDA_HOST_STAGING_BYTES: u64 = 8 * 1024 * 1024 * 1024;

const Q8_0_BLOCK_ELEMENTS: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

fn overflow() -> Error {
    Error::ModelLoadError("Qwen3.5/3.6-MoE memory estimate overflow".into())
}

/// Resident bytes of the pinned checkpoint on one backend: packed Q8_0
/// projections plus F32 dense state on CPU, expanded F16/BF16 elsewhere.
fn resident_bytes(backend: BackendKind) -> Result<u64> {
    let inventory = pinned_representation_inventory();
    let q8_bytes = inventory
        .fp8_elements
        .checked_add(Q8_0_BLOCK_ELEMENTS - 1)
        .ok_or_else(overflow)?
        / Q8_0_BLOCK_ELEMENTS
        * Q8_0_BLOCK_BYTES;
    match backend {
        BackendKind::Cpu => q8_bytes
            .checked_add(
                inventory
                    .dense_elements
                    .checked_mul(4)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow),
        BackendKind::Metal | BackendKind::Cuda => (inventory
            .fp8_elements
            .checked_add(inventory.dense_elements)
            .ok_or_else(overflow)?)
        .checked_mul(2)
        .ok_or_else(overflow),
    }
}

pub(super) fn representation_memory_estimate(backend: BackendKind) -> Result<ModelMemoryEstimate> {
    let inventory = pinned_representation_inventory();
    // Instantiation slack is material at this scale: ~60k same-sized expert
    // tensors (weights plus scale companions) pay the per-tensor allocator
    // bound while the resident representation is assembled.
    let instantiation_slack = inventory
        .tensor_count
        .checked_mul(super::PER_TENSOR_INSTANTIATION_SLACK_BYTES)
        .ok_or_else(overflow)?;
    let resident_bytes = resident_bytes(backend)?;
    let load_peak_bytes = match backend {
        BackendKind::Cpu | BackendKind::Metal => resident_bytes
            .checked_add(PORTABLE_CONVERSION_SCRATCH_BYTES)
            .ok_or_else(overflow)?,
        BackendKind::Cuda => resident_bytes
            .checked_add(CUDA_DEVICE_CONVERSION_SCRATCH_BYTES)
            .ok_or_else(overflow)?,
    }
    .checked_add(instantiation_slack)
    .ok_or_else(overflow)?;
    Ok(ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes,
    })
}

pub(super) fn resource_plan(backend: BackendKind) -> Result<ModelResourcePlan> {
    let estimate = representation_memory_estimate(backend)?;
    let mut plan = super::model_resource_plan(backend, estimate);
    if backend == BackendKind::Cuda {
        // Host memory only holds the shard/dequantization staging window; the
        // expanded BF16 representation materializes directly on the device.
        plan.load_authorization.host_bytes = ResourceAmount::Known(CUDA_HOST_STAGING_BYTES);
    }
    Ok(plan)
}

/// Fixture-mode estimate (synthetic geometry opt-in): derived from the actual
/// checkpoint inventory with a worst-case F32 expansion envelope, since the
/// fixture cannot carry the pinned element counts.
pub(super) fn synthetic_fixture_estimate(model_path: &Path) -> Result<ModelMemoryEstimate> {
    let overflow = || Error::ModelLoadError("Qwen3.5/3.6-MoE fixture memory estimate overflow".into());
    let Some(inventory) = super::checkpoint_tensor_inventory(model_path)? else {
        return Err(Error::ModelLoadError(
            "Synthetic Qwen3.5/3.6-MoE fixture has no readable tensor inventory".into(),
        ));
    };
    let resident_bytes = inventory.total_bytes.checked_mul(4).ok_or_else(overflow)?;
    let load_peak_bytes = resident_bytes
        .checked_add(PORTABLE_CONVERSION_SCRATCH_BYTES)
        .and_then(|bytes| bytes.checked_add(inventory.largest_tensor_bytes.next_power_of_two()))
        .ok_or_else(overflow)?;
    Ok(ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelVariant;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn pinned_inventory_matches_the_checkpoint_facts() {
        let inventory = pinned_representation_inventory();
        // Exact element counts derived from the pinned geometry: 40 layers ×
        // (256 experts + shared) of [512, 2048]/[2048, 512] projections, the
        // gate-fused [8192, 2048] q_proj on 10 full-attention layers, the
        // block-FP8 DeltaNet in_proj_qkv/in_proj_z on 30 GDN layers, and the
        // 248,320-row embeddings.
        assert_eq!(inventory.fp8_elements, 33_617_346_560);
        assert!(inventory.fp8_elements.is_multiple_of(Q8_0_BLOCK_ELEMENTS));
        assert_eq!(inventory.dense_elements, 1_043_264_128);
        // MoE scale: weights plus scale companions exceed 50k tensors, so the
        // per-tensor instantiation slack is a load-peak term, not noise.
        assert!(inventory.tensor_count > 50_000);
    }

    #[test]
    fn resident_representation_agrees_with_the_catalog_byte_pin() {
        let inventory = pinned_representation_inventory();
        // Source checkpoint bytes: FP8 (1 B) + dense BF16 (2 B). The catalog
        // pin carries tokenizer/metadata slack on top; hold 5% tolerance.
        // (memory_required_gb stays the deliberately conservative worst-case
        // hint; backend-specific admission replaces it.)
        let source_bytes = inventory
            .fp8_elements
            .checked_add(inventory.dense_elements.checked_mul(2).unwrap())
            .unwrap();
        let catalog_bytes = ModelVariant::Qwen36Moe35BA3BFp8.estimated_size();
        let deviation = catalog_bytes.abs_diff(source_bytes);
        assert!(
            deviation * 20 < catalog_bytes,
            "source bytes {source_bytes} drifted from the catalog pin {catalog_bytes}"
        );
    }

    #[test]
    fn resident_representation_matches_the_backend_policy() {
        let inventory = pinned_representation_inventory();
        let cpu = representation_memory_estimate(BackendKind::Cpu).unwrap();
        let metal = representation_memory_estimate(BackendKind::Metal).unwrap();
        let cuda = representation_memory_estimate(BackendKind::Cuda).unwrap();

        let expected_cpu = inventory.fp8_elements.div_ceil(Q8_0_BLOCK_ELEMENTS) * Q8_0_BLOCK_BYTES
            + inventory.dense_elements * 4;
        assert_eq!(cpu.resident_bytes, expected_cpu);
        // Q8_0 packing keeps CPU serving meaningful; the CPU projection
        // residency is far below an expanded-F32 alternative (~130 GiB).
        assert!(
            cpu.resident_bytes < 45 * GIB,
            "CPU residency {}",
            cpu.resident_bytes
        );
        assert!(cpu.load_peak_bytes > cpu.resident_bytes);

        for estimate in [metal, cuda] {
            let expected = (inventory.fp8_elements + inventory.dense_elements) * 2;
            assert_eq!(estimate.resident_bytes, expected);
            assert!(estimate.load_peak_bytes > estimate.resident_bytes);
        }
        // Metal F16/CUDA BF16 expansion stays below CPU Q8_0 + F32 dense here
        // because the dense bucket is small relative to the FP8 projections.
    }

    #[test]
    fn resource_plan_authorizes_cuda_host_staging() {
        let plan = resource_plan(BackendKind::Cuda).unwrap();
        assert_eq!(
            plan.load_authorization.host_bytes,
            ResourceAmount::Known(CUDA_HOST_STAGING_BYTES)
        );
        assert!(matches!(
            plan.load_authorization.device_bytes,
            ResourceAmount::Known(bytes) if bytes > 60 * GIB
        ));
        let cpu_plan = resource_plan(BackendKind::Cpu).unwrap();
        assert!(matches!(
            cpu_plan.load_authorization.host_bytes,
            ResourceAmount::Known(bytes) if bytes > 40 * GIB
        ));
        assert_eq!(
            cpu_plan.load_authorization.device_bytes,
            ResourceAmount::Known(0)
        );
    }
}
