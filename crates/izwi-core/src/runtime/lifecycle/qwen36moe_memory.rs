//! Admission inventory for the qwen3_5_moe persistent representation
//! (published checkpoint: Qwen3.6-35B-A3B-FP8).
//!
//! The numbers derive from the loader's own pinned tensor plan, so admission
//! can never drift from what the checkpoint actually materializes: CPU packs
//! projections as Q8_0 and keeps dense tensors in F32, Metal expands F16
//! (Apple GPUs have no FP8 path), and CUDA keeps the checkpoint's raw
//! block-FP8 bytes resident with per-tensor packed-Q8_0 fallback for tensors
//! the fp8 projection kernel cannot execute (`projection_residency_policy`).
//! Scale companions and MTP/vision tensors never become resident text-trunk
//! state.
use super::{ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::ResourceAmount;
use crate::error::{Error, Result};
use crate::models::architectures::qwen36moe::native::pinned_representation_inventory;
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
/// projections plus F32 dense state on CPU, expanded F16 on Metal, and raw
/// block-FP8 bytes plus F32 block scales on CUDA (with packed-Q8_0 fallback
/// for kernel-incompatible tensors).
fn resident_bytes(backend: BackendKind) -> Result<u64> {
    let inventory = pinned_representation_inventory();
    let q8_bytes = inventory
        .fp8_elements
        .checked_add(Q8_0_BLOCK_ELEMENTS - 1)
        .ok_or_else(overflow)?
        / Q8_0_BLOCK_ELEMENTS
        * Q8_0_BLOCK_BYTES;
    let dense_bytes = |scale: u64| {
        inventory
            .dense_elements
            .checked_mul(scale)
            .ok_or_else(overflow)
    };
    match backend {
        BackendKind::Cpu => q8_bytes
            .checked_add(dense_bytes(4)?)
            .ok_or_else(overflow),
        BackendKind::Metal => (inventory
            .fp8_elements
            .checked_add(inventory.dense_elements)
            .ok_or_else(overflow)?)
        .checked_mul(2)
        .ok_or_else(overflow),
        BackendKind::Cuda => {
            // Native block-FP8 residency: conforming tensors keep raw E4M3FN
            // bytes (1 B/element) plus their F32 block scales; the remainder
            // materializes as packed Q8_0. The runtime reconciles the lease
            // against materialized usage at publication, so any drift between
            // this estimate and the assembled representation fails loudly.
            let compatible = inventory
                .fp8_elements
                .saturating_sub(inventory.fp8_incompatible_elements);
            let fallback_q8_bytes = inventory
                .fp8_incompatible_elements
                .checked_add(Q8_0_BLOCK_ELEMENTS - 1)
                .ok_or_else(overflow)?
                / Q8_0_BLOCK_ELEMENTS
                * Q8_0_BLOCK_BYTES;
            compatible
                .checked_add(inventory.fp8_scale_bytes)
                .ok_or_else(overflow)?
                .checked_add(fallback_q8_bytes)
                .ok_or_else(overflow)?
                .checked_add(dense_bytes(2)?)
                .ok_or_else(overflow)
        }
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
        // Host memory only holds the shard/staging window; the raw block-FP8
        // representation uploads directly to the device.
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
        // The published projection geometry satisfies the CUDA fp8 kernel
        // contract (n % 64 == 0, k % 128 == 0), so the native-FP8 residency
        // estimate covers the whole FP8 bucket with no Q8_0 fallback.
        assert_eq!(inventory.fp8_incompatible_elements, 0);
        assert!(inventory.fp8_scale_bytes > 0);
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

        let expected_metal = (inventory.fp8_elements + inventory.dense_elements) * 2;
        assert_eq!(metal.resident_bytes, expected_metal);
        assert!(metal.load_peak_bytes > metal.resident_bytes);

        // CUDA native-FP8 residency: raw E4M3FN bytes (1 B/element) plus F32
        // block scales for the kernel-conforming projections (all of them in
        // the pinned census), plus BF16 dense companions.
        let expected_cuda = inventory.fp8_elements
            + inventory.fp8_scale_bytes
            + inventory.dense_elements * 2;
        assert_eq!(cuda.resident_bytes, expected_cuda);
        assert!(cuda.load_peak_bytes > cuda.resident_bytes);
        // The compact residency holds ~half of the Metal F16 expansion and
        // stays below even the CPU Q8_0 + F32-dense envelope.
        assert!(
            cuda.resident_bytes < metal.resident_bytes && cuda.resident_bytes < cpu.resident_bytes,
            "CUDA {} vs metal {} vs cpu {}",
            cuda.resident_bytes,
            metal.resident_bytes,
            cpu.resident_bytes
        );
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
            ResourceAmount::Known(bytes) if bytes > 30 * GIB && bytes < 40 * GIB
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
