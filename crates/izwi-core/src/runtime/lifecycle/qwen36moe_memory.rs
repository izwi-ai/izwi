//! Admission inventory for the qwen3_5_moe persistent representation
//! (published checkpoint: Qwen3.6-35B-A3B-FP8).
//!
//! The numbers derive from the loader's own pinned tensor plans, so admission
//! can never drift from what the checkpoint actually materializes: CPU packs
//! projections as Q8_0 and keeps dense tensors in F32, Metal expands F16
//! (Apple GPUs have no FP8 path), and CUDA keeps the checkpoint's raw
//! block-FP8 bytes resident with per-tensor packed-Q8_0 fallback for tensors
//! the fp8 projection kernel cannot execute (`projection_residency_policy`).
//! Scale companions and vision tensors never become resident text-trunk
//! state; the MTP draft manifest charges admission only when the MTP load
//! policy makes the draft head resident.
use super::{ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::ResourceAmount;
use crate::error::{Error, Result};
use crate::models::architectures::qwen36moe::native::{
    pinned_representation_inventory, resolve_mtp_load_policy, RepresentationElementBucket,
};
use std::path::Path;

const PORTABLE_CONVERSION_SCRATCH_BYTES: u64 = 1024 * 1024 * 1024;
const CUDA_DEVICE_CONVERSION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const CUDA_HOST_STAGING_BYTES: u64 = 8 * 1024 * 1024 * 1024;

const Q8_0_BLOCK_ELEMENTS: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

fn overflow() -> Error {
    Error::ModelLoadError("Qwen3.5/3.6-MoE memory estimate overflow".into())
}

/// Resident bytes of one element bucket on a backend: packed Q8_0
/// projections plus F32 dense state on CPU, expanded F16 on Metal, and raw
/// block-FP8 bytes plus F32 block scales on CUDA (with packed-Q8_0 fallback
/// for kernel-incompatible tensors).
fn bucket_resident_bytes(
    backend: BackendKind,
    bucket: &RepresentationElementBucket,
) -> Result<u64> {
    let q8_bytes = bucket
        .fp8_elements
        .checked_add(Q8_0_BLOCK_ELEMENTS - 1)
        .ok_or_else(overflow)?
        / Q8_0_BLOCK_ELEMENTS
        * Q8_0_BLOCK_BYTES;
    let dense_bytes = |scale: u64| {
        bucket
            .dense_elements
            .checked_mul(scale)
            .ok_or_else(overflow)
    };
    match backend {
        BackendKind::Cpu => q8_bytes
            .checked_add(dense_bytes(4)?)
            .ok_or_else(overflow),
        BackendKind::Metal => (bucket
            .fp8_elements
            .checked_add(bucket.dense_elements)
            .ok_or_else(overflow)?)
        .checked_mul(2)
        .ok_or_else(overflow),
        BackendKind::Cuda => {
            // Native block-FP8 residency: conforming tensors keep raw E4M3FN
            // bytes (1 B/element) plus their F32 block scales; the remainder
            // materializes as packed Q8_0. The runtime reconciles the lease
            // against materialized usage at publication, so any drift between
            // this estimate and the assembled representation fails loudly.
            let compatible = bucket
                .fp8_elements
                .saturating_sub(bucket.fp8_incompatible_elements);
            let fallback_q8_bytes = bucket
                .fp8_incompatible_elements
                .checked_add(Q8_0_BLOCK_ELEMENTS - 1)
                .ok_or_else(overflow)?
                / Q8_0_BLOCK_ELEMENTS
                * Q8_0_BLOCK_BYTES;
            compatible
                .checked_add(bucket.fp8_scale_bytes)
                .ok_or_else(overflow)?
                .checked_add(fallback_q8_bytes)
                .ok_or_else(overflow)?
                .checked_add(dense_bytes(2)?)
                .ok_or_else(overflow)
        }
    }
}

/// Resident bytes of the pinned checkpoint on one backend, plus the MTP
/// draft bucket when the load policy makes the draft head resident.
fn resident_bytes(
    backend: BackendKind,
    mtp: Option<&RepresentationElementBucket>,
) -> Result<u64> {
    let inventory = pinned_representation_inventory();
    let mut bytes = bucket_resident_bytes(backend, &inventory.trunk_bucket())?;
    if let Some(mtp) = mtp {
        bytes = bytes
            .checked_add(bucket_resident_bytes(backend, mtp)?)
            .ok_or_else(overflow)?;
    }
    Ok(bytes)
}

pub(super) fn representation_memory_estimate(
    backend: BackendKind,
    mtp_enabled: bool,
) -> Result<ModelMemoryEstimate> {
    let inventory = pinned_representation_inventory();
    // Instantiation slack is material at this scale: ~60k same-sized expert
    // tensors (weights plus scale companions) pay the per-tensor allocator
    // bound while the resident representation is assembled.
    let tensor_count = if mtp_enabled {
        inventory
            .tensor_count
            .checked_add(inventory.mtp.tensor_count)
            .ok_or_else(overflow)?
    } else {
        inventory.tensor_count
    };
    let instantiation_slack = tensor_count
        .checked_mul(super::PER_TENSOR_INSTANTIATION_SLACK_BYTES)
        .ok_or_else(overflow)?;
    let resident_bytes = resident_bytes(backend, mtp_enabled.then_some(&inventory.mtp))?;
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

pub(super) fn resource_plan(
    backend: BackendKind,
    performance: &crate::performance::PerformanceConfig,
) -> Result<ModelResourcePlan> {
    // Admission resolves the same MTP load policy the chat loader will: the
    // draft head's resident bucket joins the reservation only when enabled.
    let mtp_enabled = resolve_mtp_load_policy(backend, &performance.cuda)?
        == crate::models::architectures::qwen36moe::native::Qwen36MoeMtpLoadPolicy::Enabled;
    let estimate = representation_memory_estimate(backend, mtp_enabled)?;
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

        // The MTP draft bucket is census-exact too (revision 95a723d0):
        // 256 routed experts + shared expert of [512, 2048]/[2048, 512]
        // projections plus the gated [8192, 2048] draft attention set, and
        // the dense frame (fc, router, norms, shared-expert gate) — 1,560
        // tensors.
        assert_eq!(inventory.mtp.tensor_count, 1_560);
        assert_eq!(inventory.mtp.fp8_elements, 835_715_072);
        assert_eq!(inventory.mtp.dense_elements, 8_925_696);
        assert_eq!(inventory.mtp.fp8_incompatible_elements, 0);
        assert_eq!(inventory.mtp.fp8_scale_bytes, 204_032);
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
        // Default admission (MTP load policy disabled) charges only the
        // text trunk: the draft head's bucket stays out of the reservation.
        let cpu = representation_memory_estimate(BackendKind::Cpu, false).unwrap();
        let metal = representation_memory_estimate(BackendKind::Metal, false).unwrap();
        let cuda = representation_memory_estimate(BackendKind::Cuda, false).unwrap();

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

        // With the MTP load policy enabled, the draft bucket joins the
        // reservation under the same per-backend policy.
        let cpu_mtp = inventory.mtp.fp8_elements.div_ceil(Q8_0_BLOCK_ELEMENTS) * Q8_0_BLOCK_BYTES
            + inventory.mtp.dense_elements * 4;
        let metal_mtp = (inventory.mtp.fp8_elements + inventory.mtp.dense_elements) * 2;
        let cuda_mtp = inventory.mtp.fp8_elements
            + inventory.mtp.fp8_scale_bytes
            + inventory.mtp.dense_elements * 2;
        let cpu_enabled = representation_memory_estimate(BackendKind::Cpu, true).unwrap();
        let metal_enabled = representation_memory_estimate(BackendKind::Metal, true).unwrap();
        let cuda_enabled = representation_memory_estimate(BackendKind::Cuda, true).unwrap();
        assert_eq!(cpu_enabled.resident_bytes, cpu.resident_bytes + cpu_mtp);
        assert_eq!(metal_enabled.resident_bytes, metal.resident_bytes + metal_mtp);
        assert_eq!(cuda_enabled.resident_bytes, cuda.resident_bytes + cuda_mtp);
        // ~1,543 extra resident tensors pay the per-tensor slack too.
        assert!(cpu_enabled.load_peak_bytes > cpu.load_peak_bytes + cpu_mtp);
    }

    #[test]
    fn resource_plan_authorizes_cuda_host_staging() {
        use crate::performance::PerformanceConfig;

        // The MTP policy reads a process env var, so both plan computations
        // share one env_test_lock acquisition — parallel tests that flip the
        // handoff opt-in must not race either leg.
        let _env_guard = crate::env_test_lock().lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(crate::models::architectures::qwen36moe::native::MTP_HANDOFF_ENV);

        // Default knobs resolve the MTP load policy to disabled (the
        // handoff opt-in is unset), so the plan matches the trunk-only
        // representation.
        let disabled =
            resource_plan(BackendKind::Cuda, &PerformanceConfig::default()).unwrap();
        assert_eq!(
            disabled.load_authorization.host_bytes,
            ResourceAmount::Known(CUDA_HOST_STAGING_BYTES)
        );
        assert!(matches!(
            disabled.load_authorization.device_bytes,
            ResourceAmount::Known(bytes) if bytes > 30 * GIB && bytes < 40 * GIB
        ));
        let cpu_plan = resource_plan(BackendKind::Cpu, &PerformanceConfig::default()).unwrap();
        assert!(matches!(
            cpu_plan.load_authorization.host_bytes,
            ResourceAmount::Known(bytes) if bytes > 40 * GIB
        ));
        assert_eq!(
            cpu_plan.load_authorization.device_bytes,
            ResourceAmount::Known(0)
        );

        // The handoff opt-in makes admission reserve the draft bucket.
        std::env::set_var(crate::models::architectures::qwen36moe::native::MTP_HANDOFF_ENV, "1");
        let enabled = resource_plan(BackendKind::Cuda, &PerformanceConfig::default()).unwrap();
        std::env::remove_var(crate::models::architectures::qwen36moe::native::MTP_HANDOFF_ENV);
        assert!(matches!(
            (disabled.load_authorization.device_bytes, enabled.load_authorization.device_bytes),
            (
                ResourceAmount::Known(disabled_bytes),
                ResourceAmount::Known(enabled_bytes)
            ) if enabled_bytes > disabled_bytes
        ));
    }
}
