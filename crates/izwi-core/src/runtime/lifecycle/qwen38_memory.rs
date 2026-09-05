//! Admission inventory for the selected CUDA representation. This reads tensor
//! metadata before allocation; it does not materialize checkpoint payloads.
use super::{ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::ResourceAmount;
use crate::error::{Error, Result};
use crate::models::architectures::qwen38::native::{
    native_tensor_scope, NativeTensorScope, Qwen38NativeCheckpoint,
};
use crate::performance::{CudaProjectionBackend, PerformanceConfig};
use candle_core::DType;
use safetensors::Dtype as SafeDType;
use std::collections::BTreeSet;
use std::path::Path;

const METADATA_RESERVE: u64 = 128 * 1024 * 1024;
const DEVICE_ALIGNMENT_RESERVE: u64 = 64 * 1024 * 1024;
const GRAPH_CACHE_PER_STACK: u64 = 8 * 1024 * 1024;

fn overflow() -> Error {
    Error::ModelLoadError("Qwen3.8 selected representation inventory overflow".into())
}

fn elements(shape: &[usize]) -> Result<u64> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(Error::ModelLoadError("Empty Qwen3.8 weight shape".into()));
    }
    shape.iter().try_fold(1u64, |n, &d| {
        n.checked_mul(u64::try_from(d).map_err(|_| overflow())?)
            .ok_or_else(overflow)
    })
}

fn weight_bytes(name: &str, dtype: SafeDType, shape: &[usize], native: bool) -> Result<u64> {
    let count = elements(shape)?;
    if dtype == SafeDType::F8_E4M3 {
        if shape.len() != 2 || !shape[1].is_multiple_of(32) {
            return Err(Error::ModelLoadError(
                "Invalid Qwen3.8 CUDA projection shape".into(),
            ));
        }
        return if native {
            Ok(count)
        } else {
            (count / 32).checked_mul(34).ok_or_else(overflow)
        };
    }
    if !matches!(dtype, SafeDType::BF16 | SafeDType::F16 | SafeDType::F32) {
        return Err(Error::ModelLoadError(format!(
            "Unsupported Qwen3.8 dense dtype {dtype:?}"
        )));
    }
    // DeltaNet math constants and convolution kernels deliberately retain F32.
    // Kernel slices are views and +1/-exp initialization replaces its source.
    let f32_math = name.contains(".linear_attn.")
        && [".dt_bias", ".A_log", ".conv1d.weight", ".norm.weight"]
            .iter()
            .any(|suffix| name.ends_with(suffix));
    count
        .checked_mul(if f32_math { 4 } else { 2 })
        .ok_or_else(overflow)
}

pub(super) fn resource_plan(
    model_path: &Path,
    device: &candle_core::Device,
    performance: &PerformanceConfig,
) -> Result<ModelResourcePlan> {
    let checkpoint = Qwen38NativeCheckpoint::open_with_options(model_path, &performance.loading)?;
    let mtp = performance.cuda.enabled() && performance.cuda.mtp.enabled();
    let mut weights = 0u64;
    let mut source_shards = BTreeSet::new();
    for name in checkpoint.tensors.tensor_names() {
        if !(matches!(
            native_tensor_scope(name),
            NativeTensorScope::Text | NativeTensorScope::LmHead
        ) || mtp && native_tensor_scope(name) == NativeTensorScope::Mtp)
        {
            continue;
        }
        // Q8 incorporates source scales; native FP8 adds its F32 scale grid
        // together with the corresponding projection below.
        let info = checkpoint.tensors.tensor_info(name)?;
        source_shards.insert(info.shard.clone());
        if name.ends_with(".weight_scale_inv") {
            continue;
        }
        let native = info.dtype == SafeDType::F8_E4M3
            && info.shape.len() == 2
            && performance.cuda.enabled()
            && match performance.cuda.projection_backend {
                CudaProjectionBackend::Q8 => false,
                CudaProjectionBackend::Auto => crate::kernels::cuda::fp8::provider_auto_preferred(
                    device,
                    DType::BF16,
                    info.shape[0],
                    info.shape[1],
                ),
                CudaProjectionBackend::NativeFp8 => crate::kernels::cuda::fp8::provider_supported(
                    device,
                    DType::BF16,
                    info.shape[0],
                    info.shape[1],
                ),
            };
        weights = weights
            .checked_add(weight_bytes(name, info.dtype, &info.shape, native)?)
            .ok_or_else(overflow)?;
        if native {
            let stem = name.strip_suffix(".weight").ok_or_else(overflow)?;
            let scale = checkpoint
                .tensors
                .tensor_info(&format!("{stem}.weight_scale_inv"))?;
            let block = checkpoint.config.block_fp8.block_shape;
            if scale.shape
                != [
                    info.shape[0].div_ceil(block[0]),
                    info.shape[1].div_ceil(block[1]),
                ]
            {
                return Err(Error::ModelLoadError(
                    "Native FP8 inventory scale shape mismatch".into(),
                ));
            }
            weights = weights
                .checked_add(
                    elements(&scale.shape)?
                        .checked_mul(4)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
        }
    }
    let graphs = if performance.cuda.enabled() && performance.cuda.decode_graphs.enabled() {
        GRAPH_CACHE_PER_STACK * if mtp { 2 } else { 1 }
    } else {
        0
    };
    let resident_bytes = weights
        .checked_add(DEVICE_ALIGNMENT_RESERVE)
        .and_then(|n| n.checked_add(graphs))
        .ok_or_else(overflow)?;
    // This extra device overlap covers initialization outputs and allocator
    // padding. Every projection is uploaded into its final allocation once.
    let estimate = ModelMemoryEstimate {
        resident_bytes,
        load_peak_bytes: resident_bytes
            .checked_add(super::QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES)
            .ok_or_else(overflow)?,
    };
    let mut plan = super::model_resource_plan(BackendKind::Cuda, estimate);
    let scratch = if performance.loading.enabled() {
        let mut mapped_sizes = source_shards
            .iter()
            .map(|path| std::fs::metadata(path).map(|metadata| metadata.len()))
            .collect::<std::io::Result<Vec<_>>>()?;
        mapped_sizes.sort_unstable_by(|a, b| b.cmp(a));
        // File-backed pages are reclaimable, but reserve the complete two-map
        // window conservatively rather than confusing a staging cap with RSS.
        let mapped = mapped_sizes
            .into_iter()
            .take(2)
            .try_fold(0u64, |sum, n| sum.checked_add(n).ok_or_else(overflow))?;
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let requested = if performance.loading.workers == 0 {
            available
        } else {
            performance.loading.workers
        };
        let workers = if performance.loading.parallel_conversion.enabled() {
            requested.min(available).clamp(1, 64)
        } else {
            1
        };
        let stacks = (workers as u64 + 1) * 256 * 1024;
        u64::try_from(performance.loading.max_staging_bytes)
            .map_err(|_| overflow())?
            .checked_add(mapped)
            .and_then(|n| n.checked_add(stacks))
            .ok_or_else(overflow)?
    } else {
        super::QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES
    };
    plan.load_authorization.host_bytes =
        ResourceAmount::Known(scratch.checked_add(METADATA_RESERVE).ok_or_else(overflow)?);
    plan.resident_authorization.host_bytes = ResourceAmount::Known(METADATA_RESERVE);
    // The bounded graph cache is allocated lazily. Do not mark it physically
    // materialized before state/context fitting: that would let KV consume its
    // headroom. Keeping this small claim pending is conservative after capture.
    plan.deferred_resident_authorization.device_bytes = ResourceAmount::Known(graphs);
    tracing::info!(
        weights_bytes = weights,
        graph_capacity_bytes = graphs,
        resident_bytes,
        host_staging_bytes = scratch,
        "Selected Qwen3.8 CUDA weight inventory"
    );
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inventory_prices_compact_weights_and_f32_math_without_source_scale_double_counting() {
        assert_eq!(
            weight_bytes("projection.weight", SafeDType::F8_E4M3, &[128, 256], false).unwrap(),
            34816
        );
        assert_eq!(
            weight_bytes("projection.weight", SafeDType::F8_E4M3, &[128, 256], true).unwrap(),
            32768
        );
        assert_eq!(
            weight_bytes(
                "model.language_model.embed_tokens.weight",
                SafeDType::BF16,
                &[248320, 5120],
                false
            )
            .unwrap(),
            2542796800
        );
        for suffix in ["dt_bias", "A_log", "conv1d.weight", "norm.weight"] {
            assert_eq!(
                weight_bytes(
                    &format!("model.language_model.layers.0.linear_attn.{suffix}"),
                    SafeDType::BF16,
                    &[128],
                    false
                )
                .unwrap(),
                512
            );
        }
        assert_eq!(
            weight_bytes(
                "mtp.layers.0.input_layernorm.weight",
                SafeDType::BF16,
                &[5120],
                false
            )
            .unwrap(),
            10240
        );
    }
    #[test]
    fn malformed_inventory_fails_before_allocation() {
        assert!(weight_bytes("projection.weight", SafeDType::F8_E4M3, &[128, 33], false).is_err());
        assert!(weight_bytes("projection.weight", SafeDType::F8_E4M3, &[0, 128], true).is_err());
        assert!(weight_bytes("x", SafeDType::BF16, &[usize::MAX, usize::MAX], false).is_err());
        assert!(weight_bytes("x", SafeDType::U8, &[128], false).is_err());
    }
}
