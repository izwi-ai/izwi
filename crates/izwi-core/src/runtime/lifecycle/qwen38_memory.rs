//! Load admission for Qwen3.8 on every backend. The resident form of each
//! tensor comes from the Qwen3.8 residency policy
//! (`models/architectures/qwen38/residency.rs`), the same resolution the
//! loader uses, so admission charges exactly what the load allocates. This
//! reads tensor metadata before allocation; it does not materialize payloads.
use super::{ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::ResourceAmount;
use crate::error::{Error, Result};
use crate::models::architectures::qwen38::chat::{
    prefill_working_set_bytes, qwen38_prefill_chunk_tokens, Qwen38TextConfig,
};
use crate::models::architectures::qwen38::native::{
    native_tensor_scope, pinned_27b_config, resident_tensor_specs, NativeTensorScope,
    Qwen38NativeCheckpoint, Qwen38NativeConfig,
};
use crate::models::architectures::qwen38::residency::{
    admission_activation, backend_of, price_specs, Qwen38ResidencyPolicy, Qwen38WeightForm,
};
use crate::performance::PerformanceConfig;
use safetensors::Dtype as SafeDType;
use std::collections::BTreeSet;
use std::path::Path;

const METADATA_RESERVE: u64 = 128 * 1024 * 1024;
const DEVICE_ALIGNMENT_RESERVE: u64 = 64 * 1024 * 1024;

fn overflow() -> Error {
    Error::ModelLoadError("Qwen3.8 selected representation inventory overflow".into())
}

/// Resident and load-peak bytes of the pinned 27B checkpoint under the static
/// policy for `backend` (no device gates: CUDA prices Q8_0 here, while the
/// CUDA admission path, [`resource_plan`], resolves against the device).
pub(super) fn static_estimate(
    backend: BackendKind,
    performance: &PerformanceConfig,
) -> Result<ModelMemoryEstimate> {
    let policy = Qwen38ResidencyPolicy::resolve_static(backend, performance);
    let config = pinned_27b_config();
    let specs = resident_tensor_specs(&config, policy.mtp_loaded())?;
    let weights = price_specs(
        &policy,
        specs
            .iter()
            .map(|spec| (spec.name.as_str(), spec.dtype, spec.shape.as_slice())),
        config.block_fp8.block_shape,
    )?
    .checked_add(policy.prefill_scratch_bytes())
    .ok_or_else(overflow)?;
    let scratch = match backend {
        BackendKind::Cuda => super::QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES,
        BackendKind::Cpu | BackendKind::Metal => super::QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES,
    };
    Ok(ModelMemoryEstimate {
        resident_bytes: weights,
        load_peak_bytes: weights.checked_add(scratch).ok_or_else(overflow)?,
    })
}

/// [`static_estimate`] as a resource plan. CUDA authorizes its host
/// conversion window separately from the device peak.
#[cfg(test)]
pub(super) fn static_resource_plan(
    backend: BackendKind,
    performance: &PerformanceConfig,
) -> Result<ModelResourcePlan> {
    let mut plan = super::model_resource_plan(backend, static_estimate(backend, performance)?);
    if backend == BackendKind::Metal {
        plan.deferred_resident_authorization.unified_bytes = ResourceAmount::Known(
            Qwen38ResidencyPolicy::resolve_static(backend, performance).prefill_scratch_bytes(),
        );
    }
    if backend == BackendKind::Cuda {
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(super::QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES);
    }
    Ok(plan)
}

fn state_geometry(text: &Qwen38TextConfig) -> super::HybridStateGeometry {
    super::HybridStateGeometry {
        block_count: text.block_count,
        full_attention_interval: text.full_attention_interval,
        ssm_state_size: text.ssm_state_size,
        ssm_group_count: text.ssm_group_count,
        ssm_time_step_rank: text.ssm_time_step_rank,
        ssm_inner_size: text.ssm_inner_size,
        ssm_conv_kernel: text.ssm_conv_kernel,
    }
}

/// Recurrent state per retained or staged row, from the checkpoint config
/// alone, so state admission can run before any weight is read.
pub(super) fn per_sequence_state_bytes(model_path: &Path) -> Result<u64> {
    super::hybrid_recurrent_state_bytes_per_sequence(state_geometry(
        &Qwen38NativeConfig::load(model_path)?.text,
    ))
}

/// Admission for a load of the checkpoint at `model_path` on `device`.
pub(super) fn resource_plan(
    model_path: &Path,
    device: &candle_core::Device,
    performance: &PerformanceConfig,
) -> Result<ModelResourcePlan> {
    let backend = backend_of(device);
    let checkpoint = Qwen38NativeCheckpoint::open_with_options(model_path, &performance.loading)?;
    let policy = Qwen38ResidencyPolicy::resolve(
        device,
        admission_activation(device),
        performance,
        &checkpoint.tensors,
        &checkpoint.config,
    );
    let mtp = policy.mtp_loaded();
    let block = checkpoint.config.block_fp8.block_shape;
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
        let info = checkpoint.tensors.tensor_info(name)?;
        source_shards.insert(info.shard.clone());
        if info.dtype == SafeDType::F8_E4M3 {
            check_scale_companion(&checkpoint, &policy, name, &info.shape, block)?;
        }
        weights = weights
            .checked_add(policy.resident_bytes(name, info.dtype, &info.shape, block)?)
            .ok_or_else(overflow)?;
    }
    if backend != BackendKind::Cuda {
        // Metal FP8 prefill expands a projection to a transient F16 matrix:
        // resident (lazily allocated, so also a pending claim) beside the
        // weights.
        let prefill_scratch = policy.prefill_scratch_bytes();
        let resident_bytes = weights.checked_add(prefill_scratch).ok_or_else(overflow)?;
        // The CPU's Q8_0 conversion streams through the tiled converter:
        // its staging, two mapped shards and the worker stacks join the peak.
        let staging = if backend == BackendKind::Cpu
            && policy.plan()
                == crate::models::architectures::qwen38::residency::Qwen38ProjectionPlan::Q8
            && performance.loading.enabled()
        {
            host_conversion_scratch(performance, &source_shards)?
        } else {
            0
        };
        let estimate = ModelMemoryEstimate {
            resident_bytes,
            load_peak_bytes: resident_bytes
                .checked_add(super::QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES)
                .and_then(|n| n.checked_add(staging))
                .ok_or_else(overflow)?,
        };
        tracing::info!(
            backend = ?backend,
            weights_bytes = weights,
            representation = policy.representation().as_str(),
            mtp,
            "Selected Qwen3.8 weight inventory"
        );
        let mut plan = super::model_resource_plan(backend, estimate);
        if backend == BackendKind::Metal {
            plan.deferred_resident_authorization.unified_bytes =
                ResourceAmount::Known(prefill_scratch);
        }
        return Ok(plan);
    }
    // CUDA graph decode's working memory when the policy plans it, else the
    // bounded TensorIsland caches (the same figure the chat model reports).
    let graphs = policy.graph_capacity_bytes(&checkpoint.config.text, performance)?;
    // The prefill GEMM's one dequantized weight tile when the policy plans
    // it: lazily allocated like the graphs, so also a pending claim.
    let prefill_scratch = policy.prefill_scratch_bytes();
    // One prefill forward's activations at the chunk the load will use
    // (allocated per forward, so also a pending claim): a 1,024-token chunk
    // must not come out of the safety margin.
    let prefill_chunk = qwen38_prefill_chunk_tokens(BackendKind::Cuda, policy.fp8_prefill_gemm());
    let prefill_activations = prefill_working_set_bytes(&checkpoint.config.text, prefill_chunk)?;
    let deferred = graphs
        .checked_add(prefill_scratch)
        .and_then(|n| n.checked_add(prefill_activations))
        .ok_or_else(overflow)?;
    let resident_bytes = weights
        .checked_add(DEVICE_ALIGNMENT_RESERVE)
        .and_then(|n| n.checked_add(deferred))
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
        host_conversion_scratch(performance, &source_shards)?
    } else {
        super::QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES
    };
    // A block-FP8 LM head is quantized on the host: its bytes are staged once
    // (plus per-thread 128-row F32 temporaries inside the staging cap).
    let lm_head = [
        checkpoint.config.vocab_size,
        checkpoint.config.text.embedding_length,
    ];
    let lm_head_staging = if policy.lm_head_form(lm_head) == Qwen38WeightForm::RawFp8 {
        (lm_head[0] as u64)
            .checked_mul(lm_head[1] as u64)
            .ok_or_else(overflow)?
    } else {
        0
    };
    plan.load_authorization.host_bytes = ResourceAmount::Known(
        scratch
            .checked_add(METADATA_RESERVE)
            .and_then(|n| n.checked_add(lm_head_staging))
            .ok_or_else(overflow)?,
    );
    plan.resident_authorization.host_bytes = ResourceAmount::Known(METADATA_RESERVE);
    // The graph working memory is allocated lazily. Do not mark it physically
    // materialized before state/context fitting: that would let KV consume its
    // headroom. It stays pending after capture too, on purpose. With graph
    // decode planned (537,788,416 B for the 27B trunk), all but 64 MiB of it
    // is three state slabs that decode steps allocate afresh (the next state,
    // a pipelined step's, a verification scratch), so the claim must outlive
    // the capture. A slab that commits becomes the row's state, which the
    // managed lease records as materialized. Only the fixed part
    // (`GRAPH_DECODE_FIXED_BYTES`: graph pools, the state table, the step
    // buffer; or the 8 MiB region caches without graph decode) is held
    // between steps and so is charged twice. Its pools live in the device's
    // graph memory, which no per-model observation can measure, and
    // recording an estimate could under-charge. The prefill dequantization
    // scratch and chunk activations are likewise allocated per prefill chunk
    // and freed after it, so they stay pending for the same reason.
    plan.deferred_resident_authorization.device_bytes = ResourceAmount::Known(deferred);
    tracing::info!(
        weights_bytes = weights,
        graph_capacity_bytes = graphs,
        prefill_scratch_bytes = prefill_scratch,
        prefill_chunk_tokens = prefill_chunk,
        prefill_activation_bytes = prefill_activations,
        resident_bytes,
        host_staging_bytes = scratch,
        representation = policy.representation().as_str(),
        fp8_execution_mode = %policy.execution_mode(),
        "Selected Qwen3.8 CUDA weight inventory"
    );
    Ok(plan)
}

/// Host memory of the bounded conversion pipeline: the staging budget, the
/// two retained shard mappings and the conversion workers' stacks.
fn host_conversion_scratch(
    performance: &PerformanceConfig,
    source_shards: &BTreeSet<std::path::PathBuf>,
) -> Result<u64> {
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
        .ok_or_else(overflow)
}

/// A raw block-FP8 projection keeps its F32 scale grid resident beside it:
/// the checkpoint's scale companion must have exactly that grid.
fn check_scale_companion(
    checkpoint: &Qwen38NativeCheckpoint,
    policy: &Qwen38ResidencyPolicy,
    name: &str,
    shape: &[usize],
    block: [usize; 2],
) -> Result<()> {
    let Ok([n, k]) = <[usize; 2]>::try_from(shape) else {
        return Err(Error::ModelLoadError(format!(
            "Qwen3.8 FP8 tensor `{name}` is not a matrix"
        )));
    };
    if policy.projection_form(SafeDType::F8_E4M3, [n, k], block) != Qwen38WeightForm::RawFp8 {
        return Ok(());
    }
    let stem = name.strip_suffix(".weight").ok_or_else(overflow)?;
    let scale = checkpoint
        .tensors
        .tensor_info(&format!("{stem}.weight_scale_inv"))?;
    if scale.shape != [n.div_ceil(block[0]), k.div_ceil(block[1])] {
        return Err(Error::ModelLoadError(
            "Native FP8 inventory scale shape mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_state_bytes_match_the_27b_contract() {
        // The pinned Qwen3.8-27B geometry (`native.rs` Pinned27B policy). The
        // managed planner authorizes 156,893,184 B per row for this contract
        // (`managed.rs` qwen38_staging_width_cost_is_independent_of_retained_sessions).
        let text = Qwen38TextConfig {
            architecture: "qwen3_5".into(),
            block_count: 64,
            context_length: 262_144,
            embedding_length: 5_120,
            feed_forward_length: 17_408,
            attention_head_count: 24,
            attention_head_count_kv: 4,
            attention_key_length: 256,
            attention_value_length: 256,
            rope_dimension_sections: vec![11, 11, 10],
            rope_dimension_count: 64,
            rope_freq_base: 10_000_000.0,
            attention_layer_norm_rms_epsilon: 1e-6,
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6_144,
            full_attention_interval: 4,
        };
        assert_eq!(
            super::super::hybrid_recurrent_state_bytes_per_sequence(state_geometry(&text)).unwrap(),
            156_893_184
        );
    }
    #[test]
    fn inventory_prices_compact_weights_and_f32_math_without_source_scale_double_counting() {
        use crate::models::architectures::qwen38::native::ProjectionMaterialization;
        let block = [128, 128];
        let q8 =
            Qwen38ResidencyPolicy::resolve_static(BackendKind::Cuda, &PerformanceConfig::default());
        let native =
            Qwen38ResidencyPolicy::forced_native_fp8_for_tests(ProjectionMaterialization::BF16);
        let bytes = |policy: &Qwen38ResidencyPolicy, name: &str, dtype, shape: &[usize]| {
            policy.resident_bytes(name, dtype, shape, block).unwrap()
        };
        assert_eq!(
            bytes(&q8, "projection.weight", SafeDType::F8_E4M3, &[128, 256]),
            34816
        );
        // Raw FP8: one byte per element plus the F32 block-scale grid.
        assert_eq!(
            bytes(
                &native,
                "projection.weight",
                SafeDType::F8_E4M3,
                &[128, 256]
            ),
            32768 + 2 * 4
        );
        // The BF16 scale companion is charged with its projection.
        for policy in [&q8, &native] {
            assert_eq!(
                bytes(
                    policy,
                    "projection.weight_scale_inv",
                    SafeDType::BF16,
                    &[1, 2]
                ),
                0
            );
        }
        assert_eq!(
            bytes(
                &q8,
                "model.language_model.embed_tokens.weight",
                SafeDType::BF16,
                &[248320, 5120]
            ),
            2542796800
        );
        for suffix in ["dt_bias", "A_log", "conv1d.weight", "norm.weight"] {
            assert_eq!(
                bytes(
                    &q8,
                    &format!("model.language_model.layers.0.linear_attn.{suffix}"),
                    SafeDType::BF16,
                    &[128]
                ),
                512
            );
        }
        // Zero-centered norm gains stay F32 (`1 + w`) under every plan.
        for name in [
            "mtp.layers.0.input_layernorm.weight",
            "model.language_model.norm.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ] {
            assert_eq!(bytes(&q8, name, SafeDType::BF16, &[5120]), 20480, "{name}");
        }
        // CUDA keeps the dense alpha/beta pair and, for the router GEMV, an
        // F32 copy of its rows; with the fused decode kernels off it keeps
        // only the dense pair.
        for name in ["in_proj_a", "in_proj_b"] {
            assert_eq!(
                bytes(
                    &q8,
                    &format!("model.language_model.layers.0.linear_attn.{name}.weight"),
                    SafeDType::BF16,
                    &[48, 5120]
                ),
                48 * 5120 * (2 + 4),
                "{name}"
            );
        }
        let mut fused_off = PerformanceConfig::default();
        fused_off.cuda.fused_decode = crate::performance::OptimizationMode::Off;
        let dense_only = Qwen38ResidencyPolicy::resolve_static(BackendKind::Cuda, &fused_off);
        assert_eq!(
            bytes(
                &dense_only,
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                SafeDType::BF16,
                &[48, 5120]
            ),
            48 * 5120 * 2
        );
        // The CPU never resolves the GEMV in production.
        let cpu =
            Qwen38ResidencyPolicy::resolve_static(BackendKind::Cpu, &PerformanceConfig::default());
        assert_eq!(
            bytes(
                &cpu,
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                SafeDType::BF16,
                &[48, 5120]
            ),
            48 * 5120 * 4
        );
    }

    #[test]
    fn malformed_inventory_fails_before_allocation() {
        let block = [128, 128];
        let q8 =
            Qwen38ResidencyPolicy::resolve_static(BackendKind::Cuda, &PerformanceConfig::default());
        assert!(q8
            .resident_bytes("projection.weight", SafeDType::F8_E4M3, &[128, 33], block)
            .is_err());
        assert!(q8
            .resident_bytes("projection.weight", SafeDType::F8_E4M3, &[0, 128], block)
            .is_err());
        assert!(q8
            .resident_bytes("x", SafeDType::BF16, &[usize::MAX, usize::MAX], block)
            .is_err());
        assert!(q8
            .resident_bytes("x", SafeDType::U8, &[128], block)
            .is_err());
    }

    /// Admission and the loader resolve one policy: on the CPU stand-in for
    /// the FP8 kernels, admission charges raw FP8 (bytes plus F32 block
    /// scales); once a load-time check poisons the checkpoint, both the next
    /// admission and the next load resolve Q8_0, and admission charges Q8_0.
    #[test]
    fn inventory_prices_native_fp8_when_resolved_and_q8_when_poisoned() {
        use crate::models::architectures::qwen38::mtp::tests::{
            write_fp8_geometry_checkpoint, TestDir,
        };
        use crate::models::architectures::qwen38::native::ENV_ALLOW_SYNTHETIC_QWEN38_GEOMETRY;
        use crate::models::architectures::qwen38::residency::{
            force_cpu_native_fp8_for_tests, Qwen38ProjectionPlan,
        };
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        struct Synthetic;
        impl Drop for Synthetic {
            fn drop(&mut self) {
                std::env::remove_var(ENV_ALLOW_SYNTHETIC_QWEN38_GEOMETRY);
            }
        }
        std::env::set_var(ENV_ALLOW_SYNTHETIC_QWEN38_GEOMETRY, "1");
        let _synthetic = Synthetic;
        let dir = TestDir::new("admission-fp8");
        let config = write_fp8_geometry_checkpoint(dir.path());
        let device = candle_core::Device::Cpu;
        let performance = PerformanceConfig::default();
        let resident = |plan: &ModelResourcePlan| match plan.resident_authorization.host_bytes {
            ResourceAmount::Known(bytes) => bytes,
            ResourceAmount::Unknown => panic!("CPU residency must be known"),
        };
        let mut expanded_config = PerformanceConfig::default();
        expanded_config.cpu.projection_backend = crate::performance::CpuProjectionBackend::Expanded;
        let expanded = resident(&resource_plan(dir.path(), &device, &expanded_config).unwrap());
        // The CPU default: Q8_0 projections and head (34 bytes per 32).
        let cpu_q8 = resident(&resource_plan(dir.path(), &device, &performance).unwrap());
        let _forced = force_cpu_native_fp8_for_tests();
        let native = resident(&resource_plan(dir.path(), &device, &performance).unwrap());

        // The fixture's FP8 elements (trunk + MTP) and scale grid, priced
        // independently: 3 DeltaNet + 1 attention + 1 MTP layer.
        let (hidden, ff) = (256u64, 512u64);
        let mlp = 3 * ff * hidden;
        let linear = 2048 * hidden + hidden * 768;
        let attention = 1536 * hidden + hidden * 512;
        let fp8 = 3 * (mlp + linear) + 2 * (mlp + attention);
        let scales = |n: u64, k: u64| n.div_ceil(128) * k.div_ceil(128);
        let scale_grid = 5 * (2 * scales(ff, hidden) + scales(hidden, ff))
            + 3 * (scales(1280, hidden) + scales(768, hidden) + scales(hidden, 768))
            + 2 * (scales(1024, hidden) + 2 * scales(256, hidden) + scales(hidden, 512));
        // Expanded F32 (4 bytes) vs raw FP8 (1 byte + 4 per block scale);
        // the [512, 256] LM head goes from F32 to block FP8 as well.
        let head = 512 * 256;
        let head_fp8 = head + scales(512, 256) * 4;
        assert_eq!(
            expanded - native,
            fp8 * 3 - scale_grid * 4 + head * 4 - head_fp8
        );
        assert_eq!(
            expanded - cpu_q8,
            fp8 * 4 - fp8 / 32 * 34 + head * 4 - head / 32 * 34
        );

        let policy =
            crate::models::architectures::qwen38::residency::Qwen38ResidencyPolicy::resolve(
                &device,
                crate::models::architectures::qwen38::native::ProjectionMaterialization::F32,
                &performance,
                &crate::models::architectures::qwen38::native::IndexedSafetensors::open(dir.path())
                    .unwrap(),
                &config,
            );
        assert_eq!(
            policy.plan(),
            Qwen38ProjectionPlan::NativeFp8 { explicit: false }
        );
        policy.poison_projections("forced load-time mismatch");
        let q8 = resident(&resource_plan(dir.path(), &device, &performance).unwrap());
        // Q8_0 projections on the CPU take the CPU's Q8_0 head.
        assert_eq!(q8, cpu_q8);
        assert_eq!(
            q8 - native,
            fp8 / 32 * 34 - fp8 - scale_grid * 4 + head / 32 * 34 - head_fp8
        );
        let loader_policy =
            crate::models::architectures::qwen38::residency::Qwen38ResidencyPolicy::resolve(
                &device,
                crate::models::architectures::qwen38::native::ProjectionMaterialization::F32,
                &performance,
                &crate::models::architectures::qwen38::native::IndexedSafetensors::open(dir.path())
                    .unwrap(),
                &config,
            );
        assert_eq!(loader_policy.plan(), Qwen38ProjectionPlan::Q8);
    }

    /// Metal admission resolves the policy the Metal loader uses: raw block
    /// FP8 plus the F16 prefill scratch (also a pending claim), the F16
    /// expansion when `metal.projection_backend = "expanded"`, and the
    /// expansion again once a load-time check poisons the checkpoint.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_admission_prices_raw_fp8_and_the_expansion_once_poisoned() {
        use crate::models::architectures::qwen38::mtp::tests::{
            write_fp8_geometry_checkpoint, TestDir,
        };
        use crate::models::architectures::qwen38::native::ENV_ALLOW_SYNTHETIC_QWEN38_GEOMETRY;
        use crate::models::architectures::qwen38::residency::{
            Qwen38ProjectionPlan, METAL_FP8_PREFILL_DEQUANT_BYTES,
        };
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_admission_prices_raw_fp8_and_the_expansion_once_poisoned",
        ) else {
            return;
        };
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        struct Synthetic;
        impl Drop for Synthetic {
            fn drop(&mut self) {
                std::env::remove_var(ENV_ALLOW_SYNTHETIC_QWEN38_GEOMETRY);
            }
        }
        std::env::set_var(ENV_ALLOW_SYNTHETIC_QWEN38_GEOMETRY, "1");
        let _synthetic = Synthetic;
        let dir = TestDir::new("admission-metal-fp8");
        let config = write_fp8_geometry_checkpoint(dir.path());
        let unified = |plan: &ModelResourcePlan| match plan.resident_authorization.unified_bytes {
            ResourceAmount::Known(bytes) => bytes,
            ResourceAmount::Unknown => panic!("Metal residency must be known"),
        };
        let native_plan =
            resource_plan(dir.path(), &device, &PerformanceConfig::default()).unwrap();
        assert_eq!(
            native_plan.deferred_resident_authorization.unified_bytes,
            ResourceAmount::Known(METAL_FP8_PREFILL_DEQUANT_BYTES)
        );
        let native = unified(&native_plan);
        let mut expanded_config = PerformanceConfig::default();
        expanded_config.metal.projection_backend =
            crate::performance::MetalProjectionBackend::Expanded;
        let expanded = unified(&resource_plan(dir.path(), &device, &expanded_config).unwrap());
        // FP8 elements (trunk + MTP) and their block scales, as in the CPU test.
        let (hidden, ff) = (256u64, 512u64);
        let mlp = 3 * ff * hidden;
        let fp8 =
            3 * (mlp + 2048 * hidden + hidden * 768) + 2 * (mlp + 1536 * hidden + hidden * 512);
        let scales = |n: u64, k: u64| n.div_ceil(128) * k.div_ceil(128);
        let scale_grid = 5 * (2 * scales(ff, hidden) + scales(hidden, ff))
            + 3 * (scales(1280, hidden) + scales(768, hidden) + scales(hidden, 768))
            + 2 * (scales(1024, hidden) + 2 * scales(256, hidden) + scales(hidden, 512));
        assert_eq!(
            expanded + METAL_FP8_PREFILL_DEQUANT_BYTES + scale_grid * 4,
            native + fp8
        );
        let policy = Qwen38ResidencyPolicy::resolve(
            &device,
            crate::models::architectures::qwen38::native::ProjectionMaterialization::F16,
            &PerformanceConfig::default(),
            &crate::models::architectures::qwen38::native::IndexedSafetensors::open(dir.path())
                .unwrap(),
            &config,
        );
        assert_eq!(
            policy.plan(),
            Qwen38ProjectionPlan::NativeFp8 { explicit: false }
        );
        policy.poison_projections("forced Metal mismatch");
        let poisoned = resource_plan(dir.path(), &device, &PerformanceConfig::default()).unwrap();
        assert_eq!(unified(&poisoned), expanded);
    }
}
