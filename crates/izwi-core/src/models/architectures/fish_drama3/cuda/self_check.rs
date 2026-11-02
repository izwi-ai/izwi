//! A CUDA load's checks (§6 "Self-checks"), each with its counter in
//! [`FishDrama3CudaChecks`]:
//!
//! * **Host RoPE tables** and **FP8 GEMV vs F64 dequant** against the
//!   original tensors: the device column's table checks (`device::checks`),
//!   with BF16 activations. Either breach refuses the load: CUDA's FP8 GEMV
//!   takes no F32 activations to fall back on, and no slower path fixes a
//!   wrong scale, stacking or rotation.
//! * **Row-invariant multi-row GEMV** (`kernels::cuda::row_gemv::verify`):
//!   the kernel against its host reference at 1 to 64 rows, every row's bits
//!   the same at every row count, before any step uses it. A failure leaves
//!   steps on one GEMV per row, which is row-invariant too, only slower.
//! * **Chunked DeltaNet prefill** (`kernels::cuda::verify_chunked_deltanet`)
//!   against the sequential kernel at the trunk's head geometry. A failure
//!   leaves long spans on the per-row kernels.
//! * **Grouped MoE tensor-core tiles** (`kernels::cuda::moe::
//!   verify_grouped_mma`) against the warp kernels at the trunk's widths.
//!   The verdict is kept by the load (the process-wide switch other families
//!   read is left alone); a failure leaves spans on the warp kernels.

use candle_core::DType;

use super::super::checkpoint::FishDrama3Checkpoint;
use super::super::config::FishDrama3Config;
use super::super::device::checks;
use super::super::device::trunk::DeviceMixer;
use super::super::precision::FishDrama3CudaChecks;
use super::super::trunk::FishDrama3Trunk;
use crate::error::Result;
use crate::kernels::cuda::moe::{self, GroupedKernel};

const BACKEND: &str = "CUDA";

/// Every check of a CUDA load, switching each verified kernel on in
/// `trunk`; returns the report.
pub(crate) fn run(
    checkpoint: &FishDrama3Checkpoint,
    config: &FishDrama3Config,
    trunk: &mut FishDrama3Trunk,
) -> Result<FishDrama3CudaChecks> {
    let device = trunk.device().clone();
    let mut report = FishDrama3CudaChecks {
        activations: format!("{:?}", DType::BF16),
        kv_dtype: format!("{:?}", DType::BF16),
        ..Default::default()
    };
    checks::check_rope(
        &[
            ("trunk", config.trunk.attention),
            ("tower", config.tower.attention),
        ],
        &device,
        BACKEND,
        &mut report.rope_checks,
        &mut report.rope_table_breaches,
    )?;
    checks::check_fp8(
        checkpoint,
        config,
        trunk
            .device_trunk()
            .ok_or_else(|| checks::load_error(BACKEND, "the trunk holds no device weights"))?,
        DType::BF16,
        None,
        BACKEND,
        &mut report.fp8_checks,
        &mut report.fp8_gemv_breaches,
    )?;

    report.row_gemv = match crate::kernels::cuda::row_gemv::verify_and_enable(&device) {
        Ok(()) => "verified".into(),
        Err(error) => {
            report.row_gemv_breaches = 1;
            tracing::warn!(
                "Fish Audio Drama 3 on CUDA: the row-invariant GEMV failed its device check \
                 ({error}); decode steps run one GEMV per row"
            );
            format!("off: {error}")
        }
    };

    let la = config.trunk.linear;
    let chunked =
        if !crate::kernels::cuda::chunked_deltanet_supported(la.key_head_dim, la.value_head_dim) {
            report.chunked_deltanet = format!(
                "off: heads of {} / {} are outside the chunked kernels",
                la.key_head_dim, la.value_head_dim
            );
            false
        } else {
            match crate::kernels::cuda::verify_chunked_deltanet(
                &device,
                la.value_heads,
                la.key_head_dim,
                la.value_head_dim,
            ) {
                Ok(()) => {
                    report.chunked_deltanet = "verified".into();
                    true
                }
                Err(error) => {
                    report.chunked_deltanet_breaches = 1;
                    tracing::warn!(
                    "Fish Audio Drama 3 on CUDA: the chunked DeltaNet prefill failed its device \
                     check ({error}); spans run the per-row kernels"
                );
                    report.chunked_deltanet = format!("off: {error}");
                    false
                }
            }
        };

    let (hidden, inter, experts) = (
        config.trunk.hidden,
        config.trunk.moe.intermediate,
        config.trunk.moe.experts,
    );
    let grouped = if !moe::grouped_supported(hidden, inter, experts) {
        report.grouped_moe =
            format!("none: hidden {hidden} / expert width {inter} are outside the grouped kernels");
        GroupedKernel::Warp
    } else {
        match moe::verify_grouped_mma(&device, hidden, inter) {
            Ok(()) => {
                report.grouped_moe = "tensor-core".into();
                GroupedKernel::TensorCore
            }
            Err(error) => {
                report.grouped_mma_breaches = 1;
                tracing::warn!(
                    "Fish Audio Drama 3 on CUDA: the tensor-core grouped MoE failed its device \
                     check ({error}); spans run the warp kernels"
                );
                report.grouped_moe = format!("warp: {error}");
                GroupedKernel::Warp
            }
        }
    };

    let weights = trunk
        .device_trunk_mut()
        .ok_or_else(|| checks::load_error(BACKEND, "the trunk holds no device weights"))?;
    for layer in &mut weights.layers {
        layer.moe.grouped = grouped;
        if let DeviceMixer::DeltaNet { net, .. } = &mut layer.mixer {
            net.chunked = chunked;
        }
    }
    Ok(report)
}
