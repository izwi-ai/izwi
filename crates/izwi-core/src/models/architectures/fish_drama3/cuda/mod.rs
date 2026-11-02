//! The CUDA residency (§6, CUDA column; D5), on GPUs of compute capability
//! 8.0 or newer. Older devices are refused before any weight is read.
//!
//! | Part | Form on CUDA |
//! |---|---|
//! | Residual stream | BF16, as upstream |
//! | Dense projections (trunk, shared expert, tower, heads) | BF16 weights; BF16 activations in and out |
//! | Packing at load | trunk `q_proj` / `k_proj` / `v_proj` and `in_proj_qkv` / `in_proj_z` as one matrix each; the tower's contiguous `wqkv` repacked per head like the trunk's |
//! | Step projections, router, tower and heads | the row-invariant multi-row GEMV (`kernels::cuda::row_gemv`), weights read once per step; one GEMV per row if its device check fails |
//! | Span projections | cuBLAS GEMM |
//! | Routed experts | raw E4M3 + F32 128×128 scales stacked per layer; the fused per-(token, slot) kernels in a step (the grouped path never runs there), the grouped kernels in a span of 32 rows or more |
//! | Shared expert | BF16, unfolded, gated by its F32 sigmoid gate |
//! | Router, shared-expert gate, DeltaNet `a` / `b`, norm gains, conv, `A_log`, `dt_bias` | F32 |
//! | RMSNorm, q/k norm, attention gate | the Qwen3.6 BF16 kernels with F32 gains |
//! | RoPE | host F64 tables rounded to F32, rotated in F32 (NeoX for the trunk, interleaved pairs for the tower) |
//! | DeltaNet | `gdn_conv` + `gdn_decode` per row in steps and spans under 32 rows; from 32 rows the causal conv over the span and the chunked prefill; F32 state |
//! | Paged KV (both domains) | BF16; a decode row's split-decode partitions come from its own length (`SplitDecodePartitioning::RowLength`) |
//! | Embedding and code tables | BF16 device tables, gathered to F32 |
//! | Logits | BF16 projections into F32 |
//! | Sampling | on the device, one readback a frame (two with the fused path off) (`device::sampling`) |
//!
//! The forward code is the device column's (`device/`), shared with Metal;
//! this module holds what only CUDA runs: the SM80 floor, the load-time
//! checks (`self_check.rs`) and the CUDA device tests.
//!
//! **How it differs from Metal.** BF16 has F32's exponent range, so there
//! is no range probe and no per-layer F32 fallback: every layer runs BF16
//! with a BF16 residual, as upstream does, and the paged KV is always BF16.
//! The host expert loop (`IZWI_FISH_DRAMA3_MOE_BACKEND=loop`) runs BF16
//! activations, since the CUDA FP8 GEMV takes no F32 input. A step's
//! projections read their weights once for every row instead of once per
//! row. The kernel-side checks gate a kernel, never the load: a failed
//! row-invariant GEMV check leaves steps on one GEMV per row, a failed
//! chunked DeltaNet check leaves spans on the per-row kernels, and a failed
//! tensor-core check leaves the grouped MoE on its warp kernels. The FP8 and
//! RoPE table checks refuse the load: no slower path fixes a wrong scale or
//! rotation.
//!
//! `tests.rs` holds the device tests: component parity against the CPU
//! reference, batch invariance at widths 1, 2, 4 and 8, and the load checks.

pub(crate) mod self_check;
#[cfg(all(test, feature = "cuda"))]
mod tests;

use candle_core::Device;

use crate::error::{Error, Result};

/// Refuses anything but a CUDA device of compute capability 8.0 or newer
/// (Ampere and later): BF16 arithmetic and the FP8 kernels' PTX need it.
pub(crate) fn require_sm80(device: &Device) -> Result<()> {
    if device.is_cuda() && crate::kernels::cuda::fp8::device_is_sm80_or_newer(device) {
        return Ok(());
    }
    Err(Error::ModelLoadError(
        "Fish Audio Drama 3 on CUDA needs a GPU of compute capability 8.0 or newer (Ampere or \
         later), for BF16 and its FP8 kernels; this device is older or not a CUDA device"
            .into(),
    ))
}
