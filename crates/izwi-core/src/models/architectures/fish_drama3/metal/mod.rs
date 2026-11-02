//! The Metal residency (§6, Metal column; D5).
//!
//! | Part | Form on Metal |
//! |---|---|
//! | Residual stream | F32 |
//! | Dense projections (trunk, shared expert, tower, heads) | F16 weights; F16 activations in and out |
//! | Packing at load | trunk `q_proj` / `k_proj` / `v_proj` and `in_proj_qkv` / `in_proj_z` as one matrix each; the tower's contiguous `wqkv` repacked per head like the trunk's |
//! | Routed experts | raw E4M3 + F32 128×128 scales stacked per layer, the fused Qwen3.6 MoE kernels (F16 in and out) |
//! | Router, shared-expert gate, DeltaNet `a` / `b`, norm gains, conv, `A_log`, `dt_bias` | F32 |
//! | RMSNorm, q/k norm, attention gate | the Qwen3.6 F16 kernels with F32 gains (a norm reads the F32 residual through F16) |
//! | RoPE | host F64 tables rounded to F32, rotated in F32 (NeoX for the trunk, interleaved pairs for the tower) |
//! | DeltaNet | `gdn_conv` + `gdn_decode` (F32 gates) row by row in steps and spans, F32 state; heads other than 128-dim run the same rule as a Candle composition |
//! | Router | F32 GEMV, the fused routing kernel, weights rounded to BF16 |
//! | Paged KV (both domains) | F16, F32 after a breached load-time probe |
//! | Embedding and code tables | F16 device tables, gathered to F32 |
//! | Logits | F16 projections widened to F32 |
//! | Sampling | on the device, two readbacks a frame (`device::sampling`) |
//!
//! The forward code is the device column's (`device/`), which CUDA shares in
//! its own forms; this module holds what only Metal runs: the F16 range
//! probe (`self_check.rs`) and the Metal device tests.
//!
//! Every output of an F16 projection is added into the F32 residual, so
//! the stream itself never rounds to 16 bits; what does is each layer's
//! activations. The load-time probe (`self_check.rs`) runs every layer on
//! xorshift input and moves a layer whose activations leave F16's safe
//! range to F32 activations (its F16 weights widened per call), and a
//! breached MoE layer to the host-routed expert loop.
//!
//! **Kernel phases (§7.4).** A step's rows run together layer by layer, but
//! no reduction ever spans two rows: each projection runs one GEMV per row,
//! attention reads each row's own paged view, the DeltaNet kernels each
//! row's own state, and the fused MoE and norm kernels compute every token
//! on its own. So a row's bits never depend on its batch. A span (one
//! request's prefill) runs one GEMM per projection and, from 32 rows, the
//! grouped expert kernels when the geometry allows them; the DeltaNet
//! walks the span's rows through the same per-row kernels, so prefill and
//! decode share one recurrence.

pub(crate) mod self_check;
#[cfg(test)]
mod tests;

use candle_core::DType;

/// The activation dtype of a healthy Metal layer.
pub(crate) const HALF: DType = DType::F16;
