//! CUDA graphs of the fused decode step's stretches of Mamba and MoE layers
//! (`kernels::cuda::stretch_graphs`): each a pure function of the residual,
//! the normed hidden and its layers' Mamba states.

use candle_core::Device;

use crate::kernels::cuda::stretch_graphs::StretchGraphs;

/// Kill switch for the stretch graphs (`0`, `off`, `false`): every stretch
/// runs eagerly. The global `IZWI_CUDA_DECODE_GRAPHS=off` turns them off
/// too. Read at load.
pub const SEGMENT_GRAPHS_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_SEGMENT_GRAPHS";

/// The graphs of a decoder with `stretches` stretches on `device`, or why it
/// has none.
pub(super) fn resolve(device: &Device, stretches: usize) -> Result<StretchGraphs, String> {
    StretchGraphs::resolve(
        "Nemotron 3.5 Transcribe",
        SEGMENT_GRAPHS_ENV,
        device,
        stretches,
    )
}
