//! CUDA graphs of SOV-Transcribe-1's fixed-shape work.
//!
//! * **Encoder** (`kernels::cuda::shape_graphs`): the Whisper encoder and the
//!   projector over a batch of 30 s windows is one pure function of the
//!   batch's features, and a batch has one of eight shapes (1 to
//!   [`super::model::ENCODER_BATCH_ENV`] windows). Each shape is captured
//!   once, checked bit for bit against the eager run twice, then replayed.
//! * **Decode** (`kernels::cuda::stretch_graphs`): a decode step's work
//!   between two layers' paged attention (the output projection, norms,
//!   dense MLP, fused experts, and the next layer's projections, norms and
//!   rotation) is one stretch per layer, captured per row count.
//!
//! Any graph failure turns that set of graphs off for the process and the
//! work runs eagerly; `encoder_graphs` and `decode_graphs` in the
//! diagnostics report captures, replays and why graphs are off.

use candle_core::Device;

use crate::backends::BackendKind;
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::kernels::cuda::stretch_graphs::StretchGraphs;

/// Kill switch for the encoder graphs (`0`, `off`, `false`; read at load).
pub const ENCODER_GRAPHS_ENV: &str = "IZWI_SOV_TRANSCRIBE_ENCODER_GRAPHS";

/// Kill switch for the decode-step graphs (`0`, `off`, `false`; read at
/// load). The global `IZWI_CUDA_DECODE_GRAPHS=off` turns them off too.
pub const DECODE_GRAPHS_ENV: &str = "IZWI_SOV_TRANSCRIBE_DECODE_GRAPHS";

const MIB: u64 = 1 << 20;

/// One graph per batch size; each keeps its features (1.5 MiB a window)
/// and its BF16 audio embeddings (2 MiB a window). The pool holds the
/// largest batch's working set: per window about 140 MiB of encoder
/// activations through the 5120-wide MLP.
pub const ENCODER_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 8,
    retained_bytes: 160 * MIB,
    pool_bytes: 1_536 * MIB,
};

fn encoder_graphs_requested() -> bool {
    crate::env::flag_or(ENCODER_GRAPHS_ENV, true)
}

/// Device bytes a load on `backend` prices for the encoder graphs, which
/// allocate on their first batches: none off CUDA or with them off.
pub fn encoder_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend == BackendKind::Cuda && encoder_graphs_requested() {
        ENCODER_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// The encoder's graphs on `device`, or why it has none.
pub fn resolve_encoder(device: &Device) -> Result<ShapeGraphs<usize>, String> {
    if !encoder_graphs_requested() {
        return Err(format!("{ENCODER_GRAPHS_ENV} is off"));
    }
    ShapeGraphs::resolve("SOV-Transcribe encoder", device, ENCODER_GRAPH_LIMITS)
}

/// The row counts a decode step's graphs are captured at: a step pads its
/// rows to the next of these, so every width up to the fitted 32 replays a
/// graph (the stretches hold no per-row state, only the step's activations).
pub const DECODE_GRAPH_WIDTHS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// The captured width a step of `rows` rows pads to, or `None` past the
/// widest.
pub fn decode_graph_width(rows: usize) -> Option<usize> {
    DECODE_GRAPH_WIDTHS
        .iter()
        .copied()
        .find(|&width| width >= rows)
}

fn graphed_decode_width(rows: usize) -> bool {
    DECODE_GRAPH_WIDTHS.contains(&rows)
}

/// The decode step's graphs for `stretches` stretches on `device`, or why
/// it has none.
pub fn resolve_decode(device: &Device, stretches: usize) -> Result<StretchGraphs, String> {
    Ok(StretchGraphs::resolve(
        "SOV-Transcribe decoder",
        DECODE_GRAPHS_ENV,
        device,
        stretches,
    )?
    .with_widths(graphed_decode_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_steps_pad_to_the_next_captured_width() {
        assert_eq!(decode_graph_width(1), Some(1));
        assert_eq!(decode_graph_width(3), Some(4));
        assert_eq!(decode_graph_width(9), Some(16));
        assert_eq!(decode_graph_width(32), Some(32));
        assert_eq!(decode_graph_width(33), None);
        assert!(DECODE_GRAPH_WIDTHS.iter().all(|&w| graphed_decode_width(w)));
        assert!(!graphed_decode_width(3));
    }

    #[test]
    fn graphs_are_cuda_only() {
        assert!(resolve_encoder(&Device::Cpu).is_err());
        assert!(resolve_decode(&Device::Cpu, 31).is_err());
        assert_eq!(encoder_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(encoder_graph_device_bytes(BackendKind::Metal), 0);
    }
}
