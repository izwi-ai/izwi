//! Whole-step CUDA graphs of Voxtral's language-model decode, for the
//! realtime model's decode rows (live streams and the offline rows that run
//! as closed streams, up to 32 on CUDA). A frame step is host-bound on an
//! H100 (about 6.3 ms a frame at one row against about 2 ms of weight
//! reads), so the step from the frame embeddings to the logits is captured
//! once per padded width ([`decode_step::WIDTHS`], 1 to 32 rows) and
//! replayed.
//!
//! The graph body is the eager decode with the step's per-row values moved
//! to the device: each row's position, flat KV slot and block table come
//! from the step buffer ([`decode_step`]); RoPE rows are gathered from the
//! model's position table by those positions; K/V rows are written by
//! device slot and attention runs the split decode with row-invariant
//! partitions. Ada RMSNorm reads the model's precomputed delay gains. The
//! projections are dense, as eager. Spare rows decode a zero embedding at
//! position 0 into no KV slot.
//!
//! Per width, [`ShapeGraphs`] checks a capture against the eager body bit
//! for bit on its first two steps; the model compares a width's first
//! graphed step with its standard decode ([`decode_step::logits_agree`]).
//! Any failure turns the graphs off for the process, and every step runs
//! the standard decode. Off CUDA there are no graphs.
//!
//! [`decode_step`]: crate::kernels::cuda::decode_step
//! [`decode_step::WIDTHS`]: crate::kernels::cuda::decode_step::WIDTHS
//! [`decode_step::logits_agree`]: crate::kernels::cuda::decode_step::logits_agree

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::Device;
use serde_json::json;

use crate::backends::BackendKind;
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off;
use crate::kv::KvArenaId;

/// Kill switch (a boolean; on when unset). Read by admission pricing and the
/// loader.
pub(crate) const VOXTRAL_DECODE_GRAPHS_ENV: &str = "IZWI_VOXTRAL_DECODE_GRAPHS";

const MIB: u64 = 1024 * 1024;

/// What the decode graphs may hold. Each kept graph retains its stable
/// embeddings and step buffer and its F32 logits (`width x 131,072`: 108
/// rows across the ten widths are 57 MB); the pool holds the widest step's
/// working set beside them (the 32-row logits, split-decode partials and
/// MLP activations, about 64 MB).
pub(crate) const DECODE_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 16,
    retained_bytes: 96 * MIB,
    pool_bytes: 160 * MIB,
};

/// Whether the switch and `cuda.decode_graphs` leave the graphs on.
pub(crate) fn decode_graphs_requested() -> bool {
    crate::env::flag_or(VOXTRAL_DECODE_GRAPHS_ENV, true) && !decode_graphs_switched_off()
}

/// Device bytes a load on `backend` prices for the decode graphs, which
/// allocate lazily on their first steps: zero off CUDA or with the graphs
/// switched off.
pub(crate) fn voxtral_decode_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend == BackendKind::Cuda && decode_graphs_requested() {
        DECODE_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// What a graph's captured addresses depend on beside its inputs: its
/// width, the KV arena's backing generation (growth swaps the caches), the
/// step buffer's block-table width and the RoPE position table (rebuilt
/// larger as positions grow).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GraphKey {
    pub(super) width: usize,
    pub(super) arena: KvArenaId,
    pub(super) generation: u64,
    pub(super) max_blocks: usize,
    pub(super) rope_rows: usize,
}

/// The language model's decode graphs.
pub(crate) struct VoxtralDecodeGraph {
    pub(super) graphs: ShapeGraphs<GraphKey>,
    /// Widths whose first graphed step matched the standard decode.
    checked: Mutex<Vec<usize>>,
    /// Why the warm check turned the graphs off.
    off: Mutex<Option<String>>,
    graphed_steps: AtomicU64,
    spare_rows: AtomicU64,
    warm_checks: AtomicU64,
}

impl VoxtralDecodeGraph {
    /// The decode graphs on `device`, or why there are none: CUDA only,
    /// with the switch and `cuda.decode_graphs` on.
    pub(super) fn resolve(device: &Device) -> std::result::Result<Self, String> {
        if !crate::env::flag_or(VOXTRAL_DECODE_GRAPHS_ENV, true) {
            return Err(format!("{VOXTRAL_DECODE_GRAPHS_ENV} is off"));
        }
        if decode_graphs_switched_off() {
            return Err("cuda.decode_graphs=off".into());
        }
        Ok(Self {
            graphs: ShapeGraphs::resolve("Voxtral decode", device, DECODE_GRAPH_LIMITS)?,
            checked: Mutex::new(Vec::new()),
            off: Mutex::new(None),
            graphed_steps: AtomicU64::new(0),
            spare_rows: AtomicU64::new(0),
            warm_checks: AtomicU64::new(0),
        })
    }

    /// Whether steps still go through the graphs.
    pub(super) fn is_on(&self) -> bool {
        self.graphs.is_on() && self.off.lock().is_ok_and(|off| off.is_none())
    }

    /// Turns the graphs off for the process.
    pub(super) fn disable(&self, reason: String) {
        tracing::warn!("Voxtral CUDA decode graphs are off: {reason}");
        if let Ok(mut off) = self.off.lock() {
            off.get_or_insert(reason);
        }
    }

    /// Whether a step of `width` still needs its check against the
    /// standard decode.
    pub(super) fn needs_check(&self, width: usize) -> bool {
        !self
            .checked
            .lock()
            .is_ok_and(|checked| checked.contains(&width))
    }

    pub(super) fn mark_checked(&self, width: usize) {
        self.warm_checks.fetch_add(1, Relaxed);
        if let Ok(mut checked) = self.checked.lock() {
            if !checked.contains(&width) {
                checked.push(width);
            }
        }
    }

    pub(super) fn record_step(&self, rows: usize, width: usize) {
        self.graphed_steps.fetch_add(1, Relaxed);
        self.spare_rows
            .fetch_add(width.saturating_sub(rows) as u64, Relaxed);
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let checked = self
            .checked
            .lock()
            .map(|checked| checked.clone())
            .unwrap_or_default();
        json!({
            "off": self.off.lock().ok().and_then(|off| off.clone()),
            "graphed_steps": self.graphed_steps.load(Relaxed),
            "spare_rows": self.spare_rows.load(Relaxed),
            "warm_checks": self.warm_checks.load(Relaxed),
            "checked_widths": checked,
            "graphs": self.graphs.diagnostics(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The graphs price their memory on CUDA loads with their switch on,
    /// and never resolve off CUDA or with a switch off.
    #[test]
    fn decode_graphs_price_and_resolve_on_cuda_only() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(voxtral_decode_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(voxtral_decode_graph_device_bytes(BackendKind::Metal), 0);
        let reason = VoxtralDecodeGraph::resolve(&Device::Cpu)
            .err()
            .expect("no graphs on CPU");
        if decode_graphs_requested() {
            assert!(reason.contains("CUDA only"), "{reason}");
            assert_eq!(
                voxtral_decode_graph_device_bytes(BackendKind::Cuda),
                256 << 20
            );
        }
        for (name, value, expected) in [
            (VOXTRAL_DECODE_GRAPHS_ENV, "0", VOXTRAL_DECODE_GRAPHS_ENV),
            ("IZWI_CUDA_DECODE_GRAPHS", "off", "cuda.decode_graphs=off"),
        ] {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            let bytes = voxtral_decode_graph_device_bytes(BackendKind::Cuda);
            let reason = VoxtralDecodeGraph::resolve(&Device::Cpu).err();
            std::env::remove_var(name);
            if let Some(previous) = previous {
                std::env::set_var(name, previous);
            }
            assert_eq!(bytes, 0, "{name}");
            assert!(reason.expect("off").contains(expected), "{name}");
        }
    }
}
