//! Whole-step CUDA graphs of the Qwen3 trunk's batched decode, for the ASR
//! families that share `qwen3::core`: Qwen3-ASR and VibeVoice-ASR. Their
//! decode steps are host-bound on an H100 (about 20 launches a layer, and
//! the 1.7B model decodes within 5% of the 0.6B), so the whole step, from
//! the token embeddings to the logits, is captured once per padded width
//! ([`decode_step::WIDTHS`], 1 to 32 rows) and replayed.
//!
//! The graph body is the eager decode with three swaps, each recorded by a
//! capture:
//! - quantized projections run the izwi-owned K-quant MMVQ/MMQ
//!   (`kernels::cuda::kquant`), bit for bit what Candle's `QMatMul` gives
//!   but over caller-owned buffers (`QMatMul` keeps process-global scratch
//!   that a graph cannot capture);
//! - each row's position, flat KV slot and block table come from the step
//!   buffer ([`decode_step`]), its K/V rows are written by device slot, and
//!   attention runs the split decode with row-invariant partitions;
//! - spare rows decode token 0 at position 0 into no KV slot.
//!
//! Per width, [`ShapeGraphs`] checks a capture against the eager body bit
//! for bit on its first two steps; the model compares a width's first
//! graphed step with its standard decode ([`decode_step::logits_agree`]).
//! Any failure turns the graphs off for the process, and every step runs
//! the standard decode. Off CUDA there are no graphs.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::Device;
use serde_json::json;

use crate::backends::BackendKind;
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off;
use crate::kv::KvArenaId;

/// Qwen3-ASR's kill switch (a boolean; on when unset).
pub(crate) const QWEN3_ASR_DECODE_GRAPHS_ENV: &str = "IZWI_QWEN3_ASR_DECODE_GRAPHS";
/// VibeVoice-ASR's kill switch (a boolean; on when unset).
pub(crate) const VIBEVOICE_ASR_DECODE_GRAPHS_ENV: &str = "IZWI_VIBEVOICE_ASR_DECODE_GRAPHS";

const MIB: u64 = 1024 * 1024;

/// What a model's decode graphs may hold. Each kept graph retains its
/// stable step buffer and token ids and its logits output (`width x vocab`
/// F32 at most: 108 rows across the ten widths are 66 MB at a 152k vocab);
/// the pool holds the widest step's working set beside them (the 32-row
/// logits, split-decode partials and MLP activations, about 64 MB).
pub(crate) const DECODE_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 16,
    retained_bytes: 96 * MIB,
    pool_bytes: 160 * MIB,
};

/// Whether `switch` and `cuda.decode_graphs` leave the decode graphs on.
pub(crate) fn decode_graphs_requested(switch: &'static str) -> bool {
    crate::env::flag_or(switch, true) && !decode_graphs_switched_off()
}

/// Device bytes a load on `backend` prices for the decode graphs behind
/// `switch`, which allocate lazily on their first steps: zero off CUDA or
/// with the graphs switched off.
pub(crate) fn qwen3_decode_graph_device_bytes(backend: BackendKind, switch: &'static str) -> u64 {
    if backend == BackendKind::Cuda && decode_graphs_requested(switch) {
        DECODE_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// What a graph's captured addresses depend on beside its inputs: its
/// width, the KV arena's backing generation (growth swaps the caches), the
/// step buffer's block-table width and the RoPE table (rebuilt larger as
/// positions grow).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GraphKey {
    pub(super) width: usize,
    pub(super) arena: KvArenaId,
    pub(super) generation: u64,
    pub(super) max_blocks: usize,
    pub(super) rope_rows: usize,
}

/// One model's decode graphs.
pub(crate) struct Qwen3DecodeGraph {
    /// The model, in warnings.
    label: &'static str,
    pub(super) graphs: ShapeGraphs<GraphKey>,
    /// Widths whose first graphed step matched the standard decode.
    checked: Mutex<Vec<usize>>,
    /// Why the warm check turned the graphs off.
    off: Mutex<Option<String>>,
    graphed_steps: AtomicU64,
    spare_rows: AtomicU64,
    warm_checks: AtomicU64,
}

impl Qwen3DecodeGraph {
    /// `label`'s decode graphs on `device`, or why it has none: they run on
    /// CUDA only, with `switch` and `cuda.decode_graphs` on.
    pub(super) fn resolve(
        label: &'static str,
        switch: &'static str,
        device: &Device,
    ) -> std::result::Result<Self, String> {
        if !crate::env::flag_or(switch, true) {
            return Err(format!("{switch} is off"));
        }
        if decode_graphs_switched_off() {
            return Err("cuda.decode_graphs=off".into());
        }
        Ok(Self {
            label,
            graphs: ShapeGraphs::resolve(label, device, DECODE_GRAPH_LIMITS)?,
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
        tracing::warn!("{} CUDA decode graphs are off: {reason}", self.label);
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
        for switch in [QWEN3_ASR_DECODE_GRAPHS_ENV, VIBEVOICE_ASR_DECODE_GRAPHS_ENV] {
            assert_eq!(qwen3_decode_graph_device_bytes(BackendKind::Cpu, switch), 0);
            assert_eq!(
                qwen3_decode_graph_device_bytes(BackendKind::Metal, switch),
                0
            );
            let reason = Qwen3DecodeGraph::resolve("test", switch, &Device::Cpu)
                .err()
                .expect("no graphs on CPU");
            if decode_graphs_requested(switch) {
                assert!(reason.contains("CUDA only"), "{reason}");
                assert_eq!(
                    qwen3_decode_graph_device_bytes(BackendKind::Cuda, switch),
                    256 << 20
                );
            }
            let previous = std::env::var_os(switch);
            std::env::set_var(switch, "0");
            let bytes = qwen3_decode_graph_device_bytes(BackendKind::Cuda, switch);
            let reason = Qwen3DecodeGraph::resolve("test", switch, &Device::Cpu).err();
            std::env::remove_var(switch);
            if let Some(previous) = previous {
                std::env::set_var(switch, previous);
            }
            assert_eq!(bytes, 0);
            assert!(reason.expect("off").contains(switch));
        }
    }

    /// `cuda.decode_graphs=off` turns them off whatever the family switch.
    #[test]
    fn the_global_decode_graph_switch_turns_them_off() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("IZWI_CUDA_DECODE_GRAPHS");
        std::env::set_var("IZWI_CUDA_DECODE_GRAPHS", "off");
        let bytes = qwen3_decode_graph_device_bytes(BackendKind::Cuda, QWEN3_ASR_DECODE_GRAPHS_ENV);
        let reason =
            Qwen3DecodeGraph::resolve("test", QWEN3_ASR_DECODE_GRAPHS_ENV, &Device::Cpu).err();
        std::env::remove_var("IZWI_CUDA_DECODE_GRAPHS");
        if let Some(previous) = previous {
            std::env::set_var("IZWI_CUDA_DECODE_GRAPHS", previous);
        }
        assert_eq!(bytes, 0);
        assert_eq!(reason.as_deref(), Some("cuda.decode_graphs=off"));
    }
}
