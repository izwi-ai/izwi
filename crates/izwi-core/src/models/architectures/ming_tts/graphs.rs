//! CUDA graphs of Ming's flow path: the flow head's whole ten-step sampler
//! and the aggregator, each at the padded CUDA group
//! ([`super::FLOW_GROUP_CUDA`] requests), the shape every CUDA call has; and
//! of the VAE's fixed-shape window ([`MingVaeGraphs`]).
//!
//! A latent's flow head is about 5,000 small kernels (eight blocks, ten
//! Euler steps, both guidance branches); as one graph it is one launch for
//! every request in the group. The graphs are [`ShapeGraphs`]: captured on
//! first use with the eager run's outputs checked against a replay bit for
//! bit, verified once more over new inputs, then replayed. Any failure turns
//! them off for the process and the flow path runs eagerly, as it does on
//! the CPU and Metal. A load-time self-check runs both through capture and
//! verification on xorshift inputs, so a model that cannot graph says so in
//! its diagnostics before serving.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Tensor};

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};

/// Kill switch for the flow path's CUDA graphs (a boolean; on when unset).
/// Read once: load pricing and the runtime read it.
pub(crate) const MING_TTS_FLOW_GRAPHS_ENV: &str = "IZWI_MING_TTS_FLOW_GRAPHS";

fn flow_graphs_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(MING_TTS_FLOW_GRAPHS_ENV, true));
    ON.get()
}

/// Kill switch for the language model's decode-step graphs (a boolean; on
/// when unset). The global `IZWI_CUDA_DECODE_GRAPHS=off` turns them off too.
pub(crate) const MING_TTS_DECODE_GRAPHS_ENV: &str = "IZWI_MING_TTS_DECODE_GRAPHS";

const MIB: u64 = 1024 * 1024;

/// What the graphs may hold: the sampler and the aggregator at the loaded
/// dtype. A graph retains its inputs (the sampler's hidden states,
/// histories, noise and guidance for 32 requests, under 0.2 MiB) and its
/// output. The sampler's working set is 64 rows of three 1,024-wide
/// positions through eight blocks for ten steps, the widest
/// [192, 4096]; the pool budget leaves room for the graph allocator not
/// reusing every step's buffers. A pool that grows past it turns the graphs
/// off and the flow path runs eagerly.
const FLOW_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 4,
    retained_bytes: 8 * MIB,
    pool_bytes: 192 * MIB,
};

/// Kill switch for the VAE window's CUDA graph (a boolean; on when unset).
/// Off, CUDA windows still run at their fixed shape, eagerly, with the same
/// bits. Read once: load pricing and the runtime read it.
pub(crate) const MING_TTS_VAE_GRAPHS_ENV: &str = "IZWI_MING_TTS_VAE_GRAPHS";

fn vae_graphs_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(MING_TTS_VAE_GRAPHS_ENV, true));
    ON.get()
}

/// What the VAE window's graph may hold: one shape (34 frames over 63 kept
/// keys, F32). It retains its inputs (the frames and tables, under 32 KiB,
/// and every layer's kept keys and values, 1.5 MiB) and its outputs (the
/// head, 0.5 MiB, and the new kept keys, 1.5 MiB). Its working set is 24
/// layers of 34 rows at width 896, the widest the [34, 4864] MLP rows and
/// the [2, 7, 34, 97] scores, a few MiB a layer; the pool budget covers all
/// 24 layers' buffers should the graph allocator reuse none of them. A pool
/// that grows past it runs the window eagerly at the same shape.
const VAE_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 1,
    retained_bytes: 8 * MIB,
    pool_bytes: 128 * MIB,
};

/// Device bytes a load on `backend` prices for the flow path's and the VAE
/// window's graphs, which the load's self-checks allocate: zero off CUDA or
/// with the graphs off.
pub(crate) fn ming_tts_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend != BackendKind::Cuda {
        return 0;
    }
    let flow = if flow_graphs_requested() {
        FLOW_GRAPH_LIMITS.device_bytes()
    } else {
        0
    };
    let vae = if vae_graphs_requested() {
        VAE_GRAPH_LIMITS.device_bytes()
    } else {
        0
    };
    flow + vae
}

/// Which part of the flow path a graph runs, at which dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MingGraphKey {
    Sampler(DType),
    Aggregator(DType),
}

/// The flow path's graphs, or why there are none, and how many calls ran
/// through them.
pub(crate) struct MingFlowGraphs {
    graphs: std::result::Result<ShapeGraphs<MingGraphKey>, String>,
    graphed: AtomicU64,
    /// Calls the graphs could not take (not at the padded group).
    eager: AtomicU64,
}

impl MingFlowGraphs {
    pub(crate) fn off(reason: &str) -> Self {
        Self {
            graphs: Err(reason.into()),
            graphed: AtomicU64::new(0),
            eager: AtomicU64::new(0),
        }
    }

    /// The graphs on `device`, or why they are off (not CUDA, or the
    /// switch). The caller self-checks them before serving.
    pub(crate) fn resolve(device: &Device) -> Self {
        let graphs = if flow_graphs_requested() {
            ShapeGraphs::resolve("Ming TTS flow path", device, FLOW_GRAPH_LIMITS)
        } else {
            Err(format!("{MING_TTS_FLOW_GRAPHS_ENV} is off"))
        };
        Self {
            graphs,
            graphed: AtomicU64::new(0),
            eager: AtomicU64::new(0),
        }
    }

    /// Whether a call of `rows` rows runs through a graph.
    pub(crate) fn takes(&self, rows: usize) -> bool {
        let on = self.graphs.as_ref().is_ok_and(ShapeGraphs::is_on);
        if on && rows != super::FLOW_GROUP_CUDA {
            self.eager.fetch_add(1, Ordering::Relaxed);
        }
        on && rows == super::FLOW_GROUP_CUDA
    }

    /// `body` over `inputs` through `key`'s graph; `takes` said yes.
    pub(crate) fn run(
        &self,
        key: MingGraphKey,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let graphs = self.graphs.as_ref().map_err(|reason| {
            Error::InferenceError(format!("Ming flow graphs are off: {reason}"))
        })?;
        let bytes = |tensor: &Tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64;
        let inputs = inputs
            .iter()
            .map(Tensor::contiguous)
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut out = graphs.run(
            key,
            inputs.iter().map(bytes).sum::<u64>() * 2,
            &inputs,
            body,
        )?;
        self.graphed.fetch_add(1, Ordering::Relaxed);
        out.pop()
            .ok_or_else(|| Error::InferenceError("Ming flow graph returned nothing".into()))
    }

    /// Turn the graphs off (the load's self-check failed), keeping why.
    pub(crate) fn turn_off(&mut self, reason: String) {
        self.graphs = Err(reason);
    }

    /// Whether the graphs are still on.
    pub(crate) fn is_on(&self) -> bool {
        self.graphs.as_ref().is_ok_and(ShapeGraphs::is_on)
    }

    /// The graphs' state and counters, for the model diagnostics.
    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let mut diagnostics = match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        };
        diagnostics["requests"] = super::FLOW_GROUP_CUDA.into();
        diagnostics["graphed_calls"] = self.graphed.load(Ordering::Relaxed).into();
        diagnostics["eager_calls"] = self.eager.load(Ordering::Relaxed).into();
        diagnostics
    }
}

/// The VAE window's graph's key: its query slots and kept keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MingVaeGraphKey {
    pub(crate) queries: usize,
    pub(crate) kept: usize,
}

/// The VAE window's graph (CUDA), or why there is none: a window then runs
/// its fixed-shape body eagerly, with the same bits.
pub(crate) struct MingVaeGraphs {
    graphs: std::result::Result<ShapeGraphs<MingVaeGraphKey>, String>,
}

impl MingVaeGraphs {
    pub(crate) fn off(reason: &str) -> Self {
        Self {
            graphs: Err(reason.into()),
        }
    }

    /// The graph on `device`, or why it is off (not CUDA, or the switch).
    /// The caller self-checks it before serving.
    pub(crate) fn resolve(device: &Device) -> Self {
        let graphs = if vae_graphs_requested() {
            ShapeGraphs::resolve("Ming TTS VAE window", device, VAE_GRAPH_LIMITS)
        } else {
            Err(format!("{MING_TTS_VAE_GRAPHS_ENV} is off"))
        };
        Self { graphs }
    }

    /// `body` over `inputs` through `key`'s graph, or eagerly when the
    /// graph is off; `retained` is what the graph keeps between runs.
    pub(crate) fn run(
        &self,
        key: MingVaeGraphKey,
        retained: u64,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        match &self.graphs {
            Ok(graphs) => graphs.run(key, retained, inputs, body),
            Err(_) => body(inputs),
        }
    }

    /// Turn the graph off (the load's self-check failed), keeping why.
    pub(crate) fn turn_off(&mut self, reason: String) {
        self.graphs = Err(reason);
    }

    /// Whether the graph is still on.
    pub(crate) fn is_on(&self) -> bool {
        self.graphs.as_ref().is_ok_and(ShapeGraphs::is_on)
    }

    /// The graph's state and counters, for the model diagnostics.
    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphs_are_off_and_unpriced_away_from_cuda() {
        let graphs = MingFlowGraphs::resolve(&Device::Cpu);
        assert!(!graphs.is_on());
        assert!(!graphs.takes(super::super::FLOW_GROUP_CUDA));
        assert!(graphs.diagnostics()["off"].is_string());
        let vae = MingVaeGraphs::resolve(&Device::Cpu);
        assert!(!vae.is_on());
        assert!(vae.diagnostics()["off"].is_string());
        assert_eq!(ming_tts_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(ming_tts_graph_device_bytes(BackendKind::Metal), 0);
    }

    #[test]
    fn cuda_prices_both_graphs() {
        assert_eq!(
            ming_tts_graph_device_bytes(BackendKind::Cuda),
            FLOW_GRAPH_LIMITS.device_bytes() + VAE_GRAPH_LIMITS.device_bytes()
        );
    }
}
