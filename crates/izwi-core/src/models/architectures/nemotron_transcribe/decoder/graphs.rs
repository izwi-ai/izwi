//! CUDA graphs of the fused decode step's attention-free stretches.
//!
//! The fused step is a run of pieces ([`super::Piece`]): stretches of Mamba
//! and MoE layers, each a pure function of the residual, the normed hidden
//! and its layers' Mamba states, and the attention layers between them,
//! which write and read the paged KV under per-step metadata. On CUDA each
//! stretch is captured once per row count with [`SegmentGraph`] and replayed
//! after; attention stays eager. Per stretch and row count:
//!
//! 1. **warm**: the eager stretch under Candle's upload-cache guard;
//! 2. **capture**: the graph is captured, the eager stretch gives the step's
//!    outputs, and a replay over the same inputs must match them bit for bit;
//! 3. **verify**: the next step checks a replay against the eager stretch
//!    again, over new inputs, so a value baked into the graph shows;
//! 4. **replay**: the graph alone, its outputs copied out of graph memory.
//!
//! Any graph failure turns every graph off for the process and the stretch
//! runs eagerly; an error of the eager stretch itself is the step's.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};

use candle_core::{DType, Device, Tensor};
use serde_json::json;

use crate::error::{Error, Result};
use crate::kernels::cuda::segment_graph::{SegmentGraph, SegmentInput};

/// Kill switch for the stretch graphs (`0`, `off`, `false`): every stretch
/// runs eagerly. The global `IZWI_CUDA_DECODE_GRAPHS=off` turns them off
/// too. Read at load.
pub const SEGMENT_GRAPHS_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_SEGMENT_GRAPHS";

/// The global switch for every family's decode graphs (`cuda.decode_graphs`).
const DECODE_GRAPHS_ENV: &str = "IZWI_CUDA_DECODE_GRAPHS";

/// Widest step whose stretches are captured. Each row count keeps its own
/// graphs, with stable copies of its rows' Mamba states.
pub(super) const MAX_GRAPH_ROWS: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Warm,
    Capture,
    Verify,
    Replay,
}

#[derive(Default)]
struct Stretch {
    graph: SegmentGraph,
    phase: Phase,
}

/// Each row count's stretches, or why the graphs are off.
type Runtime = std::result::Result<HashMap<usize, Vec<Stretch>>, String>;

/// The stretch graphs of one decoder.
pub(super) struct StepGraphs {
    device: Device,
    stretches: usize,
    runtime: Mutex<Runtime>,
    warmups: AtomicU64,
    captures: AtomicU64,
    verified: AtomicU64,
    replays: AtomicU64,
}

impl StepGraphs {
    /// The graphs of a decoder with `stretches` stretches on `device`, or
    /// why it has none.
    pub(super) fn resolve(device: &Device, stretches: usize) -> std::result::Result<Self, String> {
        if !device.is_cuda() {
            return Err("stretch graphs run on CUDA only".into());
        }
        if !crate::env::flag_or(SEGMENT_GRAPHS_ENV, true) {
            return Err(format!("{SEGMENT_GRAPHS_ENV} is off"));
        }
        if crate::env::value(DECODE_GRAPHS_ENV).is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "off" | "0" | "false"
            )
        }) {
            return Err("cuda.decode_graphs=off".into());
        }
        Ok(Self {
            device: device.clone(),
            stretches,
            runtime: Mutex::new(Ok(HashMap::new())),
            warmups: AtomicU64::new(0),
            captures: AtomicU64::new(0),
            verified: AtomicU64::new(0),
            replays: AtomicU64::new(0),
        })
    }

    /// A step of `rows` rows through the graphs, or `None` when they are off,
    /// held by another step, or not made for that width.
    pub(super) fn begin(&self, rows: usize) -> Option<StepRun<'_>> {
        if rows == 0 || rows > MAX_GRAPH_ROWS {
            return None;
        }
        let guard = self.runtime.try_lock().ok()?;
        guard.is_ok().then_some(StepRun {
            graphs: self,
            guard,
            rows,
        })
    }

    pub(super) fn diagnostics(&self) -> serde_json::Value {
        let runtime = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        json!({
            "off": runtime.as_ref().err(),
            "widths": runtime.as_ref().map(|map| {
                let mut widths = map.keys().copied().collect::<Vec<_>>();
                widths.sort_unstable();
                widths
            }).unwrap_or_default(),
            "warmups": self.warmups.load(Relaxed),
            "captures": self.captures.load(Relaxed),
            "verified": self.verified.load(Relaxed),
            "replays": self.replays.load(Relaxed),
        })
    }

    /// Stretch `stretch` through its phase. `Err(Ok(error))` is the eager
    /// stretch's own error; `Err(Err(reason))` a graph failure.
    fn step(
        &self,
        stretch: &mut Stretch,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> std::result::Result<Vec<Tensor>, std::result::Result<Error, String>> {
        let stable = inputs
            .iter()
            .map(|tensor| SegmentInput {
                tensor,
                adopt: false,
            })
            .collect::<Vec<_>>();
        match stretch.phase {
            Phase::Warm => {
                let out = SegmentGraph::warm(&self.device, inputs, |inputs| {
                    body(inputs).map_err(candle_core::Error::wrap)
                })
                .map_err(|error| Ok(Error::from(error)))?;
                self.warmups.fetch_add(1, Relaxed);
                stretch.phase = Phase::Capture;
                Ok(out)
            }
            Phase::Capture | Phase::Verify => {
                if stretch.phase == Phase::Capture {
                    stretch
                        .graph
                        .capture(&stable, |inputs| {
                            body(inputs).map_err(candle_core::Error::wrap)
                        })
                        .map_err(failed("capture"))?;
                    self.captures.fetch_add(1, Relaxed);
                }
                let eager = body(inputs).map_err(Ok)?;
                let replayed = stretch.graph.replay(&stable).map_err(failed("replay"))?;
                bitwise(&eager, &replayed).map_err(Err)?;
                self.verified.fetch_add(1, Relaxed);
                stretch.phase = match stretch.phase {
                    Phase::Capture => Phase::Verify,
                    _ => Phase::Replay,
                };
                Ok(eager)
            }
            Phase::Replay => {
                let replayed = stretch.graph.replay(&stable).map_err(failed("replay"))?;
                self.replays.fetch_add(1, Relaxed);
                // Graph memory: the next replay overwrites it.
                replayed
                    .iter()
                    .map(Tensor::copy)
                    .collect::<candle_core::Result<Vec<_>>>()
                    .map_err(failed("output copy"))
            }
        }
    }
}

/// One step's hold on the graphs.
pub(super) struct StepRun<'a> {
    graphs: &'a StepGraphs,
    guard: MutexGuard<'a, Runtime>,
    rows: usize,
}

impl StepRun<'_> {
    /// Stretch `index` over `inputs`, with `body` the eager stretch. The
    /// outputs belong to the caller. A graph failure turns every graph off
    /// and the stretch runs eagerly.
    pub(super) fn run(
        &mut self,
        index: usize,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        let Ok(widths) = self.guard.as_mut() else {
            return body(inputs);
        };
        let stretches = widths.entry(self.rows).or_insert_with(|| {
            (0..self.graphs.stretches)
                .map(|_| Stretch::default())
                .collect()
        });
        let Some(stretch) = stretches.get_mut(index) else {
            return Err(Error::InferenceError(format!(
                "Nemotron 3.5 Transcribe has no decode stretch {index}"
            )));
        };
        match self.graphs.step(stretch, inputs, body) {
            Ok(outputs) => Ok(outputs),
            Err(Ok(error)) => Err(error),
            Err(Err(reason)) => {
                tracing::warn!(
                    "Nemotron 3.5 Transcribe decode graphs are off: stretch {index} at width {}: \
                     {reason}",
                    self.rows
                );
                // Dropping the graphs fences the stream before their memory goes.
                *self.guard = Err(reason);
                body(inputs)
            }
        }
    }
}

/// A graph failure from a CUDA error of `what`.
fn failed(
    what: &'static str,
) -> impl FnOnce(candle_core::Error) -> std::result::Result<Error, String> {
    move |error| Err(format!("{what}: {error}"))
}

/// Whether `a` and `b` hold the same tensors bit for bit.
fn bitwise(a: &[Tensor], b: &[Tensor]) -> std::result::Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} outputs against {}", b.len(), a.len()));
    }
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        let same = a.dims() == b.dims()
            && a.dtype() == b.dtype()
            && bits(a).map_err(|e| e.to_string())? == bits(b).map_err(|e| e.to_string())?;
        if !same {
            return Err(format!("output {index} differs from the eager stretch"));
        }
    }
    Ok(())
}

fn bits(t: &Tensor) -> candle_core::Result<Vec<u32>> {
    let t = t.flatten_all()?.to_device(&Device::Cpu)?;
    match t.dtype() {
        DType::U32 => t.to_vec1::<u32>(),
        _ => Ok(t
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect()),
    }
}
