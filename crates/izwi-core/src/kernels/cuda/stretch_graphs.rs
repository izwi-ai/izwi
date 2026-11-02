//! CUDA graphs of a decode step's stretches: the pure runs of layers between
//! the pieces that must stay eager (attention, which writes and reads the
//! paged KV under per-step metadata). A model passes each stretch's inputs
//! (its residual, its normed hidden, its layers' recurrent states) and the
//! eager stretch itself; each stretch is captured once per row count with
//! [`SegmentGraph`] and replayed after. Per stretch and row count:
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

use super::segment_graph::{SegmentGraph, SegmentInput};

/// The global switch for every family's decode graphs (`cuda.decode_graphs`).
const DECODE_GRAPHS_ENV: &str = "IZWI_CUDA_DECODE_GRAPHS";

/// Widest step whose stretches are captured. Each row count keeps its own
/// graphs, with stable copies of its rows' recurrent states.
pub const MAX_GRAPH_ROWS: usize = 4;

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

/// Each (row count, key) pair's stretches, or why the graphs are off.
type Runtime = std::result::Result<HashMap<(usize, u64), Vec<Stretch>>, String>;

/// Most (row count, key) pairs one model keeps graphs for: a new pair past
/// it drops another's.
const MAX_SHAPES: usize = 8;

/// The stretch graphs of one model's decode step.
pub struct StretchGraphs {
    /// The model, in warnings.
    label: &'static str,
    device: Device,
    stretches: usize,
    runtime: Mutex<Runtime>,
    warmups: AtomicU64,
    captures: AtomicU64,
    verified: AtomicU64,
    replays: AtomicU64,
}

impl StretchGraphs {
    /// The graphs of `label`'s step of `stretches` stretches on `device`, or
    /// why it has none: `switch` is the model's own flag (default on).
    pub fn resolve(
        label: &'static str,
        switch: &str,
        device: &Device,
        stretches: usize,
    ) -> std::result::Result<Self, String> {
        if !device.is_cuda() {
            return Err("stretch graphs run on CUDA only".into());
        }
        if !crate::env::flag_or(switch, true) {
            return Err(format!("{switch} is off"));
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
            label,
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
    pub fn begin(&self, rows: usize) -> Option<StepRun<'_>> {
        self.begin_keyed(rows, 0)
    }

    /// [`Self::begin`] for stretches whose kernels also take host values
    /// (a sampling policy, say) as launch arguments: `key` fingerprints
    /// them, and each key keeps its own graphs.
    pub fn begin_keyed(&self, rows: usize, key: u64) -> Option<StepRun<'_>> {
        if rows == 0 || rows > MAX_GRAPH_ROWS {
            return None;
        }
        let guard = self.runtime.try_lock().ok()?;
        guard.is_ok().then_some(StepRun {
            graphs: self,
            guard,
            shape: (rows, key),
        })
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        let runtime = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        json!({
            "off": runtime.as_ref().err(),
            "widths": runtime.as_ref().map(|map| {
                let mut widths = map.keys().map(|(rows, _)| *rows).collect::<Vec<_>>();
                widths.sort_unstable();
                widths.dedup();
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
pub struct StepRun<'a> {
    graphs: &'a StretchGraphs,
    guard: MutexGuard<'a, Runtime>,
    shape: (usize, u64),
}

impl StepRun<'_> {
    /// Stretch `index` over `inputs`, with `body` the eager stretch. The
    /// outputs belong to the caller. A graph failure turns every graph off
    /// and the stretch runs eagerly.
    pub fn run(
        &mut self,
        index: usize,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        let Ok(widths) = self.guard.as_mut() else {
            return body(inputs);
        };
        if !widths.contains_key(&self.shape) && widths.len() >= MAX_SHAPES {
            // Dropping a shape's graphs fences the stream before their memory goes.
            if let Some(oldest) = widths.keys().next().copied() {
                widths.remove(&oldest);
            }
        }
        let stretches = widths.entry(self.shape).or_insert_with(|| {
            (0..self.graphs.stretches)
                .map(|_| Stretch::default())
                .collect()
        });
        let Some(stretch) = stretches.get_mut(index) else {
            return Err(Error::InferenceError(format!(
                "{} has no decode stretch {index}",
                self.graphs.label
            )));
        };
        match self.graphs.step(stretch, inputs, body) {
            Ok(outputs) => Ok(outputs),
            Err(Ok(error)) => Err(error),
            Err(Err(reason)) => {
                tracing::warn!(
                    "{} decode graphs are off: stretch {index} at width {}: {reason}",
                    self.graphs.label,
                    self.shape.0
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
