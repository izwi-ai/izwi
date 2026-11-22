//! CUDA graphs of one pure segment (an encoder's layer stack, say), one
//! graph per input shape. A [`super::stretch_graphs`] step replays a few
//! widths every step; a segment run here comes in many shapes, each graph
//! holding real memory, so the cache keeps at most a few graphs within a
//! retained-memory budget and drops the least recently used. Per shape:
//!
//! 1. **capture**: the eager segment runs under Candle's upload-cache guard,
//!    which warms the parameter caches the capture must hit on this same
//!    thread; the graph is captured, and a replay over the same inputs must
//!    match the eager outputs bit for bit;
//! 2. **verify**: the next run of that shape checks a replay against the
//!    eager segment again, over new inputs, so a value baked into the graph
//!    shows;
//! 3. **replay**: the graph alone, its outputs copied out of graph memory.
//!
//! The first two phases return the eager outputs. The segment's
//! intermediates are graph memory, from the device's graph memory pool,
//! which keeps its physical pages after a launch. Graphs launched on one
//! stream share those pages, so the pool holds about the largest graph's
//! working set plus every kept graph's outputs. The cache charges the pool's
//! growth over each capture (and its first launch) to its budget; replays
//! reuse mapped pages, so growth seen around one is another cache's trim
//! being undone, never this cache's. A capture that would take the pool past
//! the budget is dropped and its shape runs eagerly from then on; the other
//! graphs keep replaying. A caller drops graphs that can never run again
//! (their buffers were replaced) with [`ShapeGraphs::retain`], so their pages
//! serve the next capture. Any graph failure turns the cache off for the
//! process: its graphs are dropped behind a stream fence, the pool's unused
//! pages are trimmed, and the segment runs eagerly. An error of the eager
//! segment is the run's.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::{Device, Tensor};
use serde_json::json;

use crate::error::{Error, Result};

use super::segment_graph::{SegmentGraph, SegmentInput};
use super::stretch_graphs::bitwise;

/// What one cache may hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeGraphLimits {
    /// Most graphs kept at once.
    pub max_graphs: usize,
    /// Most bytes the kept graphs retain between runs: their stable input
    /// copies and their graph-memory outputs.
    pub retained_bytes: u64,
    /// Most the device's graph memory pool may grow across this cache's
    /// captures: the largest working set beside every kept graph's outputs.
    pub pool_bytes: u64,
}

impl ShapeGraphLimits {
    /// Device memory a model prices for this cache: what its graphs retain
    /// and what the graph pool may hold for them.
    pub const fn device_bytes(self) -> u64 {
        self.retained_bytes + self.pool_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Verify,
    Replay,
}

struct Slot<K, G> {
    key: K,
    retained: u64,
    graph: G,
    phase: Phase,
}

/// Kept graphs, least recently used first.
struct Slots<K, G> {
    slots: Vec<Slot<K, G>>,
}

impl<K: PartialEq, G> Slots<K, G> {
    const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// `key`'s slot, now the most recently used.
    fn touch(&mut self, key: &K) -> Option<&mut Slot<K, G>> {
        let index = self.slots.iter().position(|slot| slot.key == *key)?;
        let slot = self.slots.remove(index);
        self.slots.push(slot);
        self.slots.last_mut()
    }

    fn retained(&self) -> u64 {
        self.slots.iter().map(|slot| slot.retained).sum()
    }

    /// Drops the least recently used slots until a new slot retaining
    /// `bytes` fits `limits`, and returns how many went; `None`, dropping
    /// nothing, when it would not fit alone.
    fn make_room(&mut self, bytes: u64, limits: ShapeGraphLimits) -> Option<usize> {
        if limits.max_graphs == 0 || bytes > limits.retained_bytes {
            return None;
        }
        let mut dropped = 0;
        while self.slots.len() >= limits.max_graphs
            || self.retained().saturating_add(bytes) > limits.retained_bytes
        {
            // Dropping a graph fences the stream before its memory goes.
            self.slots.remove(0);
            dropped += 1;
        }
        Some(dropped)
    }

    fn keys(&self) -> impl Iterator<Item = &K> {
        self.slots.iter().map(|slot| &slot.key)
    }

    /// Drops the slots whose key `keep` refuses, and returns how many went.
    fn retain(&mut self, keep: impl Fn(&K) -> bool) -> usize {
        let before = self.slots.len();
        // Dropping a graph fences the stream before its memory goes.
        self.slots.retain(|slot| keep(&slot.key));
        before - self.slots.len()
    }
}

/// Most shapes a cache remembers as eager (their capture would have taken
/// the pool past its budget); the oldest is forgotten first.
const MAX_EAGER_SHAPES: usize = 64;

struct State<K> {
    slots: Slots<K, SegmentGraph>,
    /// Graph-pool growth over this cache's captures.
    pool_growth: u64,
    /// Shapes that run eagerly: capturing them would have grown the pool
    /// past the budget.
    eager: Vec<K>,
}

/// One graph failure, or the eager segment's own error.
enum Failure {
    Eager(Error),
    Graph(String),
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self::Eager(error)
    }
}

/// A graph failure from a CUDA error of `what`.
fn failed(what: &'static str) -> impl FnOnce(candle_core::Error) -> Failure {
    move |error| Failure::Graph(format!("{what}: {error}"))
}

/// The graphs of one model's segment, keyed by shape.
pub struct ShapeGraphs<K> {
    /// The model and segment, in warnings.
    label: &'static str,
    device: Device,
    limits: ShapeGraphLimits,
    state: Mutex<std::result::Result<State<K>, String>>,
    captures: AtomicU64,
    verified: AtomicU64,
    replays: AtomicU64,
    evictions: AtomicU64,
    fallbacks: AtomicU64,
    over_budget: AtomicU64,
}

impl<K: PartialEq + Clone + std::fmt::Debug> ShapeGraphs<K> {
    /// The graphs of `label` on `device`, or why it has none (they run on
    /// CUDA only).
    pub fn resolve(
        label: &'static str,
        device: &Device,
        limits: ShapeGraphLimits,
    ) -> std::result::Result<Self, String> {
        if !device.is_cuda() {
            return Err("shape graphs run on CUDA only".into());
        }
        Ok(Self {
            label,
            device: device.clone(),
            limits,
            state: Mutex::new(Ok(State {
                slots: Slots::new(),
                pool_growth: 0,
                eager: Vec::new(),
            })),
            captures: AtomicU64::new(0),
            verified: AtomicU64::new(0),
            replays: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
            over_budget: AtomicU64::new(0),
        })
    }

    /// Drops every kept graph, and forgets every eager shape, whose key
    /// `keep` refuses: graphs whose captured buffers were replaced (a KV
    /// arena that grew, a rebuilt table) can never replay, and their pages
    /// then serve the next capture. Returns how many graphs went; a run
    /// holding the cache, or a cache turned off, drops nothing.
    pub fn retain(&self, keep: impl Fn(&K) -> bool) -> usize {
        let Ok(mut guard) = self.state.try_lock() else {
            return 0;
        };
        let Ok(state) = guard.as_mut() else {
            return 0;
        };
        state.eager.retain(|key| keep(key));
        let dropped = state.slots.retain(keep);
        self.evictions.fetch_add(dropped as u64, Relaxed);
        dropped
    }

    /// Whether runs still go through graphs (no failure turned them off).
    pub fn is_on(&self) -> bool {
        self.state.lock().is_ok_and(|state| state.is_ok())
    }

    /// `body` over `inputs` through `key`'s graph (see the module docs).
    /// `retained` is what that graph keeps between runs: its stable copies
    /// of `inputs` and its outputs. A shape that does not fit
    /// [`ShapeGraphLimits::retained_bytes`] alone, or a run while another
    /// holds the cache, runs eagerly. The outputs belong to the caller.
    pub fn run(
        &self,
        key: K,
        retained: u64,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        let Ok(mut guard) = self.state.try_lock() else {
            self.fallbacks.fetch_add(1, Relaxed);
            return body(inputs);
        };
        let Ok(state) = guard.as_mut() else {
            self.fallbacks.fetch_add(1, Relaxed);
            return body(inputs);
        };
        match self.step(state, &key, retained, inputs, body) {
            Ok(Some(outputs)) => Ok(outputs),
            Ok(None) => {
                self.fallbacks.fetch_add(1, Relaxed);
                body(inputs)
            }
            Err(Failure::Eager(error)) => Err(error),
            Err(Failure::Graph(reason)) => {
                tracing::warn!(
                    "{} CUDA graphs are off: shape {key:?}: {reason}",
                    self.label
                );
                // Dropping the graphs fences the stream before their memory
                // goes; the pool then returns the pages nothing uses.
                *guard = Err(reason);
                drop(guard);
                pool::trim(&self.device);
                self.fallbacks.fetch_add(1, Relaxed);
                body(inputs)
            }
        }
    }

    /// One run of `key` through its phase; `None` runs eagerly.
    fn step(
        &self,
        state: &mut State<K>,
        key: &K,
        retained: u64,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> std::result::Result<Option<Vec<Tensor>>, Failure> {
        let stable = inputs
            .iter()
            .map(|tensor| SegmentInput {
                tensor,
                adopt: false,
            })
            .collect::<Vec<_>>();
        let State {
            slots,
            pool_growth,
            eager: eager_shapes,
        } = state;
        let launch = |graph: &SegmentGraph| graph.replay(&stable).map_err(failed("replay"));
        if let Some(slot) = slots.touch(key) {
            return match slot.phase {
                Phase::Verify => {
                    let eager = body(inputs)?;
                    let replayed = launch(&slot.graph)?;
                    bitwise(&eager, &replayed).map_err(Failure::Graph)?;
                    self.verified.fetch_add(1, Relaxed);
                    slot.phase = Phase::Replay;
                    Ok(Some(eager))
                }
                Phase::Replay => {
                    let replayed = launch(&slot.graph)?;
                    self.replays.fetch_add(1, Relaxed);
                    // Graph memory: the next replay overwrites it.
                    replayed
                        .iter()
                        .map(Tensor::copy)
                        .collect::<candle_core::Result<Vec<_>>>()
                        .map(Some)
                        .map_err(failed("output copy"))
                }
            };
        }
        if eager_shapes.contains(key) {
            return Ok(None);
        }
        let Some(dropped) = slots.make_room(retained, self.limits) else {
            return Ok(None);
        };
        self.evictions.fetch_add(dropped as u64, Relaxed);
        let eager = SegmentGraph::warm(&self.device, inputs, |inputs| {
            body(inputs).map_err(candle_core::Error::wrap)
        })
        .map_err(Error::from)?;
        let mut graph = SegmentGraph::default();
        let before = pool::reserved(&self.device);
        graph
            .capture(&stable, |inputs| {
                body(inputs).map_err(candle_core::Error::wrap)
            })
            .map_err(failed("capture"))?;
        self.captures.fetch_add(1, Relaxed);
        let replayed = launch(&graph)?;
        bitwise(&eager, &replayed).map_err(Failure::Graph)?;
        let growth = growth_since(before, pool::reserved(&self.device));
        if !within_budget(*pool_growth, growth, self.limits) {
            // The capture's pages stay in the pool for the next capture;
            // this shape runs eagerly instead of growing it further.
            self.over_budget.fetch_add(1, Relaxed);
            tracing::info!(
                "{} runs shape {key:?} eagerly: its CUDA graph grew the graph memory pool {growth} bytes, past the {} bytes priced ({} already charged)",
                self.label,
                self.limits.device_bytes(),
                pool_growth
            );
            if eager_shapes.len() >= MAX_EAGER_SHAPES {
                eager_shapes.remove(0);
            }
            eager_shapes.push(key.clone());
            // The graph drops here, behind its stream fence.
            return Ok(Some(eager));
        }
        *pool_growth += growth;
        slots.slots.push(Slot {
            key: key.clone(),
            retained,
            graph,
            phase: Phase::Verify,
        });
        Ok(Some(eager))
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (shapes, retained, pool_growth, eager) = match state.as_ref() {
            Ok(state) => (
                state
                    .slots
                    .keys()
                    .map(|key| format!("{key:?}"))
                    .collect::<Vec<_>>(),
                state.slots.retained(),
                state.pool_growth,
                state.eager.len(),
            ),
            Err(_) => (Vec::new(), 0, 0, 0),
        };
        json!({
            "off": state.as_ref().err(),
            "shapes": shapes,
            "retained_bytes": retained,
            "pool_growth_bytes": pool_growth,
            "limits": {
                "max_graphs": self.limits.max_graphs,
                "retained_bytes": self.limits.retained_bytes,
                "pool_bytes": self.limits.pool_bytes,
            },
            "captures": self.captures.load(Relaxed),
            "verified": self.verified.load(Relaxed),
            "replays": self.replays.load(Relaxed),
            "evictions": self.evictions.load(Relaxed),
            "fallbacks": self.fallbacks.load(Relaxed),
            "over_budget_captures": self.over_budget.load(Relaxed),
            "eager_shapes": eager,
            "device_graph_pool": pool::diagnostics(&self.device),
        })
    }
}

/// Bytes the pool grew from `before` to `after`; nothing when either is
/// unknown (no driver attribute).
fn growth_since(before: Option<u64>, after: Option<u64>) -> u64 {
    match (before, after) {
        (Some(before), Some(after)) => after.saturating_sub(before),
        _ => 0,
    }
}

/// Whether a capture that grew the pool `growth` bytes keeps the cache's
/// charged growth (`charged`) within its budget. A capture that reused
/// mapped pages (no growth) always fits.
fn within_budget(charged: u64, growth: u64, limits: ShapeGraphLimits) -> bool {
    growth == 0 || charged.saturating_add(growth) <= limits.device_bytes()
}

/// The device's CUDA graph memory pool.
mod pool {
    use candle_core::Device;

    #[cfg(feature = "cuda")]
    fn attribute(
        device: &Device,
        attribute: candle_core::cuda_backend::cudarc::driver::sys::CUgraphMem_attribute,
    ) -> Option<u64> {
        use candle_core::cuda_backend::cudarc::driver::{result, sys};
        let Device::Cuda(cuda) = device else {
            return None;
        };
        let ctx = cuda.cuda_stream().context().clone();
        let mut value = 0u64;
        // SAFETY: the context is live; the driver writes one u64 (a
        // cuuint64_t) for every graph memory attribute.
        let status = unsafe {
            result::ctx::set_current(ctx.cu_ctx()).ok()?;
            sys::cuDeviceGetGraphMemAttribute(
                ctx.cu_device(),
                attribute,
                (&mut value as *mut u64).cast(),
            )
        };
        (status == sys::CUresult::CUDA_SUCCESS).then_some(value)
    }

    /// Bytes the pool holds mapped now.
    pub(super) fn reserved(device: &Device) -> Option<u64> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::sys::CUgraphMem_attribute;
            attribute(
                device,
                CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            None
        }
    }

    /// Returns the pool's unused pages to the device.
    pub(super) fn trim(device: &Device) {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            use candle_core::cuda_backend::cudarc::driver::{result, sys};
            let ctx = cuda.cuda_stream().context().clone();
            // SAFETY: the context is live; trimming frees only pages no
            // graph allocation uses.
            unsafe {
                if result::ctx::set_current(ctx.cu_ctx()).is_ok() {
                    let _ = sys::cuDeviceGraphMemTrim(ctx.cu_device());
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
    }

    /// The pool's current and peak use, device-wide (every model's graphs).
    pub(super) fn diagnostics(device: &Device) -> serde_json::Value {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::sys::CUgraphMem_attribute::*;
            serde_json::json!({
                "used_bytes": attribute(device, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT),
                "used_high_bytes": attribute(device, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH),
                "reserved_bytes": attribute(device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT),
                "reserved_high_bytes": attribute(device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            serde_json::Value::Null
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: ShapeGraphLimits = ShapeGraphLimits {
        max_graphs: 3,
        retained_bytes: 100,
        pool_bytes: 1000,
    };

    fn slots(entries: &[(u32, u64)]) -> Slots<u32, ()> {
        let mut slots = Slots::new();
        for &(key, retained) in entries {
            slots.slots.push(Slot {
                key,
                retained,
                graph: (),
                phase: Phase::Verify,
            });
        }
        slots
    }

    fn keys(slots: &Slots<u32, ()>) -> Vec<u32> {
        slots.keys().copied().collect()
    }

    #[test]
    fn a_touched_shape_becomes_the_most_recently_used() {
        let mut kept = slots(&[(1, 10), (2, 10), (3, 10)]);
        assert!(kept.touch(&1).is_some());
        assert_eq!(keys(&kept), [2, 3, 1]);
        assert!(kept.touch(&9).is_none());
        assert_eq!(keys(&kept), [2, 3, 1]);
    }

    #[test]
    fn room_is_made_by_dropping_the_least_recently_used() {
        // The count cap: one goes.
        let mut kept = slots(&[(1, 10), (2, 10), (3, 10)]);
        kept.touch(&1);
        assert_eq!(kept.make_room(10, LIMITS), Some(1));
        assert_eq!(keys(&kept), [3, 1]);

        // The byte budget: as many as it takes.
        let mut kept = slots(&[(1, 40), (2, 30), (3, 20)]);
        assert_eq!(kept.make_room(60, LIMITS), Some(2));
        assert_eq!(keys(&kept), [3]);

        // Room already there.
        let mut kept = slots(&[(1, 10)]);
        assert_eq!(kept.make_room(90, LIMITS), Some(0));
        assert_eq!(keys(&kept), [1]);
    }

    #[test]
    fn a_shape_past_the_budget_alone_drops_nothing() {
        let mut kept = slots(&[(1, 10), (2, 10)]);
        assert_eq!(kept.make_room(101, LIMITS), None);
        assert_eq!(keys(&kept), [1, 2]);
        let none = ShapeGraphLimits {
            max_graphs: 0,
            ..LIMITS
        };
        assert_eq!(kept.make_room(1, none), None);
        assert_eq!(keys(&kept), [1, 2]);
    }

    #[test]
    fn the_priced_bytes_cover_the_retained_and_pool_budgets() {
        assert_eq!(LIMITS.device_bytes(), 1100);
    }

    #[test]
    fn retain_drops_only_the_refused_keys_in_order() {
        let mut kept = slots(&[(1, 10), (2, 10), (3, 10), (4, 10)]);
        assert_eq!(kept.retain(|key| key % 2 == 0), 2);
        assert_eq!(keys(&kept), [2, 4]);
        assert_eq!(kept.retained(), 20);
        assert_eq!(kept.retain(|_| true), 0);
    }

    /// Captures are charged by the pool's growth; one that reused mapped
    /// pages always fits, and one past the budget does not, however little
    /// it grew (a cache never grows the pool past its price).
    #[test]
    fn captures_are_charged_by_growth_and_refused_past_the_budget() {
        assert_eq!(growth_since(Some(100), Some(160)), 60);
        // A trim between the readings is no growth of this capture's.
        assert_eq!(growth_since(Some(160), Some(100)), 0);
        assert_eq!(growth_since(None, Some(100)), 0);
        assert_eq!(growth_since(Some(100), None), 0);

        assert!(within_budget(0, 1100, LIMITS));
        assert!(!within_budget(0, 1101, LIMITS));
        assert!(within_budget(1000, 100, LIMITS));
        assert!(!within_budget(1000, 101, LIMITS));
        assert!(within_budget(5000, 0, LIMITS), "a reused-page capture fits");
    }

    #[test]
    fn graphs_resolve_on_cuda_only() {
        let error = ShapeGraphs::<u32>::resolve("test", &Device::Cpu, LIMITS)
            .err()
            .expect("CPU has no graphs");
        assert!(error.contains("CUDA only"), "{error}");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;

    fn host(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// Capture, verify and replay each return the eager segment's outputs
    /// bit for bit, over fresh inputs; a third shape past the count cap
    /// drops the least recently used.
    #[test]
    fn cuda_shape_graphs_capture_verify_and_replay() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let limits = ShapeGraphLimits {
            max_graphs: 2,
            retained_bytes: 1 << 20,
            pool_bytes: 64 << 20,
        };
        let graphs = ShapeGraphs::resolve("test", &device, limits).unwrap();
        let w = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.25], (2, 2), &device).unwrap();
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            let y = (inputs[0].matmul(&w.t()?.contiguous()?)? + 1.0)?;
            Ok(vec![candle_nn::ops::softmax_last_dim(&y)?])
        };
        for step in 0..4u32 {
            for rows in [1usize, 2] {
                let values = (0..rows * 2)
                    .map(|i| (i as f32 + step as f32) * 0.37 - 1.0)
                    .collect::<Vec<_>>();
                let x = Tensor::from_vec(values, (rows, 2), &device).unwrap();
                let expected = host(&body(std::slice::from_ref(&x)).unwrap()[0]);
                let out = graphs
                    .run(rows, 64, std::slice::from_ref(&x), &body)
                    .unwrap();
                assert_eq!(host(&out[0]), expected, "step {step} rows {rows}");
            }
        }
        let diagnostics = graphs.diagnostics();
        assert!(diagnostics["off"].is_null(), "{diagnostics}");
        assert_eq!(diagnostics["captures"], 2, "{diagnostics}");
        assert_eq!(diagnostics["verified"], 2, "{diagnostics}");
        assert_eq!(diagnostics["replays"], 4, "{diagnostics}");

        let x = Tensor::from_vec(vec![1f32; 6], (3, 2), &device).unwrap();
        graphs.run(3, 64, std::slice::from_ref(&x), &body).unwrap();
        let diagnostics = graphs.diagnostics();
        assert_eq!(diagnostics["evictions"], 1, "{diagnostics}");
        assert_eq!(diagnostics["shapes"], json!(["2", "3"]), "{diagnostics}");
    }
}
