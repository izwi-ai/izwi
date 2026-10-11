//! Piecewise CUDA graph capture for decode steps.
//!
//! A segment is a pure function of its input tensors, the model weights and a
//! device address table ([`DeviceTable`]) that runs between host-side work
//! (attention with per-step positions and KV metadata, sampling). A
//! [`SegmentGraph`] takes it through three phases:
//! 1. **warm**: run eagerly under Candle's htod-cache guard. That populates the
//!    strided-op parameter cache and the small-constant upload cache a capture
//!    must hit, since an upload inside a capture fails.
//! 2. **capture**: thread-local stream capture over stable input buffers, with
//!    cudarc's per-allocation event tracking off for the window. The graph is
//!    instantiated with `AUTO_FREE_ON_LAUNCH` and launched once, so its
//!    retained output memory is mapped.
//! 3. **replay**: copy the inputs into the stable buffers and launch. An input
//!    adopted at capture (another graph's retained output, already at a fixed
//!    address) is not copied.
//!
//! Captured outputs are graph memory at fixed addresses. They are valid until
//! this segment's next replay and are retained, with the stable inputs, until
//! the graph is destroyed behind a stream fence. The ownership rules follow
//! [`super::graphs::TensorIsland`]: nothing allocated before the capture may be
//! freed inside it, and the closure may not read device values on the host.
//! Nor may it upload host data. Candle's small-upload cache is per thread, so
//! an upload would miss after a thread switch, and a hit would bake a stale
//! value into the graph.
//! Off CUDA, only [`SegmentGraph::warm`] (eager execution) is available.
use candle_core::{Device, Result, Tensor};

/// A device array of `u64` entries (device addresses) that the host rewrites
/// between replays. A captured kernel reads the addresses of per-step tensors
/// from it (see `gdn::StateTable`).
pub struct DeviceTable {
    entries: usize,
    #[cfg(feature = "cuda")]
    slice: candle_core::cuda_backend::cudarc::driver::CudaSlice<u64>,
    #[cfg(feature = "cuda")]
    stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
}

impl DeviceTable {
    pub fn new(device: &Device, entries: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            let stream = cuda.cuda_stream();
            let slice = stream
                .alloc_zeros::<u64>(entries.max(1))
                .map_err(|error| candle_core::Error::Msg(format!("device table: {error}")))?;
            return Ok(Self {
                entries,
                slice,
                stream,
            });
        }
        let _ = entries;
        candle_core::bail!(
            "device address tables need a CUDA device, found {:?}",
            device.location()
        )
    }

    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Device address of entry 0.
    pub fn address(&self) -> u64 {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            let (address, _guard) = self.slice.device_ptr(&self.stream);
            address
        }
        #[cfg(not(feature = "cuda"))]
        0
    }

    /// Write `values` at `offset`, stream-ordered after the work already
    /// queued (a replay that reads the previous values has been launched).
    pub fn write(&mut self, offset: usize, values: &[u64]) -> Result<()> {
        let end = offset
            .checked_add(values.len())
            .filter(|end| *end <= self.entries)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "device table write {offset}+{} exceeds {} entries",
                    values.len(),
                    self.entries
                ))
            })?;
        #[cfg(feature = "cuda")]
        {
            let mut view = self.slice.slice_mut(offset..end);
            self.stream
                .memcpy_htod(values, &mut view)
                .map_err(|error| candle_core::Error::Msg(format!("device table write: {error}")))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = end;
            Ok(())
        }
    }
}

/// One input of a segment run. `adopt` marks a tensor already at a stable
/// address for the graph's lifetime (another segment's retained output): the
/// capture reads it in place, and replays expect the same tensor again.
#[derive(Clone, Copy)]
pub struct SegmentInput<'a> {
    pub tensor: &'a Tensor,
    pub adopt: bool,
}

/// A segment that can be captured once and replayed (see the module docs).
#[derive(Default)]
pub struct SegmentGraph {
    #[cfg(feature = "cuda")]
    captured: Option<device::Captured>,
}

impl SegmentGraph {
    pub fn is_captured(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.captured.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Run `f` eagerly on `inputs`. On CUDA the htod-cache guard is active, so
    /// a later capture of the same work finds its parameter caches warm.
    pub fn warm<F>(device: &Device, inputs: &[Tensor], f: F) -> Result<Vec<Tensor>>
    where
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            let _htod = cuda.enable_cuda_graph_htod_cache();
            return f(inputs);
        }
        let _ = device;
        f(inputs)
    }

    /// Capture `f` over stable copies of `inputs` (adopted inputs are used in
    /// place), then launch the graph once on those copies. On failure the
    /// stream leaves capture mode, CUDA errors the failed capture left on the
    /// context are cleared, and no graph is kept.
    pub fn capture<F>(&mut self, inputs: &[SegmentInput<'_>], f: F) -> Result<()>
    where
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        #[cfg(feature = "cuda")]
        {
            self.captured = None;
            self.captured = Some(device::capture(inputs, f)?);
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (inputs, f);
            candle_core::bail!("CUDA graph capture needs the cuda feature")
        }
    }

    /// Copy `inputs` into the stable buffers and launch the graph. Returns the
    /// retained outputs, valid until the next replay.
    pub fn replay(&self, inputs: &[SegmentInput<'_>]) -> Result<Vec<Tensor>> {
        #[cfg(feature = "cuda")]
        {
            let Some(captured) = &self.captured else {
                candle_core::bail!("segment graph replayed before capture")
            };
            captured.replay(inputs)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = inputs;
            candle_core::bail!("CUDA graph replay needs the cuda feature")
        }
    }

    /// Destroy the graph (behind a stream fence) and its retained buffers.
    pub fn reset(&mut self) {
        #[cfg(feature = "cuda")]
        {
            self.captured = None;
        }
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::SegmentInput;
    use candle_core::cuda_backend::cudarc::driver::{
        result,
        sys::{
            self, CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        },
        CudaContext,
    };
    use candle_core::{Result, Tensor};
    use std::sync::Arc;

    fn driver(what: &str, error: impl std::fmt::Display) -> candle_core::Error {
        candle_core::Error::Msg(format!("{what}: {error}"))
    }

    /// An instantiated graph held through raw driver handles. cudarc's
    /// `CudaGraph` goes through `bind_to_thread`, which first returns any
    /// error another call left on the context (`check_err`). For
    /// `end_capture`, that would return early and leave the stream in capture
    /// mode.
    struct Graph {
        graph: sys::CUgraph,
        exec: sys::CUgraphExec,
        device: candle_core::CudaDevice,
    }

    // SAFETY: CUDA graph APIs allow serialized use from any thread. Every
    // access goes through `&mut SegmentGraph` / `&SegmentGraph` behind the
    // owner's lock, and the owner fences before destruction.
    unsafe impl Send for Graph {}
    unsafe impl Sync for Graph {}

    impl Graph {
        fn launch(&self) -> Result<()> {
            let stream = self.device.cuda_stream();
            // SAFETY: the exec is live, and the stream is this device's.
            unsafe { result::graph::launch(self.exec, stream.cu_stream()) }
                .map_err(|error| driver("graph launch", error))
        }
    }

    impl Drop for Graph {
        fn drop(&mut self) {
            let ctx = self.device.cuda_stream().context().cu_ctx();
            // SAFETY: the handles are owned here and destroyed exactly once,
            // after the owner's fence; the context outlives the device handle.
            unsafe {
                let _ = result::ctx::set_current(ctx);
                let _ = result::graph::exec_destroy(self.exec);
                let _ = result::graph::destroy(self.graph);
            }
        }
    }

    /// Ends the stream capture on every exit path (error, early return,
    /// panic), so the per-thread stream never stays in capture mode.
    struct CaptureGuard {
        stream: sys::CUstream,
        active: bool,
    }

    impl CaptureGuard {
        fn finish(mut self) -> std::result::Result<sys::CUgraph, result::DriverError> {
            self.active = false;
            // SAFETY: this guard began the capture on `stream`.
            unsafe { result::stream::end_capture(self.stream) }
        }
    }

    impl Drop for CaptureGuard {
        fn drop(&mut self) {
            if !self.active {
                return;
            }
            // SAFETY: as in `finish`; a partial graph is discarded.
            unsafe {
                if let Ok(graph) = result::stream::end_capture(self.stream) {
                    if !graph.is_null() {
                        let _ = result::graph::destroy(graph);
                    }
                }
            }
        }
    }

    /// Turns cudarc's per-allocation event tracking off for the capture window.
    /// Each allocation gets read/write events, but izwi never enters
    /// multi-stream mode, so the events are never recorded. Freeing an
    /// allocation still waits on them, and inside a capture a wait on an event
    /// the capture did not record can invalidate it; the segments allocate and
    /// free intermediates inside the capture. In multi-stream mode the events
    /// do order streams, so capture is refused there.
    struct EventTrackingOff {
        ctx: Arc<CudaContext>,
        was_on: bool,
    }

    impl EventTrackingOff {
        fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
            if ctx.is_in_multi_stream_mode() {
                candle_core::bail!(
                    "segment capture needs a single-stream context (event tracking orders streams)"
                )
            }
            let was_on = ctx.is_event_tracking();
            if was_on {
                // SAFETY: single-stream mode: the events never order anything.
                unsafe { ctx.disable_event_tracking() };
            }
            Ok(Self {
                ctx: ctx.clone(),
                was_on,
            })
        }
    }

    impl Drop for EventTrackingOff {
        fn drop(&mut self) {
            if self.was_on {
                // SAFETY: restores the default for allocations made afterwards.
                unsafe { self.ctx.enable_event_tracking() };
            }
        }
    }

    pub(super) struct Captured {
        // Field order: the graph is destroyed before the buffers it references.
        graph: Option<Graph>,
        outputs: Vec<Tensor>,
        inputs: Vec<Tensor>,
        adopted: Vec<bool>,
    }

    impl Drop for Captured {
        fn drop(&mut self) {
            let Some(graph) = &self.graph else {
                return;
            };
            if graph.device.cuda_stream().synchronize().is_err() {
                // A failed fence cannot prove the last replay finished: leak
                // the graph and every buffer it may still touch.
                std::mem::forget(self.graph.take());
                std::mem::forget(std::mem::take(&mut self.outputs));
                std::mem::forget(std::mem::take(&mut self.inputs));
            }
        }
    }

    pub(super) fn capture<F>(inputs: &[SegmentInput<'_>], f: F) -> Result<Captured>
    where
        F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
    {
        let Some(first) = inputs.first() else {
            candle_core::bail!("a captured segment needs at least one input")
        };
        let device = first.tensor.device().as_cuda_device()?.clone();
        let stable = inputs
            .iter()
            .map(|input| {
                if input.adopt {
                    Ok(input.tensor.clone())
                } else {
                    input.tensor.contiguous()?.copy().map(|t| t.detach())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let stream = device.cuda_stream();
        let ctx = stream.context().clone();
        let _htod = device.enable_cuda_graph_htod_cache();
        let events = EventTrackingOff::new(&ctx)?;
        // SAFETY: the stream is this device's; the guard ends the capture.
        unsafe {
            result::stream::begin_capture(stream.cu_stream(), CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        }
        .map_err(|error| driver("begin capture", error))?;
        let guard = CaptureGuard {
            stream: stream.cu_stream(),
            active: true,
        };
        let computation = f(&stable);
        let ended = guard.finish();
        drop(events);
        let fail = |message: String| -> candle_core::Error {
            // Frees inside a failed capture record errors on the context;
            // clear them so they cannot fail an unrelated later call.
            if let Err(error) = ctx.check_err() {
                tracing::debug!(%error, "cleared CUDA errors left by a failed segment capture");
            }
            candle_core::Error::Msg(message)
        };
        let graph = match (computation, ended) {
            (Ok(outputs), Ok(graph)) if !graph.is_null() && !outputs.is_empty() => {
                // SAFETY: `graph` came from a completed capture.
                match unsafe {
                    result::graph::instantiate(
                        graph,
                        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    )
                } {
                    Ok(exec) => (
                        Graph {
                            graph,
                            exec,
                            device: device.clone(),
                        },
                        outputs,
                    ),
                    Err(error) => {
                        // SAFETY: the graph is ours and never instantiated.
                        unsafe {
                            let _ = result::graph::destroy(graph);
                        }
                        // Graph-memory outputs of an uninstantiated graph were
                        // never mapped: freeing them would be invalid.
                        std::mem::forget(outputs);
                        return Err(fail(format!("graph instantiate: {error}")));
                    }
                }
            }
            (Ok(outputs), ended) => {
                if let Ok(graph) = ended {
                    if !graph.is_null() {
                        // SAFETY: an unused graph from the capture.
                        unsafe {
                            let _ = result::graph::destroy(graph);
                        }
                    }
                }
                std::mem::forget(outputs);
                return Err(fail(match ended {
                    Err(error) => format!("end capture: {error}"),
                    Ok(_) => "segment capture recorded no work or outputs".into(),
                }));
            }
            (Err(error), ended) => {
                if let Ok(graph) = ended {
                    if !graph.is_null() {
                        // SAFETY: an unused graph from the capture.
                        unsafe {
                            let _ = result::graph::destroy(graph);
                        }
                    }
                }
                return Err(fail(format!("segment capture: {error}")));
            }
        };
        let (graph, outputs) = graph;
        // Launch once now, as `TensorIsland` does: the retained outputs are
        // graph memory, mapped only once the graph has run.
        if let Err(error) = graph.launch() {
            std::mem::forget(outputs);
            std::mem::forget(graph);
            return Err(fail(format!("first launch: {error}")));
        }
        Ok(Captured {
            graph: Some(graph),
            outputs,
            inputs: stable,
            adopted: inputs.iter().map(|input| input.adopt).collect(),
        })
    }

    impl Captured {
        pub(super) fn replay(&self, inputs: &[SegmentInput<'_>]) -> Result<Vec<Tensor>> {
            if inputs.len() != self.inputs.len() {
                candle_core::bail!(
                    "segment replay received {} inputs, captured {}",
                    inputs.len(),
                    self.inputs.len()
                )
            }
            for ((stable, input), adopted) in self.inputs.iter().zip(inputs).zip(&self.adopted) {
                if *adopted && input.tensor.id() == stable.id() {
                    continue;
                }
                if input.tensor.dims() != stable.dims() || input.tensor.dtype() != stable.dtype() {
                    candle_core::bail!(
                        "segment replay input {:?} {:?} differs from the captured {:?} {:?}",
                        input.tensor.dims(),
                        input.tensor.dtype(),
                        stable.dims(),
                        stable.dtype()
                    )
                }
                stable.slice_set(&input.tensor.contiguous()?, 0, 0)?;
            }
            let Some(graph) = &self.graph else {
                candle_core::bail!("segment graph was destroyed")
            };
            graph.launch()?;
            Ok(self.outputs.clone())
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::DType;

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// Capture runs nothing; each replay recomputes from the current inputs;
    /// adopted inputs are read in place; the table round-trips addresses.
    #[test]
    fn cuda_segment_graph_replays_fresh_inputs() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let x0 = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let w = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.25], (2, 2), &device).unwrap();
        let f = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            // A strided op (transpose + contiguous) exercises the warm caches.
            let y = (inputs[0].matmul(&w.t()?.contiguous()?)? + 1.0)?;
            Ok(vec![(&y * 2.0)?])
        };
        let warm = SegmentGraph::warm(&device, std::slice::from_ref(&x0), f).unwrap();
        let mut graph = SegmentGraph::default();
        let input = SegmentInput {
            tensor: &x0,
            adopt: false,
        };
        graph.capture(&[input], f).unwrap();
        assert!(graph.is_captured());
        assert_eq!(host(&graph.replay(&[input]).unwrap()[0]), host(&warm[0]));

        let x1 = Tensor::from_vec(vec![-1f32, 0.5, 2.0, 8.0], (2, 2), &device).unwrap();
        let expected = host(&f(std::slice::from_ref(&x1)).unwrap()[0]);
        let replayed = graph
            .replay(&[SegmentInput {
                tensor: &x1,
                adopt: false,
            }])
            .unwrap();
        assert_eq!(
            host(&replayed[0]),
            expected,
            "replay must read the new input"
        );

        // A second graph adopts the first one's retained output in place.
        let mut chained = SegmentGraph::default();
        let adopted = SegmentInput {
            tensor: &replayed[0],
            adopt: true,
        };
        chained
            .capture(&[adopted], |inputs| Ok(vec![(&inputs[0] - 3.0)?]))
            .unwrap();
        let again = graph
            .replay(&[SegmentInput {
                tensor: &x0,
                adopt: false,
            }])
            .unwrap();
        let chained_out = chained
            .replay(&[SegmentInput {
                tensor: &again[0],
                adopt: true,
            }])
            .unwrap();
        let want = host(&warm[0]).iter().map(|v| v - 3.0).collect::<Vec<_>>();
        assert_eq!(host(&chained_out[0]), want);

        let mut table = DeviceTable::new(&device, 4).unwrap();
        table.write(1, &[7, 9]).unwrap();
        assert_ne!(table.address(), 0);
        assert!(table.write(3, &[1, 2]).is_err(), "out of bounds");
        graph.reset();
        assert!(!graph.is_captured());
    }
}
