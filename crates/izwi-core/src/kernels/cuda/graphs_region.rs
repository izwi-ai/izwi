//! Model-owned multi-operation regions. See TensorIsland::run_multi for the capture
//! ownership contract. In particular, Candle Q8 fast-MMVQ has hidden growable
//! scratch and must not be captured through this interface.
use candle_core::{Result, Tensor};

pub struct IslandOutput {
    pub outputs: Vec<Tensor>,
    /// Retain every intermediate allocation referenced by a captured operation.
    pub intermediates: Vec<Tensor>,
}
#[derive(Default)]
pub struct TensorIsland {
    warmups: std::sync::atomic::AtomicU64,
    captures: std::sync::atomic::AtomicU64,
    replays: std::sync::atomic::AtomicU64,
    negative_fallbacks: std::sync::atomic::AtomicU64,
    #[cfg(feature = "cuda")]
    cache: std::sync::Mutex<std::collections::VecDeque<(device::Key, device::Entry)>>,
}
impl std::fmt::Debug for TensorIsland {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TensorIsland")
    }
}
impl TensorIsland {
    pub fn diagnostics(&self) -> serde_json::Value {
        use std::sync::atomic::Ordering::Relaxed;
        #[cfg(feature = "cuda")]
        let cache_bytes = device::current(&self.cache.lock().unwrap_or_else(|e| e.into_inner()));
        #[cfg(not(feature = "cuda"))]
        let cache_bytes = 0usize;
        serde_json::json!({"warmups":self.warmups.load(Relaxed),"captures":self.captures.load(Relaxed),"replays":self.replays.load(Relaxed),"negative_fallbacks":self.negative_fallbacks.load(Relaxed),"cache_bytes":cache_bytes})
    }
    /// Execute a pure region over stable dynamic-input copies. `owners` retains
    /// every static tensor referenced by the closure. No host reads, state
    /// mutation, hidden growable scratch, or input mutation is permitted. Return
    /// all intermediates, including contiguous/cast buffers, in IslandOutput.
    ///
    /// `allocation_bound_bytes` must bound all output/intermediate allocations
    /// made by the closure; it is reserved BEFORE capture. The observed retained
    /// tensors are checked against this bound. Budget also includes stable input
    /// buffers and a detached result copy. Failures are cached until eviction or
    /// invalidation; a failed replay propagates rather than repeating eager work.
    #[allow(clippy::too_many_arguments)]
    pub fn run_multi<F>(
        &self,
        inputs: &[&Tensor],
        owners: &[Tensor],
        generation: u64,
        provider: &'static str,
        budget_bytes: usize,
        allocation_bound_bytes: usize,
        f: F,
    ) -> Result<Option<Vec<Tensor>>>
    where
        F: Fn(&[Tensor]) -> Result<IslandOutput>,
    {
        #[cfg(feature = "cuda")]
        {
            device::run(
                self,
                inputs,
                owners,
                generation,
                provider,
                budget_bytes,
                allocation_bound_bytes,
                f,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                inputs,
                owners,
                generation,
                provider,
                budget_bytes,
                allocation_bound_bytes,
                f,
            );
            Ok(None)
        }
    }
    /// Drops graph memory and static weight references, from any worker.
    pub fn invalidate(&self) {
        #[cfg(feature = "cuda")]
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{
        sys::{
            CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        },
        CudaGraph,
    };
    use candle_core::{DType, DeviceLocation, TensorId};
    use std::{collections::HashSet, mem::ManuallyDrop};
    #[derive(PartialEq, Eq)]
    pub(super) struct Key {
        generation: u64,
        provider: &'static str,
        shapes: Vec<Vec<usize>>,
        dtypes: Vec<DType>,
        device: DeviceLocation,
        owners: Vec<TensorId>,
    }
    pub(super) enum Entry {
        Negative,
        Live(Box<Live>),
    }
    pub(super) struct Live {
        // Drop graph before tensor owners; its destructor fences the capture stream.
        graph: Option<Graph>,
        captured: Option<IslandOutput>,
        inputs: Vec<Tensor>,
        _owners: Vec<Tensor>,
        bytes: usize,
    }
    struct Graph {
        raw: ManuallyDrop<CudaGraph>,
        device: candle_core::CudaDevice,
    }
    // SAFETY: CUDA graph APIs permit serialized cross-thread use. All access is
    // private to a model-owned Mutex, including destruction. Cudarc launch binds
    // its context, and our Drop binds and fences before destroying the graph.
    unsafe impl Send for Graph {}
    impl Drop for Graph {
        fn drop(&mut self) {
            let stream = self.device.cuda_stream();
            let _ = stream.context().bind_to_thread();
            unsafe { ManuallyDrop::drop(&mut self.raw) }
        }
    }
    static BLOCKED: std::sync::LazyLock<std::sync::Mutex<HashSet<usize>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));
    impl Drop for Live {
        fn drop(&mut self) {
            if let Some(graph) = &self.graph {
                if graph.device.cuda_stream().synchronize().is_err() {
                    BLOCKED
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(graph.device.cuda_stream().context().cu_ctx() as usize);
                    // A failed fence cannot prove last use. Quarantine ALL
                    // captured owners and refuse this context until restart.
                    std::mem::forget(self.graph.take());
                    std::mem::forget(self.captured.take());
                    std::mem::forget(std::mem::take(&mut self.inputs));
                    std::mem::forget(std::mem::take(&mut self._owners));
                }
            }
        }
    }
    fn size(t: &Tensor) -> Option<usize> {
        use candle_core::{cuda_backend::CudaStorageSlice, Storage};
        let (storage, _) = t.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            return None;
        };
        // Charge the physical backing, not merely a narrow view's element
        // count. Multiple distinct views may conservatively charge it twice.
        let count = match &storage.slice {
            CudaStorageSlice::U8(s) => s.len(),
            CudaStorageSlice::U32(s) => s.len(),
            CudaStorageSlice::I16(s) => s.len(),
            CudaStorageSlice::I32(s) => s.len(),
            CudaStorageSlice::I64(s) => s.len(),
            CudaStorageSlice::F16(s) => s.len(),
            CudaStorageSlice::BF16(s) => s.len(),
            CudaStorageSlice::F32(s) => s.len(),
            CudaStorageSlice::F64(s) => s.len(),
            CudaStorageSlice::F8E4M3(s) => s.len(),
            CudaStorageSlice::F6E2M3(s) => s.len(),
            CudaStorageSlice::F6E3M2(s) => s.len(),
            CudaStorageSlice::F4(s) => s.len(),
            CudaStorageSlice::F8E8M0(s) => s.len(),
        };
        count.checked_mul(t.dtype().size_in_bytes())
    }
    fn retained(result: &IslandOutput) -> Option<usize> {
        let mut seen = HashSet::new();
        result
            .outputs
            .iter()
            .chain(&result.intermediates)
            .try_fold(0usize, |sum, t| {
                if seen.insert(t.id()) {
                    sum.checked_add(size(t)?)
                } else {
                    Some(sum)
                }
            })
    }
    pub(super) fn current(cache: &std::collections::VecDeque<(Key, Entry)>) -> usize {
        cache
            .iter()
            .map(|(_, e)| match e {
                Entry::Live(l) => l.bytes,
                Entry::Negative => 0,
            })
            .sum()
    }
    fn valid(result: &IslandOutput, input: &Tensor, bound: usize) -> bool {
        !result.outputs.is_empty()
            && retained(result).is_some_and(|n| n <= bound)
            && result
                .outputs
                .iter()
                .chain(&result.intermediates)
                .all(|t| t.device().same_device(input.device()))
    }
    fn negative(
        island: &TensorIsland,
        cache: &mut std::collections::VecDeque<(Key, Entry)>,
        key: Key,
    ) {
        island
            .negative_fallbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if cache
            .iter()
            .filter(|(_, e)| matches!(e, Entry::Negative))
            .count()
            >= 64
        {
            if let Some(i) = cache.iter().position(|(_, e)| matches!(e, Entry::Negative)) {
                cache.remove(i);
            }
        }
        cache.push_back((key, Entry::Negative));
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run<F>(
        island: &TensorIsland,
        inputs: &[&Tensor],
        owners: &[Tensor],
        generation: u64,
        provider: &'static str,
        budget: usize,
        bound: usize,
        f: F,
    ) -> Result<Option<Vec<Tensor>>>
    where
        F: Fn(&[Tensor]) -> Result<IslandOutput>,
    {
        let Some(first) = inputs.first() else {
            return Ok(None);
        };
        if !first.device().is_cuda()
            || inputs
                .iter()
                .any(|t| t.elem_count() == 0 || !t.device().same_device(first.device()))
            || owners
                .iter()
                .any(|t| !t.device().same_device(first.device()))
        {
            return Ok(None);
        }
        if BLOCKED.lock().unwrap_or_else(|e| e.into_inner()).contains(
            &(first
                .device()
                .as_cuda_device()?
                .cuda_stream()
                .context()
                .cu_ctx() as usize),
        ) {
            candle_core::bail!(
                "CUDA graph context was quarantined after a failed fence; restart the process"
            )
        }
        let Some(input_bytes) = inputs.iter().try_fold(0usize, |n, t| {
            n.checked_add(t.elem_count().checked_mul(t.dtype().size_in_bytes())?)
        }) else {
            return Ok(None);
        };
        // The output copy is at most the allocation bound. Conservative reservation
        // also handles providers returning a larger output than their input.
        let Some(required) = bound
            .checked_mul(2)
            .and_then(|n| n.checked_add(input_bytes))
        else {
            return Ok(None);
        };
        if bound == 0 || required > budget {
            return Ok(None);
        }
        let key = Key {
            generation,
            provider,
            shapes: inputs.iter().map(|t| t.dims().to_vec()).collect(),
            dtypes: inputs.iter().map(|t| t.dtype()).collect(),
            device: first.device().location(),
            owners: owners.iter().map(Tensor::id).collect(),
        };
        let mut cache = island.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = cache.iter().position(|(k, e)| {
            *k == key
                && match e {
                    Entry::Live(l) => l.inputs[0].device().same_device(first.device()),
                    Entry::Negative => true,
                }
        }) {
            let (_, entry) = cache.remove(index).unwrap();
            let Entry::Live(mut live) = entry else {
                negative(island, &mut cache, key);
                return Ok(None);
            };
            // A smaller budget or changed upper bound invalidates before replay.
            if live.bytes > budget || required != live.bytes {
                negative(island, &mut cache, key);
                return Ok(None);
            }
            while current(&cache).saturating_add(live.bytes) > budget {
                if cache.pop_front().is_none() {
                    return Ok(None);
                }
            }
            for (dst, src) in live.inputs.iter().zip(inputs) {
                dst.slice_set(&src.contiguous()?, 0, 0)?;
            }
            let was_captured = live.graph.is_some();
            if let Some(graph) = &live.graph {
                if let Err(e) = graph.raw.launch() {
                    negative(island, &mut cache, key);
                    return Err(candle_core::Error::Msg(format!(
                        "tensor region graph replay failed: {e}"
                    )));
                }
            } else {
                let device = first.device().as_cuda_device()?;
                let stream = device.cuda_stream();
                // Warmup already loaded modules and checked retained-byte geometry.
                let _htod = device.enable_cuda_graph_htod_cache();
                if stream
                    .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .is_err()
                {
                    negative(island, &mut cache, key);
                    return Ok(None);
                }
                let computation = f(&live.inputs);
                // Never short circuit end_capture, even on closure failure.
                let captured = stream.end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                let (result, graph) = match (computation, captured) {
                    (Ok(result), Ok(Some(graph))) => (result, graph),
                    _ => {
                        negative(island, &mut cache, key);
                        return Ok(None);
                    }
                };
                let graph = Graph {
                    raw: ManuallyDrop::new(graph),
                    device: device.clone(),
                };
                if !valid(&result, first, bound) {
                    drop(graph);
                    negative(island, &mut cache, key);
                    return Ok(None);
                }
                // Keep capture allocations alive until the graph has been destroyed.
                live.captured = Some(result);
                live.graph = Some(graph);
                island
                    .captures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                if let Err(e) = live.graph.as_ref().unwrap().raw.launch() {
                    negative(island, &mut cache, key);
                    return Err(candle_core::Error::Msg(format!(
                        "tensor region initial replay failed: {e}"
                    )));
                }
            }
            if was_captured {
                island
                    .replays
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let out = live
                .captured
                .as_ref()
                .unwrap()
                .outputs
                .iter()
                .map(|t| t.copy().map(|t| t.detach()))
                .collect::<Result<Vec<_>>>()?;
            cache.push_back((key, Entry::Live(live)));
            return Ok(Some(out));
        }
        // Admit a stable working subset; do not evict earlier layers on every
        // forward when the full model or a new verification width cannot fit.
        if cache
            .iter()
            .filter(|(_, e)| matches!(e, Entry::Live(_)))
            .count()
            >= 256
            || current(&cache).saturating_add(required) > budget
        {
            negative(island, &mut cache, key);
            return Ok(None);
        }
        let stable = inputs
            .iter()
            .map(|t| t.contiguous()?.copy().map(|t| t.detach()))
            .collect::<Result<Vec<_>>>()?;
        let result = match f(&stable) {
            Ok(result) => result,
            Err(e) => {
                negative(island, &mut cache, key);
                return Err(e);
            }
        };
        if !valid(&result, first, bound) {
            negative(island, &mut cache, key);
            return Ok(None);
        }
        island
            .warmups
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let out = result.outputs;
        cache.push_back((
            key,
            Entry::Live(Box::new(Live {
                graph: None,
                captured: None,
                inputs: stable,
                _owners: owners.to_vec(),
                bytes: required,
            })),
        ));
        Ok(Some(out))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_island_does_not_invoke_closure() {
        let island = TensorIsland::default();
        let t = Tensor::zeros((1, 4), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(island
            .run_multi(&[&t], &[], 1, "cpu", 4096, 256, |_| panic!(
                "unsupported closure invoked"
            ))
            .unwrap()
            .is_none());
        island.invalidate();
    }
    #[test]
    fn model_owned_island_is_send_sync() {
        fn check<T: Send + Sync>() {}
        check::<TensorIsland>();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    #[test]
    fn cuda_graph_multioutput_replay_inputs_isolation_budget_and_crossworker_invalidation() {
        let Some(device) = super::super::super::cuda_test_device() else {
            return;
        };
        let island = Arc::new(TensorIsland::default());
        let weight = Tensor::ones(128, DType::F32, &device).unwrap();
        let weight_cpu = weight.to_device(&candle_core::Device::Cpu).unwrap();
        let mut previous: Option<Vec<Tensor>> = None;
        for step in 0..4 {
            let x = Tensor::full(step as f32 + 1., (2, 128), &device).unwrap();
            let r = Tensor::ones((2, 128), DType::F32, &device).unwrap();
            let result = island
                .run_multi(
                    &[&x, &r],
                    std::slice::from_ref(&weight),
                    42,
                    "residual-rms-test",
                    16384,
                    2048,
                    |inputs| {
                        let sum = (&inputs[0] + &inputs[1])?;
                        let norm = candle_nn::ops::rms_norm(&sum, &weight, 1e-6)?;
                        Ok(IslandOutput {
                            outputs: vec![sum, norm],
                            intermediates: vec![],
                        })
                    },
                )
                .unwrap()
                .expect("required CUDA region did not execute");
            assert_eq!(result[0].to_vec2::<f32>().unwrap()[0][0], step as f32 + 2.);
            let expected = candle_nn::ops::rms_norm(
                &Tensor::full(step as f32 + 2., (2, 128), &candle_core::Device::Cpu).unwrap(),
                &weight_cpu,
                1e-6,
            )
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
            let actual = result[1].to_vec2::<f32>().unwrap();
            assert!((actual[0][0] - expected[0][0]).abs() < 1e-5);
            if let Some(previous) = previous {
                assert_eq!(
                    previous[0].to_vec2::<f32>().unwrap()[0][0],
                    step as f32 + 1.
                );
            }
            previous = Some(result);
        }
        let d = island.diagnostics();
        assert_eq!(d["warmups"], 1);
        assert_eq!(d["captures"], 1);
        assert_eq!(d["replays"], 2);
        assert!(d["cache_bytes"].as_u64().unwrap() <= 16384);
        let owned = island.clone();
        std::thread::spawn(move || owned.invalidate())
            .join()
            .unwrap();
        assert_eq!(island.diagnostics()["cache_bytes"], 0);
        let calls = AtomicUsize::new(0);
        let x = Tensor::ones((1, 128), DType::F32, &device).unwrap();
        assert!(island
            .run_multi(&[&x], &[], 43, "budget-denied", 0, 1024, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                unreachable!()
            })
            .unwrap()
            .is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // Warm succeeds, capture is deliberately rejected. A third invocation
        // must hit the negative cache without executing the closure again.
        for step in 0..3 {
            let result = island
                .run_multi(&[&x], &[], 44, "failure-injection", 16384, 1024, |inputs| {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    if call == 1 {
                        candle_core::bail!("injected capture failure")
                    };
                    let out = (&inputs[0] + 1.)?;
                    Ok(IslandOutput {
                        outputs: vec![out],
                        intermediates: vec![],
                    })
                })
                .unwrap();
            assert_eq!(result.is_some(), step == 0);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        island.invalidate();
    }
}
