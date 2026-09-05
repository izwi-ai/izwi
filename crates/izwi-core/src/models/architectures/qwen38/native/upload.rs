//! Single final Q8 CUDA allocation with a bounded two-buffer pinned upload ring.
//! The exclusive destination is never exposed before all copies complete.
use super::*;
use candle_core::quantized::cuda::QCudaStorage;
use cudarc::driver::{CudaEvent, CudaStream, PinnedHostSlice};
use std::sync::{Mutex, OnceLock};

static FAILED_CONTEXTS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub(super) fn ensure_context_healthy(device: &Device) -> Result<()> {
    if let Device::Cuda(cuda) = device {
        check_context(&cuda.cuda_stream())?;
    }
    Ok(())
}
fn check_context(stream: &CudaStream) -> Result<()> {
    let failed = FAILED_CONTEXTS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if failed.contains(&(stream.context().cu_ctx() as usize)) {
        return Err(cuda_error(
            "CUDA context quarantined after an unprovable upload completion; restart the process",
        ));
    }
    Ok(())
}

struct Slot {
    buffer: PinnedHostSlice<u8>,
    done: CudaEvent,
    pending: bool,
}
pub(super) struct Q8Upload {
    storage: Option<QCudaStorage>,
    stream: Arc<CudaStream>,
    slots: Vec<Slot>,
    next: usize,
    bytes: usize,
    complete: bool,
}
fn cuda_error(e: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Q8 CUDA upload: {e}"))
}
impl Q8Upload {
    pub fn new(
        device: &candle_core::CudaDevice,
        elements: usize,
        tile_bytes: usize,
        pinned: bool,
    ) -> Result<Self> {
        let bytes = (elements / 32)
            .checked_mul(34)
            .ok_or_else(|| cuda_error("size overflow"))?;
        let stream = device.cuda_stream();
        check_context(&stream)?;
        // QCudaStorage::zeros allocates the final padded destination once. Unlike
        // quantize_onto, no replacement quantized device allocation is made.
        let storage = QCudaStorage::zeros(device, elements, GgmlDType::Q8_0)?;
        let mut slots = Vec::new();
        if pinned {
            for _ in 0..2 {
                // SAFETY: initialized in full before any copy reads this memory.
                let allocated = unsafe { stream.context().alloc_pinned::<u8>(tile_bytes) };
                let Ok(mut buffer) = allocated else {
                    tracing::debug!(
                        "Pinned Q8 staging unavailable; using synchronous bounded transfers"
                    );
                    slots.clear();
                    break;
                };
                buffer.as_mut_slice().map_err(cuda_error)?.fill(0);
                let done = stream.context().new_event(None).map_err(cuda_error)?;
                slots.push(Slot {
                    buffer,
                    done,
                    pending: false,
                });
            }
        }
        Ok(Self {
            storage: Some(storage),
            stream,
            slots,
            next: 0,
            bytes,
            complete: false,
        })
    }
    pub fn uses_pinned(&self) -> bool {
        !self.slots.is_empty()
    }
    pub fn push(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty()
            || !bytes.len().is_multiple_of(34)
            || !offset.is_multiple_of(34)
            || offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > self.bytes)
        {
            return Err(cuda_error("invalid tile bounds"));
        }
        self.stream.context().bind_to_thread().map_err(cuda_error)?;
        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| cuda_error("upload already finished"))?;
        let (ptr, _guard) = storage.device_ptr_with_guard(&self.stream)?;
        let destination = (ptr as u64)
            .checked_add(offset as u64)
            .ok_or_else(|| cuda_error("pointer overflow"))?;
        if self.slots.is_empty() {
            // Complete zero initialization before a synchronous default-stream
            // transfer. Source Vec is not pinned and may be released on return.
            self.stream.synchronize().map_err(cuda_error)?;
            // SAFETY: exclusive destination, checked offset/length, live source,
            // bound CUDA context, synchronous lifetime and no in-flight writers.
            unsafe { cudarc::driver::result::memcpy_htod_sync(destination, bytes) }
                .map_err(cuda_error)?;
        } else {
            let slot = &mut self.slots[self.next];
            if slot.pending {
                slot.done.synchronize().map_err(cuda_error)?;
                slot.pending = false;
            }
            let host = slot.buffer.as_mut_slice().map_err(cuda_error)?;
            if bytes.len() > host.len() {
                return Err(cuda_error("tile exceeds pinned staging"));
            }
            host[..bytes.len()].copy_from_slice(bytes);
            // SAFETY: exclusive final allocation with validated range; pinned
            // buffer remains alive and is not reused until its completion event.
            // All writes and Candle zeroing use this same stream. Drop fences
            // even error paths before releasing either source or destination.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    destination,
                    &host[..bytes.len()],
                    self.stream.cu_stream(),
                )
            }
            .map_err(cuda_error)?;
            slot.done.record(&self.stream).map_err(cuda_error)?;
            slot.pending = true;
            self.next = (self.next + 1) % self.slots.len();
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<QCudaStorage> {
        self.stream.synchronize().map_err(cuda_error)?;
        self.complete = true;
        self.storage
            .take()
            .ok_or_else(|| cuda_error("upload already finished"))
    }
}
impl Drop for Q8Upload {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        // A failed copy/event recording may still have enqueued DMA. Only a
        // successful stream fence proves that pinned buffers can be freed.
        if let Err(error) = self.stream.synchronize() {
            let context = self.stream.context().cu_ctx() as usize;
            FAILED_CONTEXTS
                .get_or_init(|| Mutex::new(HashSet::new()))
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(context);
            let host_bytes: usize = self.slots.iter().map(|s| s.buffer.num_bytes()).sum();
            tracing::error!(%error,context,quarantined_host_bytes=host_bytes,quarantined_device_bytes=self.bytes+544,"Q8 upload fence failed; retaining DMA owners and blocking further loader allocations on this context; process restart is required");
            // Retaining the context also prevents its identity being reused.
            // A repeated load on it is rejected before allocation. At most this
            // failed upload's two pinned tiles and single final projection leak.
            super::loading::release_after_fence(
                (
                    std::mem::take(&mut self.slots),
                    self.storage.take(),
                    self.stream.clone(),
                ),
                false,
            );
        }
    }
}

/// Aligned, bounded conversion for a dense tensor whose source and execution
/// dtypes differ (or whose Safetensors payload is unaligned).
pub(super) fn dense_from_values(
    values: impl Iterator<Item = Result<f32>>,
    shape: &[usize],
    target: ProjectionMaterialization,
    device: &Device,
    budget: usize,
) -> Result<Tensor> {
    use cudarc::driver::DevicePtr;
    let Device::Cuda(cuda) = device else {
        return Err(cuda_error("dense tile device is not CUDA"));
    };
    ensure_context_healthy(device)?;
    let tensor = Tensor::zeros(shape, target.dtype(), device)?;
    let stream = cuda.cuda_stream();
    let (storage, _) = tensor.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        return Err(cuda_error("dense CUDA storage missing"));
    };
    macro_rules! upload {
        ($ty:ty,$convert:expr) => {{
            let capacity = (budget / std::mem::size_of::<$ty>())
                .min(4 * 1024 * 1024 / std::mem::size_of::<$ty>());
            if capacity == 0 {
                return Err(cuda_error("dense staging budget below one element"));
            }
            let mut staging = Vec::<$ty>::with_capacity(capacity);
            let (ptr, _guard) = storage.as_cuda_slice::<$ty>()?.device_ptr(&stream);
            // Complete initialization; all following copies are synchronous and
            // the fresh tensor remains exclusively owned until successful return.
            stream.synchronize().map_err(cuda_error)?;
            stream.context().bind_to_thread().map_err(cuda_error)?;
            let mut offset = 0usize;
            for value in values {
                let converted = ($convert)(value?);
                if !converted.is_finite() {
                    return Err(cuda_error("dense value overflows execution dtype"));
                }
                staging.push(converted);
                if staging.len() == capacity {
                    copy_dense(ptr, offset, &staging, tensor.elem_count())?;
                    offset += staging.len();
                    staging.clear();
                }
            }
            if !staging.is_empty() {
                copy_dense(ptr, offset, &staging, tensor.elem_count())?;
                offset += staging.len();
            }
            if offset != tensor.elem_count() {
                return Err(cuda_error("incomplete dense tensor"));
            }
        }};
    }
    match target {
        ProjectionMaterialization::F32 => upload!(f32, |v: f32| v),
        ProjectionMaterialization::F16 => upload!(f16, f16::from_f32),
        ProjectionMaterialization::BF16 => upload!(bf16, bf16::from_f32),
    }
    // Clone the handle while the storage read guard remains in scope.
    Ok(tensor.clone())
}
fn copy_dense<T>(pointer: u64, offset: usize, values: &[T], total: usize) -> Result<()> {
    if offset.checked_add(values.len()).is_none_or(|n| n > total) {
        return Err(cuda_error("dense copy out of bounds"));
    }
    let byte_offset = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| cuda_error("dense offset overflow"))?;
    let destination = pointer
        .checked_add(byte_offset as u64)
        .ok_or_else(|| cuda_error("dense pointer overflow"))?;
    // SAFETY: the caller holds exclusive ownership of a newly allocated tensor;
    // exact typed range checked above, live aligned staging and synchronous DMA.
    unsafe { cudarc::driver::result::memcpy_htod_sync(destination, values) }.map_err(cuda_error)
}
