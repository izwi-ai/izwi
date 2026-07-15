//! Backing-storage accounting for model-owned session tensors.

use std::collections::HashSet;

use candle_core::{CpuStorage, Storage, Tensor};

/// Copy a tensor into storage whose backing allocation matches its logical size.
///
/// Candle's Metal scratch allocator may satisfy a small operation from a larger
/// pooled buffer. Persistent recurrent/KV state must not retain that transient
/// backing allocation because physical cache accounting would become inaccurate.
pub(crate) fn compact_tensor_storage(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let tensor = tensor.contiguous()?;

    #[cfg(feature = "metal")]
    if let candle_core::Device::Metal(device) = tensor.device() {
        use candle_core::{op::BackpropOp, MetalStorage};

        let element_count = tensor.elem_count();
        if element_count == 0 {
            return Ok(tensor);
        }
        let dtype = tensor.dtype();
        let byte_count = element_count
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| {
                candle_core::Error::Msg("persistent Metal tensor byte size overflow".to_string())
            })?;
        let target = device.new_private_buffer(element_count, dtype, "persistent-state")?;
        let (storage, layout) = tensor.storage_and_layout();
        let Storage::Metal(source) = &*storage else {
            return Err(candle_core::Error::Msg(
                "Metal tensor exposed non-Metal storage".to_string(),
            ));
        };
        let source_offset = layout
            .start_offset()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| {
                candle_core::Error::Msg("persistent Metal tensor offset overflow".to_string())
            })?;
        {
            let mut blit = device.blit_command_encoder()?;
            blit.copy_from_buffer(source.buffer(), source_offset, &target, 0, byte_count);
        }
        let storage = MetalStorage::new(target, device.clone(), element_count, dtype);
        return Ok(Tensor::from_storage(
            Storage::Metal(storage),
            tensor.shape().clone(),
            BackpropOp::none(),
            false,
        ));
    }

    Ok(tensor)
}

/// Deep-copy persistent state without routing Metal through Candle's pooled
/// allocator, which may return an oversized backing allocation.
pub(crate) fn deep_copy_tensor_storage(tensor: &Tensor) -> candle_core::Result<Tensor> {
    if tensor.device().is_metal() {
        compact_tensor_storage(tensor)
    } else {
        tensor.copy()
    }
}

/// Accumulates the backing allocations retained by a set of Candle tensors.
///
/// Tensor views share a Candle storage allocation. Accounting logical tensor
/// shapes would therefore undercount a narrow view that keeps a larger backing
/// allocation alive. This collector keys allocations by their shared storage
/// object and counts each backing allocation once.
#[derive(Debug, Default)]
pub(crate) struct TensorStorageAccounting {
    seen: HashSet<usize>,
    bytes: u64,
}

impl TensorStorageAccounting {
    pub(crate) fn add_tensor(&mut self, tensor: &Tensor) -> Option<()> {
        let (storage, _) = tensor.storage_and_layout();
        let storage_key = std::ptr::from_ref::<Storage>(&storage) as usize;
        if !self.seen.insert(storage_key) {
            return Some(());
        }

        self.add_bytes(storage_bytes(&storage)?)
    }

    pub(crate) fn add_bytes(&mut self, bytes: u64) -> Option<()> {
        self.bytes = self.bytes.checked_add(bytes)?;
        Some(())
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

fn storage_bytes(storage: &Storage) -> Option<u64> {
    match storage {
        Storage::Cpu(storage) => cpu_storage_bytes(storage),
        Storage::Cuda(storage) => cuda_storage_bytes(storage),
        Storage::Metal(storage) => metal_storage_bytes(storage),
    }
}

fn allocation_bytes(capacity: usize, bytes_per_element: usize) -> Option<u64> {
    let bytes = capacity.checked_mul(bytes_per_element)?;
    u64::try_from(bytes).ok()
}

fn cpu_storage_bytes(storage: &CpuStorage) -> Option<u64> {
    match storage {
        CpuStorage::U8(values) => allocation_bytes(values.capacity(), 1),
        CpuStorage::U32(values) => allocation_bytes(values.capacity(), 4),
        CpuStorage::I16(values) => allocation_bytes(values.capacity(), 2),
        CpuStorage::I32(values) => allocation_bytes(values.capacity(), 4),
        CpuStorage::I64(values) => allocation_bytes(values.capacity(), 8),
        CpuStorage::BF16(values) => allocation_bytes(values.capacity(), 2),
        CpuStorage::F16(values) => allocation_bytes(values.capacity(), 2),
        CpuStorage::F32(values) => allocation_bytes(values.capacity(), 4),
        CpuStorage::F64(values) => allocation_bytes(values.capacity(), 8),
        CpuStorage::F8E4M3(values) => allocation_bytes(values.capacity(), 1),
        CpuStorage::F6E2M3(values)
        | CpuStorage::F6E3M2(values)
        | CpuStorage::F4(values)
        | CpuStorage::F8E8M0(values) => allocation_bytes(values.capacity(), 1),
    }
}

#[cfg(feature = "cuda")]
fn cuda_storage_bytes(storage: &candle_core::CudaStorage) -> Option<u64> {
    use candle_core::cuda_backend::CudaStorageSlice;

    match &storage.slice {
        CudaStorageSlice::U8(values) => allocation_bytes(values.len(), 1),
        CudaStorageSlice::U32(values) => allocation_bytes(values.len(), 4),
        CudaStorageSlice::I16(values) => allocation_bytes(values.len(), 2),
        CudaStorageSlice::I32(values) => allocation_bytes(values.len(), 4),
        CudaStorageSlice::I64(values) => allocation_bytes(values.len(), 8),
        CudaStorageSlice::BF16(values) => allocation_bytes(values.len(), 2),
        CudaStorageSlice::F16(values) => allocation_bytes(values.len(), 2),
        CudaStorageSlice::F32(values) => allocation_bytes(values.len(), 4),
        CudaStorageSlice::F64(values) => allocation_bytes(values.len(), 8),
        CudaStorageSlice::F8E4M3(values) => allocation_bytes(values.len(), 1),
        CudaStorageSlice::F6E2M3(values)
        | CudaStorageSlice::F6E3M2(values)
        | CudaStorageSlice::F4(values)
        | CudaStorageSlice::F8E8M0(values) => allocation_bytes(values.len(), 1),
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_storage_bytes(_storage: &candle_core::CudaStorage) -> Option<u64> {
    // A CUDA storage cannot be constructed without Candle's CUDA feature.
    None
}

#[cfg(feature = "metal")]
fn metal_storage_bytes(storage: &candle_core::MetalStorage) -> Option<u64> {
    u64::try_from(storage.buffer().length()).ok()
}

#[cfg(not(feature = "metal"))]
fn metal_storage_bytes(_storage: &candle_core::MetalStorage) -> Option<u64> {
    // A Metal storage cannot be constructed without Candle's Metal feature.
    None
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "metal")]
    use super::compact_tensor_storage;
    use super::TensorStorageAccounting;
    use candle_core::{Device, IndexOp, Tensor};

    #[test]
    fn shared_views_count_their_backing_allocation_once() {
        let tensor = Tensor::from_vec(vec![0f32; 128], (8, 16), &Device::Cpu).unwrap();
        let left = tensor.i((.., ..4)).unwrap();
        let right = tensor.i((.., 12..)).unwrap();

        let mut base = TensorStorageAccounting::default();
        base.add_tensor(&tensor).unwrap();

        let mut views = TensorStorageAccounting::default();
        views.add_tensor(&left).unwrap();
        views.add_tensor(&right).unwrap();

        assert_eq!(views.bytes(), base.bytes());
        assert!(views.bytes() >= 128 * std::mem::size_of::<f32>() as u64);
    }

    #[test]
    fn independent_allocations_are_added() {
        let first = Tensor::from_vec(vec![0f32; 32], (32,), &Device::Cpu).unwrap();
        let second = Tensor::from_vec(vec![0f32; 32], (32,), &Device::Cpu).unwrap();

        let mut first_only = TensorStorageAccounting::default();
        first_only.add_tensor(&first).unwrap();

        let mut both = TensorStorageAccounting::default();
        both.add_tensor(&first).unwrap();
        both.add_tensor(&second).unwrap();

        assert!(both.bytes() >= first_only.bytes().saturating_mul(2));
    }

    #[test]
    fn byte_overflow_fails_closed() {
        let mut accounting = TensorStorageAccounting::default();
        accounting.add_bytes(u64::MAX).unwrap();
        assert!(accounting.add_bytes(1).is_none());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn persistent_metal_copy_has_exact_backing_size() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };
        let pooled = Tensor::zeros((3,), candle_core::DType::F32, &device).unwrap();
        let compact = compact_tensor_storage(&pooled).unwrap();
        let mut accounting = TensorStorageAccounting::default();
        accounting.add_tensor(&compact).unwrap();

        assert_eq!(accounting.bytes(), 3 * std::mem::size_of::<f32>() as u64);
        assert_eq!(compact.to_vec1::<f32>().unwrap(), vec![0.0; 3]);
    }
}
