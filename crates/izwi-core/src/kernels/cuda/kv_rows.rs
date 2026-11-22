//! Paged KV writes on CUDA: each source row copied into the destination row
//! its slot names, in place and in one launch.
//!
//! Candle's `scatter_set` needed an index as large as the written K/V (a u32
//! per element, materialized from a broadcast) and its kernel runs one
//! thread per column, looping over the rows serially: a prefill write of
//! hundreds of rows took about 40 µs a layer on the H100. This kernel copies
//! every row in parallel in the widest unit (16, 4, 2 or 1 bytes) that
//! divides the row and both offsets, so any storage dtype (FP8 included)
//! takes it. The CPU runs the same copy as a test reference.

use candle_core::{bail, CpuStorage, InplaceOp3, Layout, Result, Tensor};

/// Copy `source` rows into the `destination` rows named by `slots` (u32),
/// in place; a slot of [`SKIP_SLOT`] writes nothing. Shapes beyond the
/// first dimension must agree; callers must have bounds-checked every other
/// slot against the destination rows.
pub fn write_kv_rows(destination: &Tensor, source: &Tensor, slots: &Tensor) -> Result<()> {
    if destination.rank() == 0
        || source.rank() != destination.rank()
        || source.dims()[1..] != destination.dims()[1..]
        || source.dim(0)? != slots.elem_count()
        || source.dtype() != destination.dtype()
        || slots.dtype() != candle_core::DType::U32
    {
        bail!(
            "izwi-kv-write-rows: destination {:?} {:?}, source {:?} {:?}, {} {:?} slots",
            destination.dims(),
            destination.dtype(),
            source.dims(),
            source.dtype(),
            slots.elem_count(),
            slots.dtype()
        )
    }
    if slots.elem_count() == 0 {
        return Ok(());
    }
    destination.inplace_op3(&source.contiguous()?, &slots.contiguous()?, &KvWriteRows)
}

/// The slot that writes nothing (scatter's convention for an idle row).
pub const SKIP_SLOT: u32 = u32::MAX;

struct KvWriteRows;

/// `(rows, row_elements)` of a write, validated against the destination.
fn geometry(destination: &Layout, source: &Layout, slots: &Layout) -> Result<(usize, usize)> {
    if !destination.is_contiguous() || !source.is_contiguous() || !slots.is_contiguous() {
        bail!("izwi-kv-write-rows requires contiguous tensors")
    }
    let rows = slots.shape().elem_count();
    let elements = source.shape().elem_count();
    if rows == 0 || !elements.is_multiple_of(rows) {
        bail!("izwi-kv-write-rows source does not split into slot rows")
    }
    let row_elements = elements / rows;
    if row_elements == 0
        || !destination
            .shape()
            .elem_count()
            .is_multiple_of(row_elements)
    {
        bail!("izwi-kv-write-rows destination rows do not match the source")
    }
    Ok((rows, row_elements))
}

impl InplaceOp3 for KvWriteRows {
    fn name(&self) -> &'static str {
        "izwi-kv-write-rows"
    }

    fn cpu_fwd(
        &self,
        destination: &mut CpuStorage,
        destination_layout: &Layout,
        source: &CpuStorage,
        source_layout: &Layout,
        slots: &CpuStorage,
        slots_layout: &Layout,
    ) -> Result<()> {
        let (rows, width) = geometry(destination_layout, source_layout, slots_layout)?;
        let CpuStorage::U32(slots) = slots else {
            bail!("izwi-kv-write-rows takes u32 slots")
        };
        let slots = &slots[slots_layout.start_offset()..slots_layout.start_offset() + rows];
        let capacity = destination_layout.shape().elem_count() / width;
        if let Some(slot) = slots
            .iter()
            .find(|slot| **slot != SKIP_SLOT && **slot as usize >= capacity)
        {
            bail!("izwi-kv-write-rows slot {slot} is outside {capacity} rows")
        }
        macro_rules! copy {
            ($($variant:ident),*) => {
                match (destination, source) {
                    $((CpuStorage::$variant(d), CpuStorage::$variant(s)) => {
                        let d = &mut d[destination_layout.start_offset()..];
                        let s = &s[source_layout.start_offset()..];
                        for (row, slot) in slots.iter().enumerate() {
                            if *slot == SKIP_SLOT {
                                continue;
                            }
                            let at = *slot as usize * width;
                            d[at..at + width].copy_from_slice(&s[row * width..(row + 1) * width]);
                        }
                    })*
                    _ => bail!("izwi-kv-write-rows: unsupported or mismatched dtypes"),
                }
            };
        }
        copy!(U8, U32, I64, BF16, F16, F32, F64, F8E4M3);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        destination: &mut candle_core::CudaStorage,
        destination_layout: &Layout,
        source: &candle_core::CudaStorage,
        source_layout: &Layout,
        slots: &candle_core::CudaStorage,
        slots_layout: &Layout,
    ) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{
            DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
        let (rows, row_elements) = geometry(destination_layout, source_layout, slots_layout)?;
        let element = destination.dtype().size_in_bytes();
        if element == 0 || source.dtype() != destination.dtype() {
            bail!("izwi-kv-write-rows requires matching byte-sized dtypes")
        }
        let row_bytes = row_elements * element;
        let destination_offset = destination_layout.start_offset() * element;
        let source_offset = source_layout.start_offset() * element;
        let (unit, function) = [
            (16, "izwi_kv_write_rows_u128"),
            (4, "izwi_kv_write_rows_u32"),
            (2, "izwi_kv_write_rows_u16"),
            (1, "izwi_kv_write_rows_u8"),
        ]
        .into_iter()
        .find(|(unit, _)| {
            [row_bytes, destination_offset, source_offset]
                .iter()
                .all(|bytes| bytes.is_multiple_of(*unit))
        })
        .expect("a one-byte unit divides everything");
        let units = (row_bytes / unit) as u64;
        let device = destination.device.clone();
        let stream = device.cuda_stream();
        let CudaStorageSlice::U32(slot_slice) = &slots.slice else {
            bail!("izwi-kv-write-rows takes u32 slots")
        };
        let slot_view = slot_slice.slice(slots_layout.start_offset()..);
        macro_rules! pointers {
            ($($variant:ident),*) => {
                match (&mut destination.slice, &source.slice) {
                    $((CudaStorageSlice::$variant(d), CudaStorageSlice::$variant(s)) => {
                        let (d, dg) = d.device_ptr_mut(&stream);
                        let (s, sg) = s.device_ptr(&stream);
                        (d, s, (dg, sg))
                    })*
                    _ => bail!("izwi-kv-write-rows: unsupported or mismatched dtypes"),
                }
            };
        }
        let (destination_ptr, source_ptr, _guards) =
            pointers!(U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3);
        let destination_ptr = destination_ptr + destination_offset as u64;
        let source_ptr = source_ptr + source_offset as u64;
        let rows = rows as u64;
        let total = rows * units;
        let blocks = total.div_ceil(256).clamp(1, 8192) as u32;
        let func = device.get_or_load_custom_func(
            function,
            "izwi_physical_state",
            super::cuda_ptx::PHYSICAL_STATE,
        )?;
        let mut builder = func.builder();
        builder.arg(&destination_ptr);
        builder.arg(&source_ptr);
        builder.arg(&slot_view);
        builder.arg(&rows);
        builder.arg(&units);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments follow `izwi_kv_write_rows_*`; both pointers
        // are offset to their views and aligned to the copy unit, the source
        // holds `rows * units` units, and the caller bounds-checked every slot
        // against the destination's rows.
        unsafe { builder.launch(config) }.w()?;
        Ok(())
    }
}

/// Overwrite every element of the contiguous u32 `tensor` with `values`, in
/// place: on CUDA one host-to-device copy into the existing buffer (a diff
/// scattered through uploaded indices and values cost two uploads and a
/// kernel). Candle routes the copy through its graph-capture cache, as it
/// does for any upload.
pub fn overwrite_u32(tensor: &Tensor, values: &[u32]) -> Result<()> {
    if tensor.dtype() != candle_core::DType::U32 || tensor.elem_count() != values.len() {
        bail!(
            "izwi-overwrite-u32: {:?} {:?} tensor, {} values",
            tensor.dims(),
            tensor.dtype(),
            values.len()
        )
    }
    tensor.inplace_op1(&OverwriteU32 {
        values: values.to_vec(),
    })
}

struct OverwriteU32 {
    values: Vec<u32>,
}

impl candle_core::InplaceOp1 for OverwriteU32 {
    fn name(&self) -> &'static str {
        "izwi-overwrite-u32"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()> {
        let CpuStorage::U32(data) = storage else {
            bail!("izwi-overwrite-u32 takes a u32 tensor")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            bail!("izwi-overwrite-u32 takes a contiguous tensor")
        };
        data[start..end].copy_from_slice(&self.values);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, storage: &mut candle_core::CudaStorage, layout: &Layout) -> Result<()> {
        use candle_core::cuda_backend::CudaStorageSlice;
        let Some((start, end)) = layout.contiguous_offsets() else {
            bail!("izwi-overwrite-u32 takes a contiguous tensor")
        };
        let device = storage.device.clone();
        let CudaStorageSlice::U32(data) = &mut storage.slice else {
            bail!("izwi-overwrite-u32 takes a u32 tensor")
        };
        let mut view = data.slice_mut(start..end);
        device.memcpy_htod(self.values.as_slice(), &mut view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn check(device: &Device, dtype: DType) {
        let destination = Tensor::zeros((6, 2, 4), DType::F32, device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let source = Tensor::arange(1f32, 25., device)
            .unwrap()
            .reshape((3, 2, 4))
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let slots = Tensor::new(&[4u32, SKIP_SLOT, 0, 2], device).unwrap();
        let source = Tensor::cat(&[source.narrow(0, 0, 1).unwrap(), source.clone()], 0).unwrap();
        write_kv_rows(&destination, &source, &slots).unwrap();
        let got = destination
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let source = source
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        let zero = vec![vec![0f32; 4]; 2];
        // Row 1 is skipped; rows 0, 2 and 3 land in slots 4, 0 and 2.
        let want = [&source[2], &zero, &source[3], &zero, &source[0], &zero];
        for (row, (a, b)) in got.iter().zip(want).enumerate() {
            assert_eq!(a, b, "row {row} ({dtype:?})");
        }
    }

    #[test]
    fn rows_land_in_their_slots_and_nothing_else_changes() {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            check(&Device::Cpu, dtype);
        }
    }

    #[test]
    fn overwrite_replaces_every_element_in_place() {
        let tensor = Tensor::new(&[1u32, 2, 3, 4], &Device::Cpu).unwrap();
        let alias = tensor.clone();
        overwrite_u32(&tensor, &[9, 8, 7, 6]).unwrap();
        assert_eq!(alias.to_vec1::<u32>().unwrap(), [9, 8, 7, 6]);
        assert!(overwrite_u32(&tensor, &[1, 2]).is_err());
    }

    #[test]
    fn a_slot_past_the_destination_is_refused() {
        let destination = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let source = Tensor::ones((1, 4), DType::F32, &Device::Cpu).unwrap();
        let slots = Tensor::new(&[2u32], &Device::Cpu).unwrap();
        assert!(write_kv_rows(&destination, &source, &slots).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rows_land_in_their_slots_at_every_copy_unit() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        // BF16 rows of 8 bytes take the u32 unit, F32 rows of 32 the u128.
        for dtype in [DType::F32, DType::BF16, DType::F16, DType::F8E4M3] {
            if dtype == DType::F8E4M3 {
                // Candle has no CUDA cast into or out of FP8: build and read
                // the tensors on the host.
                let fp8 = |t: Tensor| t.to_dtype(dtype).unwrap().to_device(&device).unwrap();
                let destination = fp8(Tensor::zeros((6, 3), DType::F32, &Device::Cpu).unwrap());
                let source = fp8(Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap());
                let slots = Tensor::new(&[5u32, 1], &device).unwrap();
                write_kv_rows(&destination, &source, &slots).unwrap();
                let got = destination
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                for (row, values) in got.iter().enumerate() {
                    let want = if row == 5 || row == 1 { 1.0 } else { 0.0 };
                    assert!(values.iter().all(|v| *v == want), "row {row}: {values:?}");
                }
            } else {
                check(&device, dtype);
            }
        }
    }
}
