//! Zero-filled state tensors whose device buffer is exactly their size.
//!
//! Candle's Metal `Tensor::zeros` draws from its buffer pool, which rounds
//! every allocation up to the next power of two (`candle-core` 0.11
//! `metal_backend/device.rs`, `buf_size`) and zero-fills the rounded length.
//! Long-lived state (paged KV arenas, invocation state, static attention
//! memories) is leased at its exact byte count, so a pooled state tensor
//! could hold up to twice what the ledger authorized: an LFM2.5 Audio load
//! observed 1.82 GB of state growth against 1.43 GB authorized (ADM-3).
//! These tensors are allocated outside the pool at their exact size and
//! cleared with one blit; they are freed when the last reference drops.

use candle_core::{DType, Device, Shape, Tensor, Var};

use crate::error::Result;

/// A zero-filled tensor of `shape` and `dtype` on `device` whose Metal
/// buffer is exactly its size. CPU and CUDA allocations are already exact.
pub(crate) fn exact_zeros<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let shape = shape.into();
    #[cfg(feature = "metal")]
    if let Some(tensor) = metal_exact_zeros(&shape, dtype, device, false)? {
        return Ok(tensor);
    }
    Ok(Tensor::zeros(shape, dtype, device)?)
}

/// `parts` joined along dimension 0 into one tensor whose Metal buffer is
/// exactly its size. `Tensor::cat` draws its result from Candle's pool,
/// rounded up to the next power of two, so a fused projection weight built
/// with it can waste up to half its size for the model's life (Granite
/// Speech's joined Q/K/V: 9.8 MB per layer held in 16 MiB).
pub(crate) fn exact_concat_rows(parts: &[&Tensor]) -> Result<Tensor> {
    let first = parts.first().ok_or_else(|| {
        crate::error::Error::InvalidInput("cannot concatenate zero tensors".into())
    })?;
    let mut dims = first.dims().to_vec();
    if dims.is_empty() {
        return Err(crate::error::Error::InvalidInput(
            "cannot concatenate scalar tensors".into(),
        ));
    }
    let mut rows = 0;
    for part in parts {
        if part.dtype() != first.dtype() || part.dims()[1..] != dims[1..] {
            return Err(crate::error::Error::InvalidInput(format!(
                "cannot concatenate {:?} {:?} with {:?} {:?} along rows",
                first.dtype(),
                first.dims(),
                part.dtype(),
                part.dims()
            )));
        }
        rows += part.dim(0)?;
    }
    dims[0] = rows;
    let joined = exact_zeros(dims, first.dtype(), first.device())?;
    let mut offset = 0;
    for part in parts {
        joined.slice_set(&part.contiguous()?, 0, offset)?;
        offset += part.dim(0)?;
    }
    Ok(joined)
}

/// [`exact_zeros`] as a [`Var`], for state written in place.
pub(crate) fn exact_zeros_var<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Var> {
    let shape = shape.into();
    #[cfg(feature = "metal")]
    if let Some(tensor) = metal_exact_zeros(&shape, dtype, device, true)? {
        // Already a variable, so this wraps it without a copy.
        return Ok(Var::from_tensor(&tensor)?);
    }
    Ok(Var::zeros(shape, dtype, device)?)
}

/// A private Metal buffer of exactly `count` elements of `dtype`, freed when
/// its last reference drops.
///
/// Candle's `MetalDevice::new_private_buffer` adds its buffer to the device's
/// `MTLResidencySet`, which retains it, and Candle removes only pooled
/// buffers from that set (`drop_unused_buffers`), so a buffer from it is
/// never freed: an unloaded LFM2.5 kept its 0.40 GB KV arena, Granite Speech
/// 2.85 GB of state and fused weights, and every Qwen3.5 prefill its copied
/// recurrent state. Taken from the raw device, the buffer lives exactly as
/// long as its references: Metal makes a buffer bound to an encoder resident
/// for the command buffer that uses it, and a command buffer retains the
/// buffers it references until it completes.
#[cfg(feature = "metal")]
pub(crate) fn exact_private_buffer(
    metal: &candle_core::MetalDevice,
    count: usize,
    dtype: DType,
    label: &str,
) -> candle_core::Result<std::sync::Arc<candle_metal_kernels::metal::Buffer>> {
    use candle_metal_kernels::metal::MTLResourceOptions;

    let bytes = count
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| candle_core::Error::Msg("exact Metal buffer size overflow".into()))?;
    // Candle's `PRIVATE_RESOURCE_OPTIONS`, which it does not export.
    let buffer = metal
        .device()
        .new_buffer(
            bytes,
            MTLResourceOptions::StorageModePrivate
                | MTLResourceOptions::HazardTrackingModeUntracked,
        )
        .map_err(candle_core::MetalError::from)?;
    buffer.set_label(label);
    Ok(std::sync::Arc::new(buffer))
}

#[cfg(feature = "metal")]
fn metal_exact_zeros(
    shape: &Shape,
    dtype: DType,
    device: &Device,
    is_variable: bool,
) -> Result<Option<Tensor>> {
    use candle_core::op::BackpropOp;
    use candle_core::{MetalStorage, Storage};

    let Device::Metal(metal) = device else {
        return Ok(None);
    };
    let count = shape.elem_count();
    if count == 0 {
        return Ok(None);
    }
    let bytes = count.checked_mul(dtype.size_in_bytes()).ok_or_else(|| {
        crate::error::Error::Overloaded("exact state tensor size overflow".into())
    })?;
    let buffer = exact_private_buffer(metal, count, dtype, "izwi_state")?;
    {
        let mut blit = metal.blit_command_encoder()?;
        blit.set_label("izwi_state_zeros");
        blit.fill_buffer(&buffer, (0, bytes), 0);
    }
    Ok(Some(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, metal.clone(), count, dtype)),
        shape.clone(),
        BackpropOp::none(),
        is_variable,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_concat_rows_equals_cat() {
        let device = Device::Cpu;
        let a = Tensor::arange(0f32, 6.0, &device)
            .unwrap()
            .reshape((2, 3))
            .unwrap();
        let b = Tensor::arange(6f32, 15.0, &device)
            .unwrap()
            .reshape((3, 3))
            .unwrap();
        let joined = exact_concat_rows(&[&a, &b]).unwrap();
        let expected = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(
            joined.to_vec2::<f32>().unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
        assert!(exact_concat_rows(&[&a, &a.t().unwrap()]).is_err());
    }

    #[test]
    fn exact_zeros_is_zeroed_with_the_requested_shape() {
        let tensor = exact_zeros((2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[2, 3, 4]);
        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| *value == 0.0));
        let var = exact_zeros_var((4, 2), DType::F16, &Device::Cpu).unwrap();
        assert_eq!(var.as_tensor().dims(), &[4, 2]);
        assert!(var.as_tensor().is_variable());
    }

    /// The LFM2.5 Audio retained KV arena, from the published checkpoint's
    /// geometry (6 attention layers of 8 KV heads × 64 dims, F32 on Metal,
    /// 16-token pages): one K and one V tensor per layer. Leased at its exact
    /// size, an arena of 1,100 pages held 1.86× that through Candle's pooled
    /// power-of-two allocation, past the reconciliation tolerance; exact
    /// allocation holds what the lease authorized.
    #[test]
    fn pooled_arena_allocation_exceeds_its_exact_lease() {
        const ATTENTION_LAYERS: u64 = 6;
        const PAGE_TOKENS: u64 = 16;
        const KV_HEADS: u64 = 8;
        const HEAD_DIM: u64 = 64;
        let pages = 1_100;
        let tensor_bytes = pages * PAGE_TOKENS * KV_HEADS * HEAD_DIM * 4;
        let lease = 2 * ATTENTION_LAYERS * tensor_bytes;
        let pooled = 2 * ATTENTION_LAYERS * tensor_bytes.next_power_of_two();
        let tolerance = (lease / 8).max(256 * 1024 * 1024);
        assert!(
            pooled > lease + tolerance,
            "pooled {pooled} vs lease {lease}"
        );
        let exact = exact_zeros(
            (
                pages as usize,
                PAGE_TOKENS as usize,
                KV_HEADS as usize,
                HEAD_DIM as usize,
            ),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            2 * ATTENTION_LAYERS * (exact.elem_count() * exact.dtype().size_in_bytes()) as u64,
            lease
        );
    }

    /// Exact state and its compact copies return their memory to the device
    /// once the last reference drops; registered in Candle's residency set,
    /// they stayed allocated after an unload for the life of the process.
    #[cfg(feature = "metal")]
    #[test]
    fn exact_metal_buffers_are_freed_when_their_last_reference_drops() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "exact_metal_buffers_are_freed_when_their_last_reference_drops",
        ) else {
            return;
        };
        let metal = device.as_metal_device().unwrap().clone();
        device.synchronize().unwrap();
        let before = metal.device().current_allocated_size();
        let state = exact_zeros_var((16, 1024, 1024), DType::F32, &device).unwrap();
        let compact = crate::models::shared::memory::accounting::deep_copy_tensor_storage(
            &state.as_tensor().narrow(0, 0, 4).unwrap(),
        )
        .unwrap();
        device.synchronize().unwrap();
        assert!(metal.device().current_allocated_size() >= before + (80 << 20));
        drop((state, compact));
        device.synchronize().unwrap();
        assert!(metal.device().current_allocated_size() <= before + (1 << 20));
    }
}
