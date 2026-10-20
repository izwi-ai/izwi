//! Safetensors loading that bounds Candle's Metal load staging.
//!
//! Candle's own memory-mapped backend uploads each tensor at its stored dtype
//! and casts it on the device. On Metal that leaves two costs until the next
//! device synchronization, which a loader otherwise reaches only at its load
//! fence: the released stored-dtype upload is kept in Candle's buffer pool, and
//! the kernel-written cast is rounded up to a power of two. Whisper (F16 stored,
//! F32 on Metal) peaked at 6.3 GB for 3.0 GiB of weights this way.
//!
//! [`metal_staged_safetensors`] fixes both on Metal: each tensor is converted
//! to its requested dtype on the host and uploaded once, at its exact size, and
//! the device is synchronized every [`METAL_LOAD_SYNC_INTERVAL_BYTES`] of
//! uploads, which drops every buffer the loader's own device ops (fusions,
//! casts, pre-scales) released in between. [`metal_synced_safetensors`] keeps
//! Candle's device casts and only adds the synchronizations, for loaders whose
//! host copies would cost more than the rounding. Other devices keep Candle's
//! backend. Load admission prices both forms (`runtime/lifecycle/asr_memory.rs`).
use std::path::Path;
use std::sync::Mutex;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;

/// Uploaded bytes between two device synchronizations during a staged load.
///
/// Released buffers stay in Candle's pool until a synchronization, so this,
/// plus the loader's largest single release, bounds what a load holds beyond
/// its resident weights: 256 MiB is under 2% of a 16 GB
/// Mac. A synchronization only waits for the queued casts and concatenations
/// (uploads are host copies into shared buffers), so the 24 a 6 GB checkpoint
/// pays cost well under a second, while a smaller interval would trade
/// little memory for more stalls of the queue.
pub const METAL_LOAD_SYNC_INTERVAL_BYTES: u64 = 256 * 1024 * 1024;

/// Counts uploaded bytes and reports when a synchronization is due.
#[derive(Debug)]
pub struct UploadSyncCounter {
    interval: u64,
    pending: Mutex<u64>,
}

impl UploadSyncCounter {
    pub fn new(interval: u64) -> Self {
        Self {
            interval: interval.max(1),
            pending: Mutex::new(0),
        }
    }

    /// Records `bytes` uploaded. Returns `true` when the uploads since the last
    /// synchronization reached the interval; the count then restarts, so the
    /// caller must synchronize.
    pub fn record(&self, bytes: u64) -> bool {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *pending = pending.saturating_add(bytes);
        if *pending >= self.interval {
            *pending = 0;
            true
        } else {
            false
        }
    }
}

/// Synchronizes the device by uploaded bytes, converting each tensor on the
/// host and uploading it at its exact size ([`metal_staged_safetensors`]), or
/// uploading it at its stored dtype and casting on the device as Candle does
/// ([`metal_synced_safetensors`]).
struct HostCastSafetensors {
    inner: MmapedSafetensors,
    counter: UploadSyncCounter,
    host_cast: bool,
}

impl HostCastSafetensors {
    fn load(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let (tensor, bytes) = if self.host_cast {
            let host = self.inner.load(name, &Device::Cpu)?.to_dtype(dtype)?;
            let bytes = (host.elem_count() as u64).saturating_mul(dtype.size_in_bytes() as u64);
            (host.to_device(device)?, bytes)
        } else {
            let upload = self.inner.load(name, device)?;
            let bytes =
                (upload.elem_count() as u64).saturating_mul(upload.dtype().size_in_bytes() as u64);
            (upload.to_dtype(dtype)?, bytes)
        };
        if self.counter.record(bytes) {
            device.synchronize()?;
        }
        Ok(tensor)
    }
}

impl SimpleBackend for HostCastSafetensors {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load(name, dtype, device)?;
        if tensor.shape() != &shape {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.load(name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

/// A `VarBuilder` over memory-mapped safetensors `paths`, staged as described
/// in the module docs on Metal and Candle's own backend elsewhere.
///
/// # Safety
///
/// As [`VarBuilder::from_mmaped_safetensors`]: the files must not be modified
/// while mapped.
pub unsafe fn metal_staged_safetensors<P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    if !device.is_metal() {
        return VarBuilder::from_mmaped_safetensors(paths, dtype, device);
    }
    host_cast_safetensors(paths, dtype, device, METAL_LOAD_SYNC_INTERVAL_BYTES, true)
}

/// A `VarBuilder` over memory-mapped safetensors `paths` that keeps Candle's
/// own reads (each tensor uploaded straight from the mapping at its stored
/// dtype and cast on the device, so no host copy is made) but, on Metal,
/// synchronizes the device every [`METAL_LOAD_SYNC_INTERVAL_BYTES`] of
/// uploads, so released uploads and casts never pile up until the load
/// fence. Large host copies stay in the process's footprint after they are
/// freed (0.7 GiB after loading the dense forced aligner staged), which costs
/// more than the power-of-two rounding of the device casts this keeps.
///
/// # Safety
///
/// As [`VarBuilder::from_mmaped_safetensors`].
pub unsafe fn metal_synced_safetensors<P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    if !device.is_metal() {
        return VarBuilder::from_mmaped_safetensors(paths, dtype, device);
    }
    host_cast_safetensors(paths, dtype, device, METAL_LOAD_SYNC_INTERVAL_BYTES, false)
}

unsafe fn host_cast_safetensors<P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
    interval: u64,
    host_cast: bool,
) -> candle_core::Result<VarBuilder<'static>> {
    let backend = HostCastSafetensors {
        inner: MmapedSafetensors::multi(paths)?,
        counter: UploadSyncCounter::new(interval),
        host_cast,
    };
    Ok(VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_requests_a_synchronization_each_time_uploads_reach_the_interval() {
        let counter = UploadSyncCounter::new(100);
        assert!(!counter.record(40));
        assert!(!counter.record(59));
        assert!(counter.record(1), "99 + 1 reaches the interval");
        assert!(!counter.record(99), "the count restarts after a request");
        assert!(counter.record(250), "one large upload requests at once");
        assert!(!counter.record(0));
        assert!(
            UploadSyncCounter::new(0).record(1),
            "a zero interval is one byte"
        );
    }

    #[test]
    fn host_cast_backend_matches_candles_backend() {
        let dir = std::env::temp_dir().join(format!(
            "izwi-staged-safetensors-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("model.safetensors");
        let weight = Tensor::arange(0f32, 24.0, &Device::Cpu)
            .and_then(|t| t.reshape((4, 6)))
            .and_then(|t| (t * 0.37)?.to_dtype(DType::BF16))
            .expect("weight");
        let bias = Tensor::ones(6, DType::F16, &Device::Cpu).expect("bias");
        candle_core::safetensors::save(
            &[("layer.weight", weight), ("layer.bias", bias)]
                .into_iter()
                .map(|(name, tensor)| (name.to_string(), tensor))
                .collect(),
            &path,
        )
        .expect("save");

        let staged = unsafe { host_cast_safetensors(&[&path], DType::F32, &Device::Cpu, 1, true) }
            .expect("staged");
        let synced = unsafe { host_cast_safetensors(&[&path], DType::F32, &Device::Cpu, 1, false) }
            .expect("synced");
        let candle =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu) }
                .expect("candle");
        for (name, shape) in [("layer.weight", (4, 6).into()), ("layer.bias", 6.into())] {
            let shape: Shape = shape;
            let right = candle.get(shape.clone(), name).expect("candle tensor");
            for vb in [&staged, &synced] {
                let left = vb.get(shape.clone(), name).expect("tensor");
                assert_eq!(left.dtype(), DType::F32);
                assert_eq!(
                    left.flatten_all()
                        .and_then(|t| t.to_vec1::<f32>())
                        .expect("left"),
                    right
                        .flatten_all()
                        .and_then(|t| t.to_vec1::<f32>())
                        .expect("right"),
                );
            }
        }
        let f16 = staged
            .to_dtype(DType::F16)
            .get((4, 6), "layer.weight")
            .expect("f16 tensor");
        assert_eq!(f16.dtype(), DType::F16);
        assert!(
            staged.get((6, 4), "layer.weight").is_err(),
            "shape is checked"
        );
        assert!(staged.contains_tensor("layer.bias"));
        assert!(!staged.contains_tensor("layer.missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A staged load casts on the host what Candle casts on the Metal
    /// device; both round BF16 to F16 the same way for every bit pattern
    /// (NaNs stay NaN), so a staged F16 load holds the same weights.
    #[cfg(feature = "metal")]
    #[test]
    fn host_and_metal_bf16_to_f16_casts_agree() {
        let Some((metal, _lease)) =
            crate::backends::metal_test_device_or_skip("host_and_metal_bf16_to_f16_casts_agree")
        else {
            return;
        };
        let patterns = (0..=u16::MAX)
            .map(half::bf16::from_bits)
            .collect::<Vec<_>>();
        let source = Tensor::from_vec(patterns, u16::MAX as usize + 1, &Device::Cpu).expect("bf16");
        let host = source
            .to_dtype(DType::F16)
            .and_then(|t| t.to_vec1::<half::f16>())
            .expect("host cast");
        let device = source
            .to_device(&metal)
            .and_then(|t| t.to_dtype(DType::F16))
            .and_then(|t| t.to_vec1::<half::f16>())
            .expect("metal cast");
        for (index, (host, device)) in host.iter().zip(&device).enumerate() {
            assert!(
                host.to_bits() == device.to_bits() || (host.is_nan() && device.is_nan()),
                "bf16 {index:#06x}: host {:#06x}, metal {:#06x}",
                host.to_bits(),
                device.to_bits()
            );
        }
    }
}
