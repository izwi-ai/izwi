//! Delayed CUDA stream timing without a host/device synchronization. Query the
//! pending interval after the next normal sampling readback; incomplete events
//! return None. Timings include all stream work between the two record points.
use candle_core::{Device, Result};
use std::time::Duration;

pub struct CudaTimer {
    #[cfg(feature = "cuda")]
    inner: Pair,
}
pub struct PendingCudaTimer {
    #[cfg(feature = "cuda")]
    inner: Pair,
}
#[cfg(feature = "cuda")]
struct Pair {
    start: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    end: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
}
impl CudaTimer {
    pub fn start(device: &Device) -> Result<Option<Self>> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::{
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT, WrapErr,
            };
            let Ok(device) = device.as_cuda_device() else {
                return Ok(None);
            };
            let stream = device.cuda_stream();
            let start = stream.context().new_event(Some(CU_EVENT_DEFAULT)).w()?;
            let end = stream.context().new_event(Some(CU_EVENT_DEFAULT)).w()?;
            start.record(&stream).w()?;
            Ok(Some(Self {
                inner: Pair { start, end, stream },
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            Ok(None)
        }
    }
    pub fn finish(self) -> Result<PendingCudaTimer> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::WrapErr;
            self.inner.end.record(&self.inner.stream).w()?;
            Ok(PendingCudaTimer { inner: self.inner })
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::bail!("CUDA timer requires a CUDA build")
        }
    }
}
impl PendingCudaTimer {
    pub fn try_elapsed(&self) -> Result<Option<Duration>> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::{
                cudarc::driver::{result, sys::cudaError_enum::CUDA_ERROR_NOT_READY},
                WrapErr,
            };
            self.inner.stream.context().bind_to_thread().w()?;
            // SAFETY: owned timing-enabled events on the same retained context. A
            // successful end query proves both record points completed. Use the raw
            // elapsed call because cudarc's safe elapsed_ms synchronizes internally.
            match unsafe { result::event::query(self.inner.end.cu_event()) } {
                Ok(()) => {}
                Err(e) if e.0 == CUDA_ERROR_NOT_READY => return Ok(None),
                Err(e) => return Err(e).w(),
            }
            let ms = unsafe {
                result::event::elapsed(self.inner.start.cu_event(), self.inner.end.cu_event())
            }
            .w()?;
            if !ms.is_finite() || ms < 0. {
                candle_core::bail!("invalid CUDA event elapsed time")
            }
            Ok(Some(Duration::from_secs_f64(ms as f64 / 1000.)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(None)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_cuda_timer_for_cpu() {
        assert!(CudaTimer::start(&Device::Cpu).unwrap().is_none());
    }
    #[test]
    fn pending_timing_can_be_owned_by_a_request_across_workers() {
        fn check<T: Send + Sync>() {}
        check::<CudaTimer>();
        check::<PendingCudaTimer>();
    }
}

#[cfg(all(test, feature = "cuda"))]
#[test]
fn cuda_timer_reads_completed_interval_without_adding_a_fence() {
    let Some(device) = super::cuda_test_device() else {
        return;
    };
    let timer = CudaTimer::start(&device).unwrap().unwrap();
    let x = candle_core::Tensor::ones((32, 128), candle_core::DType::F32, &device).unwrap();
    let y = x.sqr().unwrap();
    let pending = timer.finish().unwrap();
    // This is the normal consumer readback; try_elapsed itself never fences.
    let _ = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(pending.try_elapsed().unwrap().is_some());
}
