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
/// Mean device milliseconds per call of `work` on `device`'s stream: two
/// raw driver events around `iterations` calls, after `warmup` untimed calls
/// (module loads, allocator growth). The context is made current and
/// the events are created, recorded, waited on and destroyed through raw
/// driver calls only: cudarc's safe event and context wrappers bind through
/// `bind_to_thread`, which returns and clears an error another component
/// latched on the context, so a measurement could fail on, or swallow,
/// someone else's error. `None` off CUDA; a driver failure is this
/// measurement's own error.
pub fn device_ms_per_call(
    device: &Device,
    warmup: usize,
    iterations: usize,
    mut work: impl FnMut() -> Result<()>,
) -> Result<Option<f64>> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::{result, sys};
        let Ok(cuda) = device.as_cuda_device() else {
            return Ok(None);
        };
        let failed = |what: &str, error: result::DriverError| {
            candle_core::Error::Msg(format!("CUDA timing {what} failed: {error:?}"))
        };
        let stream = cuda.cuda_stream();
        // SAFETY: the device keeps its context alive for this call.
        unsafe { result::ctx::set_current(stream.context().cu_ctx()) }
            .map_err(|e| failed("context", e))?;
        for _ in 0..warmup {
            work()?;
        }
        struct Events(Vec<sys::CUevent>);
        impl Drop for Events {
            fn drop(&mut self) {
                for event in &self.0 {
                    // SAFETY: created below and destroyed exactly once.
                    let _ = unsafe { result::event::destroy(*event) };
                }
            }
        }
        let mut events = Events(Vec::with_capacity(2));
        for _ in 0..2 {
            events.0.push(
                result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
                    .map_err(|e| failed("event creation", e))?,
            );
        }
        let iterations = iterations.max(1);
        // SAFETY: live events and the device's stream on the current context.
        unsafe { result::event::record(events.0[0], stream.cu_stream()) }
            .map_err(|e| failed("start record", e))?;
        for _ in 0..iterations {
            work()?;
        }
        // SAFETY: as above; the elapsed time is read after the end event
        // completed, from two timing-enabled events on one stream.
        let ms = unsafe {
            result::event::record(events.0[1], stream.cu_stream())
                .map_err(|e| failed("end record", e))?;
            result::event::synchronize(events.0[1]).map_err(|e| failed("synchronize", e))?;
            result::event::elapsed(events.0[0], events.0[1]).map_err(|e| failed("elapsed", e))?
        };
        if !ms.is_finite() || ms < 0.0 {
            candle_core::bail!("invalid CUDA event elapsed time {ms}")
        }
        Ok(Some(f64::from(ms) / iterations as f64))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (device, warmup, iterations, &mut work);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_cuda_timer_for_cpu() {
        assert!(CudaTimer::start(&Device::Cpu).unwrap().is_none());
        let mut calls = 0;
        let measured = device_ms_per_call(&Device::Cpu, 1, 3, || {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert!(measured.is_none());
        assert_eq!(calls, 0, "no work runs off CUDA");
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
    let mut calls = 0;
    let ms = device_ms_per_call(&device, 1, 4, || {
        calls += 1;
        x.sqr().map(drop)
    })
    .unwrap()
    .expect("CUDA measures");
    assert_eq!(calls, 5, "one warm-up call and four timed calls");
    assert!(ms.is_finite() && ms >= 0.0, "{ms}");
}
