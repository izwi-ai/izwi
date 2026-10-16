//! One owned OS thread per compute device.
//!
//! Candle's Metal device shares command-buffer state between the threads that
//! use it: two threads driving one device read back wrong tokens. CUDA work
//! runs on the calling thread's default stream, and cudarc binds the context
//! of a freed buffer through event waits on whatever thread drops it. Device
//! work spread over tokio workers, load tasks and blocking threads is therefore
//! wrong on Metal and fragile on CUDA. Every device operation (model execution,
//! KV page zero/copy, offload capture/restore, load, warm-up and unload drops)
//! runs as a job on its device's thread instead.
//!
//! The thread for a device lives as long as the process: two threads must
//! never drive one device, not even while a retiring thread drains its queue
//! beside its replacement. [`DeviceThread::for_location`] returns it, started
//! on first use.
//!
//! [`probe_device_op`] marks device-op entry points. A call that is not on a
//! device thread counts in [`device_ops_off_thread_total`]: the deterministic
//! check that replaces a racy corruption test.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use candle_core::{Device, DeviceLocation};
use tokio::sync::oneshot;
use tracing::debug;

use crate::error::{Error, Result};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// The device a [`DeviceThread`] owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceThreadKey {
    Cpu,
    Metal(usize),
    Cuda(usize),
}

impl DeviceThreadKey {
    pub fn from_location(location: DeviceLocation) -> Self {
        match location {
            DeviceLocation::Cpu => Self::Cpu,
            DeviceLocation::Metal { gpu_id } => Self::Metal(gpu_id),
            DeviceLocation::Cuda { gpu_id } => Self::Cuda(gpu_id),
        }
    }

    /// The metric label of this device: `cpu`, `metal:0`, `cuda:1`.
    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Metal(ordinal) => format!("metal:{ordinal}"),
            Self::Cuda(ordinal) => format!("cuda:{ordinal}"),
        }
    }

    fn thread_name(self) -> String {
        match self {
            Self::Cpu => "izwi-device-cpu-0".to_string(),
            Self::Metal(ordinal) => format!("izwi-device-metal-{ordinal}"),
            Self::Cuda(ordinal) => format!("izwi-device-cuda-{ordinal}"),
        }
    }
}

/// An owned OS thread that runs every job for one device, in order.
pub struct DeviceThread {
    key: DeviceThreadKey,
    id: u64,
    queue: Mutex<Option<mpsc::Sender<Job>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

static NEXT_DEVICE_THREAD_ID: AtomicU64 = AtomicU64::new(1);
static DEVICE_OPS_OFF_THREAD: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The id of the device thread (or a scope delegated by one) this thread
    /// is running device work for.
    static CURRENT_DEVICE_THREAD: Cell<Option<u64>> = const { Cell::new(None) };
    /// Test-only sink that attributes off-thread device ops to one test.
    static OFF_THREAD_WATCH: RefCell<Option<Arc<AtomicU64>>> = const { RefCell::new(None) };
}

fn registry() -> &'static Mutex<HashMap<DeviceThreadKey, Arc<DeviceThread>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<DeviceThreadKey, Arc<DeviceThread>>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

impl DeviceThread {
    /// The thread for `location`'s device, started on first use.
    pub fn for_location(location: DeviceLocation) -> Result<Arc<Self>> {
        let key = DeviceThreadKey::from_location(location);
        let mut threads = registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(thread) = threads.get(&key) {
            return Ok(thread.clone());
        }
        let thread = Arc::new(Self::spawn(key)?);
        threads.insert(key, thread.clone());
        Ok(thread)
    }

    /// The thread for `device`, started on first use.
    pub fn for_device(device: &Device) -> Result<Arc<Self>> {
        Self::for_location(device.location())
    }

    fn spawn(key: DeviceThreadKey) -> Result<Self> {
        let id = NEXT_DEVICE_THREAD_ID.fetch_add(1, Ordering::Relaxed);
        let (queue, jobs) = mpsc::channel::<Job>();
        let handle = std::thread::Builder::new()
            .name(key.thread_name())
            .spawn(move || {
                CURRENT_DEVICE_THREAD.with(|current| current.set(Some(id)));
                // Jobs contain their own panics; this only guards the loop.
                while let Ok(job) = jobs.recv() {
                    job();
                }
            })
            .map_err(|error| {
                Error::InferenceError(format!(
                    "failed to start the {key:?} device thread: {error}"
                ))
            })?;
        Ok(Self {
            key,
            id,
            queue: Mutex::new(Some(queue)),
            handle: Mutex::new(Some(handle)),
        })
    }

    pub fn key(&self) -> DeviceThreadKey {
        self.key
    }

    /// Whether the calling thread runs device work for this thread.
    pub fn is_current(&self) -> bool {
        CURRENT_DEVICE_THREAD.with(Cell::get) == Some(self.id)
    }

    fn enqueue(&self, job: Job) -> Result<()> {
        let queue = self
            .queue
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let stopped = || Error::InferenceError(format!("the {:?} device thread stopped", self.key));
        queue
            .as_ref()
            .ok_or_else(stopped)?
            .send(job)
            .map_err(|_| stopped())
    }

    /// Run `job` on the device thread and resolve with its result. A panic in
    /// the job is contained and returned as an error; the thread keeps
    /// serving later jobs.
    pub fn submit<T, F>(&self, job: F) -> impl Future<Output = Result<T>> + Send + 'static
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (done, result) = oneshot::channel();
        let enqueued = self.enqueue(Box::new(move || {
            let _ = done.send(catch_unwind(AssertUnwindSafe(job)));
        }));
        let key = self.key;
        async move {
            enqueued?;
            match result.await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(payload)) => Err(job_panicked(key, payload.as_ref())),
                Err(_) => Err(Error::InferenceError(format!(
                    "the {key:?} device thread stopped before running a job"
                ))),
            }
        }
    }

    /// Run `job` on the device thread and block the caller until it returns.
    /// Runs inline when the caller already is the device thread. Only call it
    /// from a blocking context or for short leaf operations (KV page ops);
    /// async code awaits [`Self::submit`] instead.
    pub fn run_sync<'env, T, F>(&self, job: F) -> Result<T>
    where
        T: Send + 'env,
        F: FnOnce() -> T + Send + 'env,
    {
        if self.is_current() {
            return catch_unwind(AssertUnwindSafe(job))
                .map_err(|payload| job_panicked(self.key, payload.as_ref()));
        }
        let (done, result) = mpsc::sync_channel(1);
        let scoped = ScopedJob {
            job: Some(job),
            done,
        };
        let boxed: Box<dyn FnOnce() + Send + 'env> = Box::new(move || scoped.run());
        // SAFETY: the job may borrow from the caller's stack ('env). This
        // function does not return before the device thread has finished
        // with it: `result.recv()` returns only once `done` has sent the
        // result or has been dropped, and `ScopedJob` drops `job` before
        // `done` (field order). A job that is never run (the thread stopped)
        // is dropped with its queue, which also drops `job` before `done`, or
        // is handed back by `enqueue`'s error and dropped here.
        let boxed: Job =
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'env>, Job>(boxed) };
        self.enqueue(boxed)?;
        match result.recv() {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(payload)) => Err(job_panicked(self.key, payload.as_ref())),
            Err(_) => Err(Error::InferenceError(format!(
                "the {:?} device thread stopped before running a job",
                self.key
            ))),
        }
    }

    /// Stop accepting jobs, let queued jobs finish and join the thread.
    /// Idempotent; a no-op join when called from the device thread itself.
    pub fn shutdown(&self) {
        self.queue
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        let handle = self
            .handle
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if let Some(handle) = handle {
            if !self.is_current() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for DeviceThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for DeviceThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceThread")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// A borrowed job for [`DeviceThread::run_sync`]. Field order is load
/// bearing: `job` (which may borrow the caller's stack) drops before `done`
/// releases the waiting caller.
struct ScopedJob<F, T> {
    job: Option<F>,
    done: mpsc::SyncSender<std::thread::Result<T>>,
}

impl<F: FnOnce() -> T, T> ScopedJob<F, T> {
    fn run(mut self) {
        let job = self.job.take().expect("a scoped device job runs once");
        let result = catch_unwind(AssertUnwindSafe(job));
        let _ = self.done.send(result);
    }
}

fn job_panicked(key: DeviceThreadKey, payload: &(dyn std::any::Any + Send)) -> Error {
    let message = payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string());
    Error::InferenceError(format!("{key:?} device job panicked: {message}"))
}

/// Proof that the current thread runs a device job, for handing that job's
/// device context to helper threads it joins before returning (the CPU
/// request-parallel fan-out).
#[derive(Debug, Clone, Copy)]
pub struct DeviceScope(u64);

impl DeviceScope {
    /// The scope of the device job running on this thread, if any.
    pub fn current() -> Option<Self> {
        CURRENT_DEVICE_THREAD.with(Cell::get).map(Self)
    }

    /// Run device work for this scope on the calling thread until the guard
    /// drops. The caller must join this thread before the owning job returns.
    pub fn enter(self) -> DeviceScopeGuard {
        let previous = CURRENT_DEVICE_THREAD.with(|current| current.replace(Some(self.0)));
        DeviceScopeGuard { previous }
    }
}

pub struct DeviceScopeGuard {
    previous: Option<u64>,
}

impl Drop for DeviceScopeGuard {
    fn drop(&mut self) {
        CURRENT_DEVICE_THREAD.with(|current| current.set(self.previous));
    }
}

/// Mark a device-op entry point. A call that is not on a device thread (or a
/// scope delegated by one) is counted and logged.
pub fn probe_device_op(op: &'static str) {
    if CURRENT_DEVICE_THREAD.with(Cell::get).is_some() {
        return;
    }
    DEVICE_OPS_OFF_THREAD.fetch_add(1, Ordering::Relaxed);
    OFF_THREAD_WATCH.with(|watch| {
        if let Some(watch) = watch.borrow().as_ref() {
            watch.fetch_add(1, Ordering::Relaxed);
        }
    });
    debug!(
        op,
        thread = std::thread::current().name().unwrap_or("unnamed"),
        "Device operation ran off its device thread"
    );
}

/// Device operations that ran off a device thread since process start.
pub fn device_ops_off_thread_total() -> u64 {
    DEVICE_OPS_OFF_THREAD.load(Ordering::Relaxed)
}

/// Attribute off-thread device ops on the calling thread to `watch` (tests
/// install it on every thread of a runtime they own). `None` removes it.
#[cfg(test)]
pub(crate) fn watch_off_thread_ops(watch: Option<Arc<AtomicU64>>) {
    OFF_THREAD_WATCH.with(|slot| *slot.borrow_mut() = watch);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_thread() -> Arc<DeviceThread> {
        // A private thread: the shared CPU thread may be serving other tests.
        Arc::new(DeviceThread::spawn(DeviceThreadKey::Cpu).expect("device thread"))
    }

    #[test]
    fn run_sync_runs_on_the_device_thread_and_borrows_the_caller() {
        let thread = test_thread();
        let mut values = vec![1, 2, 3];
        let name = thread
            .run_sync(|| {
                values.push(4);
                std::thread::current().name().map(str::to_string)
            })
            .expect("job");
        assert_eq!(name.as_deref(), Some("izwi-device-cpu-0"));
        assert_eq!(values, [1, 2, 3, 4]);
        assert!(!thread.is_current());
    }

    #[test]
    fn run_sync_is_reentrant_on_the_device_thread() {
        let thread = test_thread();
        let inner = thread.clone();
        let nested = thread
            .run_sync(move || {
                assert!(inner.is_current());
                inner.run_sync(|| 7).expect("inline job")
            })
            .expect("outer job");
        assert_eq!(nested, 7);
    }

    #[tokio::test]
    async fn submit_resolves_with_the_job_result() {
        let thread = test_thread();
        let value = thread.submit(|| 40 + 2).await.expect("job");
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn a_panicking_job_is_contained_and_the_thread_keeps_serving() {
        let thread = test_thread();
        let error = thread
            .submit(|| -> usize { panic!("boom") })
            .await
            .expect_err("panic surfaces as an error");
        assert!(error.to_string().contains("boom"), "{error}");
        let error = thread
            .run_sync(|| -> usize { panic!("again") })
            .expect_err("panic surfaces as an error");
        assert!(error.to_string().contains("again"), "{error}");
        assert_eq!(thread.submit(|| 5).await.expect("later job"), 5);
    }

    #[tokio::test]
    async fn shutdown_finishes_queued_jobs_and_refuses_new_ones() {
        let thread = test_thread();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let first = thread.submit(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            1
        });
        let queued = thread.submit(|| 2);
        started_rx.recv().unwrap();
        let stopper = {
            let thread = thread.clone();
            std::thread::spawn(move || thread.shutdown())
        };
        release_tx.send(()).unwrap();
        stopper.join().unwrap();
        assert_eq!(first.await.expect("running job"), 1);
        assert_eq!(queued.await.expect("queued job"), 2);
        assert!(thread.submit(|| 3).await.is_err());
        assert!(thread.run_sync(|| 3).is_err());
    }

    #[test]
    fn threads_are_shared_per_device() {
        let first = DeviceThread::for_location(DeviceLocation::Cpu).expect("thread");
        let second = DeviceThread::for_location(DeviceLocation::Cpu).expect("thread");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.key(), DeviceThreadKey::Cpu);
    }

    #[test]
    fn the_probe_counts_only_device_ops_off_a_device_thread() {
        let thread = test_thread();
        let watch = Arc::new(AtomicU64::new(0));
        watch_off_thread_ops(Some(watch.clone()));
        thread
            .run_sync(|| probe_device_op("test.on_thread"))
            .expect("job");
        let scope = thread.run_sync(DeviceScope::current).expect("job");
        std::thread::scope(|threads| {
            threads.spawn(|| {
                let _scope = scope.expect("device scope").enter();
                probe_device_op("test.delegated");
            });
        });
        assert_eq!(watch.load(Ordering::Relaxed), 0);
        probe_device_op("test.off_thread");
        assert_eq!(watch.load(Ordering::Relaxed), 1);
        watch_off_thread_ops(None);
    }
}
