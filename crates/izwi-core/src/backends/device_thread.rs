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
//! Jobs do not run in arrival order. Control jobs (KV page ops, loads,
//! frees, cleanup: short leaf operations that callers wait on, often under
//! the engine lock) run first, in order. Model dispatches carry a
//! [`DeviceJobTag`]: each execution lane is a flow, and the flow with the
//! least weighted device time served runs next, so one lane's long jobs
//! cannot hold the device while another lane's short steps queue behind
//! them. A running job is never preempted; arbitration happens between jobs.
//!
//! [`probe_device_op`] marks device-op entry points. A call that is not on a
//! device thread counts in [`device_ops_off_thread_total`]: the deterministic
//! check that replaces a racy corruption test.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use candle_core::{Device, DeviceLocation};
use tokio::sync::oneshot;
use tracing::debug;

use crate::error::{Error, Result};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// The device a [`DeviceThread`] owns: Candle's device location, so a Metal
/// device is keyed by its registry id and a CUDA device by its ordinal.
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

    /// The metric label of this device: `cpu`, `metal` (a Mac has one
    /// Metal device; its registry id is not a readable label), `cuda:1`.
    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Metal(_) => "metal".to_string(),
            Self::Cuda(ordinal) => format!("cuda:{ordinal}"),
        }
    }

    fn thread_name(self) -> String {
        match self {
            Self::Cpu => "izwi-device-cpu-0".to_string(),
            Self::Metal(registry_id) => format!("izwi-device-metal-{registry_id}"),
            Self::Cuda(ordinal) => format!("izwi-device-cuda-{ordinal}"),
        }
    }
}

/// The flow a model dispatch belongs to (its execution lane) and that
/// flow's share of device time relative to other flows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceJobTag {
    pub flow: u64,
    /// Relative share of device time; zero counts as one.
    pub weight: u64,
}

/// Weighted device time is kept in nanoseconds scaled by this factor, so
/// integer division by a weight keeps sub-nanosecond resolution.
const VIRTUAL_TIME_SCALE: u128 = 840;

struct QueuedJob {
    job: Job,
    order: u64,
    weight: u64,
}

struct Flow {
    /// Weighted device time this flow has been served.
    served: u128,
    jobs: VecDeque<QueuedJob>,
}

/// The device's job queue: control jobs first, then the backlogged flow
/// with the least weighted device time served.
#[derive(Default)]
struct Arbiter {
    control: VecDeque<Job>,
    flows: HashMap<u64, Flow>,
    /// Weighted time of the last flow that started a job. A flow that
    /// becomes backlogged starts here, so idle time is not banked as credit.
    virtual_time: u128,
    next_order: u64,
    stopped: bool,
}

impl Arbiter {
    fn push(&mut self, job: Job, tag: Option<DeviceJobTag>) {
        let Some(tag) = tag else {
            self.control.push_back(job);
            return;
        };
        let order = self.next_order;
        self.next_order += 1;
        let virtual_time = self.virtual_time;
        let flow = self.flows.entry(tag.flow).or_insert_with(|| Flow {
            served: virtual_time,
            jobs: VecDeque::new(),
        });
        if flow.jobs.is_empty() {
            flow.served = flow.served.max(virtual_time);
        }
        flow.jobs.push_back(QueuedJob {
            job,
            order,
            weight: tag.weight.max(1),
        });
    }

    /// The next job, and the flow and weight to charge for its device time.
    fn pop(&mut self) -> Option<(Job, Option<(u64, u64)>)> {
        if let Some(job) = self.control.pop_front() {
            return Some((job, None));
        }
        let virtual_time = self.virtual_time;
        // A drained flow with no debt left carries nothing a new flow at
        // `virtual_time` would not.
        self.flows
            .retain(|_, flow| !flow.jobs.is_empty() || flow.served > virtual_time);
        let (&id, flow) = self
            .flows
            .iter_mut()
            .filter(|(_, flow)| !flow.jobs.is_empty())
            .min_by_key(|(_, flow)| (flow.served, flow.jobs[0].order))?;
        self.virtual_time = self.virtual_time.max(flow.served);
        let queued = flow
            .jobs
            .pop_front()
            .expect("only backlogged flows compete");
        Some((queued.job, Some((id, queued.weight))))
    }

    fn charge(&mut self, flow: u64, weight: u64, elapsed: Duration) {
        if let Some(flow) = self.flows.get_mut(&flow) {
            flow.served = flow
                .served
                .saturating_add(elapsed.as_nanos() * VIRTUAL_TIME_SCALE / u128::from(weight));
        }
    }
}

#[derive(Default)]
struct JobQueue {
    arbiter: Mutex<Arbiter>,
    ready: Condvar,
}

impl JobQueue {
    fn lock(&self) -> std::sync::MutexGuard<'_, Arbiter> {
        self.arbiter
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Serve jobs until the queue is stopped and drained.
    fn serve(&self) {
        loop {
            let next = {
                let mut arbiter = self.lock();
                loop {
                    if let Some(next) = arbiter.pop() {
                        break Some(next);
                    }
                    if arbiter.stopped {
                        break None;
                    }
                    arbiter = self
                        .ready
                        .wait(arbiter)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
            };
            let Some((job, charge)) = next else {
                return;
            };
            let started = Instant::now();
            // Jobs contain their own panics.
            job();
            if let Some((flow, weight)) = charge {
                self.lock().charge(flow, weight, started.elapsed());
            }
        }
    }

    /// Run one queued control job, waiting up to `timeout` for one.
    fn serve_control_job(&self, timeout: Duration) {
        let job = {
            let mut arbiter = self.lock();
            if arbiter.control.is_empty() {
                arbiter = self
                    .ready
                    .wait_timeout(arbiter, timeout)
                    .unwrap_or_else(|poison| poison.into_inner())
                    .0;
            }
            arbiter.control.pop_front()
        };
        if let Some(job) = job {
            job();
        }
    }
}

/// An owned OS thread that runs every job for one device.
pub struct DeviceThread {
    key: DeviceThreadKey,
    id: u64,
    queue: Arc<JobQueue>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

static NEXT_DEVICE_THREAD_ID: AtomicU64 = AtomicU64::new(1);
static DEVICE_OPS_OFF_THREAD: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The id of the device thread (or a scope delegated by one) this thread
    /// is running device work for.
    static CURRENT_DEVICE_THREAD: Cell<Option<u64>> = const { Cell::new(None) };
    /// The queue of the device thread this thread is.
    static CURRENT_QUEUE: RefCell<Option<Arc<JobQueue>>> = const { RefCell::new(None) };
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
        let queue = Arc::new(JobQueue::default());
        let jobs = queue.clone();
        let handle = std::thread::Builder::new()
            .name(key.thread_name())
            .spawn(move || {
                CURRENT_DEVICE_THREAD.with(|current| current.set(Some(id)));
                CURRENT_QUEUE.with(|queue| *queue.borrow_mut() = Some(jobs.clone()));
                jobs.serve();
            })
            .map_err(|error| {
                Error::InferenceError(format!(
                    "failed to start the {key:?} device thread: {error}"
                ))
            })?;
        Ok(Self {
            key,
            id,
            queue,
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

    fn enqueue(&self, job: Job, tag: Option<DeviceJobTag>) -> Result<()> {
        let mut arbiter = self.queue.lock();
        if arbiter.stopped {
            return Err(Error::InferenceError(format!(
                "the {:?} device thread stopped",
                self.key
            )));
        }
        arbiter.push(job, tag);
        drop(arbiter);
        self.queue.ready.notify_one();
        Ok(())
    }

    /// Run `job` on the device thread as a control job and resolve with its
    /// result. A panic in the job is contained and returned as an error; the
    /// thread keeps serving later jobs.
    pub fn submit<T, F>(&self, job: F) -> impl Future<Output = Result<T>> + Send + 'static
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.submit_job(None, job)
    }

    /// [`Self::submit`] for a model dispatch of the flow `tag` names.
    pub fn submit_tagged<T, F>(
        &self,
        tag: DeviceJobTag,
        job: F,
    ) -> impl Future<Output = Result<T>> + Send + 'static
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.submit_job(Some(tag), job)
    }

    fn submit_job<T, F>(
        &self,
        tag: Option<DeviceJobTag>,
        job: F,
    ) -> impl Future<Output = Result<T>> + Send + 'static
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (done, result) = oneshot::channel();
        let enqueued = self.enqueue(
            Box::new(move || {
                let _ = done.send(catch_unwind(AssertUnwindSafe(job)));
            }),
            tag,
        );
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

    /// Run `job` on the device thread as a control job and block the caller
    /// until it returns. Runs inline when the caller already is the device
    /// thread. Only call it
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
        // `done` (field order). The thread runs every accepted job before it
        // exits; a job `enqueue` refuses is dropped there, before it returns.
        let boxed: Job =
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'env>, Job>(boxed) };
        self.enqueue(boxed, None)?;
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
        self.queue.lock().stopped = true;
        self.queue.ready.notify_all();
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

/// Wait for `reply` without stalling the device: `None` when its sender is
/// dropped. On a device thread the wait keeps serving the device's control
/// jobs (loads, frees, KV page ops), so a device job that waits on async work
/// which itself needs the device (a load whose precision-reload admission
/// evicts another model) cannot deadlock its own thread. Model dispatches
/// stay queued until the job returns. Elsewhere it is a blocking receive.
pub fn recv_serving_device<T>(mut reply: oneshot::Receiver<T>) -> Option<T> {
    let Some(queue) = CURRENT_QUEUE.with(|queue| queue.borrow().clone()) else {
        return reply.blocking_recv().ok();
    };
    loop {
        match reply.try_recv() {
            Ok(value) => return Some(value),
            Err(oneshot::error::TryRecvError::Closed) => return None,
            // The reply does not notify the queue; the bounded wait is only
            // paid while a job is waiting, on a rare path.
            Err(oneshot::error::TryRecvError::Empty) => {
                queue.serve_control_job(Duration::from_millis(1))
            }
        }
    }
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

    /// A device job that waits for a reply produced by a control job queued
    /// behind it (a load waiting on an admission that evicts a model) serves
    /// that job instead of deadlocking its own thread.
    #[tokio::test]
    async fn a_waiting_job_keeps_serving_control_jobs() {
        let thread = test_thread();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (reply_tx, reply_rx) = oneshot::channel();
        let waiter = thread.submit(move || {
            started_tx.send(()).unwrap();
            recv_serving_device(reply_rx)
        });
        started_rx.recv().unwrap();
        let replier = thread.clone();
        let (sent_tx, sent_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let sent = replier.run_sync(move || reply_tx.send(7).is_ok());
            let _ = sent_tx.send(sent);
        });
        let reply = tokio::time::timeout(Duration::from_secs(10), waiter)
            .await
            .expect("the waiting job deadlocked its device thread")
            .unwrap();
        assert_eq!(reply, Some(7));
        assert!(sent_rx.recv().unwrap().unwrap());
        // A dropped sender ends the wait.
        let (reply_tx, reply_rx) = oneshot::channel::<u8>();
        drop(reply_tx);
        assert_eq!(
            thread
                .submit(move || recv_serving_device(reply_rx))
                .await
                .unwrap(),
            None
        );
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

    /// Hold `thread` busy until the returned sender fires, so later jobs
    /// queue behind it.
    fn block(thread: &DeviceThread) -> std::sync::mpsc::Sender<()> {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        drop(thread.submit(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        }));
        started_rx.recv().unwrap();
        release_tx
    }

    /// Queue `count` jobs of `flow` that each record the flow and then
    /// sleep for `busy`.
    fn queue_flow(
        thread: &DeviceThread,
        flow: u64,
        weight: u64,
        count: usize,
        busy: Duration,
        order: &Arc<Mutex<Vec<u64>>>,
    ) -> Vec<impl Future<Output = Result<()>>> {
        (0..count)
            .map(|_| {
                let order = order.clone();
                thread.submit_tagged(DeviceJobTag { flow, weight }, move || {
                    order.lock().unwrap().push(flow);
                    std::thread::sleep(busy);
                })
            })
            .collect()
    }

    #[tokio::test]
    async fn control_jobs_run_before_queued_dispatches() {
        let thread = test_thread();
        let order = Arc::new(Mutex::new(Vec::new()));
        let release = block(&thread);
        let dispatches = queue_flow(&thread, 1, 1, 2, Duration::ZERO, &order);
        let control = {
            let order = order.clone();
            thread.submit(move || order.lock().unwrap().push(0))
        };
        release.send(()).unwrap();
        control.await.expect("control job");
        for dispatch in dispatches {
            dispatch.await.expect("dispatch");
        }
        assert_eq!(*order.lock().unwrap(), [0, 1, 1]);
    }

    #[tokio::test]
    async fn a_lane_of_long_jobs_does_not_hold_the_device() {
        let thread = test_thread();
        let order = Arc::new(Mutex::new(Vec::new()));
        let release = block(&thread);
        // Lane 1 queued four long jobs before lane 2's short steps arrived.
        let (long, short) = (Duration::from_millis(100), Duration::from_millis(1));
        let mut jobs = queue_flow(&thread, 1, 1, 4, long, &order);
        jobs.extend(queue_flow(&thread, 2, 1, 8, short, &order));
        release.send(()).unwrap();
        for job in jobs {
            job.await.expect("job");
        }
        // After one long job, every short step runs before the next long one.
        assert_eq!(*order.lock().unwrap(), [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]);
    }

    #[test]
    fn lanes_share_the_device_by_weight() {
        // Charged with a fixed time: a slept job's measured time varies on a
        // loaded host, and so would the order.
        let mut arbiter = Arbiter::default();
        for (flow, weight) in [(1, 1), (2, 6)] {
            for _ in 0..12 {
                arbiter.push(Box::new(|| {}), Some(DeviceJobTag { flow, weight }));
            }
        }
        let mut order = Vec::new();
        while let Some((_, charge)) = arbiter.pop() {
            let (flow, weight) = charge.expect("tagged job");
            arbiter.charge(flow, weight, Duration::from_millis(5));
            order.push(flow);
        }
        // Equal jobs at weights 1 and 6: one light job per six heavy.
        let round = [1, 2, 2, 2, 2, 2, 2];
        assert_eq!(order, [&round[..], &round, &[1; 10]].concat());
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
