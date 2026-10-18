//! The engine step driver, completion waiters and the guards that own a
//! request until the engine releases it.

use super::*;

#[derive(Debug)]
pub(super) struct StreamOutputOrder {
    pub(super) last_sequence: Option<usize>,
    pub(super) final_seen: bool,
    pub(super) allow_gaps: bool,
}

impl StreamOutputOrder {
    pub(super) fn new(policy: EngineStreamPolicy) -> Self {
        Self {
            last_sequence: None,
            final_seen: false,
            allow_gaps: policy == EngineStreamPolicy::DropNewest,
        }
    }

    pub(super) fn observe(&mut self, request_id: &str, chunk: &StreamingOutput) -> Result<()> {
        if chunk.request_id != request_id {
            return Err(Error::InferenceError(format!(
                "stream output for {request_id} carried request ID {}",
                chunk.request_id
            )));
        }
        if self.final_seen {
            return Err(Error::InferenceError(format!(
                "stream output for {request_id} arrived after its final marker"
            )));
        }
        if self
            .last_sequence
            .is_some_and(|last| chunk.sequence <= last)
        {
            return Err(Error::InferenceError(format!(
                "stream output sequence {} for {request_id} was not greater than its predecessor",
                chunk.sequence
            )));
        }
        let expected = self.last_sequence.map_or(0, |last| last.saturating_add(1));
        if !self.allow_gaps && chunk.sequence != expected {
            return Err(Error::InferenceError(format!(
                "stream output sequence {} for {request_id} did not match expected {expected}",
                chunk.sequence
            )));
        }
        self.last_sequence = Some(chunk.sequence);
        self.final_seen = chunk.is_final;
        Ok(())
    }

    pub(super) fn require_final(&self, request_id: &str) -> Result<()> {
        if self.final_seen {
            Ok(())
        } else {
            Err(Error::InferenceError(format!(
                "stream for {request_id} closed without a final marker"
            )))
        }
    }
}

pub(super) struct RuntimeCompletionWaiter {
    pub(super) registration_id: u64,
    pub(super) session_epoch: Option<u64>,
    pub(super) sender: oneshot::Sender<Result<EngineOutput>>,
}

pub(super) type RuntimeCompletionWaiters = Mutex<HashMap<String, RuntimeCompletionWaiter>>;

pub(super) async fn remove_waiter_registration(
    waiters: &RuntimeCompletionWaiters,
    request_id: &str,
    registration_id: u64,
) -> bool {
    let mut waiters = waiters.lock().await;
    if waiters
        .get(request_id)
        .is_some_and(|waiter| waiter.registration_id == registration_id)
    {
        waiters.remove(request_id);
        true
    } else {
        false
    }
}

pub(super) async fn bind_waiter_registration(
    waiters: &RuntimeCompletionWaiters,
    request_id: &str,
    registration_id: u64,
    session_epoch: u64,
) -> bool {
    let mut waiters = waiters.lock().await;
    let Some(waiter) = waiters.get_mut(request_id) else {
        return false;
    };
    if waiter.registration_id != registration_id || waiter.session_epoch.is_some() {
        return false;
    }
    waiter.session_epoch = Some(session_epoch);
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StepOutcome {
    Produced,
    Idle,
    Failed,
}

/// The task that steps the engine and routes terminal outputs: one loop
/// over whole-engine steps, or one loop per execution lane with lane step
/// drivers.
#[derive(Clone)]
pub(super) struct StepDriver {
    pub(super) engine: Arc<CoreEngine>,
    pub(super) waiters: Arc<RuntimeCompletionWaiters>,
    pub(super) telemetry: Arc<RuntimeTelemetryCollector>,
    pub(super) wakeup: Arc<Notify>,
}

impl StepDriver {
    pub(super) async fn drive_engine(self) {
        let mut idle_backoff_ms = 1u64;
        loop {
            if !self.engine.has_pending_work().await {
                self.engine
                    .idle_wait(Duration::from_millis(idle_backoff_ms), Some(&self.wakeup))
                    .await;
                idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
                continue;
            }
            match self.step(None).await {
                StepOutcome::Produced => idle_backoff_ms = 1,
                StepOutcome::Idle => {
                    // Sleeps until the next retry or deadline the scheduler
                    // holds, or a wakeup, at the latest.
                    self.engine
                        .idle_wait(Duration::from_millis(idle_backoff_ms), Some(&self.wakeup))
                        .await;
                    idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
                }
                StepOutcome::Failed => {}
            }
        }
    }

    /// Run one step driver per lane with scheduler work, so one lane's long
    /// dispatch does not hold another lane's steps. Work no lane holds
    /// (terminal outbox, cleanup retries, delivery lanes) runs as a
    /// whole-engine step while no lane is stepping.
    pub(super) async fn drive_lanes(self) {
        let lane_wakeup = Arc::new(Notify::new());
        let mut lanes = HashMap::new();
        loop {
            lanes.retain(|_, driver: &mut tokio::task::JoinHandle<()>| !driver.is_finished());
            for lane in self.engine.lanes_with_work().await {
                lanes.entry(lane).or_insert_with(|| {
                    tokio::spawn(self.clone().drive_lane(lane, lane_wakeup.clone()))
                });
            }
            if lanes.is_empty()
                && self.engine.has_pending_work().await
                && self.step(None).await == StepOutcome::Produced
            {
                continue;
            }
            self.engine
                .idle_wait(Duration::from_millis(50), Some(&self.wakeup))
                .await;
            lane_wakeup.notify_waiters();
        }
    }

    /// Step one lane until it holds no scheduler work.
    pub(super) async fn drive_lane(self, lane: Option<crate::engine::ModelInstanceId>, wakeup: Arc<Notify>) {
        let mut idle_backoff_ms = 1u64;
        while self.engine.lanes_with_work().await.contains(&lane) {
            if self.step(Some(lane)).await == StepOutcome::Produced {
                idle_backoff_ms = 1;
                continue;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(idle_backoff_ms)) => {}
                _ = wakeup.notified() => {}
            }
            idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
        }
    }

    /// Run one step of `lane` (the whole engine for `None`) and route its
    /// terminal outputs.
    pub(super) async fn step(&self, lane: Option<Option<crate::engine::ModelInstanceId>>) -> StepOutcome {
        let step = match lane {
            Some(lane) => self.engine.step_lane_for_dispatch(lane).boxed(),
            None => self.engine.step_for_dispatch().boxed(),
        };
        match std::panic::AssertUnwindSafe(step).catch_unwind().await {
            Ok(Ok(outputs)) => {
                let produced = if outputs.is_empty() {
                    StepOutcome::Idle
                } else {
                    StepOutcome::Produced
                };
                for output in outputs.into_iter().filter(|output| output.is_finished) {
                    route_terminal_output(
                        self.engine.as_ref(),
                        self.waiters.as_ref(),
                        self.telemetry.as_ref(),
                        output,
                    )
                    .await;
                }
                produced
            }
            Ok(Err(err)) => {
                error!(
                    error = %err,
                    "Engine step failed before commit; scheduled quanta were rolled back"
                );
                tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                StepOutcome::Failed
            }
            Err(payload) => {
                let panic_message = panic_payload_to_string(payload.as_ref());
                self.telemetry.record_worker_panic();
                let mut w = self.waiters.lock().await;
                let pending: Vec<_> = w.drain().collect();
                drop(w);
                let request_ids: Vec<_> = pending.iter().map(|(id, _)| id.as_str()).collect();
                self.telemetry.record_forced_failures(request_ids).await;
                let _ = self.engine.abort_all_requests().await;
                for (_, waiter) in pending {
                    let _ = waiter.sender.send(Err(Error::InferenceError(format!(
                        "Engine worker panicked: {}",
                        panic_message
                    ))));
                }
                error!(
                    "Engine step worker panicked ({}); continuing with isolated loop",
                    panic_message
                );
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                StepOutcome::Failed
            }
        }
    }
}

pub(super) async fn route_terminal_output(
    engine: &CoreEngine,
    waiters: &RuntimeCompletionWaiters,
    telemetry: &RuntimeTelemetryCollector,
    output: EngineOutput,
) {
    debug_assert!(output.is_finished);

    // Removing the waiter is the routing hand-off. A missing waiter or a
    // dropped receiver still means there is no live runtime consumer left, so
    // either case must release the exact-session delivery fence.
    let waiter = loop {
        let mut waiters = waiters.lock().await;
        match waiters.get(&output.request_id) {
            Some(waiter) if waiter.session_epoch == Some(output.sequence_id) => {
                break waiters.remove(&output.request_id);
            }
            // A registration is installed before its engine session exists.
            // Let admission bind it before deciding whether this terminal
            // belongs to that waiter; otherwise an old output can steal a
            // newly reused public request ID.
            Some(waiter) if waiter.session_epoch.is_none() => {
                drop(waiters);
                tokio::task::yield_now().await;
            }
            // A waiter for a later exact session must remain registered.
            Some(_) | None => break None,
        }
    };
    if let Some(waiter) = waiter {
        let _ = waiter.sender.send(runtime_completion(output.clone()));
    }

    // Do not make the public request ID reusable until the runtime has
    // attempted terminal delivery to the waiter selected above.
    if !engine.acknowledge_dispatched_terminal(&output).await {
        warn!(
            request_id = %output.request_id,
            session_epoch = output.sequence_id,
            "Runtime terminal output had no matching delivery fence"
        );
    }
    telemetry.record_request_finished(&output).await;
}

pub(super) struct WaiterRegistrationGuard {
    pub(super) request_id: String,
    pub(super) registration_id: u64,
    pub(super) completion_waiters: Arc<RuntimeCompletionWaiters>,
    pub(super) active: bool,
}

impl WaiterRegistrationGuard {
    pub(super) fn new(
        request_id: String,
        registration_id: u64,
        completion_waiters: Arc<RuntimeCompletionWaiters>,
    ) -> Self {
        Self {
            request_id,
            registration_id,
            completion_waiters,
            active: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for WaiterRegistrationGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let request_id = self.request_id.clone();
        let registration_id = self.registration_id;
        let waiters = self.completion_waiters.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                remove_waiter_registration(waiters.as_ref(), &request_id, registration_id).await;
            });
        }
    }
}

pub(super) struct PendingRequestGuard {
    pub(super) session: SessionKey,
    pub(super) core_engine: Arc<CoreEngine>,
    pub(super) completion_waiters: Arc<RuntimeCompletionWaiters>,
    pub(super) waiter_registration_id: u64,
    pub(super) telemetry: Arc<RuntimeTelemetryCollector>,
    pub(super) job: Option<JobLease>,
    pub(super) residency_lease: Option<ModelResidencyLease>,
    pub(super) active: bool,
}

/// Owns cancellation-sensitive admission state after its request future is
/// gone. Dropping the detached cleanup task (runtime shutdown, panic, or task
/// abortion) deliberately leaks these leases instead of allowing a model or
/// resource reservation to disappear underneath native work that was never
/// proven stopped.
pub(super) struct DeferredRequestOwnership {
    pub(super) job: Option<JobLease>,
    pub(super) residency_lease: Option<ModelResidencyLease>,
    pub(super) release_confirmed: bool,
}

impl DeferredRequestOwnership {
    pub(super) fn new(job: Option<JobLease>, residency_lease: Option<ModelResidencyLease>) -> Self {
        Self {
            job,
            residency_lease,
            release_confirmed: false,
        }
    }

    pub(super) fn release(mut self) {
        self.release_confirmed = true;
    }
}

impl Drop for DeferredRequestOwnership {
    fn drop(&mut self) {
        if self.release_confirmed {
            return;
        }
        if let Some(residency_lease) = self.residency_lease.take() {
            std::mem::forget(residency_lease);
        }
        if let Some(job) = self.job.take() {
            std::mem::forget(job);
        }
    }
}

pub(super) async fn cleanup_pending_request(
    session: SessionKey,
    engine: Arc<CoreEngine>,
    waiters: Arc<RuntimeCompletionWaiters>,
    waiter_registration_id: u64,
    telemetry: Arc<RuntimeTelemetryCollector>,
    ownership: DeferredRequestOwnership,
) {
    remove_waiter_registration(
        waiters.as_ref(),
        &session.request_id,
        waiter_registration_id,
    )
    .await;

    let aborted = match engine.abort_request_session(&session).await {
        Ok(aborted) => aborted,
        Err(err) => {
            warn!(
                request_id = %session.request_id,
                session_epoch = session.epoch,
                error = %err,
                "Exact-session cancellation cleanup failed; retaining request ownership"
            );
            return;
        }
    };
    // The exact abort waits for any in-flight engine step and its first
    // physical cleanup attempt. Only that successful hand-off permits the
    // model and coordinator admission to be released.
    ownership.release();
    if aborted {
        telemetry
            .record_request_cancelled(&session.request_id)
            .await;
    }
}

impl PendingRequestGuard {
    pub(super) fn new(
        session: SessionKey,
        core_engine: Arc<CoreEngine>,
        completion_waiters: Arc<RuntimeCompletionWaiters>,
        waiter_registration_id: u64,
        telemetry: Arc<RuntimeTelemetryCollector>,
        job: JobLease,
        residency_lease: Option<ModelResidencyLease>,
    ) -> Self {
        Self {
            session,
            core_engine,
            completion_waiters,
            waiter_registration_id,
            telemetry,
            job: Some(job),
            residency_lease,
            active: true,
        }
    }

    pub(super) fn disarm(&mut self) {
        self.active = false;
        self.job.take();
        self.residency_lease.take();
    }

    /// Cancel and fence the exact Engine session before releasing admission.
    ///
    /// This future is cancellation-safe: if its owner disappears while the
    /// Engine step lock is held, `Drop` transfers the still-owned leases into
    /// the existing fail-closed detached cleanup path.
    pub(super) async fn confirm_cleanup(&mut self) -> Result<bool> {
        remove_waiter_registration(
            self.completion_waiters.as_ref(),
            &self.session.request_id,
            self.waiter_registration_id,
        )
        .await;
        let aborted = self
            .core_engine
            .abort_request_session(&self.session)
            .await?;
        if aborted {
            self.telemetry
                .record_request_cancelled(&self.session.request_id)
                .await;
        }
        self.disarm();
        Ok(aborted)
    }

    /// Transfer cancellation cleanup to a detached exact-session task without
    /// waiting for the engine core lock. Streaming callbacks execute outside
    /// the engine, so a failed or timed-out transport must be able to return
    /// promptly even while a native step still owns that lock.
    pub(super) fn defer_cleanup(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;

        let session = self.session.clone();
        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();
        let waiter_registration_id = self.waiter_registration_id;
        let telemetry = self.telemetry.clone();
        let job = self.job.take();
        let residency_lease = self.residency_lease.take();
        let ownership = DeferredRequestOwnership::new(job, residency_lease);

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(cleanup_pending_request(
                session,
                engine,
                waiters,
                waiter_registration_id,
                telemetry,
                ownership,
            ));
        } else {
            drop(ownership);
        }
    }
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        self.defer_cleanup();
    }
}

impl RuntimeService {
    pub(super) async fn ensure_step_driver_started(&self) {
        let mut guard = self.step_driver_task.lock().await;
        let restart_needed = match guard.as_ref() {
            Some(handle) if !handle.is_finished() => false,
            Some(_) => true,
            None => true,
        };

        if !restart_needed {
            self.step_driver_started.store(true, Ordering::Release);
            return;
        }

        if guard.is_some() {
            self.telemetry.record_worker_restart();
        }

        let driver = StepDriver {
            engine: self.core_engine.clone(),
            waiters: self.completion_waiters.clone(),
            telemetry: self.telemetry.clone(),
            wakeup: self.step_driver_wakeup.clone(),
        };
        let task = if driver.engine.lane_step_drivers() {
            tokio::spawn(driver.drive_lanes())
        } else {
            tokio::spawn(driver.drive_engine())
        };

        *guard = Some(task);
        self.step_driver_started.store(true, Ordering::Release);
    }

    pub(super) async fn register_waiter(
        &self,
        request_id: &str,
    ) -> Result<(u64, oneshot::Receiver<Result<EngineOutput>>)> {
        use std::collections::hash_map::Entry;

        let registration_id = self
            .next_completion_waiter_registration
            .fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        let mut waiters = self.completion_waiters.lock().await;
        match waiters.entry(request_id.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(RuntimeCompletionWaiter {
                    registration_id,
                    session_epoch: None,
                    sender: tx,
                });
                Ok((registration_id, rx))
            }
            Entry::Occupied(_) => Err(Error::InvalidInput(format!(
                "request {request_id} already has a completion waiter"
            ))),
        }
    }

    pub(super) async fn remove_waiter(&self, request_id: &str, registration_id: u64) {
        remove_waiter_registration(
            self.completion_waiters.as_ref(),
            request_id,
            registration_id,
        )
        .await;
    }

    pub(super) async fn bind_waiter(
        &self,
        request_id: &str,
        registration_id: u64,
        session_epoch: u64,
    ) -> Result<()> {
        if bind_waiter_registration(
            self.completion_waiters.as_ref(),
            request_id,
            registration_id,
            session_epoch,
        )
        .await
        {
            Ok(())
        } else {
            Err(Error::InferenceError(format!(
                "request {request_id} lost its completion waiter before session binding"
            )))
        }
    }

    pub(super) async fn await_completion(
        &self,
        request_id: &str,
        rx: oneshot::Receiver<Result<EngineOutput>>,
        deadline: Option<std::time::Instant>,
    ) -> Result<EngineOutput> {
        let completion = async {
            rx.await.map_err(|_| {
                Error::InferenceError(format!(
                    "Request {} completion channel closed unexpectedly",
                    request_id
                ))
            })?
        };
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), completion)
                .await
                .map_err(|_| Error::Timeout(request_id.to_string()))?,
            None => completion.await,
        }
    }
}
