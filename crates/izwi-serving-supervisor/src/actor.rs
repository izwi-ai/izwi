//! One supervision task per worker (CP-6).
//!
//! Each worker gets an actor task that owns its process: it waits for a
//! model-load permit, launches the worker and waits for readiness, watches
//! for its exit and probes its liveness, and restarts it under the bounded
//! restart policy. The supervisor's main loop only sends commands and reads
//! the view each actor publishes, so one worker's slow load or crash loop
//! never stalls exit handling, restarts, diagnostics or autoscaling for the
//! others. The node's `max_parallel_model_loads` bounds how many actors are
//! between spawn and readiness at once.

use std::{
    collections::BTreeMap,
    ffi::OsString,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{IncarnationId, ServiceCredentials, WorkerId};
use tokio::{
    sync::{mpsc, oneshot, watch, Semaphore},
    task::JoinHandle,
    time::Instant,
};
use tracing::{info, warn};

use crate::probe::{self, Liveness, LivenessTracker};
use crate::{
    build_child_launch_spec, ExpectedWorkerIdentity, LifecycleError, LockNamespace,
    ResolvedWorkerSecret, RestartController, RestartDecision, SupervisedWorker,
    ValidatedNodeConfig, WorkerLockPaths,
};

/// How long the actor waits before observing the process again after an
/// exit observation failed.
const EXIT_OBSERVATION_RETRY: Duration = Duration::from_secs(1);

/// Supervisor counters, shared by every actor and the main loop.
#[derive(Debug, Default)]
pub struct SupervisorMetrics {
    pub launch_attempts: AtomicU64,
    pub readiness_successes: AtomicU64,
    pub readiness_failures: AtomicU64,
    pub restarts_scheduled: AtomicU64,
    pub quarantines: AtomicU64,
    pub unexpected_exits: AtomicU64,
    pub exit_observation_failures: AtomicU64,
    pub liveness_restarts: AtomicU64,
    pub stop_failures: AtomicU64,
    pub autoscale_ups: AtomicU64,
    pub autoscale_downs: AtomicU64,
    pub rollouts_committed: AtomicU64,
    pub rollouts_aborted: AtomicU64,
    pub rollouts_degraded: AtomicU64,
}

/// Adds one to a supervisor counter.
pub fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Reads a supervisor counter.
pub fn count(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

/// What every actor of one supervisor shares.
#[derive(Debug)]
pub struct Supervision {
    pub locks: LockNamespace,
    pub inherited_environment: BTreeMap<OsString, OsString>,
    /// One permit per simultaneous launch-to-readiness stage
    /// (`max_parallel_model_loads`).
    pub load_permits: Semaphore,
    pub metrics: SupervisorMetrics,
    pub started_at: Instant,
}

impl Supervision {
    pub fn new(
        locks: LockNamespace,
        inherited_environment: BTreeMap<OsString, OsString>,
        max_parallel_model_loads: u32,
    ) -> Self {
        Self {
            locks,
            inherited_environment,
            load_permits: Semaphore::new(max_parallel_model_loads as usize),
            metrics: SupervisorMetrics::default(),
            started_at: Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPhase {
    /// Not running and not scheduled: an autoscaling standby, a replacement
    /// awaiting its start, or a stopped worker.
    Idle,
    /// Waiting for a model-load permit.
    WaitingForLoad,
    /// Spawned and waiting for readiness.
    Starting,
    Ready,
    /// Admission stopped by a scale-down; the worker exits on its own.
    Draining,
    /// Down after a failure, waiting for its scheduled restart.
    RestartPending {
        at: Instant,
    },
    /// Exceeded its restart budget; retried once at `retry_at`, if set.
    Quarantined {
        retry_at: Option<Instant>,
    },
}

/// What an actor publishes about its worker.
#[derive(Debug, Clone)]
pub struct WorkerView {
    pub phase: WorkerPhase,
    /// Launches that reached readiness: a new incarnation each time.
    pub ready_launches: u64,
    /// Launches that failed before readiness.
    pub failed_launches: u64,
    pub process_id: Option<u32>,
    pub incarnation: Option<IncarnationId>,
    /// A client for the running incarnation's status endpoint.
    pub client: Option<WorkerClient>,
    pub exit_observation_failed: bool,
}

impl WorkerView {
    fn idle() -> Self {
        Self {
            phase: WorkerPhase::Idle,
            ready_launches: 0,
            failed_launches: 0,
            process_id: None,
            incarnation: None,
            client: None,
            exit_observation_failed: false,
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.phase, WorkerPhase::Ready | WorkerPhase::Draining)
    }

    /// Down after a failure and coming back on its own.
    pub fn is_restarting(&self) -> bool {
        matches!(
            self.phase,
            WorkerPhase::RestartPending { .. }
                | WorkerPhase::WaitingForLoad
                | WorkerPhase::Starting
        )
    }

    pub fn launch_counts(&self) -> LaunchCounts {
        LaunchCounts {
            ready: self.ready_launches,
            failed: self.failed_launches,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchCounts {
    pub ready: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchOutcome {
    Ready,
    Failed,
}

enum Command {
    /// Launch an idle worker (a standby or a replacement).
    Start,
    /// Close the ownership pipe: the worker stops admission and exits once
    /// its work drains, without counting as a failure.
    BeginDrain,
    /// Drain and stop the worker (or cancel its pending restart) and stay
    /// idle; the sender learns when it is done.
    Stop(oneshot::Sender<()>),
    /// An operator cleared the quarantine: relaunch with a fresh budget.
    ClearQuarantine,
}

/// The main loop's handle on one worker actor.
pub struct WorkerActor {
    worker_id: WorkerId,
    node: Arc<ValidatedNodeConfig>,
    commands: mpsc::UnboundedSender<Command>,
    view: watch::Receiver<WorkerView>,
    task: JoinHandle<()>,
}

impl WorkerActor {
    /// Starts an idle actor for one configured worker.
    pub fn spawn(
        node: Arc<ValidatedNodeConfig>,
        worker_id: WorkerId,
        secret: ResolvedWorkerSecret,
        supervision: Arc<Supervision>,
    ) -> Result<Self, LifecycleError> {
        let restart = RestartController::for_worker(&node, &worker_id)?;
        let (commands, receiver) = mpsc::unbounded_channel();
        let (view_sender, view) = watch::channel(WorkerView::idle());
        let actor = Actor {
            worker_id: worker_id.clone(),
            node: Arc::clone(&node),
            secret,
            restart,
            supervision,
            commands: receiver,
            view: view_sender,
        };
        let task = tokio::spawn(actor.run());
        Ok(Self {
            worker_id,
            node,
            commands,
            view,
            task,
        })
    }

    pub fn worker_id(&self) -> &WorkerId {
        &self.worker_id
    }

    /// The node config this worker belongs to (the rollout target for a
    /// replacement).
    pub fn node(&self) -> &ValidatedNodeConfig {
        &self.node
    }

    pub fn view(&self) -> WorkerView {
        self.view.borrow().clone()
    }

    pub fn start(&self) {
        let _ = self.commands.send(Command::Start);
    }

    pub fn begin_drain(&self) {
        let _ = self.commands.send(Command::BeginDrain);
    }

    /// Clears a quarantine; a worker that is not quarantined ignores it.
    pub fn clear_quarantine(&self) {
        let _ = self.commands.send(Command::ClearQuarantine);
    }

    /// Asks the actor to stop its worker without waiting for it.
    pub fn request_stop(&self) -> oneshot::Receiver<()> {
        let (done, stopped) = oneshot::channel();
        let _ = self.commands.send(Command::Stop(done));
        stopped
    }

    /// Stops the worker and waits until it is gone.
    pub async fn stop(&self) {
        let _ = self.request_stop().await;
    }

    /// Waits for the first launch outcome after `before`.
    pub async fn launch_outcome(&self, before: LaunchCounts) -> LaunchOutcome {
        let mut view = self.view.clone();
        let outcome = view
            .wait_for(|view| {
                view.ready_launches > before.ready || view.failed_launches > before.failed
            })
            .await
            .map(|view| view.ready_launches > before.ready);
        match outcome {
            Ok(true) => LaunchOutcome::Ready,
            _ => LaunchOutcome::Failed,
        }
    }

    /// Stops the worker and ends the actor.
    pub async fn shut_down(self) {
        self.stop().await;
        drop(self.commands);
        let _ = self.task.await;
    }
}

/// Stops many actors at once and waits for all of them.
pub async fn stop_all<'a>(actors: impl IntoIterator<Item = &'a WorkerActor>) {
    let stopping: Vec<_> = actors.into_iter().map(WorkerActor::request_stop).collect();
    for stopped in stopping {
        let _ = stopped.await;
    }
}

/// Where the actor goes after a launch or a running process ends.
enum Next {
    Idle,
    /// Wait for a restart (or a quarantine retry) due at `until`; never
    /// when `until` is `None`.
    Wait {
        until: Option<Instant>,
        quarantined: bool,
    },
    Exit,
}

struct Actor {
    worker_id: WorkerId,
    node: Arc<ValidatedNodeConfig>,
    secret: ResolvedWorkerSecret,
    restart: RestartController,
    supervision: Arc<Supervision>,
    commands: mpsc::UnboundedReceiver<Command>,
    view: watch::Sender<WorkerView>,
}

impl Actor {
    async fn run(mut self) {
        let mut next = Next::Idle;
        loop {
            next = match next {
                Next::Exit => return,
                Next::Idle => {
                    self.set_phase(WorkerPhase::Idle);
                    match self.commands.recv().await {
                        None => Next::Exit,
                        Some(Command::Start) => self.launch().await,
                        Some(Command::Stop(done)) => {
                            let _ = done.send(());
                            Next::Idle
                        }
                        Some(Command::BeginDrain | Command::ClearQuarantine) => Next::Idle,
                    }
                }
                Next::Wait { until, quarantined } => {
                    self.set_phase(match (quarantined, until) {
                        (true, retry_at) => WorkerPhase::Quarantined { retry_at },
                        (false, Some(at)) => WorkerPhase::RestartPending { at },
                        (false, None) => WorkerPhase::Idle,
                    });
                    let due = async {
                        match until {
                            Some(at) => tokio::time::sleep_until(at).await,
                            None => std::future::pending().await,
                        }
                    };
                    tokio::select! {
                        () = due => {
                            if quarantined {
                                info!("worker {} quarantine retry: launching it once", self.worker_id);
                            }
                            self.launch().await
                        }
                        command = self.commands.recv() => match command {
                            None => Next::Exit,
                            Some(Command::Start) => self.launch().await,
                            Some(Command::Stop(done)) => {
                                let _ = done.send(());
                                Next::Idle
                            }
                            Some(Command::ClearQuarantine) if quarantined => {
                                self.restart.clear_quarantine();
                                info!(
                                    "worker {} quarantine cleared by the operator; relaunching",
                                    self.worker_id
                                );
                                self.launch().await
                            }
                            Some(Command::BeginDrain | Command::ClearQuarantine) => {
                                Next::Wait { until, quarantined }
                            }
                        },
                    }
                }
            };
        }
    }

    fn set_phase(&self, phase: WorkerPhase) {
        self.view.send_modify(|view| {
            view.phase = phase;
            if !view.is_running() {
                view.process_id = None;
                view.incarnation = None;
                view.client = None;
                view.exit_observation_failed = false;
            }
        });
    }

    fn metrics(&self) -> &SupervisorMetrics {
        &self.supervision.metrics
    }

    /// One launch: wait for a load permit, spawn, wait for readiness, then
    /// supervise the running process until it ends.
    async fn launch(&mut self) -> Next {
        self.set_phase(WorkerPhase::WaitingForLoad);
        let supervision = Arc::clone(&self.supervision);
        let permit = loop {
            tokio::select! {
                permit = supervision.load_permits.acquire() => {
                    break permit.expect("the load semaphore is never closed");
                }
                command = self.commands.recv() => match command {
                    None => return Next::Exit,
                    Some(Command::Stop(done)) => {
                        let _ = done.send(());
                        return Next::Idle;
                    }
                    Some(Command::Start | Command::BeginDrain | Command::ClearQuarantine) => {}
                },
            }
        };
        self.set_phase(WorkerPhase::Starting);
        bump(&self.metrics().launch_attempts);
        let mut process = match self.spawn_process() {
            Ok(process) => process,
            Err(error) => {
                warn!("worker {} launch failed: {error}", self.worker_id);
                return self.launch_failed(Duration::ZERO);
            }
        };
        let process_started_at = Instant::now();
        let readiness_policy = self.node.config().readiness.clone();
        let readiness = {
            let ready = process.wait_until_ready(&readiness_policy);
            tokio::pin!(ready);
            loop {
                tokio::select! {
                    result = &mut ready => break Ok(result),
                    command = self.commands.recv() => match command {
                        None => break Err(None),
                        Some(Command::Stop(done)) => break Err(Some(done)),
                        Some(Command::Start | Command::BeginDrain | Command::ClearQuarantine) => {}
                    },
                }
            }
        };
        drop(permit);
        match readiness {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                warn!("worker {} failed readiness: {error}", self.worker_id);
                self.stop_process(process).await;
                return self.launch_failed(process_started_at.elapsed());
            }
            Err(done) => {
                self.stop_process(process).await;
                return match done {
                    Some(done) => {
                        let _ = done.send(());
                        Next::Idle
                    }
                    None => Next::Exit,
                };
            }
        }
        bump(&self.metrics().readiness_successes);
        self.restart.note_ready();
        let incarnation = process.readiness().expected().incarnation_id().clone();
        info!(
            "worker {} ready as incarnation {}",
            self.worker_id, incarnation
        );
        self.view.send_modify(|view| {
            view.phase = WorkerPhase::Ready;
            view.ready_launches += 1;
            view.process_id = process.process_id();
            view.incarnation = Some(incarnation);
            view.client = Some(process.client().clone());
            view.exit_observation_failed = false;
        });
        self.supervise(process, process_started_at).await
    }

    fn spawn_process(&self) -> Result<SupervisedWorker, String> {
        let node = &self.node;
        let incarnation = IncarnationId::new(uuid::Uuid::new_v4().simple().to_string())
            .expect("UUID incarnation is a valid bounded identity");
        let worker = node
            .worker(&self.worker_id)
            .expect("actors are spawned for configured workers");
        let worker_locks = WorkerLockPaths::for_worker(
            &self.supervision.locks,
            &self.worker_id,
            &worker.assignment,
        );
        let spec = build_child_launch_spec(
            node,
            &self.worker_id,
            &incarnation,
            &worker.bearer_token_env,
            &self.secret,
            &self.supervision.inherited_environment,
            &worker_locks,
        )
        .map_err(|error| format!("launch specification: {error}"))?;
        let expected = ExpectedWorkerIdentity::from_config(node, &self.worker_id, incarnation)
            .map_err(|error| format!("identity setup: {error}"))?;
        let credentials = ServiceCredentials {
            credential_id: worker.credential_id.clone(),
            bearer_token: self.secret.bearer_token.clone(),
        };
        SupervisedWorker::spawn(&spec, expected, credentials, WorkerClientConfig::default())
            .map_err(|error| format!("spawn: {error}"))
    }

    /// Watches a ready process until it exits or is stopped.
    async fn supervise(&mut self, mut process: SupervisedWorker, started_at: Instant) -> Next {
        let liveness = self.node.config().liveness.clone();
        let period = Duration::from_millis(liveness.period_ms);
        let mut probe_timer = tokio::time::interval_at(Instant::now() + period, period);
        probe_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut tracker = LivenessTracker::new(&liveness);
        let mut draining = false;
        loop {
            let event = tokio::select! {
                exit = process.wait_for_exit() => Event::Exited(exit),
                command = self.commands.recv() => Event::Command(command),
                _ = probe_timer.tick(), if !draining => Event::Probe,
            };
            match event {
                Event::Probe => {
                    let answer = tokio::time::timeout(
                        Duration::from_millis(liveness.timeout_ms),
                        process.client().status(),
                    )
                    .await
                    .ok();
                    let verdict = probe::judge(process.readiness().expected(), answer);
                    match tracker.observe(verdict) {
                        Liveness::Alive => {}
                        Liveness::Suspect { failures, reason } => warn!(
                            "worker {} liveness probe failed ({failures} of {}): {reason}",
                            self.worker_id, liveness.failure_threshold
                        ),
                        Liveness::Dead { reason } => {
                            bump(&self.metrics().liveness_restarts);
                            warn!(
                                "worker {} failed {} consecutive liveness probes ({reason}); restarting it",
                                self.worker_id, liveness.failure_threshold
                            );
                            self.terminate_process(process).await;
                            return self.failed(started_at.elapsed());
                        }
                    }
                }
                Event::Exited(Ok(status)) if draining => {
                    info!(
                        "worker {} exited after its drain ({status})",
                        self.worker_id
                    );
                    return Next::Idle;
                }
                Event::Exited(Ok(status)) => {
                    bump(&self.metrics().unexpected_exits);
                    warn!("worker {} exited with {status}", self.worker_id);
                    return self.failed(started_at.elapsed());
                }
                Event::Exited(Err(error)) => {
                    // A failed observation does not prove the process
                    // exited: keep ownership and look again shortly.
                    if !self.view.borrow().exit_observation_failed {
                        bump(&self.metrics().exit_observation_failures);
                        warn!("worker {} exit observation failed: {error}", self.worker_id);
                        self.view
                            .send_modify(|view| view.exit_observation_failed = true);
                    }
                    tokio::time::sleep(EXIT_OBSERVATION_RETRY).await;
                }
                Event::Command(Some(Command::BeginDrain)) => {
                    if let Err(error) = process.begin_drain().await {
                        warn!("worker {} admission stop failed: {error}", self.worker_id);
                    }
                    draining = true;
                    self.view
                        .send_modify(|view| view.phase = WorkerPhase::Draining);
                }
                Event::Command(Some(Command::Start | Command::ClearQuarantine)) => {}
                Event::Command(Some(Command::Stop(done))) => {
                    self.stop_process(process).await;
                    let _ = done.send(());
                    return Next::Idle;
                }
                Event::Command(None) => {
                    self.stop_process(process).await;
                    return Next::Exit;
                }
            }
        }
    }

    /// Stops a worker that stopped answering, without a cooperative drain.
    async fn terminate_process(&self, process: SupervisedWorker) {
        match process.terminate(&self.node.config().shutdown).await {
            Ok(report) => info!(
                "worker {} stopped with {:?} ({})",
                self.worker_id, report.outcome, report.exit_status
            ),
            Err(error) => {
                bump(&self.metrics().stop_failures);
                warn!("worker {} stop failed: {error}", self.worker_id);
            }
        }
    }

    async fn stop_process(&self, process: SupervisedWorker) {
        match process.drain_and_stop(&self.node.config().shutdown).await {
            Ok(report) => info!(
                "worker {} stopped with {:?} ({})",
                self.worker_id, report.outcome, report.exit_status
            ),
            Err(error) => {
                bump(&self.metrics().stop_failures);
                warn!("worker {} stop failed: {error}", self.worker_id);
            }
        }
    }

    fn launch_failed(&mut self, uptime: Duration) -> Next {
        bump(&self.metrics().readiness_failures);
        self.view.send_modify(|view| view.failed_launches += 1);
        self.failed(uptime)
    }

    /// Schedules the restart the policy allows, or quarantines.
    fn failed(&mut self, uptime: Duration) -> Next {
        let now = self.supervision.started_at.elapsed();
        match self.restart.record_failure(now, uptime) {
            RestartDecision::RestartAfter(delay) => {
                bump(&self.metrics().restarts_scheduled);
                info!(
                    "worker {} unavailable; restart scheduled in {} ms",
                    self.worker_id,
                    delay.as_millis()
                );
                Next::Wait {
                    until: Some(Instant::now() + delay),
                    quarantined: false,
                }
            }
            RestartDecision::Quarantine => {
                bump(&self.metrics().quarantines);
                let retry = self.restart.quarantine_retry();
                match retry {
                    Some(retry) => warn!(
                        "worker {} exceeded its bounded restart budget and is quarantined; retrying it in {} s",
                        self.worker_id,
                        retry.as_secs()
                    ),
                    None => warn!(
                        "worker {} exceeded its bounded restart budget and is quarantined until an operator clears it",
                        self.worker_id
                    ),
                }
                Next::Wait {
                    until: retry.map(|retry| Instant::now() + retry),
                    quarantined: true,
                }
            }
        }
    }
}

enum Event {
    Exited(Result<std::process::ExitStatus, LifecycleError>),
    Command(Option<Command>),
    Probe,
}
