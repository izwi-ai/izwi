//! The `izwi-serving-supervisor` process: its command line, the run loop
//! that owns the worker actors, the rollout and autoscale drivers, and the
//! operator diagnostics. The binary is a thin wrapper over `run_from_cli`.

mod autoscale_driver;
mod cli;
mod diagnostics;
mod env_config;
mod error;
mod rollout_driver;

pub use error::SupervisorError;

use autoscale_driver::{run_autoscale_tick, sync_autoscaler};
use cli::{print_usage, CliOptions, ParseOutcome};
use diagnostics::{
    runtime_diagnostic, supervisor_prometheus, validation_diagnostic, write_metrics_textfile,
};
use rollout_driver::{
    execute_rollout, prepare_rollout, print_rollout_status, reconcile_fresh_start_state,
    rollout_abort_command, RolloutOutcome,
};

use izwi_serving_protocol::env::EnvSnapshot;
use izwi_serving_protocol::{BackendKind, DeploymentId, ServiceBearerToken, WorkerId};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, MissedTickBehavior};
use tracing::{info, warn};

use crate::actor::{
    bump, stop_all, LaunchCounts, LaunchOutcome, Supervision, WorkerActor, WorkerPhase,
};
use crate::autoscale::Autoscaler;
use crate::{
    autoscale, request_quarantine_clear, signals, take_quarantine_clear_requests, BinaryCatalog,
    BinaryRecord, CudaDeviceInventory, HostInventory, LockNamespace, MetalDeviceInventory,
    NodeConfig, QuarantineClear, ResolvedWorkerSecret, ValidatedNodeConfig, WorkerBinaryFlavor,
    WorkerConfig, MAX_NODE_CONFIG_BYTES,
};

const SUPERVISION_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// How often the supervisor looks for `--clear-quarantine` requests.
const QUARANTINE_CLEAR_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// The supervisor process: parse the command line, then supervise this
/// node's workers until shutdown.
pub async fn run_from_cli() -> Result<(), SupervisorError> {
    let env = env_config::capture();
    init_tracing(&env)?;
    let options = match CliOptions::parse(env::args_os().skip(1))? {
        ParseOutcome::Run(options) => options,
        ParseOutcome::Help => {
            print_usage();
            return Ok(());
        }
    };
    let metrics_textfile = env
        .get_os(env_config::METRICS_TEXTFILE_ENV)
        .map(PathBuf::from);
    run(*options, metrics_textfile).await
}

/// Supervisor logs go to stderr: text by default, one JSON object per line
/// with `IZWI_LOG_FORMAT=json`. `RUST_LOG` overrides the info-level filter.
fn init_tracing(env: &EnvSnapshot) -> Result<(), SupervisorError> {
    use std::io::IsTerminal;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "izwi_serving_supervisor=info".into());
    let registry = tracing_subscriber::registry().with(filter);
    match env.get(env_config::LOG_FORMAT_ENV)? {
        None | Some("text") => registry
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_ansi(std::io::stderr().is_terminal()),
            )
            .init(),
        Some("json") => registry
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .flatten_event(true)
                    .with_writer(std::io::stderr),
            )
            .init(),
        Some(other) => {
            return Err(SupervisorError::InvalidOption(format!(
                "IZWI_LOG_FORMAT={other:?} (expected text or json)"
            )))
        }
    }
    Ok(())
}

async fn run(
    options: CliOptions,
    metrics_textfile: Option<PathBuf>,
) -> Result<(), SupervisorError> {
    let validate_only = options.validate_only;
    let config_bytes = read_bounded(&options.config, MAX_NODE_CONFIG_BYTES)?;
    let config = NodeConfig::parse_bounded(&config_bytes)?;

    // --rollout-status only reports persisted state; no validation, locks,
    // or launch. The abort command runs after full node validation below.
    if options.rollout_status {
        return print_rollout_status(&config);
    }

    ensure_flavor_binaries(&config, &options)?;

    let inventory = HostInventory {
        effective_cpu_ids: options.cpu_ids.clone(),
        allocatable_host_memory_bytes: options.allocatable_host_memory_bytes,
        metal_devices: options
            .metal_devices
            .iter()
            .map(|(device_id, index)| MetalDeviceInventory {
                device_id: device_id.clone(),
                process_local_device_index: *index,
                // Declared Metal devices are Apple Silicon unified-memory GPUs;
                // config validation rejects any non-unified declaration.
                unified_memory: true,
            })
            .collect(),
        cuda_devices: options
            .cuda_devices
            .iter()
            .map(
                |(device_uuid, host_index, total_memory_bytes)| CudaDeviceInventory {
                    device_uuid: device_uuid.clone(),
                    host_device_index: *host_index,
                    total_memory_bytes: *total_memory_bytes,
                },
            )
            .collect(),
    };
    let mut binary_records = Vec::new();
    if let Some(path) = options.cpu_worker_binary.as_ref() {
        binary_records.push((
            WorkerBinaryFlavor::Cpu,
            BinaryRecord {
                path: path.clone(),
                supported_backends: vec![BackendKind::Cpu],
            },
        ));
    }
    if let Some(path) = options.metal_worker_binary.as_ref() {
        binary_records.push((
            WorkerBinaryFlavor::Metal,
            BinaryRecord {
                path: path.clone(),
                supported_backends: vec![BackendKind::Metal],
            },
        ));
    }
    if let Some(path) = options.cuda_worker_binary.as_ref() {
        binary_records.push((
            WorkerBinaryFlavor::Cuda,
            BinaryRecord {
                path: path.clone(),
                supported_backends: vec![BackendKind::Cuda],
            },
        ));
    }
    let binaries = BinaryCatalog::new(binary_records);
    let node = Arc::new(config.validate(&inventory, &binaries)?);
    // --clear-quarantine leaves a request for the running supervisor; it
    // takes no lock and launches nothing.
    if let Some(target) = options.clear_quarantine.as_deref() {
        return request_clear_quarantine(&node, target);
    }
    let secrets = resolve_secrets(node.config().workers.iter())?;
    if validate_only {
        print!("{}", validation_diagnostic(&node));
        return Ok(());
    }

    // --rollout-abort restores the pre-rollout approval view without
    // launching anything. Acquiring the node lease and generation barrier
    // proves no supervisor is live and no orphaned worker holds a fence.
    if options.rollout_abort {
        return rollout_abort_command(&node);
    }

    let locks = LockNamespace::open(&node.config().runtime_directory)?;
    let lock_metadata = format!("node={} pid={}", node.config().node_id, std::process::id());
    let _supervisor_lease = locks.try_node_supervisor(lock_metadata.as_bytes())?;
    // This proves every worker from the previous supervisor generation has released
    // its shared fence. Holding the node lease prevents a second new supervisor from
    // entering the gap after this exclusive barrier is released.
    let generation_barrier = locks.try_generation_barrier(lock_metadata.as_bytes())?;
    drop(generation_barrier);

    // DS6: rollout preparation runs under the supervisor lease so state
    // staging cannot race another supervisor. A fresh start without a plan
    // must reconcile any persisted rollout state fail-closed.
    // DS7: autoscaling and coordinated rollout are mutually exclusive in one
    // supervisor run — both own the shared approvals view. The check must
    // precede rollout preparation so a conflicting plan never stages state.
    if options.rollout_plan.is_some() && node.config().autoscaling.is_some() {
        return Err(SupervisorError::RolloutAutoscalingConflict);
    }
    let mut rollout = match options.rollout_plan.as_ref() {
        Some(plan_path) => Some(prepare_rollout(
            plan_path, &node, &inventory, &binaries, &options,
        )?),
        None => {
            reconcile_fresh_start_state(&node, &config_bytes)?;
            None
        }
    };
    let resume_draining_old = rollout
        .as_ref()
        .is_some_and(|prepared| prepared.resumed && prepared.state.phase.old_generation_retired());

    let supervision = Arc::new(Supervision::new(
        locks.clone(),
        crate::env::inherited_environment(),
        node.config().max_parallel_model_loads,
    ));
    let mut slots = Vec::with_capacity(secrets.len());
    for (worker_id, secret) in secrets {
        slots.push(WorkerSlot::new(WorkerActor::spawn(
            Arc::clone(&node),
            worker_id,
            secret,
            Arc::clone(&supervision),
        )?));
    }
    if let Some(prepared) = rollout.as_mut() {
        for (worker_id, secret) in std::mem::take(&mut prepared.replacement_secrets) {
            prepared.replacements.push(WorkerActor::spawn(
                Arc::clone(&prepared.target),
                worker_id,
                secret,
                Arc::clone(&supervision),
            )?);
        }
    }

    let mut autoscaler = Autoscaler::from_config(&node);
    if let Some(scaler) = autoscaler.as_mut() {
        // Partition slots: the min set keeps the static posture
        // (`autoscale: None`); every other declared worker of an autoscaled
        // deployment becomes a standby that only a scale-up decision starts.
        for slot in slots.iter_mut() {
            let Some(state) = scaler.deployment_of(slot.worker_id()) else {
                continue;
            };
            if state.core_set().contains(slot.worker_id()) {
                continue;
            }
            slot.autoscale = Some(AutoscaleSlotState::standby(state.deployment_id().clone()));
        }
        // Reconcile the shared view to the min set before any launch: lines
        // for this node's non-running autoscaled workers must not stay
        // approved, or the gateway would route to endpoints that do not exist.
        let approvals_path = node
            .config()
            .autoscaling
            .as_ref()
            .expect("the autoscaler exists only with the autoscaling block")
            .shared_approvals_path
            .clone();
        let desired: Vec<&WorkerConfig> = scaler
            .deployments()
            .values()
            .flat_map(|state| state.core_set().iter())
            .filter_map(|worker_id| node.worker(worker_id))
            .collect();
        autoscale::reconcile_min_set(&approvals_path, &node, &desired)
            .map_err(|error| SupervisorError::Autoscale(error.to_string()))?;
        info!(
            "autoscaling: reconciled {} to the min set of {} autoscaled deployment(s)",
            approvals_path.display(),
            scaler.deployments().len()
        );
    }
    if let Some(canary_id) = options.canary_worker_id.as_ref() {
        if slots
            .iter()
            .any(|slot| slot.worker_id() == canary_id && slot.is_standby())
        {
            return Err(SupervisorError::Autoscale(format!(
                "canary worker {canary_id} is an autoscaling standby; canaries must belong to the startup min set"
            )));
        }
    }

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        signals::shutdown_requested().await;
        let _ = shutdown_tx.send(true);
    });
    let (diagnostic_tx, mut diagnostic_rx) = mpsc::channel(1);
    signals::forward_diagnostic_requests(diagnostic_tx);
    let (quarantine_clear_tx, mut quarantine_clear_rx) = mpsc::channel(1);
    signals::forward_quarantine_clear_requests(quarantine_clear_tx);
    // DS6: SIGUSR2 requests an in-process rollout abort while the rollout is
    // still reversible (launch or window phases).
    let mut abort_rx = signals::rollout_abort_requests();

    if let Some(canary_id) = options.canary_worker_id.as_ref() {
        let canary = slots
            .iter()
            .find(|slot| slot.worker_id() == canary_id)
            .ok_or_else(|| SupervisorError::CanaryWorkerNotFound {
                worker_id: canary_id.clone(),
            })?;
        info!(
            "launching canary worker {} first; remaining workers will start after it reaches readiness",
            canary_id
        );
        match start_and_wait(&canary.actor, &mut shutdown_rx).await {
            Some(LaunchOutcome::Ready) => info!(
                "canary worker {} is ready; launching remaining workers",
                canary_id
            ),
            Some(LaunchOutcome::Failed) => {
                warn!(
                    "canary worker {} failed to reach readiness; aborting rollout",
                    canary_id
                );
                shut_down_all(slots).await;
                return Err(SupervisorError::CanaryReadinessFailed {
                    worker_id: canary_id.clone(),
                });
            }
            None => {
                shut_down_all(slots).await;
                return Ok(());
            }
        }
    }

    let mut started = Vec::new();
    for slot in &slots {
        if options.canary_worker_id.as_ref() == Some(slot.worker_id()) {
            continue;
        }
        // On resume past the drain the old generation is retired and its
        // workers exited before the generation barrier released; never
        // relaunch them.
        if resume_draining_old
            && rollout
                .as_ref()
                .is_some_and(|prepared| prepared.spec.is_old_worker(slot.worker_id().as_str()))
        {
            continue;
        }
        // DS7: autoscaling standbys wait for a scale-up decision.
        if slot.is_standby() {
            continue;
        }
        started.push((slot.actor.view().launch_counts(), &slot.actor));
        slot.actor.start();
    }

    // DS6: run the rollout state machine to commit or abort. The current
    // generation settles first, as a rollout starts from a running node.
    // After a commit the replacements join supervision.
    if let Some(prepared) = rollout.as_mut() {
        let settled = tokio::select! {
            () = async {
                for (before, actor) in &started {
                    actor.launch_outcome(*before).await;
                }
            } => true,
            _ = shutdown_rx.wait_for(|stop| *stop) => false,
        };
        drop(started);
        if settled {
            let outcome =
                execute_rollout(prepared, &node, &mut slots, &mut shutdown_rx, &mut abort_rx).await;
            let metrics = &supervision.metrics;
            match outcome {
                RolloutOutcome::Committed | RolloutOutcome::DegradedAbort => {
                    bump(if matches!(outcome, RolloutOutcome::Committed) {
                        &metrics.rollouts_committed
                    } else {
                        &metrics.rollouts_degraded
                    });
                    let prepared = rollout.take().expect("rollout is present");
                    slots.extend(prepared.replacements.into_iter().map(WorkerSlot::new));
                }
                RolloutOutcome::Aborted => {
                    bump(&metrics.rollouts_aborted);
                    rollout.take();
                }
            }
        }
    } else {
        drop(started);
    }

    let mut written_metrics = String::new();
    let mut poll = tokio::time::interval(SUPERVISION_POLL_INTERVAL);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut next_autoscale_eval = Instant::now()
        + autoscaler
            .as_ref()
            .map_or(Duration::ZERO, |scaler| scaler.evaluation_interval());
    let mut next_clear_check = Instant::now();
    while !*shutdown_rx.borrow() {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    break;
                }
            }
            Some(()) = diagnostic_rx.recv() => {
                eprint!("{}", runtime_diagnostic(&node, &slots, &supervision));
            }
            Some(()) = quarantine_clear_rx.recv() => {
                clear_quarantines(&slots, &QuarantineClear::All);
            }
            _ = poll.tick() => {
                if Instant::now() >= next_clear_check {
                    next_clear_check = Instant::now() + QUARANTINE_CLEAR_CHECK_INTERVAL;
                    for request in take_quarantine_clear_requests(&node.config().runtime_directory) {
                        clear_quarantines(&slots, &request);
                    }
                }
                if let Some(scaler) = autoscaler.as_mut() {
                    sync_autoscaler(scaler, &mut slots);
                }
                if let Some(path) = metrics_textfile.as_deref() {
                    let rendered = supervisor_prometheus(&slots, &supervision.metrics);
                    if rendered != written_metrics {
                        match write_metrics_textfile(path, &rendered) {
                            Ok(()) => written_metrics = rendered,
                            Err(error) => warn!(
                                "supervisor metrics textfile {} write failed: {error}",
                                path.display()
                            ),
                        }
                    }
                }
                // DS7: one autoscaling evaluation pass per configured interval.
                if Instant::now() >= next_autoscale_eval {
                    if let Some(scaler) = autoscaler.as_mut() {
                        next_autoscale_eval = Instant::now() + scaler.evaluation_interval();
                        run_autoscale_tick(scaler, &node, &mut slots, &supervision).await;
                    }
                }
            }
        }
    }

    shut_down_all(slots).await;
    Ok(())
}

/// Applies one operator quarantine clear to the matching workers.
fn clear_quarantines(slots: &[WorkerSlot], target: &QuarantineClear) {
    let mut matched = false;
    for slot in slots {
        if matches!(target, QuarantineClear::Worker(worker) if worker != slot.worker_id()) {
            continue;
        }
        matched = true;
        if matches!(slot.actor.view().phase, WorkerPhase::Quarantined { .. }) {
            info!("clearing the quarantine of worker {}", slot.worker_id());
            slot.actor.clear_quarantine();
        }
    }
    if !matched {
        warn!("quarantine clear request names an unknown worker: {target:?}");
    }
}

/// The `--clear-quarantine` command: validate the target against the node
/// config and leave a request for the running supervisor.
fn request_clear_quarantine(
    node: &ValidatedNodeConfig,
    target: &str,
) -> Result<(), SupervisorError> {
    let request = if target == "all" {
        QuarantineClear::All
    } else {
        let worker = WorkerId::new(target)
            .ok()
            .filter(|worker| node.worker(worker).is_some())
            .ok_or_else(|| {
                SupervisorError::InvalidOption(format!(
                "--clear-quarantine {target:?} names no configured worker (use a worker id or all)"
            ))
            })?;
        QuarantineClear::Worker(worker)
    };
    request_quarantine_clear(&node.config().runtime_directory, &request).map_err(|source| {
        SupervisorError::ReadConfig {
            path: node
                .config()
                .runtime_directory
                .join(crate::QUARANTINE_CLEAR_DIRECTORY),
            source,
        }
    })?;
    eprintln!(
        "quarantine clear requested for {target}; a running supervisor applies it within a second"
    );
    Ok(())
}

/// Starts an idle worker and waits for its first launch outcome; `None`
/// when shutdown came first.
async fn start_and_wait(
    actor: &WorkerActor,
    shutdown: &mut watch::Receiver<bool>,
) -> Option<LaunchOutcome> {
    let before = actor.view().launch_counts();
    actor.start();
    tokio::select! {
        outcome = actor.launch_outcome(before) => Some(outcome),
        _ = shutdown.wait_for(|stop| *stop) => None,
    }
}

/// Drains every worker in parallel and ends their actors.
pub(crate) async fn shut_down_all(slots: Vec<WorkerSlot>) {
    stop_all(slots.iter().map(|slot| &slot.actor)).await;
    for slot in slots {
        slot.actor.shut_down().await;
    }
}

/// Milliseconds since the supervisor started, from the monotonic clock.
pub(crate) fn monotonic_ms(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0)
}

/// Lifecycle phase of an autoscaling slot (DS7). Core (min-set) workers keep
/// `autoscale: None` and behave exactly like static workers; only standby
/// replicas and scale-event transitions carry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AutoscalePhase {
    /// Declared standby: not launched; only a scale-up decision starts it.
    Standby,
    /// Scale-up started the actor; the approvals line is published once its
    /// launch after `since` reaches readiness.
    Admitting { since: LaunchCounts },
    /// Running and approved in the shared approvals view.
    Active,
    /// Scale-down in progress: unapproved, admission stopped, draining.
    Draining { stop_requested: bool },
}

#[derive(Debug, Clone)]
pub(crate) struct AutoscaleSlotState {
    deployment: DeploymentId,
    phase: AutoscalePhase,
}

impl AutoscaleSlotState {
    fn standby(deployment: DeploymentId) -> Self {
        Self {
            deployment,
            phase: AutoscalePhase::Standby,
        }
    }
}

/// One supervised worker: its actor, plus the autoscaling bookkeeping the
/// main loop keeps for it.
pub(crate) struct WorkerSlot {
    actor: WorkerActor,
    autoscale: Option<AutoscaleSlotState>,
    /// The actor's ready-launch count at the last autoscaler sync.
    seen_ready_launches: u64,
    /// Whether the worker was running at the last autoscaler sync.
    was_running: bool,
}

impl WorkerSlot {
    fn new(actor: WorkerActor) -> Self {
        Self {
            actor,
            autoscale: None,
            seen_ready_launches: 0,
            was_running: false,
        }
    }

    fn worker_id(&self) -> &WorkerId {
        self.actor.worker_id()
    }

    fn is_standby(&self) -> bool {
        self.autoscale
            .as_ref()
            .is_some_and(|state| state.phase == AutoscalePhase::Standby)
    }

    /// Whether the restart controller, not a scale decision, owns this
    /// worker's process: static and core workers, and active scale-ups.
    fn restart_managed(&self) -> bool {
        self.autoscale
            .as_ref()
            .is_none_or(|state| state.phase == AutoscalePhase::Active)
    }
}

/// Resolves each worker's bearer token from the supervisor's environment.
pub(crate) fn resolve_secrets<'a>(
    workers: impl Iterator<Item = &'a WorkerConfig>,
) -> Result<Vec<(WorkerId, ResolvedWorkerSecret)>, SupervisorError> {
    workers
        .map(|worker| {
            let value =
                env::var(&worker.bearer_token_env).map_err(|_| SupervisorError::MissingSecret {
                    worker: worker.worker_id.clone(),
                    environment: worker.bearer_token_env.clone(),
                })?;
            let bearer_token = ServiceBearerToken::new(value).map_err(|source| {
                SupervisorError::InvalidSecret {
                    worker: worker.worker_id.clone(),
                    environment: worker.bearer_token_env.clone(),
                    source,
                }
            })?;
            Ok((
                worker.worker_id.clone(),
                ResolvedWorkerSecret { bearer_token },
            ))
        })
        .collect()
}

/// Every binary flavor referenced by the node configuration must have an
/// operator-provided path. Configuration validation rejects flavor/backend
/// mismatches; this only proves the operator supplied the binaries they
/// configured. The supervisor never probes or initializes accelerators:
/// device inventory is operator-declared and verified by the worker's own
/// assigned-device selection at startup.
pub(crate) fn ensure_flavor_binaries(
    config: &NodeConfig,
    options: &CliOptions,
) -> Result<(), SupervisorError> {
    for worker in &config.workers {
        let (provided, option) = match worker.binary {
            WorkerBinaryFlavor::Cpu => (&options.cpu_worker_binary, "--cpu-worker-binary"),
            WorkerBinaryFlavor::Metal => (&options.metal_worker_binary, "--metal-worker-binary"),
            WorkerBinaryFlavor::Cuda => (&options.cuda_worker_binary, "--cuda-worker-binary"),
        };
        if provided.is_none() {
            return Err(SupervisorError::MissingFlavorBinary {
                option,
                worker: worker.worker_id.clone(),
                binary: worker.binary,
            });
        }
    }
    Ok(())
}

pub(crate) fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>, SupervisorError> {
    let file = File::open(path).map_err(|source| SupervisorError::ReadConfig {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::with_capacity(maximum.min(64 * 1024));
    file.take((maximum as u64) + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| SupervisorError::ReadConfig {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > maximum {
        return Err(SupervisorError::ConfigTooLarge {
            actual_at_least: bytes.len(),
            maximum,
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
