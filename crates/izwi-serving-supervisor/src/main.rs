use izwi_serving_client::WorkerClientConfig;
use izwi_serving_protocol::{
    BackendKind, DeviceAssignment, IncarnationId, ServiceBearerToken, ServiceCredentials, WorkerId,
};
use izwi_serving_supervisor::{
    build_child_launch_spec, BinaryCatalog, BinaryRecord, HostInventory, LockNamespace, NodeConfig,
    ResolvedWorkerSecret, RestartController, RestartDecision, SupervisedWorker,
    ValidatedNodeConfig, WorkerBinaryFlavor, WorkerLockPaths, MAX_NODE_CONFIG_BYTES,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::{OsStr, OsString},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{
    sync::{mpsc, watch},
    task::JoinSet,
    time::{Instant, MissedTickBehavior},
};

const MAX_CLI_ARGUMENTS: usize = 16;
const MAX_CPU_IDS: usize = 1024;
const MAX_DIAGNOSTIC_RESPONSE_BYTES: usize = 16 * 1024;
const TRUNCATED_DIAGNOSTIC_SUFFIX: &str = "\ndiagnostic_output=truncated\n";
const SUPERVISION_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[tokio::main]
async fn main() -> Result<(), SupervisorError> {
    let options = match CliOptions::parse(env::args_os().skip(1))? {
        ParseOutcome::Run(options) => options,
        ParseOutcome::Help => {
            print_usage();
            return Ok(());
        }
    };
    run(options).await
}

async fn run(options: CliOptions) -> Result<(), SupervisorError> {
    let validate_only = options.validate_only;
    let config_bytes = read_bounded(&options.config, MAX_NODE_CONFIG_BYTES)?;
    let config = NodeConfig::parse_bounded(&config_bytes)?;
    require_cpu_only(&config)?;

    let inventory = HostInventory {
        effective_cpu_ids: options.cpu_ids,
        allocatable_host_memory_bytes: options.allocatable_host_memory_bytes,
        metal_devices: Vec::new(),
        cuda_devices: Vec::new(),
    };
    let binaries = BinaryCatalog::new([(
        WorkerBinaryFlavor::Cpu,
        BinaryRecord {
            path: options.cpu_worker_binary,
            supported_backends: vec![BackendKind::Cpu],
        },
    )]);
    let node = config.validate(&inventory, &binaries)?;
    let mut slots = resolve_slots(&node)?;
    if validate_only {
        print!("{}", validation_diagnostic(&node));
        return Ok(());
    }

    let inherited_environment = inherited_environment();

    let locks = LockNamespace::open(&node.config().runtime_directory)?;
    let lock_metadata = format!("node={} pid={}", node.config().node_id, std::process::id());
    let _supervisor_lease = locks.try_node_supervisor(lock_metadata.as_bytes())?;
    // This proves every worker from the previous supervisor generation has released
    // its shared fence. Holding the node lease prevents a second new supervisor from
    // entering the gap after this exclusive barrier is released.
    let generation_barrier = locks.try_generation_barrier(lock_metadata.as_bytes())?;
    drop(generation_barrier);

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        wait_for_shutdown_request().await;
        let _ = shutdown_tx.send(true);
    });
    let (diagnostic_tx, mut diagnostic_rx) = mpsc::channel(1);
    spawn_diagnostic_signal_listener(diagnostic_tx);

    let started_at = Instant::now();
    let mut metrics = SupervisorMetrics::default();
    for slot in &mut slots {
        if *shutdown_rx.borrow() {
            break;
        }
        launch_slot(
            &node,
            &locks,
            &inherited_environment,
            slot,
            &mut shutdown_rx,
            started_at,
            &mut metrics,
        )
        .await;
    }

    let mut poll = tokio::time::interval(SUPERVISION_POLL_INTERVAL);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    while !*shutdown_rx.borrow() {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    break;
                }
            }
            Some(()) = diagnostic_rx.recv() => {
                eprint!("{}", runtime_diagnostic(&node, &slots, &metrics, started_at));
            }
            _ = poll.tick() => {
                observe_exits(&mut slots, started_at, &mut metrics);
                if let Some(slot) = next_restart_slot(&mut slots) {
                    launch_slot(
                        &node,
                        &locks,
                        &inherited_environment,
                        slot,
                        &mut shutdown_rx,
                        started_at,
                        &mut metrics,
                    ).await;
                }
            }
        }
    }

    drain_all(slots, node.config().shutdown.clone(), &mut metrics).await;
    Ok(())
}

struct WorkerSlot {
    worker_id: WorkerId,
    secret_environment_name: String,
    secret: ResolvedWorkerSecret,
    restart: RestartController,
    process: Option<SupervisedWorker>,
    process_started_at: Option<Instant>,
    restart_at: Option<Instant>,
    exit_observation_failed: bool,
}

#[derive(Debug, Default)]
struct SupervisorMetrics {
    launch_attempts: u64,
    readiness_successes: u64,
    readiness_failures: u64,
    restarts_scheduled: u64,
    quarantines: u64,
    unexpected_exits: u64,
    exit_observation_failures: u64,
    stop_failures: u64,
}

impl WorkerSlot {
    fn restart_due(&self) -> bool {
        self.process.is_none()
            && !self.restart.is_quarantined()
            && self
                .restart_at
                .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn record_failure(
        &mut self,
        supervisor_started_at: Instant,
        uptime: Duration,
        metrics: &mut SupervisorMetrics,
    ) {
        self.process = None;
        self.process_started_at = None;
        self.exit_observation_failed = false;
        match self
            .restart
            .record_failure(supervisor_started_at.elapsed(), uptime)
        {
            RestartDecision::RestartAfter(delay) => {
                metrics.restarts_scheduled = metrics.restarts_scheduled.saturating_add(1);
                self.restart_at = Some(Instant::now() + delay);
                eprintln!(
                    "worker {} unavailable; restart scheduled in {} ms",
                    self.worker_id,
                    delay.as_millis()
                );
            }
            RestartDecision::Quarantine => {
                metrics.quarantines = metrics.quarantines.saturating_add(1);
                self.restart_at = None;
                eprintln!(
                    "worker {} exceeded its bounded restart budget and is quarantined",
                    self.worker_id
                );
            }
        }
    }
}

fn resolve_slots(node: &ValidatedNodeConfig) -> Result<Vec<WorkerSlot>, SupervisorError> {
    node.config()
        .workers
        .iter()
        .map(|worker| {
            let value =
                env::var(&worker.bearer_token_env).map_err(|_| SupervisorError::MissingSecret {
                    worker: worker.worker_id.clone(),
                    environment: worker.bearer_token_env.clone(),
                })?;
            let secret = ResolvedWorkerSecret {
                bearer_token: ServiceBearerToken::new(value).map_err(|source| {
                    SupervisorError::InvalidSecret {
                        worker: worker.worker_id.clone(),
                        environment: worker.bearer_token_env.clone(),
                        source,
                    }
                })?,
            };
            Ok(WorkerSlot {
                worker_id: worker.worker_id.clone(),
                secret_environment_name: worker.bearer_token_env.clone(),
                secret,
                restart: RestartController::for_worker(node, &worker.worker_id)?,
                process: None,
                process_started_at: None,
                restart_at: Some(Instant::now()),
                exit_observation_failed: false,
            })
        })
        .collect()
}

fn validation_diagnostic(node: &ValidatedNodeConfig) -> String {
    let mut output = BoundedDiagnostic::new();
    output.push_line(&format!(
        "validation=ok mode=validate-only node={} workers={} credentials=validated-redacted",
        node.config().node_id,
        node.config().workers.len()
    ));
    for worker in &node.config().workers {
        output.push_line(&format!(
            "worker={} backend={:?} bind={} deployment={} generation={} task={:?} precision={} execution={} streaming={} max_active_invocations={} secret=redacted",
            worker.worker_id,
            worker.assignment.backend(),
            worker.bind,
            worker.deployment.deployment_id,
            worker.deployment.model_generation.get(),
            worker.deployment.task,
            worker.deployment.precision,
            worker.deployment.execution_representation,
            worker.deployment.capability.streaming,
            worker.max_active_invocations,
        ));
    }
    output.finish()
}

fn runtime_diagnostic(
    node: &ValidatedNodeConfig,
    slots: &[WorkerSlot],
    metrics: &SupervisorMetrics,
    started_at: Instant,
) -> String {
    let mut output = BoundedDiagnostic::new();
    let running = slots.iter().filter(|slot| slot.process.is_some()).count();
    let quarantined = slots
        .iter()
        .filter(|slot| slot.restart.is_quarantined())
        .count();
    let restart_pending = slots
        .iter()
        .filter(|slot| slot.process.is_none() && slot.restart_at.is_some())
        .count();
    output.push_line(&format!(
        "supervisor_status version={} node={} uptime_ms={} workers={} running={} restart_pending={} quarantined={} launch_attempts_total={} readiness_successes_total={} readiness_failures_total={} restarts_scheduled_total={} quarantines_total={} unexpected_exits_total={} exit_observation_failures_total={} stop_failures_total={}",
        env!("CARGO_PKG_VERSION"),
        node.config().node_id,
        started_at.elapsed().as_millis(),
        slots.len(),
        running,
        restart_pending,
        quarantined,
        metrics.launch_attempts,
        metrics.readiness_successes,
        metrics.readiness_failures,
        metrics.restarts_scheduled,
        metrics.quarantines,
        metrics.unexpected_exits,
        metrics.exit_observation_failures,
        metrics.stop_failures,
    ));
    for slot in slots {
        let configured = node
            .worker(&slot.worker_id)
            .expect("worker slots are derived from validated configuration");
        let (state, process_id, incarnation, assignment_source) =
            if let Some(process) = &slot.process {
                (
                    if slot.exit_observation_failed {
                        "observation-uncertain"
                    } else {
                        "ready"
                    },
                    process
                        .process_id()
                        .map_or_else(|| "unavailable".to_string(), |id| id.to_string()),
                    process.readiness().expected().incarnation_id().to_string(),
                    "verified",
                )
            } else if slot.restart.is_quarantined() {
                (
                    "quarantined",
                    "none".to_string(),
                    "none".to_string(),
                    "configured",
                )
            } else if slot.restart_at.is_some() {
                (
                    "restart-pending",
                    "none".to_string(),
                    "none".to_string(),
                    "configured",
                )
            } else {
                (
                    "unavailable",
                    "none".to_string(),
                    "none".to_string(),
                    "configured",
                )
            };
        output.push_line(&format!(
            "worker={} state={} pid={} incarnation={} assignment_source={} backend={:?} assignment={:?} deployment={} generation={} secret=redacted",
            slot.worker_id,
            state,
            process_id,
            incarnation,
            assignment_source,
            configured.assignment.backend(),
            configured.assignment,
            configured.deployment.deployment_id,
            configured.deployment.model_generation.get(),
        ));
    }
    output.finish()
}

fn spawn_diagnostic_signal_listener(sender: mpsc::Sender<()>) {
    #[cfg(unix)]
    tokio::spawn(async move {
        // This is a local administrative surface: Unix signal permissions gate
        // access, and no unauthenticated network listener is introduced.
        let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())
        else {
            return;
        };
        while signal.recv().await.is_some() {
            // Coalesce repeated operator requests instead of buffering diagnostics.
            let _ = sender.try_send(());
        }
    });
    #[cfg(not(unix))]
    drop(sender);
}

struct BoundedDiagnostic {
    output: String,
    truncated: bool,
}

impl BoundedDiagnostic {
    fn new() -> Self {
        Self {
            output: String::with_capacity(MAX_DIAGNOSTIC_RESPONSE_BYTES),
            truncated: false,
        }
    }

    fn push_line(&mut self, line: &str) {
        if self.truncated {
            return;
        }
        let payload_limit = MAX_DIAGNOSTIC_RESPONSE_BYTES - TRUNCATED_DIAGNOSTIC_SUFFIX.len();
        let required = line.len().saturating_add(1);
        if self.output.len().saturating_add(required) <= payload_limit {
            self.output.push_str(line);
            self.output.push('\n');
            return;
        }

        let mut remaining = payload_limit
            .saturating_sub(self.output.len())
            .min(line.len());
        while !line.is_char_boundary(remaining) {
            remaining -= 1;
        }
        self.output.push_str(&line[..remaining]);
        self.output.push_str(TRUNCATED_DIAGNOSTIC_SUFFIX);
        self.truncated = true;
    }

    fn finish(self) -> String {
        debug_assert!(self.output.len() <= MAX_DIAGNOSTIC_RESPONSE_BYTES);
        self.output
    }
}

async fn launch_slot(
    node: &ValidatedNodeConfig,
    locks: &LockNamespace,
    inherited_environment: &BTreeMap<OsString, OsString>,
    slot: &mut WorkerSlot,
    shutdown: &mut watch::Receiver<bool>,
    supervisor_started_at: Instant,
    metrics: &mut SupervisorMetrics,
) {
    metrics.launch_attempts = metrics.launch_attempts.saturating_add(1);
    slot.restart_at = None;
    let incarnation = IncarnationId::new(uuid::Uuid::new_v4().simple().to_string())
        .expect("UUID incarnation is a valid bounded identity");
    let worker = node
        .worker(&slot.worker_id)
        .expect("worker slots are derived from the validated node");
    let worker_locks = WorkerLockPaths::for_worker(locks, &slot.worker_id, &worker.assignment);
    let spec = match build_child_launch_spec(
        node,
        &slot.worker_id,
        &incarnation,
        &slot.secret_environment_name,
        &slot.secret,
        inherited_environment,
        &worker_locks,
    ) {
        Ok(spec) => spec,
        Err(error) => {
            eprintln!(
                "worker {} launch specification failed: {error}",
                slot.worker_id
            );
            metrics.readiness_failures = metrics.readiness_failures.saturating_add(1);
            slot.record_failure(supervisor_started_at, Duration::ZERO, metrics);
            return;
        }
    };
    let expected = match izwi_serving_supervisor::ExpectedWorkerIdentity::from_config(
        node,
        &slot.worker_id,
        incarnation,
    ) {
        Ok(expected) => expected,
        Err(error) => {
            eprintln!("worker {} identity setup failed: {error}", slot.worker_id);
            metrics.readiness_failures = metrics.readiness_failures.saturating_add(1);
            slot.record_failure(supervisor_started_at, Duration::ZERO, metrics);
            return;
        }
    };
    let credentials = ServiceCredentials {
        credential_id: worker.credential_id.clone(),
        bearer_token: slot.secret.bearer_token.clone(),
    };
    let mut process = match SupervisedWorker::spawn(
        &spec,
        expected,
        credentials,
        WorkerClientConfig::default(),
    ) {
        Ok(process) => process,
        Err(error) => {
            eprintln!("worker {} spawn failed: {error}", slot.worker_id);
            metrics.readiness_failures = metrics.readiness_failures.saturating_add(1);
            slot.record_failure(supervisor_started_at, Duration::ZERO, metrics);
            return;
        }
    };
    let process_started_at = Instant::now();
    let readiness = tokio::select! {
        result = process.wait_until_ready(&node.config().readiness) => Some(result),
        changed = shutdown.changed() => {
            let _ = changed;
            None
        }
    };
    match readiness {
        Some(Ok(_)) => {
            metrics.readiness_successes = metrics.readiness_successes.saturating_add(1);
            eprintln!(
                "worker {} ready as incarnation {}",
                slot.worker_id,
                process.readiness().expected().incarnation_id()
            );
            slot.process = Some(process);
            slot.process_started_at = Some(process_started_at);
            slot.exit_observation_failed = false;
        }
        Some(Err(error)) => {
            metrics.readiness_failures = metrics.readiness_failures.saturating_add(1);
            eprintln!("worker {} failed readiness: {error}", slot.worker_id);
            if let Err(stop_error) = process.drain_and_stop(&node.config().shutdown).await {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("worker {} cleanup failed: {stop_error}", slot.worker_id);
            }
            slot.record_failure(supervisor_started_at, process_started_at.elapsed(), metrics);
        }
        None => {
            if let Err(error) = process.drain_and_stop(&node.config().shutdown).await {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("worker {} shutdown cleanup failed: {error}", slot.worker_id);
            }
        }
    }
}

fn observe_exits(
    slots: &mut [WorkerSlot],
    supervisor_started_at: Instant,
    metrics: &mut SupervisorMetrics,
) {
    for slot in slots {
        let Some(process) = slot.process.as_mut() else {
            continue;
        };
        match process.try_wait() {
            Ok(Some(status)) => {
                metrics.unexpected_exits = metrics.unexpected_exits.saturating_add(1);
                eprintln!("worker {} exited with {status}", slot.worker_id);
                let uptime = slot
                    .process_started_at
                    .map_or(Duration::ZERO, |started| started.elapsed());
                slot.record_failure(supervisor_started_at, uptime, metrics);
            }
            Ok(None) => slot.exit_observation_failed = false,
            Err(error) => {
                if !slot.exit_observation_failed {
                    metrics.exit_observation_failures =
                        metrics.exit_observation_failures.saturating_add(1);
                    eprintln!("worker {} exit observation failed: {error}", slot.worker_id);
                    slot.exit_observation_failed = true;
                }
                // A failed observation does not prove the process exited. Retain
                // ownership and do not schedule a replacement until exit is known.
            }
        }
    }
}

fn next_restart_slot(slots: &mut [WorkerSlot]) -> Option<&mut WorkerSlot> {
    let index = slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.restart_due())
        .min_by_key(|(_, slot)| slot.restart_at.expect("restart-ready slot has a deadline"))
        .map(|(index, _)| index)?;
    slots.get_mut(index)
}

async fn drain_all(
    slots: Vec<WorkerSlot>,
    policy: izwi_serving_supervisor::ShutdownPolicy,
    metrics: &mut SupervisorMetrics,
) {
    let mut drains = JoinSet::new();
    for slot in slots {
        if let Some(process) = slot.process {
            let worker_id = slot.worker_id;
            let policy = policy.clone();
            drains.spawn(async move { (worker_id, process.drain_and_stop(&policy).await) });
        }
    }
    while let Some(result) = drains.join_next().await {
        match result {
            Ok((worker_id, Ok(report))) => eprintln!(
                "worker {worker_id} stopped with {:?} ({})",
                report.outcome, report.exit_status
            ),
            Ok((worker_id, Err(error))) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("worker {worker_id} shutdown failed: {error}")
            }
            Err(error) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("worker shutdown task failed: {error}")
            }
        }
    }
}

fn require_cpu_only(config: &NodeConfig) -> Result<(), SupervisorError> {
    for worker in &config.workers {
        if !matches!(worker.assignment, DeviceAssignment::Cpu { .. })
            || worker.binary != WorkerBinaryFlavor::Cpu
        {
            return Err(SupervisorError::UnsupportedDeviceLane {
                worker: worker.worker_id.clone(),
                backend: worker.assignment.backend(),
                binary: worker.binary,
            });
        }
    }
    Ok(())
}

fn inherited_environment() -> BTreeMap<OsString, OsString> {
    izwi_serving_supervisor::INHERITED_ENV_ALLOWLIST
        .iter()
        .filter_map(|name| env::var_os(name).map(|value| (OsString::from(name), value)))
        .collect()
}

fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>, SupervisorError> {
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

async fn wait_for_shutdown_request() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM listener");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    config: PathBuf,
    cpu_worker_binary: PathBuf,
    cpu_ids: Vec<u16>,
    allocatable_host_memory_bytes: u64,
    validate_only: bool,
}

enum ParseOutcome {
    Run(CliOptions),
    Help,
}

impl CliOptions {
    fn parse(
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<ParseOutcome, SupervisorError> {
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        if arguments.len() > MAX_CLI_ARGUMENTS {
            return Err(SupervisorError::TooManyArguments);
        }
        if arguments.len() == 1 && arguments[0] == OsStr::new("--help") {
            return Ok(ParseOutcome::Help);
        }
        let mut config = None;
        let mut cpu_worker_binary = None;
        let mut cpu_ids = None;
        let mut allocatable_host_memory_bytes = None;
        let mut validate_only = false;
        let mut index = 0;
        while index < arguments.len() {
            let name = arguments[index]
                .to_str()
                .ok_or(SupervisorError::NonUtf8OptionName)?;
            if name == "--validate-only" {
                if validate_only {
                    return Err(SupervisorError::DuplicateOption(name.to_string()));
                }
                validate_only = true;
                index += 1;
                continue;
            }
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| SupervisorError::MissingOptionValue(name.to_string()))?;
            match name {
                "--config" => set_once(&mut config, PathBuf::from(value), name)?,
                "--cpu-worker-binary" => {
                    set_once(&mut cpu_worker_binary, PathBuf::from(value), name)?
                }
                "--cpu-ids" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut cpu_ids, parse_cpu_ids(value)?, name)?;
                }
                "--allocatable-host-memory-bytes" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    let parsed = value
                        .parse::<u64>()
                        .ok()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut allocatable_host_memory_bytes, parsed, name)?;
                }
                _ => return Err(SupervisorError::UnknownOption(name.to_string())),
            }
            index += 2;
        }
        Ok(ParseOutcome::Run(Self {
            config: config.ok_or(SupervisorError::MissingOption("--config"))?,
            cpu_worker_binary: cpu_worker_binary
                .ok_or(SupervisorError::MissingOption("--cpu-worker-binary"))?,
            cpu_ids: cpu_ids.ok_or(SupervisorError::MissingOption("--cpu-ids"))?,
            allocatable_host_memory_bytes: allocatable_host_memory_bytes.ok_or(
                SupervisorError::MissingOption("--allocatable-host-memory-bytes"),
            )?,
            validate_only,
        }))
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), SupervisorError> {
    if slot.replace(value).is_some() {
        Err(SupervisorError::DuplicateOption(name.to_string()))
    } else {
        Ok(())
    }
}

fn parse_cpu_ids(value: &str) -> Result<Vec<u16>, SupervisorError> {
    if value.is_empty() || value.len() > 16 * 1024 {
        return Err(SupervisorError::InvalidCpuIds);
    }
    let mut seen = BTreeSet::new();
    for item in value.split(',') {
        let id = item
            .parse::<u16>()
            .map_err(|_| SupervisorError::InvalidCpuIds)?;
        if !seen.insert(id) || seen.len() > MAX_CPU_IDS {
            return Err(SupervisorError::InvalidCpuIds);
        }
    }
    Ok(seen.into_iter().collect())
}

fn print_usage() {
    eprintln!(
        "Usage: izwi-serving-supervisor \\\n  --config PATH \\\n  --cpu-worker-binary PATH \\\n  --cpu-ids 0,1,... \\\n  --allocatable-host-memory-bytes BYTES\n\n\
This executable intentionally accepts CPU workers only. CPU IDs and allocatable host memory\n\
must come from an operator or a trusted launcher; it does not probe or initialize accelerators."
    );
    eprintln!(
        "Optional: --validate-only resolves configuration and service credentials, prints bounded redacted diagnostics, and exits without acquiring locks or launching workers."
    );
    #[cfg(unix)]
    eprintln!(
        "Send SIGUSR1 to a running supervisor for one bounded, redacted status snapshot on stderr."
    );
}

#[derive(Debug, thiserror::Error)]
enum SupervisorError {
    #[error("too many command-line arguments")]
    TooManyArguments,
    #[error("command-line option name is not UTF-8")]
    NonUtf8OptionName,
    #[error("unknown option {0}")]
    UnknownOption(String),
    #[error("option {0} is repeated")]
    DuplicateOption(String),
    #[error("option {0} requires a value")]
    MissingOptionValue(String),
    #[error("required option {0} is missing")]
    MissingOption(&'static str),
    #[error("option {0} has an invalid value")]
    InvalidOption(String),
    #[error("CPU IDs must be a non-empty, bounded, unique comma-separated u16 list")]
    InvalidCpuIds,
    #[error("failed to read node configuration {path}: {source}")]
    ReadConfig { path: PathBuf, source: io::Error },
    #[error("node configuration is at least {actual_at_least} bytes; maximum is {maximum}")]
    ConfigTooLarge {
        actual_at_least: usize,
        maximum: usize,
    },
    #[error(transparent)]
    Config(#[from] izwi_serving_supervisor::ConfigError),
    #[error(transparent)]
    Lock(#[from] izwi_serving_supervisor::LockError),
    #[error(transparent)]
    Lifecycle(#[from] izwi_serving_supervisor::LifecycleError),
    #[error("worker {worker} requires unsupported {backend:?}/{binary:?}; this executable supports explicit CPU assignments only")]
    UnsupportedDeviceLane {
        worker: WorkerId,
        backend: BackendKind,
        binary: WorkerBinaryFlavor,
    },
    #[error("worker {worker} secret environment variable {environment} is not set")]
    MissingSecret {
        worker: WorkerId,
        environment: String,
    },
    #[error("worker {worker} secret environment variable {environment} is invalid: {source}")]
    InvalidSecret {
        worker: WorkerId,
        environment: String,
        source: izwi_serving_protocol::IdentifierError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_protocol::{
        ArtifactRevision, CancellationBehavior, CredentialId, DeploymentId, DeviceId, InputFormat,
        ModelAlias, ModelGeneration, NodeId, OutputFormat, TaskKind,
    };
    use izwi_serving_supervisor::{
        CapabilityProfileConfig, DeploymentConfig, ReadinessPolicy, RestartPolicy, ShutdownPolicy,
        WorkerConfig, NODE_CONFIG_SCHEMA_VERSION,
    };
    use std::collections::BTreeSet;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn config(assignment: DeviceAssignment, binary: WorkerBinaryFlavor) -> NodeConfig {
        NodeConfig {
            schema_version: NODE_CONFIG_SCHEMA_VERSION,
            node_id: id::<NodeId>("node-a"),
            working_directory: PathBuf::from("/work"),
            runtime_directory: PathBuf::from("/run"),
            host_memory_budget_bytes: 1024,
            workers: vec![WorkerConfig {
                worker_id: id("worker-a"),
                bind: "127.0.0.1:9470".parse().unwrap(),
                binary,
                credential_id: id::<CredentialId>("credential-a"),
                bearer_token_env: "WORKER_TOKEN".into(),
                assignment,
                deployment: DeploymentConfig {
                    deployment_id: id::<DeploymentId>("deployment-a"),
                    public_model: id::<ModelAlias>("model-a"),
                    artifact_revision: id::<ArtifactRevision>("revision-a"),
                    model_generation: ModelGeneration::new(1).unwrap(),
                    task: TaskKind::Chat,
                    backend: BackendKind::Cpu,
                    precision: "gguf-q4_k_m".into(),
                    execution_representation: "native-lfm2".into(),
                    tokenizer_revision: None,
                    capability: CapabilityProfileConfig {
                        streaming: true,
                        realtime: false,
                        cancellation: CancellationBehavior::Cooperative,
                        accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                        output_formats: BTreeSet::from([OutputFormat::Text]),
                        max_input_bytes: 1024,
                        max_context_tokens: Some(32),
                        max_output_tokens: Some(32),
                    },
                    models_directory: PathBuf::from("/models"),
                },
                max_active_invocations: 1,
                max_request_bytes: 1024,
                max_retained_attempts: 1,
                attempt_retention_secs: 60,
                streaming: true,
            }],
            readiness: ReadinessPolicy::default(),
            restart: RestartPolicy::default(),
            shutdown: ShutdownPolicy::default(),
        }
    }

    #[test]
    fn cli_requires_explicit_bounded_inventory() {
        let arguments = [
            "--config",
            "/config.toml",
            "--validate-only",
            "--cpu-worker-binary",
            "/worker",
            "--cpu-ids",
            "3,1",
            "--allocatable-host-memory-bytes",
            "4096",
        ]
        .map(OsString::from);
        let ParseOutcome::Run(options) = CliOptions::parse(arguments).unwrap() else {
            panic!("expected runnable options")
        };
        assert_eq!(options.cpu_ids, vec![1, 3]);
        assert_eq!(options.allocatable_host_memory_bytes, 4096);
        assert!(options.validate_only);
        assert!(matches!(
            CliOptions::parse([OsString::from("--config"), OsString::from("/x")]),
            Err(SupervisorError::MissingOption("--cpu-worker-binary"))
        ));
        assert!(matches!(
            parse_cpu_ids("1,1"),
            Err(SupervisorError::InvalidCpuIds)
        ));
        assert!(matches!(
            CliOptions::parse(
                [
                    "--validate-only",
                    "--validate-only",
                    "--config",
                    "/config.toml",
                    "--cpu-worker-binary",
                    "/worker",
                    "--cpu-ids",
                    "1",
                    "--allocatable-host-memory-bytes",
                    "4096",
                ]
                .map(OsString::from)
            ),
            Err(SupervisorError::DuplicateOption(option)) if option == "--validate-only"
        ));
    }

    #[test]
    fn diagnostic_buffer_is_hard_bounded() {
        let mut diagnostic = BoundedDiagnostic::new();
        diagnostic.push_line(&"x".repeat(MAX_DIAGNOSTIC_RESPONSE_BYTES * 2));
        diagnostic.push_line("must-not-appear");
        let output = diagnostic.finish();

        assert_eq!(output.len(), MAX_DIAGNOSTIC_RESPONSE_BYTES);
        assert!(output.ends_with(TRUNCATED_DIAGNOSTIC_SUFFIX));
        assert!(!output.contains("must-not-appear"));
    }

    #[test]
    fn runtime_diagnostic_is_bounded_and_redacts_credentials() {
        let root = tempfile::tempdir().unwrap();
        let working_directory = root.path().join("work");
        let runtime_directory = root.path().join("run");
        let models_directory = root.path().join("models");
        std::fs::create_dir(&working_directory).unwrap();
        std::fs::create_dir(&runtime_directory).unwrap();
        std::fs::create_dir(&models_directory).unwrap();
        let worker_binary = root.path().join("worker");
        std::fs::write(&worker_binary, b"worker").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&worker_binary, std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let assignment = DeviceAssignment::Cpu {
            thread_budget: 1,
            affinity: vec![1],
            host_memory_limit_bytes: 512,
        };
        let mut config = config(assignment, WorkerBinaryFlavor::Cpu);
        config.working_directory = working_directory;
        config.runtime_directory = runtime_directory;
        config.workers[0].deployment.models_directory = models_directory;
        let inventory = HostInventory {
            effective_cpu_ids: vec![1],
            allocatable_host_memory_bytes: 4096,
            metal_devices: Vec::new(),
            cuda_devices: Vec::new(),
        };
        let binaries = BinaryCatalog::new([(
            WorkerBinaryFlavor::Cpu,
            BinaryRecord {
                path: worker_binary,
                supported_backends: vec![BackendKind::Cpu],
            },
        )]);
        let node = config.validate(&inventory, &binaries).unwrap();
        let slot = WorkerSlot {
            worker_id: id("worker-a"),
            secret_environment_name: "WORKER_TOKEN".into(),
            secret: ResolvedWorkerSecret {
                bearer_token: ServiceBearerToken::new("private-worker-secret").unwrap(),
            },
            restart: RestartController::for_worker(&node, &id("worker-a")).unwrap(),
            process: None,
            process_started_at: None,
            restart_at: Some(Instant::now()),
            exit_observation_failed: false,
        };
        let metrics = SupervisorMetrics {
            launch_attempts: 2,
            readiness_successes: 1,
            readiness_failures: 1,
            restarts_scheduled: 1,
            ..SupervisorMetrics::default()
        };

        let diagnostic = runtime_diagnostic(&node, &[slot], &metrics, Instant::now());

        assert!(diagnostic.len() <= MAX_DIAGNOSTIC_RESPONSE_BYTES);
        assert!(diagnostic.contains("supervisor_status"));
        assert!(diagnostic.contains("launch_attempts_total=2"));
        assert!(diagnostic.contains("worker=worker-a state=restart-pending"));
        assert!(diagnostic.contains("backend=Cpu"));
        assert!(diagnostic.contains("secret=redacted"));
        for secret in ["private-worker-secret", "WORKER_TOKEN", "credential-a"] {
            assert!(!diagnostic.contains(secret));
        }
    }

    #[test]
    fn accelerator_lanes_are_rejected_without_fallback() {
        let assignment = DeviceAssignment::Metal {
            device_id: id::<DeviceId>("metal:1"),
            process_local_device_index: 0,
            shared_memory_limit_bytes: 512,
        };
        assert!(matches!(
            require_cpu_only(&config(assignment, WorkerBinaryFlavor::Metal)),
            Err(SupervisorError::UnsupportedDeviceLane {
                backend: BackendKind::Metal,
                ..
            })
        ));
    }
}
