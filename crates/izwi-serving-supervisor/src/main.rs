use izwi_serving_protocol::{
    BackendKind, DeploymentId, DeviceId, GatewayApprovalsStatus, GatewayWorkerApproval,
    ServiceBearerToken, WorkerId,
};
use izwi_serving_supervisor::actor::{
    bump, count, stop_all, LaunchCounts, LaunchOutcome, Supervision, SupervisorMetrics,
    WorkerActor, WorkerPhase, WorkerView,
};
use izwi_serving_supervisor::autoscale::{
    self, Autoscaler, DeploymentObservation, ScaleDecision, WorkerSignals,
};
use izwi_serving_supervisor::gateway_ack::GatewayAck;
use izwi_serving_supervisor::rollout::{self, RolloutPhase, RolloutStrategy};
use izwi_serving_supervisor::{approvals_file, signals};
use izwi_serving_supervisor::{
    request_quarantine_clear, take_quarantine_clear_requests, BinaryCatalog, BinaryRecord,
    CudaDeviceInventory, HostInventory, LockNamespace, MetalDeviceInventory, NodeConfig,
    QuarantineClear, ResolvedWorkerSecret, ValidatedNodeConfig, WorkerBinaryFlavor, WorkerConfig,
    MAX_NODE_CONFIG_BYTES,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::{OsStr, OsString},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc, watch},
    time::{Instant, MissedTickBehavior},
};
use tracing::{debug, info, warn};

const MAX_CLI_ARGUMENTS: usize = 24;
const MAX_CPU_IDS: usize = 1024;
const MAX_DEVICE_DECLARATIONS: usize = 64;
const MAX_DIAGNOSTIC_RESPONSE_BYTES: usize = 16 * 1024;
const TRUNCATED_DIAGNOSTIC_SUFFIX: &str = "\ndiagnostic_output=truncated\n";
const SUPERVISION_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// How often the supervisor looks for `--clear-quarantine` requests.
const QUARANTINE_CLEAR_CHECK_INTERVAL: Duration = Duration::from_secs(1);
/// A path the supervisor keeps a Prometheus textfile at (for a node
/// exporter's textfile collector); unset writes none.
const METRICS_TEXTFILE_ENV: &str = "IZWI_SUPERVISOR_METRICS_TEXTFILE";

#[tokio::main]
async fn main() -> Result<(), SupervisorError> {
    init_tracing()?;
    let options = match CliOptions::parse(env::args_os().skip(1))? {
        ParseOutcome::Run(options) => options,
        ParseOutcome::Help => {
            print_usage();
            return Ok(());
        }
    };
    run(*options).await
}

/// Supervisor logs go to stderr: text by default, one JSON object per line
/// with `IZWI_LOG_FORMAT=json`. `RUST_LOG` overrides the info-level filter.
fn init_tracing() -> Result<(), SupervisorError> {
    use std::io::IsTerminal;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "izwi_serving_supervisor=info".into());
    let registry = tracing_subscriber::registry().with(filter);
    match env::var("IZWI_LOG_FORMAT").ok().as_deref() {
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

async fn run(options: CliOptions) -> Result<(), SupervisorError> {
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
        izwi_serving_supervisor::env::inherited_environment(),
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

    let metrics_textfile = env::var_os(METRICS_TEXTFILE_ENV).map(PathBuf::from);
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
                .join(izwi_serving_supervisor::QUARANTINE_CLEAR_DIRECTORY),
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
async fn shut_down_all(slots: Vec<WorkerSlot>) {
    stop_all(slots.iter().map(|slot| &slot.actor)).await;
    for slot in slots {
        slot.actor.shut_down().await;
    }
}

/// A prepared DS6 rollout: validated spec, staged state, target config, and
/// idle replacement actors awaiting launch.
struct PreparedRollout {
    spec: rollout::RolloutSpec,
    state: rollout::RolloutState,
    target: Arc<ValidatedNodeConfig>,
    approvals_path: PathBuf,
    /// The replacement lines the rollout adds to the shared approvals file.
    approval_lines: Vec<GatewayWorkerApproval>,
    replacement_secrets: Vec<(WorkerId, ResolvedWorkerSecret)>,
    replacements: Vec<WorkerActor>,
    /// Whether this run resumes persisted state rather than starting fresh.
    resumed: bool,
    /// The gateways to ask before moving on, when the plan names them.
    ack: Option<GatewayAck>,
}

enum RolloutOutcome {
    /// The new generation is admission-eligible and the old generation
    /// drained; replacements join supervision under the target config.
    Committed,
    /// The rollout aborted cleanly; the previous generation never stopped
    /// serving and the replacement workers were stopped.
    Aborted,
    /// Abort could not restore the approvals; everything keeps running and
    /// the state stays non-terminal for a manual `--rollout-abort` retry.
    DegradedAbort,
}

/// Milliseconds since the supervisor started, from the monotonic clock.
fn monotonic_ms(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0)
}

fn prepare_rollout(
    plan_path: &Path,
    current: &ValidatedNodeConfig,
    inventory: &HostInventory,
    binaries: &BinaryCatalog,
    options: &CliOptions,
) -> Result<PreparedRollout, SupervisorError> {
    let plan_bytes = read_bounded(plan_path, rollout::MAX_ROLLOUT_PLAN_BYTES)?;
    let plan = rollout::RolloutPlan::parse_bounded(&plan_bytes)?;
    let target_bytes = read_bounded(plan.target_node_config(), MAX_NODE_CONFIG_BYTES)?;
    let target_config = NodeConfig::parse_bounded(&target_bytes)?;
    ensure_flavor_binaries(&target_config, options)?;
    let target = target_config.validate(inventory, binaries)?;
    let spec = plan.validate(current, &target)?;
    let strategy = rollout::resolve_strategy(
        plan.strategy(),
        current,
        &target,
        &spec,
        inventory,
        binaries,
    )?;

    let runtime_directory = &current.config().runtime_directory;
    let plan_digest = plan.digest(&target_bytes);
    let existing = rollout::load_state(runtime_directory)?;
    let state = match existing {
        Some(state) if !state.phase.is_terminal() => {
            if state.plan_digest != plan_digest {
                return Err(SupervisorError::RolloutStateDigestMismatch);
            }
            info!(
                "resuming rollout in state {} (plan digest matches)",
                state.phase.as_str()
            );
            Some(state)
        }
        Some(state) => {
            rollout::clear_state(runtime_directory)?;
            info!(
                "cleared terminal rollout state ({}) before staging the new rollout",
                state.phase.as_str()
            );
            None
        }
        None => None,
    };

    // Once the old generation is retired its lines are gone from the file,
    // so only a rollout that has not reached that point checks for them.
    if state
        .as_ref()
        .is_none_or(|state| !state.phase.old_generation_retired())
    {
        let approvals = approvals_file::read(plan.shared_approvals_path())
            .map_err(rollout::RolloutError::from)?;
        rollout::verify_approvals_precondition(&approvals, &spec)?;
    }
    let approval_lines = rollout::replacement_approvals(&spec, &target)?;
    let replacement_secrets = resolve_replacement_secrets(&target, &spec)?;
    let ack = match plan.gateway_ack() {
        Some(config) => {
            let admin_key = env::var(&config.admin_key_env).map_err(|_| {
                SupervisorError::MissingGatewayAdminKey {
                    environment: config.admin_key_env.clone(),
                }
            })?;
            Some(GatewayAck::new(config, admin_key).map_err(|error| {
                SupervisorError::InvalidOption(format!("gateway_ack client: {error}"))
            })?)
        }
        None => None,
    };

    let resumed = state.is_some();
    let state = match state {
        Some(state) => state,
        None => {
            let now = now_ms();
            rollout::RolloutState {
                schema_version: rollout::ROLLOUT_STATE_SCHEMA_VERSION,
                plan_digest,
                phase: if strategy == RolloutStrategy::Recreate {
                    rollout::RolloutPhase::RecreateDrainingOld
                } else {
                    rollout::RolloutPhase::LaunchingReplacement
                },
                strategy,
                started_at_ms: now,
                updated_at_ms: now,
                window_deadline_ms: None,
                replacement_workers: Vec::new(),
                shared_approvals_path: plan.shared_approvals_path().to_string_lossy().into_owned(),
                target_node_config: plan.target_node_config().to_string_lossy().into_owned(),
                target_config_digest: rollout::digest_bytes(&target_bytes),
                approval_lines: approval_lines
                    .iter()
                    .map(GatewayWorkerApproval::render_line)
                    .collect(),
                removed_lines: Vec::new(),
            }
        }
    };
    rollout::persist_state(runtime_directory, &state)?;
    info!(
        "rollout prepared: strategy {}, {} rolling deployment(s), canary {}, window {}s, commit grace {}s, abort grace {}s, {}",
        state.strategy.as_str(),
        spec.rolling.len(),
        spec.canary_worker_id,
        spec.window.as_secs(),
        spec.commit_grace.as_secs(),
        spec.abort_grace.as_secs(),
        match &ack {
            Some(ack) => format!("waiting for {} gateway(s) to acknowledge each view", ack.gateways()),
            None => "timed gateway refresh (no gateway_ack)".to_string(),
        }
    );
    Ok(PreparedRollout {
        spec,
        state,
        target: Arc::new(target),
        approvals_path: plan.shared_approvals_path().to_path_buf(),
        approval_lines,
        replacement_secrets,
        replacements: Vec::new(),
        resumed,
        ack,
    })
}

/// A fresh supervisor start without a plan must not silently relaunch the
/// old generation mid-rollout: non-terminal state fails closed, committed
/// state requires the committed target config, aborted state clears.
fn reconcile_fresh_start_state(
    node: &ValidatedNodeConfig,
    config_bytes: &[u8],
) -> Result<(), SupervisorError> {
    let runtime_directory = &node.config().runtime_directory;
    let Some(state) = rollout::load_state(runtime_directory)? else {
        return Ok(());
    };
    if !state.phase.is_terminal() {
        return Err(rollout::RolloutError::RolloutAlreadyInProgress {
            phase: state.phase.as_str().to_string(),
        }
        .into());
    }
    let supplied_digest = rollout::digest_bytes(config_bytes);
    if state.phase == RolloutPhase::Committed && state.target_config_digest != supplied_digest {
        return Err(SupervisorError::RolloutCommittedConfigMismatch {
            recorded: state.target_config_digest,
            actual: supplied_digest,
        });
    }
    rollout::clear_state(runtime_directory)?;
    info!("cleared terminal rollout state ({})", state.phase.as_str());
    Ok(())
}

/// Runs a prepared rollout with its strategy.
async fn execute_rollout(
    prepared: &mut PreparedRollout,
    node: &ValidatedNodeConfig,
    current_slots: &mut Vec<WorkerSlot>,
    shutdown: &mut watch::Receiver<bool>,
    abort_requested: &mut watch::Receiver<bool>,
) -> RolloutOutcome {
    match prepared.state.strategy {
        RolloutStrategy::BlueGreen => {
            execute_blue_green(prepared, node, current_slots, shutdown, abort_requested).await
        }
        RolloutStrategy::Recreate => {
            execute_recreate(prepared, node, current_slots, shutdown, abort_requested).await
        }
    }
}

enum ReplacementLaunch {
    Ready,
    Failed(WorkerId),
    Interrupted(&'static str),
}

/// Launches the replacements: the canary first, then the rest together.
async fn launch_replacements(
    prepared: &mut PreparedRollout,
    shutdown: &mut watch::Receiver<bool>,
    abort_requested: &mut watch::Receiver<bool>,
) -> ReplacementLaunch {
    let canary = prepared.spec.canary_worker_id.clone();
    prepared
        .replacements
        .sort_by_key(|actor| actor.worker_id().as_str() != canary.as_str());
    let (first, rest) = prepared.replacements.split_at(1);
    for wave in [first, rest] {
        let waiting: Vec<_> = wave
            .iter()
            .map(|actor| {
                let before = actor.view().launch_counts();
                actor.start();
                (before, actor)
            })
            .collect();
        for (before, actor) in waiting {
            tokio::select! {
                outcome = actor.launch_outcome(before) => {
                    if outcome == LaunchOutcome::Failed {
                        return ReplacementLaunch::Failed(actor.worker_id().clone());
                    }
                }
                _ = shutdown.wait_for(|stop| *stop) => {
                    return ReplacementLaunch::Interrupted("shutdown requested");
                }
                _ = abort_requested.wait_for(|abort| *abort) => {
                    return ReplacementLaunch::Interrupted("operator abort (SIGUSR2)");
                }
            }
        }
    }
    prepared.state.replacement_workers = prepared
        .replacements
        .iter()
        .map(|actor| rollout::RolloutWorkerRecord {
            worker_id: actor.worker_id().to_string(),
            pid: actor.view().process_id.unwrap_or(0),
        })
        .collect();
    prepared.state.updated_at_ms = now_ms();
    ReplacementLaunch::Ready
}

/// Waits until every gateway acknowledges a view; the error names why not.
async fn await_ack(
    ack: &GatewayAck,
    adopted: impl Fn(&GatewayApprovalsStatus) -> bool,
    shutdown: &mut watch::Receiver<bool>,
    abort_requested: &mut watch::Receiver<bool>,
) -> Result<(), &'static str> {
    tokio::select! {
        result = ack.wait_for(adopted) => result.map_err(|gateways| {
            warn!("gateways {gateways:?} did not acknowledge the approvals view in time");
            "the gateways did not acknowledge the approvals view"
        }),
        _ = shutdown.wait_for(|stop| *stop) => Err("shutdown requested"),
        _ = abort_requested.wait_for(|abort| *abort) => Err("operator abort (SIGUSR2)"),
    }
}

/// Persists rollout state, logging a failure.
fn persist(runtime_directory: &Path, state: &rollout::RolloutState) -> bool {
    match rollout::persist_state(runtime_directory, state) {
        Ok(()) => true,
        Err(error) => {
            warn!("rollout state persist failed: {error}");
            false
        }
    }
}

async fn execute_blue_green(
    prepared: &mut PreparedRollout,
    node: &ValidatedNodeConfig,
    current_slots: &mut Vec<WorkerSlot>,
    shutdown: &mut watch::Receiver<bool>,
    abort_requested: &mut watch::Receiver<bool>,
) -> RolloutOutcome {
    let runtime_directory = node.config().runtime_directory.clone();
    // On window resume the crashed supervisor's replacements self-drained,
    // so they relaunch here too.
    if matches!(
        prepared.state.phase,
        RolloutPhase::LaunchingReplacement | RolloutPhase::WindowOpen
    ) {
        match launch_replacements(prepared, shutdown, abort_requested).await {
            ReplacementLaunch::Ready => {
                persist(&runtime_directory, &prepared.state);
            }
            ReplacementLaunch::Failed(worker_id) => {
                warn!("replacement worker {worker_id} failed to reach readiness; aborting rollout");
                return abort_rollout(
                    prepared,
                    &runtime_directory,
                    current_slots,
                    AbortMode::KeepOld,
                    "replacement readiness failed",
                )
                .await;
            }
            ReplacementLaunch::Interrupted(reason) => {
                return abort_rollout(
                    prepared,
                    &runtime_directory,
                    current_slots,
                    AbortMode::KeepOld,
                    reason,
                )
                .await;
            }
        }
    }

    // Open (or re-open on resume) the window: both generations are approved,
    // the gateway cuts over atomically when the successor first observes
    // Ready. The old generation never stops serving during the window.
    if let Err(error) = approvals_file::edit(&prepared.approvals_path, |document| {
        rollout::apply_window_view(document, &prepared.approval_lines);
        Ok::<_, rollout::RolloutError>(())
    }) {
        warn!("rollout window view failed to write: {error}; nothing changed");
        return RolloutOutcome::DegradedAbort;
    }
    prepared.state.phase = RolloutPhase::WindowOpen;
    prepared.state.updated_at_ms = now_ms();
    persist(&runtime_directory, &prepared.state);
    // With gateway acknowledgement the soak starts once every gateway
    // reports the window view adopted and cut over to the successor.
    if let Some(ack) = prepared.ack.as_ref() {
        info!(
            "rollout window open: waiting for {} gateway(s) to adopt the window view and cut over",
            ack.gateways()
        );
        let (spec, lines) = (&prepared.spec, &prepared.approval_lines);
        let acknowledged = await_ack(
            ack,
            |status| rollout::window_adopted(status, spec, lines),
            shutdown,
            abort_requested,
        )
        .await;
        if let Err(reason) = acknowledged {
            return abort_rollout(
                prepared,
                &runtime_directory,
                current_slots,
                AbortMode::KeepOldAfterGrace,
                reason,
            )
            .await;
        }
        info!("rollout window: every gateway adopted the window view and cut over");
    }
    // The window always runs in full from now on the monotonic clock: a
    // resumed rollout re-arms it, so the gateways adopt the window view
    // again before the commit view lands.
    let window_end = Instant::now() + prepared.spec.window;
    prepared.state.window_deadline_ms =
        Some(now_ms().saturating_add(prepared.spec.window.as_millis() as u64));
    persist(&runtime_directory, &prepared.state);
    info!(
        "rollout window: both generations approved; cutover fires when the successor generation is observed Ready; commit in {}s",
        prepared.spec.window.as_secs()
    );
    let mut window_watch = tokio::time::interval(Duration::from_millis(500));
    window_watch.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            () = tokio::time::sleep_until(window_end) => break,
            _ = window_watch.tick() => {}
            _ = abort_requested.changed() => {}
            _ = shutdown.changed() => {}
        }
        let reason = if *shutdown.borrow() {
            Some("shutdown requested")
        } else if *abort_requested.borrow() {
            Some("operator abort (SIGUSR2)")
        } else if let Some(actor) = prepared
            .replacements
            .iter()
            .find(|actor| actor.view().phase != WorkerPhase::Ready)
        {
            warn!(
                "replacement worker {} left readiness during the rollout window; aborting rollout",
                actor.worker_id()
            );
            Some("replacement exited during the window")
        } else {
            None
        };
        if let Some(reason) = reason {
            return abort_rollout(
                prepared,
                &runtime_directory,
                current_slots,
                AbortMode::KeepOldAfterGrace,
                reason,
            )
            .await;
        }
    }

    // The commit view removes the old generation's lines; the lines it
    // removed are recorded so an abort before the drain can put them back.
    info!("rollout commit: writing the commit approval view");
    let mut removed = Vec::new();
    if let Err(error) = approvals_file::edit(&prepared.approvals_path, |document| {
        removed = rollout::apply_commit_view(document, &prepared.spec, &prepared.approval_lines);
        Ok::<_, rollout::RolloutError>(())
    }) {
        warn!("rollout commit view failed to write: {error}; the window view stays in place");
        return RolloutOutcome::DegradedAbort;
    }
    if !removed.is_empty() {
        prepared.state.removed_lines = removed;
    }
    prepared.state.window_deadline_ms = None;
    prepared.state.updated_at_ms = now_ms();
    persist(&runtime_directory, &prepared.state);
    // The old generation drains only once no gateway routes to it: with
    // acknowledgement, once every gateway reports the commit view adopted
    // (not in time, the rollout aborts and the old generation never stopped
    // serving); otherwise after the commit grace. Shutdown drains everything
    // anyway, so it ends the wait.
    match prepared.ack.as_ref() {
        Some(ack) => {
            let spec = &prepared.spec;
            let acknowledged = await_ack(
                ack,
                |status| rollout::old_generation_unapproved(status, spec),
                shutdown,
                abort_requested,
            )
            .await;
            if let Err(reason) = acknowledged {
                if !*shutdown.borrow() {
                    return abort_rollout(
                        prepared,
                        &runtime_directory,
                        current_slots,
                        AbortMode::KeepOldAfterGrace,
                        reason,
                    )
                    .await;
                }
            }
        }
        None if !prepared.spec.commit_grace.is_zero() => {
            info!(
                "waiting {}s for gateway approval refresh before draining the old generation",
                prepared.spec.commit_grace.as_secs()
            );
            tokio::select! {
                () = tokio::time::sleep(prepared.spec.commit_grace) => {}
                _ = shutdown.wait_for(|stop| *stop) => {}
            }
        }
        None => {}
    }
    // DrainingOld is the point of no return: the old generation's workers
    // stop admitting. It is entered only once persisted, so a crash can
    // never resume an earlier phase and relaunch the drained generation. On
    // resume into this phase the old workers already exited (the
    // generation barrier proved it), so the drain finds nothing to stop.
    prepared.state.phase = RolloutPhase::DrainingOld;
    prepared.state.updated_at_ms = now_ms();
    if !persist(&runtime_directory, &prepared.state) {
        prepared.state.phase = RolloutPhase::WindowOpen;
        return abort_rollout(
            prepared,
            &runtime_directory,
            current_slots,
            AbortMode::KeepOldAfterGrace,
            "the draining_old state could not be persisted",
        )
        .await;
    }
    info!("rollout drain: draining the old generation (point of no return)");
    let (old, kept): (Vec<WorkerSlot>, Vec<WorkerSlot>) = std::mem::take(current_slots)
        .into_iter()
        .partition(|slot| prepared.spec.is_old_worker(slot.worker_id().as_str()));
    *current_slots = kept;
    shut_down_all(old).await;
    prepared.state.phase = RolloutPhase::Committed;
    prepared.state.updated_at_ms = now_ms();
    if let Err(error) = rollout::persist_state(&runtime_directory, &prepared.state) {
        warn!("rollout state persist failed: {error}");
    }
    info!(
        "rollout committed: the target config's generations are admission-eligible; repoint the service definition at the target node config"
    );
    RolloutOutcome::Committed
}

async fn execute_recreate(
    prepared: &mut PreparedRollout,
    node: &ValidatedNodeConfig,
    current_slots: &mut Vec<WorkerSlot>,
    shutdown: &mut watch::Receiver<bool>,
    abort_requested: &mut watch::Receiver<bool>,
) -> RolloutOutcome {
    let runtime_directory = node.config().runtime_directory.clone();
    if prepared.state.phase == RolloutPhase::RecreateDrainingOld {
        // Cut the gateways over first: the old generation's lines go and the
        // replacements' lines come, adopted once the replacements answer.
        // The deployment is unavailable from the drain until they are ready.
        let mut removed = Vec::new();
        if let Err(error) = approvals_file::edit(&prepared.approvals_path, |document| {
            removed =
                rollout::apply_commit_view(document, &prepared.spec, &prepared.approval_lines);
            Ok::<_, rollout::RolloutError>(())
        }) {
            warn!("rollout commit view failed to write: {error}");
            return abort_rollout(
                prepared,
                &runtime_directory,
                current_slots,
                AbortMode::KeepOld,
                "the commit view could not be written",
            )
            .await;
        }
        // A resumed rollout finds the lines already gone; keep the record.
        if !removed.is_empty() {
            prepared.state.removed_lines = removed;
        }
        prepared.state.updated_at_ms = now_ms();
        if !persist(&runtime_directory, &prepared.state) {
            return abort_rollout(
                prepared,
                &runtime_directory,
                current_slots,
                AbortMode::KeepOld,
                "the commit view could not be recorded",
            )
            .await;
        }
        info!(
            "rollout (recreate): commit view written; the old generation drains once the gateways stop routing to it, and the deployment is unavailable until the replacements are ready"
        );
        match prepared.ack.as_ref() {
            Some(ack) => {
                let spec = &prepared.spec;
                let acknowledged = await_ack(
                    ack,
                    |status| rollout::old_generation_unapproved(status, spec),
                    shutdown,
                    abort_requested,
                )
                .await;
                if let Err(reason) = acknowledged {
                    return abort_rollout(
                        prepared,
                        &runtime_directory,
                        current_slots,
                        AbortMode::KeepOld,
                        reason,
                    )
                    .await;
                }
            }
            None if !prepared.spec.commit_grace.is_zero() => {
                tokio::select! {
                    () = tokio::time::sleep(prepared.spec.commit_grace) => {}
                    _ = shutdown.wait_for(|stop| *stop) => {}
                }
            }
            None => {}
        }
        stop_all(
            current_slots
                .iter()
                .filter(|slot| prepared.spec.is_old_worker(slot.worker_id().as_str()))
                .map(|slot| &slot.actor),
        )
        .await;
        prepared.state.phase = RolloutPhase::RecreateLaunching;
        prepared.state.updated_at_ms = now_ms();
        persist(&runtime_directory, &prepared.state);
    }

    match launch_replacements(prepared, shutdown, abort_requested).await {
        ReplacementLaunch::Ready => {}
        ReplacementLaunch::Failed(worker_id) => {
            warn!("replacement worker {worker_id} failed to reach readiness; rolling back to the old generation");
            return abort_rollout(
                prepared,
                &runtime_directory,
                current_slots,
                AbortMode::RelaunchOld,
                "replacement readiness failed",
            )
            .await;
        }
        ReplacementLaunch::Interrupted(reason) => {
            let mode = if *shutdown.borrow() {
                AbortMode::KeepOld
            } else {
                AbortMode::RelaunchOld
            };
            return abort_rollout(prepared, &runtime_directory, current_slots, mode, reason).await;
        }
    }
    let (old, kept): (Vec<WorkerSlot>, Vec<WorkerSlot>) = std::mem::take(current_slots)
        .into_iter()
        .partition(|slot| prepared.spec.is_old_worker(slot.worker_id().as_str()));
    *current_slots = kept;
    shut_down_all(old).await;
    prepared.state.phase = RolloutPhase::Committed;
    prepared.state.updated_at_ms = now_ms();
    persist(&runtime_directory, &prepared.state);
    info!(
        "rollout committed (recreate): the replacements are ready and approved; repoint the service definition at the target node config"
    );
    RolloutOutcome::Committed
}

/// What an abort does with the old generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbortMode {
    /// The old generation is still serving; stop the replacements at once.
    KeepOld,
    /// The old generation is still serving; give the gateways the abort
    /// grace to stop routing to the replacements before stopping them.
    KeepOldAfterGrace,
    /// A recreate stopped the old generation: start it again.
    RelaunchOld,
}

/// Aborts a rollout: restore the approval view first (remove the
/// replacement lines, put back any old lines the rollout removed) so the
/// gateway stops routing to the replacements within one refresh TTL, then
/// stop the replacements, and relaunch the old generation if a recreate had
/// stopped it.
async fn abort_rollout(
    prepared: &mut PreparedRollout,
    runtime_directory: &Path,
    current_slots: &[WorkerSlot],
    mode: AbortMode,
    reason: &str,
) -> RolloutOutcome {
    warn!("rollout aborted ({reason}): restoring the previous approval view");
    if let Err(error) = approvals_file::edit(&prepared.approvals_path, |document| {
        rollout::restore_view(
            document,
            &prepared.approval_lines,
            &prepared.state.removed_lines,
        );
        Ok::<_, rollout::RolloutError>(())
    }) {
        warn!(
            "rollout abort could not restore approvals: {error}; workers keep running and the state stays resumable"
        );
        return RolloutOutcome::DegradedAbort;
    }
    if mode == AbortMode::KeepOldAfterGrace && !prepared.spec.abort_grace.is_zero() {
        info!(
            "waiting {}s for gateway approval refresh before stopping replacement workers",
            prepared.spec.abort_grace.as_secs()
        );
        tokio::time::sleep(prepared.spec.abort_grace).await;
    }
    stop_all(&prepared.replacements).await;
    for actor in prepared.replacements.drain(..) {
        actor.shut_down().await;
    }
    if mode == AbortMode::RelaunchOld {
        for slot in current_slots
            .iter()
            .filter(|slot| prepared.spec.is_old_worker(slot.worker_id().as_str()))
        {
            slot.actor.start();
        }
    }
    prepared.state.phase = RolloutPhase::Aborted;
    prepared.state.updated_at_ms = now_ms();
    persist(runtime_directory, &prepared.state);
    let _ = rollout::clear_state(runtime_directory);
    info!(
        "rollout abort complete: the previous approval view is restored and the replacement workers are stopped"
    );
    RolloutOutcome::Aborted
}

fn rollout_abort_command(node: &ValidatedNodeConfig) -> Result<(), SupervisorError> {
    let runtime_directory = &node.config().runtime_directory;
    let Some(state) = rollout::load_state(runtime_directory)? else {
        eprintln!(
            "no rollout state found in {}; nothing to abort",
            runtime_directory.display()
        );
        return Err(SupervisorError::RolloutNothingToAbort);
    };
    if state.phase.is_terminal() {
        rollout::clear_state(runtime_directory)?;
        eprintln!("cleared terminal rollout state ({})", state.phase.as_str());
        return Ok(());
    }
    let locks = LockNamespace::open(runtime_directory)?;
    let lease_metadata = format!("rollout-abort pid={}", std::process::id());
    if locks
        .try_node_supervisor(lease_metadata.as_bytes())
        .is_err()
    {
        return Err(SupervisorError::RolloutSupervisorBusy);
    }
    if locks
        .try_generation_barrier(lease_metadata.as_bytes())
        .is_err()
    {
        return Err(SupervisorError::RolloutFenceContended);
    }
    let added = rollout::parse_approval_lines(&state.approval_lines)?;
    approvals_file::edit(Path::new(&state.shared_approvals_path), |document| {
        rollout::restore_view(document, &added, &state.removed_lines);
        Ok::<_, rollout::RolloutError>(())
    })?;
    rollout::clear_state(runtime_directory)?;
    if state.phase.old_generation_retired() {
        eprintln!(
            "rollout aborted: the rollout's approval edits are undone; start the supervisor with the current config to serve the previous generation again"
        );
    } else {
        eprintln!(
            "rollout aborted: the replacement approval lines are removed; the previous generation never stopped serving"
        );
    }
    Ok(())
}

fn print_rollout_status(config: &NodeConfig) -> Result<(), SupervisorError> {
    let runtime_directory = &config.runtime_directory;
    match rollout::load_state(runtime_directory)? {
        Some(state) => {
            eprintln!(
                "rollout_state={} node={}",
                state.phase.as_str(),
                config.node_id
            );
            eprintln!(
                "plan_digest={} target_node_config={}",
                state.plan_digest, state.target_node_config
            );
            eprintln!("shared_approvals_path={}", state.shared_approvals_path);
            if let Some(deadline) = state.window_deadline_ms {
                eprintln!(
                    "window_deadline_ms={deadline} (remaining {}s)",
                    deadline.saturating_sub(now_ms()) / 1000
                );
            }
            for worker in &state.replacement_workers {
                eprintln!("replacement_worker={} pid={}", worker.worker_id, worker.pid);
            }
            if state.phase.is_terminal() {
                eprintln!(
                    "note: terminal rollout state clears automatically on the next supervisor start or via --rollout-abort"
                );
            }
            Ok(())
        }
        None => {
            eprintln!("no rollout state in {}", runtime_directory.display());
            Ok(())
        }
    }
}

fn resolve_replacement_secrets(
    target: &ValidatedNodeConfig,
    spec: &rollout::RolloutSpec,
) -> Result<Vec<(WorkerId, ResolvedWorkerSecret)>, SupervisorError> {
    resolve_secrets(
        target
            .config()
            .workers
            .iter()
            .filter(|worker| spec.rolling_for(worker.worker_id.as_str()).is_some()),
    )
}

/// Lifecycle phase of an autoscaling slot (DS7). Core (min-set) workers keep
/// `autoscale: None` and behave exactly like static workers; only standby
/// replicas and scale-event transitions carry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoscalePhase {
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
struct AutoscaleSlotState {
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
struct WorkerSlot {
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
fn resolve_secrets<'a>(
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
    if let Some(autoscaling) = node.config().autoscaling.as_ref() {
        output.push_line(&format!(
            "autoscaling enabled deployments={} evaluation_interval_ms={} shared_approvals={}",
            autoscaling.deployments.len(),
            autoscaling.evaluation_interval_ms,
            autoscaling.shared_approvals_path.display(),
        ));
        for (deployment, policy) in &autoscaling.deployments {
            output.push_line(&format!(
                "autoscale deployment={deployment} min_workers={} max_workers={} scale_up_queue_depth={} scale_up_sustained_polls={} scale_down_stabilization_window_ms={}",
                policy.min_workers,
                policy.max_workers,
                policy.scale_up_queue_depth,
                policy.scale_up_sustained_polls,
                policy.scale_down_stabilization_window_ms,
            ));
        }
    }
    output.finish()
}

/// The diagnostic state name of one worker.
fn slot_state(view: &WorkerView, autoscale: Option<&AutoscaleSlotState>) -> &'static str {
    match (autoscale.map(|state| state.phase), view.phase) {
        (Some(AutoscalePhase::Admitting { .. }), _) => "admitting",
        (_, WorkerPhase::Draining) | (Some(AutoscalePhase::Draining { .. }), _) => "draining",
        (_, WorkerPhase::Ready) if view.exit_observation_failed => "observation-uncertain",
        (_, WorkerPhase::Ready) => "ready",
        (_, WorkerPhase::WaitingForLoad | WorkerPhase::Starting) => "starting",
        (_, WorkerPhase::RestartPending { .. }) => "restart-pending",
        (_, WorkerPhase::Quarantined { .. }) => "quarantined",
        (Some(AutoscalePhase::Standby), WorkerPhase::Idle) => "standby",
        (_, WorkerPhase::Idle) => "stopped",
    }
}

fn runtime_diagnostic(
    node: &ValidatedNodeConfig,
    slots: &[WorkerSlot],
    supervision: &Supervision,
) -> String {
    let views: Vec<WorkerView> = slots.iter().map(|slot| slot.actor.view()).collect();
    render_runtime_diagnostic(
        node,
        slots.iter().zip(&views).map(|(slot, view)| SlotSnapshot {
            worker_id: slot.worker_id(),
            config: slot
                .actor
                .node()
                .worker(slot.worker_id())
                .expect("actors are spawned for configured workers"),
            view,
            autoscale: slot.autoscale.as_ref(),
        }),
        &supervision.metrics,
        supervision.started_at,
    )
}

/// What diagnostics read about one worker.
struct SlotSnapshot<'a> {
    worker_id: &'a WorkerId,
    config: &'a WorkerConfig,
    view: &'a WorkerView,
    autoscale: Option<&'a AutoscaleSlotState>,
}

fn render_runtime_diagnostic<'a>(
    node: &ValidatedNodeConfig,
    slots: impl Iterator<Item = SlotSnapshot<'a>>,
    metrics: &SupervisorMetrics,
    started_at: Instant,
) -> String {
    let slots: Vec<SlotSnapshot<'a>> = slots.collect();
    let mut output = BoundedDiagnostic::new();
    let in_phase = |predicate: &dyn Fn(&WorkerView) -> bool| {
        slots.iter().filter(|slot| predicate(slot.view)).count()
    };
    output.push_line(&format!(
        "supervisor_status version={} node={} uptime_ms={} workers={} running={} restart_pending={} quarantined={} launch_attempts_total={} readiness_successes_total={} readiness_failures_total={} restarts_scheduled_total={} quarantines_total={} unexpected_exits_total={} exit_observation_failures_total={} liveness_restarts_total={} stop_failures_total={} autoscale_ups_total={} autoscale_downs_total={}",
        env!("CARGO_PKG_VERSION"),
        node.config().node_id,
        started_at.elapsed().as_millis(),
        slots.len(),
        in_phase(&WorkerView::is_running),
        in_phase(&|view| matches!(view.phase, WorkerPhase::RestartPending { .. })),
        in_phase(&|view| matches!(view.phase, WorkerPhase::Quarantined { .. })),
        count(&metrics.launch_attempts),
        count(&metrics.readiness_successes),
        count(&metrics.readiness_failures),
        count(&metrics.restarts_scheduled),
        count(&metrics.quarantines),
        count(&metrics.unexpected_exits),
        count(&metrics.exit_observation_failures),
        count(&metrics.liveness_restarts),
        count(&metrics.stop_failures),
        count(&metrics.autoscale_ups),
        count(&metrics.autoscale_downs),
    ));
    let mut autoscale_summary: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for slot in &slots {
        if let Some(state) = slot.autoscale {
            let entry = autoscale_summary
                .entry(state.deployment.to_string())
                .or_default();
            match state.phase {
                AutoscalePhase::Standby => entry[1] += 1,
                AutoscalePhase::Admitting { .. } | AutoscalePhase::Active => entry[0] += 1,
                AutoscalePhase::Draining { .. } => entry[2] += 1,
            }
        }
    }
    for (deployment, [running_capacity, standby, draining]) in autoscale_summary {
        output.push_line(&format!(
            "autoscale deployment={deployment} running={running_capacity} standby={standby} draining={draining}"
        ));
    }
    for slot in &slots {
        let running = slot.view.is_running();
        output.push_line(&format!(
            "worker={} state={} pid={} incarnation={} assignment_source={} backend={:?} assignment={:?} deployment={} generation={} secret=redacted",
            slot.worker_id,
            slot_state(slot.view, slot.autoscale),
            slot.view
                .process_id
                .map_or_else(|| "none".to_string(), |id| id.to_string()),
            slot.view
                .incarnation
                .as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            if running { "verified" } else { "configured" },
            slot.config.assignment.backend(),
            slot.config.assignment,
            slot.config.deployment.deployment_id,
            slot.config.deployment.model_generation.get(),
        ));
    }
    output.finish()
}

/// Supervisor counters and worker-state gauges in the Prometheus text format.
/// Only events change it, so it is rewritten only when it differs.
fn supervisor_prometheus(slots: &[WorkerSlot], metrics: &SupervisorMetrics) -> String {
    use std::fmt::Write as _;
    let views: Vec<WorkerView> = slots.iter().map(|slot| slot.actor.view()).collect();
    let mut output = String::new();
    let mut metric = |name: &str, kind: &str, help: &str, value: u64| {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} {kind}");
        let _ = writeln!(output, "{name} {value}");
    };
    let in_phase = |predicate: &dyn Fn(&WorkerView) -> bool| {
        views.iter().filter(|view| predicate(view)).count() as u64
    };
    metric(
        "izwi_supervisor_workers",
        "gauge",
        "Configured worker slots.",
        views.len() as u64,
    );
    metric(
        "izwi_supervisor_workers_running",
        "gauge",
        "Worker slots with a ready process.",
        in_phase(&WorkerView::is_running),
    );
    metric(
        "izwi_supervisor_workers_quarantined",
        "gauge",
        "Worker slots quarantined after exhausting their restart budget.",
        in_phase(&|view| matches!(view.phase, WorkerPhase::Quarantined { .. })),
    );
    metric(
        "izwi_supervisor_workers_restart_pending",
        "gauge",
        "Worker slots waiting for a scheduled restart.",
        in_phase(&|view| matches!(view.phase, WorkerPhase::RestartPending { .. })),
    );
    for (name, help, value) in [
        (
            "izwi_supervisor_launch_attempts_total",
            "Worker launch attempts.",
            &metrics.launch_attempts,
        ),
        (
            "izwi_supervisor_readiness_failures_total",
            "Launches that failed readiness.",
            &metrics.readiness_failures,
        ),
        (
            "izwi_supervisor_restarts_scheduled_total",
            "Worker restarts scheduled.",
            &metrics.restarts_scheduled,
        ),
        (
            "izwi_supervisor_quarantines_total",
            "Workers quarantined.",
            &metrics.quarantines,
        ),
        (
            "izwi_supervisor_unexpected_exits_total",
            "Workers that exited unexpectedly.",
            &metrics.unexpected_exits,
        ),
        (
            "izwi_supervisor_liveness_restarts_total",
            "Workers restarted after failing their liveness probes.",
            &metrics.liveness_restarts,
        ),
        (
            "izwi_supervisor_stop_failures_total",
            "Worker stops that failed.",
            &metrics.stop_failures,
        ),
        (
            "izwi_supervisor_autoscale_ups_total",
            "Autoscaling scale-ups.",
            &metrics.autoscale_ups,
        ),
        (
            "izwi_supervisor_autoscale_downs_total",
            "Autoscaling scale-downs.",
            &metrics.autoscale_downs,
        ),
        (
            "izwi_supervisor_rollouts_committed_total",
            "Rollouts committed.",
            &metrics.rollouts_committed,
        ),
        (
            "izwi_supervisor_rollouts_aborted_total",
            "Rollouts aborted cleanly.",
            &metrics.rollouts_aborted,
        ),
        (
            "izwi_supervisor_rollouts_degraded_total",
            "Rollout aborts that could not restore the approvals.",
            &metrics.rollouts_degraded,
        ),
    ] {
        metric(name, "counter", help, count(value));
    }
    output
}

/// Replace `path` atomically, so a collector never reads a partial file.
fn write_metrics_textfile(path: &Path, contents: &str) -> io::Result<()> {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, contents)?;
    std::fs::rename(&temporary, path)
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

/// Keeps the autoscaler's running set in step with the actors: a crash of a
/// restart-managed worker leaves it, a relaunch that reached readiness
/// rejoins it, and a scale-down drain that ended returns its slot to the
/// standby pool.
fn sync_autoscaler(autoscaler: &mut Autoscaler, slots: &mut [WorkerSlot]) {
    for slot in slots.iter_mut() {
        let view = slot.actor.view();
        let worker = slot.worker_id().clone();
        let Some(state) = autoscaler.deployment_of_mut(&worker) else {
            continue;
        };
        if slot.restart_managed() {
            if view.ready_launches > slot.seen_ready_launches && view.phase == WorkerPhase::Ready {
                state.note_launched(&worker);
            } else if slot.was_running && !view.is_running() {
                state.note_worker_lost(&worker);
            }
        }
        slot.seen_ready_launches = view.ready_launches;
        slot.was_running = view.is_running();
        if let Some(autoscale) = slot.autoscale.as_mut() {
            if matches!(autoscale.phase, AutoscalePhase::Draining { .. })
                && view.phase == WorkerPhase::Idle
            {
                autoscale.phase = AutoscalePhase::Standby;
                state.note_stopped();
                info!("autoscaling: draining worker {worker} exited cleanly; slot returned to standby");
            }
        }
    }
}

/// One DS7 autoscaling evaluation pass: complete pending admissions, poll
/// the running workers of each deployment, then act on at most one scale
/// decision per deployment. Every step is conservative — a failed approvals
/// write, a failed status poll, or a launch in flight freezes that
/// deployment's decisions until the state is consistent again. Nothing here
/// waits for a launch: the actors run them.
async fn run_autoscale_tick(
    autoscaler: &mut Autoscaler,
    node: &ValidatedNodeConfig,
    slots: &mut [WorkerSlot],
    supervision: &Supervision,
) {
    let Some(autoscaling) = node.config().autoscaling.as_ref() else {
        return;
    };
    let approvals_path = autoscaling.shared_approvals_path.clone();
    let metrics = &supervision.metrics;
    let deployment_ids: Vec<DeploymentId> = autoscaler.deployments().keys().cloned().collect();
    for deployment_id in deployment_ids {
        // Monotonic scale-event time: a wall-clock step or a desktop sleep
        // must not open or close a stabilization window.
        let now_ms = monotonic_ms(supervision.started_at);
        if !admit_pending_workers(
            autoscaler,
            &approvals_path,
            node,
            &deployment_id,
            slots,
            now_ms,
            metrics,
        ) {
            continue;
        }
        let Some(observation) = poll_deployment(
            autoscaler,
            slots,
            &deployment_id,
            autoscaler.evaluation_interval(),
        )
        .await
        else {
            continue;
        };
        let now_ms = monotonic_ms(supervision.started_at);
        let decision = autoscaler
            .deployment_mut(&deployment_id)
            .expect("deployment ids come from the autoscaler")
            .evaluate(now_ms, &observation);
        let Some(slot) = (match &decision {
            ScaleDecision::NoAction => None,
            ScaleDecision::ScaleUp { worker }
            | ScaleDecision::MarkDraining { worker }
            | ScaleDecision::StopDrained { worker } => {
                slots.iter_mut().find(|slot| slot.worker_id() == worker)
            }
        }) else {
            continue;
        };
        match decision {
            ScaleDecision::NoAction => {}
            ScaleDecision::ScaleUp { worker } => {
                info!(
                    "autoscaling: deployment {deployment_id} scaled up; launching standby {worker}"
                );
                slot.autoscale = Some(AutoscaleSlotState {
                    deployment: deployment_id.clone(),
                    phase: AutoscalePhase::Admitting {
                        since: slot.actor.view().launch_counts(),
                    },
                });
                slot.actor.start();
            }
            ScaleDecision::MarkDraining { worker } => {
                // Remove the worker's approvals line (the gateway stops
                // routing new work after its view refresh) and close its
                // ownership pipe so it stops admission at once, covering the
                // gateway's refresh gap.
                if let Err(error) = autoscale::remove_worker(&approvals_path, node, &worker) {
                    warn!(
                        "autoscaling: could not unapprove worker {worker}: {error}; retrying next evaluation"
                    );
                    continue;
                }
                slot.actor.begin_drain();
                slot.autoscale = Some(AutoscaleSlotState {
                    deployment: deployment_id.clone(),
                    phase: AutoscalePhase::Draining {
                        stop_requested: false,
                    },
                });
                autoscaler
                    .deployment_mut(&deployment_id)
                    .expect("deployment ids come from the autoscaler")
                    .note_draining(now_ms, &worker);
                bump(&metrics.autoscale_downs);
                info!(
                    "autoscaling: worker {worker} marked draining; waiting for zero active work before stopping it"
                );
            }
            ScaleDecision::StopDrained { worker } => {
                // The worker is fully quiet: stop it. Its slot returns to the
                // standby pool once the actor reports it gone.
                if let Some(AutoscaleSlotState {
                    phase: AutoscalePhase::Draining { stop_requested },
                    ..
                }) = slot.autoscale.as_mut()
                {
                    if !*stop_requested {
                        *stop_requested = true;
                        drop(slot.actor.request_stop());
                        info!("autoscaling: worker {worker} drained; stopping it");
                    }
                }
            }
        }
    }
}

/// Publishes approvals lines for scale-ups whose launch reached readiness,
/// and returns failed scale-ups to the standby pool. Returns false while a
/// scale-up is still launching or a publication failed, so the deployment
/// defers further decisions.
fn admit_pending_workers(
    autoscaler: &mut Autoscaler,
    approvals_path: &Path,
    node: &ValidatedNodeConfig,
    deployment_id: &DeploymentId,
    slots: &mut [WorkerSlot],
    now_ms: u64,
    metrics: &SupervisorMetrics,
) -> bool {
    let mut settled = true;
    for slot in slots.iter_mut() {
        let Some(AutoscaleSlotState {
            deployment,
            phase: AutoscalePhase::Admitting { since },
        }) = slot.autoscale.clone()
        else {
            continue;
        };
        if &deployment != deployment_id {
            continue;
        }
        let worker = slot.worker_id().clone();
        let view = slot.actor.view();
        if view.ready_launches > since.ready && view.phase == WorkerPhase::Ready {
            match autoscale::add_worker(approvals_path, node, &worker) {
                Ok(()) => {
                    slot.autoscale = Some(AutoscaleSlotState {
                        deployment,
                        phase: AutoscalePhase::Active,
                    });
                    slot.seen_ready_launches = view.ready_launches;
                    slot.was_running = true;
                    if let Some(state) = autoscaler.deployment_mut(deployment_id) {
                        state.note_scaled_up(now_ms, &worker);
                    }
                    bump(&metrics.autoscale_ups);
                    info!("autoscaling: worker {worker} is ready and approved; scale-up complete");
                }
                Err(error) => {
                    warn!(
                        "autoscaling: approvals publish for worker {worker} failed: {error}; retrying next evaluation"
                    );
                    settled = false;
                }
            }
        } else if view.failed_launches > since.failed {
            // Cancel the restart the actor scheduled: a standby only runs
            // on a scale-up decision.
            drop(slot.actor.request_stop());
            slot.autoscale = Some(AutoscaleSlotState::standby(deployment));
            if let Some(state) = autoscaler.deployment_mut(deployment_id) {
                state.note_launch_abandoned(&worker);
            }
            warn!("autoscaling: standby {worker} failed its launch; it remains standby");
        } else {
            settled = false;
        }
    }
    settled
}

/// Polls the status of every running worker of a deployment. `None` means
/// the deployment is not fully observable this tick (a restart pending, a
/// failed or timed-out poll) and must make no decisions.
async fn poll_deployment(
    autoscaler: &Autoscaler,
    slots: &[WorkerSlot],
    deployment_id: &DeploymentId,
    poll_budget: Duration,
) -> Option<DeploymentObservation> {
    let state = autoscaler.deployment(deployment_id)?;
    if let Some(slot) = slots.iter().find(|slot| {
        state.worker_order().contains(slot.worker_id())
            && slot.restart_managed()
            && slot.actor.view().is_restarting()
    }) {
        debug!(
            "autoscaling: deployment {deployment_id} holds its decisions while worker {} is restart-pending",
            slot.worker_id()
        );
        return None;
    }
    let mut observation = DeploymentObservation::default();
    for worker in state.active() {
        let slot = slots.iter().find(|slot| slot.worker_id() == worker)?;
        let client = slot.actor.view().client?;
        let status = match tokio::time::timeout(poll_budget, client.status()).await {
            Ok(Ok(status)) => status,
            Ok(Err(error)) => {
                warn!("autoscaling: worker {worker} status poll failed: {error}");
                return None;
            }
            Err(_) => {
                warn!("autoscaling: worker {worker} status poll timed out");
                return None;
            }
        };
        observation.signals.insert(
            worker.clone(),
            WorkerSignals {
                queued_invocations: u64::from(status.capacity.queued_invocations),
                active_invocations: status.capacity.active_invocations,
                reserved_sessions: status.capacity.reserved_sessions,
            },
        );
    }
    Some(observation)
}

/// Every binary flavor referenced by the node configuration must have an
/// operator-provided path. Configuration validation rejects flavor/backend
/// mismatches; this only proves the operator supplied the binaries they
/// configured. The supervisor never probes or initializes accelerators:
/// device inventory is operator-declared and verified by the worker's own
/// assigned-device selection at startup.
fn ensure_flavor_binaries(
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    config: PathBuf,
    cpu_worker_binary: Option<PathBuf>,
    metal_worker_binary: Option<PathBuf>,
    cuda_worker_binary: Option<PathBuf>,
    cpu_ids: Vec<u16>,
    metal_devices: Vec<(DeviceId, u32)>,
    cuda_devices: Vec<(DeviceId, u32, u64)>,
    allocatable_host_memory_bytes: u64,
    validate_only: bool,
    canary_worker_id: Option<WorkerId>,
    rollout_plan: Option<PathBuf>,
    rollout_abort: bool,
    rollout_status: bool,
    clear_quarantine: Option<String>,
}

enum ParseOutcome {
    Run(Box<CliOptions>),
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
        let mut metal_worker_binary = None;
        let mut cuda_worker_binary = None;
        let mut cpu_ids = None;
        let mut metal_devices = Vec::new();
        let mut cuda_devices = Vec::new();
        let mut allocatable_host_memory_bytes = None;
        let mut validate_only = false;
        let mut canary_worker_id = None;
        let mut rollout_plan = None;
        let mut rollout_abort = false;
        let mut rollout_status = false;
        let mut clear_quarantine = None;
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
            if name == "--rollout-abort" {
                if rollout_abort {
                    return Err(SupervisorError::DuplicateOption(name.to_string()));
                }
                rollout_abort = true;
                index += 1;
                continue;
            }
            if name == "--rollout-status" {
                if rollout_status {
                    return Err(SupervisorError::DuplicateOption(name.to_string()));
                }
                rollout_status = true;
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
                "--metal-worker-binary" => {
                    set_once(&mut metal_worker_binary, PathBuf::from(value), name)?
                }
                "--cuda-worker-binary" => {
                    set_once(&mut cuda_worker_binary, PathBuf::from(value), name)?
                }
                "--metal-devices" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    metal_devices.extend(parse_metal_devices(value)?);
                }
                "--cuda-devices" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    cuda_devices.extend(parse_cuda_devices(value)?);
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
                "--canary-worker-id" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    let id = WorkerId::new(value)
                        .map_err(|_| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut canary_worker_id, id, name)?;
                }
                "--rollout-plan" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut rollout_plan, PathBuf::from(value), name)?;
                }
                "--clear-quarantine" => {
                    let value = value
                        .to_str()
                        .ok_or_else(|| SupervisorError::InvalidOption(name.to_string()))?;
                    set_once(&mut clear_quarantine, value.to_string(), name)?;
                }
                _ => return Err(SupervisorError::UnknownOption(name.to_string())),
            }
            index += 2;
        }
        let rollout_modes = usize::from(rollout_abort)
            + usize::from(rollout_status)
            + usize::from(rollout_plan.is_some());
        if rollout_modes > 1 {
            return Err(SupervisorError::InvalidOption(
                "--rollout-plan, --rollout-abort, and --rollout-status are mutually exclusive"
                    .to_string(),
            ));
        }
        if (rollout_abort || rollout_status || rollout_plan.is_some())
            && (validate_only || canary_worker_id.is_some())
        {
            return Err(SupervisorError::InvalidOption(
                "--rollout-* modes cannot be combined with --validate-only or --canary-worker-id"
                    .to_string(),
            ));
        }
        if clear_quarantine.is_some()
            && (rollout_modes > 0 || validate_only || canary_worker_id.is_some())
        {
            return Err(SupervisorError::InvalidOption(
                "--clear-quarantine is a command of its own".to_string(),
            ));
        }
        Ok(ParseOutcome::Run(Box::new(Self {
            config: config.ok_or(SupervisorError::MissingOption("--config"))?,
            cpu_worker_binary,
            metal_worker_binary,
            cuda_worker_binary,
            cpu_ids: cpu_ids.ok_or(SupervisorError::MissingOption("--cpu-ids"))?,
            metal_devices,
            cuda_devices,
            allocatable_host_memory_bytes: allocatable_host_memory_bytes.ok_or(
                SupervisorError::MissingOption("--allocatable-host-memory-bytes"),
            )?,
            validate_only,
            canary_worker_id,
            rollout_plan,
            rollout_abort,
            rollout_status,
            clear_quarantine,
        })))
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

/// Declares Metal devices as `device_id@process_local_index` pairs. Devices
/// are declared by the operator (from `system_profiler`/`ioreg`) because the
/// supervisor never probes or initializes accelerators; the worker's own
/// assigned-device selection verifies the real identity at startup.
fn parse_metal_devices(value: &str) -> Result<Vec<(DeviceId, u32)>, SupervisorError> {
    const OPTION: &str = "--metal-devices";
    let invalid = |reason: String| SupervisorError::InvalidDeviceDeclaration {
        option: OPTION,
        reason,
    };
    if value.is_empty() || value.len() > 16 * 1024 {
        return Err(invalid("expected device_id@index".into()));
    }
    let mut seen = BTreeSet::new();
    let mut devices = Vec::new();
    for item in value.split(',') {
        let Some((id, index)) = item.split_once('@') else {
            return Err(invalid(format!("expected device_id@index, got {item:?}")));
        };
        let device_id =
            DeviceId::new(id).map_err(|_| invalid(format!("invalid device id {id:?}")))?;
        let index: u32 = index
            .parse()
            .map_err(|_| invalid(format!("invalid process-local index {index:?}")))?;
        if !seen.insert(device_id.clone()) || seen.len() > MAX_DEVICE_DECLARATIONS {
            return Err(invalid("duplicate or excessive device declarations".into()));
        }
        devices.push((device_id, index));
    }
    Ok(devices)
}

/// Declares CUDA devices as `device_uuid@host_device_index@total_memory_bytes`.
fn parse_cuda_devices(value: &str) -> Result<Vec<(DeviceId, u32, u64)>, SupervisorError> {
    const OPTION: &str = "--cuda-devices";
    let invalid = |reason: String| SupervisorError::InvalidDeviceDeclaration {
        option: OPTION,
        reason,
    };
    if value.is_empty() || value.len() > 16 * 1024 {
        return Err(invalid(
            "expected uuid@host_index@total_memory_bytes".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut devices = Vec::new();
    for item in value.split(',') {
        let fields: Vec<&str> = item.split('@').collect();
        let [uuid, host_index, total_memory_bytes] = fields.as_slice() else {
            return Err(invalid(format!(
                "expected uuid@host_index@total_memory_bytes, got {item:?}"
            )));
        };
        let device_uuid =
            DeviceId::new(*uuid).map_err(|_| invalid(format!("invalid device uuid {uuid:?}")))?;
        let host_index: u32 = host_index
            .parse()
            .map_err(|_| invalid(format!("invalid host device index {host_index:?}")))?;
        let total_memory_bytes: u64 = total_memory_bytes
            .parse()
            .map_err(|_| invalid(format!("invalid total memory bytes {total_memory_bytes:?}")))?;
        if total_memory_bytes == 0
            || !seen.insert(device_uuid.clone())
            || seen.len() > MAX_DEVICE_DECLARATIONS
        {
            return Err(invalid(
                "duplicate, zero-memory, or excessive device declarations".into(),
            ));
        }
        devices.push((device_uuid, host_index, total_memory_bytes));
    }
    Ok(devices)
}

fn print_usage() {
    eprintln!(
        "Usage: izwi-serving-supervisor \\\n  --config PATH \\\n  --cpu-ids 0,1,... \\\n  --allocatable-host-memory-bytes BYTES \\\n  [--cpu-worker-binary PATH] [--metal-worker-binary PATH] [--cuda-worker-binary PATH] \\\n  [--metal-devices id@index,...] [--cuda-devices uuid@host_index@total_memory_bytes,...]\n\n\
Every binary flavor referenced by the node configuration must be provided. Device\n\
declarations come from the operator (the supervisor never probes or initializes\n\
accelerators); each worker verifies its assigned device identity at startup and\n\
the supervisor admits nothing before that worker reports readiness."
    );
    eprintln!(
        "Optional: --validate-only resolves configuration and service credentials, prints bounded redacted diagnostics, and exits without acquiring locks or launching workers."
    );
    eprintln!(
        "Rollout (DS6): --rollout-plan PATH starts or resumes the rollout declared by the plan (blue-green when\n\
  both generations fit the node, otherwise recreate);\n\
  --rollout-abort restores the pre-rollout approvals and clears state; --rollout-status reports persisted rollout state.\n\
  All three require --config."
    );
    eprintln!(
        "Logs go to stderr as text, or one JSON object per line with IZWI_LOG_FORMAT=json; worker\n\
  stdout and stderr are relayed line by line with the worker id. IZWI_SUPERVISOR_METRICS_TEXTFILE=PATH\n\
  keeps supervisor counters and worker-state gauges there in the Prometheus text format."
    );
    eprintln!(
        "--clear-quarantine WORKER_ID|all asks the running supervisor (same --config) to relaunch a\n\
  quarantined worker at once; quarantined workers are otherwise retried every restart.quarantine_retry_ms."
    );
    #[cfg(unix)]
    eprintln!(
        "Send SIGUSR1 to a running supervisor for one bounded, redacted status snapshot on stderr; SIGUSR2 aborts a running rollout while it is still reversible; SIGHUP clears every quarantine."
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
    #[error(
        "worker {worker} references binary flavor {binary:?}; provide its binary via {option}"
    )]
    MissingFlavorBinary {
        option: &'static str,
        worker: WorkerId,
        binary: WorkerBinaryFlavor,
    },
    #[error("{option} is invalid: {reason}")]
    InvalidDeviceDeclaration {
        option: &'static str,
        reason: String,
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
    #[error("canary worker {worker_id} was not found in the node configuration")]
    CanaryWorkerNotFound { worker_id: WorkerId },
    #[error("canary worker {worker_id} failed to reach readiness; rollout aborted")]
    CanaryReadinessFailed { worker_id: WorkerId },
    #[error(transparent)]
    Rollout(#[from] rollout::RolloutError),
    #[error("persisted rollout state belongs to a different plan; resume with its original plan or abort with --rollout-abort")]
    RolloutStateDigestMismatch,
    #[error("a committed rollout recorded target config digest {recorded}; the supplied config digest is {actual}. Start with the committed target config or clear the state with --rollout-abort")]
    RolloutCommittedConfigMismatch { recorded: String, actual: String },
    #[error("cannot abort: a live supervisor holds the node lease (send SIGUSR2 for in-process abort, or stop the supervisor first)")]
    RolloutSupervisorBusy,
    #[error("cannot abort: orphaned rollout workers still hold their generation fence; they self-drain after the supervisor exited, retry shortly")]
    RolloutFenceContended,
    #[error("no rollout state found to abort")]
    RolloutNothingToAbort,
    #[error(
        "autoscaling and coordinated rollout are mutually exclusive in one supervisor run; disable one before using the other"
    )]
    RolloutAutoscalingConflict,
    #[error("autoscaling: {0}")]
    Autoscale(String),
    #[error("the rollout plan's gateway_ack.admin_key_env {environment} is not set")]
    MissingGatewayAdminKey { environment: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_protocol::DeviceAssignment;
    use izwi_serving_protocol::{
        ArtifactRevision, CancellationBehavior, CredentialId, DeploymentId, DeviceId, InputFormat,
        ModelAlias, ModelGeneration, NodeId, OutputFormat, TaskKind,
    };
    use izwi_serving_supervisor::{
        CapabilityProfileConfig, DeploymentConfig, ReadinessPolicy, RestartPolicy, ShutdownPolicy,
        WorkerConfig, DEFAULT_MODEL_LOAD_SLOTS, NODE_CONFIG_SCHEMA_VERSION,
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
            max_parallel_model_loads: DEFAULT_MODEL_LOAD_SLOTS,
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
                host_kv_pool_budget_bytes: 0,
                environment: Default::default(),
            }],
            readiness: ReadinessPolicy::default(),
            liveness: Default::default(),
            restart: RestartPolicy::default(),
            shutdown: ShutdownPolicy::default(),
            autoscaling: None,
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
        // Worker binary paths are optional at the CLI level: the missing one is
        // reported against the flavor the node configuration actually references.
        assert_eq!(options.cpu_worker_binary, Some(PathBuf::from("/worker")));
        assert!(options.metal_worker_binary.is_none());
        assert!(options.cuda_worker_binary.is_none());
        assert!(matches!(
            CliOptions::parse([OsString::from("--config"), OsString::from("/x")]),
            Err(SupervisorError::MissingOption("--cpu-ids"))
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
        let view = WorkerView {
            phase: WorkerPhase::RestartPending { at: Instant::now() },
            ready_launches: 1,
            failed_launches: 1,
            process_id: None,
            incarnation: None,
            client: None,
            exit_observation_failed: false,
        };
        let metrics = SupervisorMetrics::default();
        for (counter, value) in [
            (&metrics.launch_attempts, 2),
            (&metrics.readiness_successes, 1),
            (&metrics.readiness_failures, 1),
            (&metrics.restarts_scheduled, 1),
        ] {
            counter.store(value, std::sync::atomic::Ordering::Relaxed);
        }
        let worker_id = id("worker-a");
        let snapshot = SlotSnapshot {
            worker_id: &worker_id,
            config: node.worker(&worker_id).unwrap(),
            view: &view,
            autoscale: None,
        };

        let diagnostic =
            render_runtime_diagnostic(&node, std::iter::once(snapshot), &metrics, Instant::now());

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
    fn ensure_flavor_binaries_requires_the_configured_lanes() {
        let metal = DeviceAssignment::Metal {
            device_id: id::<DeviceId>("metal:1"),
            process_local_device_index: 0,
            shared_memory_limit_bytes: 512,
        };
        let options = CliOptions {
            config: PathBuf::from("/x"),
            cpu_worker_binary: Some(PathBuf::from("/cpu-worker")),
            metal_worker_binary: None,
            cuda_worker_binary: None,
            cpu_ids: vec![0, 1],
            metal_devices: Vec::new(),
            cuda_devices: Vec::new(),
            allocatable_host_memory_bytes: 4294967296,
            validate_only: false,
            canary_worker_id: None,
            rollout_plan: None,
            rollout_abort: false,
            rollout_status: false,
            clear_quarantine: None,
        };
        assert!(matches!(
            ensure_flavor_binaries(&config(metal.clone(), WorkerBinaryFlavor::Metal), &options),
            Err(SupervisorError::MissingFlavorBinary {
                option: "--metal-worker-binary",
                binary: WorkerBinaryFlavor::Metal,
                ..
            })
        ));

        let mut with_metal = options.clone();
        with_metal.metal_worker_binary = Some(PathBuf::from("/metal-worker"));
        ensure_flavor_binaries(&config(metal, WorkerBinaryFlavor::Metal), &with_metal)
            .expect("a declared metal lane with its binary passes the flavor check");
    }

    #[test]
    fn device_declarations_parse_and_fail_closed() {
        let metal = parse_metal_devices("metal:0x1@0,metal:0x2@1").unwrap();
        assert_eq!(
            metal,
            vec![
                (DeviceId::new("metal:0x1").unwrap(), 0),
                (DeviceId::new("metal:0x2").unwrap(), 1)
            ]
        );
        assert!(parse_metal_devices("metal:0x1").is_err());
        assert!(parse_metal_devices("metal:0x1@0,metal:0x1@0").is_err());
        assert!(parse_metal_devices("").is_err());

        let cuda = parse_cuda_devices("GPU-abc@0@8589934592").unwrap();
        assert_eq!(
            cuda,
            vec![(DeviceId::new("GPU-abc").unwrap(), 0, 8_589_934_592u64)]
        );
        assert!(parse_cuda_devices("GPU-abc@0").is_err());
        assert!(parse_cuda_devices("GPU-abc@0@0").is_err());
        assert!(parse_cuda_devices("GPU-abc@1@8589934592@extra").is_err());
    }

    #[test]
    fn canary_worker_id_is_parsed_and_validated() {
        let ParseOutcome::Run(options) = CliOptions::parse([
            OsString::from("--config"),
            OsString::from("/x"),
            OsString::from("--cpu-worker-binary"),
            OsString::from("/w"),
            OsString::from("--cpu-ids"),
            OsString::from("0,1"),
            OsString::from("--allocatable-host-memory-bytes"),
            OsString::from("4294967296"),
            OsString::from("--canary-worker-id"),
            OsString::from("canary-worker-1"),
        ])
        .unwrap() else {
            panic!("expected Run outcome");
        };
        assert_eq!(
            options.canary_worker_id.as_ref().unwrap().as_str(),
            "canary-worker-1"
        );
    }

    #[test]
    fn canary_worker_id_is_optional() {
        let ParseOutcome::Run(options) = CliOptions::parse([
            OsString::from("--config"),
            OsString::from("/x"),
            OsString::from("--cpu-worker-binary"),
            OsString::from("/w"),
            OsString::from("--cpu-ids"),
            OsString::from("0,1"),
            OsString::from("--allocatable-host-memory-bytes"),
            OsString::from("4294967296"),
        ])
        .unwrap() else {
            panic!("expected Run outcome");
        };
        assert!(options.canary_worker_id.is_none());
    }

    #[test]
    fn duplicate_canary_worker_id_is_rejected() {
        let result = CliOptions::parse([
            OsString::from("--config"),
            OsString::from("/x"),
            OsString::from("--cpu-worker-binary"),
            OsString::from("/w"),
            OsString::from("--cpu-ids"),
            OsString::from("0,1"),
            OsString::from("--allocatable-host-memory-bytes"),
            OsString::from("4294967296"),
            OsString::from("--canary-worker-id"),
            OsString::from("canary-1"),
            OsString::from("--canary-worker-id"),
            OsString::from("canary-2"),
        ]);
        assert!(result.is_err());
        let error = result.err().unwrap().to_string();
        assert!(error.contains("repeated"));
    }
}
