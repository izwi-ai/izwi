use izwi_serving_client::WorkerClientConfig;
use izwi_serving_protocol::{
    BackendKind, DeploymentId, DeviceId, IncarnationId, ServiceBearerToken, ServiceCredentials,
    WorkerId,
};
use izwi_serving_supervisor::autoscale::{
    self, Autoscaler, DeploymentObservation, ResourceLedger, ScaleDecision, WorkerSignals,
};
use izwi_serving_supervisor::rollout::{self, RolloutPhase};
use izwi_serving_supervisor::{
    build_child_launch_spec, BinaryCatalog, BinaryRecord, CudaDeviceInventory, HostInventory,
    LockNamespace, MetalDeviceInventory, NodeConfig, ResolvedWorkerSecret, RestartController,
    RestartDecision, ShutdownPolicy, SupervisedWorker, ValidatedNodeConfig, WorkerBinaryFlavor,
    WorkerConfig, WorkerLockPaths, MAX_NODE_CONFIG_BYTES,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::{OsStr, OsString},
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc, watch},
    task::JoinSet,
    time::{Instant, MissedTickBehavior},
};

const MAX_CLI_ARGUMENTS: usize = 24;
const MAX_CPU_IDS: usize = 1024;
const MAX_DEVICE_DECLARATIONS: usize = 64;
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
    run(*options).await
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
    let node = config.validate(&inventory, &binaries)?;
    let mut slots = resolve_slots(&node)?;
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

    let inherited_environment = inherited_environment();

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
        .is_some_and(|prepared| prepared.state.phase == RolloutPhase::DrainingOld);

    let mut autoscaler = Autoscaler::from_config(&node);
    let mut ledger = ResourceLedger::new(&inventory, &node);
    if let Some(scaler) = autoscaler.as_mut() {
        // Partition slots: the min set keeps the static posture
        // (`autoscale: None`); every other declared worker of an autoscaled
        // deployment becomes a standby that only a scale-up decision starts.
        for slot in slots.iter_mut() {
            let Some(state) = scaler.deployment_of(&slot.worker_id) else {
                continue;
            };
            if state.core_set().contains(&slot.worker_id) {
                continue;
            }
            slot.autoscale = Some(AutoscaleSlotState {
                deployment: state.deployment_id().clone(),
                phase: AutoscalePhase::Standby,
            });
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
        eprintln!(
            "autoscaling: reconciled {} to the min set of {} autoscaled deployment(s)",
            approvals_path.display(),
            scaler.deployments().len()
        );
    }
    if let Some(canary_id) = options.canary_worker_id.as_ref() {
        if slots.iter().any(|slot| {
            &slot.worker_id == canary_id
                && slot
                    .autoscale
                    .as_ref()
                    .is_some_and(|state| state.phase == AutoscalePhase::Standby)
        }) {
            return Err(SupervisorError::Autoscale(format!(
                "canary worker {canary_id} is an autoscaling standby; canaries must belong to the startup min set"
            )));
        }
    }

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        wait_for_shutdown_request().await;
        let _ = shutdown_tx.send(true);
    });
    let (diagnostic_tx, mut diagnostic_rx) = mpsc::channel(1);
    spawn_diagnostic_signal_listener(diagnostic_tx);
    // DS6: SIGUSR2 requests an in-process rollout abort while the rollout is
    // still reversible (launch or window phases).
    let (abort_tx, mut abort_rx) = watch::channel(false);
    #[cfg(unix)]
    tokio::spawn(async move {
        let mut signal =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined2())
                .expect("install SIGUSR2 listener");
        signal.recv().await;
        let _ = abort_tx.send(true);
    });

    let started_at = Instant::now();
    let mut metrics = SupervisorMetrics::default();

    if let Some(canary_id) = options.canary_worker_id.as_ref() {
        let canary_index = slots
            .iter()
            .position(|slot| &slot.worker_id == canary_id)
            .ok_or_else(|| SupervisorError::CanaryWorkerNotFound {
                worker_id: canary_id.clone(),
            })?;
        eprintln!(
            "launching canary worker {} first; remaining workers will start after it reaches readiness",
            canary_id
        );
        launch_slot(
            &node,
            &locks,
            &inherited_environment,
            &mut slots[canary_index],
            &mut shutdown_rx,
            started_at,
            &mut metrics,
        )
        .await;
        note_slot_launched(&mut autoscaler, &slots[canary_index]);
        if slots[canary_index].process.is_none() {
            eprintln!(
                "canary worker {} failed to reach readiness; aborting rollout",
                canary_id
            );
            return Err(SupervisorError::CanaryReadinessFailed {
                worker_id: canary_id.clone(),
            });
        }
        eprintln!(
            "canary worker {} is ready; launching remaining workers",
            canary_id
        );
    }

    for slot in &mut slots {
        if *shutdown_rx.borrow() {
            break;
        }
        if options
            .canary_worker_id
            .as_ref()
            .is_some_and(|id| &slot.worker_id == id)
        {
            continue;
        }
        // On resume into DrainingOld the old generation is retired and its
        // workers exited before the generation barrier released; never
        // relaunch them.
        if resume_draining_old
            && rollout
                .as_ref()
                .is_some_and(|prepared| prepared.spec.is_old_worker(slot.worker_id.as_str()))
        {
            continue;
        }
        // DS7: autoscaling standbys wait for a scale-up decision.
        if slot
            .autoscale
            .as_ref()
            .is_some_and(|state| state.phase == AutoscalePhase::Standby)
        {
            continue;
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
        note_slot_launched(&mut autoscaler, slot);
    }

    // DS6: run the rollout state machine to commit or abort. After a commit
    // the replacement slots join supervision under the target node config.
    let mut replacement_slots: Vec<WorkerSlot> = Vec::new();
    let mut target_node: Option<ValidatedNodeConfig> = None;
    if rollout.is_some() {
        match execute_rollout(
            rollout.as_mut().expect("rollout is present"),
            &node,
            &locks,
            &inherited_environment,
            &mut slots,
            &mut shutdown_rx,
            &mut abort_rx,
            started_at,
            &mut metrics,
        )
        .await
        {
            RolloutOutcome::Committed | RolloutOutcome::DegradedAbort => {
                let prepared = rollout.take().expect("rollout is present");
                replacement_slots = prepared.replacements;
                target_node = Some(prepared.target);
            }
            RolloutOutcome::Aborted => {
                rollout.take();
            }
        }
    }

    let mut poll = tokio::time::interval(SUPERVISION_POLL_INTERVAL);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut next_autoscale_eval = Instant::now()
        + autoscaler
            .as_ref()
            .map_or(Duration::ZERO, |scaler| scaler.evaluation_interval());
    while !*shutdown_rx.borrow() {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    break;
                }
            }
            Some(()) = diagnostic_rx.recv() => {
                eprint!("{}", runtime_diagnostic(&node, &slots, &metrics, started_at));
                if !replacement_slots.is_empty() {
                    if let Some(target) = target_node.as_ref() {
                        eprint!(
                            "{}",
                            runtime_diagnostic(target, &replacement_slots, &metrics, started_at)
                        );
                    }
                }
            }
            _ = poll.tick() => {
                observe_exits(
                    &mut slots,
                    autoscaler.as_mut(),
                    &mut ledger,
                    started_at,
                    &mut metrics,
                );
                observe_exits(&mut replacement_slots, None, &mut ledger, started_at, &mut metrics);
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
                    note_slot_launched(&mut autoscaler, slot);
                }
                if !replacement_slots.is_empty() {
                    if let Some(target) = target_node.as_ref() {
                        if let Some(slot) = next_restart_slot(&mut replacement_slots) {
                            launch_slot(
                                target,
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
                // DS7: one autoscaling evaluation pass per configured interval.
                if Instant::now() >= next_autoscale_eval {
                    if let Some(scaler) = autoscaler.as_mut() {
                        next_autoscale_eval = Instant::now() + scaler.evaluation_interval();
                        run_autoscale_tick(
                            scaler,
                            &node,
                            &mut ledger,
                            &mut slots,
                            &locks,
                            &inherited_environment,
                            &mut shutdown_rx,
                            started_at,
                            &mut metrics,
                        )
                        .await;
                    }
                }
            }
        }
    }

    drain_all(slots, node.config().shutdown.clone(), &mut metrics).await;
    let replacement_policy = target_node
        .as_ref()
        .map(|target| target.config().shutdown.clone())
        .unwrap_or_else(|| node.config().shutdown.clone());
    drain_all(replacement_slots, replacement_policy, &mut metrics).await;
    Ok(())
}

/// A prepared DS6 rollout: validated spec, staged state, target config, and
/// resolved replacement slots awaiting launch.
struct PreparedRollout {
    spec: rollout::RolloutSpec,
    state: rollout::RolloutState,
    target: ValidatedNodeConfig,
    approvals_path: PathBuf,
    classified: Vec<rollout::ClassifiedApproval>,
    replacements: Vec<WorkerSlot>,
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

    let runtime_directory = &current.config().runtime_directory;
    let plan_digest = plan.digest(&target_bytes);
    let existing = rollout::load_state(runtime_directory)?;
    let state = match existing {
        Some(state) if !state.phase.is_terminal() => {
            if state.plan_digest != plan_digest {
                return Err(SupervisorError::RolloutStateDigestMismatch);
            }
            eprintln!(
                "resuming rollout in state {} (plan digest matches)",
                state.phase.as_str()
            );
            Some(state)
        }
        Some(state) => {
            rollout::clear_state(runtime_directory)?;
            rollout::clear_approvals_backup(runtime_directory)?;
            eprintln!(
                "cleared terminal rollout state ({}) before staging the new rollout",
                state.phase.as_str()
            );
            None
        }
        None => None,
    };

    let approvals_text = rollout::read_approvals_file(plan.shared_approvals_path())?;
    let classified = rollout::classify_approvals(&approvals_text)?;
    rollout::verify_approvals_precondition(&classified, &spec)?;
    let replacements = resolve_replacement_slots(&target, &spec)?;

    let state = match state {
        Some(state) => state,
        None => {
            let now = now_ms();
            let approvals_backup_digest =
                rollout::backup_approvals(runtime_directory, &approvals_text)?;
            rollout::RolloutState {
                schema_version: 1,
                plan_digest,
                phase: rollout::RolloutPhase::LaunchingReplacement,
                started_at_ms: now,
                updated_at_ms: now,
                window_deadline_ms: None,
                replacement_workers: Vec::new(),
                shared_approvals_path: plan.shared_approvals_path().to_string_lossy().into_owned(),
                target_node_config: plan.target_node_config().to_string_lossy().into_owned(),
                target_config_digest: rollout::digest_bytes(&target_bytes),
                approvals_backup_digest,
            }
        }
    };
    rollout::persist_state(runtime_directory, &state)?;
    eprintln!(
        "rollout prepared: {} rolling deployment(s), canary {}, window {}s, abort grace {}s",
        spec.rolling.len(),
        spec.canary_worker_id,
        spec.window.as_secs(),
        spec.abort_grace.as_secs()
    );
    Ok(PreparedRollout {
        spec,
        state,
        target,
        approvals_path: plan.shared_approvals_path().to_path_buf(),
        classified,
        replacements,
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
    rollout::clear_approvals_backup(runtime_directory)?;
    eprintln!("cleared terminal rollout state ({})", state.phase.as_str());
    Ok(())
}

// Single-call-site orchestrator: every parameter is a distinct handle the
// rollout state machine needs in place, so grouping them would only add
// indirection without reducing real coupling.
#[allow(clippy::too_many_arguments)]
async fn execute_rollout(
    prepared: &mut PreparedRollout,
    node: &ValidatedNodeConfig,
    locks: &LockNamespace,
    inherited_environment: &BTreeMap<OsString, OsString>,
    current_slots: &mut Vec<WorkerSlot>,
    shutdown: &mut watch::Receiver<bool>,
    abort_requested: &mut watch::Receiver<bool>,
    started_at: Instant,
    metrics: &mut SupervisorMetrics,
) -> RolloutOutcome {
    let runtime_directory = node.config().runtime_directory.clone();
    let shutdown_policy = node.config().shutdown.clone();
    let resuming_window = prepared.state.phase == RolloutPhase::WindowOpen;

    // Launch the replacements canary-first. On window resume the crashed
    // supervisor's replacements self-drained, so they relaunch here too.
    if prepared.state.phase == RolloutPhase::LaunchingReplacement || resuming_window {
        let canary = prepared.spec.canary_worker_id.clone();
        prepared
            .replacements
            .sort_by_key(|slot| slot.worker_id.as_str() != canary.as_str());
        prepared.state.replacement_workers.clear();
        for slot in prepared.replacements.iter_mut() {
            if *shutdown.borrow() {
                break;
            }
            launch_slot(
                &prepared.target,
                locks,
                inherited_environment,
                slot,
                shutdown,
                started_at,
                metrics,
            )
            .await;
            if slot.process.is_none() {
                eprintln!(
                    "replacement worker {} failed to reach readiness; aborting rollout",
                    slot.worker_id
                );
                let restored = abort_rollout(
                    prepared,
                    &runtime_directory,
                    metrics,
                    false,
                    &shutdown_policy,
                    "replacement readiness failed",
                )
                .await;
                return if restored {
                    RolloutOutcome::Aborted
                } else {
                    RolloutOutcome::DegradedAbort
                };
            }
            let pid = slot
                .process
                .as_ref()
                .and_then(SupervisedWorker::process_id)
                .unwrap_or(0);
            prepared
                .state
                .replacement_workers
                .push(rollout::RolloutWorkerRecord {
                    worker_id: slot.worker_id.to_string(),
                    pid,
                });
            prepared.state.updated_at_ms = now_ms();
            if let Err(error) = rollout::persist_state(&runtime_directory, &prepared.state) {
                eprintln!("rollout state persist failed: {error}");
            }
        }
        if *shutdown.borrow() {
            let restored = abort_rollout(
                prepared,
                &runtime_directory,
                metrics,
                false,
                &shutdown_policy,
                "shutdown requested",
            )
            .await;
            return if restored {
                RolloutOutcome::Aborted
            } else {
                RolloutOutcome::DegradedAbort
            };
        }
    }

    // Open (or re-open on resume) the window: both generations are approved,
    // the gateway cuts over atomically when the successor first observes
    // Ready. The old generation never stops serving during the window.
    if let Err(error) =
        rollout::build_window_view(&prepared.classified, &prepared.spec, &prepared.target)
            .and_then(|view| {
                rollout::write_atomic(&prepared.approvals_path, view.as_bytes())?;
                Ok(view)
            })
    {
        eprintln!("rollout window view failed to write: {error}; nothing changed");
        return RolloutOutcome::DegradedAbort;
    }
    let deadline_ms = prepared
        .state
        .window_deadline_ms
        .unwrap_or_else(|| now_ms().saturating_add(prepared.spec.window.as_millis() as u64));
    prepared.state.phase = RolloutPhase::WindowOpen;
    prepared.state.window_deadline_ms = Some(deadline_ms);
    prepared.state.updated_at_ms = now_ms();
    if let Err(error) = rollout::persist_state(&runtime_directory, &prepared.state) {
        eprintln!("rollout state persist failed: {error}");
    }
    eprintln!(
        "rollout window open: both generations approved; cutover fires when the successor generation is observed Ready; drain starts in {}s",
        deadline_ms.saturating_sub(now_ms()) / 1000
    );
    let mut window_watch = tokio::time::interval(Duration::from_millis(500));
    window_watch.set_missed_tick_behavior(MissedTickBehavior::Delay);
    while now_ms() < deadline_ms {
        tokio::select! {
            _ = window_watch.tick() => {
                let operator_abort = *abort_requested.borrow();
                if *shutdown.borrow() || operator_abort {
                    let reason = if *shutdown.borrow() {
                        "shutdown requested"
                    } else {
                        "operator abort (SIGUSR2)"
                    };
                    let restored = abort_rollout(
                        prepared,
                        &runtime_directory,
                        metrics,
                        true,
                        &shutdown_policy,
                        reason,
                    )
                    .await;
                    return if restored {
                        RolloutOutcome::Aborted
                    } else {
                        RolloutOutcome::DegradedAbort
                    };
                }
                for slot in prepared.replacements.iter_mut() {
                    let Some(process) = slot.process.as_mut() else {
                        continue;
                    };
                    if matches!(process.try_wait(), Ok(Some(_))) {
                        eprintln!(
                            "replacement worker {} exited during the rollout window; aborting rollout",
                            slot.worker_id
                        );
                        let restored = abort_rollout(
                            prepared,
                            &runtime_directory,
                            metrics,
                            true,
                            &shutdown_policy,
                            "replacement exited during the window",
                        )
                        .await;
                        return if restored {
                            RolloutOutcome::Aborted
                        } else {
                            RolloutOutcome::DegradedAbort
                        };
                    }
                }
            }
            _ = abort_requested.changed() => {}
        }
    }

    // DrainingOld is the point of no return: the commit view is written and
    // the old generation's workers stop admitting. On resume into this
    // phase the old workers already exited (the generation barrier proved
    // it), so the drain finds nothing to stop.
    prepared.state.phase = RolloutPhase::DrainingOld;
    prepared.state.window_deadline_ms = None;
    prepared.state.updated_at_ms = now_ms();
    if let Err(error) = rollout::persist_state(&runtime_directory, &prepared.state) {
        eprintln!("rollout state persist failed: {error}");
    }
    eprintln!(
        "rollout drain: writing the commit approval view and draining the old generation (point of no return)"
    );
    if let Err(error) =
        rollout::build_commit_view(&prepared.classified, &prepared.spec, &prepared.target)
            .and_then(|view| {
                rollout::write_atomic(&prepared.approvals_path, view.as_bytes())?;
                Ok(view)
            })
    {
        eprintln!(
            "rollout commit view failed to write: {error}; the window view stays in place"
        );
        return RolloutOutcome::DegradedAbort;
    }
    drain_selected(
        current_slots,
        |slot| prepared.spec.is_old_worker(slot.worker_id.as_str()),
        &shutdown_policy,
        metrics,
    )
    .await;
    prepared.state.phase = RolloutPhase::Committed;
    prepared.state.updated_at_ms = now_ms();
    if let Err(error) = rollout::persist_state(&runtime_directory, &prepared.state) {
        eprintln!("rollout state persist failed: {error}");
    }
    eprintln!(
        "rollout committed: the target config's generations are admission-eligible; repoint the service definition at the target node config"
    );
    RolloutOutcome::Committed
}

/// Aborts a reversible rollout: restore the pre-rollout approvals first so
/// the gateway stops routing to the replacements within one refresh TTL,
/// wait the abort grace, then drain the replacement workers. The previous
/// generation never stopped serving. Returns whether the approvals were
/// restored.
async fn abort_rollout(
    prepared: &mut PreparedRollout,
    runtime_directory: &Path,
    metrics: &mut SupervisorMetrics,
    honor_grace: bool,
    policy: &ShutdownPolicy,
    reason: &str,
) -> bool {
    eprintln!("rollout aborted ({reason}): restoring the previous approval view");
    if let Err(error) = rollout::restore_approvals(
        runtime_directory,
        &prepared.approvals_path,
        &prepared.state.approvals_backup_digest,
    ) {
        eprintln!(
            "rollout abort could not restore approvals: {error}; workers keep running and the state stays resumable"
        );
        return false;
    }
    if honor_grace && !prepared.spec.abort_grace.is_zero() {
        eprintln!(
            "waiting {}s for gateway approval refresh before stopping replacement workers",
            prepared.spec.abort_grace.as_secs()
        );
        tokio::time::sleep(prepared.spec.abort_grace).await;
    }
    let mut drains = JoinSet::new();
    for slot in prepared.replacements.iter_mut() {
        if let Some(process) = slot.process.take() {
            let worker_id = slot.worker_id.clone();
            let policy = policy.clone();
            drains.spawn(async move { (worker_id, process.drain_and_stop(&policy).await) });
        }
    }
    while let Some(result) = drains.join_next().await {
        match result {
            Ok((worker_id, Ok(report))) => eprintln!(
                "replacement worker {worker_id} stopped with {:?} ({})",
                report.outcome, report.exit_status
            ),
            Ok((worker_id, Err(error))) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("replacement worker {worker_id} stop failed: {error}");
            }
            Err(error) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("replacement worker stop task failed: {error}");
            }
        }
    }
    prepared.state.phase = RolloutPhase::Aborted;
    prepared.state.updated_at_ms = now_ms();
    if let Err(error) = rollout::persist_state(runtime_directory, &prepared.state) {
        eprintln!("rollout state persist failed: {error}");
    }
    let _ = rollout::clear_state(runtime_directory);
    let _ = rollout::clear_approvals_backup(runtime_directory);
    eprintln!(
        "rollout abort complete: the previous approval view is restored and the replacement workers are stopped"
    );
    true
}

/// Drains a selected subset of slots and removes them from supervision.
async fn drain_selected(
    slots: &mut Vec<WorkerSlot>,
    select: impl Fn(&WorkerSlot) -> bool,
    policy: &ShutdownPolicy,
    metrics: &mut SupervisorMetrics,
) {
    let mut drains = JoinSet::new();
    let mut selected = Vec::new();
    for (index, slot) in slots.iter_mut().enumerate() {
        if select(slot) {
            if let Some(process) = slot.process.take() {
                let worker_id = slot.worker_id.clone();
                let policy = policy.clone();
                drains.spawn(async move { (worker_id, process.drain_and_stop(&policy).await) });
            }
            selected.push(index);
        }
    }
    while let Some(result) = drains.join_next().await {
        match result {
            Ok((worker_id, Ok(report))) => eprintln!(
                "worker {worker_id} drained with {:?} ({})",
                report.outcome, report.exit_status
            ),
            Ok((worker_id, Err(error))) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("worker {worker_id} drain failed: {error}");
            }
            Err(error) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("worker drain task failed: {error}");
            }
        }
    }
    let mut cursor = 0;
    slots.retain(|_| {
        let keep = !selected.contains(&cursor);
        cursor += 1;
        keep
    });
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
        rollout::clear_approvals_backup(runtime_directory)?;
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
    rollout::restore_approvals(
        runtime_directory,
        Path::new(&state.shared_approvals_path),
        &state.approvals_backup_digest,
    )?;
    rollout::clear_approvals_backup(runtime_directory)?;
    rollout::clear_state(runtime_directory)?;
    eprintln!(
        "rollout aborted: the previous approval view is restored byte-identically; the previous generation never stopped serving"
    );
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

fn resolve_replacement_slots(
    target: &ValidatedNodeConfig,
    spec: &rollout::RolloutSpec,
) -> Result<Vec<WorkerSlot>, SupervisorError> {
    target
        .config()
        .workers
        .iter()
        .filter(|worker| spec.rolling_for(worker.worker_id.as_str()).is_some())
        .map(|worker| resolve_slot(target, worker))
        .collect()
}

/// Lifecycle phase of an autoscaling slot (DS7). Core (min-set) workers keep
/// `autoscale: None` and behave exactly like static workers; only standby
/// replicas and scale-event transitions carry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoscalePhase {
    /// Declared standby: not launched; only a scale-up decision starts it.
    Standby,
    /// Scale-up launched the process; the approvals line is not yet published.
    Admitting,
    /// Running and approved in the shared approvals view.
    Active,
    /// Scale-down in progress: unapproved, admission stopped, draining.
    Draining,
}

#[derive(Debug, Clone)]
struct AutoscaleSlotState {
    deployment: DeploymentId,
    phase: AutoscalePhase,
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
    autoscale: Option<AutoscaleSlotState>,
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
    autoscale_ups: u64,
    autoscale_downs: u64,
}

impl WorkerSlot {
    fn restart_due(&self) -> bool {
        if self
            .autoscale
            .as_ref()
            .is_some_and(|state| state.phase != AutoscalePhase::Active)
        {
            // Standbys launch only via scale-up decisions; admitting and
            // draining slots are mid-transition and never auto-restart.
            return false;
        }
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
        .map(|worker| resolve_slot(node, worker))
        .collect()
}

fn resolve_slot(
    node: &ValidatedNodeConfig,
    worker: &izwi_serving_supervisor::WorkerConfig,
) -> Result<WorkerSlot, SupervisorError> {
    let value = env::var(&worker.bearer_token_env).map_err(|_| SupervisorError::MissingSecret {
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
        autoscale: None,
    })
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
        "supervisor_status version={} node={} uptime_ms={} workers={} running={} restart_pending={} quarantined={} launch_attempts_total={} readiness_successes_total={} readiness_failures_total={} restarts_scheduled_total={} quarantines_total={} unexpected_exits_total={} exit_observation_failures_total={} stop_failures_total={} autoscale_ups_total={} autoscale_downs_total={}",
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
        metrics.autoscale_ups,
        metrics.autoscale_downs,
    ));
    let mut autoscale_summary: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for slot in slots {
        if let Some(state) = slot.autoscale.as_ref() {
            let entry = autoscale_summary
                .entry(state.deployment.to_string())
                .or_default();
            match state.phase {
                AutoscalePhase::Standby => entry[1] += 1,
                AutoscalePhase::Admitting | AutoscalePhase::Active => entry[0] += 1,
                AutoscalePhase::Draining => entry[2] += 1,
            }
        }
    }
    for (deployment, [running_capacity, standby, draining]) in autoscale_summary {
        output.push_line(&format!(
            "autoscale deployment={deployment} running={running_capacity} standby={standby} draining={draining}"
        ));
    }
    for slot in slots {
        let configured = node
            .worker(&slot.worker_id)
            .expect("worker slots are derived from validated configuration");
        let (state, process_id, incarnation, assignment_source) =
            if let Some(process) = &slot.process {
                let transition_state = || {
                    slot.autoscale
                        .as_ref()
                        .and_then(|autoscale| match autoscale.phase {
                            AutoscalePhase::Admitting => Some("admitting"),
                            AutoscalePhase::Draining => Some("draining"),
                            _ => None,
                        })
                };
                (
                    transition_state().unwrap_or(if slot.exit_observation_failed {
                        "observation-uncertain"
                    } else {
                        "ready"
                    }),
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
            } else if slot
                .autoscale
                .as_ref()
                .is_some_and(|state| state.phase == AutoscalePhase::Standby)
            {
                (
                    "standby",
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
    mut autoscaler: Option<&mut Autoscaler>,
    ledger: &mut ResourceLedger,
    supervisor_started_at: Instant,
    metrics: &mut SupervisorMetrics,
) {
    for slot in slots {
        let Some(process) = slot.process.as_mut() else {
            continue;
        };
        match process.try_wait() {
            Ok(Some(status)) => {
                if slot
                    .autoscale
                    .as_ref()
                    .is_some_and(|state| state.phase == AutoscalePhase::Draining)
                {
                    // Expected completion of a scale-down: admission had
                    // stopped, the worker finished its bounded shutdown on
                    // its own, and the slot returns to the standby pool.
                    slot.process = None;
                    slot.process_started_at = None;
                    slot.exit_observation_failed = false;
                    if let Some(state) = slot.autoscale.as_mut() {
                        state.phase = AutoscalePhase::Standby;
                    }
                    ledger.release(&slot.worker_id);
                    if let Some(state) = autoscaler
                        .as_mut()
                        .and_then(|scaler| scaler.deployment_of_mut(&slot.worker_id))
                    {
                        state.note_stopped();
                    }
                    eprintln!(
                        "autoscaling: draining worker {} exited cleanly ({status}); slot returned to standby",
                        slot.worker_id
                    );
                    continue;
                }
                metrics.unexpected_exits = metrics.unexpected_exits.saturating_add(1);
                eprintln!("worker {} exited with {status}", slot.worker_id);
                let uptime = slot
                    .process_started_at
                    .map_or(Duration::ZERO, |started| started.elapsed());
                slot.record_failure(supervisor_started_at, uptime, metrics);
                if let Some(state) = autoscaler
                    .as_mut()
                    .and_then(|scaler| scaler.deployment_of_mut(&slot.worker_id))
                {
                    state.note_worker_lost(&slot.worker_id);
                }
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

/// Driver confirmation after any successful launch of a slot that belongs to
/// an autoscaled deployment (startup min set, canary ordering, or crash
/// restart): the state machine counts it toward running capacity.
fn note_slot_launched(autoscaler: &mut Option<Autoscaler>, slot: &WorkerSlot) {
    if slot.process.is_none() {
        return;
    }
    if let Some(state) = autoscaler
        .as_mut()
        .and_then(|scaler| scaler.deployment_of_mut(&slot.worker_id))
    {
        state.note_launched(&slot.worker_id);
    }
}

/// One DS7 autoscaling evaluation pass: complete pending admissions, poll
/// the running workers of each deployment, then act on at most one scale
/// decision per deployment. Every step is conservative — a failed approvals
/// write, a failed status poll, or a launch in flight freezes that
/// deployment's decisions until the state is consistent again.
#[allow(clippy::too_many_arguments)]
async fn run_autoscale_tick(
    autoscaler: &mut Autoscaler,
    node: &ValidatedNodeConfig,
    ledger: &mut ResourceLedger,
    slots: &mut [WorkerSlot],
    locks: &LockNamespace,
    inherited_environment: &BTreeMap<OsString, OsString>,
    shutdown: &mut watch::Receiver<bool>,
    supervisor_started_at: Instant,
    metrics: &mut SupervisorMetrics,
) {
    let now_ms = now_ms();
    let Some(autoscaling) = node.config().autoscaling.as_ref() else {
        return;
    };
    let approvals_path = autoscaling.shared_approvals_path.clone();
    let deployment_ids: Vec<DeploymentId> = autoscaler.deployments().keys().cloned().collect();
    for deployment_id in deployment_ids {
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
        let decision = autoscaler
            .deployment_mut(&deployment_id)
            .expect("deployment ids come from the autoscaler")
            .evaluate(now_ms, &observation);
        match decision {
            ScaleDecision::NoAction => {}
            ScaleDecision::ScaleUp { worker } => {
                scale_up(
                    node,
                    &deployment_id,
                    ledger,
                    slots,
                    locks,
                    inherited_environment,
                    shutdown,
                    supervisor_started_at,
                    metrics,
                    worker,
                )
                .await;
            }
            ScaleDecision::MarkDraining { worker } => {
                mark_draining(
                    autoscaler,
                    node,
                    &deployment_id,
                    &approvals_path,
                    slots,
                    now_ms,
                    metrics,
                    worker,
                )
                .await;
            }
            ScaleDecision::StopDrained { worker } => {
                stop_drained(
                    autoscaler,
                    node,
                    &deployment_id,
                    ledger,
                    slots,
                    metrics,
                    worker,
                )
                .await;
            }
        }
    }
}

/// Publishes approvals lines for slots that finished scale-up readiness but
/// are not yet admitted. Returns false while any publication is outstanding
/// so the deployment defers further decisions.
fn admit_pending_workers(
    autoscaler: &mut Autoscaler,
    approvals_path: &Path,
    node: &ValidatedNodeConfig,
    deployment_id: &DeploymentId,
    slots: &mut [WorkerSlot],
    now_ms: u64,
    metrics: &mut SupervisorMetrics,
) -> bool {
    let pending: Vec<WorkerId> = slots
        .iter()
        .filter(|slot| {
            slot.autoscale.as_ref().is_some_and(|state| {
                &state.deployment == deployment_id && state.phase == AutoscalePhase::Admitting
            })
        })
        .map(|slot| slot.worker_id.clone())
        .collect();
    let mut all_admitted = true;
    for worker in pending {
        match autoscale::add_worker(approvals_path, node, &worker) {
            Ok(()) => {
                if let Some(slot) = slots.iter_mut().find(|slot| slot.worker_id == worker) {
                    if let Some(state) = slot.autoscale.as_mut() {
                        state.phase = AutoscalePhase::Active;
                    }
                }
                if let Some(state) = autoscaler.deployment_mut(deployment_id) {
                    state.note_scaled_up(now_ms, &worker);
                }
                metrics.autoscale_ups = metrics.autoscale_ups.saturating_add(1);
                eprintln!("autoscaling: worker {worker} is ready and approved; scale-up complete");
            }
            Err(error) => {
                eprintln!(
                    "autoscaling: approvals publish for worker {worker} failed: {error}; retrying next evaluation"
                );
                all_admitted = false;
            }
        }
    }
    all_admitted
}

/// Polls the status of every running worker of a deployment. `None` means
/// the deployment is not fully observable this tick (a launch in flight, a
/// restart pending, a failed or timed-out poll) and must make no decisions.
async fn poll_deployment(
    autoscaler: &Autoscaler,
    slots: &[WorkerSlot],
    deployment_id: &DeploymentId,
    poll_budget: Duration,
) -> Option<DeploymentObservation> {
    let state = autoscaler.deployment(deployment_id)?;
    let mut observation = DeploymentObservation::default();
    for worker in state.active() {
        let slot = slots.iter().find(|slot| &slot.worker_id == worker)?;
        let process = slot.process.as_ref()?;
        let status = match tokio::time::timeout(poll_budget, process.client().status()).await {
            Ok(Ok(status)) => status,
            Ok(Err(error)) => {
                eprintln!("autoscaling: worker {worker} status poll failed: {error}");
                return None;
            }
            Err(_) => {
                eprintln!("autoscaling: worker {worker} status poll timed out");
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

/// Scale-up: reserve the candidate's budgets, then launch it through the
/// existing launch/readiness path. Readiness is awaited inline (as the
/// startup and canary paths do); the approvals line is published on the
/// next evaluation pass (Admitting phase).
#[allow(clippy::too_many_arguments)]
async fn scale_up(
    node: &ValidatedNodeConfig,
    deployment_id: &DeploymentId,
    ledger: &mut ResourceLedger,
    slots: &mut [WorkerSlot],
    locks: &LockNamespace,
    inherited_environment: &BTreeMap<OsString, OsString>,
    shutdown: &mut watch::Receiver<bool>,
    supervisor_started_at: Instant,
    metrics: &mut SupervisorMetrics,
    worker: WorkerId,
) {
    let Some(slot) = slots.iter_mut().find(|slot| slot.worker_id == worker) else {
        eprintln!("autoscaling: scale-up candidate {worker} has no slot; abandoned");
        return;
    };
    if let Err(rejection) = ledger.reserve(&worker) {
        eprintln!("autoscaling: {rejection}");
        return;
    }
    slot.restart_at = None;
    slot.autoscale = Some(AutoscaleSlotState {
        deployment: deployment_id.clone(),
        phase: AutoscalePhase::Admitting,
    });
    eprintln!("autoscaling: deployment {deployment_id} scaled up; launching standby {worker}");
    launch_slot(
        node,
        locks,
        inherited_environment,
        slot,
        shutdown,
        supervisor_started_at,
        metrics,
    )
    .await;
    if slot.process.is_none() {
        ledger.release(&worker);
        slot.autoscale = Some(AutoscaleSlotState {
            deployment: deployment_id.clone(),
            phase: AutoscalePhase::Standby,
        });
        eprintln!("autoscaling: standby {worker} failed its launch; it remains standby");
    }
}

/// Scale-down step 1: remove the worker's approvals line (the gateway stops
/// routing new work after its view refresh) and close the ownership pipe so
/// the worker stops admission immediately, covering the gateway's refresh
/// TTL gap.
#[allow(clippy::too_many_arguments)]
async fn mark_draining(
    autoscaler: &mut Autoscaler,
    node: &ValidatedNodeConfig,
    deployment_id: &DeploymentId,
    approvals_path: &Path,
    slots: &mut [WorkerSlot],
    now_ms: u64,
    metrics: &mut SupervisorMetrics,
    worker: WorkerId,
) {
    if let Err(error) = autoscale::remove_worker(approvals_path, node, &worker) {
        eprintln!(
            "autoscaling: could not unapprove worker {worker}: {error}; retrying next evaluation"
        );
        return;
    }
    let Some(slot) = slots.iter_mut().find(|slot| slot.worker_id == worker) else {
        return;
    };
    if let Some(process) = slot.process.as_mut() {
        if let Err(error) = process.begin_drain().await {
            eprintln!("autoscaling: worker {worker} admission stop failed: {error}");
        }
    }
    slot.autoscale = Some(AutoscaleSlotState {
        deployment: deployment_id.clone(),
        phase: AutoscalePhase::Draining,
    });
    autoscaler
        .deployment_mut(deployment_id)
        .expect("deployment ids come from the autoscaler")
        .note_draining(now_ms, &worker);
    metrics.autoscale_downs = metrics.autoscale_downs.saturating_add(1);
    eprintln!(
        "autoscaling: worker {worker} marked draining; waiting for zero active work before stopping it"
    );
}

/// Scale-down step 2: the worker is fully quiet — stop it via the existing
/// drain_and_stop path (mostly-immediate cooperative stop) and return the
/// slot to the standby pool.
async fn stop_drained(
    autoscaler: &mut Autoscaler,
    node: &ValidatedNodeConfig,
    deployment_id: &DeploymentId,
    ledger: &mut ResourceLedger,
    slots: &mut [WorkerSlot],
    metrics: &mut SupervisorMetrics,
    worker: WorkerId,
) {
    let Some(slot) = slots.iter_mut().find(|slot| slot.worker_id == worker) else {
        return;
    };
    if let Some(process) = slot.process.take() {
        match process
            .drain_and_stop(&node.config().shutdown.clone())
            .await
        {
            Ok(report) => eprintln!(
                "autoscaling: worker {worker} stopped with {:?} ({})",
                report.outcome, report.exit_status
            ),
            Err(error) => {
                metrics.stop_failures = metrics.stop_failures.saturating_add(1);
                eprintln!("autoscaling: worker {worker} stop failed: {error}");
            }
        }
    }
    slot.process_started_at = None;
    slot.autoscale = Some(AutoscaleSlotState {
        deployment: deployment_id.clone(),
        phase: AutoscalePhase::Standby,
    });
    ledger.release(&worker);
    autoscaler
        .deployment_mut(deployment_id)
        .expect("deployment ids come from the autoscaler")
        .note_stopped();
    eprintln!(
        "autoscaling: worker {worker} drained and stopped; the slot returned to the standby pool"
    );
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
        "Rollout (DS6): --rollout-plan PATH starts or resumes a blue-green rollout declared by the plan;\n\
  --rollout-abort restores the pre-rollout approvals and clears state; --rollout-status reports persisted rollout state.\n\
  All three require --config."
    );
    #[cfg(unix)]
    eprintln!(
        "Send SIGUSR1 to a running supervisor for one bounded, redacted status snapshot on stderr; SIGUSR2 aborts a running rollout while it is still reversible."
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
            }],
            readiness: ReadinessPolicy::default(),
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
            autoscale: None,
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
