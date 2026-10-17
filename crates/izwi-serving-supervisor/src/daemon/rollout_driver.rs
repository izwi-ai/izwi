//! Driving a rolling replacement (blue-green or recreate) from a rollout plan, and aborting one.

use izwi_serving_protocol::{GatewayApprovalsStatus, GatewayWorkerApproval, WorkerId};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::{Instant, MissedTickBehavior};
use tracing::{info, warn};

use super::cli::CliOptions;
use super::error::SupervisorError;
use super::{
    ensure_flavor_binaries, now_ms, read_bounded, resolve_secrets, shut_down_all, WorkerSlot,
};
use crate::actor::{stop_all, LaunchOutcome, WorkerActor, WorkerPhase};
use crate::gateway_ack::GatewayAck;
use crate::rollout::{RolloutPhase, RolloutStrategy};
use crate::{
    approvals_file, rollout, BinaryCatalog, HostInventory, LockNamespace, NodeConfig,
    ResolvedWorkerSecret, ValidatedNodeConfig, MAX_NODE_CONFIG_BYTES,
};

/// A prepared DS6 rollout: validated spec, staged state, target config, and
/// idle replacement actors awaiting launch.
pub(crate) struct PreparedRollout {
    pub(crate) spec: rollout::RolloutSpec,
    pub(crate) state: rollout::RolloutState,
    pub(crate) target: Arc<ValidatedNodeConfig>,
    pub(crate) approvals_path: PathBuf,
    /// The replacement lines the rollout adds to the shared approvals file.
    pub(crate) approval_lines: Vec<GatewayWorkerApproval>,
    pub(crate) replacement_secrets: Vec<(WorkerId, ResolvedWorkerSecret)>,
    pub(crate) replacements: Vec<WorkerActor>,
    /// Whether this run resumes persisted state rather than starting fresh.
    pub(crate) resumed: bool,
    /// The gateways to ask before moving on, when the plan names them.
    pub(crate) ack: Option<GatewayAck>,
}

pub(crate) enum RolloutOutcome {
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

pub(crate) fn prepare_rollout(
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
pub(crate) fn reconcile_fresh_start_state(
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
pub(crate) async fn execute_rollout(
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

pub(crate) fn rollout_abort_command(node: &ValidatedNodeConfig) -> Result<(), SupervisorError> {
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

pub(crate) fn print_rollout_status(config: &NodeConfig) -> Result<(), SupervisorError> {
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
