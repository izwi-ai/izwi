//! Feeding worker observations to the autoscaler and applying its decisions.

use izwi_serving_protocol::DeploymentId;
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info, warn};

use super::{monotonic_ms, AutoscalePhase, AutoscaleSlotState, WorkerSlot};
use crate::actor::{bump, Supervision, SupervisorMetrics, WorkerPhase};
use crate::autoscale::{Autoscaler, DeploymentObservation, ScaleDecision, WorkerSignals};
use crate::{autoscale, ValidatedNodeConfig};

/// Keeps the autoscaler's running set in step with the actors: a crash of a
/// restart-managed worker leaves it, a relaunch that reached readiness
/// rejoins it, and a scale-down drain that ended returns its slot to the
/// standby pool.
pub(crate) fn sync_autoscaler(autoscaler: &mut Autoscaler, slots: &mut [WorkerSlot]) {
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
pub(crate) async fn run_autoscale_tick(
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
pub(crate) async fn poll_deployment(
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
