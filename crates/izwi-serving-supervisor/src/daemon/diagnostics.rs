//! Operator diagnostics: the validation and runtime reports and the Prometheus textfile.

use izwi_serving_protocol::WorkerId;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::{env, io};
use tokio::time::Instant;

use super::{AutoscalePhase, AutoscaleSlotState, WorkerSlot};
use crate::actor::{count, Supervision, SupervisorMetrics, WorkerPhase, WorkerView};
use crate::{ValidatedNodeConfig, WorkerConfig};

pub(crate) const MAX_DIAGNOSTIC_RESPONSE_BYTES: usize = 16 * 1024;
pub(crate) const TRUNCATED_DIAGNOSTIC_SUFFIX: &str = "\ndiagnostic_output=truncated\n";

pub(crate) fn validation_diagnostic(node: &ValidatedNodeConfig) -> String {
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

pub(crate) fn runtime_diagnostic(
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
pub(crate) struct SlotSnapshot<'a> {
    pub(crate) worker_id: &'a WorkerId,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) view: &'a WorkerView,
    pub(crate) autoscale: Option<&'a AutoscaleSlotState>,
}

pub(crate) fn render_runtime_diagnostic<'a>(
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
pub(crate) fn supervisor_prometheus(slots: &[WorkerSlot], metrics: &SupervisorMetrics) -> String {
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
pub(crate) fn write_metrics_textfile(path: &Path, contents: &str) -> io::Result<()> {
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, contents)?;
    std::fs::rename(&temporary, path)
}

pub(crate) struct BoundedDiagnostic {
    pub(crate) output: String,
    pub(crate) truncated: bool,
}

impl BoundedDiagnostic {
    pub(crate) fn new() -> Self {
        Self {
            output: String::with_capacity(MAX_DIAGNOSTIC_RESPONSE_BYTES),
            truncated: false,
        }
    }

    pub(crate) fn push_line(&mut self, line: &str) {
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

    pub(crate) fn finish(self) -> String {
        debug_assert!(self.output.len() <= MAX_DIAGNOSTIC_RESPONSE_BYTES);
        self.output
    }
}
