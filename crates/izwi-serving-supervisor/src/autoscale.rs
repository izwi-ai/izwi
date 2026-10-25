//! DS7 signal-driven worker autoscaling primitives.
//!
//! Two process-free concerns live here, each unit-testable in isolation:
//!
//! 1. The per-deployment scale state machine: capacity signals → scale
//!    decisions under the configured bounds and the per-direction
//!    stabilization windows.
//!    Decisions are pure observations; the supervision driver confirms the
//!    lifecycle transitions it actually performed, so a failed launch or a
//!    failed approvals write never desynchronizes the state machine.
//! 2. The shared approvals view editor: this node's autoscaled workers are
//!    approved through v1 pinned lines that the supervisor alone adds and
//!    removes, under the shared file lock and preserving every unrelated
//!    line verbatim (see [`crate::approvals_file`]).
//!
//! Scale-ups need no resource check of their own: config validation proves
//! that every declared worker, standbys included, fits the node at once.
//!
//! Scale state is intentionally in-memory: a supervisor restart relaunches
//! the declared min set and reconciles the view to it, which is the fail-closed
//! posture (a crashed supervisor never leaves phantom capacity behind).

use crate::approvals_file::{self, ApprovalsDocument, SharedApprovalsError};
use crate::{DeploymentAutoscalingPolicy, ValidatedNodeConfig, WorkerConfig};
use izwi_serving_protocol::{
    DeploymentId, GatewayWorkerApproval, GatewayWorkerApprovalIdentity, NodeId, WorkerId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

/// Capacity signals observed from one running worker during an evaluation
/// tick (a projection of the worker status capacity snapshot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerSignals {
    pub queued_invocations: u64,
    pub active_invocations: u32,
    pub reserved_sessions: u32,
}

impl WorkerSignals {
    /// A worker counts as busy while it has queued or active invocations or
    /// reserved realtime sessions; scale-down waits for a fully quiet worker.
    pub fn is_busy(&self) -> bool {
        self.queued_invocations > 0 || self.active_invocations > 0 || self.reserved_sessions > 0
    }

    /// Invocations the worker holds: running plus queued. Workers do not
    /// queue today, so this is their active count.
    pub fn outstanding(&self) -> u64 {
        self.queued_invocations
            .saturating_add(u64::from(self.active_invocations))
    }
}

/// Observations for one evaluation tick of one deployment. The driver only
/// includes signals from workers it successfully polled, and it skips a
/// deployment's evaluation entirely when any of its running workers failed
/// to answer — no capacity decision is ever made on partial observability.
#[derive(Debug, Clone, Default)]
pub struct DeploymentObservation {
    pub signals: BTreeMap<WorkerId, WorkerSignals>,
}

/// One scale decision. Decisions never mutate the state machine; the driver
/// confirms the outcomes it observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaleDecision {
    NoAction,
    /// Activate this standby through the existing launch/readiness path.
    ScaleUp {
        worker: WorkerId,
    },
    /// Remove the worker's approvals line and close its ownership pipe
    /// (stop admission); keep observing until its work drains.
    MarkDraining {
        worker: WorkerId,
    },
    /// The draining worker is fully quiet; stop it and return the slot to
    /// the standby pool.
    StopDrained {
        worker: WorkerId,
    },
}

/// Per-deployment scale state. `min_workers` of the declared workers (config
/// order) are the core set: they launch at startup and never scale down, so
/// the static posture is always recoverable. Scale-up activates standbys in
/// config order; scale-down drains non-core workers, preferring the first
/// fully-idle eligible candidate in config order (deterministic).
#[derive(Debug)]
pub struct DeploymentScaleState {
    deployment_id: DeploymentId,
    policy: DeploymentAutoscalingPolicy,
    worker_order: Vec<WorkerId>,
    active: BTreeSet<WorkerId>,
    draining: Option<WorkerId>,
    /// Workers whose process went down without a scale decision (crashed,
    /// restarting or quarantined). Scale-up never picks one as a standby:
    /// the restart controller owns whether it comes back.
    lost: BTreeSet<WorkerId>,
    sustained_up_polls: u32,
    last_scale_event_ms: Option<u64>,
    idle_since_ms: BTreeMap<WorkerId, u64>,
}

impl DeploymentScaleState {
    pub fn new(
        deployment_id: DeploymentId,
        policy: DeploymentAutoscalingPolicy,
        worker_order: Vec<WorkerId>,
    ) -> Self {
        assert_eq!(
            worker_order.len(),
            policy.max_workers,
            "the declared worker order must cover the policy's max_workers"
        );
        Self {
            deployment_id,
            policy,
            worker_order,
            active: BTreeSet::new(),
            draining: None,
            lost: BTreeSet::new(),
            sustained_up_polls: 0,
            last_scale_event_ms: None,
            idle_since_ms: BTreeMap::new(),
        }
    }

    pub fn deployment_id(&self) -> &DeploymentId {
        &self.deployment_id
    }

    pub fn policy(&self) -> &DeploymentAutoscalingPolicy {
        &self.policy
    }

    /// Declared workers in config order; indexes below `min_workers` are the
    /// startup core set.
    pub fn worker_order(&self) -> &[WorkerId] {
        &self.worker_order
    }

    /// The workers that launch at startup (config order, `min_workers` long).
    pub fn core_set(&self) -> &[WorkerId] {
        let core = self.worker_order.len().min(self.policy.min_workers);
        &self.worker_order[..core]
    }

    pub fn active(&self) -> &BTreeSet<WorkerId> {
        &self.active
    }

    pub fn draining(&self) -> Option<&WorkerId> {
        self.draining.as_ref()
    }

    pub fn is_core(&self, worker: &WorkerId) -> bool {
        self.core_set().contains(worker)
    }

    pub fn last_scale_event_ms(&self) -> Option<u64> {
        self.last_scale_event_ms
    }

    /// Driver confirmation: a launch reached readiness as part of the
    /// startup min set. Records running capacity but is not a scale event.
    pub fn note_launched(&mut self, worker: &WorkerId) {
        self.lost.remove(worker);
        self.active.insert(worker.clone());
    }

    /// Driver confirmation: a scale-up launch reached readiness. Records the
    /// scale event for hysteresis.
    pub fn note_scaled_up(&mut self, now_ms: u64, worker: &WorkerId) {
        self.lost.remove(worker);
        self.active.insert(worker.clone());
        self.last_scale_event_ms = Some(now_ms);
    }

    /// Driver confirmation: a scale-up launch failed; the slot stays standby.
    pub fn note_launch_abandoned(&mut self, worker: &WorkerId) {
        self.active.remove(worker);
    }

    /// Driver confirmation: a process exited unexpectedly (crash). The
    /// restart controller owns whether it comes back; the autoscaler stops
    /// counting it toward the deployment's running capacity and never picks
    /// it as a standby until a relaunch confirms it again.
    pub fn note_worker_lost(&mut self, worker: &WorkerId) {
        self.active.remove(worker);
        self.idle_since_ms.remove(worker);
        self.lost.insert(worker.clone());
    }

    /// Driver confirmation: the approvals line was removed and admission
    /// stopped; the worker is now draining for scale-down. The scale event
    /// is recorded at initiation — the deployment's capacity posture changed
    /// the moment the worker left the eligible pool.
    pub fn note_draining(&mut self, now_ms: u64, worker: &WorkerId) {
        self.draining = Some(worker.clone());
        self.last_scale_event_ms = Some(now_ms);
    }

    /// Driver confirmation: the drained worker stopped; the slot is standby
    /// again.
    pub fn note_stopped(&mut self) {
        if let Some(worker) = self.draining.take() {
            self.active.remove(&worker);
            self.idle_since_ms.remove(&worker);
        }
    }

    fn next_standby(&self) -> Option<WorkerId> {
        self.worker_order
            .iter()
            .find(|worker| {
                !self.active.contains(*worker)
                    && !self.lost.contains(*worker)
                    && !self.is_core(worker)
            })
            .cloned()
    }

    /// One evaluation step. Emits at most one decision per call; a decision
    /// only takes effect through its driver confirmation, so repeated calls
    /// before confirmation re-emit the same decision without corrupting the
    /// recorded transitions (the sustained and idle counters are
    /// observation-derived bookkeeping and update every tick).
    pub fn evaluate(&mut self, now_ms: u64, observation: &DeploymentObservation) -> ScaleDecision {
        if let Some(worker) = self.draining.clone() {
            let quiet = observation
                .signals
                .get(&worker)
                .is_none_or(|signals| !signals.is_busy());
            return if quiet {
                ScaleDecision::StopDrained { worker }
            } else {
                ScaleDecision::NoAction
            };
        }

        for (worker, signals) in &observation.signals {
            if signals.is_busy() {
                self.idle_since_ms.remove(worker);
            } else {
                self.idle_since_ms.entry(worker.clone()).or_insert(now_ms);
            }
        }
        self.idle_since_ms
            .retain(|worker, _| self.active.contains(worker));

        let up_allowed = self
            .policy
            .scale_up_allowed(now_ms, self.last_scale_event_ms);
        let max_outstanding = observation
            .signals
            .values()
            .map(WorkerSignals::outstanding)
            .max()
            .unwrap_or(0);
        let saturated = max_outstanding >= self.policy.scale_up_queue_depth;
        if saturated && up_allowed {
            self.sustained_up_polls = self.sustained_up_polls.saturating_add(1);
        } else {
            self.sustained_up_polls = 0;
        }
        if saturated
            && up_allowed
            && self.sustained_up_polls >= self.policy.scale_up_sustained_polls
            && self.active.len() < self.policy.max_workers
        {
            if let Some(worker) = self.next_standby() {
                return ScaleDecision::ScaleUp { worker };
            }
        }

        let down_allowed = self
            .policy
            .scale_down_allowed(now_ms, self.last_scale_event_ms);
        if down_allowed && self.active.len() > self.policy.min_workers {
            let window = self.policy.scale_down_stabilization_window_ms;
            let candidate = self
                .worker_order
                .iter()
                .filter(|worker| self.active.contains(*worker) && !self.is_core(worker))
                .find(|worker| {
                    self.idle_since_ms
                        .get(*worker)
                        .is_some_and(|since| now_ms.saturating_sub(*since) >= window)
                })
                .cloned();
            if let Some(worker) = candidate {
                return ScaleDecision::MarkDraining { worker };
            }
        }

        ScaleDecision::NoAction
    }
}

/// Node-level autoscaler built from a validated config; `None` when the
/// autoscaling block is absent (the static supervisor posture).
#[derive(Debug)]
pub struct Autoscaler {
    deployments: BTreeMap<DeploymentId, DeploymentScaleState>,
    evaluation_interval: Duration,
}

impl Autoscaler {
    pub fn from_config(node: &ValidatedNodeConfig) -> Option<Self> {
        let autoscaling = node.config().autoscaling.as_ref()?;
        let mut deployments = BTreeMap::new();
        for (deployment_name, policy) in &autoscaling.deployments {
            let worker_order: Vec<WorkerId> = node
                .config()
                .workers
                .iter()
                .filter(|worker| worker.deployment.deployment_id.as_str() == deployment_name)
                .map(|worker| worker.worker_id.clone())
                .collect();
            let deployment_id = DeploymentId::new(deployment_name.clone())
                .expect("validated deployment identifiers reparse");
            deployments.insert(
                deployment_id.clone(),
                DeploymentScaleState::new(deployment_id, policy.clone(), worker_order),
            );
        }
        Some(Self {
            deployments,
            evaluation_interval: Duration::from_millis(autoscaling.evaluation_interval_ms),
        })
    }

    pub fn evaluation_interval(&self) -> Duration {
        self.evaluation_interval
    }

    pub fn deployments(&self) -> &BTreeMap<DeploymentId, DeploymentScaleState> {
        &self.deployments
    }

    pub fn deployment(&self, deployment_id: &DeploymentId) -> Option<&DeploymentScaleState> {
        self.deployments.get(deployment_id)
    }

    pub fn deployment_mut(
        &mut self,
        deployment_id: &DeploymentId,
    ) -> Option<&mut DeploymentScaleState> {
        self.deployments.get_mut(deployment_id)
    }

    /// The deployment that owns a worker, if the worker is autoscaled.
    pub fn deployment_of(&self, worker: &WorkerId) -> Option<&DeploymentScaleState> {
        self.deployments
            .values()
            .find(|deployment| deployment.worker_order.contains(worker))
    }

    /// The deployment owning a worker, mutably.
    pub fn deployment_of_mut(&mut self, worker: &WorkerId) -> Option<&mut DeploymentScaleState> {
        self.deployments
            .values_mut()
            .find(|deployment| deployment.worker_order.contains(worker))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AutoscaleError {
    #[error("shared approvals view is unusable: {0}")]
    View(String),
    #[error(transparent)]
    File(#[from] SharedApprovalsError),
    #[error(
        "standalone approval line for {endpoint} collides with this node's autoscaled worker {worker}; the fleet profile requires v1 pinned lines"
    )]
    CollidingStandaloneLine { endpoint: String, worker: WorkerId },
}

/// The autoscaled workers declared on this node, keyed for line ownership.
struct AutoscaledWorkers<'a> {
    node_id: &'a NodeId,
    workers: Vec<&'a WorkerConfig>,
}

impl<'a> AutoscaledWorkers<'a> {
    fn from_node(node: &'a ValidatedNodeConfig) -> Result<Self, AutoscaleError> {
        let Some(autoscaling) = node.config().autoscaling.as_ref() else {
            return Err(AutoscaleError::View(
                "the node config does not enable autoscaling".to_string(),
            ));
        };
        let workers: Vec<&WorkerConfig> = node
            .config()
            .workers
            .iter()
            .filter(|worker| {
                autoscaling
                    .deployments
                    .contains_key(worker.deployment.deployment_id.as_str())
            })
            .collect();
        Ok(Self {
            node_id: &node.config().node_id,
            workers,
        })
    }

    fn worker(&self, worker_id: &WorkerId) -> Result<&'a WorkerConfig, AutoscaleError> {
        self.workers
            .iter()
            .copied()
            .find(|worker| &worker.worker_id == worker_id)
            .ok_or_else(|| {
                AutoscaleError::View(format!(
                    "worker {worker_id} is not a declared autoscaled worker of this node"
                ))
            })
    }

    /// Rejects a standalone-form line whose endpoint equals one of this
    /// node's autoscaled workers: the supervisor cannot own a line it cannot
    /// pin, so the operator must migrate it to the v1 fleet form.
    fn reject_standalone_collisions(
        &self,
        document: &ApprovalsDocument,
    ) -> Result<(), AutoscaleError> {
        for approval in document.approvals() {
            if let GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint =
                approval.identity
            {
                if let Some(worker) = self
                    .workers
                    .iter()
                    .find(|worker| approval.endpoint == format!("http://{}", worker.bind))
                {
                    return Err(AutoscaleError::CollidingStandaloneLine {
                        endpoint: approval.endpoint.clone(),
                        worker: worker.worker_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Whether a line is a v1 pinned line of this node (optionally narrowed
    /// to one worker).
    fn is_own_line(&self, approval: &GatewayWorkerApproval, worker: Option<&WorkerId>) -> bool {
        let GatewayWorkerApprovalIdentity::V1 { node_id, worker_id } = &approval.identity else {
            return false;
        };
        if node_id != self.node_id {
            return false;
        }
        match worker {
            Some(worker) => worker == worker_id,
            None => self
                .workers
                .iter()
                .any(|declared| &declared.worker_id == worker_id),
        }
    }
}

/// Startup reconciliation: the view approves exactly the min-set workers of
/// this node's autoscaled deployments — lines for other workers of those
/// deployments are removed (standbys are not running), every unrelated line
/// passes through verbatim, and a missing file is created.
pub fn reconcile_min_set(
    path: &Path,
    node: &ValidatedNodeConfig,
    desired: &[&WorkerConfig],
) -> Result<(), AutoscaleError> {
    let own = AutoscaledWorkers::from_node(node)?;
    let desired: Vec<GatewayWorkerApproval> = desired
        .iter()
        .map(|worker| approvals_file::pinned_approval(own.node_id, worker))
        .collect();
    approvals_file::edit(path, |document| {
        own.reject_standalone_collisions(document)?;
        document.retain(|approval| !own.is_own_line(approval, None) || desired.contains(approval));
        for approval in &desired {
            document.ensure(approval);
        }
        Ok(())
    })
    .map(drop)
}

/// Scale-up: appends one worker's pinned line if it is not already present.
pub fn add_worker(
    path: &Path,
    node: &ValidatedNodeConfig,
    worker_id: &WorkerId,
) -> Result<(), AutoscaleError> {
    let own = AutoscaledWorkers::from_node(node)?;
    let worker = own.worker(worker_id)?;
    let line = approvals_file::pinned_approval(own.node_id, worker);
    approvals_file::edit(path, |document| {
        own.reject_standalone_collisions(document)?;
        document.ensure(&line);
        Ok(())
    })
    .map(drop)
}

/// Scale-down: removes one worker's pinned line (idempotent).
pub fn remove_worker(
    path: &Path,
    node: &ValidatedNodeConfig,
    worker_id: &WorkerId,
) -> Result<(), AutoscaleError> {
    let own = AutoscaledWorkers::from_node(node)?;
    own.worker(worker_id)?;
    approvals_file::edit(path, |document| {
        own.reject_standalone_collisions(document)?;
        document.retain(|approval| !own.is_own_line(approval, Some(worker_id)));
        Ok(())
    })
    .map(drop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BinaryCatalog, BinaryRecord, CapabilityProfileConfig, DeploymentConfig, HostInventory,
        NodeConfig, WorkerBinaryFlavor,
    };
    use izwi_serving_protocol::{
        BackendKind, CancellationBehavior, DeviceAssignment, InputFormat, ModelGeneration,
        OutputFormat, TaskKind,
    };
    use std::collections::BTreeSet;
    use std::io::Write as IoWrite;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn signals(queued: u64, active: u32) -> WorkerSignals {
        WorkerSignals {
            queued_invocations: queued,
            active_invocations: active,
            reserved_sessions: 0,
        }
    }

    fn observation(entries: &[(&'static str, WorkerSignals)]) -> DeploymentObservation {
        DeploymentObservation {
            signals: entries
                .iter()
                .map(|(worker, signals)| (id::<WorkerId>(worker), *signals))
                .collect(),
        }
    }

    fn policy(min_workers: usize, max_workers: usize) -> DeploymentAutoscalingPolicy {
        DeploymentAutoscalingPolicy {
            min_workers,
            max_workers,
            scale_up_queue_depth: 4,
            scale_up_sustained_polls: 3,
            scale_up_stabilization_window_ms: 0,
            scale_down_stabilization_window_ms: 1_000,
        }
    }

    fn state(min_workers: usize, max_workers: usize) -> DeploymentScaleState {
        let declared: Vec<WorkerId> = ["worker-a", "worker-b", "worker-c"]
            .into_iter()
            .take(max_workers)
            .map(id)
            .collect();
        DeploymentScaleState::new(
            id("deployment-1"),
            policy(min_workers, max_workers),
            declared,
        )
    }

    #[test]
    fn sustained_queue_depth_scales_up_then_respects_the_ceiling() {
        let mut state = state(1, 2);
        state.note_launched(&id("worker-a"));

        let busy = observation(&[("worker-a", signals(4, 1))]);
        assert_eq!(state.evaluate(0, &busy), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(1, &busy), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(2, &busy),
            ScaleDecision::ScaleUp {
                worker: id("worker-b")
            }
        );
        state.note_scaled_up(2, &id("worker-b"));

        // At max_workers no further scale-up fires even while saturated.
        assert_eq!(state.evaluate(3, &busy), ScaleDecision::NoAction);
    }

    #[test]
    fn queue_depth_must_be_sustained_across_evaluations() {
        let mut state = state(1, 2);
        state.note_launched(&id("worker-a"));

        let busy = observation(&[("worker-a", signals(9, 1))]);
        assert_eq!(state.evaluate(0, &busy), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(1, &busy), ScaleDecision::NoAction);

        // One quiet evaluation resets the sustained counter.
        let quiet = observation(&[("worker-a", signals(0, 0))]);
        assert_eq!(state.evaluate(2, &quiet), ScaleDecision::NoAction);

        assert_eq!(state.evaluate(3, &busy), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(4, &busy), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(5, &busy),
            ScaleDecision::ScaleUp {
                worker: id("worker-b")
            }
        );
    }

    #[test]
    fn hysteresis_blocks_scale_down_within_the_stabilization_window() {
        let mut state = state(1, 2);
        state.note_launched(&id("worker-a"));
        state.note_scaled_up(0, &id("worker-b"));

        let idle = observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        // Both workers go idle at t=0, the same instant as the scale-up
        // event: the hysteresis window pins the scale-down until t=1000.
        assert_eq!(state.evaluate(0, &idle), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(500, &idle), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(1_000, &idle),
            ScaleDecision::MarkDraining {
                worker: id("worker-b")
            }
        );
    }

    #[test]
    fn active_invocations_count_toward_the_scale_up_threshold() {
        let mut state = state(1, 2);
        state.note_launched(&id("worker-a"));
        // Workers do not queue: a full worker reports active work only.
        let full = observation(&[("worker-a", signals(0, 4))]);
        assert_eq!(state.evaluate(0, &full), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(1, &full), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(2, &full),
            ScaleDecision::ScaleUp {
                worker: id("worker-b")
            }
        );
    }

    /// Scale out right after a scale-in: the scale-down window must not
    /// hold back a burst (scale-up stabilization defaults to zero).
    #[test]
    fn a_burst_scales_up_right_after_a_scale_down() {
        let mut state = state(1, 2);
        state.note_launched(&id("worker-a"));
        state.note_scaled_up(0, &id("worker-b"));
        let idle = observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        assert_eq!(state.evaluate(0, &idle), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(1_000, &idle),
            ScaleDecision::MarkDraining {
                worker: id("worker-b")
            }
        );
        state.note_draining(1_000, &id("worker-b"));
        assert_eq!(
            state.evaluate(1_001, &idle),
            ScaleDecision::StopDrained {
                worker: id("worker-b")
            }
        );
        state.note_stopped();

        let busy = observation(&[("worker-a", signals(0, 4))]);
        assert_eq!(state.evaluate(1_002, &busy), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(1_003, &busy), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(1_004, &busy),
            ScaleDecision::ScaleUp {
                worker: id("worker-b")
            }
        );
    }

    #[test]
    fn the_scale_up_window_holds_scale_up_after_a_scale_event() {
        let mut policy = policy(1, 3);
        policy.scale_up_stabilization_window_ms = 500;
        let mut state = DeploymentScaleState::new(
            id("deployment-1"),
            policy,
            vec![id("worker-a"), id("worker-b"), id("worker-c")],
        );
        state.note_launched(&id("worker-a"));
        state.note_scaled_up(0, &id("worker-b"));
        let busy = observation(&[("worker-a", signals(0, 4)), ("worker-b", signals(0, 4))]);
        for now in [100, 200, 300, 499] {
            assert_eq!(state.evaluate(now, &busy), ScaleDecision::NoAction);
        }
        assert_eq!(state.evaluate(500, &busy), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(501, &busy), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(502, &busy),
            ScaleDecision::ScaleUp {
                worker: id("worker-c")
            }
        );
    }

    #[test]
    fn a_crashed_worker_is_never_picked_as_a_standby() {
        let mut state = state(1, 3);
        state.note_launched(&id("worker-a"));
        state.note_scaled_up(0, &id("worker-b"));
        // worker-b crashes: the restart controller owns it now.
        state.note_worker_lost(&id("worker-b"));

        let busy = observation(&[("worker-a", signals(0, 4))]);
        assert_eq!(state.evaluate(1, &busy), ScaleDecision::NoAction);
        assert_eq!(state.evaluate(2, &busy), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(3, &busy),
            ScaleDecision::ScaleUp {
                worker: id("worker-c")
            },
            "scale-up must take the next free standby, not the crashed worker"
        );

        // Once its restart reaches readiness it counts as running again.
        state.note_launched(&id("worker-b"));
        assert!(state.active().contains(&id("worker-b")));
    }

    #[test]
    fn scale_down_waits_for_a_fully_quiet_worker_then_stops_it() {
        let mut state = state(1, 2);
        state.note_launched(&id("worker-a"));
        state.note_scaled_up(0, &id("worker-b"));

        let idle = observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        // Idle from t=0; the scale-down eligibility window closes at t=1000.
        assert_eq!(state.evaluate(0, &idle), ScaleDecision::NoAction);
        assert_eq!(
            state.evaluate(1_000, &idle),
            ScaleDecision::MarkDraining {
                worker: id("worker-b")
            }
        );
        state.note_draining(1_000, &id("worker-b"));

        // In-flight work keeps the draining worker in place.
        let draining_busy =
            observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 1))]);
        assert_eq!(
            state.evaluate(1_100, &draining_busy),
            ScaleDecision::NoAction
        );
        let draining_session =
            observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        // Reserved sessions also count as outstanding work.
        let mut with_session = draining_session.clone();
        with_session
            .signals
            .get_mut(&id("worker-b"))
            .unwrap()
            .reserved_sessions = 2;
        assert_eq!(
            state.evaluate(1_200, &with_session),
            ScaleDecision::NoAction
        );

        assert_eq!(
            state.evaluate(1_300, &draining_session),
            ScaleDecision::StopDrained {
                worker: id("worker-b")
            }
        );
        state.note_stopped();
        assert_eq!(state.active().len(), 1);
        assert!(state.draining().is_none());
    }

    #[test]
    fn scale_down_never_touches_the_core_set_or_dips_below_min() {
        let mut state = state(2, 3);
        state.note_launched(&id("worker-a"));
        state.note_launched(&id("worker-b"));

        let idle = observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        // Active == min: nothing to scale down even though both are idle.
        assert_eq!(state.evaluate(10_000, &idle), ScaleDecision::NoAction);

        state.note_scaled_up(10_000, &id("worker-c"));
        // Only the core workers are idle; the non-core worker is busy.
        let mixed = observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        assert_eq!(state.evaluate(20_000, &mixed), ScaleDecision::NoAction);
    }

    #[test]
    fn a_quiesced_static_policy_never_scales() {
        let mut state = state(2, 2);
        state.note_launched(&id("worker-a"));
        state.note_launched(&id("worker-b"));

        let saturated = observation(&[("worker-a", signals(50, 1)), ("worker-b", signals(50, 1))]);
        for now in [0, 1, 2, 3, 4] {
            assert_eq!(state.evaluate(now, &saturated), ScaleDecision::NoAction);
        }
        let idle = observation(&[("worker-a", signals(0, 0)), ("worker-b", signals(0, 0))]);
        assert_eq!(state.evaluate(10_000, &idle), ScaleDecision::NoAction);
    }

    #[test]
    fn busy_activity_resets_the_idle_window_and_candidates_are_deterministic() {
        let mut state = state(1, 3);
        state.note_launched(&id("worker-a"));
        state.note_scaled_up(0, &id("worker-b"));
        state.note_scaled_up(0, &id("worker-c"));

        // Both non-core workers idle at t=0, then worker-b turns busy.
        let idle = observation(&[
            ("worker-a", signals(0, 0)),
            ("worker-b", signals(0, 0)),
            ("worker-c", signals(0, 0)),
        ]);
        let _ = state.evaluate(0, &idle);
        let busy_b = observation(&[
            ("worker-a", signals(0, 0)),
            ("worker-b", signals(1, 1)),
            ("worker-c", signals(0, 0)),
        ]);
        let _ = state.evaluate(900, &busy_b);

        // At t=1000 worker-b is idle again (since 1000) but worker-c has
        // been idle since 0; worker-c is the eligible candidate.
        assert_eq!(
            state.evaluate(1_000, &idle),
            ScaleDecision::MarkDraining {
                worker: id("worker-c")
            }
        );
    }

    fn executable(directory: &std::path::Path) -> std::path::PathBuf {
        let path = directory.join("worker");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"worker")
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        path
    }

    /// Two CPU workers on one deployment with autoscaling min=1 max=2.
    fn autoscaled_node(directory: &std::path::Path) -> ValidatedNodeConfig {
        let inventory = HostInventory {
            effective_cpu_ids: vec![0, 1, 2, 3],
            allocatable_host_memory_bytes: 2048,
            metal_devices: Vec::new(),
            cuda_devices: Vec::new(),
        };
        let binaries = BinaryCatalog::new([(
            WorkerBinaryFlavor::Cpu,
            BinaryRecord {
                path: executable(directory),
                supported_backends: vec![BackendKind::Cpu],
            },
        )]);
        let worker = |name: &'static str, port: u16| crate::WorkerConfig {
            worker_id: id(name),
            bind: format!("127.0.0.1:{port}").parse().unwrap(),
            binary: WorkerBinaryFlavor::Cpu,
            credential_id: id("credential-1"),
            bearer_token_env: "TEST_WORKER_TOKEN".into(),
            assignment: DeviceAssignment::Cpu {
                thread_budget: 2,
                affinity: vec![0, 1],
                host_memory_limit_bytes: 512,
            },
            deployment: DeploymentConfig {
                deployment_id: id("deployment-1"),
                public_model: id("model-1"),
                artifact_revision: id("revision-1"),
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
                    max_input_bytes: 4096,
                    max_context_tokens: Some(32),
                    max_output_tokens: Some(32),
                },
                models_directory: directory.to_path_buf(),
            },
            max_active_invocations: Some(4),
            max_request_bytes: 4096,
            max_retained_attempts: 4,
            attempt_retention_secs: 30,
            streaming: true,
            host_kv_pool_budget_bytes: 0,
            environment: Default::default(),
        };
        let config = NodeConfig {
            schema_version: crate::NODE_CONFIG_SCHEMA_VERSION,
            node_id: id("node-a"),
            working_directory: directory.to_path_buf(),
            runtime_directory: directory.join("run"),
            host_memory_budget_bytes: 2048,
            workers: vec![worker("cpu-1", 9471), worker("cpu-2", 9472)],
            readiness: Default::default(),
            liveness: Default::default(),
            restart: Default::default(),
            shutdown: Default::default(),
            max_parallel_model_loads: crate::DEFAULT_MODEL_LOAD_SLOTS,
            autoscaling: Some(crate::AutoscalingConfig {
                shared_approvals_path: directory.join("shared-approvals"),
                evaluation_interval_ms: crate::DEFAULT_AUTOSCALING_EVALUATION_INTERVAL_MS,
                deployments: BTreeMap::from([("deployment-1".to_string(), policy(1, 2))]),
            }),
        };
        config.validate(&inventory, &binaries).unwrap()
    }

    fn view(path: &std::path::Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn reconcile_creates_the_min_set_view_idempotently() {
        let directory = tempfile::tempdir().unwrap();
        let node = autoscaled_node(directory.path());
        let path = directory.path().join("shared-approvals");
        let core = node.worker(&id("cpu-1")).unwrap();

        reconcile_min_set(&path, &node, &[core]).unwrap();
        let first = view(&path);
        assert_eq!(first.len(), 1, "only the core line is approved: {first:?}");
        assert!(first[0].starts_with("v1|http://127.0.0.1:9471|node-a|cpu-1|"));

        reconcile_min_set(&path, &node, &[core]).unwrap();
        assert_eq!(view(&path), first, "reconciliation is idempotent");
    }

    #[cfg(unix)]
    #[test]
    fn reconciled_views_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let node = autoscaled_node(directory.path());
        let path = directory.path().join("shared-approvals");
        let core = node.worker(&id("cpu-1")).unwrap();

        reconcile_min_set(&path, &node, &[core]).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "approvals views are owner-only");
    }

    #[test]
    fn reconcile_removes_standby_lines_and_preserves_foreign_lines() {
        let directory = tempfile::tempdir().unwrap();
        let node = autoscaled_node(directory.path());
        let path = directory.path().join("shared-approvals");
        let foreign =
            "v1|https://worker-b.internal:9470|node-z|worker-z|chat|model-1|deployment-1|1";
        std::fs::write(
            &path,
            format!(
                "{foreign}\nv1|http://127.0.0.1:9471|node-a|cpu-1|chat|model-1|deployment-1|1\nv1|http://127.0.0.1:9472|node-a|cpu-2|chat|model-1|deployment-1|1\n"
            ),
        )
        .unwrap();

        let core = node.worker(&id("cpu-1")).unwrap();
        reconcile_min_set(&path, &node, &[core]).unwrap();
        let lines = view(&path);
        assert_eq!(lines.len(), 2, "foreign line plus own core line: {lines:?}");
        assert_eq!(lines[0], foreign);
        assert!(lines[1].contains("|node-a|cpu-1|"));
        assert!(
            !lines.iter().any(|line| line.contains("|node-a|cpu-2|")),
            "the standby line must be removed: {lines:?}"
        );
    }

    #[test]
    fn scale_up_adds_and_scale_down_removes_the_worker_line() {
        let directory = tempfile::tempdir().unwrap();
        let node = autoscaled_node(directory.path());
        let path = directory.path().join("shared-approvals");
        let core = node.worker(&id("cpu-1")).unwrap();
        reconcile_min_set(&path, &node, &[core]).unwrap();

        add_worker(&path, &node, &id("cpu-2")).unwrap();
        assert_eq!(view(&path).len(), 2);
        add_worker(&path, &node, &id("cpu-2")).unwrap();
        assert_eq!(view(&path).len(), 2, "adding twice is idempotent");

        remove_worker(&path, &node, &id("cpu-2")).unwrap();
        assert_eq!(view(&path).len(), 1);
        remove_worker(&path, &node, &id("cpu-2")).unwrap();
        assert_eq!(view(&path).len(), 1, "removing twice is idempotent");
    }

    #[test]
    fn a_standalone_line_for_an_own_endpoint_fails_closed() {
        let directory = tempfile::tempdir().unwrap();
        let node = autoscaled_node(directory.path());
        let path = directory.path().join("shared-approvals");
        std::fs::write(&path, "http://127.0.0.1:9471|chat|model-1|deployment-1|1\n").unwrap();

        let core = node.worker(&id("cpu-1")).unwrap();
        let error = reconcile_min_set(&path, &node, &[core]).unwrap_err();
        assert!(matches!(
            error,
            AutoscaleError::CollidingStandaloneLine { .. }
        ));
    }
}
