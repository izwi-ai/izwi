//! DS6 coordinated blue-green rollout: plan, state, and approvals views.
//!
//! A rollout replaces one node's deployment generations in one operator
//! action. The declarative plan names a target node config (the new
//! generations), the shared approvals file the gateway reads, a canary
//! worker, and a soak window. The coordinator state machine is
//!
//! `launching replacement (canary first) → window open (both generations
//! approved; the gateway cuts over when the successor first observes Ready)
//! → draining old → committed`, with an abort path that removes exactly the
//! approval lines the rollout added (so the file is byte-identical to the
//! pre-rollout file unless a peer edited it meanwhile, and a peer's edit is
//! kept) and stops only the replacement workers — the current generation
//! never stops serving, and the drain step is the single point of no return
//! (worker drain closes the ownership pipe irreversibly).
//!
//! Rollout state persists in the supervisor's runtime directory so a
//! crashed supervisor resumes or aborts explicitly instead of silently
//! relaunching the old generation mid-rollout. The shared approvals file
//! stays the only supervisor-to-gateway coupling: the coordinator edits it
//! through [`crate::approvals_file`] (read, change and atomic replace under
//! one lock), and the gateway adopts changed views through the DINV-07
//! eligibility rule.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use izwi_serving_protocol::{
    GatewayApprovalsStatus, GatewayWorkerApproval, GatewayWorkerApprovalIdentity, ModelGeneration,
    NodeId, TaskKind,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::approvals_file::{self, ApprovalsDocument, SharedApprovalsError};
use crate::config::{ValidatedNodeConfig, WorkerConfig};

/// Hard ceiling on the rollout plan file.
pub const MAX_ROLLOUT_PLAN_BYTES: usize = 64 * 1024;
/// Hard ceiling on the soak window.
pub const MAX_ROLLOUT_WINDOW_SECS: u64 = 3600;
/// Hard ceiling on the post-abort routing grace.
pub const MAX_ROLLOUT_ABORT_GRACE_SECS: u64 = 300;
/// Hard ceiling on the gateway approvals refresh interval a plan may declare
/// (the gateway's `IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS` upper bound).
pub const MAX_GATEWAY_APPROVALS_REFRESH_SECS: u64 = 3600;
/// Rollout state file name inside the supervisor's runtime directory.
pub const ROLLOUT_STATE_FILE: &str = "rollout-state.json";
/// Version 2 records the replacement approval lines instead of a backup of
/// the whole approvals file.
pub const ROLLOUT_STATE_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, thiserror::Error)]
pub enum RolloutError {
    #[error("rollout plan exceeds its {limit} byte size limit")]
    PlanTooLarge { limit: usize },
    #[error("rollout plan is not valid TOML: {0}")]
    InvalidToml(String),
    #[error("rollout plan {0}")]
    InvalidPlan(String),
    #[error("rollout state {0}")]
    InvalidState(String),
    #[error("rollout state IO failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("approvals file {0}")]
    InvalidApprovals(String),
    #[error(transparent)]
    Approvals(#[from] SharedApprovalsError),
    #[error("a rollout is already recorded in state {phase}; resume with --rollout-plan or abort with --rollout-abort")]
    RolloutAlreadyInProgress { phase: String },
}

/// The declarative rollout plan (DS6.1).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutPlan {
    schema_version: u16,
    /// Node config holding the new deployment generations. Its non-rolling
    /// workers and node identity must match the running config exactly.
    target_node_config: PathBuf,
    /// The shared approvals file the gateway reads. The coordinator writes
    /// the window and commit views here atomically.
    shared_approvals_path: PathBuf,
    /// The replacement worker launched first; the rollout aborts if it fails
    /// readiness.
    canary_worker_id: String,
    /// How often the gateways re-read the shared approvals file: their
    /// `IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS`, in seconds. The coordinator
    /// cannot ask the gateways when they adopted a view, so the window and
    /// the commit grace are sized from this interval.
    #[serde(default = "default_gateway_approvals_refresh_secs")]
    gateway_approvals_refresh_secs: u64,
    /// Soak window between opening the rollout window and writing the
    /// commit view. At least two gateway refresh intervals, so every gateway
    /// adopts the window view (and cuts over) before the commit view lands;
    /// unset defaults to exactly two.
    window_secs: Option<u64>,
    /// Grace between restoring the approvals on abort and stopping the
    /// replacement workers, so the gateway stops routing to them first.
    #[serde(default = "default_abort_grace_secs")]
    abort_grace_secs: u64,
    /// Grace between writing the commit view and draining the old
    /// generation, so every gateway adopts the commit view and stops routing
    /// to the old workers first. At least one gateway refresh interval;
    /// unset defaults to exactly one.
    commit_grace_secs: Option<u64>,
    /// How the generations change places. `auto` (the default) runs them
    /// side by side when both fit the node, and otherwise recreates.
    #[serde(default)]
    strategy: StrategyChoice,
    /// The gateways' admin endpoints. With them the rollout moves on only
    /// once every gateway reports a view adopted, instead of waiting out
    /// timers sized from `gateway_approvals_refresh_secs`.
    gateway_ack: Option<GatewayAckConfig>,
}

/// Where and how to ask the gateways which approvals view they adopted.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayAckConfig {
    /// Each gateway's base URL; the rollout polls
    /// `GET <url>/internal/admin/approvals` on every one.
    pub admin_urls: Vec<String>,
    /// The supervisor environment variable holding the gateways' admin key.
    pub admin_key_env: String,
    /// How long every gateway has to acknowledge a view. A window view not
    /// acknowledged in time aborts the rollout; so does a commit view,
    /// before the old generation drains.
    #[serde(default = "default_gateway_ack_timeout_secs")]
    pub timeout_secs: u64,
}

pub const MAX_GATEWAY_ACK_URLS: usize = 16;

fn default_gateway_ack_timeout_secs() -> u64 {
    120
}

/// The rollout strategy a plan asks for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyChoice {
    #[default]
    Auto,
    BlueGreen,
    Recreate,
}

/// How a rollout replaces the old generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutStrategy {
    /// Both generations run during a window; the gateway cuts over without
    /// a gap. Needs room for both on the node.
    BlueGreen,
    /// Drain the old generation, then start the new one: the only way on a
    /// node with one GPU, at the cost of downtime while the new generation
    /// loads (a Kubernetes rolling update with maxSurge 0).
    Recreate,
}

impl RolloutStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            RolloutStrategy::BlueGreen => "blue_green",
            RolloutStrategy::Recreate => "recreate",
        }
    }
}

impl GatewayAckConfig {
    fn validate(&self) -> Result<(), RolloutError> {
        if self.admin_urls.is_empty() || self.admin_urls.len() > MAX_GATEWAY_ACK_URLS {
            return Err(RolloutError::InvalidPlan(format!(
                "gateway_ack.admin_urls must list 1 to {MAX_GATEWAY_ACK_URLS} gateways"
            )));
        }
        for url in &self.admin_urls {
            let scheme_ok = url.starts_with("http://") || url.starts_with("https://");
            if !scheme_ok || url.len() > 2048 || url.chars().any(char::is_whitespace) {
                return Err(RolloutError::InvalidPlan(format!(
                    "gateway_ack.admin_urls entry {url:?} must be an http(s) URL"
                )));
            }
        }
        crate::config::validate_env_name(&self.admin_key_env).map_err(|_| {
            RolloutError::InvalidPlan(
                "gateway_ack.admin_key_env must name an environment variable".to_string(),
            )
        })?;
        if !(1..=MAX_ROLLOUT_WINDOW_SECS).contains(&self.timeout_secs) {
            return Err(RolloutError::InvalidPlan(format!(
                "gateway_ack.timeout_secs must be between 1 and {MAX_ROLLOUT_WINDOW_SECS}"
            )));
        }
        Ok(())
    }
}

/// Whether a gateway adopted the window view and cut every rolling
/// deployment over: it approves every replacement and routes new work to
/// the new generation (which needs it to have observed a replacement Ready).
pub fn window_adopted(
    status: &GatewayApprovalsStatus,
    spec: &RolloutSpec,
    replacements: &[GatewayWorkerApproval],
) -> bool {
    replacements
        .iter()
        .all(|approval| status.approvals.contains(&approval.render_line()))
        && spec.rolling.iter().all(|deployment| {
            status.pools.iter().any(|pool| {
                pool.deployment_id.as_str() == deployment.deployment_id
                    && pool.eligible_generation == Some(deployment.new_generation)
            })
        })
}

/// Whether a gateway adopted a view that no longer approves this node's old
/// generation, so draining it sheds nothing.
pub fn old_generation_unapproved(status: &GatewayApprovalsStatus, spec: &RolloutSpec) -> bool {
    status.approvals.iter().all(|line| {
        line.parse::<GatewayWorkerApproval>()
            .map_or(true, |approval| !spec.is_old_line(&approval))
    })
}

/// Picks the strategy for a validated rollout. Blue-green needs the
/// current workers and the replacements to fit the node together — host
/// memory, CPU threads, distinct worker ids and binds, and exclusive
/// devices — which a one-GPU node never satisfies; `auto` recreates then.
pub fn resolve_strategy(
    choice: StrategyChoice,
    current: &ValidatedNodeConfig,
    target: &ValidatedNodeConfig,
    spec: &RolloutSpec,
    inventory: &crate::HostInventory,
    binaries: &crate::BinaryCatalog,
) -> Result<RolloutStrategy, RolloutError> {
    if choice == StrategyChoice::Recreate {
        return Ok(RolloutStrategy::Recreate);
    }
    let mut union = current.config().clone();
    union.workers.extend(
        target
            .config()
            .workers
            .iter()
            .filter(|worker| spec.rolling_for(worker.worker_id.as_str()).is_some())
            .cloned(),
    );
    match (choice, union.validate(inventory, binaries)) {
        (_, Ok(_)) => Ok(RolloutStrategy::BlueGreen),
        (StrategyChoice::BlueGreen, Err(error)) => Err(RolloutError::InvalidPlan(format!(
            "blue-green runs both generations at once, but the current workers and the replacements do not fit this node together: {error}; use strategy = \"recreate\" or \"auto\""
        ))),
        (_, Err(_)) => Ok(RolloutStrategy::Recreate),
    }
}

fn default_gateway_approvals_refresh_secs() -> u64 {
    30
}

fn default_abort_grace_secs() -> u64 {
    30
}

/// One deployment whose generation the rollout advances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollingDeployment {
    pub deployment_id: String,
    pub task: TaskKind,
    pub public_model: String,
    pub old_generation: ModelGeneration,
    pub new_generation: ModelGeneration,
    pub old_worker_ids: Vec<String>,
    /// The old workers' endpoints, which match their standalone-form lines.
    pub old_endpoints: Vec<String>,
    pub new_worker_ids: Vec<String>,
}

/// The validated rollout: what rolls, in what order, and with what policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutSpec {
    pub node_id: NodeId,
    pub rolling: Vec<RollingDeployment>,
    pub canary_worker_id: String,
    pub window: Duration,
    /// After abort restores the previous approval view, the coordinator
    /// waits this long before stopping the replacement workers so the
    /// gateway's approvals refresh stops routing to them first.
    pub abort_grace: Duration,
    /// After the commit view is written, the coordinator waits this long
    /// before draining the old generation so the gateways' approvals
    /// refresh stops routing to it first.
    pub commit_grace: Duration,
}

impl RolloutSpec {
    /// The rolling deployment a worker belongs to, if any.
    pub fn rolling_for(&self, worker_id: &str) -> Option<&RollingDeployment> {
        self.rolling
            .iter()
            .find(|deployment| deployment.new_worker_ids.iter().any(|id| id == worker_id))
    }

    pub fn is_old_worker(&self, worker_id: &str) -> bool {
        self.rolling
            .iter()
            .any(|deployment| deployment.old_worker_ids.iter().any(|id| id == worker_id))
    }

    /// Whether an approval line is this node's line for an old worker of a
    /// rolling deployment: the lines the commit view removes.
    pub fn is_old_line(&self, approval: &GatewayWorkerApproval) -> bool {
        self.rolling.iter().any(|deployment| {
            approval.deployment_id.as_str() == deployment.deployment_id
                && approval.model_generation == deployment.old_generation
                && match &approval.identity {
                    GatewayWorkerApprovalIdentity::V1 { node_id, worker_id } => {
                        node_id == &self.node_id
                            && deployment
                                .old_worker_ids
                                .iter()
                                .any(|id| id == worker_id.as_str())
                    }
                    _ => deployment
                        .old_endpoints
                        .iter()
                        .any(|endpoint| endpoint == &approval.endpoint),
                }
        })
    }
}

impl RolloutPlan {
    /// Parses the bounded plan file.
    pub fn parse_bounded(bytes: &[u8]) -> Result<Self, RolloutError> {
        if bytes.len() > MAX_ROLLOUT_PLAN_BYTES {
            return Err(RolloutError::PlanTooLarge {
                limit: MAX_ROLLOUT_PLAN_BYTES,
            });
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|error| RolloutError::InvalidToml(error.to_string()))?;
        let plan: RolloutPlan =
            toml::from_str(text).map_err(|error| RolloutError::InvalidToml(error.to_string()))?;
        plan.validate_shape()?;
        Ok(plan)
    }

    /// The target node config path.
    pub fn target_node_config(&self) -> &Path {
        &self.target_node_config
    }

    /// The shared approvals file path the coordinator rewrites.
    pub fn strategy(&self) -> StrategyChoice {
        self.strategy
    }

    pub fn gateway_ack(&self) -> Option<&GatewayAckConfig> {
        self.gateway_ack.as_ref()
    }

    pub fn shared_approvals_path(&self) -> &Path {
        &self.shared_approvals_path
    }

    fn validate_shape(&self) -> Result<(), RolloutError> {
        if self.schema_version != 1 {
            return Err(RolloutError::InvalidPlan(format!(
                "unsupported schema_version {}; this supervisor implements 1",
                self.schema_version
            )));
        }
        for (label, path) in [
            ("target_node_config", &self.target_node_config),
            ("shared_approvals_path", &self.shared_approvals_path),
        ] {
            if path.as_os_str().is_empty() {
                return Err(RolloutError::InvalidPlan(format!(
                    "{label} must not be empty"
                )));
            }
            if path.to_string_lossy().len() > 4096 {
                return Err(RolloutError::InvalidPlan(format!(
                    "{label} exceeds its path length limit"
                )));
            }
            if !path.is_absolute() {
                return Err(RolloutError::InvalidPlan(format!(
                    "{label} must be an absolute path"
                )));
            }
        }
        if self.canary_worker_id.is_empty() || self.canary_worker_id.len() > 256 {
            return Err(RolloutError::InvalidPlan(
                "canary_worker_id must be 1..=256 bytes".to_string(),
            ));
        }
        let refresh = self.gateway_approvals_refresh_secs;
        if !(1..=MAX_GATEWAY_APPROVALS_REFRESH_SECS).contains(&refresh) {
            return Err(RolloutError::InvalidPlan(format!(
                "gateway_approvals_refresh_secs must be between 1 and {MAX_GATEWAY_APPROVALS_REFRESH_SECS}"
            )));
        }
        if self.window_secs() > MAX_ROLLOUT_WINDOW_SECS {
            return Err(RolloutError::InvalidPlan(format!(
                "window_secs must be at most {MAX_ROLLOUT_WINDOW_SECS}"
            )));
        }
        if let Some(ack) = &self.gateway_ack {
            ack.validate()?;
        }
        // Acknowledged adoption replaces the refresh-sized timers.
        let timed = self.gateway_ack.is_none();
        if timed && self.window_secs() < refresh.saturating_mul(2) {
            return Err(RolloutError::InvalidPlan(format!(
                "window_secs must be at least twice gateway_approvals_refresh_secs ({}s) so every gateway adopts the window view before the commit view",
                refresh.saturating_mul(2)
            )));
        }
        if self.abort_grace_secs > MAX_ROLLOUT_ABORT_GRACE_SECS {
            return Err(RolloutError::InvalidPlan(format!(
                "abort_grace_secs must be at most {MAX_ROLLOUT_ABORT_GRACE_SECS}"
            )));
        }
        if self.commit_grace_secs() > MAX_ROLLOUT_WINDOW_SECS {
            return Err(RolloutError::InvalidPlan(format!(
                "commit_grace_secs must be at most {MAX_ROLLOUT_WINDOW_SECS}"
            )));
        }
        if timed && self.commit_grace_secs() < refresh {
            return Err(RolloutError::InvalidPlan(format!(
                "commit_grace_secs must be at least gateway_approvals_refresh_secs ({refresh}s) so every gateway stops routing to the old generation before it drains"
            )));
        }
        Ok(())
    }

    /// The soak window. Without acknowledgement it must cover two gateway
    /// refreshes; with it the window is a pure soak after the gateways
    /// confirm the cutover, none by default.
    fn window_secs(&self) -> u64 {
        self.window_secs.unwrap_or_else(|| match self.gateway_ack {
            Some(_) => 0,
            None => self.gateway_approvals_refresh_secs.saturating_mul(2),
        })
    }

    /// The wait before draining the old generation when no gateway can be
    /// asked; with acknowledgement the drain waits for the gateways.
    fn commit_grace_secs(&self) -> u64 {
        self.commit_grace_secs.unwrap_or(match self.gateway_ack {
            Some(_) => 0,
            None => self.gateway_approvals_refresh_secs,
        })
    }

    /// Digest binding the plan to its target config bytes; persisted rollout
    /// state is only resumable against the same digest.
    pub fn digest(&self, target_config_bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.target_node_config.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(self.shared_approvals_path.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(self.canary_worker_id.as_bytes());
        hasher.update([0]);
        hasher.update(self.window_secs().to_le_bytes());
        hasher.update([0]);
        hasher.update(self.abort_grace_secs.to_le_bytes());
        hasher.update([0]);
        hasher.update(self.gateway_approvals_refresh_secs.to_le_bytes());
        hasher.update([0]);
        hasher.update(self.commit_grace_secs().to_le_bytes());
        hasher.update([0]);
        hasher.update([self.strategy as u8]);
        hasher.update([0]);
        hasher.update(target_config_bytes);
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    /// Validates the plan against the running and target node configs and
    /// derives the rollout spec. Validation is fail-closed: a rollout only
    /// ever moves generations forward, only rewrites rolling deployments,
    /// and requires the target node to be this node.
    pub fn validate(
        &self,
        current: &ValidatedNodeConfig,
        target: &ValidatedNodeConfig,
    ) -> Result<RolloutSpec, RolloutError> {
        if current.config().node_id != target.config().node_id {
            return Err(RolloutError::InvalidPlan(format!(
                "target node config is for node {} but this node is {}",
                target.config().node_id,
                current.config().node_id
            )));
        }

        let mut current_deployments: BTreeMap<String, &crate::config::DeploymentConfig> =
            BTreeMap::new();
        for worker in &current.config().workers {
            current_deployments
                .entry(worker.deployment.deployment_id.to_string())
                .or_insert(&worker.deployment);
        }
        let mut target_deployments: BTreeMap<String, &crate::config::DeploymentConfig> =
            BTreeMap::new();
        for worker in &target.config().workers {
            target_deployments
                .entry(worker.deployment.deployment_id.to_string())
                .or_insert(&worker.deployment);
        }

        for deployment_id in current_deployments.keys() {
            if !target_deployments.contains_key(deployment_id) {
                return Err(RolloutError::InvalidPlan(format!(
                    "target config removes deployment {deployment_id}; rollouts only advance generations"
                )));
            }
        }
        for deployment_id in target_deployments.keys() {
            if !current_deployments.contains_key(deployment_id) {
                return Err(RolloutError::InvalidPlan(format!(
                    "target config adds deployment {deployment_id}; rollouts only advance generations"
                )));
            }
        }

        let mut rolling: Vec<RollingDeployment> = Vec::new();
        for (deployment_id, current_deployment) in &current_deployments {
            let target_deployment = target_deployments[deployment_id];
            let old_generation = current_deployment.model_generation;
            let new_generation = target_deployment.model_generation;
            if new_generation < old_generation {
                return Err(RolloutError::InvalidPlan(format!(
                    "deployment {deployment_id} regresses from generation {} to {}; rollback is the abort path, not a rollout",
                    old_generation.get(),
                    new_generation.get()
                )));
            }
            if new_generation == old_generation {
                // Non-rolling deployment: every target worker must be an
                // unchanged copy of a current worker, byte for byte.
                for worker in
                    target.config().workers.iter().filter(|worker| {
                        worker.deployment.deployment_id.to_string() == *deployment_id
                    })
                {
                    let Some(current_worker) = current.worker(&worker.worker_id) else {
                        return Err(RolloutError::InvalidPlan(format!(
                            "target config changes worker set of non-rolling deployment {deployment_id}: unknown worker {}",
                            worker.worker_id
                        )));
                    };
                    if current_worker != worker {
                        return Err(RolloutError::InvalidPlan(format!(
                            "worker {} of non-rolling deployment {deployment_id} differs between current and target configs; rollouts only advance generations",
                            worker.worker_id
                        )));
                    }
                }
                continue;
            }
            let (old_worker_ids, old_endpoints) = current
                .config()
                .workers
                .iter()
                .filter(|worker| worker.deployment.deployment_id.to_string() == *deployment_id)
                .map(|worker| {
                    (
                        worker.worker_id.to_string(),
                        format!("http://{}", worker.bind),
                    )
                })
                .unzip();
            let new_worker_ids: Vec<String> = target
                .config()
                .workers
                .iter()
                .filter(|worker| worker.deployment.deployment_id.to_string() == *deployment_id)
                .map(|worker| worker.worker_id.to_string())
                .collect();
            if new_worker_ids.is_empty() {
                return Err(RolloutError::InvalidPlan(format!(
                    "deployment {deployment_id} advances to generation {} but the target config has no workers for it",
                    new_generation.get()
                )));
            }
            for worker in target
                .config()
                .workers
                .iter()
                .filter(|worker| worker.deployment.deployment_id.to_string() == *deployment_id)
            {
                if worker.deployment.model_generation != new_generation {
                    return Err(RolloutError::InvalidPlan(format!(
                        "worker {} mixes generation {} into rolling deployment {deployment_id} (target generation {})",
                        worker.worker_id,
                        worker.deployment.model_generation.get(),
                        new_generation.get()
                    )));
                }
            }
            rolling.push(RollingDeployment {
                deployment_id: deployment_id.clone(),
                task: current_deployment.task,
                public_model: current_deployment.public_model.to_string(),
                old_generation,
                new_generation,
                old_worker_ids,
                old_endpoints,
                new_worker_ids,
            });
        }
        if rolling.is_empty() {
            return Err(RolloutError::InvalidPlan(
                "the target config does not advance any deployment generation".to_string(),
            ));
        }

        let canary_is_replacement = rolling
            .iter()
            .any(|deployment| deployment.new_worker_ids.contains(&self.canary_worker_id));
        if !canary_is_replacement {
            return Err(RolloutError::InvalidPlan(format!(
                "canary worker {} is not a replacement worker of a rolling deployment",
                self.canary_worker_id
            )));
        }

        Ok(RolloutSpec {
            node_id: current.config().node_id.clone(),
            rolling,
            canary_worker_id: self.canary_worker_id.clone(),
            window: Duration::from_secs(self.window_secs()),
            abort_grace: Duration::from_secs(self.abort_grace_secs),
            commit_grace: Duration::from_secs(self.commit_grace_secs()),
        })
    }
}

/// Persisted rollout state (DS6.2 resume-after-crash).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutState {
    pub schema_version: u16,
    pub plan_digest: String,
    pub phase: RolloutPhase,
    pub strategy: RolloutStrategy,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    #[serde(default)]
    pub window_deadline_ms: Option<u64>,
    #[serde(default)]
    pub replacement_workers: Vec<RolloutWorkerRecord>,
    pub shared_approvals_path: String,
    pub target_node_config: String,
    pub target_config_digest: String,
    /// The replacement lines this rollout adds to the shared approvals
    /// file; an abort removes exactly these.
    pub approval_lines: Vec<String>,
    /// The old generation's lines the commit view removed; an abort after
    /// it puts them back.
    #[serde(default)]
    pub removed_lines: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutPhase {
    LaunchingReplacement,
    WindowOpen,
    DrainingOld,
    /// Recreate: the commit view is written and the old generation drains.
    RecreateDrainingOld,
    /// Recreate: the old generation is gone and the replacements launch.
    RecreateLaunching,
    Committed,
    Aborted,
}

impl RolloutPhase {
    /// Terminal phases are safe to auto-clear on a fresh supervisor start:
    /// nothing remains to resume or roll back.
    pub fn is_terminal(self) -> bool {
        matches!(self, RolloutPhase::Committed | RolloutPhase::Aborted)
    }

    /// Phases in which the old generation has been retired: a resumed
    /// supervisor never relaunches it.
    pub fn old_generation_retired(self) -> bool {
        matches!(
            self,
            RolloutPhase::DrainingOld
                | RolloutPhase::RecreateDrainingOld
                | RolloutPhase::RecreateLaunching
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RolloutPhase::LaunchingReplacement => "launching_replacement",
            RolloutPhase::WindowOpen => "window_open",
            RolloutPhase::DrainingOld => "draining_old",
            RolloutPhase::RecreateDrainingOld => "recreate_draining_old",
            RolloutPhase::RecreateLaunching => "recreate_launching",
            RolloutPhase::Committed => "committed",
            RolloutPhase::Aborted => "aborted",
        }
    }
}

/// One launched replacement worker, for post-crash accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutWorkerRecord {
    pub worker_id: String,
    pub pid: u32,
}

pub fn state_path(runtime_directory: &Path) -> PathBuf {
    runtime_directory.join(ROLLOUT_STATE_FILE)
}

/// Loads persisted rollout state, if any.
pub fn load_state(runtime_directory: &Path) -> Result<Option<RolloutState>, RolloutError> {
    let path = state_path(runtime_directory);
    let Ok(bytes) = std::fs::read(&path) else {
        return Ok(None);
    };
    if bytes.len() > 256 * 1024 {
        return Err(RolloutError::InvalidState(
            "rollout state file exceeds its size limit".to_string(),
        ));
    }
    let state: RolloutState = serde_json::from_slice(&bytes)
        .map_err(|error| RolloutError::InvalidState(error.to_string()))?;
    if state.schema_version != ROLLOUT_STATE_SCHEMA_VERSION {
        return Err(RolloutError::InvalidState(format!(
            "unsupported rollout state schema_version {}; finish or abort it with the supervisor that started it",
            state.schema_version
        )));
    }
    Ok(Some(state))
}

/// Persists rollout state atomically and durably (temp file, fsync,
/// rename, directory fsync; owner-only). The node lease serializes writers.
pub fn persist_state(runtime_directory: &Path, state: &RolloutState) -> Result<(), RolloutError> {
    let bytes = serde_json::to_vec_pretty(state)
        .map_err(|error| RolloutError::InvalidState(error.to_string()))?;
    approvals_file::replace_file(&state_path(runtime_directory), &bytes)?;
    Ok(())
}

/// Removes persisted rollout state.
pub fn clear_state(runtime_directory: &Path) -> Result<(), RolloutError> {
    match std::fs::remove_file(state_path(runtime_directory)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Verifies the rollout precondition: the approvals file currently approves
/// the old generation of every rolling deployment. A rollout that rewrites
/// a file the gateway is not actually serving from would silently strand
/// admission.
pub fn verify_approvals_precondition(
    document: &ApprovalsDocument,
    spec: &RolloutSpec,
) -> Result<(), RolloutError> {
    for deployment in &spec.rolling {
        let approved = document.approvals().any(|approval| {
            approval.deployment_id.as_str() == deployment.deployment_id
                && approval.model_generation == deployment.old_generation
        });
        if !approved {
            return Err(RolloutError::InvalidApprovals(format!(
                "the shared approvals file does not approve deployment {} at the current generation {}; refusing to roll out",
                deployment.deployment_id,
                deployment.old_generation.get()
            )));
        }
    }
    Ok(())
}

/// The pinned approval of every replacement worker: the lines a rollout
/// adds to the shared view, and the only lines its abort removes.
pub fn replacement_approvals(
    spec: &RolloutSpec,
    target: &ValidatedNodeConfig,
) -> Result<Vec<GatewayWorkerApproval>, RolloutError> {
    let node_id = &target.config().node_id;
    spec.rolling
        .iter()
        .flat_map(|deployment| &deployment.new_worker_ids)
        .map(|worker_id| {
            let worker: &WorkerConfig = target
                .config()
                .workers
                .iter()
                .find(|worker| worker.worker_id.as_str() == worker_id)
                .ok_or_else(|| {
                    RolloutError::InvalidPlan(format!(
                        "replacement worker {worker_id} is missing from the target config"
                    ))
                })?;
            Ok(approvals_file::pinned_approval(node_id, worker))
        })
        .collect()
}

/// The window view: every existing line stays (the old generation remains
/// approved and admission-eligible until the gateway observes the successor
/// Ready) and the replacement lines are appended. Applying it again changes
/// nothing, so a resumed rollout rewrites it safely.
pub fn apply_window_view(document: &mut ApprovalsDocument, replacements: &[GatewayWorkerApproval]) {
    for approval in replacements {
        document.ensure(approval);
    }
}

/// The commit view: this node's old-generation lines of the rolling
/// deployments are removed and the replacement lines are kept. Lines of
/// other nodes and other deployments are untouched. Returns the removed
/// lines.
pub fn apply_commit_view(
    document: &mut ApprovalsDocument,
    spec: &RolloutSpec,
    replacements: &[GatewayWorkerApproval],
) -> Vec<String> {
    let removed = document.retain(|approval| !spec.is_old_line(approval));
    for approval in replacements {
        document.ensure(approval);
    }
    removed
}

/// Undoes a rollout's view edits: removes exactly the replacement lines it
/// added and puts back the lines it removed. Every other line, including a
/// peer's change made since, stays as it is; with no peer change the file
/// is byte-identical to the pre-rollout file.
pub fn restore_view(
    document: &mut ApprovalsDocument,
    added: &[GatewayWorkerApproval],
    removed: &[String],
) {
    document.retain(|approval| !added.contains(approval));
    for line in removed {
        document.append_line(line);
    }
}

/// Parses the replacement lines a persisted state recorded.
pub fn parse_approval_lines(lines: &[String]) -> Result<Vec<GatewayWorkerApproval>, RolloutError> {
    lines
        .iter()
        .map(|line| {
            line.parse::<GatewayWorkerApproval>().map_err(|error| {
                RolloutError::InvalidState(format!("recorded approval line {line:?}: {error}"))
            })
        })
        .collect()
}

pub fn digest_bytes(bytes: &[u8]) -> String {
    approvals_file::digest(bytes)
}

#[cfg(test)]
mod tests {
    use izwi_serving_protocol::NodeId;

    use super::*;

    fn plan_toml(target: &str, approvals: &str, canary: &str, window: u64) -> String {
        format!(
            "schema_version = 1\ntarget_node_config = \"{target}\"\nshared_approvals_path = \"{approvals}\"\ncanary_worker_id = \"{canary}\"\ngateway_approvals_refresh_secs = 5\nwindow_secs = {window}\n"
        )
    }

    fn base_worker(
        worker_id: &str,
        bind: &str,
        deployment_id: &str,
        public_model: &str,
        task: &str,
        generation: u64,
    ) -> String {
        let token = if worker_id.ends_with('a') || worker_id.ends_with('c') {
            "TOKEN_A"
        } else {
            "TOKEN_B"
        };
        let formats = if task == "chat" {
            "[\"chat_messages\"]"
        } else {
            "[\"encoded_audio\"]"
        };
        // Both fixture tasks emit text only.
        let output = "[\"text\"]";
        format!(
            "[[workers]]\nworker_id = \"{worker_id}\"\nbind = \"{bind}\"\nbinary = \"cpu\"\ncredential_id = \"cred-a\"\nbearer_token_env = \"{token}\"\n\n[workers.assignment]\nbackend = \"cpu\"\nthread_budget = 1\naffinity = [0]\nhost_memory_limit_bytes = 1024\n\n[workers.deployment]\ndeployment_id = \"{deployment_id}\"\npublic_model = \"{public_model}\"\nartifact_revision = \"rev-1\"\nmodel_generation = {generation}\ntask = \"{task}\"\nbackend = \"cpu\"\nprecision = \"f32\"\nexecution_representation = \"gguf\"\nmodels_directory = \"/tmp/izwi-models/chat\"\n\n[workers.deployment.capability]\nstreaming = true\nrealtime = false\ncancellation = \"cooperative\"\naccepted_input_formats = {formats}\noutput_formats = {output}\nmax_input_bytes = 1048576\nmax_context_tokens = 32\nmax_output_tokens = 32\n"
        )
    }

    /// Current shape: worker-a serves chat at the old generation, worker-c
    /// serves ASR (never rolled in these fixtures).
    fn current_toml(chat_generation: u64) -> String {
        format!(
            "schema_version = 2\nnode_id = \"node-a\"\nworking_directory = \"/tmp/izwi-work\"\nruntime_directory = \"/tmp/izwi-run\"\nhost_memory_budget_bytes = 8589934592\n\n{}\n{}",
            base_worker("worker-a", "127.0.0.1:9470", "chat-prod", "chat-model", "chat", chat_generation),
            base_worker("worker-c", "127.0.0.1:9471", "asr-prod", "asr-model", "speech_to_text", 4)
        )
    }

    /// Target shape: worker-b replaces worker-a for chat at the new
    /// generation; ASR is byte-identical.
    fn target_toml(chat_generation: u64) -> String {
        format!(
            "schema_version = 2\nnode_id = \"node-a\"\nworking_directory = \"/tmp/izwi-work\"\nruntime_directory = \"/tmp/izwi-run\"\nhost_memory_budget_bytes = 8589934592\n\n{}\n{}",
            base_worker("worker-b", "127.0.0.1:9480", "chat-prod", "chat-model", "chat", chat_generation),
            base_worker("worker-c", "127.0.0.1:9471", "asr-prod", "asr-model", "speech_to_text", 4)
        )
    }

    fn inventory() -> crate::config::HostInventory {
        crate::config::HostInventory {
            effective_cpu_ids: vec![0, 1, 2, 3],
            allocatable_host_memory_bytes: u64::MAX,
            metal_devices: vec![],
            cuda_devices: vec![],
        }
    }

    /// Validation needs a regular executable, and nothing here runs it: the
    /// test binary is one everywhere. `/usr/bin/true` is a symlink on
    /// Ubuntu 26.04, which validation rejects.
    fn binaries() -> crate::config::BinaryCatalog {
        crate::config::BinaryCatalog::new(vec![(
            crate::config::WorkerBinaryFlavor::Cpu,
            crate::config::BinaryRecord {
                path: std::env::current_exe().unwrap(),
                supported_backends: vec![izwi_serving_protocol::BackendKind::Cpu],
            },
        )])
    }

    fn validated(toml_text: &str) -> ValidatedNodeConfig {
        std::fs::create_dir_all("/tmp/izwi-work").unwrap();
        std::fs::create_dir_all("/tmp/izwi-run").unwrap();
        std::fs::create_dir_all("/tmp/izwi-models/chat").unwrap();
        let config = crate::config::NodeConfig::parse_bounded(toml_text.as_bytes()).unwrap();
        config.validate(&inventory(), &binaries()).unwrap()
    }

    fn spec_for(current: &ValidatedNodeConfig, target: &ValidatedNodeConfig) -> RolloutSpec {
        let plan: RolloutPlan = toml::from_str(&plan_toml(
            "/tmp/target.toml",
            "/tmp/approvals",
            "worker-b",
            10,
        ))
        .unwrap();
        plan.validate(current, target).unwrap()
    }

    #[test]
    fn plan_parse_is_bounded_and_strict() {
        let plan = RolloutPlan::parse_bounded(
            plan_toml("/tmp/t.toml", "/tmp/a.txt", "worker-b", 30).as_bytes(),
        )
        .unwrap();
        assert_eq!(plan.window_secs(), 30);
        assert_eq!(plan.commit_grace_secs(), 5);
        assert_eq!(plan.canary_worker_id, "worker-b");

        let oversized = vec![b' '; MAX_ROLLOUT_PLAN_BYTES + 1];
        assert!(matches!(
            RolloutPlan::parse_bounded(&oversized).unwrap_err(),
            RolloutError::PlanTooLarge { .. }
        ));

        let unknown = "schema_version = 1\nsurprise = true\n";
        assert!(matches!(
            RolloutPlan::parse_bounded(unknown.as_bytes()).unwrap_err(),
            RolloutError::InvalidToml(_)
        ));

        let relative = "schema_version = 1\ntarget_node_config = \"relative.toml\"\nshared_approvals_path = \"/tmp/a\"\ncanary_worker_id = \"w\"\n";
        assert!(matches!(
            RolloutPlan::parse_bounded(relative.as_bytes()).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));

        let unbounded_window = plan_toml(
            "/tmp/t.toml",
            "/tmp/a.txt",
            "w",
            MAX_ROLLOUT_WINDOW_SECS + 1,
        );
        assert!(matches!(
            RolloutPlan::parse_bounded(unbounded_window.as_bytes()).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));
    }

    #[test]
    fn window_and_commit_grace_cover_the_gateway_approvals_refresh() {
        let header = "schema_version = 1\ntarget_node_config = \"/tmp/t.toml\"\nshared_approvals_path = \"/tmp/a.txt\"\ncanary_worker_id = \"w\"\n";
        // Unset values derive from the gateway's default 30s refresh: two
        // refresh intervals of window and one of commit grace.
        let defaults = RolloutPlan::parse_bounded(header.as_bytes()).unwrap();
        assert_eq!(defaults.window_secs(), 60);
        assert_eq!(defaults.commit_grace_secs(), 30);

        let declared = format!("{header}gateway_approvals_refresh_secs = 2\n");
        let plan = RolloutPlan::parse_bounded(declared.as_bytes()).unwrap();
        assert_eq!(plan.window_secs(), 4);
        assert_eq!(plan.commit_grace_secs(), 2);

        // A window shorter than two refreshes can commit before a gateway
        // ever adopted the window view; a commit grace shorter than one
        // refresh drains the old generation while it is still routed.
        for invalid in [
            format!("{header}window_secs = 30\n"),
            format!("{header}gateway_approvals_refresh_secs = 2\nwindow_secs = 3\n"),
            format!("{header}gateway_approvals_refresh_secs = 2\ncommit_grace_secs = 1\n"),
            format!("{header}gateway_approvals_refresh_secs = 0\n"),
            format!("{header}gateway_approvals_refresh_secs = 3601\n"),
        ] {
            assert!(
                matches!(
                    RolloutPlan::parse_bounded(invalid.as_bytes()).unwrap_err(),
                    RolloutError::InvalidPlan(_)
                ),
                "{invalid} must be rejected"
            );
        }
    }

    #[test]
    fn gateway_ack_replaces_the_refresh_timers_and_is_validated() {
        let header = "schema_version = 1\ntarget_node_config = \"/tmp/t.toml\"\nshared_approvals_path = \"/tmp/a.txt\"\ncanary_worker_id = \"w\"\n";
        let ack = "[gateway_ack]\nadmin_urls = [\"http://127.0.0.1:8080\"]\nadmin_key_env = \"IZWI_GATEWAY_ADMIN_KEY\"\n";
        let plan = RolloutPlan::parse_bounded(format!("{header}{ack}").as_bytes()).unwrap();
        assert_eq!(plan.window_secs(), 0, "the soak starts after the ack");
        assert_eq!(plan.commit_grace_secs(), 0, "the drain waits for the ack");
        assert_eq!(plan.gateway_ack().unwrap().timeout_secs, 120);
        // Short timers no longer need to cover the refresh interval.
        RolloutPlan::parse_bounded(format!("{header}window_secs = 1\n{ack}").as_bytes()).unwrap();

        for invalid in [
            "[gateway_ack]\nadmin_urls = []\nadmin_key_env = \"KEY\"\n",
            "[gateway_ack]\nadmin_urls = [\"ftp://gateway\"]\nadmin_key_env = \"KEY\"\n",
            "[gateway_ack]\nadmin_urls = [\"http://gateway\"]\nadmin_key_env = \"lower\"\n",
            "[gateway_ack]\nadmin_urls = [\"http://gateway\"]\nadmin_key_env = \"KEY\"\ntimeout_secs = 0\n",
        ] {
            assert!(
                matches!(
                    RolloutPlan::parse_bounded(format!("{header}{invalid}").as_bytes()).unwrap_err(),
                    RolloutError::InvalidPlan(_)
                ),
                "{invalid} must be rejected"
            );
        }
    }

    #[test]
    fn adoption_predicates_read_the_gateway_report() {
        let spec = sample_spec();
        let pool = |eligible: u64| izwi_serving_protocol::GatewayPoolStatus {
            task: TaskKind::Chat,
            public_model: izwi_serving_protocol::ModelAlias::new("chat-model").unwrap(),
            deployment_id: izwi_serving_protocol::DeploymentId::new("chat-prod").unwrap(),
            eligible_generation: Some(ModelGeneration::new(eligible).unwrap()),
            generations: vec![
                ModelGeneration::new(1).unwrap(),
                ModelGeneration::new(2).unwrap(),
            ],
        };
        let old_line = "v1|http://127.0.0.1:9470|node-a|worker-a|chat|chat-model|chat-prod|1";
        let status = |approvals: Vec<String>, eligible: u64| GatewayApprovalsStatus {
            file_digest: None,
            adoptions: 1,
            approvals,
            pools: vec![pool(eligible)],
        };
        let new_lines: Vec<String> = replacements()
            .iter()
            .map(|line| line.render_line())
            .collect();
        let window = [vec![old_line.to_string()], new_lines.clone()].concat();

        assert!(!window_adopted(
            &status(vec![old_line.to_string()], 1),
            &spec,
            &replacements()
        ));
        assert!(
            !window_adopted(&status(window.clone(), 1), &spec, &replacements()),
            "adopted but not yet cut over"
        );
        assert!(window_adopted(
            &status(window.clone(), 2),
            &spec,
            &replacements()
        ));

        assert!(!old_generation_unapproved(&status(window, 2), &spec));
        assert!(old_generation_unapproved(&status(new_lines, 2), &spec));
    }

    #[test]
    fn validate_derives_rolling_deployments_and_rejects_regressions() {
        let current = validated(&current_toml(1));
        let target = validated(&target_toml(2));
        let spec = spec_for(&current, &target);
        assert_eq!(spec.rolling.len(), 1);
        let deployment = &spec.rolling[0];
        assert_eq!(deployment.deployment_id, "chat-prod");
        assert_eq!(deployment.old_generation, ModelGeneration::new(1).unwrap());
        assert_eq!(deployment.new_generation, ModelGeneration::new(2).unwrap());
        assert_eq!(deployment.old_worker_ids, vec!["worker-a"]);
        assert_eq!(deployment.new_worker_ids, vec!["worker-b"]);
        assert_eq!(spec.canary_worker_id, "worker-b");
        assert!(spec.is_old_worker("worker-a"));
        assert!(!spec.is_old_worker("worker-b"));
        assert_eq!(
            spec.rolling_for("worker-b").unwrap().deployment_id,
            "chat-prod"
        );
        assert!(spec.rolling_for("worker-a").is_none());

        // Generation regression is the abort path, never a rollout.
        let regressed = validated(&current_toml(3));
        let rolled_back = validated(&target_toml(2));
        let plan: RolloutPlan =
            toml::from_str(&plan_toml("/tmp/t.toml", "/tmp/a", "worker-b", 10)).unwrap();
        assert!(matches!(
            plan.validate(&regressed, &rolled_back).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));

        // A different node id is a config mistake, not a rollout.
        let other = validated(&current_toml(1).replacen("node-a", "node-b", 1));
        assert!(matches!(
            plan.validate(&current, &other).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));

        // The canary must be a replacement worker of a rolling deployment.
        let wrong_canary: RolloutPlan =
            toml::from_str(&plan_toml("/tmp/t.toml", "/tmp/a", "worker-c", 10)).unwrap();
        assert!(matches!(
            wrong_canary.validate(&current, &target).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));
    }

    /// A node that cannot hold both generations (here the replacement
    /// takes the old worker's bind, as a one-GPU node would share its
    /// device) recreates under `auto` and refuses an explicit blue-green.
    #[test]
    fn strategy_follows_whether_both_generations_fit() {
        let current = validated(&current_toml(1));
        let target = validated(&target_toml(2));
        let spec = spec_for(&current, &target);
        let resolve = |choice, target: &ValidatedNodeConfig| {
            resolve_strategy(choice, &current, target, &spec, &inventory(), &binaries())
        };
        assert_eq!(
            resolve(StrategyChoice::Auto, &target).unwrap(),
            RolloutStrategy::BlueGreen
        );
        assert_eq!(
            resolve(StrategyChoice::Recreate, &target).unwrap(),
            RolloutStrategy::Recreate
        );

        let shared = validated(&target_toml(2).replace("127.0.0.1:9480", "127.0.0.1:9470"));
        assert_eq!(
            resolve(StrategyChoice::Auto, &shared).unwrap(),
            RolloutStrategy::Recreate
        );
        assert!(matches!(
            resolve(StrategyChoice::BlueGreen, &shared).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));
    }

    #[test]
    fn non_rolling_workers_must_match_between_configs() {
        let current = validated(&current_toml(1));
        // Same generations, but the non-rolling worker changed its bind.
        let changed = current_toml(1).replace("127.0.0.1:9471", "127.0.0.1:9500");
        let target = validated(&changed);
        let plan: RolloutPlan =
            toml::from_str(&plan_toml("/tmp/t.toml", "/tmp/a", "worker-b", 10)).unwrap();
        assert!(matches!(
            plan.validate(&current, &target).unwrap_err(),
            RolloutError::InvalidPlan(_)
        ));
    }

    const ORIGINAL: &str = "# fleet\nv1|http://127.0.0.1:9470|node-a|worker-a|chat|chat-model|chat-prod|1\nv1|http://127.0.0.1:9600|node-a|asr-worker|speech_to_text|asr-model|asr-prod|4\n";

    fn document(text: &str) -> ApprovalsDocument {
        ApprovalsDocument::parse(text).unwrap()
    }

    fn sample_spec() -> RolloutSpec {
        RolloutSpec {
            node_id: NodeId::new("node-a").unwrap(),
            rolling: vec![RollingDeployment {
                deployment_id: "chat-prod".to_string(),
                task: TaskKind::Chat,
                public_model: "chat-model".to_string(),
                old_generation: ModelGeneration::new(1).unwrap(),
                new_generation: ModelGeneration::new(2).unwrap(),
                old_worker_ids: vec!["worker-a".to_string()],
                old_endpoints: vec!["http://127.0.0.1:9470".to_string()],
                new_worker_ids: vec!["worker-b".to_string()],
            }],
            canary_worker_id: "worker-b".to_string(),
            window: Duration::from_secs(10),
            abort_grace: Duration::from_secs(5),
            commit_grace: Duration::from_secs(5),
        }
    }

    fn replacements() -> Vec<GatewayWorkerApproval> {
        replacement_approvals(&sample_spec(), &validated(&target_toml(2))).unwrap()
    }

    #[test]
    fn window_view_keeps_every_line_and_appends_pinned_replacements() {
        let mut view = document(ORIGINAL);
        apply_window_view(&mut view, &replacements());
        let window = view.render();
        assert!(
            window.starts_with(ORIGINAL),
            "existing lines pass through: {window}"
        );
        let parsed = izwi_serving_protocol::parse_approvals_text(&window).unwrap();
        assert_eq!(parsed.len(), 3);
        let replacement = parsed
            .iter()
            .find(|approval| approval.endpoint == "http://127.0.0.1:9480")
            .unwrap();
        assert_eq!(
            replacement.pinned_identity(),
            Some((
                &NodeId::new("node-a").unwrap(),
                &izwi_serving_protocol::WorkerId::new("worker-b").unwrap()
            ))
        );
        assert_eq!(
            replacement.model_generation,
            ModelGeneration::new(2).unwrap()
        );

        apply_window_view(&mut view, &replacements());
        assert_eq!(view.render(), window, "the window view is idempotent");
    }

    #[test]
    fn commit_view_drops_only_this_nodes_old_lines() {
        let foreign = "v1|http://10.0.0.9:9470|node-z|worker-z|chat|chat-model|chat-prod|1\n";
        let standalone = "http://127.0.0.1:9470|chat|chat-model|chat-prod|1\n";
        let mut view = document(&format!("{ORIGINAL}{foreign}{standalone}"));
        apply_window_view(&mut view, &replacements());
        let removed = apply_commit_view(&mut view, &sample_spec(), &replacements());
        assert_eq!(
            removed.len(),
            2,
            "the v1 and standalone old lines: {removed:?}"
        );
        let commit = view.render();
        assert!(!commit.contains("worker-a|chat|chat-model|chat-prod|1"));
        assert!(!commit.contains(standalone));
        assert!(commit.contains(foreign), "another node's line stays");
        assert!(commit.contains("asr-prod|4"));
        assert!(commit.contains("worker-b|chat|chat-model|chat-prod|2"));

        assert!(
            apply_commit_view(&mut view, &sample_spec(), &replacements()).is_empty(),
            "the commit view is idempotent"
        );
        assert_eq!(view.render(), commit);
    }

    #[test]
    fn restore_removes_only_the_rollouts_lines() {
        let mut view = document(ORIGINAL);
        apply_window_view(&mut view, &replacements());
        let mut untouched = view.clone();
        restore_view(&mut untouched, &replacements(), &[]);
        assert_eq!(
            untouched.render(),
            ORIGINAL,
            "with no peer change the abort restores the file byte-identically"
        );

        // A peer adds a line while the window is open; the abort keeps it.
        let peer = "v1|http://127.0.0.1:9700|node-b|worker-p|chat|chat-model|chat-prod|1";
        view.ensure(&peer.parse().unwrap());
        restore_view(&mut view, &replacements(), &[]);
        assert_eq!(view.render(), format!("{ORIGINAL}{peer}\n"));
    }

    #[test]
    fn approvals_precondition_requires_the_current_generation() {
        let spec = sample_spec();
        verify_approvals_precondition(&document(ORIGINAL), &spec).unwrap();

        let stale =
            document("v1|http://127.0.0.1:9470|node-a|worker-a|chat|chat-model|chat-prod|9\n");
        assert!(matches!(
            verify_approvals_precondition(&stale, &spec).unwrap_err(),
            RolloutError::InvalidApprovals(_)
        ));
    }

    #[test]
    fn recorded_approval_lines_parse_back() {
        let lines: Vec<String> = replacements()
            .iter()
            .map(|line| line.render_line())
            .collect();
        assert_eq!(parse_approval_lines(&lines).unwrap(), replacements());
        assert!(matches!(
            parse_approval_lines(&["http://bad|line".to_string()]).unwrap_err(),
            RolloutError::InvalidState(_)
        ));
    }

    #[test]
    fn state_round_trips_and_persists_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let state = RolloutState {
            schema_version: ROLLOUT_STATE_SCHEMA_VERSION,
            plan_digest: "sha256:abc".to_string(),
            phase: RolloutPhase::WindowOpen,
            strategy: RolloutStrategy::BlueGreen,
            started_at_ms: 1,
            updated_at_ms: 2,
            window_deadline_ms: Some(3),
            replacement_workers: vec![RolloutWorkerRecord {
                worker_id: "worker-b".to_string(),
                pid: 42,
            }],
            shared_approvals_path: "/tmp/approvals".to_string(),
            target_node_config: "/tmp/target.toml".to_string(),
            target_config_digest: "sha256:def".to_string(),
            approval_lines: vec![
                "v1|http://127.0.0.1:9480|node-a|worker-b|chat|chat-model|chat-prod|2".to_string(),
            ],
            removed_lines: Vec::new(),
        };
        persist_state(dir.path(), &state).unwrap();
        assert_eq!(load_state(dir.path()).unwrap().as_ref(), Some(&state));
        assert!(state_path(dir.path()).extension().is_some());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(state_path(dir.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "rollout state is owner-only");
        }
        clear_state(dir.path()).unwrap();
        assert_eq!(load_state(dir.path()).unwrap(), None);
        clear_state(dir.path()).unwrap(); // idempotent

        assert!(!RolloutPhase::WindowOpen.is_terminal());
        assert!(RolloutPhase::Committed.is_terminal());
        assert!(RolloutPhase::Aborted.is_terminal());
    }
}
