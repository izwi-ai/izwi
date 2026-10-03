//! DS6 coordinated blue-green rollout: plan, state, and approvals views.
//!
//! A rollout replaces one node's deployment generations in one operator
//! action. The declarative plan names a target node config (the new
//! generations), the shared approvals file the gateway reads, a canary
//! worker, and a soak window. The coordinator state machine is
//!
//! `launching replacement (canary first) → window open (both generations
//! approved; the gateway cuts over when the successor first observes Ready)
//! → draining old → committed`, with an abort path that restores the
//! pre-rollout approvals byte-identically and stops only the replacement
//! workers — the current generation never stops serving, and the drain step
//! is the single point of no return (worker drain closes the ownership pipe
//! irreversibly).
//!
//! Rollout state persists in the supervisor's runtime directory so a
//! crashed supervisor resumes or aborts explicitly instead of silently
//! relaunching the old generation mid-rollout. The shared approvals file
//! stays the only supervisor-to-gateway coupling: the coordinator writes
//! whole approval views atomically (temp file plus rename under a sidecar
//! lock), and the gateway adopts changed views through the DINV-07
//! eligibility rule.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use fs2::FileExt;
use izwi_serving_protocol::{
    GatewayWorkerApproval, GatewayWorkerApprovalIdentity, ModelGeneration, TaskKind,
    MAX_APPROVALS_FILE_BYTES,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ValidatedNodeConfig, WorkerConfig};

/// Hard ceiling on the rollout plan file.
pub const MAX_ROLLOUT_PLAN_BYTES: usize = 64 * 1024;
/// Hard ceiling on the soak window.
pub const MAX_ROLLOUT_WINDOW_SECS: u64 = 3600;
/// Hard ceiling on the post-abort routing grace.
pub const MAX_ROLLOUT_ABORT_GRACE_SECS: u64 = 300;
/// Rollout state file name inside the supervisor's runtime directory.
pub const ROLLOUT_STATE_FILE: &str = "rollout-state.json";
/// Pre-rollout approvals backup file name inside the runtime directory.
pub const ROLLOUT_APPROVALS_BACKUP_FILE: &str = "rollout-approvals-backup";

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
    /// Soak window between opening the rollout window and draining the old
    /// generation. Zero skips the wait.
    #[serde(default = "default_window_secs")]
    window_secs: u64,
    /// Grace between restoring the approvals on abort and stopping the
    /// replacement workers, so the gateway stops routing to them first.
    #[serde(default = "default_abort_grace_secs")]
    abort_grace_secs: u64,
}

fn default_window_secs() -> u64 {
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
    pub new_worker_ids: Vec<String>,
}

/// The validated rollout: what rolls, in what order, and with what policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutSpec {
    pub rolling: Vec<RollingDeployment>,
    pub canary_worker_id: String,
    pub window: Duration,
    /// After abort restores the previous approval view, the coordinator
    /// waits this long before stopping the replacement workers so the
    /// gateway's approvals refresh stops routing to them first.
    pub abort_grace: Duration,
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
        if self.window_secs > MAX_ROLLOUT_WINDOW_SECS {
            return Err(RolloutError::InvalidPlan(format!(
                "window_secs must be at most {MAX_ROLLOUT_WINDOW_SECS}"
            )));
        }
        if self.abort_grace_secs > MAX_ROLLOUT_ABORT_GRACE_SECS {
            return Err(RolloutError::InvalidPlan(format!(
                "abort_grace_secs must be at most {MAX_ROLLOUT_ABORT_GRACE_SECS}"
            )));
        }
        Ok(())
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
        hasher.update(self.window_secs.to_le_bytes());
        hasher.update([0]);
        hasher.update(self.abort_grace_secs.to_le_bytes());
        hasher.update([0]);
        hasher.update(target_config_bytes);
        format!("sha256:{:x}", hasher.finalize())
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
                    if format!("{current_worker:?}") != format!("{worker:?}") {
                        return Err(RolloutError::InvalidPlan(format!(
                            "worker {} of non-rolling deployment {deployment_id} differs between current and target configs; rollouts only advance generations",
                            worker.worker_id
                        )));
                    }
                }
                continue;
            }
            let old_worker_ids = current
                .config()
                .workers
                .iter()
                .filter(|worker| worker.deployment.deployment_id.to_string() == *deployment_id)
                .map(|worker| worker.worker_id.to_string())
                .collect();
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
            rolling,
            canary_worker_id: self.canary_worker_id.clone(),
            window: Duration::from_secs(self.window_secs),
            abort_grace: Duration::from_secs(self.abort_grace_secs),
        })
    }
}

/// Persisted rollout state (DS6.2 resume-after-crash).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutState {
    pub schema_version: u16,
    pub plan_digest: String,
    pub phase: RolloutPhase,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    #[serde(default)]
    pub window_deadline_ms: Option<u64>,
    #[serde(default)]
    pub replacement_workers: Vec<RolloutWorkerRecord>,
    pub shared_approvals_path: String,
    pub target_node_config: String,
    pub target_config_digest: String,
    pub approvals_backup_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutPhase {
    LaunchingReplacement,
    WindowOpen,
    DrainingOld,
    Committed,
    Aborted,
}

impl RolloutPhase {
    /// Terminal phases are safe to auto-clear on a fresh supervisor start:
    /// nothing remains to resume or roll back.
    pub fn is_terminal(self) -> bool {
        matches!(self, RolloutPhase::Committed | RolloutPhase::Aborted)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RolloutPhase::LaunchingReplacement => "launching_replacement",
            RolloutPhase::WindowOpen => "window_open",
            RolloutPhase::DrainingOld => "draining_old",
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
    if state.schema_version != 1 {
        return Err(RolloutError::InvalidState(format!(
            "unsupported rollout state schema_version {}",
            state.schema_version
        )));
    }
    Ok(Some(state))
}

/// Persists rollout state atomically (temp file plus rename, 0600).
pub fn persist_state(runtime_directory: &Path, state: &RolloutState) -> Result<(), RolloutError> {
    let bytes = serde_json::to_vec_pretty(state)
        .map_err(|error| RolloutError::InvalidState(error.to_string()))?;
    write_atomic(&state_path(runtime_directory), &bytes)
}

/// Removes persisted rollout state.
pub fn clear_state(runtime_directory: &Path) -> Result<(), RolloutError> {
    match std::fs::remove_file(state_path(runtime_directory)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// One approvals line with its parsed meaning, preserving the operator's
/// raw text for verbatim pass-through in generated views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedApproval {
    pub raw: String,
    pub approval: GatewayWorkerApproval,
}

/// Parses an approvals file into classified lines. Any invalid entry fails
/// the whole file — the coordinator never rewrites a file it cannot fully
/// understand.
pub fn classify_approvals(text: &str) -> Result<Vec<ClassifiedApproval>, RolloutError> {
    let mut classified = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let entry = line.split('#').next().unwrap_or("").trim();
        if entry.is_empty() {
            continue;
        }
        let approval = entry.parse::<GatewayWorkerApproval>().map_err(|error| {
            RolloutError::InvalidApprovals(format!("line {}: {error}", index + 1))
        })?;
        classified.push(ClassifiedApproval {
            raw: entry.to_string(),
            approval,
        });
    }
    Ok(classified)
}

/// Verifies the rollout precondition: the approvals file currently approves
/// the old generation of every rolling deployment. A rollout that rewrites
/// a file the gateway is not actually serving from would silently strand
/// admission.
pub fn verify_approvals_precondition(
    classified: &[ClassifiedApproval],
    spec: &RolloutSpec,
) -> Result<(), RolloutError> {
    for deployment in &spec.rolling {
        let approved = classified.iter().any(|entry| {
            entry.approval.deployment_id.as_str() == deployment.deployment_id
                && entry.approval.model_generation == deployment.old_generation
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

fn replacement_lines(
    spec: &RolloutSpec,
    target: &ValidatedNodeConfig,
) -> Result<Vec<GatewayWorkerApproval>, RolloutError> {
    let node_id = target.config().node_id.clone();
    let mut lines = Vec::new();
    for deployment in &spec.rolling {
        for worker_id in &deployment.new_worker_ids {
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
            lines.push(GatewayWorkerApproval {
                endpoint: format!("http://{}", worker.bind),
                identity: GatewayWorkerApprovalIdentity::V1 {
                    node_id: node_id.clone(),
                    worker_id: worker.worker_id.clone(),
                },
                task: deployment.task,
                public_model: worker.deployment.public_model.clone(),
                deployment_id: worker.deployment.deployment_id.clone(),
                model_generation: deployment.new_generation,
            });
        }
    }
    Ok(lines)
}

/// Builds the window view: the old generation's operator lines stay
/// verbatim (it remains approved and admission-eligible until the gateway
/// observes the successor Ready), and one pinned line per replacement
/// worker is appended. Regenerating this view from an already-written
/// window file yields identical content, so resume rewrites are safe.
pub fn build_window_view(
    classified: &[ClassifiedApproval],
    spec: &RolloutSpec,
    target: &ValidatedNodeConfig,
) -> Result<String, RolloutError> {
    let mut text = String::new();
    for entry in classified {
        if is_rolling_new_line(entry, spec) {
            continue;
        }
        text.push_str(&entry.raw);
        text.push('\n');
    }
    for approval in replacement_lines(spec, target)? {
        text.push_str(&approval.render_line());
        text.push('\n');
    }
    Ok(text)
}

/// Builds the commit view: every line of the rolling deployments' old
/// generation is removed and the replacement lines stay. Adopting this view
/// makes the new generation the only approved one.
pub fn build_commit_view(
    classified: &[ClassifiedApproval],
    spec: &RolloutSpec,
    target: &ValidatedNodeConfig,
) -> Result<String, RolloutError> {
    let mut text = String::new();
    for entry in classified {
        if is_rolling_line(entry, spec) {
            continue;
        }
        text.push_str(&entry.raw);
        text.push('\n');
    }
    for approval in replacement_lines(spec, target)? {
        text.push_str(&approval.render_line());
        text.push('\n');
    }
    Ok(text)
}

/// Whether one classified line is a replacement line of a rolling
/// deployment (regenerated by the views, never passed through).
fn is_rolling_new_line(entry: &ClassifiedApproval, spec: &RolloutSpec) -> bool {
    spec.rolling.iter().any(|deployment| {
        entry.approval.deployment_id.as_str() == deployment.deployment_id
            && entry.approval.model_generation == deployment.new_generation
    })
}

/// Whether one classified line belongs to a rolling deployment at either
/// generation (the commit view drops both and regenerates replacements).
fn is_rolling_line(entry: &ClassifiedApproval, spec: &RolloutSpec) -> bool {
    spec.rolling.iter().any(|deployment| {
        entry.approval.deployment_id.as_str() == deployment.deployment_id
            && (entry.approval.model_generation == deployment.old_generation
                || entry.approval.model_generation == deployment.new_generation)
    })
}

/// Reads a bounded approvals file.
pub fn read_approvals_file(path: &Path) -> Result<String, RolloutError> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| RolloutError::InvalidApprovals(format!("{}: {error}", path.display())))?;
    if !metadata.is_file() {
        return Err(RolloutError::InvalidApprovals(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_APPROVALS_FILE_BYTES {
        return Err(RolloutError::InvalidApprovals(format!(
            "{} exceeds the {} byte approvals size limit",
            path.display(),
            MAX_APPROVALS_FILE_BYTES
        )));
    }
    std::fs::read_to_string(path)
        .map_err(|error| RolloutError::InvalidApprovals(format!("{}: {error}", path.display())))
}

/// Writes bytes atomically: temp file in the target directory (0600) plus
/// rename, serialized by an exclusive sidecar lock so a concurrent writer
/// can never interleave.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), RolloutError> {
    let directory = path.parent().ok_or_else(|| {
        RolloutError::Io(std::io::Error::other(format!(
            "{} has no parent directory",
            path.display()
        )))
    })?;
    std::fs::create_dir_all(directory)?;
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "rollout-file".to_string());
    let lock_path = directory.join(format!(".{file_name}.rollout-lock"));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    lock.lock_exclusive()
        .map_err(|error| RolloutError::Io(std::io::Error::other(error.to_string())))?;
    let temp_path = directory.join(format!(
        ".{file_name}.rollout-temp-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| {
        {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .mode(0o600)
                .open(&temp_path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&temp_path, path)
    })();
    let _ = fs2::FileExt::unlock(&lock);
    drop(lock);
    let _ = std::fs::remove_file(&lock_path);
    result.map_err(|error| {
        let _ = std::fs::remove_file(&temp_path);
        RolloutError::Io(error)
    })
}

/// Saves the pre-rollout approvals bytes and returns their digest.
pub fn backup_approvals(runtime_directory: &Path, text: &str) -> Result<String, RolloutError> {
    let path = runtime_directory.join(ROLLOUT_APPROVALS_BACKUP_FILE);
    write_atomic(&path, text.as_bytes())?;
    Ok(digest_bytes(text.as_bytes()))
}

/// Restores the pre-rollout approvals bytes, refusing a backup whose digest
/// no longer matches the recorded one.
pub fn restore_approvals(
    runtime_directory: &Path,
    approvals_path: &Path,
    expected_digest: &str,
) -> Result<(), RolloutError> {
    let backup_path = runtime_directory.join(ROLLOUT_APPROVALS_BACKUP_FILE);
    let bytes = std::fs::read(&backup_path).map_err(|error| {
        RolloutError::InvalidApprovals(format!(
            "pre-rollout approvals backup {} is unreadable: {error}",
            backup_path.display()
        ))
    })?;
    let digest = digest_bytes(&bytes);
    if digest != expected_digest {
        return Err(RolloutError::InvalidApprovals(format!(
            "pre-rollout approvals backup digest {digest} does not match the recorded {expected_digest}"
        )));
    }
    write_atomic(approvals_path, &bytes)
}

/// Removes the approvals backup after a terminal phase.
pub fn clear_approvals_backup(runtime_directory: &Path) -> Result<(), RolloutError> {
    match std::fs::remove_file(runtime_directory.join(ROLLOUT_APPROVALS_BACKUP_FILE)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub fn digest_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use izwi_serving_protocol::NodeId;

    use super::*;

    fn plan_toml(target: &str, approvals: &str, canary: &str, window: u64) -> String {
        format!(
            "schema_version = 1\ntarget_node_config = \"{target}\"\nshared_approvals_path = \"{approvals}\"\ncanary_worker_id = \"{canary}\"\nwindow_secs = {window}\n"
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
        let output = if task == "chat" {
            "[\"text\"]"
        } else {
            "[\"text\"]"
        };
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

    fn validated(toml_text: &str) -> ValidatedNodeConfig {
        std::fs::create_dir_all("/tmp/izwi-work").unwrap();
        std::fs::create_dir_all("/tmp/izwi-run").unwrap();
        std::fs::create_dir_all("/tmp/izwi-models/chat").unwrap();
        let config = crate::config::NodeConfig::parse_bounded(toml_text.as_bytes()).unwrap();
        let inventory = crate::config::HostInventory {
            effective_cpu_ids: vec![0, 1, 2, 3],
            allocatable_host_memory_bytes: u64::MAX,
            metal_devices: vec![],
            cuda_devices: vec![],
        };
        let binaries = crate::config::BinaryCatalog::new(vec![(
            crate::config::WorkerBinaryFlavor::Cpu,
            crate::config::BinaryRecord {
                path: "/usr/bin/true".into(),
                supported_backends: vec![izwi_serving_protocol::BackendKind::Cpu],
            },
        )]);
        config.validate(&inventory, &binaries).unwrap()
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
        assert_eq!(plan.window_secs, 30);
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

    fn classified(text: &str) -> Vec<ClassifiedApproval> {
        classify_approvals(text).unwrap()
    }

    fn sample_classified() -> Vec<ClassifiedApproval> {
        classified(
            "# fleet\nv1|http://127.0.0.1:9470|node-a|worker-a|chat|chat-model|chat-prod|1\nv1|http://127.0.0.1:9600|node-a|asr-worker|speech_to_text|asr-model|asr-prod|4\n",
        )
    }

    fn sample_spec() -> RolloutSpec {
        RolloutSpec {
            rolling: vec![RollingDeployment {
                deployment_id: "chat-prod".to_string(),
                task: TaskKind::Chat,
                public_model: "chat-model".to_string(),
                old_generation: ModelGeneration::new(1).unwrap(),
                new_generation: ModelGeneration::new(2).unwrap(),
                old_worker_ids: vec!["worker-a".to_string()],
                new_worker_ids: vec!["worker-b".to_string()],
            }],
            canary_worker_id: "worker-b".to_string(),
            window: Duration::from_secs(10),
            abort_grace: Duration::from_secs(5),
        }
    }

    fn sample_target() -> ValidatedNodeConfig {
        validated(&target_toml(2))
    }

    #[test]
    fn window_view_keeps_old_lines_and_appends_pinned_replacements() {
        let classified = sample_classified();
        let spec = sample_spec();
        let window = build_window_view(&classified, &spec, &sample_target()).unwrap();
        let parsed = izwi_serving_protocol::parse_approvals_text(&window).unwrap();
        assert_eq!(parsed.len(), 3);
        assert!(window.contains("worker-a|chat|chat-model|chat-prod|1"));
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
    }

    #[test]
    fn views_are_idempotent_over_an_already_written_window_file() {
        let spec = sample_spec();
        let window_once = build_window_view(&sample_classified(), &spec, &sample_target()).unwrap();
        let window_twice = build_window_view(
            &classify_approvals(&window_once).unwrap(),
            &spec,
            &sample_target(),
        )
        .unwrap();
        assert_eq!(window_once, window_twice);
        let commit = build_commit_view(
            &classify_approvals(&window_once).unwrap(),
            &spec,
            &sample_target(),
        )
        .unwrap();
        let commit_twice = build_commit_view(
            &classify_approvals(&commit).unwrap(),
            &spec,
            &sample_target(),
        )
        .unwrap();
        assert_eq!(commit, commit_twice);
    }

    #[test]
    fn commit_view_drops_the_old_generation_and_keeps_the_rest() {
        let classified = sample_classified();
        let spec = sample_spec();
        let commit = build_commit_view(&classified, &spec, &sample_target()).unwrap();
        let parsed = izwi_serving_protocol::parse_approvals_text(&commit).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(!commit.contains("worker-a|chat|chat-model|chat-prod|1"));
        assert!(commit.contains("asr-prod|4"));
        assert!(commit.contains("worker-b|chat|chat-model|chat-prod|2"));
    }

    #[test]
    fn approvals_precondition_requires_the_current_generation() {
        let spec = sample_spec();
        verify_approvals_precondition(&sample_classified(), &spec).unwrap();

        let stale =
            classified("v1|http://127.0.0.1:9470|node-a|worker-a|chat|chat-model|chat-prod|9\n");
        assert!(matches!(
            verify_approvals_precondition(&stale, &spec).unwrap_err(),
            RolloutError::InvalidApprovals(_)
        ));

        assert!(matches!(
            classify_approvals("http://bad|line\n").unwrap_err(),
            RolloutError::InvalidApprovals(_)
        ));
    }

    #[test]
    fn state_round_trips_and_persists_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let state = RolloutState {
            schema_version: 1,
            plan_digest: "sha256:abc".to_string(),
            phase: RolloutPhase::WindowOpen,
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
            approvals_backup_digest: "sha256:123".to_string(),
        };
        persist_state(dir.path(), &state).unwrap();
        assert_eq!(load_state(dir.path()).unwrap().as_ref(), Some(&state));
        assert!(matches!(
            state_path(dir.path()).extension(),
            Some(_) // .json
        ));
        clear_state(dir.path()).unwrap();
        assert_eq!(load_state(dir.path()).unwrap(), None);
        clear_state(dir.path()).unwrap(); // idempotent

        assert!(!RolloutPhase::WindowOpen.is_terminal());
        assert!(RolloutPhase::Committed.is_terminal());
        assert!(RolloutPhase::Aborted.is_terminal());
    }

    #[test]
    fn approvals_backup_restore_is_digest_guarded() {
        let dir = tempfile::tempdir().unwrap();
        let approvals = dir.path().join("approvals.txt");
        std::fs::write(
            &approvals,
            "http://127.0.0.1:9470|chat|chat-model|chat-prod|1\n",
        )
        .unwrap();

        let original = read_approvals_file(&approvals).unwrap();
        assert_eq!(classify_approvals(&original).unwrap().len(), 1);
        let digest = backup_approvals(dir.path(), &original).unwrap();

        // The rollout rewrites the file; the abort restores it byte-identically.
        write_atomic(
            &approvals,
            b"v1|http://127.0.0.1:9471|node-a|worker-b|chat|chat-model|chat-prod|2\n",
        )
        .unwrap();
        restore_approvals(dir.path(), &approvals, &digest).unwrap();
        assert_eq!(read_approvals_file(&approvals).unwrap(), original);

        // A tampered backup is refused.
        std::fs::write(dir.path().join(ROLLOUT_APPROVALS_BACKUP_FILE), b"tampered").unwrap();
        assert!(matches!(
            restore_approvals(dir.path(), &approvals, &digest).unwrap_err(),
            RolloutError::InvalidApprovals(_)
        ));

        // The window view actually replaces the file through the atomic path.
        let window = build_window_view(
            &classify_approvals(&original).unwrap(),
            &sample_spec(),
            &sample_target(),
        )
        .unwrap();
        write_atomic(&approvals, window.as_bytes()).unwrap();
        assert_eq!(parse_len(&approvals), 2);
        clear_approvals_backup(dir.path()).unwrap();
    }

    fn parse_len(path: &Path) -> usize {
        izwi_serving_protocol::parse_approvals_text(&read_approvals_file(path).unwrap())
            .unwrap()
            .len()
    }

    #[test]
    fn oversized_approvals_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("approvals.txt");
        std::fs::write(&path, vec![b'a'; MAX_APPROVALS_FILE_BYTES as usize + 1]).unwrap();
        assert!(matches!(
            read_approvals_file(&path).unwrap_err(),
            RolloutError::InvalidApprovals(_)
        ));
    }
}
