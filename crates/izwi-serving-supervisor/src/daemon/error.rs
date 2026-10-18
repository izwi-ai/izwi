//! The supervisor process error.

use izwi_serving_protocol::WorkerId;
use std::io;
use std::path::PathBuf;

use crate::{rollout, WorkerBinaryFlavor};

#[derive(Debug, thiserror::Error)]
pub enum SupervisorError {
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
    Config(#[from] crate::ConfigError),
    #[error(transparent)]
    Lock(#[from] crate::LockError),
    #[error(transparent)]
    Lifecycle(#[from] crate::LifecycleError),
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
    #[error(transparent)]
    Environment(#[from] izwi_serving_protocol::env::EnvError),
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
