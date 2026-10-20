//! The environment a supervised worker starts with.
//!
//! The supervisor clears the child environment, restores a short inherited
//! allowlist, applies the operator's validated `[workers.environment]`
//! table, and finally sets the worker's identity, assignment and deployment
//! variables itself. The table exists for operator switches (`IZWI_*`, for
//! example `IZWI_KV_HOST_OFFLOAD=0`) and loader paths (`LD_LIBRARY_PATH`,
//! `CUDA_HOME`, `HF_HOME`); it can never carry a secret or a variable the
//! supervisor owns.

use std::collections::BTreeMap;
use std::ffi::OsString;

use izwi_serving_protocol::WorkerId;

use crate::ConfigError;

/// Environment variables that may be deliberately inherited by a worker.
/// Everything else is removed before these and the assignment variables are set.
/// The allocator switch passes an operator's own value through; unset, a
/// worker gets the serving default ([`izwi_env::serving_allocator_env`]).
pub const INHERITED_ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "RUST_LOG",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    izwi_env::allocator::MALLOC_LARGE_CACHE_ENV,
];

pub const MAX_WORKER_ENVIRONMENT_ENTRIES: usize = 64;
pub const MAX_WORKER_ENVIRONMENT_VALUE_BYTES: usize = 4 * 1024;
pub const MAX_WORKER_ENVIRONMENT_TOTAL_BYTES: usize = 64 * 1024;

/// Names outside the `IZWI_` prefix a worker table may set: dynamic loader
/// and runtime paths a CUDA or model-cache install may need.
pub const WORKER_ENVIRONMENT_PASSTHROUGH: &[&str] = &[
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "RUST_BACKTRACE",
];

/// Every variable the supervisor sets for a worker. A worker table may not
/// name one: the supervisor's value always wins, so setting it could only
/// mislead.
pub const SUPERVISOR_OWNED_ENVIRONMENT: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "IZWI_BACKEND",
    "IZWI_CPU_MEMORY_BUDGET_BYTES",
    "IZWI_CUDA_DEVICE_ORDINAL",
    "IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES",
    "IZWI_CUDA_MEMORY_BUDGET_BYTES",
    "IZWI_KV_HOST_POOL_BUDGET_BYTES",
    "IZWI_METAL_DEVICE_ORDINAL",
    "IZWI_METAL_MEMORY_BUDGET_BYTES",
    "IZWI_MODELS_DIR",
    "IZWI_WORKER_ARTIFACT_REVISION",
    "IZWI_WORKER_ATTEMPT_RETENTION_SECS",
    "IZWI_WORKER_BEARER_TOKEN",
    "IZWI_WORKER_BIND",
    "IZWI_WORKER_CANCELLATION_GRACE_MS",
    "IZWI_WORKER_CPU_AFFINITY",
    "IZWI_WORKER_CPU_THREADS",
    "IZWI_WORKER_CREDENTIAL_ID",
    "IZWI_WORKER_DEPLOYMENT_ID",
    "IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES",
    "IZWI_WORKER_DRAIN_GRACE_MS",
    "IZWI_WORKER_EXECUTION_REPRESENTATION",
    "IZWI_WORKER_EXPECTED_DEVICE_ID",
    "IZWI_WORKER_EXPECTED_DEVICE_UUID",
    "IZWI_WORKER_GENERATION_FENCE_LOCK",
    "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
    "IZWI_WORKER_ID",
    "IZWI_WORKER_INCARNATION_ID",
    "IZWI_WORKER_MANAGED",
    "IZWI_WORKER_MAX_ACTIVE",
    "IZWI_WORKER_MAX_CONTEXT_TOKENS",
    "IZWI_WORKER_MAX_OUTPUT_TOKENS",
    "IZWI_WORKER_MAX_REQUEST_BYTES",
    "IZWI_WORKER_MAX_RETAINED_ATTEMPTS",
    "IZWI_WORKER_MODEL",
    "IZWI_WORKER_MODEL_GENERATION",
    "IZWI_WORKER_MODEL_LOAD_LOCK",
    "IZWI_WORKER_MODEL_LOAD_SLOTS",
    "IZWI_WORKER_NODE_ID",
    "IZWI_WORKER_OWNERSHIP_LOCK",
    "IZWI_WORKER_PRECISION",
    "IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES",
    "IZWI_WORKER_STREAMING",
    "IZWI_WORKER_TASK",
    "IZWI_WORKER_TERMINATION_GRACE_MS",
];

/// Name fragments that mark a credential. Secrets reach a worker only
/// through its `bearer_token_env`, resolved from the supervisor's own
/// environment, never through config text.
const SECRET_NAME_MARKERS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "API_KEY",
    "PRIVATE_KEY",
];

/// The allowlisted variables of the supervisor's own environment.
pub fn inherited_environment() -> BTreeMap<OsString, OsString> {
    INHERITED_ENV_ALLOWLIST
        .iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (OsString::from(name), value)))
        .collect()
}

/// Validates one worker's `[workers.environment]` table.
pub fn validate_worker_environment(
    worker: &WorkerId,
    bearer_token_env: &str,
    environment: &BTreeMap<String, String>,
) -> Result<(), ConfigError> {
    let forbidden = |name: &str, reason: &'static str| ConfigError::ForbiddenWorkerEnvironment {
        worker: worker.clone(),
        name: name.to_string(),
        reason,
    };
    if environment.len() > MAX_WORKER_ENVIRONMENT_ENTRIES {
        return Err(ConfigError::WorkerEnvironmentTooLarge {
            worker: worker.clone(),
        });
    }
    let mut total = 0_usize;
    for (name, value) in environment {
        crate::config::validate_env_name(name)?;
        if !name.starts_with("IZWI_") && !WORKER_ENVIRONMENT_PASSTHROUGH.contains(&name.as_str()) {
            return Err(forbidden(
                name,
                "only IZWI_* switches and the loader paths LD_LIBRARY_PATH, CUDA_HOME, CUDA_PATH, HF_HOME, HF_HUB_OFFLINE and RUST_BACKTRACE may be set",
            ));
        }
        if SUPERVISOR_OWNED_ENVIRONMENT.contains(&name.as_str()) {
            return Err(forbidden(
                name,
                "the supervisor sets it from the worker's validated config",
            ));
        }
        if name == bearer_token_env
            || SECRET_NAME_MARKERS
                .iter()
                .any(|marker| name.contains(marker))
            || name.ends_with("_KEY")
        {
            return Err(forbidden(
                name,
                "it names a credential; pass secrets only through bearer_token_env",
            ));
        }
        if value.len() > MAX_WORKER_ENVIRONMENT_VALUE_BYTES || value.contains('\0') {
            return Err(forbidden(
                name,
                "its value is longer than 4 KiB or contains a NUL byte",
            ));
        }
        total = total.saturating_add(name.len()).saturating_add(value.len());
    }
    if total > MAX_WORKER_ENVIRONMENT_TOTAL_BYTES {
        return Err(ConfigError::WorkerEnvironmentTooLarge {
            worker: worker.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker() -> WorkerId {
        WorkerId::new("worker-a").unwrap()
    }

    fn validate(entries: &[(&str, &str)]) -> Result<(), ConfigError> {
        let environment = entries
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        validate_worker_environment(&worker(), "WORKER_A_TOKEN", &environment)
    }

    #[test]
    fn operator_switches_and_loader_paths_are_accepted() {
        validate(&[
            ("IZWI_KV_HOST_OFFLOAD", "0"),
            ("IZWI_ENABLE_PREFIX_CACHING", "0"),
            ("IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS", "60000"),
            ("IZWI_WORKER_EXIT_ON_RUNTIME_FAILURE", "false"),
            ("LD_LIBRARY_PATH", "/usr/local/cuda/lib64"),
            ("CUDA_HOME", "/usr/local/cuda"),
            ("HF_HOME", "/var/lib/izwi/hf"),
        ])
        .unwrap();
    }

    #[test]
    fn owned_secret_and_foreign_names_are_rejected() {
        for name in [
            "IZWI_WORKER_ID",
            "IZWI_BACKEND",
            "CUDA_VISIBLE_DEVICES",
            "IZWI_WORKER_BEARER_TOKEN",
            "IZWI_API_KEY",
            "IZWI_GATEWAY_ADMIN_API_KEY_REF",
            "IZWI_DB_PASSWORD",
            "HF_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "PATH",
            "DYLD_INSERT_LIBRARIES",
            "lowercase",
        ] {
            assert!(validate(&[(name, "x")]).is_err(), "{name} must be rejected");
        }
        let environment = BTreeMap::from([("IZWI_WORKER_A".to_string(), "x".to_string())]);
        assert!(
            validate_worker_environment(&worker(), "IZWI_WORKER_A", &environment).is_err(),
            "the worker's own secret variable is rejected"
        );
    }

    #[test]
    fn values_and_tables_are_bounded() {
        let long = "x".repeat(MAX_WORKER_ENVIRONMENT_VALUE_BYTES + 1);
        assert!(validate(&[("IZWI_X", long.as_str())]).is_err());
        assert!(validate(&[("IZWI_X", "a\0b")]).is_err());
        let names: Vec<String> = (0..=MAX_WORKER_ENVIRONMENT_ENTRIES)
            .map(|index| format!("IZWI_X{index}"))
            .collect();
        let entries: Vec<(&str, &str)> = names.iter().map(|name| (name.as_str(), "1")).collect();
        assert!(validate(&entries).is_err());
    }
}
