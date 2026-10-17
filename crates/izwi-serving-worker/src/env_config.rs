//! Every environment variable `izwi-serving-worker` reads (HYG-E1).
//!
//! `main` captures these once at startup into an [`EnvSnapshot`] and builds
//! the worker's typed configuration from it; nothing reads the process
//! environment afterwards. The registry also renders
//! `docs/dev/env/serving-worker.md`, and the tests below keep that file and
//! the registry in step with every `IZWI_*` name in this crate's source.
//! The model runtime (izwi-core) reads further engine variables of its own.

use izwi_core::serve_runtime::{
    ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS, ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
    ENV_ASR_REALTIME_MAX_SESSIONS, ENV_KV_HOST_OFFLOAD, ENV_KV_HOST_POOL_BUDGET_BYTES,
    ENV_KV_OFFLOAD_HIGH_WATERMARK, ENV_KV_OFFLOAD_LOW_WATERMARK,
    ENV_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES, ENV_KV_OFFLOAD_MAX_PROMOTION_PAGES,
    ENV_MAX_TOKENS_PER_STEP, ENV_PREFIX_REUSE_AUTO,
};
use izwi_serving_fence::{
    MANAGED_WORKER_ENV, WORKER_EXECUTION_REPRESENTATION_ENV, WORKER_GENERATION_FENCE_ENV,
    WORKER_INCARNATION_ENV, WORKER_MAX_CONTEXT_TOKENS_ENV, WORKER_MAX_OUTPUT_TOKENS_ENV,
    WORKER_MODEL_LOAD_LOCK_ENV, WORKER_MODEL_LOAD_SLOTS_ENV, WORKER_OWNERSHIP_LOCK_ENV,
    WORKER_PRECISION_ENV, WORKER_TASK_ENV,
};
use izwi_serving_protocol::env::{EnvSnapshot, EnvVar};

pub(crate) const WORKER_TLS_CERT_REF_ENV: &str = "IZWI_WORKER_TLS_CERT_REF";
pub(crate) const WORKER_TLS_KEY_REF_ENV: &str = "IZWI_WORKER_TLS_KEY_REF";
pub(crate) const WORKER_TLS_CLIENT_CA_REF_ENV: &str = "IZWI_WORKER_TLS_CLIENT_CA_REF";

const fn var(name: &'static str, default: &'static str, description: &'static str) -> EnvVar {
    EnvVar {
        name,
        default,
        description,
    }
}

pub(crate) static WORKER_ENV: &[EnvVar] = &[
    // Process
    var("IZWI_LOG_FORMAT", "text", "Log line format: `text` or `json`."),
    var(
        "IZWI_OTLP_TRACES_ENDPOINT",
        "",
        "OTLP/HTTP collector for request spans; unset exports none.",
    ),
    var(
        "IZWI_WORKER_BIND",
        "127.0.0.1:9470",
        "Listen address. A non-loopback address needs TLS or IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND.",
    ),
    var(
        "IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND",
        "false",
        "Acknowledge plaintext on a non-loopback address (container networking).",
    ),
    var(
        WORKER_TLS_CERT_REF_ENV,
        "",
        "`file:/absolute/path` of the server certificate PEM; set with the key to serve TLS.",
    ),
    var(
        WORKER_TLS_KEY_REF_ENV,
        "",
        "`file:/absolute/path` of the server private key PEM (warned about when other users can read it).",
    ),
    var(
        WORKER_TLS_CLIENT_CA_REF_ENV,
        "",
        "`file:/absolute/path` of a client CA PEM; set to require client certificates (mTLS).",
    ),
    var(
        "IZWI_MODELS_DIR",
        "the platform data directory's izwi/models",
        "Model artifact directory.",
    ),
    // Identity
    var("IZWI_WORKER_ID", "local-cpu-1", "Worker identity."),
    var("IZWI_WORKER_NODE_ID", "local-node", "Node identity."),
    var(
        WORKER_INCARNATION_ENV,
        "a random UUID",
        "Process incarnation identity.",
    ),
    var(
        "IZWI_WORKER_CREDENTIAL_ID",
        "",
        "Required. Credential id the gateway presents.",
    ),
    var(
        "IZWI_WORKER_BEARER_TOKEN",
        "",
        "Required. Bearer token the gateway presents.",
    ),
    var(
        MANAGED_WORKER_ENV,
        "false",
        "Set by the supervisor: the worker takes its node locks and requires exact memory budgets.",
    ),
    // Deployment
    var(
        WORKER_TASK_ENV,
        "chat",
        "`chat`, `speech_to_text` or `text_to_speech`.",
    ),
    var(
        "IZWI_WORKER_MODEL",
        "the task's default model",
        "Model to serve: LFM2.5-1.2B-Instruct-GGUF (chat), Nemotron-3.5-ASR-Streaming-0.6B (speech_to_text), Kokoro-82M (text_to_speech).",
    ),
    var(
        "IZWI_WORKER_DEPLOYMENT_ID",
        "lfm25-cpu-v1",
        "Deployment id advertised to the gateway.",
    ),
    var(
        "IZWI_WORKER_ARTIFACT_REVISION",
        "",
        "Required. Model artifact revision; must match the artifact manifest.",
    ),
    var(
        "IZWI_WORKER_MODEL_GENERATION",
        "1",
        "Model generation advertised to the gateway.",
    ),
    var(
        WORKER_PRECISION_ENV,
        "derived from the model",
        "Precision the supervisor approved; refused when the loaded model differs.",
    ),
    var(
        WORKER_EXECUTION_REPRESENTATION_ENV,
        "derived from the model",
        "Execution representation the supervisor approved.",
    ),
    var(
        WORKER_MAX_CONTEXT_TOKENS_ENV,
        "derived from the model",
        "Context limit the supervisor approved; must not exceed the fitted context.",
    ),
    var(
        WORKER_MAX_OUTPUT_TOKENS_ENV,
        "derived from the model",
        "Output-token limit the supervisor approved.",
    ),
    // Admission and lifecycle
    var(
        "IZWI_WORKER_MAX_ACTIVE",
        "1",
        "Concurrent invocations (non-zero).",
    ),
    var(
        "IZWI_WORKER_MAX_REQUEST_BYTES",
        "1048576",
        "Largest invocation body in bytes.",
    ),
    var(
        "IZWI_WORKER_MAX_RETAINED_ATTEMPTS",
        "1024",
        "Finished attempts kept for status queries.",
    ),
    var(
        "IZWI_WORKER_ATTEMPT_RETENTION_SECS",
        "300",
        "How long a finished attempt stays queryable.",
    ),
    var(
        "IZWI_WORKER_STREAMING",
        "true",
        "Advertise and serve streaming invocations.",
    ),
    var(
        "IZWI_WORKER_DRAIN_GRACE_MS",
        "30000",
        "Shutdown drain grace (non-zero, at most one day).",
    ),
    var(
        "IZWI_WORKER_CANCELLATION_GRACE_MS",
        "10000",
        "Grace for in-flight work after cancellation (non-zero, at most one day).",
    ),
    var(
        "IZWI_WORKER_TERMINATION_GRACE_MS",
        "5000",
        "How long open connections may outlast the drain and cancellation graces, or a managed worker's SIGTERM, before the worker exits (non-zero, at most one day).",
    ),
    var(
        "IZWI_WORKER_EXIT_ON_RUNTIME_FAILURE",
        "true",
        "Exit when the runtime fails permanently so the supervisor restarts the worker.",
    ),
    var(
        WORKER_OWNERSHIP_LOCK_ENV,
        "",
        "Set by the supervisor (managed workers): resource ownership lock path.",
    ),
    var(
        WORKER_GENERATION_FENCE_ENV,
        "",
        "Set by the supervisor (managed workers): generation fence lock path.",
    ),
    var(
        WORKER_MODEL_LOAD_LOCK_ENV,
        "",
        "Set by the supervisor (managed workers): node model-load lock path.",
    ),
    var(
        WORKER_MODEL_LOAD_SLOTS_ENV,
        "1",
        "Set by the supervisor (managed workers): concurrent model loads per node.",
    ),
    // Device assignment
    var("IZWI_BACKEND", "cpu", "`cpu`, `metal` or `cuda`."),
    var(
        "IZWI_WORKER_CPU_THREADS",
        "1",
        "CPU thread budget (cpu backend, non-zero).",
    ),
    var(
        "IZWI_WORKER_CPU_AFFINITY",
        "",
        "Comma-separated CPU ids (advisory on this build).",
    ),
    var(
        "IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES",
        "2147483648",
        "Host memory assignment (cpu and cuda backends).",
    ),
    var(
        "IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES",
        "2147483648",
        "Unified memory assignment (metal backend).",
    ),
    var(
        "IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES",
        "2147483648",
        "Device memory assignment (cuda backend).",
    ),
    var(
        "IZWI_WORKER_EXPECTED_DEVICE_ID",
        "",
        "Required on metal: the assigned device's registry id.",
    ),
    var(
        "IZWI_WORKER_EXPECTED_DEVICE_UUID",
        "",
        "Required on cuda: the assigned device's UUID.",
    ),
    var(
        "IZWI_METAL_DEVICE_ORDINAL",
        "0",
        "Process-local Metal device index.",
    ),
    var(
        "IZWI_CUDA_DEVICE_ORDINAL",
        "0",
        "Process-local CUDA device index.",
    ),
    var(
        "IZWI_CPU_MEMORY_BUDGET_BYTES",
        "",
        "Runtime host memory budget; must equal the host assignment (required when managed).",
    ),
    var(
        "IZWI_METAL_MEMORY_BUDGET_BYTES",
        "",
        "Runtime Metal memory budget; must equal the unified assignment (required when managed).",
    ),
    var(
        "IZWI_CUDA_MEMORY_BUDGET_BYTES",
        "",
        "Runtime CUDA device budget; must equal the device assignment (required when managed).",
    ),
    var(
        "IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES",
        "",
        "Runtime CUDA host budget; must equal the host assignment (required when managed).",
    ),
    // Engine policy
    var(
        "IZWI_ENABLE_PREFIX_CACHING",
        "catalog auto",
        "Prefix caching: true enables it everywhere (needs a salt), false turns it off; unset leaves it to the catalog.",
    ),
    var(
        ENV_PREFIX_REUSE_AUTO,
        "true",
        "With IZWI_ENABLE_PREFIX_CACHING unset, reuse prefixes where the catalog has evidence.",
    ),
    var(
        "IZWI_MANAGED_PREFIX_CACHE_SALT",
        "",
        "Prefix-cache namespace; required when IZWI_ENABLE_PREFIX_CACHING is true.",
    ),
    var(
        "IZWI_MAX_PREFIX_CACHE_PAGES",
        "sized per model at load",
        "Prefix cache page budget.",
    ),
    var(
        "IZWI_ENABLE_CHUNKED_PREFILL",
        "false",
        "Split long prompts into prefill chunks.",
    ),
    var(
        "IZWI_CHUNKED_PREFILL_THRESHOLD",
        "the engine default",
        "Prompt length that turns chunked prefill on.",
    ),
    var(
        ENV_MAX_TOKENS_PER_STEP,
        "the engine default",
        "Token budget of one scheduler step.",
    ),
    var(
        "IZWI_MAX_SEQUENCE_LENGTH",
        "the model's context",
        "Explicit context-length cap.",
    ),
    var(
        ENV_KV_HOST_OFFLOAD,
        "on when a host pool budget is set",
        "false turns hierarchical KV offload off even with a budget.",
    ),
    var(
        ENV_KV_HOST_POOL_BUDGET_BYTES,
        "0 (off)",
        "Host KV pool budget for offload.",
    ),
    var(
        ENV_KV_OFFLOAD_HIGH_WATERMARK,
        "the engine default",
        "Device KV occupancy that starts offload.",
    ),
    var(
        ENV_KV_OFFLOAD_LOW_WATERMARK,
        "the engine default",
        "Device KV occupancy that stops offload.",
    ),
    var(
        ENV_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES,
        "the engine default",
        "Offload pages in flight at once.",
    ),
    var(
        ENV_KV_OFFLOAD_MAX_PROMOTION_PAGES,
        "the engine default",
        "Pages promoted back per step.",
    ),
    // Realtime ASR sessions
    var(
        ENV_ASR_REALTIME_MAX_SESSIONS,
        "16 on CUDA, 4 on CPU and Metal",
        "Concurrent realtime ASR sessions (also the state arena rows a load reserves).",
    ),
    var(
        ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
        "28800 (8 hours)",
        "Seconds one realtime ASR session may live.",
    ),
    var(
        ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS,
        "90",
        "Seconds without audio before a realtime ASR session is reaped.",
    ),
];

/// The environment `main` reads once at startup.
pub(crate) fn capture() -> EnvSnapshot {
    EnvSnapshot::capture(WORKER_ENV)
}

/// A snapshot of explicit values, for tests.
#[cfg(test)]
pub(crate) fn snapshot<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> EnvSnapshot {
    EnvSnapshot::from_pairs(WORKER_ENV, pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_protocol::env::{production_env_names, render_reference};
    use std::path::Path;

    const REFERENCE: &str = "docs/dev/env/serving-worker.md";

    fn reference() -> String {
        render_reference(
            "izwi-serving-worker environment",
            "The variables `izwi-serving-worker` reads, once, at startup. Booleans accept \
             `true`, `false`, `1`, `0`, `yes`, `no`, `on` or `off`; any other value, or a \
             number that does not parse, stops the worker with an error naming the \
             variable. Under the supervisor, the variables marked as set by it come from \
             the node config; the model runtime reads further engine variables of its own \
             (core.md).",
            WORKER_ENV,
        )
    }

    fn workspace_root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap()
    }

    #[test]
    fn every_variable_is_registered_once() {
        let mut names: Vec<_> = WORKER_ENV.iter().map(|var| var.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "a variable is registered twice");
    }

    /// Every `IZWI_*` name in this crate's production source is registered,
    /// so a new variable cannot ship undocumented.
    #[test]
    fn every_variable_in_the_source_is_registered() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let unregistered: Vec<String> = production_env_names(&source)
            .into_iter()
            .filter(|name| WORKER_ENV.iter().all(|var| var.name != name))
            .collect();
        assert!(
            unregistered.is_empty(),
            "register these in env_config.rs: {unregistered:?}"
        );
    }

    /// `docs/dev/env/serving-worker.md` is the rendered registry. Run with
    /// `UPDATE_ENV_REFERENCE=1` to rewrite it.
    #[test]
    fn reference_doc_is_current() {
        let path = workspace_root().join(REFERENCE);
        let expected = reference();
        if std::env::var_os("UPDATE_ENV_REFERENCE").is_some() {
            std::fs::write(&path, &expected).unwrap();
        }
        let actual = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            actual == expected,
            "{REFERENCE} is stale; rerun with UPDATE_ENV_REFERENCE=1"
        );
    }
}
