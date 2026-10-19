# izwi-serving-supervisor environment

<!-- Generated from the binary's environment registry; the crate's env_reference test keeps it current. -->

The variables `izwi-serving-supervisor` reads, once, at startup. Worker bearer tokens and the gateway admin key come from the variables the node config names (`bearer_token_env`, `gateway_ack.admin_key_env`).

| Variable | Default | Meaning |
|---|---|---|
| `IZWI_LOG_FORMAT` | `text` | Log line format on stderr: `text` or `json`. |
| `IZWI_SUPERVISOR_METRICS_TEXTFILE` | unset | Path of a Prometheus textfile the supervisor keeps current (for a node exporter's textfile collector); unset writes none. |

## Set for each worker

A worker starts with a cleared environment: the inherited allowlist below, the node config's `[workers.environment]` table, then these variables, which the supervisor owns and a table may not set. Their meaning is in [serving-worker.md](serving-worker.md).

- `CUDA_VISIBLE_DEVICES`
- `IZWI_BACKEND`
- `IZWI_CPU_MEMORY_BUDGET_BYTES`
- `IZWI_CUDA_DEVICE_ORDINAL`
- `IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES`
- `IZWI_CUDA_MEMORY_BUDGET_BYTES`
- `IZWI_KV_HOST_POOL_BUDGET_BYTES`
- `IZWI_METAL_DEVICE_ORDINAL`
- `IZWI_METAL_MEMORY_BUDGET_BYTES`
- `IZWI_MODELS_DIR`
- `IZWI_WORKER_ARTIFACT_REVISION`
- `IZWI_WORKER_ATTEMPT_RETENTION_SECS`
- `IZWI_WORKER_BEARER_TOKEN`
- `IZWI_WORKER_BIND`
- `IZWI_WORKER_CANCELLATION_GRACE_MS`
- `IZWI_WORKER_CPU_AFFINITY`
- `IZWI_WORKER_CPU_THREADS`
- `IZWI_WORKER_CREDENTIAL_ID`
- `IZWI_WORKER_DEPLOYMENT_ID`
- `IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES`
- `IZWI_WORKER_DRAIN_GRACE_MS`
- `IZWI_WORKER_EXECUTION_REPRESENTATION`
- `IZWI_WORKER_EXPECTED_DEVICE_ID`
- `IZWI_WORKER_EXPECTED_DEVICE_UUID`
- `IZWI_WORKER_GENERATION_FENCE_LOCK`
- `IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES`
- `IZWI_WORKER_ID`
- `IZWI_WORKER_INCARNATION_ID`
- `IZWI_WORKER_MANAGED`
- `IZWI_WORKER_MAX_ACTIVE`
- `IZWI_WORKER_MAX_CONTEXT_TOKENS`
- `IZWI_WORKER_MAX_OUTPUT_TOKENS`
- `IZWI_WORKER_MAX_REQUEST_BYTES`
- `IZWI_WORKER_MAX_RETAINED_ATTEMPTS`
- `IZWI_WORKER_MODEL`
- `IZWI_WORKER_MODEL_GENERATION`
- `IZWI_WORKER_MODEL_LOAD_LOCK`
- `IZWI_WORKER_MODEL_LOAD_SLOTS`
- `IZWI_WORKER_NODE_ID`
- `IZWI_WORKER_OWNERSHIP_LOCK`
- `IZWI_WORKER_PRECISION`
- `IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES`
- `IZWI_WORKER_STREAMING`
- `IZWI_WORKER_TASK`
- `IZWI_WORKER_TERMINATION_GRACE_MS`

Inherited from the supervisor when set:

- `PATH`
- `RUST_LOG`
- `SSL_CERT_FILE`
- `SSL_CERT_DIR`
- `TMPDIR`
- `TEMP`
- `TMP`
- `SYSTEMROOT`
- `MallocLargeCache`

On macOS a worker starts with `MallocLargeCache=0` (libmalloc's large-allocation cache off, so a model load's freed host buffers leave its footprint) unless the supervisor's own environment sets it.
