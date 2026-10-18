# izwi-serving-worker environment

<!-- Generated from the binary's environment registry; the crate's env_reference test keeps it current. -->

The variables `izwi-serving-worker` reads, once, at startup. Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, `on` or `off`; any other value, or a number that does not parse, stops the worker with an error naming the variable. Under the supervisor, the variables marked as set by it come from the node config; the model runtime reads further engine variables of its own (core.md).

| Variable | Default | Meaning |
|---|---|---|
| `IZWI_LOG_FORMAT` | `text` | Log line format: `text` or `json`. |
| `IZWI_OTLP_TRACES_ENDPOINT` | unset | OTLP/HTTP collector for request spans; unset exports none. |
| `IZWI_WORKER_BIND` | `127.0.0.1:9470` | Listen address. A non-loopback address needs TLS or IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND. |
| `IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND` | `false` | Acknowledge plaintext on a non-loopback address (container networking). |
| `IZWI_WORKER_TLS_CERT_REF` | unset | `file:/absolute/path` of the server certificate PEM; set with the key to serve TLS. |
| `IZWI_WORKER_TLS_KEY_REF` | unset | `file:/absolute/path` of the server private key PEM (warned about when other users can read it). |
| `IZWI_WORKER_TLS_CLIENT_CA_REF` | unset | `file:/absolute/path` of a client CA PEM; set to require client certificates (mTLS). |
| `IZWI_MODELS_DIR` | the platform data directory's izwi/models | Model artifact directory. |
| `IZWI_WORKER_ID` | `local-cpu-1` | Worker identity. |
| `IZWI_WORKER_NODE_ID` | `local-node` | Node identity. |
| `IZWI_WORKER_INCARNATION_ID` | a random UUID | Process incarnation identity. |
| `IZWI_WORKER_CREDENTIAL_ID` | unset | Required. Credential id the gateway presents. |
| `IZWI_WORKER_BEARER_TOKEN` | unset | Required. Bearer token the gateway presents. |
| `IZWI_WORKER_MANAGED` | `false` | Set by the supervisor: the worker takes its node locks and requires exact memory budgets. |
| `IZWI_WORKER_TASK` | `chat` | `chat`, `speech_to_text` or `text_to_speech`. |
| `IZWI_WORKER_MODEL` | the task's default model | Model to serve: LFM2.5-1.2B-Instruct-GGUF (chat), Nemotron-3.5-ASR-Streaming-0.6B (speech_to_text), Kokoro-82M (text_to_speech). |
| `IZWI_WORKER_DEPLOYMENT_ID` | `lfm25-cpu-v1` | Deployment id advertised to the gateway. |
| `IZWI_WORKER_ARTIFACT_REVISION` | unset | Required. Model artifact revision; must match the artifact manifest. |
| `IZWI_WORKER_MODEL_GENERATION` | `1` | Model generation advertised to the gateway. |
| `IZWI_WORKER_PRECISION` | derived from the model | Precision the supervisor approved; refused when the loaded model differs. |
| `IZWI_WORKER_EXECUTION_REPRESENTATION` | derived from the model | Execution representation the supervisor approved. |
| `IZWI_WORKER_MAX_CONTEXT_TOKENS` | derived from the model | Context limit the supervisor approved; must not exceed the fitted context. |
| `IZWI_WORKER_MAX_OUTPUT_TOKENS` | derived from the model | Output-token limit the supervisor approved. |
| `IZWI_WORKER_MAX_ACTIVE` | `1` | Concurrent invocations (non-zero). |
| `IZWI_WORKER_MAX_REQUEST_BYTES` | `1048576` | Largest invocation body in bytes. |
| `IZWI_WORKER_MAX_RETAINED_ATTEMPTS` | `1024` | Finished attempts kept for status queries. |
| `IZWI_WORKER_ATTEMPT_RETENTION_SECS` | `300` | How long a finished attempt stays queryable. |
| `IZWI_WORKER_STREAMING` | `true` | Advertise and serve streaming invocations. |
| `IZWI_WORKER_DRAIN_GRACE_MS` | `30000` | Shutdown drain grace (non-zero, at most one day). |
| `IZWI_WORKER_CANCELLATION_GRACE_MS` | `10000` | Grace for in-flight work after cancellation (non-zero, at most one day). |
| `IZWI_WORKER_TERMINATION_GRACE_MS` | `5000` | How long open connections may outlast the drain and cancellation graces, or a managed worker's SIGTERM, before the worker exits (non-zero, at most one day). |
| `IZWI_WORKER_EXIT_ON_RUNTIME_FAILURE` | `true` | Exit when the runtime fails permanently so the supervisor restarts the worker. |
| `IZWI_WORKER_OWNERSHIP_LOCK` | unset | Set by the supervisor (managed workers): resource ownership lock path. |
| `IZWI_WORKER_GENERATION_FENCE_LOCK` | unset | Set by the supervisor (managed workers): generation fence lock path. |
| `IZWI_WORKER_MODEL_LOAD_LOCK` | unset | Set by the supervisor (managed workers): node model-load lock path. |
| `IZWI_WORKER_MODEL_LOAD_SLOTS` | `1` | Set by the supervisor (managed workers): concurrent model loads per node. |
| `IZWI_BACKEND` | `cpu` | `cpu`, `metal` or `cuda`. |
| `IZWI_WORKER_CPU_THREADS` | `1` | CPU thread budget (cpu backend, non-zero). |
| `IZWI_WORKER_CPU_AFFINITY` | unset | Comma-separated CPU ids (advisory on this build). |
| `IZWI_WORKER_HOST_MEMORY_LIMIT_BYTES` | `2147483648` | Host memory assignment (cpu and cuda backends). |
| `IZWI_WORKER_SHARED_MEMORY_LIMIT_BYTES` | `2147483648` | Unified memory assignment (metal backend). |
| `IZWI_WORKER_DEVICE_MEMORY_LIMIT_BYTES` | `2147483648` | Device memory assignment (cuda backend). |
| `IZWI_WORKER_EXPECTED_DEVICE_ID` | unset | Required on metal: the assigned device's registry id. |
| `IZWI_WORKER_EXPECTED_DEVICE_UUID` | unset | Required on cuda: the assigned device's UUID. |
| `IZWI_METAL_DEVICE_ORDINAL` | `0` | Process-local Metal device index. |
| `IZWI_CUDA_DEVICE_ORDINAL` | `0` | Process-local CUDA device index. |
| `IZWI_CPU_MEMORY_BUDGET_BYTES` | unset | Runtime host memory budget; must equal the host assignment (required when managed). |
| `IZWI_METAL_MEMORY_BUDGET_BYTES` | unset | Runtime Metal memory budget; must equal the unified assignment (required when managed). |
| `IZWI_CUDA_MEMORY_BUDGET_BYTES` | unset | Runtime CUDA device budget; must equal the device assignment (required when managed). |
| `IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES` | unset | Runtime CUDA host budget; must equal the host assignment (required when managed). |
| `IZWI_ENABLE_PREFIX_CACHING` | catalog auto | Prefix caching: true enables it everywhere (needs a salt), false turns it off; unset leaves it to the catalog. |
| `IZWI_PREFIX_REUSE_AUTO` | `true` | With IZWI_ENABLE_PREFIX_CACHING unset, reuse prefixes where the catalog has evidence. |
| `IZWI_MANAGED_PREFIX_CACHE_SALT` | unset | Prefix-cache namespace; required when IZWI_ENABLE_PREFIX_CACHING is true. |
| `IZWI_MAX_PREFIX_CACHE_PAGES` | sized per model at load | Prefix cache page budget. |
| `IZWI_ENABLE_CHUNKED_PREFILL` | `false` | Split long prompts into prefill chunks. |
| `IZWI_CHUNKED_PREFILL_THRESHOLD` | the engine default | Prompt length that turns chunked prefill on. |
| `IZWI_MAX_TOKENS_PER_STEP` | the engine default | Token budget of one scheduler step. |
| `IZWI_MAX_SEQUENCE_LENGTH` | the model's context | Explicit context-length cap. |
| `IZWI_KV_HOST_OFFLOAD` | on when a host pool budget is set | false turns hierarchical KV offload off even with a budget. |
| `IZWI_KV_HOST_POOL_BUDGET_BYTES` | 0 (off) | Host KV pool budget for offload. |
| `IZWI_KV_OFFLOAD_HIGH_WATERMARK` | the engine default | Device KV occupancy that starts offload. |
| `IZWI_KV_OFFLOAD_LOW_WATERMARK` | the engine default | Device KV occupancy that stops offload. |
| `IZWI_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES` | the engine default | Offload pages in flight at once. |
| `IZWI_KV_OFFLOAD_MAX_PROMOTION_PAGES` | the engine default | Pages promoted back per step. |
| `IZWI_ASR_REALTIME_MAX_SESSIONS` | 16 on CUDA, 4 on CPU and Metal | Concurrent realtime ASR sessions (also the state arena rows a load reserves). |
| `IZWI_ASR_REALTIME_MAX_LIFETIME_SECS` | 28800 (8 hours) | Seconds one realtime ASR session may live. |
| `IZWI_ASR_REALTIME_IDLE_TIMEOUT_SECS` | `90` | Seconds without audio before a realtime ASR session is reaped. |
