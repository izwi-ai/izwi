# izwi-server gateway role environment

<!-- Generated from the binary's environment registry; the crate's env_reference test keeps it current. -->

The gateway role (`izwi-server --role gateway`) reads these variables once, at startup. Every value is validated and a bad one stops the gateway with an error. Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, `on` or `off`.

| Variable | Default | Meaning |
|---|---|---|
| `IZWI_GATEWAY_API_KEY_REF` | `env:IZWI_GATEWAY_API_KEY` | `env:VARIABLE` naming the public inference API key (16-4096 printable bytes). |
| `IZWI_GATEWAY_METRICS_API_KEY_REF` | unset | `env:VARIABLE` naming the /metrics key; unset turns /metrics off. Must differ from the inference key. |
| `IZWI_GATEWAY_ADMIN_API_KEY_REF` | unset | `env:VARIABLE` naming the admin API key; unset turns the admin API off. Must differ from the inference key. |
| `IZWI_GATEWAY_API_PRINCIPAL_ID` | `gateway-api-key` | Principal id of the perimeter key. |
| `IZWI_GATEWAY_TENANT_ID` | `default` | Tenant of the perimeter key. |
| `IZWI_GATEWAY_MAX_CHAT_BODY_BYTES` | `1048576` | Largest chat request body (1 KiB to 32 MiB). |
| `IZWI_GATEWAY_TRUSTED_INGRESS_TLS` | `false` | TLS terminates at a trusted ingress, allowing a non-loopback plaintext bind. |
| `IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST` | unset | Path of the scoped principal-key manifest; unset leaves the durable key store untouched. |
| `IZWI_DB_PATH` | the platform data directory's izwi.sqlite3 | SQLite store for scoped principal keys (read only when a manifest is set). |
| `IZWI_DATABASE_URL` | unset | Database URL for scoped principal keys instead of IZWI_DB_PATH. |
| `IZWI_GATEWAY_FLEET_SIZE` | unset | Gateways in a partitioned fleet; each takes 1/N of every per-tenant budget. |
| `IZWI_GATEWAY_FLEET_PARTITION` | unset | This gateway's partition index (0..N); required with IZWI_GATEWAY_FLEET_SIZE. |
| `IZWI_GATEWAY_FLEET_DB_PATH` | unset | Shared coordination store (SQLite path or postgres:// URL with db-postgres) for fleet capacity claims. |
| `IZWI_GATEWAY_ID` | a random id per boot | This gateway's identity for fleet claim ownership. |
| `IZWI_GATEWAY_FLEET_CLAIM_TTL_MS` | `30000` | Lifetime of a fleet capacity claim. |
| `IZWI_GATEWAY_FLEET_CLAIM_TIMEOUT_MS` | `100` | How long a claim may wait on the coordination store. |
| `IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE` | `600` | Per-tenant request rate. |
| `IZWI_GATEWAY_TENANT_BURST_REQUESTS` | `64` | Per-tenant request burst. |
| `IZWI_GATEWAY_TENANT_TOKENS_PER_MINUTE` | unset | Per-tenant token rate; unset leaves tokens unlimited. |
| `IZWI_GATEWAY_MAX_TRACKED_TENANTS` | `16384` | Tenants the rate limiter tracks at once. |
| `IZWI_GATEWAY_TENANT_MAX_CONCURRENT` | 8 (at most the gateway's max in flight) | Concurrent requests per tenant. |
| `IZWI_GATEWAY_SHARED_APPROVALS_PATH` | unset | Absolute path of a fleet-wide worker approvals file every gateway reads. |
| `IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS` | `30000` | How often the shared approvals file is re-read (1 s to 1 h). |
| `IZWI_GATEWAY_WORKER_TLS_CA_REFS` | unset | Comma-separated `file:/absolute/path` private CA roots for worker TLS. |
| `IZWI_GATEWAY_WORKER_TLS_CLIENT_CERT_REF` | unset | `file:/absolute/path` client certificate for worker mTLS (with the key). |
| `IZWI_GATEWAY_WORKER_TLS_CLIENT_KEY_REF` | unset | `file:/absolute/path` client private key for worker mTLS (warned about when other users can read it). |
| `IZWI_DESKTOP_OWNER_PIPE` | unset | `1` when a parent process owns the gateway: stdin closing shuts it down. |
| `IZWI_OTLP_TRACES_ENDPOINT` | unset | OTLP/HTTP collector for request spans; unset exports none. |

## Flags with an environment variable

Each of these flags can also be set through its variable; the flag wins when both are given.

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `IZWI_SERVER_ROLE` | `--role` | `local` | Process role: local inference server or hardware-independent gateway |
| `IZWI_LOG_FORMAT` | `--log-format` | `text` | Log output format (`text`, `json`) |
| `IZWI_GATEWAY_WORKER_ENDPOINT` | `--worker-endpoint` | unset | Legacy one-worker form of `--gateway-worker-endpoint`: the gateway routes to this worker through the same registry as an endpoint list |
| `IZWI_GATEWAY_TOPOLOGY` | `--gateway-topology` | `standalone` | Worker-network policy (`standalone` or `fleet-one-gateway`) |
| `IZWI_GATEWAY_WORKER_ENDPOINTS` | `--gateway-worker-endpoint` | unset | Approved private worker URLs for registry-backed routing. Repeat this option (or use a comma-separated environment value) to add capacity |
| `IZWI_GATEWAY_WORKER_APPROVALS` | `--gateway-worker-approval` | unset | Statically approved worker routing entries. Repeat this option (or use a comma-separated environment value) with the standalone five-field form or v1\|URL\|NODE_ID\|WORKER_ID\|TASK\|PUBLIC_MODEL\|DEPLOYMENT_ID\| MODEL_GENERATION for fleet mode. This cannot be combined with the legacy gateway worker endpoint list |
| `IZWI_GATEWAY_WORKER_CREDENTIAL_ID` | `--worker-credential-id` | unset | Rotatable private worker credential identifier required by gateway mode |
| `IZWI_GATEWAY_WORKER_BEARER_TOKEN` | `--worker-bearer-token` | unset | Private worker bearer token required by gateway mode |
| `IZWI_GATEWAY_WORKER_INCARNATION` | `--worker-incarnation` | unset | Accepted with `--worker-endpoint` for compatibility and ignored: the registry fences every attempt to the incarnation it observed and adopts a restarted worker |
| `IZWI_GATEWAY_WORKER_DEPLOYMENT` | `--worker-deployment` | unset | Chat deployment identifier of the legacy worker endpoint forms |
| `IZWI_GATEWAY_PUBLIC_MODEL` | `--public-model` | unset | Public model served through the gateway |
| `IZWI_GATEWAY_MODEL_GENERATION` | `--worker-model-generation` | unset | Expected non-zero model generation required by gateway mode |
| `IZWI_GATEWAY_POLICY_REVISION` | `--gateway-policy-revision` | `local-policy-v1` | Policy revision attested on private worker requests |
| `IZWI_GATEWAY_MAX_IN_FLIGHT` | `--gateway-max-in-flight` | `32` | Maximum gateway requests concurrently in flight to the pinned worker |
| `IZWI_GATEWAY_WORKER_QUEUE_WAIT_MS` | `--gateway-worker-queue-wait-ms` | `250` | Maximum time the selected worker may spend establishing runtime ownership |
| `IZWI_GATEWAY_WORKER_ADMISSION_TIMEOUT_MS` | `--gateway-worker-admission-timeout-ms` | `10000` | Maximum wait for private invocation response headers and admission |
| `IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_MS` | `--gateway-worker-first-output-timeout-ms` | `60000` | Maximum wait after admission for the first useful worker output |
| `IZWI_GATEWAY_WORKER_FIRST_OUTPUT_TIMEOUT_PER_KIB_MS` | `--gateway-worker-first-output-timeout-per-kib-ms` | `1000` | Extra first-output allowance per KiB of prompt, so a long prompt's prefill (slow on CPU) is not cancelled by the short-prompt deadline |
| `IZWI_GATEWAY_WORKER_PROGRESS_IDLE_TIMEOUT_MS` | `--gateway-worker-progress-idle-timeout-ms` | `30000` | Maximum idle time between useful worker output events |
| `IZWI_GATEWAY_SHUTDOWN_BUDGET_SECS` | `--gateway-shutdown-budget-secs` | `25` | Total gateway shutdown budget after SIGTERM: the pre-stop window, then draining in-flight requests. The default fits Kubernetes' default 30 s termination grace period |
| `IZWI_GATEWAY_PRESTOP_SECS` | `--gateway-prestop-secs` | `5` | Pre-stop window at the start of the shutdown budget: readiness turns unready while requests are still served, so load balancers stop routing here before the listener closes |
| `IZWI_GATEWAY_BODY_READ_TIMEOUT_MS` | `--gateway-body-read-timeout-ms` | `10000` | Deadline for receiving a public request body. A slow upload holds a global admission permit at most this long |
| `IZWI_GATEWAY_SLOW_CONSUMER_TIMEOUT_MS` | `--gateway-slow-consumer-timeout-ms` | `5000` | Maximum wait while relaying an event to a connected slow consumer |
| `IZWI_GATEWAY_WORKER_STATUS_TTL_MS` | `--gateway-worker-status-ttl-ms` | `10000` | Receiver-clock lifetime of a worker status observation |
| `IZWI_GATEWAY_WORKER_STATUS_POLL_MS` | `--gateway-worker-status-poll-ms` | `2000` | Status refresh interval for each configured registry worker |
| `IZWI_GATEWAY_ROUTER_PREFIX_AFFINITY` | `--gateway-router-prefix-affinity` | `on` | Prefix-aware routing: prefer the worker recently routed the longest leading part of a chat prompt, within the balance threshold below |
| `IZWI_GATEWAY_ROUTER_PREFIX_BLOCK_BYTES` | `--gateway-router-prefix-block-bytes` | `256` | Bytes of message stream per hashed prefix block |
| `IZWI_GATEWAY_ROUTER_PREFIX_MAX_BLOCKS` | `--gateway-router-prefix-max-blocks` | `128` | Leading prefix blocks considered per prompt |
| `IZWI_GATEWAY_ROUTER_PREFIX_ENTRIES` | `--gateway-router-prefix-entries` | `16384` | Prefix block hashes remembered per worker (LRU) |
| `IZWI_GATEWAY_ROUTER_BALANCE_THRESHOLD` | `--gateway-router-balance-threshold` | `4` | Outstanding requests a prefix-holding worker may carry beyond the least-loaded worker and still be preferred |
| `IZWI_GATEWAY_ROUTER_KV_PRESSURE_PCT` | `--gateway-router-kv-pressure-pct` | `90` | Managed-KV utilization percent at which a worker is avoided while any eligible worker has headroom (0-100] |
| `IZWI_GATEWAY_REALTIME` | `--gateway-realtime` | `off` | Realtime relay (DS3.6): expose /v1/realtime/ws and forward izwi-realtime-v1 sessions to eligible speech_to_text workers. `off` keeps the gateway byte-identical to the pre-realtime shape |
| `IZWI_GATEWAY_REALTIME_MAX_SESSIONS` | `--gateway-realtime-max-sessions` | `64` | Bounded concurrent realtime sessions through the relay |
| `IZWI_GATEWAY_REALTIME_SESSION_BUDGET_MS` | `--gateway-realtime-session-budget-ms` | `600000` | End-to-end budget minted into each relayed worker session (ms) |
