# Single-node production-serving operations runbook

Status: implementation-aligned deployment guide, not a production-readiness
certificate

This runbook describes only the production-serving behavior currently present
in this checkout. It covers a single trusted node with a separately started
public gateway, the supervisor executable, and one or more private workers.
The supervisor launches CPU, Apple Metal, and NVIDIA CUDA worker lanes from an
operator-declared device inventory; the evidence differs per lane (CUDA
execution has not been run). See the
[evidence matrix](#evidence-and-support-matrix) before choosing a profile.

Use the [exact-profile support matrix](PRODUCTION_SERVING_SUPPORT_MATRIX.md)
for profile approval. Node configuration
examples live in [`config/serving/examples`](../../config/serving/examples/README.md).

## Supported topology

The supported operational shape in this runbook is:

```text
TLS ingress (required before a non-loopback public bind)
        |
        v
izwi-gateway                     hardware-independent; no model/runtime/DB
        |
        | authenticated HTTP on numeric loopback
        v
one or more izwi-serving-worker processes
        ^
        |
izwi-serving-supervisor          validates, starts, fences, restarts, drains
```

- The gateway owns the public OpenAI-compatible request contract, authentication,
  policy call, request bounds, routing, and response/SSE translation. The
  `izwi-gateway` binary links no inference engine at all: no izwi-core, Candle,
  Metal or CUDA code is in it (`scripts/ci/check-serving-boundary.sh` enforces
  this), so it is smaller and runs on any host. `izwi-server --role gateway`
  takes the same flags and environment and still works; it runs the same
  gateway code inside the larger server binary. Prefer `izwi-gateway` for new
  deployments; the Docker `production-gateway` target is a slim image of it.
- The supervisor assigns each child its CPU threads/memory or accelerator
  identity/memory before process startup. It clears the child environment,
  restores only an allowlist plus the worker's validated
  `[workers.environment]` table, and injects the validated assignment. Each
  worker has its own supervision task, so one slow load or crash loop never
  stalls the others; `max_parallel_model_loads` (default 1) bounds how many
  workers load at once.
- Each worker owns exactly one loaded model, runtime, scheduler, model
  generation, and authoritative admission semaphore. One invocation never
  crosses workers or devices.
- Multiple workers may be independent replicas of one deployment or distinct
  deployments. The gateway selects from a bounded, fresh registry. A worker's
  admission result remains authoritative when gateway load observations are
  stale.
- No Kubernetes, broker, or cloud service is required. The gateway and
  synchronous chat path do not require SQLite. Single-node profiles need no
  external database; the multi-gateway fleet profile shares one coordination
  database (SQLite on one host or PostgreSQL for a server-backed fleet).
  Durable job and artifact code remains in the local server profile and is not
  made a gateway dependency.

The `izwi-serving-supervisor` executable supervises CPU, Metal, and CUDA
worker lanes, requiring an operator-declared device inventory and
allocatable host-memory ceiling: `--cpu-worker-binary` with `--cpu-ids`,
`--metal-worker-binary` with `--metal-devices device_id@index[,...]`, and
`--cuda-worker-binary` with `--cuda-devices uuid@host_index@total_memory_bytes[,...]`.
The supervisor never probes or initializes an accelerator; each worker
verifies its assigned device identity at startup. Never change an explicit
Metal or CUDA assignment to CPU merely to make validation pass.

## Security boundary

### Public gateway credential

Gateway API authentication has no secret-valued CLI option. It resolves one API
key through an environment reference:

| Variable | Meaning | Default |
|---|---|---|
| `IZWI_GATEWAY_API_KEY_REF` | Bounded `env:VARIABLE` reference | `env:IZWI_GATEWAY_API_KEY` |
| referenced variable | 16-4096 byte printable bearer key | required |
| `IZWI_GATEWAY_API_PRINCIPAL_ID` | Server-authored principal | `gateway-api-key` |
| `IZWI_GATEWAY_TENANT_ID` | Server-authored tenant | `default` |
| `IZWI_GATEWAY_TENANT_MAX_CONCURRENT` | Accepted or acceptance-uncertain work owned per server-authored tenant | min(8, `--gateway-max-in-flight`) |
| `IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE` | Per-tenant request starts per minute (token-bucket refill), 1–1,000,000 | `600` |
| `IZWI_GATEWAY_TENANT_BURST_REQUESTS` | Per-tenant burst bucket size, 1–100,000 | `64` |
| `IZWI_GATEWAY_MAX_TRACKED_TENANTS` | Bound on tenants with live rate state, 1–65,536 | `16384` |

The server assigns the `inference` role/scope. Public `x-principal-id`,
`x-tenant-id`, `x-scopes`, and `x-roles` headers are not authority. The
enterprise inference policy hook runs after authentication and fails closed on
an error.

Keep secrets in the service manager or secret store that creates the process
environment. Do not put them in node TOML, command-line arguments, shell
history, logs, or health-check URLs.

### Scoped per-principal keys (DS0.5)

Beyond the shared root key, the gateway accepts individually scoped API keys.
Every scoped key carries a server-authored principal id, a role set
(`inference`/`admin`/`metrics`), and an optional tenant scope. Rate quotas and
tenant concurrency leases derive from the authenticated principal's tenant
scope, so two tenants never share a bucket even when they share a gateway.

| Variable | Meaning | Default |
|---|---|---|
| `IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST` | Bounded path to a JSON principals manifest; unset keeps the durable store untouched | unset |

Manifest shape (all fields are validated fail-closed; unknown fields are
rejected; `key_ref` must be a bounded `env:VARIABLE` or absolute `file:/PATH` reference —
inline key material is rejected):

```json
{
  "version": 1,
  "principals": [
    {
      "principal_id": "svc-alpha",
      "roles": ["inference"],
      "tenant_id": "tenant-alpha",
      "key_ref": "env:ALPHA_GATEWAY_KEY"
    }
  ]
}
```

Semantics:

- The shared `IZWI_GATEWAY_API_KEY` remains valid as the bootstrap root
  principal and is unchanged. Scoped key material must never reuse any
  perimeter credential or another scoped key.
- Roles gate routes: `inference` for `/v1/chat/completions`, `metrics` for
  `/internal/metrics`, `admin` for `/internal/admin/drain`. A principal
  without the `inference` role is rejected with 403 on inference routes.
- Only salted HMAC-SHA256 digests are persisted, in the
  `gateway_principal_keys` table of the durable store. Key material lives
  exclusively in the referenced environment variables or files.
- Bootstrap is idempotent: each boot re-provisions manifest entries (rotation
  = change the reference, restart) and leaves unlisted store entries intact.
  Revocation = delete the row (`DELETE FROM gateway_principal_keys WHERE
  principal_id = ...`) and restart.

### Private worker credential

Each node worker entry contains a `credential_id` and a `bearer_token_env`
*name*, never the token. The supervisor resolves that variable, injects the
token into the child, and redacts it from validate-only diagnostics. The gateway
currently has one private credential pair for all configured workers:

- `IZWI_GATEWAY_WORKER_CREDENTIAL_ID`
- `IZWI_GATEWAY_WORKER_BEARER_TOKEN`

Consequently, every worker selected by one gateway must currently accept that
same pair. Configure the same credential ID and secret value across replica
entries, or use separate gateways. Per-worker gateway credentials are not
implemented. One optional client identity can be configured for all HTTPS
worker links made by a gateway.

### Binds and TLS

- `--gateway-topology standalone` is the default and accepts only numeric-
  loopback HTTP worker endpoints. `--gateway-topology fleet-one-gateway`
  requires the gateway (`izwi-gateway`, or `izwi-server --role gateway`), HTTPS,
  a configured client certificate/private key, and versioned worker approvals
  that pin both node and worker identity.
- Supervised workers bind numeric loopback only: node config validation
  rejects any other bind. Plaintext worker client URLs must use a numeric
  loopback host such as `127.0.0.1` or `::1`; `localhost` is deliberately not
  accepted.
- The worker client accepts certificate-verified HTTPS endpoints, augments the
  platform trust store with up to 16 private CA PEM files, and can present one
  client certificate/key identity. Configure bounded absolute `file:`
  references with `IZWI_GATEWAY_WORKER_TLS_CA_REFS` (comma-separated),
  `IZWI_GATEWAY_WORKER_TLS_CLIENT_CERT_REF`, and
  `IZWI_GATEWAY_WORKER_TLS_CLIENT_KEY_REF`. Certificate and key references must
  be supplied together. Each PEM file is limited to 256 KiB and all private CA
  files together to 1 MiB. TLS material on a plaintext endpoint is rejected.
- A worker started outside the supervisor terminates TLS itself when
  `IZWI_WORKER_TLS_CERT_REF` and `IZWI_WORKER_TLS_KEY_REF` name bounded
  absolute `file:` PEM paths (256 KiB each), and requires client certificates
  (mTLS) when `IZWI_WORKER_TLS_CLIENT_CA_REF` is also set. Without TLS it
  refuses a non-loopback bind unless `IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND=1`
  acknowledges a private network such as a container bridge; the gateway's
  `standalone` topology still reaches workers only over numeric-loopback HTTP,
  and `fleet-one-gateway` only over HTTPS. This is not a supported
  multi-machine profile until a deployment has separate-machine transport
  evidence.
- The gateway itself does not terminate TLS. Bind it to loopback behind a TLS
  ingress. A non-loopback gateway bind is rejected unless
  `IZWI_GATEWAY_TRUSTED_INGRESS_TLS=1` explicitly acknowledges that trusted
  ingress. That variable does not enable encryption.
- Leave CORS disabled unless browser access is required. If enabled, configure
  an explicit `IZWI_CORS_ORIGINS` allowlist; wildcard or empty gateway CORS is
  rejected.
- Private endpoints are bearer-authenticated but are not administrative APIs.
  Do not expose them directly to an untrusted network.

## Prepare one CPU node

Start from the packaged
[`izwi-serving-node.example.toml`](../../config/serving/izwi-serving-node.example.toml)
or the development
[`one-device-cpu.toml`](../../config/serving/examples/one-device-cpu.toml).
Both use schema version 2 and are templates, not certified resource profiles.

Before validation:

1. Replace every absolute working, runtime, model, and binary path.
2. Pin the exact artifact revision and use a nonzero model generation.
3. Set `host_memory_budget_bytes`, worker host-memory limits, CPU thread budgets,
   and optional CPU affinity from the actual service allocation. Empty affinity
   is allowed; affinity enforcement is advisory in the current worker build.
4. Keep each worker on a distinct numeric-loopback port. The schema permits at
   most 64 workers per node and rejects duplicate endpoints or exclusive
   devices and aggregate CPU/host-memory overcommit.
5. Keep each worker deployment contract within what the worker serves. Chat
   workers serve `LFM2.5-1.2B-Instruct-GGUF`, `Qwen3.8-27B-FP8`,
   `Qwen3-30B-A3B-GGUF`, and `Qwen3.6-35B-A3B-FP8`; `speech_to_text` workers
   serve the Nemotron ASR and Voxtral realtime families; `text_to_speech`
   workers serve the streaming TTS families. A chat deployment must declare
   `max_context_tokens` and `max_output_tokens` (output within context). The
   supervisor passes the declared precision, execution representation and
   token limits to the worker (`IZWI_WORKER_PRECISION`,
   `IZWI_WORKER_EXECUTION_REPRESENTATION`, `IZWI_WORKER_MAX_CONTEXT_TOKENS`,
   `IZWI_WORKER_MAX_OUTPUT_TOKENS`); the worker advertises exactly those once
   the loaded model proves them and refuses to start otherwise, so its status
   equals the approval.
6. Set every `bearer_token_env` in the supervisor's environment. Use the same
   private credential ID and secret for all workers routed by one gateway.

The worker verifies its artifact manifest before constructing the runtime. It
then initializes the exact assigned backend without fallback, loads and warms
the model, and only then binds its private listener.

### Worker process environment

The supervisor sets each worker's identity, assignment and deployment
variables itself, including the declared `IZWI_WORKER_PRECISION`,
`IZWI_WORKER_EXECUTION_REPRESENTATION`, `IZWI_WORKER_MAX_CONTEXT_TOKENS` and
`IZWI_WORKER_MAX_OUTPUT_TOKENS`. It clears everything else except a short
allowlist (`PATH`, `RUST_LOG`, `SSL_CERT_FILE`, `SSL_CERT_DIR`, temp
directories) and the worker's own `[workers.environment]` table:

```toml
[[workers]]
worker_id = "chat-cuda-1"
# ...
[workers.environment]
IZWI_KV_HOST_OFFLOAD = "0"                      # any IZWI_* switch
IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS = "60000"
LD_LIBRARY_PATH = "/usr/local/cuda/lib64"      # loader and cache paths
```

The table accepts `IZWI_*` names and `LD_LIBRARY_PATH`, `CUDA_HOME`,
`CUDA_PATH`, `HF_HOME`, `HF_HUB_OFFLINE` and `RUST_BACKTRACE`. Validation
rejects every variable the supervisor sets itself (identity, assignment,
budgets, locks, `CUDA_VISIBLE_DEVICES`), anything that looks like a
credential (`TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `API_KEY`, `*_KEY`)
and the worker's `bearer_token_env`, and bounds the table (64 entries, 4 KiB
per value, 64 KiB in all). The supervisor's own variables are set last, so
they always win. The variables below can be set this way:

| Variable | Meaning | Default |
|---|---|---|
| `IZWI_WORKER_PRECISION`, `IZWI_WORKER_EXECUTION_REPRESENTATION` | Declared execution profile; the worker refuses to start if the loaded model does not match. Unset: derived from the served model | derived |
| `IZWI_WORKER_MAX_CONTEXT_TOKENS`, `IZWI_WORKER_MAX_OUTPUT_TOKENS` | Declared chat token limits; a context above the runtime's fitted context fails startup. Unset: the fitted context, and output up to it | derived |
| `IZWI_WORKER_EXIT_ON_RUNTIME_FAILURE` | Exit with status 70 once the runtime fails permanently (a panic after native entry or a latched fatal device fault), so a supervisor restarts it | `true` |
| `IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS` | How long streamed text may wait on a stalled reader before the invocation fails; while the reader lags, deltas are merged rather than dropped. Also read by the local server | `30000` |

A worker's `/internal/v1/status` reflects runtime health: a permanent failure
reports `process_state` and readiness `Failed`, and a drain-recoverable fault
keeps the process `Running` with readiness `Failed`. The gateway and the
supervisor's readiness wait only route to `Running` and `Ready`.

## Validate without starting workers

Run the installed supervisor with all required explicit inputs:

```sh
izwi-serving-supervisor \
  --config /absolute/path/to/node.toml \
  --cpu-worker-binary /absolute/path/to/izwi-serving-worker \
  --cpu-ids 0,1 \
  --allocatable-host-memory-bytes 4294967296 \
  --validate-only
```

The referenced worker secret variables must already exist in this process
environment. Validate-only performs the same bounded TOML, directory,
executable, CPU inventory, resource, deployment, capability, and credential
validation used before launch. Successful output is redacted and capped at 16
KiB. It exits before inheriting the allowlisted environment, opening lock files,
launching a child, loading a model, or initializing an accelerator.

Metal and CUDA nodes validate the same way with their worker binary and
device declarations (`--metal-worker-binary` plus `--metal-devices`,
`--cuda-worker-binary` plus `--cuda-devices`). A device that is not declared,
or a worker binary that does not support the lane, fails validation; that is
fail-closed behavior, not a request to edit the backend to CPU.

## Start and verify

### 1. Start the supervisor

Remove `--validate-only` from the validated command and run it under the node's
service manager. Keep its standard input/control pipe intact; the supervisor
uses it to prove child ownership.

The supervisor:

1. acquires the single-supervisor and prior-generation fences;
2. creates a new worker incarnation for each launch;
3. launches each child with a cleared, bounded environment and exact resource
   assignment;
4. polls authenticated descriptor and status endpoints;
5. accepts readiness only when worker/node/incarnation, assignment, deployment,
   artifact revision, model generation, task, execution profile, capability,
   and capacity match the validated node contract; and
6. monitors exits with bounded exponential backoff, jitter, stable-uptime reset,
   and restart-window quarantine.

Gateway startup reads every approved endpoint once. Approvals that cannot work
at all (an empty or duplicated endpoint, an endpoint the topology forbids, two
endpoints answering as one logical worker) fail startup. An approved worker
that is unavailable or does not yet match its approval does not: the gateway
starts without it, logs a warning, and retries the adoption every status-poll
interval, so one down replica never takes a restarting gateway down. `/readyz`
stays 503 until at least one fresh compatible worker is adopted.

### 2. Start the gateway

Prefer typed approvals over the legacy `--worker-endpoint` form. For one CPU worker,
the process shape is:

```sh
izwi-gateway \
  --gateway-topology standalone \
  --host 127.0.0.1 \
  --port 8080 \
  --public-model LFM2.5-1.2B-Instruct-GGUF \
  --gateway-worker-approval 'http://127.0.0.1:9470|chat|LFM2.5-1.2B-Instruct-GGUF|lfm25-cpu-v1|1' \
  --gateway-max-in-flight 32 \
  --gateway-worker-queue-wait-ms 250 \
  --gateway-worker-admission-timeout-ms 10000 \
  --gateway-worker-first-output-timeout-ms 60000 \
  --gateway-worker-progress-idle-timeout-ms 30000 \
  --gateway-slow-consumer-timeout-ms 5000 \
  --gateway-worker-status-ttl-ms 10000 \
  --gateway-worker-status-poll-ms 2000
```

Supply the public and private credential environment described above.
`izwi-server --role gateway` accepts exactly the same flags. The gateway reads
no configuration file: its listener takes `--host`/`--port` (or `IZWI_HOST`,
`IZWI_PORT`), `IZWI_TIMEOUT` and `IZWI_CORS`/`IZWI_CORS_ORIGINS`, and every
gateway variable is listed in [`env/gateway.md`](env/gateway.md). Its log
targets are `izwi_gateway` (the default filter logs warnings), so debug a
gateway with `RUST_LOG=izwi_gateway=debug`, not `izwi_server`. It logs, and
exports spans, as service `izwi-gateway` (`izwi-server` when the server runs
the role). The standalone numeric-loopback compatibility syntax is:

```text
URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION
```

The fleet-one-gateway syntax is versioned and pins operator-approved logical
identities rather than trusting the first descriptor returned by an endpoint:

```text
v1|HTTPS_URL|NODE_ID|WORKER_ID|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION
```

Fleet mode rejects the legacy `--worker-endpoint` form, legacy repeated
`--gateway-worker-endpoint` values, five-field approvals, plaintext endpoints,
and missing client identity before descriptor/status network I/O. Bearer
service authentication remains required in addition to mTLS. This policy does
not by itself make a remote worker a supported deployment: the worker
terminates TLS/mTLS itself (see "Binds and TLS"), but no separate-machine
handshake has been exercised, so retain real separate-machine handshake
evidence before relying on it.
Configurations that previously used remote HTTPS under the implicit local
policy must migrate explicitly to `fleet-one-gateway`, v1 approvals, and client
certificate/key references; there is no legacy remote-HTTPS compatibility mode.

Repeat `--gateway-worker-approval` for replicas. A pool is keyed by task and
public model. Approvals may name several chat models: the gateway serves each
one (`--public-model` stays the primary model, listed first, and names the
realtime stages and the legacy endpoint forms' model), lists them all on
`/v1/models`, and routes each request by its `model`; an unserved model is a
404. `/readyz` stays ready while any served model has a ready worker, and its
check message names the models without one. A chat model added to the shared
approvals file after boot needs a gateway restart. All replicas in a pool must agree on deployment ID, generation,
artifact revision, execution representation, tokenizer revision, and exact
capability; backend and precision may differ and remain worker properties. The
CLI pins task/model/deployment/generation, while the gateway freezes the rest of
the authenticated initial status contract. The node TOML is therefore the
operator's artifact-revision source of truth for this single-node profile.

Legacy `--worker-endpoint` is the one-worker form of
`--gateway-worker-endpoint`: the gateway routes to it through the same
registry, so it polls its status, fences every attempt to the incarnation it
observed, and adopts a restarted worker. `--worker-incarnation` is still
accepted with it and ignored. Keep it only for standalone compatibility.
Legacy `--gateway-worker-endpoint` entries (and `--worker-endpoint`) use one
chat deployment/generation supplied separately. Do not mix legacy endpoints
with typed approvals.

The registry supports at most 256 configured workers; deployment tables support
at most 64 pools and 256 replicas per pool. Poll interval must be nonzero and
less than status TTL. The gateway uses receiver-monotonic observation time and
strictly increasing status sequence numbers rather than trusting remote clocks.
Admission/header timeout is configurable from 1 ms to 60 seconds, first-output
and inter-output idle timeouts from 1 ms to one hour, and slow-consumer relay
timeout from 1 ms to 60 seconds. Queue wait cannot exceed admission timeout.
The first-output deadline grows with the prompt:
`--gateway-worker-first-output-timeout-per-kib-ms` (default 1000) is added for
every KiB of invocation, so a long prompt's CPU prefill is not cancelled by the
deadline sized for short prompts. Every phase remains capped by the original
request's total remaining deadline; transport or metadata trickles do not
extend it.

Routing among a pool's eligible workers:

- `--gateway-router-kv-pressure-pct` (default 90): a worker whose managed-KV
  usage is at or above this is chosen only when every eligible worker is.
- `--gateway-router-prefix-affinity` (default `on`): the gateway hashes each
  prompt's message stream in `--gateway-router-prefix-block-bytes` (256)
  blocks, up to `--gateway-router-prefix-max-blocks` (128), and remembers per
  worker the last `--gateway-router-prefix-entries` (16,384) block hashes it
  routed there. The worker holding the longest prefix of the prompt wins
  unless it carries more than `--gateway-router-balance-threshold` (4)
  outstanding requests beyond the least-loaded worker; then load decides.
  Only hashes are kept. A wrong guess costs one prefill, never a request.
- Otherwise the capacity-weighted least-loaded worker wins; equal load
  prefers the higher observed output rate.

The earlier `--gateway-router-cache-*` and `--gateway-session-pin*` flags are
removed; delete them from service definitions.

HTTPS workers serve HTTP/2 (negotiated by ALPN), so a gateway multiplexes its
invocations and status polls over one connection per worker. Plaintext
loopback workers stay on HTTP/1.1 keep-alive.

### 3. Check probes and one public request

`GET /livez`, `GET /readyz`, `/openapi.json`, and `/docs` are intentionally
public. `/readyz` returns 200 only when the gateway lifecycle is Ready, it is not
draining, and a receiver-fresh compatible worker supports both JSON and SSE
chat. Exhausted capacity is an admission condition and does not by itself make
the service unready.

All `/v1` requests require:

```text
Authorization: Bearer <public gateway API key>
```

Send a bounded text-only `POST /v1/chat/completions` first without streaming,
then with the existing OpenAI-compatible SSE form. Preserve and inspect the
returned `x-request-id`. Do not use an inference request as a liveness probe.

### 4. Scrape bounded gateway metrics

Metrics are absent (404) unless `IZWI_GATEWAY_METRICS_API_KEY_REF` names a
bounded `env:VARIABLE` secret. That secret must differ from the public inference
key. With it configured, authenticate separately and scrape either
`GET /internal/metrics` or `GET /internal/metrics/prometheus`.

The response is capped at 256 KiB. Every label comes from a closed vocabulary:
the parsed model name (a known `ModelVariant`, never request text), the
gateway's own route paths, HTTP status codes and a fixed outcome set. A labelled
family keeps at most 16 series; further label sets share an `other` series.
There are no request IDs, tenant IDs, prompts or worker-controlled labels.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `izwi_gateway_chat_request_duration_seconds` | histogram | `model`, `stream`, `outcome` (`ok`/`error`) | Gateway ingress to the last byte of the answer. `_count` is requests. A stream dropped before its terminal event is an `error`. |
| `izwi_gateway_chat_ttft_seconds` | histogram | `model` | Streaming: ingress to the first answer text sent. |
| `izwi_gateway_chat_inter_token_latency_seconds` | histogram | `model` | Gap between successive text deltas of one stream (a worker may coalesce several tokens into one delta). |
| `izwi_gateway_chat_prompt_tokens_total`, `izwi_gateway_chat_completion_tokens_total` | counter | `model` | Usage of completed requests. |
| `izwi_gateway_worker_dispatch_seconds` | histogram | `outcome` (`accepted` or the HTTP status answered) | Worker call latency; streams stop at acceptance. |
| `izwi_gateway_http_request_duration_seconds` | histogram | `route`, `status` | Time to response headers. Unknown paths share `route="other"`. |
| `izwi_gateway_active_requests`, `izwi_gateway_active_streams`, `izwi_gateway_active_tenant_work` | gauge | none | Admitted work in flight. |
| `izwi_gateway_*_rejections_total`, `izwi_gateway_routing_dispatch_failures_total`, `izwi_gateway_dispatch_calls_total`, `izwi_gateway_dispatch_latency_microseconds_total`, `izwi_gateway_http_responses_{2xx,3xx,4xx,5xx,other}_total` | counter | none | Unchanged unlabelled counters. |

The same private service credential used for descriptor, status, invocation,
query, and cancellation authorizes a worker scrape at
`GET /internal/v1/metrics/prometheus`. Its response is capped at 8 KiB and uses
only fixed unlabeled process counters/gauges, including active and retained
attempts, admissions/rejections, terminal outcomes, cancellations, and their
accumulated timings. Keep this endpoint on the same trusted private boundary;
it is not a public monitoring API.

On Unix, send `SIGUSR1` to the supervisor process to print one redacted status
snapshot to stderr. The snapshot is capped at 16 KiB and includes bounded
supervisor lifecycle totals plus configured worker identity, state, process,
incarnation, assignment, deployment, and generation. Repeated signals are
coalesced through a one-entry channel. This local signal surface is for service
manager diagnostics; it does not create an unauthenticated network listener.

Set `IZWI_SUPERVISOR_METRICS_TEXTFILE=/path/supervisor.prom` to have the
supervisor keep a Prometheus textfile for a node exporter's textfile collector.
It is replaced atomically whenever it changes and holds the gauges
`izwi_supervisor_workers`, `_workers_running`, `_workers_quarantined` and
`_workers_restart_pending`, and the counters
`izwi_supervisor_{launch_attempts,readiness_failures,restarts_scheduled,quarantines,unexpected_exits,liveness_restarts,stop_failures,autoscale_ups,autoscale_downs,rollouts_committed,rollouts_aborted,rollouts_degraded}_total`.
A rollout runs before the supervision loop, so rollout counters appear once it
has finished.

### 5. Logs and traces

- The gateway and the local server log text by default; `--log-format json`
  (or `IZWI_LOG_FORMAT=json`) writes one JSON object per line with the event
  fields at the top level, plus `span` and `spans` carrying `correlation_id`,
  `trace_id`, `span_id` and `parent_span_id`. The worker and the supervisor
  read the same `IZWI_LOG_FORMAT` variable.
- Trace context: a valid W3C `traceparent` from the caller is continued; any
  other request starts a trace. The gateway sends its own span as the parent on
  every worker invocation (`traceparent` header) and realtime dial (WebSocket
  upgrade header), and the worker continues it, so one `trace_id` finds the
  request in the gateway's and the worker's logs.
- The supervisor relays each worker's stdout and stderr line by line: JSON
  lines gain a leading `"worker_id"` field, other lines a `<worker_id> | `
  prefix.
- Span export is off by default. Set `IZWI_OTLP_TRACES_ENDPOINT` (for an
  OpenTelemetry Collector, `http://collector:4318/v1/traces`) on the gateway,
  server or worker to export request spans (`http_request`, `worker_request`)
  as OTLP/HTTP JSON. Spans queue in a bounded buffer and are dropped, never
  waited on, when the collector falls behind.
- Request audit and observability hooks run after the response body ends, on
  their own task. A streamed answer is audited with how it ended: a 200 SSE
  stream that reported an error in-band is a `Failure`, and the audit metadata
  `body` is `completed`, `failed` or `interrupted` (client disconnect).

## Admission, retry, timeout, and cancellation semantics

- Gateway admission is fail-fast and bounded by `--gateway-max-in-flight`; it
  does not create an unbounded waiting queue. The lifecycle is checked both
  before and after taking a permit so drain cannot admit a racing request.
- Tenant concurrent-work admission is separately fail-fast and defaults to
  `min(8, --gateway-max-in-flight)`, configurable with
  `IZWI_GATEWAY_TENANT_MAX_CONCURRENT` but never above global ownership. The
  tenant is derived from authenticated server state, not caller identity
  headers. A tenant-limit rejection is 429; exhausted or unavailable global
  ownership state is 503.
- Tenant request starts are also rate-limited per gateway process: a token
  bucket of `IZWI_GATEWAY_TENANT_BURST_REQUESTS` (default 64) refilled at
  `IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE` (default 600), with rate state
  for at most `IZWI_GATEWAY_MAX_TRACKED_TENANTS` (default 16,384) tenants. An
  over-rate request is 429. The limits are per process: with several
  gateways, use the fleet partition (below) or the totals multiply.
- `IZWI_GATEWAY_TENANT_TOKENS_PER_MINUTE` (default 0: off) adds a per-tenant
  token budget (prompt plus completion tokens, one minute's worth as burst).
  A request reserves its output budget plus prompt bytes / 4 at admission
  and is settled to the worker's reported usage when it completes; a request
  that fails before streaming is refunded, and one that started streaming
  keeps its estimate. The fleet partition divides it like the request rate.
- The tenant checks run before the global permit: a tenant over its request
  rate or concurrent-work limit is refused before it takes one of the
  gateway's `--gateway-max-in-flight` permits or sends its body, so one
  tenant's slow uploads cannot exhaust admission for others. The body must
  arrive within `--gateway-body-read-timeout-ms` (default 10000, at most
  60000); a late body is 408 and is counted in
  `izwi_gateway_body_timeout_rejections_total`. The rate token itself is
  spent only after the body validates, and the local limits run before the
  enterprise quota hook, so a request they refuse is never charged by it.
- Chat responses carry `x-ratelimit-{limit,remaining,reset}-requests` (and
  `-tokens` with a token budget); 429 and 503 answers carry `Retry-After`.
- An enterprise `QuotaLimiter` receives `commit(QuotaUsage)` after each
  completed chat with the admitted estimate and the actual input and output
  tokens, for metering. It is best effort (2 s, logged on failure); the
  default implementation does nothing.
- Once an exact worker attempt is bound, tenant ownership survives a response
  timeout, dropped JSON response, or SSE disconnect. It is released only by an
  authenticated terminal worker event, a cancellation disposition that proves
  teardown, or an exact-attempt query that proves execution stopped. The
  reconciler retries idempotent exact cancellation with bounded backoff; stale
  status and an expired public deadline are not teardown proof.
- Registry capacity is advisory. Selection uses only an approved, fresh,
  running, capability-compatible worker and subtracts bounded local dispatch
  reservations, but the selected worker atomically accepts or rejects the exact
  attempt.
- One alternate worker may be selected only before acceptance is proven. Safe
  cases are a local client-permit deadline, a connection that was provably never
  established, or a valid `accepted: false` transient rejection for capacity,
  queue wait, wrong incarnation, wrong generation, unknown deployment, model
  not ready, or worker draining. The alternate excludes the first incarnation,
  keeps the logical request ID, creates a new attempt ID, and receives only the
  remaining original deadline. There is no third attempt.
- No alternate is used after a response-header timeout, generic/malformed HTTP
  response, protocol or NDJSON error, stream-progress/total timeout, EOF without
  a terminal event, acceptance, or any streamed output. Those cases may have
  executed and are never replayed.
- A valid worker rejection proves the worker was reachable. Transport/unknown
  failures count toward an incarnation-fenced circuit. The default opens after
  three strikes for 30 seconds; cooldown plus a newer authenticated status is
  required for one half-open probe. A replacement incarnation starts with a
  fresh circuit.
- On disconnect, timeout, malformed accepted stream, or dropped response, the
  client sends best-effort cancellation to the exact worker/incarnation/model
  attempt. Cancellation is cooperative. A requested cancellation, dropped HTTP
  future, missing attempt, or expired retention record does not prove execution
  stopped.
- The worker retains its authoritative execution permit until its executor
  reports completed, cancelled, or otherwise confirmed teardown. It suppresses
  further output after cancellation and treats EOF without a terminal event as
  interrupted/unknown.

Clients must treat an interrupted accepted stream as an unknown result, not as
permission to resubmit. If their application needs idempotency, it must use a
documented route-level contract; the migrated chat route does not promise a
durable result ledger.

## Bounded state and data paths

The important current bounds are:

| Layer | Bound |
|---|---|
| Gateway chat body | 1 MiB default; `IZWI_GATEWAY_MAX_CHAT_BODY_BYTES` permits 1 KiB-32 MiB |
| Gateway headers | 128 fields, 32 KiB total, 8 KiB per value |
| Gateway request ID | one safe-ASCII value, at most 128 bytes |
| Gateway admissions | configured semaphore, fail-fast |
| Gateway tenant work | process-local per-tenant and total ownership, fail-fast; hard cap 100,000 and configured total no greater than `--gateway-max-in-flight` |
| Configured registry | 256 workers; bounded deployment/local-dispatch tables |
| Worker request body | node `max_request_bytes`, at most 64 MiB by schema |
| Worker execution | `max_active_invocations`, atomic and fail-fast; no transport queue |
| Worker event channel | 4 events; 1 MiB encoded event limit in the real worker |
| Worker attempt records | configured `max_retained_attempts`, at least active capacity, max 65,536; time-bounded by `attempt_retention_secs` (1-86,400) |
| Private client JSON | 1 MiB request, 512 KiB control body, 16 KiB error body defaults |
| Gateway private deadlines | 10 s admission/header, 60 s first output, 30 s inter-output idle; each is capped by the original 300 s default total invocation deadline |
| Gateway slow consumer | 5 s relay wait before exact-attempt cancellation; configurable from 1 ms to 60 s |
| Private NDJSON | Gateway chat: 1 MiB line, 16 MiB total, 8,192 events; generic client defaults remain independently bounded |
| Remote chat output | gateway ceiling 4,096 tokens/512 KiB, further restricted by the worker capability |

Backpressure is cancellation-safe: bounded channel saturation or a slow/dropped
consumer requests cancellation instead of retaining unbounded events. The
worker still owns capacity until executor teardown.

The local durable batch store is not loaded in gateway mode. In the local
profile its maintenance defaults to 64 records and clamps each recovery or
reconciliation pass to 512, with deterministic ordering and one shared repair
budget. SQLite reopen/claim/attempt-token completion has focused restart proof.
This does not establish multi-gateway database ownership. The opaque artifact
facade similarly has bounded, integrity-checked reads and tombstone-first
deletion, and its retention labels drive expiry (see below), but no gateway
artifact route or fleet adapter is enabled.

Opaque artifact creation requires reserved-write protocol v1 from the selected
media provider. Its database reservation exists before bytes leave the process,
expires after one minute, and is atomically consumed by metadata publication.
Provider calls are bounded to thirty seconds. Maintenance uses exact cleanup
claim tokens and recovers by write ID, including reservations with no recorded
provider key. `NotFound` means both absent and unable to commit later for that
expired write ID. The reservation ledger is capped per tenant and in total
(see below) and each cleanup scan at 64. New durable Fish PCM replay chunks use tenant-scoped opaque artifact
IDs and an atomic attempt-publication transaction; provider keys remain private.
Existing raw-key replay rows stay compatible. Durable Fish final WAV publication
can be enabled with `IZWI_TTS_OPAQUE_FINAL_WAV_ENABLED=1` after the reserved-file
provider and atomic settlement checks pass; the default remains legacy for
rollout safety. Saved voices, other history-audio producers, references, other
media, and gateway audio transport retain their existing provider path until
explicitly migrated. Speech-history storage can consume an exact tenant-scoped
opaque reference and exposes it through the existing bounded streaming
response; it rejects incomplete, ambiguous, wrong-tenant, metadata-mismatched,
or multiply-owned references. Opaque deletion atomically removes the history
row, tombstones the media row, and retains its bounded cleanup intent; provider
failure leaves that intent for retry. The final settlement transaction binds
the opaque media, exact attempt output, Ready projection, checkpoint, stage,
job, and reservation in one commit. This does not enable a gateway audio route
or migrate the remaining local producers.

Reserved-write protocol v1 exposes reserved file publication as a separate
capability. The built-in local provider streams a finalized file with a fixed
64 KiB buffer, validates its declared length and SHA-256, and publishes under
the existing per-write lock and expiry fence. Do not advertise final speech
artifacts through a bytes-only provider; Izwi does not fall back to legacy
`put_file` for an opaque reserved-file request.

The local reserved-write provider uses the shared image/video/audio MIME-to-file
extension mapping because reads reconstruct MIME metadata from the object key.
Canonical mapped types (for example `image/png`, `video/mp4`, `audio/wav`,
`audio/mpeg`, and `audio/pcm-f32le`) round-trip. Aliases such as `audio/x-wav` are rejected before
publication rather than being stored under `.wav` and later misreported as
`audio/wav`; callers should submit the canonical media type.

### Durable state retention, quotas and idempotency

Durable creates (`POST /v1/text-to-speech` without streaming, including
reference and saved voices, and `POST /v1/speech-to-text/jobs`) accept an
`Idempotency-Key` header. The key is scoped to the authenticated tenant and
reserved before any work. The job, its inputs, its stages and the replay
result commit in one transaction. A retry with the same body returns the
original record (202). The same key with a different body, or while the
first request is still being accepted, is a 409. A streaming request with a
key is a 400. Results are replayable for 24 hours.

Idempotency records, provider write reservations and pending artifact
cleanup intents are counted per tenant and in total by counters kept in the
same transactions as the records; a full tenant is refused (503 for
idempotency) without affecting others. Cleanup intents are capped at 16,384
per tenant and 65,536 in total; a delete past the cap fails before it hides
the artifact.

| Setting | Default | Meaning |
|---|---|---|
| `IZWI_IDEMPOTENCY_MAX_RECORDS_PER_TENANT` | 65,536 | Live idempotency records (reserved or replayable) per tenant |
| `IZWI_IDEMPOTENCY_MAX_RECORDS` | 1,048,576 | Live idempotency records in total |
| `IZWI_PROVIDER_WRITE_MAX_RESERVATIONS_PER_TENANT` | 16,384 | Outstanding provider write reservations per tenant |
| `IZWI_PROVIDER_WRITE_MAX_RESERVATIONS` | 65,536 | Outstanding provider write reservations in total |

The local batch worker's maintenance pass (every 30 s, at most 64 rows of each
kind per pass, each kind in its own short transaction) deletes state past its
retention:

| Setting | Default | What expires |
|---|---|---|
| `IZWI_JOB_RETENTION_SECONDS` | 2,592,000 (30 days) | Terminal jobs with their stages, artifact rows and text inputs, measured from the job's last update. Opaque objects labelled `job` expire with their job. A job that still owns live job- or ephemeral-labelled objects waits until they are deleted. |
| `IZWI_ARTIFACT_EPHEMERAL_RETENTION_SECONDS` | 3,600 | Opaque objects labelled `ephemeral`, from creation. Objects labelled `durable` (speech-history audio) never expire here. Also the minimum age of unreferenced input media (below). |
| `IZWI_TTS_REPLAY_RETENTION_SECONDS` | 86,400 | Replay PCM of terminal speech jobs (unchanged). |
| `IZWI_WORKER_HEARTBEAT_RETENTION_SECONDS` | 86,400 | Heartbeat rows of workers that stopped heartbeating. |
| `IZWI_MEDIA_TOMBSTONE_RETENTION_SECONDS` | 86,400 | Media rows tombstoned after their object was deleted and that nothing references. |

Local input media labelled `default` (uploads, canonical WAVs, TTS reference
and output audio) has no clock of its own: it is deleted, object and row, once
nothing uses it and it is older than the ephemeral retention. It is in use while
a job takes it as input or an artifact row points at it (so it lives at least as
long as its job), or while a transcription, diarization, speech-history item or
saved voice holds its audio. A transcription's canonical WAV therefore goes with
its job while the record keeps the upload; the canonical copy of a saved voice's
or history item's audio is kept as long as that audio, since every speech
request with that voice reuses it. A failed object delete leaves the row for the
next pass.

Expired idempotency records go on every pass; rows of the retired legacy
`idempotency_keys` table (no longer written) go after seven days.
Reconciliation repairs pending projections of jobs cancelled in the last seven
days.

Database pool bounds apply to SQLite and PostgreSQL alike:
`IZWI_DB_MAX_CONNECTIONS` (1-1,024, default 5) and `IZWI_DB_ACQUIRE_TIMEOUT_MS`
(default 3,000). Database URLs are redacted (password and credential query
parameters) in errors and debug output.

An upgrade creates `runtime_quota_usage` and seeds it from existing records.
Operators of a provider-managed schema must create that table (`quota_kind`,
`scope`, `used`, primary key on the first two; the empty scope is the global
row), seed it the same way, and add `runtime_jobs.replay_swept_at INTEGER
NULL` before upgrading.

### The durable store on PostgreSQL

Release and Docker builds of `izwi-server` include PostgreSQL support (with
TLS, for example `?sslmode=require`): set
`IZWI_DATABASE_URL=postgres://user:pass@host:5432/izwi` instead of a SQLite
path. An enterprise database hook may choose Izwi-managed or provider-managed
migrations on PostgreSQL. Every node that shares the database runs its own
batch worker:

- **Migrations** are versioned in `schema_migrations` (component, version).
  Each boot applies the pending ones in one transaction under a PostgreSQL
  advisory lock, so nodes starting together apply each migration once. A
  database an earlier unversioned release created upgrades in place.
- **Claims** take one stage per statement with `FOR UPDATE SKIP LOCKED`, so
  workers on several nodes take different stages instead of queueing on one
  row. A stage enqueued on a node wakes that node's idle worker slots at
  once; other nodes pick it up within `idle_poll_interval` (1 s).
- **Time**: leases, retry delays and retention cutoffs use the database
  server's clock (read every 30 s and advanced locally in between), so clock
  skew between nodes cannot expire a live lease early.

For a provider-managed schema, `docs/dev/postgres-schema.sql` is the full
application schema, generated from the migrator: apply it to an empty schema
and Izwi finds every migration recorded. The PostgreSQL CI lane applies it and
checks it against a migrated database.

To run the store suite locally against a disposable database (each test
creates its own schema in it):

```
createdb izwi_store_test
export IZWI_TEST_STORE_PG_URL=postgres://$(whoami)@localhost/izwi_store_test
cargo test -p izwi-db --features db-postgres
cargo test -p izwi-server --features db-postgres --lib -- \
  batch_runtime artifact_store saved_voice_store
```

## Route migration gates

Gateway mode currently exposes text-only
`POST /v1/chat/completions` plus, behind an explicit preview flag, the
realtime transcription relay `GET /v1/realtime/ws` and its public-client
alias `/v1/speech-to-text/realtime/ws` (see below). The following
representative routes return 404 in gateway mode and remain available only
through their existing local/desktop profile where applicable:

- `/v1/models`
- `/v1/audio/speech`
- `/v1/audio/transcriptions`
- `/v1/responses`
- `/v1/media`
- `/v1/voices`
- `/v1/studio/projects`
- `/v1/jobs`
- `/v1/admin/models`
- `/v1/voice/sessions`

Multimodal chat is also local-only. Realtime transcription/voice, agent and chat
workflows, durable jobs/history, media, saved voices, Studio, and model administration
must not be published through the gateway until their state, artifact,
streaming, ownership, and cancellation contracts are migrated.

Do not replace the local server with the gateway for desktop or local workflows.
`--role local` remains the compatibility profile and continues to own its
runtime, SQLite/provider state, and full route set.

### Realtime relay preview (DS3.6, off by default)

`IZWI_GATEWAY_REALTIME=on` exposes `GET /v1/realtime/ws` (WebSocket,
subprotocol `izwi-realtime-v1`) and relays sessions to approved realtime
workers, resolving each admit's task to its stage pool: the approved
speech_to_text deployment serves ASR-stream sessions, and an approved
text_to_speech deployment (same public model) serves TTS-stream sessions.
The gateway is off unless the operator sets the flag. Boot fails closed only
when no realtime stage can be
served at all; a missing individual stage refuses its admits with a policy
close at session time. Knobs:

| Variable | Meaning | Default |
|---|---|---|
| `IZWI_GATEWAY_REALTIME` | `on`/`off`; `on` requires an approved speech_to_text or text_to_speech deployment at boot or the gateway exits fail-closed | `off` |
| `IZWI_GATEWAY_REALTIME_MAX_SESSIONS` | Bounded concurrent relayed sessions per gateway | `64` |
| `IZWI_GATEWAY_REALTIME_SESSION_BUDGET_MS` | End-to-end worker session budget minted into each admit | `600000` |

The relay dials HTTPS workers with the same trust as the HTTP client: the
public web roots plus the configured private CA roots, presenting the mTLS
client identity when one is configured. Each direction of a session is its
own task with a 5 s write deadline, so a client that stops reading cannot
delay its own cancel or audio toward the worker; a peer that misses the
deadline ends the session.

Worker side: ASR workers are configured with `IZWI_WORKER_TASK=speech_to_text`
and `IZWI_WORKER_MODEL=Nemotron-3.5-ASR-Streaming-0.6B` (or a Voxtral
realtime variant). TTS workers use `IZWI_WORKER_TASK=text_to_speech` with a
TTS synthesis family (e.g. `Kokoro-82M`); boot includes a bounded streaming
warm-up and fails closed if synthesis does not produce audio. Each stage
session takes its own tenant lease and dispatch slot; the relay never holds
one stage's permit while awaiting another stage's admission.

A TTS-stream admit (`"input": {"kind": "text_stream"}`) may name a built-in
`speaker` and a `language`, and set `"incremental": true` to have each run of
complete sentences synthesized as its `input` frame arrives instead of the
whole text at `finish`; audio frames stay in order and the last one is
final-flagged. TTS workers advertise `realtime_tts_text_stream` for this, and
the relay sends an admit that sets any of these fields only to such workers
(older ones would ignore them). Client identity:
the gateway mints the attested caller context from the authenticated
principal; the client's session/request/attempt ids pass through, so a
worker-subprotocol client library works against the gateway unchanged. Owner
loss (worker stream lost without a terminal) surfaces as an explicit
internal-error event followed by a close; reconnecting is always a new
session with no resume.

### Public transcription-realtime clients (DS3.4, same flag)

At the same route the gateway dispatches on the subprotocol offer. Clients
offering `izwi-realtime-v1` get the byte-identical passthrough relay above;
clients offering no worker subprotocol get the public transcription-realtime
surface translated onto the worker session, so a client of the single-node
`/v1/speech-to-text/realtime/ws` socket can repoint at the gateway unchanged
— either keeping the single-node path (an alias of it is mounted at the
gateway) or switching to `/v1/realtime/ws`. Both public wire modes are
served: the legacy `transcription_realtime_v2` JSON lifecycle
(`session_start`/`session_stop`/`ping`, ITRW binary audio frames,
`transcript_partial`/`session_done`) and the typed `transcription_realtime`
v3 envelopes (`session_ready`/`session_started`, per-frame ingress events
with audio-gap detection, `transcript_final`/`closing`/`closed`); typed v3
resume requests are rejected exactly as on the single-node surface. Audio is
re-encoded from the public ITRW framing onto the worker's IRTA framing with
the client's sequence numbers preserved; admission (worker selection and
dialing) defers to the first audio frame because the public envelope declares
the sample rate per frame, and a session that never streams audio finishes
locally without touching a worker. Worker deltas accumulate into the public
replaceable partial hypothesis; the worker's terminal delta carries the full
final text and replaces the accumulation. Pings are answered locally; owner
loss and worker errors map onto the public error vocabulary with the same
close semantics as the passthrough mode. Unlike the passthrough relay, the
public surface carries no client session ids, so the gateway mints them —
operator-visible registry entries use gateway-minted ids. Session capacity,
tenant leases, and the session budget are shared with the passthrough mode.

### Realtime ASR session limits (SPK-F9)

Every runtime that serves realtime ASR (a local server or a speech_to_text
worker) bounds its sessions with three limits. Each one is read from the
engine config's `asr_realtime` field when set, else from its environment
variable, else from the default:

| Limit | Environment | Default |
|---|---|---|
| Concurrent sessions | `IZWI_ASR_REALTIME_MAX_SESSIONS` | 16 on CUDA, 4 on CPU and Metal |
| Absolute lifetime (s) | `IZWI_ASR_REALTIME_MAX_LIFETIME_SECS` | 28,800 (8 hours) |
| Idle timeout (s) | `IZWI_ASR_REALTIME_IDLE_TIMEOUT_SECS` | 90 |

The session count is also the number of rows in the retained state arena
that a Nemotron or Parakeet load allocates up front. Nemotron's arena is
about 74 MB per session in F32, all of it held while the model is resident.
Size the count from the concurrency you need, not as high as possible.

Voxtral rolls its stream over at a pause before the model's ceiling, and
Nemotron's streaming state is bounded. The lifetime therefore only reaps a
session that is never closed; it does not limit transcript length.

The idle timeout counts time without audio frames. A client that stops
sending while its microphone is muted keeps its session for that long. Make
the timeout longer than such pauses, or have the client keep sending
silence.

The old defaults were a 10-minute lifetime and a 30 s idle timeout. A
deployment that relied on them must now set them explicitly.

### Hierarchical KV offload (DS4, explicit opt-in)

DS4 demotes cold committed prefix pages from the device KV arenas into a
bounded host pool and promotes them back when a later request's prefix
lookup continues into the host tier. It is off unless the supervisor
assignment carries `host_kv_pool_budget_bytes` (validated against the
assignment's own host/shared limit) or the worker gets
`IZWI_KV_HOST_POOL_BUDGET_BYTES`; the budget is part of the model's
load-time resource authorization, so an over-large budget fails at load,
not under pressure. Kill switch: `IZWI_KV_HOST_OFFLOAD=0`. Tuning knobs:
`IZWI_KV_OFFLOAD_HIGH_WATERMARK`/`IZWI_KV_OFFLOAD_LOW_WATERMARK` (defaults
0.85/0.70 of arena capacity), `IZWI_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES` (8),
`IZWI_KV_OFFLOAD_MAX_PROMOTION_PAGES` (64).

Operationally: pages move only between tiers of the same worker process;
demotion runs synchronously at manager safe points, so a referenced page is
never a victim and active work keeps resolving through the existing preemption
ladder; a promotion that cannot restore truncates and the scheduler re-plans
cold, so reuse degrades before correctness does. On CUDA the pool is
additional capacity across PCIe; on Metal and CPU it is charged to the shared
host/unified ledger (DINV-05) and the win is prefix retention plus admission
headroom, never extra memory. Watch `izwi_engine_kv_cache_host_pages` (gauge,
must stay inside the budget),
`izwi_engine_kv_cache_demotions_total`/`izwi_engine_kv_cache_promotions_total`
(counters; on a healthy shared-prefix workload both advance and the gauge
returns toward zero after promotion), and the per-deployment additive status
fields of the same names. Evidence rig:
`scripts/bench/run-ds4-offload-benchmark.sh` (off/on legs; hard gates on
completion, demotion, promotion, and budget containment).

## Drain, shutdown, and restart

For planned maintenance, drain the gateway before stopping the supervisor:

1. Remove the gateway from external ingress/load-balancer rotation, or let
   the pre-stop window do it.
2. Send SIGTERM (or Ctrl+C) to the gateway. Shutdown takes at most
   `--gateway-shutdown-budget-secs` (default 25, inside Kubernetes' default
   30 s grace period). For the first `--gateway-prestop-secs` (default 5)
   `/readyz` is 503 while the listener stays open and requests are still
   served, so endpoints are withdrawn before anything is refused.
3. Then admission closes and the listener stops accepting; accepted
   responses finish on their own. Shortly before the budget ends (a quarter
   of the drain, at most 2 s, before it) streams still open receive a 503
   `service_unavailable` error event and `[DONE]`, and their exact private
   attempts request cancellation; anything left at the end of the budget is
   dropped. Completion is still worker-authoritative.
   The local role follows the same plan: `IZWI_HTTP_SHUTDOWN_PRESTOP_SECS`
   (default 0, so a desktop or command-line server stops at once) and then
   `IZWI_HTTP_SHUTDOWN_GRACE_SECS` (default 20) of drain, with the same
   stream notice for open chat streams (chat completions, Responses and
   thread chats); it then stops its batch worker and unloads its models.
4. Send SIGTERM (or Ctrl+C) to the supervisor. Closing each parent-owned control
   pipe makes the worker stop admission. The worker waits `drain_grace_ms`,
   requests cancellation for remaining attempts, then waits
   `cancellation_grace_ms`.
5. The supervisor waits for that cooperative window, sends TERM if the child is
   still present, waits `termination_grace_ms`, and finally kills and reaps it.
   The worker reads the same grace (`IZWI_WORKER_TERMINATION_GRACE_MS`): after
   its cooperative window, or at once on the supervisor's TERM, it cancels what
   is left, gives open connections that grace to close, and then exits on its
   own (status 1 when connections were still open). A worker nothing
   supervises treats SIGTERM like Ctrl+C and starts the cooperative drain.
   Review each logged stop outcome: `Cooperative`, `Terminated`, or `Killed`.

If the supervisor dies, managed workers observe control-pipe EOF and enter the
same local drain path. The supervisor also uses kill-on-drop as a final ownership
fence. Do not start a replacement supervisor until the node generation barrier
confirms prior workers released their fences.

An unexpected worker exit is restarted with the configured bounded policy.
So is a ready worker that stops answering: every `[liveness] period_ms`
(default 5000) the supervisor asks the worker's status; an answer that does
not come within `timeout_ms` (3000), comes from another identity, or
reports a failed runtime counts as a failure, and `failure_threshold` (3)
failures in a row terminate the worker (no cooperative drain: it cannot be
asked anything) and restart it like a crash. A deployment that is merely
not ready (a drain-recoverable fault) does not count.

A worker that exceeds `max_restarts_per_window` is quarantined. It is
retried once every `restart.quarantine_retry_ms` (default 600000; 0 waits
for an operator), and a retry that stays up for `stable_reset_ms` ends the
quarantine. Clear it at once, after diagnosing the cause, with SIGHUP (every
quarantined worker) or, on any platform,

```
izwi-serving-supervisor --config /etc/izwi/node.toml <inventory flags> --clear-quarantine chat-cpu-1   # or: all
```

which leaves a request in the runtime directory that the running supervisor
applies within a second. A cleared worker relaunches with a fresh restart
budget. A healthy replacement receives a new incarnation. Gateway polling
may approve it only when logical worker/node, capacity, deployment,
artifact, generation, execution profile, and capability remain unchanged.

There is no hot config or model reload in this profile. For a model update or
rollback:

1. drain gateway, then supervisor;
2. retain the last known-good node and gateway service definitions;
3. change the pinned artifact and assign a new nonzero model generation (also
   use a fresh generation when rolling back, to avoid an ABA identity);
4. run validate-only;
5. start the supervisor and wait for exact readiness;
6. update the gateway approval generation, start the gateway, and check
   `/readyz`; and
7. restore ingress only after bounded JSON and SSE smoke requests pass.

If startup fails, stop the partial rollout, restore the known-good artifact and
configuration with another fresh generation, and repeat the same sequence.

For a pure model-generation cutover on one node, prefer the coordinated
rollout below: it replaces the manual approval-edit step (step 6) with an
atomic, supervisor-computed approvals view and adds an automatic abort path.
The manual sequence remains the fallback for changes a rollout plan cannot
express (node identity, capacity, non-rolling workers).

### Coordinated rollout (DS6)

One supervisor invocation performs the cutover with automatic abort. Write a
rollout plan that names the already-validated target node config (new nonzero
generations), the gateway's shared approvals file, the canary replacement, and
how to learn that the gateways adopted each view:

```toml
schema_version = 1
target_node_config = "/etc/izwi/node-v2.toml"   # absolute path; validated in full before anything changes
shared_approvals_path = "/var/lib/izwi/shared-approvals"  # the gateway's IZWI_GATEWAY_SHARED_APPROVALS_PATH file
canary_worker_id = "chat-worker-2"              # replacement launched first; its readiness failure aborts
strategy = "auto"                               # auto (default), blue_green or recreate
abort_grace_secs = 30                           # wait after an abort restore before stopping replacements (0–300)

# Recommended: ask the gateways instead of guessing from their refresh interval.
[gateway_ack]
admin_urls = ["http://127.0.0.1:8080"]          # every gateway reading the file
admin_key_env = "IZWI_GATEWAY_ADMIN_KEY"        # supervisor env var holding the gateways' admin key
timeout_secs = 120                              # per view; not acknowledged in time aborts (1–3600)
```

With `[gateway_ack]` the supervisor polls each gateway's
`GET /internal/admin/approvals` (admin key; it reports the adopted file's
digest, the approval lines and the generation each pool routes to) after
every view it writes, and moves on only when every gateway reports the view
adopted. `window_secs` (default 0) is then a pure soak after the
acknowledged cutover, and no commit grace is needed. Without it the
supervisor cannot know when a view was adopted and falls back to timers
sized from the refresh interval the plan declares:

```toml
gateway_approvals_refresh_secs = 30   # the gateways' IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS, in seconds (1–3600, default 30)
window_secs = 60                      # ≥ 2× the refresh, ≤ 3600; default 2× the refresh
commit_grace_secs = 30                # ≥ the refresh, ≤ 3600; default the refresh
```

A window of two refreshes lets every gateway adopt the window view (and cut
over) before the commit view lands, and a commit grace of one refresh lets
every gateway stop routing to the old generation before it drains. Declare
the gateways' real refresh interval; plan validation rejects shorter timers.

The target config's non-rolling workers and node identity must match the
running config exactly; only the rolling deployments' generations advance.

**Strategy.** Blue-green runs both generations side by side, so plan
validation checks that the current workers and the replacements fit the
node together: host memory, CPU threads, distinct worker ids and binds, and
exclusive devices. A single Metal device or GPU never fits twice, so `auto`
(the default) uses blue-green when the union fits and recreate otherwise;
`blue_green` refuses a plan that does not fit and says why. Recreate drains
the old generation and then starts the new one, like a Kubernetes rolling
update with `maxSurge: 0`: the deployment is unavailable from the drain
until the replacements are ready (the model load time).

Start (or restart) the supervisor with the plan — `--config` stays the current
(old) config; the plan carries the target:

```
izwi-serving-supervisor --config /etc/izwi/node-v1.toml \
  --cpu-worker-binary <path> --cpu-ids <ids> --allocatable-host-memory-bytes <bytes> \
  --rollout-plan /etc/izwi/rollout.toml
```

A blue-green rollout then:

1. launches the current generation's workers, then the replacements
   canary-first (canary readiness failure aborts immediately);
2. adds the replacements' lines to the shared file (the window view) and
   persists `window_open`; each gateway adopts the view at runtime and
   moves admission to the new generation the moment it observes it Ready
   (never two eligible generations, never zero). With `[gateway_ack]` the
   rollout waits until every gateway reports that cutover; not in time, it
   aborts;
3. after the soak, writes the commit view (this node's old-generation lines
   removed), waits for every gateway to acknowledge it (or the commit grace),
   persists `draining_old` — the point of no return — and drains the old
   workers, then records `committed` and keeps supervising the
   replacements. A commit view not acknowledged in time aborts: the old
   lines go back and the old generation never stopped serving.

A recreate rollout writes the commit view first (`recreate_draining_old`),
waits for the acknowledgement or the commit grace, stops the old
generation, launches the replacements canary-first (`recreate_launching`)
and records `committed` once they are ready. If a replacement fails
readiness, or an operator aborts, the view is restored and the old
generation is launched again.

Repoint the service definition at the target config before the next
supervisor restart. Under systemd, run the rollout as the unit: add a
drop-in (`systemctl edit izwi-serving-supervisor`) whose `ExecStart` appends
`--rollout-plan /etc/izwi/rollout.toml`, and restart the unit (any
supervisor restart drains and relaunches the current generation first).
Once `--rollout-status` reports `committed`, change the drop-in to the
target `--config` without the plan and run `systemctl daemon-reload`; the
next restart then finds the committed state matching its config. Leaving
the plan in place would stage the same rollout again on that restart.

Monitor and intervene without guessing:

- `--rollout-status` reports the persisted phase (`launching_replacement`,
  `window_open`, `draining_old`, `recreate_draining_old`,
  `recreate_launching`, `committed`, `aborted`) and window deadline.
- Abort while reversible: send SIGUSR2 to the supervisor, or stop it —
  shutdown during launch/window aborts and restores automatically. A
  replacement that exits during the window aborts too. The abort removes
  exactly the lines the rollout added (and puts back any it removed) under
  the approvals file lock, so the file is byte-identical to the pre-rollout
  file unless someone else edited it meanwhile, and their edit is kept. It
  then waits `abort_grace_secs` so the gateway stops routing to the
  replacements and drains them.
- If the supervisor dies mid-rollout, managed workers self-drain and a fresh
  start fails closed until you rerun `--rollout-plan` (resume with the same
  plan: the old generation relaunches only in pre-drain states, replacements
  relaunch alongside it, and a resumed window runs in full again; past the
  drain the old generation is never relaunched) or `--rollout-abort`
  (restore the approvals without launching anything; after a recreate's
  drain, start the supervisor with the old config to serve the old
  generation again). Terminal states clear automatically; a `committed`
  state requires the committed target config's digest.
- Rollback after `committed` is a new rollout plan back to the previous
  configuration with a fresh generation — never a generation reuse.

Limit the blast radius honestly: `draining_old` is irreversible (worker
control-pipe EOF cannot be un-sent), the coordinator covers one supervisor
over one shared approvals file, and the views it writes use the workers'
loopback endpoints, so every gateway must run on the same host.
Hand-editing the shared approvals file is still possible and now safe
against the supervisor's own edits (they all take the file's lock and start
from what is there), but it has runtime effect.

### Signal-driven worker autoscaling (DS7)

**Scope.** This is an optional, default-off capability for one host. The
desktop and single-node profile does not need it (a single GPU can never
hold a second worker), and fleets should scale with Kubernetes and KEDA on
the workers' metrics instead (see "Service managers and Kubernetes"). The
supervisor can act as a capacity manager for one deployment on its node:
scale out on sustained load, scale in after a stabilized idle window. It owns
the v1 pinned approval lines of its autoscaled deployments in the gateway's
shared approvals file; those lines carry the workers' loopback endpoints, so
the gateways must run on the same host.

Autoscaling is off unless the node config carries an `[autoscaling]` block.
All workers of an autoscaled deployment are declared in the node config as
usual (own id, bind, assignment, budgets); `max_workers` names the full
declared replica set and `min_workers` the subset launched at startup. The
rest are standbys that only a scale-up decision starts:

```toml
[autoscaling]
shared_approvals_path = "/var/lib/izwi/shared-approvals"  # the gateway's IZWI_GATEWAY_SHARED_APPROVALS_PATH file
# evaluation_interval_ms = 1000                           # 100–60000, default 1000

[autoscaling.deployments.chat-prod]
min_workers = 1                              # launched at startup, never scaled down
max_workers = 3                              # must equal the declared worker count for the deployment
scale_up_queue_depth = 4                     # a running worker holding >= this many invocations (active + queued)...
scale_up_sustained_polls = 3                 # ...for this many consecutive evaluations
# scale_up_stabilization_window_ms = 0       # hold scale-up this long after any scale event (default 0)
scale_down_stabilization_window_ms = 60000   # 1000–86400000; hold scale-down this long, and the idle window
```

Startup behavior with the block present: the supervisor reconciles the shared
approvals view to the min set (standby lines are removed so the gateway never
routes to a not-running endpoint; unrelated and foreign-node lines pass
through untouched), launches exactly the min set, and evaluates every
`evaluation_interval_ms` thereafter.

Scale semantics:

- **Scale-up** requires the deployment below `max_workers`, a standby
  available, the scale-up window clear, and a running worker holding at
  least `scale_up_queue_depth` invocations (active plus queued; workers do
  not queue today, so this is the active count and validation rejects a
  depth above the workers' `max_active_invocations`) for
  `scale_up_sustained_polls` consecutive evaluations. The standby starts
  through the standard launch path; the supervisor does not wait for it and
  publishes its approvals line once it is ready. A worker that crashed is
  never picked as a standby.
- **Scale-down** requires the deployment above `min_workers`, the
  scale-down window clear, and a non-core worker fully idle (zero queued,
  zero active, zero reserved sessions) for the whole window. The worker's
  line is removed first (the gateway stops routing new work after its view
  refresh), its control pipe closes immediately (it stops admission without
  waiting for the gateway TTL), the supervisor waits for zero active work,
  and only then stops it. The slot returns to the standby pool; scale-down
  never touches the startup min set.
- **Windows** are per direction, like the Kubernetes HPA: after any scale
  event, scale-up waits `scale_up_stabilization_window_ms` (default 0, so a
  burst scales out at once) and scale-down waits
  `scale_down_stabilization_window_ms`. They run on the monotonic clock.
- **Budgets**: config validation proves the whole declared replica set
  (standbys included) fits the node at once — host memory, CPU threads and
  device exclusivity — so a scale-up needs no check of its own.

Operate it like the rest of the supervisor:

- `--validate-only` prints the parsed policy; SIGUSR1 diagnostics include an
  `autoscale deployment=... running=N standby=N draining=N` line per
  autoscaled deployment plus `autoscale_ups_total`/`autoscale_downs_total`.
- Decisions freeze (never guess) while a worker of the deployment is
  restart-pending or loading, a status poll fails, or an approvals write
  fails; the deployment resumes deciding once observability is whole again.
  A quarantined worker does not freeze it.
- Standbys are ordinary declared workers: they resolve secrets at startup
  and are covered by the restart controller once activated.
- Autoscaling and `--rollout-plan` are mutually exclusive in one supervisor
  run (both own the shared view). Stop autoscaling by removing the block
  before rolling; scale state is in-memory by design — a supervisor restart
  returns to the declared min set and reconciles the view to it.
- Hand-editing lines for autoscaled deployments of this node is futile (the
  next reconciliation overwrites them) and a standalone-form line colliding
  with an autoscaled endpoint fails the view write closed; the v1 pinned
  form is required.

## Troubleshooting

| Symptom | Check | Safe response |
|---|---|---|
| Supervisor validate-only reports a missing secret | The exact `bearer_token_env` name and service-manager environment | Supply the secret out of band; do not add it to TOML or argv |
| Config rejects host memory, CPUs, endpoint, or device | Operator inventory, aggregate budgets, real directories/executable, loopback ports | Correct the allocation; never enable backend fallback |
| Worker never becomes Ready | Worker log for manifest, strict backend, budget-pair, load/warm-up, or identity failure | Leave it unadvertised; fix the pinned contract or resource assignment |
| Gateway exits during startup | Empty, duplicated or topology-forbidden endpoints; two endpoints reporting the same logical worker | Fix the approvals; do not weaken approval checks |
| Approved worker never adopted | Warning "Approved worker is not adoptable yet" at startup, then retries at debug level; the endpoint must answer authenticated descriptor/status and match task/model/deployment/generation | Start or fix the worker; the gateway adopts it without a restart |
| `/readyz` is 503 | Lifecycle/draining and `compatible_worker_ready` detail; status TTL/poll; matching JSON+SSE capability | Restore a fresh matching worker or correct the static approval |
| Public route returns 401 | Public bearer key reference/value | Rotate/fix the environment; caller identity headers cannot repair auth |
| Public route returns 403 or 503 from policy | Enterprise inference policy result or hook health | Repair policy service/rules; do not bypass the hook |
| Request returns 400/413 | Request ID/header/body/model/multimodal bounds | Fix the request or explicitly configured bounded limit |
| Request returns overload/unavailable | Gateway semaphore, fresh eligible workers, worker authoritative capacity, drain state, circuit logs | Shed load or add an independently budgeted worker; do not create an unbounded proxy queue |
| Same tenant receives 429 | Concurrency: `IZWI_GATEWAY_TENANT_MAX_CONCURRENT` and exact attempts still awaiting terminal teardown. Rate: `IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE` / `IZWI_GATEWAY_TENANT_BURST_REQUESTS` | Wait for confirmed completion/teardown or raise the limits only with measured capacity; do not release on client disconnect |
| Stream stops without terminal output | Timeout, proxy disconnect, slow consumer, worker/protocol error | Treat result as unknown; inspect the exact attempt and never replay automatically |
| Worker repeatedly restarts or is quarantined | First failure, resource limits, manifest/model load, lock ownership, stable-uptime threshold; `failed N consecutive liveness probes` means it stopped answering | Fix the root cause, then `--clear-quarantine <worker>` (or SIGHUP); otherwise it is retried every `quarantine_retry_ms` |
| New worker incarnation is ignored | Logical worker/node, capacity, artifact/generation/profile/capability drift | Make node config and gateway approval agree; do not accept status-advertised drift |

Use structured service logs and request IDs, but do not log authorization
headers, bearer values, raw private error bodies, or tenant payloads. Gateway
and worker metrics plus supervisor diagnostics remain operational evidence, not
teardown proof; only the exact-attempt terminal paths described above release
accepted-work ownership.

## Fleet operation (multi-gateway)

Multi-gateway operation is validated on two coordination-store lanes:
a shared SQLite file on one host, or a shared PostgreSQL database for a
server-backed fleet (see the coordination database section below). The
authority contract is unchanged from single-node operation: the worker
remains the atomic admission arbiter, capacity claims only steer selection,
and an unreachable coordination store degrades to worker-authoritative
admission — dispatches keep succeeding, nothing panics. Remote
TLS termination and separate-machine handshakes remain explicit release
gates.

**Partitioned quotas.** Set `IZWI_GATEWAY_FLEET_SIZE=N` and
`IZWI_GATEWAY_FLEET_PARTITION=i` (0-based, `i < N`) on each gateway. Every
gateway owns a strict 1/N slice (floored to 1) of the configured tenant
request-rate and concurrency budgets, so the fleet total cannot exceed the
configured limits; a crashed gateway's slice simply goes unused until it
restarts, and its peers neither absorb it nor are affected by it. No shared
atomic counter or coordination service is required. Partitioning is the
explicitly-chosen quota fallback; without it, each gateway's tenant budget is
enforced worker-authoritatively. Every gateway logs its effective posture at
startup (`selection_mode`, `quota_mode`).

**Shared approvals.** Point every gateway at the same bounded approvals file
with `IZWI_GATEWAY_SHARED_APPROVALS_PATH=/absolute/path/approvals.txt`
(max 64 KiB, 256 entries, one 5-field standalone or 8-field v1 approval per
line, `#` comments and blank lines skipped). CLI approvals remain explicit;
shared entries augment them and duplicate endpoints fail closed at startup.
Each gateway refreshes its cached view every
`IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS` (1s–1h, default 30s) and adopts a
changed view at runtime: workers the view adds are adopted (and retried until
they answer), pollers for removed workers stop, and generation changes apply
through the rollout cutover rule. In-flight streams are never moved.

**Fleet coordination database (optional).** Point every gateway at the same
coordination database — a SQLite file with
`IZWI_GATEWAY_FLEET_DB_PATH=/absolute/path/fleet.sqlite3`, or a shared
PostgreSQL database with
`IZWI_GATEWAY_FLEET_DB_PATH=postgres://user:pass@db-host:5432/izwi_fleet` —
to share in-transit capacity claims (each gateway polls every worker's status
itself). Each gateway claims one short-lived capacity unit per dispatch;
selection steers
away from peer-filled workers while the worker remains the atomic admission
arbiter, so a lost race degrades to one alternate dispatch. A claim covers
only the transit from selection to the worker's acceptance and is released
when the worker accepts: from then on the worker's own status reports the
work. Peer views count other gateways' claims only, since each gateway already
tracks its own dispatches locally. Unreleased claims expire by
TTL — `IZWI_GATEWAY_FLEET_CLAIM_TTL_MS`, bounded 100ms–300s, default 30000 —
which is the crash-recovery path: no explicit recovery protocol. A slow
background sweep additionally reaps expired claims, so a long-lived fleet
does not leak rows. Each gateway's claims carry its identity,
`IZWI_GATEWAY_ID` (1–128 ASCII letters, digits, `-` or `_`; an invalid value
fails boot). Unset, the gateway takes a random identity for this boot, never
one derived from its process id (every container is PID 1); with a stable
operator-set identity a restarted gateway also releases its leftover claims
at startup instead of waiting out the TTL. Unset
`IZWI_GATEWAY_FLEET_DB_PATH` means single-gateway operation with purely
process-local state; that is the default.

The coordination store never sits on a request's or a status poll's critical
path. A dispatch waits at most `IZWI_GATEWAY_FLEET_CLAIM_TIMEOUT_MS` (1ms–5s,
default 100) for its claim and otherwise invokes without one; a claim that
lands after that is released at once. Status polls record each observation
locally first and refresh the worker's peer-claim count from the store in
the background, so a slow or hung store cannot make healthy workers look
stale. When the store cannot be
read, a gateway stops counting peer claims for that worker and admission is
worker-authoritative until the store answers again. PostgreSQL coordination
sessions carry a 5 s `statement_timeout` and
`idle_in_transaction_session_timeout`, so a stuck statement or abandoned
claim transaction cannot hold a worker's advisory lock.

PostgreSQL is the validated server-backed store: the migrator promotes the
shared DDL per backend, and the fleet store execution suite and the durable
store suite run against real PostgreSQL in CI
(`scripts/ci/check-backend-truth.sh cargo-fleet-stores`). The coordination
database holds only the fleet tables (its own `fleet` migrations).
To run it locally against a disposable Homebrew PostgreSQL database:

```
brew services start postgresql@18
createdb izwi_ds5_test
IZWI_TEST_FLEET_PG_URL=postgres://$(whoami)@localhost/izwi_ds5_test \
  cargo test -p izwi-gateway --features db-postgres --lib fleet_postgres
```

MySQL is not supported: `mysql://` database URLs are rejected at startup.

**Operator drain.** Configure `IZWI_GATEWAY_ADMIN_API_KEY_REF` to a bounded
`env:VARIABLE` secret that differs from the inference key. Then
`POST /internal/admin/drain` with that bearer credential returns 202 and
marks the gateway draining (same path as SIGTERM). The endpoint returns 404
when no admin key is configured and 401 for the inference key.

**Canary launch ordering.** `--canary-worker-id <worker-id>` launches that
worker first within a single plain boot; remaining workers start only after
the canary reaches readiness, and the supervisor exits if it fails. It orders
launches only — it never moves gateway routing between generations and
cannot be combined with `--rollout-*`. For a generation cutover use the
coordinated rollout (see "Drain, shutdown, and restart"), which
adds the dual-generation approval window, automatic abort, and resume.

**Worker TLS.** The worker terminates TLS when both
`IZWI_WORKER_TLS_CERT_REF` and `IZWI_WORKER_TLS_KEY_REF` name bounded
absolute `file:` PEM paths (256 KiB each), optionally requiring client
certificates via `IZWI_WORKER_TLS_CLIENT_CA_REF` (mTLS). Non-loopback binds
are rejected without TLS unless `IZWI_WORKER_ALLOW_INSECURE_PLAINTEXT_BIND=1`
acknowledges a private network. Supervised workers bind loopback only.
Remote deployment needs separate-machine handshake evidence before any
support claim.

## Service managers and Kubernetes (scope D1)

The supported scopes are deliberately narrow:

- **Desktop and single node** (first-class): the supervisor with its
  per-worker supervision, liveness probing, bounded restarts and
  quarantine, plus the coordinated rollout (recreate on a one-GPU node).
  Run the supervisor and the gateway under a service manager:
  `config/serving/systemd/izwi-serving-supervisor.service` and
  `izwi-gateway.service` are ready-made units (secrets in mode-0600
  environment files, `KillMode=mixed` so stopping the supervisor drains its
  workers, `RuntimeDirectory=izwi` for the node lease and fences).
- **Optional, default off**: supervisor autoscaling (an `[autoscaling]`
  block) and multi-gateway coordination (`IZWI_GATEWAY_FLEET_DB_PATH`).
  Both stay runtime switches in the one release binary rather than cargo
  features, so nothing changes for a node that does not set them and a
  fleet operator needs no custom build. Neither is needed on one host.
- **Fleets**: let Kubernetes supervise, scale and roll out.
  `config/serving/kubernetes/` holds a worker StatefulSet with a GPU
  request and liveness on the worker status (`worker.yaml`; a StatefulSet so
  every replica keeps the stable endpoint and identity the gateway's v1
  approvals pin), a gateway Deployment and Service in the
  `fleet-one-gateway` topology with mutual TLS, identities from pod names
  and the approvals in a ConfigMap (`gateway.yaml`), and a KEDA
  ScaledObject that scales the workers on `izwi_worker_active_invocations`
  with immediate scale-out and a five-minute scale-in window, plus the
  Prometheus scrape job the workers need (`autoscaling.yaml`). Approve every
  worker ordinal up to the scaler's maximum; the gateway adopts each worker
  when it answers.

Rollouts on Kubernetes: Argo Rollouts and plain rolling updates manage
Deployments' ReplicaSets, and a worker generation is pinned in the
approvals, so roll a generation out as the supervisor does: create a second
StatefulSet for the new generation (its own ordinals and deployment
generation), add its lines to the `izwi-approvals` ConfigMap next to the old
ones (the window view; each gateway cuts over when it sees a new worker
Ready, never two eligible generations), confirm with each gateway's
`GET /internal/admin/approvals` that the pool routes to the new generation,
then remove the old lines, wait for the gateways to report that view, and
scale the old StatefulSet to zero. The one approval the fleet-one-gateway
topology requires on the command line moves with the gateway Deployment's
arguments, which a rolling update of the gateways applies. Argo Rollouts fits the gateway
Deployment (a canary of a new gateway build behind the Service), and an
Argo Rollouts `AnalysisTemplate` can gate a worker generation's promotion
on the gateways' error rate or on `izwi_worker_failed_total`.

## Evidence and support matrix

The labels below apply to this separated production-serving architecture, not
to unrelated local-engine tests elsewhere in the repository.

| Lane | Current evidence | Operational status |
|---|---|---|
| Scripted-runtime worker | The real worker service over a scripted runtime (`izwi-serving-worker` `test-support`), on real loopback sockets: the client contract suites and public gateway tests cover auth/version/bounds, success, incompatible model/deployment, authoritative capacity rejection, timeout, lost acknowledgement, partial stream, cancellation, two replicas, circuit fencing, and the bounded one-alternate policy. The focused alternate-dispatch suite passed 7/7 in this work session, including bounded backoff/jitter. | Implementation/test evidence only; not production readiness |
| Real CPU execution | A subprocess worker loaded/warmed a generated tiny supported LFM2 GGUF and completed public JSON and SSE chat through the gateway | Proven development-path process separation; not a production model/resource/performance certificate |
| Metal compilation and execution | A metal-feature worker served 120 gateway-routed generations on the tiny fixture; `backend_parity.rs` proves greedy Metal outputs equal CPU outputs; supervisor launch with exact device identity is process-evidenced (`multi_lane_launch.rs`) | Development lane proven on the fixture; not a production model, capacity, soak, or performance certificate |
| CUDA compilation | Compile-checked by the `cargo-cuda-compile` lane; no serving-specific CUDA execution was run | Compilation only |
| Real CUDA execution | The supervisor accepts, validates and launches CUDA assignments with the exact declared UUID (`multi_lane_launch.rs`); no NVIDIA device was available to execute on | Not established |
| Physical multi-GPU | Parser/topology and scripted-runtime replica tests only | Not established |
| Multi-machine | Worker binary terminates server-side TLS/mTLS from bounded file references (unit-tested parsing; plaintext non-loopback rejected). Client HTTPS trust, fleet topology policy, and v1 approvals exist. No separate-machine handshake, remote artifact/state path, or partition test was exercised | Not supported as an operational profile |
| Multi-gateway | Two real gateway processes over one shared coordination database (`tests/fleet_rig.rs`, SQLite and PostgreSQL lanes) prove atomic admission, partitioned quotas that are not multiplied, shared approvals, claim steering, gateway-crash plus TTL recovery, and store-outage degradation; unit tests cover a hung store (status polls and dispatch proceed). Circuit state and accepted-work ownership stay per-gateway | Supported on the validated store lanes (SQLite on one host, PostgreSQL); not a production HA certificate. See the [support matrix](PRODUCTION_SERVING_SUPPORT_MATRIX.md) |
| Soak/load/performance | Closed-loop gateway chat benchmark harness exists (`scripts/bench/run-gateway-chat-benchmark.py`) reporting TTFT/latency percentiles, throughput, and completed/rejected/failed rates; no hours-long soak or representative production load matrix has been run | Not established |

Before any production claim, complete the open route, quota, observability,
artifact/state, mTLS, HA, accelerator, overload, soak, and performance gates in
the implementation plan. Two scripted-runtime workers returning responses—or one tiny CPU
fixture completing chat—does not make the system production-ready.
