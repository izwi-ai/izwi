# Single-node production-serving operations runbook

Status: implementation-aligned deployment guide, not a production-readiness
certificate

This runbook describes only the production-serving behavior currently present
in this checkout. It covers a single trusted node with a separately started
public gateway, CPU-only supervisor executable, and one or more private CPU
workers. The contracts are designed for CPU, Apple Metal, and NVIDIA CUDA, but
the currently packaged supervisor command rejects Metal and CUDA assignments.
See the [evidence matrix](#evidence-and-support-matrix) before choosing a
profile.

The architecture and route decisions are recorded in
[ADR 0002](adr/0002-production-worker-boundary.md), the
[route migration ledger](PRODUCTION_SERVING_DISCOVERY.md), and the
[packaging evidence](PRODUCTION_SERVING_PACKAGING.md). Use the
[release checklist and risk register](PRODUCTION_SERVING_RELEASE_CHECKLIST.md)
and the [exact-profile support matrix](PRODUCTION_SERVING_SUPPORT_MATRIX.md)
for profile approval. Node configuration
examples live in [`config/serving/examples`](../../config/serving/examples/README.md).

## Supported topology

The supported operational shape in this runbook is:

```text
TLS ingress (required before a non-loopback public bind)
        |
        v
izwi-server --role gateway       hardware-independent; no model/runtime/DB
        |
        | authenticated HTTP on numeric loopback
        v
one or more izwi-serving-worker processes
        ^
        |
izwi-serving-supervisor          validates, starts, fences, restarts, drains
```

- The gateway owns the public OpenAI-compatible request contract, authentication,
  policy call, request bounds, routing, and response/SSE translation. Gateway
  mode branches before `RuntimeService`, persistence, preload, local batch
  workers, or accelerator delegation are initialized.
- The supervisor assigns each child its CPU threads/memory or accelerator
  identity/memory before process startup. It clears the child environment,
  restores only an allowlist, and injects the validated assignment. It starts
  initial workers serially and serializes their allocation-heavy load stage.
- Each worker owns exactly one loaded model, runtime, scheduler, model
  generation, and authoritative admission semaphore. One invocation never
  crosses workers or devices.
- Multiple workers may be independent replicas of one deployment or distinct
  deployments. The gateway selects from a bounded, fresh registry. A worker's
  admission result remains authoritative when gateway load observations are
  stale.
- No Kubernetes, broker, external database, or cloud service is required. The
  gateway and synchronous chat path do not require SQLite. Existing durable job
  and artifact code remains in the local server profile and is not made a
  gateway dependency.

The `izwi-serving-supervisor` executable currently accepts CPU workers only and
requires an operator-supplied CPU-ID inventory and allocatable host-memory
ceiling. Although the shared schema and worker support strict Metal/CUDA
assignments, there is no launcher inventory integration for those lanes yet.
Never change an explicit Metal or CUDA assignment to CPU merely to make
validation pass.

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

The server assigns the `inference` role/scope. Public `x-principal-id`,
`x-tenant-id`, `x-scopes`, and `x-roles` headers are not authority. The
enterprise inference policy hook runs after authentication and fails closed on
an error.

Keep secrets in the service manager or secret store that creates the process
environment. Do not put them in node TOML, command-line arguments, shell
history, logs, or health-check URLs.

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

- The bundled worker rejects every non-loopback bind. Plaintext worker client
  URLs must use a numeric loopback host such as `127.0.0.1` or `::1`; `localhost`
  is deliberately not accepted.
- The worker client accepts certificate-verified HTTPS endpoints, augments the
  platform trust store with up to 16 private CA PEM files, and can present one
  client certificate/key identity. Configure bounded absolute `file:`
  references with `IZWI_GATEWAY_WORKER_TLS_CA_REFS` (comma-separated),
  `IZWI_GATEWAY_WORKER_TLS_CLIENT_CERT_REF`, and
  `IZWI_GATEWAY_WORKER_TLS_CLIENT_KEY_REF`. Certificate and key references must
  be supplied together. Each PEM file is limited to 256 KiB and all private CA
  files together to 1 MiB. TLS material on a plaintext endpoint is rejected.
- The bundled worker does not terminate TLS. A remote HTTPS topology therefore
  still needs a separately managed trusted TLS/mTLS ingress or sidecar proxying
  to the worker's loopback listener. This is not a native multi-machine profile
  until that deployment has separate-machine transport evidence.
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
5. Keep the current worker deployment contract unchanged unless the worker code
   supports the replacement. The real worker currently supports only
   `LFM2.5-1.2B-Instruct-GGUF`, task `chat`, GGUF `q4_k_m`, native LFM2 execution,
   chat-message input, text output, and cooperative cancellation.
6. Set every `bearer_token_env` in the supervisor's environment. Use the same
   private credential ID and secret for all workers routed by one gateway.

The worker verifies its artifact manifest before constructing the runtime. It
then initializes the exact assigned backend without fallback, loads and warms
the model, and only then binds its private listener.

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

Validation is currently CPU-only. A Metal or CUDA node file is expected to fail
at the executable's device-lane boundary; this is fail-closed behavior, not a
request to edit the backend to CPU.

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

Do not start the gateway until every configured worker needed at gateway startup
is Ready. Gateway startup reads every approved endpoint synchronously and fails
if one is unavailable or violates its approval.

### 2. Start the gateway

Prefer typed approvals over the legacy pinned-worker form. For one CPU worker,
the process shape is:

```sh
izwi-server \
  --role gateway \
  --host 127.0.0.1 \
  --port 8080 \
  --public-model LFM2.5-1.2B-Instruct-GGUF \
  --gateway-worker-approval 'http://127.0.0.1:9470|chat|LFM2.5-1.2B-Instruct-GGUF|lfm25-cpu-v1|1' \
  --gateway-max-in-flight 32 \
  --gateway-worker-queue-wait-ms 250 \
  --gateway-worker-status-ttl-ms 10000 \
  --gateway-worker-status-poll-ms 2000
```

Supply the public and private credential environment described above. Approval
syntax is exactly:

```text
URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION
```

Repeat `--gateway-worker-approval` for replicas. A pool is keyed by task and
public model. All replicas in a pool must agree on deployment ID, generation,
artifact revision, execution representation, tokenizer revision, and exact
capability; backend and precision may differ and remain worker properties. The
CLI pins task/model/deployment/generation, while the gateway freezes the rest of
the authenticated initial status contract. The node TOML is therefore the
operator's artifact-revision source of truth for this single-node profile.

Legacy `--worker-endpoint` pins one exact incarnation and should be retained
only for compatibility. Legacy repeated `--gateway-worker-endpoint` entries use
one chat deployment/generation supplied separately. Do not mix legacy endpoints
with typed approvals.

The registry supports at most 256 configured workers; deployment tables support
at most 64 pools and 256 replicas per pool. Poll interval must be nonzero and
less than status TTL. The gateway uses receiver-monotonic observation time and
strictly increasing status sequence numbers rather than trusting remote clocks.

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

The response is capped at 4 KiB and contains only a fixed set of unlabeled
counters/gauges for active admitted requests and streams, auth/quota/body
rejections, routing or dispatch failures, dispatch calls and accumulated
latency, and HTTP status classes. It intentionally contains no request IDs,
tenant IDs, model names, prompts, or worker-controlled labels. A dropped SSE
body counts as a stream failure unless a terminal completion was observed.

## Admission, retry, timeout, and cancellation semantics

- Gateway admission is fail-fast and bounded by `--gateway-max-in-flight`; it
  does not create an unbounded waiting queue. The lifecycle is checked both
  before and after taking a permit so drain cannot admit a racing request.
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
| Configured registry | 256 workers; bounded deployment/local-dispatch tables |
| Worker request body | node `max_request_bytes`, at most 64 MiB by schema |
| Worker execution | `max_active_invocations`, atomic and fail-fast; no transport queue |
| Worker event channel | 4 events; 1 MiB encoded event limit in the real worker |
| Worker attempt records | configured `max_retained_attempts`, at least active capacity, max 65,536; time-bounded by `attempt_retention_secs` (1-86,400) |
| Private client JSON | 1 MiB request, 512 KiB control body, 16 KiB error body defaults |
| Private NDJSON | 256 KiB line, 16 MiB total, 4,096 events by default |
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
deletion, but no gateway artifact route, automatic expiry/GC, or fleet adapter
is enabled.

## Route migration gates

Gateway mode currently exposes only text-only
`POST /v1/chat/completions`. The following representative routes return 404 in
gateway mode and remain available only through their existing local/desktop
profile where applicable:

- `/v1/models`
- `/v1/audio/speech`
- `/v1/audio/transcriptions`
- `/v1/responses`
- `/v1/admin/models`
- `/v1/voice/sessions`

Multimodal chat is also local-only. Realtime transcription/voice, agent and chat
workflows, durable jobs/history, media/saved voices, and model administration
must not be published through the gateway until their state, artifact,
streaming, ownership, and cancellation contracts are migrated. The complete
ownership decision is in the
[route migration ledger](PRODUCTION_SERVING_DISCOVERY.md#route-migration-ledger).

Do not replace the local server with the gateway for desktop or local workflows.
`--role local` remains the compatibility profile and continues to own its
runtime, SQLite/provider state, and full route set.

## Drain, shutdown, and restart

For planned maintenance, drain the gateway before stopping the supervisor:

1. Remove the gateway from external ingress/load-balancer rotation.
2. Send SIGTERM (or Ctrl+C) to the gateway. It marks itself draining and closes
   admission before beginning HTTP graceful shutdown. `/readyz` becomes 503.
3. Allow accepted responses to finish. `IZWI_HTTP_SHUTDOWN_GRACE_SECS` bounds
   this wait to 1-300 seconds (default 20). At expiry, remaining HTTP
   connections are dropped and their exact private attempts request
   cancellation; completion is still worker-authoritative.
4. Send SIGTERM (or Ctrl+C) to the supervisor. Closing each parent-owned control
   pipe makes the worker stop admission. The worker waits `drain_grace_ms`,
   requests cancellation for remaining attempts, then waits
   `cancellation_grace_ms`.
5. The supervisor waits for that cooperative window, sends TERM if the child is
   still present, waits `termination_grace_ms`, and finally kills and reaps it.
   Review each logged stop outcome: `Cooperative`, `Terminated`, or `Killed`.

If the supervisor dies, managed workers observe control-pipe EOF and enter the
same local drain path. The supervisor also uses kill-on-drop as a final ownership
fence. Do not start a replacement supervisor until the node generation barrier
confirms prior workers released their fences.

An unexpected worker exit is restarted with the configured bounded policy. A
worker that exceeds `max_restarts_per_window` is quarantined; restart the
supervisor only after diagnosing the cause. A healthy replacement receives a
new incarnation. Gateway polling may approve it only when logical worker/node,
capacity, deployment, artifact, generation, execution profile, and capability
remain unchanged.

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

## Troubleshooting

| Symptom | Check | Safe response |
|---|---|---|
| Supervisor validate-only reports a missing secret | The exact `bearer_token_env` name and service-manager environment | Supply the secret out of band; do not add it to TOML or argv |
| Config rejects host memory, CPUs, endpoint, or device | Operator inventory, aggregate budgets, real directories/executable, loopback ports | Correct the allocation; never enable backend fallback |
| Worker never becomes Ready | Worker log for manifest, strict backend, budget-pair, load/warm-up, or identity failure | Leave it unadvertised; fix the pinned contract or resource assignment |
| Gateway exits during startup | Every approved endpoint must answer authenticated descriptor/status and match task/model/deployment/generation | Start/fix workers first; do not weaken approval checks |
| `/readyz` is 503 | Lifecycle/draining and `compatible_worker_ready` detail; status TTL/poll; matching JSON+SSE capability | Restore a fresh matching worker or correct the static approval |
| Public route returns 401 | Public bearer key reference/value | Rotate/fix the environment; caller identity headers cannot repair auth |
| Public route returns 403 or 503 from policy | Enterprise inference policy result or hook health | Repair policy service/rules; do not bypass the hook |
| Request returns 400/413 | Request ID/header/body/model/multimodal bounds | Fix the request or explicitly configured bounded limit |
| Request returns overload/unavailable | Gateway semaphore, fresh eligible workers, worker authoritative capacity, drain state, circuit logs | Shed load or add an independently budgeted worker; do not create an unbounded proxy queue |
| Stream stops without terminal output | Timeout, proxy disconnect, slow consumer, worker/protocol error | Treat result as unknown; inspect the exact attempt and never replay automatically |
| Worker repeatedly restarts or is quarantined | First failure, resource limits, manifest/model load, lock ownership, stable-uptime threshold | Drain the gateway, fix root cause, validate, then restart supervisor |
| New worker incarnation is ignored | Logical worker/node, capacity, artifact/generation/profile/capability drift | Make node config and gateway approval agree; do not accept status-advertised drift |

Use structured service logs and request IDs, but do not log authorization
headers, bearer values, raw private error bodies, or tenant payloads. The
current diagnostics and metrics surface is incomplete; absence of a log or
metric is not proof of teardown.

## Evidence and support matrix

The labels below apply to this separated production-serving architecture, not
to unrelated local-engine tests elsewhere in the repository.

| Lane | Current evidence | Operational status |
|---|---|---|
| Deterministic mock worker | Real-loopback HTTP contract and public gateway tests cover auth/version/bounds, success, incompatible model/deployment, authoritative capacity rejection, timeout, lost acknowledgement, partial stream, cancellation, two replicas, circuit fencing, and the bounded one-alternate policy. The focused alternate-dispatch suite passed 7/7 in this work session, including bounded backoff/jitter. | Implementation/test evidence only; not production readiness |
| Real CPU execution | A subprocess worker loaded/warmed a generated tiny supported LFM2 GGUF and completed public JSON and SSE chat through the gateway | Proven development-path process separation; not a production model/resource/performance certificate |
| Metal compilation | No serving-specific Metal build was run in this work session | Not established |
| Real Metal execution | No separated gateway/supervisor/worker inference was run on Metal | Not established; supervisor executable rejects Metal configs |
| CUDA compilation | No CUDA toolchain was available and no serving-specific CUDA build was run | Not established |
| Real CUDA execution | No NVIDIA device was available | Not established; supervisor executable rejects CUDA configs |
| Physical multi-GPU | Parser/topology and mock-replica tests only | Not established |
| Multi-machine | Worker-client URL validation accepts certificate-verified HTTPS; no native worker TLS listener, remote artifact/state path, partition test, or deployment was exercised | Not supported as an operational profile |
| Soak/load/performance | No hours-long soak, representative production load matrix, latency/throughput measurement, or resource-efficiency benchmark for this architecture | Not established |

Before any production claim, complete the open route, quota, observability,
artifact/state, mTLS, HA, accelerator, overload, soak, and performance gates in
the implementation plan. Two mock workers returning responses—or one tiny CPU
fixture completing chat—does not make the system production-ready.
