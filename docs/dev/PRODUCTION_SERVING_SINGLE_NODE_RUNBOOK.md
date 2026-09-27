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
- No Kubernetes, broker, or cloud service is required. The gateway and
  synchronous chat path do not require SQLite. Single-node profiles need no
  external database; the multi-gateway fleet profile shares one coordination
  database (SQLite on one host or PostgreSQL for a server-backed fleet).
  Durable job and artifact code remains in the local server profile and is not
  made a gateway dependency.

The `izwi-serving-supervisor` executable supervises CPU, Metal, and CUDA
worker lanes (ADR 0003), requiring an operator-declared device inventory and
allocatable host-memory ceiling. Never change an explicit Metal or CUDA
assignment to CPU merely to make validation pass.

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
rejected; `key_ref` must be a bounded `env:VARIABLE` or `file:PATH` reference —
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
  requires `--role gateway`, HTTPS, a configured client certificate/private
  key, and versioned worker approvals that pin both node and worker identity.
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

Supply the public and private credential environment described above. The
standalone numeric-loopback compatibility syntax is:

```text
URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION
```

The fleet-one-gateway syntax is versioned and pins operator-approved logical
identities rather than trusting the first descriptor returned by an endpoint:

```text
v1|HTTPS_URL|NODE_ID|WORKER_ID|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION
```

Fleet mode rejects the pinned `--worker-endpoint` form, legacy repeated
`--gateway-worker-endpoint` values, five-field approvals, plaintext endpoints,
and missing client identity before descriptor/status network I/O. Bearer
service authentication remains required in addition to mTLS. This policy does
not make the bundled loopback-only worker a supported remote deployment: use a
reviewed TLS/mTLS proxy and retain real separate-machine handshake evidence.
Configurations that previously used remote HTTPS under the implicit local
policy must migrate explicitly to `fleet-one-gateway`, v1 approvals, and client
certificate/key references; there is no legacy remote-HTTPS compatibility mode.

Repeat `--gateway-worker-approval` for replicas. A pool is keyed by task and
public model. All replicas in a pool must agree on deployment ID, generation,
artifact revision, execution representation, tokenizer revision, and exact
capability; backend and precision may differ and remain worker properties. The
CLI pins task/model/deployment/generation, while the gateway freezes the rest of
the authenticated initial status contract. The node TOML is therefore the
operator's artifact-revision source of truth for this single-node profile.

Legacy `--worker-endpoint` pins one exact incarnation and should be retained
only for standalone compatibility. Legacy repeated `--gateway-worker-endpoint`
entries use one chat deployment/generation supplied separately. Do not mix
legacy endpoints with typed approvals.

The registry supports at most 256 configured workers; deployment tables support
at most 64 pools and 256 replicas per pool. Poll interval must be nonzero and
less than status TTL. The gateway uses receiver-monotonic observation time and
strictly increasing status sequence numbers rather than trusting remote clocks.
Admission/header timeout is configurable from 1 ms to 60 seconds, first-output
and inter-output idle timeouts from 1 ms to one hour, and slow-consumer relay
timeout from 1 ms to 60 seconds. Queue wait cannot exceed admission timeout.
Every phase remains capped by the original request's total remaining deadline;
transport or metadata trickles do not extend it.

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
deletion, but no gateway artifact route, automatic expiry/GC, or fleet adapter
is enabled.

Opaque artifact creation requires reserved-write protocol v1 from the selected
media provider. Its database reservation exists before bytes leave the process,
expires after one minute, and is atomically consumed by metadata publication.
Provider calls are bounded to thirty seconds. Maintenance uses exact cleanup
claim tokens and recovers by write ID, including reservations with no recorded
provider key. `NotFound` means both absent and unable to commit later for that
expired write ID. The reservation ledger is capped at 65,536 and each cleanup
scan at 64. New durable Fish PCM replay chunks use tenant-scoped opaque artifact
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
streaming, ownership, and cancellation contracts are migrated. The complete
ownership decision is in the
[route migration ledger](PRODUCTION_SERVING_DISCOVERY.md#route-migration-ledger).

Do not replace the local server with the gateway for desktop or local workflows.
`--role local` remains the compatibility profile and continues to own its
runtime, SQLite/provider state, and full route set.

### Realtime relay preview (DS3.6, off by default)

`IZWI_GATEWAY_REALTIME=on` exposes `GET /v1/realtime/ws` (WebSocket,
subprotocol `izwi-realtime-v1`) and relays sessions to approved realtime
workers, resolving each admit's task to its stage pool: the approved
speech_to_text deployment serves ASR-stream sessions, and an approved
text_to_speech deployment (same public model) serves TTS-stream sessions.
The gateway is off unless the operator sets the flag; pinned single-worker
gateway mode rejects it. Boot fails closed only when no realtime stage can be
served at all; a missing individual stage refuses its admits with a policy
close at session time. Knobs:

| Variable | Meaning | Default |
|---|---|---|
| `IZWI_GATEWAY_REALTIME` | `on`/`off`; `on` requires an approved speech_to_text or text_to_speech deployment at boot or the gateway exits fail-closed | `off` |
| `IZWI_GATEWAY_REALTIME_MAX_SESSIONS` | Bounded concurrent relayed sessions per gateway | `64` |
| `IZWI_GATEWAY_REALTIME_SESSION_BUDGET_MS` | End-to-end worker session budget minted into each admit | `600000` |

Worker side: ASR workers are configured with `IZWI_WORKER_TASK=speech_to_text`
and `IZWI_WORKER_MODEL=Nemotron-3.5-ASR-Streaming-0.6B` (or a Voxtral
realtime variant). TTS workers use `IZWI_WORKER_TASK=text_to_speech` with a
TTS synthesis family (e.g. `Kokoro-82M`); boot includes a bounded streaming
warm-up and fails closed if synthesis does not produce audio. Each stage
session takes its own tenant lease and dispatch slot; the relay never holds
one stage's permit while awaiting another stage's admission. Client identity:
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

Operationally: pages move only between tiers of the same worker process
(ADR 0004); demotion runs synchronously at manager safe points, so a
referenced page is never a victim and active work keeps resolving through
the existing preemption ladder; a promotion that cannot restore truncates
and the scheduler re-plans cold, so reuse degrades before correctness does.
On CUDA the pool is additional capacity across PCIe; on Metal and CPU it is
charged to the shared host/unified ledger (DINV-05) and the win is prefix
retention plus admission headroom, never extra memory. Watch
`izwi_engine_kv_cache_host_pages` (gauge, must stay inside the budget),
`izwi_engine_kv_cache_demotions_total`/`izwi_engine_kv_cache_promotions_total`
(counters; on a healthy shared-prefix workload both advance and the gauge
returns toward zero after promotion), and the per-deployment additive
status fields of the same names. Evidence rig:
`scripts/bench/run-ds4-offload-benchmark.sh` (off/on legs; hard gates on
completion, demotion, promotion, and budget containment).

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

For a pure model-generation cutover on one node, prefer the coordinated
rollout below: it replaces the manual approval-edit step (step 6) with an
atomic, supervisor-computed approvals view and adds an automatic abort path.
The manual sequence remains the fallback for changes a rollout plan cannot
express (node identity, capacity, non-rolling workers).

### Coordinated blue-green rollout (DS6)

One supervisor invocation performs the cutover with automatic abort
(ADR 0006). Write a rollout plan that names the already-validated target node
config (new nonzero generations), the gateway's shared approvals file, the
canary replacement, and the soak/abort windows:

```toml
schema_version = 1
target_node_config = "/etc/izwi/node-v2.toml"   # absolute path; validated in full before anything changes
shared_approvals_path = "/etc/izwi/shared-approvals.toml"  # the gateway's IZWI_GATEWAY_SHARED_APPROVALS_PATH file
canary_worker_id = "chat-worker-2"              # replacement launched first; its readiness failure aborts
window_secs = 30                                # soak window between cutover and draining the old generation (0–3600)
abort_grace_secs = 30                           # wait after an abort restore before stopping replacements (0–300)
```

The target config's non-rolling workers and node identity must match the
running config exactly; only the rolling deployments' generations advance.

Start (or restart) the supervisor with the plan — `--config` stays the current
(old) config; the plan carries the target:

```
izwi-serving-supervisor --config /etc/izwi/node-v1.toml \
  --cpu-worker-binary <path> --cpu-ids <ids> --allocatable-host-memory-bytes <bytes> \
  --rollout-plan /etc/izwi/rollout.toml
```

The supervisor then:

1. launches the current generation's workers, then the replacements
   canary-first (canary readiness failure aborts immediately);
2. writes the window view — both generations approved, one atomic file
   write — and persists `window_open` state; the gateway adopts the view at
   runtime and moves admission to the new generation the moment it observes
   it Ready (never two eligible generations, never zero);
3. after the soak window writes the commit view (new generation only) and
   drains the old workers — the point of no return — then records
   `committed` and keeps supervising the replacements; repoint the service
   definition at the target config for the next supervisor restart.

Monitor and intervene without guessing:

- `--rollout-status` reports the persisted phase (`launching_replacement`,
  `window_open`, `draining_old`, `committed`, `aborted`) and window deadline.
- Abort while reversible: send SIGUSR2 to the supervisor, or stop it —
  shutdown during launch/window aborts and restores automatically. A
  replacement that exits during the window aborts too. The abort restores
  the pre-rollout approvals byte-identically first, waits
  `abort_grace_secs` so the gateway stops routing to the replacements, then
  drains them; the old generation never stopped serving.
- If the supervisor dies mid-rollout, managed workers self-drain and a fresh
  start fails closed until you rerun `--rollout-plan` (resume with the same
  plan: the old generation relaunches only in pre-drain states, replacements
  relaunch alongside it; in `draining_old` the old generation is never
  relaunched) or `--rollout-abort` (restore without launching anything).
  Terminal states clear automatically; a `committed` state requires the
  committed target config's digest.
- Rollback after `committed` is a new rollout plan back to the previous
  configuration with a fresh generation — never a generation reuse.

Limit the blast radius honestly: `draining_old` is irreversible (worker
control-pipe EOF cannot be un-sent), the coordinator covers one supervisor
and one gateway over one shared approvals file, and hand-editing the shared
approvals file is still possible — it now has runtime effect, so leave the
file to the rollout command.

### Signal-driven worker autoscaling (DS7)

The supervisor can act as a capacity manager for one deployment on its node:
scale out on sustained queue depth, scale in after a stabilized idle window
(ADR 0007). It is supervisor-managed — no Kubernetes, no external scaler —
and fleet-profile only: the supervisor owns the v1 pinned approval lines of
its autoscaled deployments in the gateway's shared approvals file.

Autoscaling is off unless the node config carries an `[autoscaling]` block.
All workers of an autoscaled deployment are declared in the node config as
usual (own id, bind, assignment, budgets); `max_workers` names the full
declared replica set and `min_workers` the subset launched at startup. The
rest are standbys that only a scale-up decision starts:

```toml
[autoscaling]
shared_approvals_path = "/etc/izwi/shared-approvals"  # the gateway's IZWI_GATEWAY_SHARED_APPROVALS_PATH file
# evaluation_interval_ms = 1000                       # 100–60000, default 1000

[autoscaling.deployments.chat-prod]
min_workers = 1                              # launched at startup, never scaled down
max_workers = 3                              # must equal the declared worker count for the deployment
scale_up_queue_depth = 4                     # a running worker reporting queued >= this depth...
scale_up_sustained_polls = 3                 # ...for this many consecutive evaluations
scale_down_stabilization_window_ms = 60000   # 1000–86400000; hysteresis + idle window
```

Startup behavior with the block present: the supervisor reconciles the shared
approvals view to the min set (standby lines are removed so the gateway never
routes to a not-running endpoint; unrelated and foreign-node lines pass
through untouched), launches exactly the min set, and evaluates every
`evaluation_interval_ms` thereafter.

Scale semantics:

- **Scale-up** requires the deployment below `max_workers`, a standby
  available, hysteresis clear, and a running worker at or above
  `scale_up_queue_depth` for `scale_up_sustained_polls` consecutive
  evaluations. The standby launches through the standard launch/readiness
  path (readiness is awaited inline, like the canary path) and gains its
  approvals line only after readiness.
- **Scale-down** requires the deployment above `min_workers`, hysteresis
  clear, and a non-core worker fully idle (zero queued, zero active, zero
  reserved sessions) for the whole stabilization window. The worker's line
  is removed first (the gateway stops routing new work after its view
  refresh), its control pipe closes immediately (it stops admission without
  waiting for the gateway TTL), the supervisor waits for zero active work,
  and only then runs the standard drain/stop. The slot returns to the
  standby pool; scale-down never touches the startup min set.
- **Hysteresis** is shared by both directions: no scale event may start
  within `scale_down_stabilization_window_ms` of the previous scale event.
- **Budgets** (DS7.3): every scale-up reserves the candidate's declared
  budgets in the node resource ledger first — host memory, CPU threads,
  device exclusivity — and a reservation that would overcommit is rejected
  with a diagnostic instead of launched. Config validation already proves
  the full declared replica set fits the node, so a rejection means ledger
  misuse, not reachable config state.

Operate it like the rest of the supervisor:

- `--validate-only` prints the parsed policy; SIGUSR1 diagnostics include an
  `autoscale deployment=... running=N standby=N draining=N` line per
  autoscaled deployment plus `autoscale_ups_total`/`autoscale_downs_total`.
- Decisions freeze (never guess) while a worker is restart-pending, a status
  poll fails, or an approvals write fails; the affected deployment resumes
  deciding once observability is whole again.
- Standbys are ordinary declared workers: they resolve secrets at startup
  and are covered by the restart controller once activated. A quarantined
  standby stays standby until an operator clears it.
- Autoscaling and `--rollout-plan` are mutually exclusive in one supervisor
  run (both own the shared view). Stop autoscaling by removing the block
  before rolling; scale state is in-memory by design — a supervisor restart
  returns to the declared min set and reconciles the view to it.
- Hand-editing lines for autoscaled deployments of this node is futile (the
  next reconciliation overwrites them) and a standalone-form line colliding
  with an autoscaled endpoint fails the view write closed; the fleet
  profile's v1 pinned form is required.

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
| Same tenant receives 429 | `IZWI_GATEWAY_TENANT_MAX_CONCURRENT` and exact attempts still awaiting terminal teardown | Wait for confirmed completion/teardown or raise the limit only with measured capacity; do not release on client disconnect |
| Stream stops without terminal output | Timeout, proxy disconnect, slow consumer, worker/protocol error | Treat result as unknown; inspect the exact attempt and never replay automatically |
| Worker repeatedly restarts or is quarantined | First failure, resource limits, manifest/model load, lock ownership, stable-uptime threshold | Drain the gateway, fix root cause, validate, then restart supervisor |
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
admission — dispatches keep succeeding, nothing panics (ADR 0005). Remote
TLS termination and separate-machine handshakes remain explicit release
gates.

**Partitioned quotas.** Set `IZWI_GATEWAY_FLEET_SIZE=N` and
`IZWI_GATEWAY_FLEET_PARTITION=i` (0-based, `i < N`) on each gateway. Every
gateway owns a strict 1/N slice (floored to 1) of the configured tenant
request-rate and concurrency budgets, so the fleet total cannot exceed the
configured limits and a crashed gateway releases its partition with no effect
on its peers. No shared atomic counter or coordination service is required.
Partitioning is the explicitly-chosen quota fallback; without it, each
gateway's tenant budget is enforced worker-authoritatively. Every gateway
logs its effective posture at startup (`selection_mode`, `quota_mode`).

**Shared approvals.** Point every gateway at the same bounded approvals file
with `IZWI_GATEWAY_SHARED_APPROVALS_PATH=/absolute/path/approvals.txt`
(max 64 KiB, 256 entries, one 5-field standalone or 8-field v1 approval per
line, `#` comments and blank lines skipped). CLI approvals remain explicit;
shared entries augment them and duplicate endpoints fail closed at startup.
Each gateway refreshes its cached view every
`IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS` (1s–1h, default 30s) and logs drift;
newly approved workers are adopted on rolling gateway restart, never
mid-stream.

**Fleet coordination database (optional).** Point every gateway at the same
coordination database — a SQLite file with
`IZWI_GATEWAY_FLEET_DB_PATH=/absolute/path/fleet.sqlite3`, or a shared
PostgreSQL database with
`IZWI_GATEWAY_FLEET_DB_PATH=postgres://user:pass@db-host:5432/izwi_fleet` —
to share worker observations and capacity claims. Each gateway publishes its
polled worker statuses (monotonic per incarnation; incarnation changes always
win) and claims one short-lived capacity unit per dispatch; selection steers
away from peer-filled workers while the worker remains the atomic admission
arbiter, so a lost race degrades to one alternate dispatch. Claims expire by
TTL — `IZWI_GATEWAY_FLEET_CLAIM_TTL_MS`, bounded 100ms–300s, default 30000 —
which is the crash-recovery path: no explicit recovery protocol. A slow
background sweep additionally reaps expired claims and prunes observation
rows for workers that stopped reporting beyond 24h, so a long-lived fleet
does not leak rows. Each gateway also releases its own leftover claims at
startup (same `IZWI_GATEWAY_ID` after a restart), so capacity frees
immediately instead of waiting out the TTL. Unset means single-gateway
operation with purely process-local state.

PostgreSQL is the validated server-backed store: the migrator promotes the
shared DDL per backend and the fleet store execution suite runs against real
PostgreSQL in CI (`scripts/ci/check-backend-truth.sh cargo-fleet-stores`).
To run it locally against a disposable Homebrew PostgreSQL database:

```
brew services start postgresql@18
createdb izwi_ds5_test
IZWI_TEST_FLEET_PG_URL=postgres://$(whoami)@localhost/izwi_ds5_test \
  cargo test -p izwi-server --features db-postgres --lib fleet_postgres
```

MySQL dialect SQL remains written but unvalidated — it has never been
executed against a live MySQL server, so MySQL-backed fleet coordination
stays explicitly unsupported.

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
coordinated blue-green rollout (see "Drain, shutdown, and restart"), which
adds the dual-generation approval window, automatic abort, and resume.

**Worker TLS.** The worker terminates TLS when both
`IZWI_WORKER_TLS_CERT_REF` and `IZWI_WORKER_TLS_KEY_REF` name bounded
absolute `file:` PEM paths (256 KiB each), optionally requiring client
certificates via `IZWI_WORKER_TLS_CLIENT_CA_REF` (mTLS). Non-loopback binds
are rejected without TLS. The bundled supervisor still launches CPU workers
only; remote deployment needs an operator-managed ingress plus
separate-machine handshake evidence before any support claim.

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
| Multi-machine | Worker binary terminates server-side TLS/mTLS from bounded file references (unit-tested parsing; plaintext non-loopback rejected). Client HTTPS trust, fleet topology policy, and v1 approvals exist. No separate-machine handshake, remote artifact/state path, or partition test was exercised | Not supported as an operational profile |
| Multi-gateway | Conservatively partitioned quotas (1/N slices, unit-tested), shared approvals file with TTL-cached views (unit-tested), shared SQLite coordination for observations and atomic capacity claims (tested, incl. multi-connection same-file atomicity and concurrent idempotency keys never double-acquiring), backend-conditional fleet SQL for PostgreSQL/MySQL written best-effort but unvalidated, registry circuit/replacement-incarnation fencing (unit-tested). Single-host equivalents proven: worker kill fails over to a replacement incarnation, gateway crash cannot duplicate execution, crashing workers are quarantined, reconnects re-approve same-generation workers and fail closed on generation drift. No multi-machine/store-outage tests | Not supported as an operational profile |
| Soak/load/performance | Closed-loop gateway chat benchmark harness exists (`scripts/bench/run-gateway-chat-benchmark.py`) reporting TTFT/latency percentiles, throughput, and completed/rejected/failed rates; no hours-long soak or representative production load matrix has been run | Not established |

Before any production claim, complete the open route, quota, observability,
artifact/state, mTLS, HA, accelerator, overload, soak, and performance gates in
the implementation plan. Two mock workers returning responses—or one tiny CPU
fixture completing chat—does not make the system production-ready.
