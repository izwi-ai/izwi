# Fish S2 multi-user serving qualification

Capacity belongs to an exact GPU/model/provider/build/workload profile. A number
of HTTP connections, active sequences, tensor rows and replicas are different
controls. There is no universal five-user limit and no GPU-memory-only throughput
promise. Load tests measure the deployed system; local fixtures do not certify it.

## Repeatable concurrent and open-loop measurements

`run-fish-streaming-benchmark.py` opens a separate connection and writes separate
PCM/WAV evidence for every request. Its `--concurrency` bounds outstanding work;
it is no longer descriptive metadata. A request file can contain either one
speech-history streaming request object or an array of objects cycled across
arrivals. Use the actual deployed POST URL and authorized test voices. The test
creates real history records and audio artifacts.

```sh
python3 scripts/bench/run-fish-streaming-benchmark.py \
  --url https://YOUR-TEST-SERVER/v1/text-to-speech \
  --request /tmp/fish-workload.json --metadata /tmp/fish-deployment.json \
  --requests 1200 --concurrency 12 --output /tmp/fish-closed-c12

python3 scripts/bench/run-fish-streaming-benchmark.py \
  --url https://YOUR-TEST-SERVER/v1/text-to-speech \
  --request /tmp/fish-workload.json --metadata /tmp/fish-deployment.json \
  --requests 1200 --concurrency 24 --arrival-rate 0.3 \
  --output /tmp/fish-open-r03
```

These example numbers describe experiments, not recommended serving capacity.
Choose enough requests for the intended duration: open-loop arrival duration is
approximately `(requests - 1) / arrival_rate`, plus final drain. Sweep geometric
concurrency through saturation and irregular widths 3/5/7, the resolved limit and
one above it. Repeat open-loop rates below, around and above sustainable capacity.
Run an initial multi-hour soak and longer tests appropriate to operational SLOs.

Metadata requires `deployed_sha`, `gpu`, `dtype`, `runtime_versions`, `checkpoint`,
`build_features`, `attention_provider`, `effective_context`, `concurrency` and
`cache_state`. Record exact GPU/MIG identity, VRAM, driver/CUDA/cuDNN versions,
checkpoint revision and loaded capacity profile, not marketing-family aliases.
CLI concurrency overwrites the saved concurrency field with the actual setting.
Use separate output directories for every run; existing evidence is never replaced.

Open-loop scheduling uses absolute monotonic arrival times, with bounded pending
work. If the generator reaches its connection ceiling, an arrival is recorded as
`generator_dropped`; it is never silently queued or relabeled as server rejection.
Increase generator capacity or use multiple generators before drawing server
saturation conclusions from such a run. HTTP 429/503 count as server rejections;
other HTTP, network and malformed/partial stream failures remain failures. No
automatic request retries hide failures or duplicate speech jobs. `--timeout` is
a socket-operation timeout, not a hard end-to-end deadline.

Each run saves metadata, exact workloads, per-request events without duplicate
base64 audio, partial PCM on failures, validated successful WAVs, request JSONL
and `summary.json`. Summary metrics include:

- Offered/sent/completed/rejected/failed/generator-dropped counts.
- Completed requests and generated audio seconds per wall second, including drain.
- Client PCM receipt TTFA, request-to-last-PCM RTF and request wall time percentiles.
- Actual sample count/rate, sequence integrity, per-request chunk gaps and server timings.
- Generator scheduling delay and peak outstanding client work.

Outstanding client work is not proof of native tensor batching or simultaneous
CUDA launches. Server queue wait, stage occupancy/padding, physical memory,
per-tenant fairness and cache state need separately retained server traces. PCM
receipt is not audible playback: playback latency and underruns remain null until
measured in a real client. Percentiles use completed requests only; rejection and
failure counts must accompany them. Save generator CPU/network/disk headroom;
storing audio and tracing can itself bottleneck a load generator.

## Evaluate one profile against explicit load SLOs

```sh
python3 scripts/bench/report-fish-serving-capacity.py \
  /tmp/fish-open-r03 /tmp/fish-open-r04 \
  --ttfa-p99-ms 1000 --rtf-p99 1 \
  --min-completed 1000 --min-wall-seconds 3600 \
  --max-failure-fraction 0 --operating-headroom 0.2 \
  --output /tmp/fish-capacity-report.json
```

The numerical thresholds are examples; ratify them for each workload before
claiming capacity. This tool refuses to merge different deployment metadata or
workloads. It fails runs with too few samples, insufficient duration, generator
drops, missing/nonfinite latency, excess failure/rejection, or exceeded SLOs. Only
passing open-loop runs nominate a measured arrival-capacity candidate. The report
applies explicit operating headroom to measured completion throughput and always
sets `production_certified: false`: numerical load SLOs alone cannot certify a
release. Closed-loop results remain useful comparison evidence.

Production evidence additionally requires native multi-row traces, state/RNG and
codec parity, listening/content/voice review, mixed workload/tenant fairness,
slow readers, disconnects at every stage, memory and disk plateaus, lease/fence
faults, real proxy behavior and replica drain/restart. Test hot/cold/distinct
references, short/long/multilingual targets, context boundaries and competing
models. Preserve unsuccessful runs. Never extrapolate the paper's H200 RTF into
a users-per-GPU claim.

## Heterogeneous replica deployment contract

This checkout contains Docker deployment files but no actual Modal wrapper for
the reported endpoint. The wrapper, account quotas, routing policy and production
storage must be changed and tested in their owning deployment repository. Do not
apply an untested example Modal function as if it were that service.

For each GPU pool, keep a versioned operational profile with the following fields:

| Field | Required value/evidence |
| --- | --- |
| Identity | Source SHA, checkpoint revision, GPU/MIG, provider, dtypes, runtime versions |
| Runtime capacity | Loaded sealed profile and safe active/AR/codec/prefill/output ceilings |
| HTTP limits | Target inflight for latency and hard maximum aligned with admission |
| Fleet bounds | Warm minimum, buffer, maximum replicas, quota/region limits |
| Scaling | Queue-to-first-audio and rejection pressure, cold-start measurement, cooldown |
| Placement | Compatible warmed engine, predicted queue delay, optional cache affinity |
| Ownership | Globally unique worker identity, lease generation and stage-attempt fencing |
| Persistence | Shared authoritative job database and durable object storage accessible to every replica |
| Drain | Stop new assignments, renew active leases, complete streams or truthfully fail at deadline |
| Release evidence | Exact profile's load report, traces, correctness/failure/soak artifacts |

Start with one persistent shared engine per GPU replica. Increase tensor width
within its sealed profile independently of input concurrency and replica count.
Input concurrency at a proxy/container only allows concurrent handlers; it does
not implement tensor batching. Never load one model copy per incoming request.
A local SQLite file or local artifact path is not a shared fleet database/storage
contract. Verify the actual deployment's consistency, publication and retrieval
semantics before routing durable jobs across replicas.

Route new work to compatible warm capacity using serving pressure. Reference
cache affinity must not send work to an overloaded replica. Pin each live stream
to its owner; do not restart an already emitted stream on another GPU. Retry only
safe pre-output operations with idempotency. Readiness requires successful model
load/capability publication and warmup. Quarantine device-fatal failures.

At scale-in or rollout: withdraw readiness/new assignments, keep lease renewals
alive, drain bounded active requests and artifact persistence, then terminate.
The platform termination grace must exceed the configured application drain
window plus cleanup. Exercise forced expiry and check truthful terminal failures,
no duplicate audio and stale worker fencing. Start with a single qualified GPU
pool canary; compare equal workloads against the baseline. Roll back by draining
and starting the prior compatible profile for new requests, never by recomputing
an audible request.

Verify SSE passes through the actual proxy without buffering and with appropriate
idle/request timeouts. Exercise speech-history and OpenAI streaming formats,
durable jobs, client disconnects and slow readers separately. In Modal, verify
async cancellation behavior, input limits, warm/buffer/max containers and the
external storage topology in the real wrapper. No production deployment or GPU
capacity qualification is performed by these scripts.

## Tool regression tests

```sh
python3 scripts/bench/test-fish-streaming-benchmark.py
python3 scripts/bench/test-fish-serving-capacity.py
```

The streaming tests include a local threaded HTTP server proving simultaneous
connections and correct per-request WAV output, a saturated open-loop generator,
and malformed/partial-stream accounting. Localhost socket permission is required.
The report tests cover sample/duration gates, saturation, missing/nonfinite
latency and rejections. These are tooling tests, not model performance results.

## Runtime rollout controls

`IZWI_FISH_NATIVE_BATCHING=true` opts a deployment into the native Fish batch
path. It defaults to scalar execution until the exact GPU/provider profile is
qualified. `false` selects the scalar path for newly loaded models. Changing the
variable cannot resize a loaded adapter; drain and reload to change the profile.
Unknown physical planning headroom also keeps the scalar path. Enabling this flag
is an experimental rollout decision, not a hardware qualification result.

At load, Fish derives a conservative row ceiling from authoritative free planning
headroom after weight reconciliation, requested-context KV geometry (native for
automatic context), Fast AR
state and per-row workspaces, retaining 20% planning headroom. Explicit
`IZWI_MAX_BATCH_SIZE` and scheduler/retained/staged ceilings can lower that bound.
The state allocator and per-request resource authority still determine whether
actual state and stage work fit; no requested context is shortened by this profile.
CUDA graphs and concurrent physical launches are not enabled by this flag.

Scheduler, retained-sequence and staged-transaction defaults now use zero as an
automatic policy. Runtime startup resolves CUDA administrative ceilings from total
selected-device memory (one administrative row per 256 MiB), with a scalar fallback
when device memory is unknown. Portable backends resolve these administrative
ceilings to eight. These numbers bound bookkeeping; they do not authorize model
memory or claim sustainable users. Positive operator values remain explicit caps.
Fish's loaded stage profile governs native width; other models retain their
existing generic tensor-width policies.

Authenticated speech admission also accepts `IZWI_MAX_SPEECH_REQUESTS_PER_TENANT`.
Its default is the global HTTP speech admission capacity. The tenant key comes
from the trusted authenticated principal namespace, not an arbitrary request
field. A streaming request retains its permit until response EOF or disconnect;
opening an SSE connection does not release its concurrency credit. Ratify each
tenant's limit together with global admission and measured GPU capacity.

Durable workers use a unique UUID-based identity per process. Optional
`IZWI_BATCH_WORKER_CONCURRENCY` is a positive ceiling clamped to the smaller of
resolved runtime retained capacity and queue capacity. It controls active durable
leases feeding the shared engine, not tensor width.

Persistence and delivery controls are process-wide unless noted:

| Control | Default | Purpose |
| --- | --- | --- |
| `IZWI_TTS_TOTAL_SPOOL_BYTES` | 1 GiB | Aggregate temporary speech WAV allocation |
| `IZWI_SPEECH_SPOOL_DIR` | System temporary directory `izwi/speech-spool` child | Owner-only local scratch root for process-locked speech temporary files |
| `IZWI_TTS_TOTAL_UPLOAD_BYTES` | 256 MiB | Aggregate in-progress upload buffers |
| `IZWI_AUDIO_STREAM_TOTAL_EVENT_BYTES` | 64 MiB | Aggregate queued SSE event bytes |
| `IZWI_AUDIO_STREAM_MAX_EVENT_BYTES` | 4 MiB | Per-stream queued SSE event bytes |
| `IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS` | 30 seconds | Bound stalled delivery waits |

Core PCM output credits are separately bounded at 4 MiB per request and 64 MiB
per process. These are output-memory ceilings, not GPU batch limits. A row without
credits yields before codec dispatch. Persistent ordered delivery lanes keep a
slow connection from blocking other streams and defer terminal completion until
its PCM delivery completes. The low-level scalar `EngineCore::step` API retains
its synchronous delivery behavior. Keep both paths covered by regression tests.

Prefill quanta are bounded by a model-derived per-token workspace envelope covering
dense buffers, full-context attention and logits. The stage declares a finite
aggregate token budget, so a long prompt progresses in admitted chunks rather
than claiming an unpriced full-context step. Loaded managed-state row limits use
the same profile during actual allocation and portable context fitting. Scalar
tensor fallback can still retain and interleave multiple independently owned
requests within those limits.

Streaming-only request admission prices the 16 MiB core staging budget and every
independent PCM queue: the frozen engine `StreamingOutput` queue, the caller's
`AudioChunk` channel capacity, and three in-flight chunk buffers, including
per-message overhead. `IZWI_STREAM_AUDIO_QUEUE_CAPACITY` (fallback
`IZWI_STREAM_QUEUE_CAPACITY`) defaults to eight for TTS; its resolved value is
frozen on the request and used by the actual channel constructor. Larger caller
channels therefore increase the reservation instead of hiding uncharged PCM.
A Fish request reaching an already-admitted streaming helper without frozen output
capacity is rejected before execution. First-party Fish TTS freezes and prices
these capacities before admission.

## Qualification limits of this implementation

The native paths are implementation candidates behind the explicit rollout flag.
Local numerical and transaction tests do not establish GPU throughput or user
capacity. The loaded profile is a conservative geometry/memory bound; it does not
contain a measured per-GPU latency curve or an online SLO feedback controller.
Dispatch uses bounded ready work inside that profile. Tenant concurrency admission, class-weighted tenant service and bounded prefill
quanta provide resource isolation. Tenant fairness uses logical work units and
bounded aging; it is not a measured GPU-time entitlement or a client-playback-aware
audio deadline scheduler. Playback acknowledgments, measured underrun targets and
audio-specific dynamic batching hysteresis require workload/device evidence before
enabling such a policy.

The exact-history codec batches compatible states and reports actual subgroup
width. An outer mixed-age request list can contain singleton native groups;
concurrency alone is not useful tensor occupancy. CUDA graphs, device sampling,
precision changes, AR/codec CUDA stream overlap and reference-prefix KV reuse
remain separate profile-driven optimizations. No speed gain from these mechanisms
is claimed or enabled by the native batching flag.

Fleet placement/autoscaling configuration, shared persistence integration and
proxy failure qualification must be completed against the actual deployment
repository and intended GPU fleet. This checkout's runbook and load tooling do not
supply an operational Modal router or certify a production service. Publish
supported capacity only after all applicable release evidence above is retained.

## Trusted tenant scheduling

The API derives a fixed-size SHA256 scheduling key from the authenticated tenant
or principal namespace. This server-only key travels with runtime context and
internal durable jobs; public speech request fields cannot choose it. Legacy and
local requests without a key share one bucket, rather than receiving a new tenant
identity per request.

The default weighted-fair policy first preserves workload-class service weights,
then orders tenants by logical work already selected, with aging to bound catch-up
starvation. Prefill/AR tokens and codec frames are charged; these are explicit work
proxies, not claims of equal GPU milliseconds. Multiple requests from one tenant
share its account. Decode batch construction simulates these charges, and prefill
and codec selection reconsider order after each row, allowing healthy tenants
to share native batches. Existing first-audio/prefill/AR progress guards remain
independent of tenant selection.

An unexecuted, rejected quantum refunds its exact logical cost under the existing
session/plan fence. A row waiting for output credits also refunds its codec cost
before retrying, because it has not generated PCM. Executed failed work still
consumes service. Tenant metadata
is removed after its last request leaves; new tenants begin at the class service
floor, so adding requests does not create free historical credit. No raw tenant
identifier is emitted as a high-cardinality scheduling metric. Tests cover a
seven-request tenant competing with a one-request peer, mixed-tenant decode
batches, codec-frame refunds, stale sessions, cleanup and aged costly tenants.

## Implementation verification (2026-09-08)

Portable core regression run: 2,603 passed, nine ignored and two optional local
LFM weight-loading smoke tests excluded. The final output-credit retry/refund
refinement also passed its targeted regression. Full server suite: 416 passed.
Workspace Clippy with warnings denied, all-target compilation, changed-file
formatting, shell fixtures and all 14 Fish benchmark/report tests passed.
Native model fixtures include seven-row AR and mixed-age codec batches, row
reordering, RNG rollback, fresh-state abort, and independent stream delivery.
These results establish local regression coverage; they are not CUDA numerical,
whole-model audio quality, throughput, latency, or fleet reliability certification.

The CUDA device CI lane now explicitly runs the native batched codec, incremental
rollback and attention parity probes. Run it on each proposed provider/GPU profile
before opt-in rollout, followed by real-model and end-to-end load qualification.
