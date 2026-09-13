# Production serving discovery and route migration ledger

Date: 2026-09-12

Baseline branch: `production-serving`

Baseline revision: `7804eee5a9445031a7668017502b9c24e6db4ba7`

Baseline subject: `Enable concurrent Fish Audio serving with reliable streaming and adaptive batching (#215)`

The working tree was clean before this work began. The checkout used Rust and
Cargo 1.93.0 on macOS arm64. The available machine is a 10-core, 16 GB Apple M1
Pro with Metal support. It has no NVIDIA device or CUDA toolchain, so no CUDA
compile, runtime, correctness, or performance result is claimed here.

## Baseline checks

| Check | Result |
|---|---|
| `cargo test -p izwi-core --lib` | Passed: 2,621 tests, 0 failed. |
| `cargo test -p izwi-server --lib` | Passed: 450 tests, 0 failed. |
| `cargo test --locked -p izwi-hooks` | Passed: 4 tests, 0 failed. |
| `cargo tree --locked -p izwi-server --depth 2` | Passed; confirms the current server transitively owns `izwi-core`, Candle, and backend dependencies. |
| `cargo test --locked -p izwi-server api::openai::compat_harness -- --test-threads=1` | Did not reach tests in a parallel follow-up build: compilation stopped with `No space left on device`. This is an environment failure, not a test assertion failure. |

The successful core and server suites were run before the build volume filled.
Cargo build artifacts may be cleaned and rebuilt; no source or durable product
data needs to be removed.

## Verified current ownership

- `crates/izwi-server/src/lib.rs` always constructs `RuntimeService`, may preload
  and warm models, and starts the in-process durable batch worker. The current
  server binary is therefore a combined API and execution process.
- `crates/izwi-server/src/state.rs` requires `Arc<RuntimeService>` and also owns
  request admission, process-local response/session maps, durable stores, media
  ingestion, and batch-worker health. A gateway-only process must not construct
  a placeholder `RuntimeService` because its constructor selects/probes a device
  and initializes execution infrastructure.
- `crates/izwi-core/src/runtime/service.rs` is the reusable worker-side boundary.
  It already owns model lifecycle, model generations, coordinator admission,
  runtime execution, cancellation cleanup, resource accounting, and telemetry.
- `crates/izwi-core/src/runtime/coordinator.rs` supplies bounded, atomic,
  fail-fast worker-local admission. Its leases and the runtime's pending-request
  guards retain capacity until physical completion or confirmed cleanup.
- `crates/izwi-core/src/backends` already represents CPU, Metal, and CUDA and
  rejects an explicit backend mismatch at `RuntimeService` construction. CUDA's
  process-local ordinal environment is suitable only when set by a supervisor
  before child startup.
- `crates/izwi-server/src/batch_runtime` already has useful typed precedent for
  worker instance IDs, resources, capabilities, heartbeats, leases, and fenced
  attempts. It is durable database job machinery, not the synchronous private
  inference transport, so Phase 1 will reuse concepts without making a database
  an HTTP dispatch dependency.
- Public request identity is constructed by the gateway middleware in
  `api/request_context.rs`. Community hooks currently authenticate a local
  anonymous principal. Private worker authentication must be a separate,
  fail-closed service credential and must receive a gateway-attested caller
  identity instead of forwarded public identity headers.
- Existing chat, transcription, TTS, and realtime producers use bounded
  channels and contain useful cancellation/backpressure behavior. There is no
  existing bounded incremental NDJSON parser.

## Route migration ledger

Gateway mode mounts only text-only `POST /v1/chat/completions`. Every other
family in this ledger is absent (404) in gateway mode and remains available only
through the existing local/desktop profile where documented; a shared database
or media-provider option alone is not evidence of shared ownership correctness.

| Route family | Maturity | Current inference owner | State and artifact dependencies | Streaming/cancellation | Initial topology decision |
|---|---|---|---|---|---|
| `POST /v1/chat/completions` | Stable | Local profile: in-process `RuntimeService`; gateway profile: one selected private worker | Stateless at the gateway; model/tokenizer remain worker-owned | JSON and SSE cross the bounded worker transport; disconnect/timeout requests exact-attempt cancellation without retry | Enabled remotely for text-only chat in the single-node gateway profile. The registry supports independent replicas, but plaintext worker URLs are restricted to numeric loopback until authenticated TLS is implemented. Multimodal chat remains local-only. |
| `POST /v1/audio/speech` | Stable | In-process `RuntimeService` | Saved voices, reference audio, codecs, long-form spool files, speech history | Binary or SSE, model-specific timeout and bounded audio/event queues | Keep local-only until worker artifact/voice resolution and binary-stream contracts are explicit. |
| `POST /v1/audio/transcriptions` | Stable | In-process `RuntimeService` | Multipart/JSON audio, optional alignment model, timestamp/subtitle formatting | JSON or SSE with bounded terminal delivery | Keep local-only initially; migrate after bounded artifact/stream transfer is available. |
| `GET /v1/models` | Stable | Reads live local runtime/catalog | Catalog and loaded-model state are currently coupled | Non-streaming | Preserve locally; gateway needs a catalog separate from ready deployment status before migration. |
| Chat threads and multimodal history | Preview | In-process chat runtime | Durable rows have no tenant column; reads are unpaged; per-thread read/generate/persist locks are process-local. Media content parts retain URL/path/data-like sources rather than tenant artifacts. | JSON/SSE | Keep local-only until tenant-filtered bounded history, shared fenced turn ownership, and opaque worker-readable media artifacts exist. |
| OpenAI Responses | Preview | In-process runtime | Bounded process-local response map with no tenant owner; terminal records appear only after execution; `store: false` is intentionally ephemeral | JSON/SSE | Keep local-only. Restart/eviction loses records and `/cancel` does not own or fence a discoverable in-flight attempt. |
| Agent sessions and chat workflows | Preview | In-process agent/chat coordinators | Bounded process-local session metadata plus a separate durable chat thread; neither has shared owner/turn fencing or tenant-scoped workflow mutation | JSON/SSE | Keep local-only. Restart can leave an orphan durable thread; tool side effects have no fleet idempotency contract. |
| Realtime transcription | Preview | In-process realtime app/runtime | Process-local rolling state and bounded queues; only correlation identity survives upgrade | WebSocket; owner-bound state | Keep local-only until tenant session ownership, gateway affinity, and a private realtime protocol exist. |
| Realtime voice | Preview | In-process workflow coordinator | ASR/chat/TTS state, barge-in, streaming input, active turn, and agent session are process-owned | WebSocket with multi-stage cancellation | Keep local-only; bind every stage to worker incarnation/deployment generation and explicitly interrupt on owner loss rather than implying migration. |
| Voice-session records | Preview | Local voice workflow/store | Durable session/turn rows lack tenant, owner incarnation, and deployment generation; live state remains process-local | REST plus realtime owner | Preserve locally. Durable rows do not make a live session migratable or fleet-owned. |
| Jobs and speech history | Preview | In-process DB-polling batch worker plus runtime | Jobs/stages/artifacts have transactional claims and attempt fencing. Text-only TTS has atomic acceptance and tenant-scoped idempotency. Durable Fish PCM replay now uses tenant-scoped opaque artifacts with atomic attempt publication, while the final WAV, history rows, references, and other speech routes retain legacy provider paths and incomplete tenant predicates. | Poll/SSE/cancel with durable state | Preserve locally. Opaque replay storage alone does not establish fleet ownership; require tenant predicates, final-artifact migration, and deployed shared-provider conformance before fleet exposure. |
| Media uploads | Preview | Local/server media provider | Public routes address provider storage keys directly; `media_assets` metadata does not enforce tenant ownership and the route does not use `ArtifactStore` | HTTP upload/download | Keep local-only until upload/download use authorized opaque artifacts and never expose provider keys. |
| Saved voices | Preview | Local saved-voice store and media provider | Rows contain provider paths and no tenant owner; `local_owner` describes use class, not authenticated caller ownership; blob/row publication is not transactional | HTTP CRUD plus TTS reuse | Keep local-only until tenant-filtered metadata, opaque artifacts, crash-safe publication/deletion, and worker-side authorized resolution exist. |
| Studio projects and rendering | Preview | In-process Studio workflow and TTS runtime | Projects, segments, snapshots, and render metadata are durable but unscoped; rendering attaches speech records separately and export accumulates segment audio in memory | REST plus background render metadata | Keep local-only until tenant ownership, fenced render jobs, transactional result publication, shared voice/audio artifacts, and bounded export exist. |
| Model administration | Preview/operator | Direct local runtime and filesystem mutation | Downloads, model files, lifecycle locks | Progress plus load/unload/delete | Never expose through the inference worker surface; require separate operator authorization and resource-safe lifecycle control. |

## Requirement map

| Requirement | Existing code to reuse | Initial change |
|---|---|---|
| Backend-neutral private contract | Serializable worker concepts in `batch_runtime/types.rs`; realtime version/sequence conventions | Add a small accelerator-free protocol crate with explicit protocol/schema versions, identities, assignment, deployment/generation, status/capacity, request/events, cancellation, and stable errors. |
| Real HTTP separation | Axum/Tokio server stack and workspace Reqwest client | Implemented for chat over a bounded loopback-only private client. Explicit fleet policy now requires HTTPS, mTLS client identity, and versioned node/worker-pinned approvals; operational remote support remains disabled until TLS termination enforcement and separate-machine evidence land. |
| Worker-authoritative capacity | `InferenceCoordinator` and RAII execution/resource leases | Mock uses atomic fail-fast admission; real worker later adapts `RuntimeService` rather than adding a competing scheduler. |
| Bounded parsing and retention | Existing bounded route channels/audio parsers | Limit request bytes, NDJSON line bytes, total stream bytes, event count, client pool concurrency, and mock attempt retention. EOF without terminal is interrupted/unknown. |
| Cancellation safety | Runtime pending guards, exact abort, terminal quarantine, detached blocking ownership | Gateway requests cancellation but never releases worker capacity. Mock must retain its permit through simulated teardown after disconnect/timeout. |
| Incarnation/model generation | Batch worker `instance_id`; lifecycle model instance generation | Carry and validate both on every invocation, status, acknowledgement, lookup, and cancellation. |
| Service authentication | Enterprise auth/policy abstraction and request context | Add a separate scoped bearer credential with fixed-time verification. Never forward caller authorization headers. |
| Gateway-only operation | Runtime-independent liveness is reusable; current readiness/health are not | Phase 2 must introduce a process/state path that does not call `RuntimeService::new`, preload, warm, or start the local batch worker. |
| Real CPU worker | Existing `RuntimeService` with explicit CPU backend and current chat methods | Only after Phase 1 transport tests pass, host it in a separate worker process with a small already-supported fixture. |

## First vertical slice

The first migrated behavior was text-only `POST /v1/chat/completions`.
Non-streaming JSON and SSE now retain public OpenAI parsing/encoding in the
gateway and cross the same private worker boundary. The private request carries
only normalized, typed chat input and a gateway-attested caller context. The
worker validates its own incarnation, deployment generation, readiness, and
capacity before acknowledging admission. Registry routing selects once and does
not replay an uncertain or partially streamed invocation on another worker.
Gateway chat applies separate bounded admission/header, first-output,
inter-output idle, and total invocation deadlines. Only non-empty text output
advances the semantic progress clock; raw fragments, usage metadata, and empty
deltas cannot keep an invocation alive. Its private parser accepts at most a
1 MiB line, 16 MiB total, and 8,192 events, coherently covering the advertised
4,096-token/512 KiB route ceiling while remaining bounded.

Multimodal chat remains on the current explicit local execution path until its
artifact transport semantics have dedicated tests. This is a migration-ledger
restriction, not removal of the working local behavior.

## Phase 6 artifact foundation

The artifact foundation adds a route-independent `ArtifactStore` facade over
the existing `MediaStorageProvider` and SQLite-compatible `media_assets` table.
It issues opaque UUID references, keeps storage keys internal, records tenant
ownership in versioned server-authored metadata, validates size, content type,
SHA-256 and provider tenant metadata, and materializes reads through a fixed-size
streaming buffer with a configured hard byte limit. The local filesystem
provider and a deterministic remote-like provider run through the same
conformance contract. Standalone mode requires no external service.

Deletion atomically tombstones the durable row and inserts a bounded deletion
intent before attempting the provider. Success and provider `NotFound` complete
the intent; failures survive restart with bounded errors, exponential backoff,
per-call deadlines, and fixed-size maintenance batches. Local batch-worker
maintenance processes only explicitly tombstoned facade objects, never a global
"unreferenced" sweep or a lease-expiry guess. Retention class is recorded, but
automatic expiry is not implemented. Opaque `ArtifactStore` creation now commits
a bounded provider-write reservation before sending bytes. That reservation
retains an independently versioned, bounded copy of the exact typed namespace,
record identity, filename, content type, and provider metadata. The opt-in
versioned provider operation is therefore recoverable by its original request
even when a crash prevents its returned key reaching the database; metadata
publication atomically consumes the reservation. Historical rows without an
envelope are recoverable only for the provable `artifact-store` request shape;
unknown legacy shapes remain fenced for operator remediation. Provider
`NotFound` completes recovery only when the object is absent and a future commit
for the expired write ID is fenced.

Durable Fish PCM replay is the first route-state adopter: each new chunk is a
tenant-scoped opaque object, and its publication marker, media row, exact active
attempt reference, and reservation consumption commit together. Runtime rows do
not expose provider keys. Existing raw-key journals remain readable and
deletable for local upgrade compatibility. Speech-history rows can now retain an
exact tenant plus opaque artifact ID and stream that artifact with terminal size
and SHA-256 verification; legacy path rows remain readable and deletable for
local upgrade compatibility. The unique artifact-reference index prevents two
history rows from claiming the same object. No public speech producer writes the
opaque history form yet, and opaque completion, replacement, and deletion stay
fail-closed until they share one transactional settlement path. The final speech
WAV, saved voices, references, other job outputs, Studio, media routes, and
multimodal inputs still retain legacy provider paths. Their tenant and fleet
ownership gates remain Phase 6 work; this partial adoption does not enable a new
gateway route.

Reserved-write protocol v1 also has a separately advertised file capability.
It streams a finalized local file through fixed-size buffers while enforcing the
same pre-recorded write ID, length, digest, deadline, idempotency, and recovery
fence as byte writes. Bytes-only reserved-write providers remain valid for PCM
chunks and are not silently accepted for final files. The local provider passes
this contract. Artifact streaming reads and the read-only speech-history opaque-
reference consumer are implemented. Opaque responses omit `Content-Length` so a
terminal length or digest failure remains observable, although already-read
chunks are not authenticated independently. Final Fish WAV publication and
lifecycle settlement remain separate adoption gates, so this primitive alone
exposes no new route.
