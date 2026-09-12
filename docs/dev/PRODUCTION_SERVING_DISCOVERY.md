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

| Route family | Maturity | Current inference owner | State and artifact dependencies | Streaming/cancellation | Initial topology decision |
|---|---|---|---|---|---|
| `POST /v1/chat/completions` | Stable | Local profile: in-process `RuntimeService`; gateway profile: one selected private worker | Stateless at the gateway; model/tokenizer remain worker-owned | JSON and SSE cross the bounded worker transport; disconnect/timeout requests exact-attempt cancellation without retry | Enabled remotely for text-only chat in the single-node gateway profile. The registry supports independent replicas, but plaintext worker URLs are restricted to numeric loopback until authenticated TLS is implemented. Multimodal chat remains local-only. |
| `POST /v1/audio/speech` | Stable | In-process `RuntimeService` | Saved voices, reference audio, codecs, long-form spool files, speech history | Binary or SSE, model-specific timeout and bounded audio/event queues | Keep local-only until worker artifact/voice resolution and binary-stream contracts are explicit. |
| `POST /v1/audio/transcriptions` | Stable | In-process `RuntimeService` | Multipart/JSON audio, optional alignment model, timestamp/subtitle formatting | JSON or SSE with bounded terminal delivery | Keep local-only initially; migrate after bounded artifact/stream transfer is available. |
| `GET /v1/models` | Stable | Reads live local runtime/catalog | Catalog and loaded-model state are currently coupled | Non-streaming | Preserve locally; gateway needs a catalog separate from ready deployment status before migration. |
| Chat threads | Preview | In-process chat runtime | SQLite history and per-thread process locks | JSON/SSE | Keep local-only until durable ownership and remote execution are separated. |
| Realtime transcription | Preview | In-process realtime app/runtime | Process-local session, rolling audio, bounded frame/command/output queues | WebSocket; owner-bound state | Keep local-only until session affinity and private realtime protocol exist. |
| Realtime voice | Preview | In-process workflow coordinator | ASR/chat/TTS stage state, barge-in, voice persistence, session admission | WebSocket with multi-stage cancellation | Keep local-only; later bind each stateful stage to an explicit worker owner. |
| Jobs and speech history | Preview | In-process DB-polling batch worker plus runtime | Transactional SQLite jobs/stages/artifacts, leases, attempt tokens | Poll/SSE/cancel with durable state | Reuse stores and fencing later; do not route as synchronous HTTP work in Phase 1. |
| Media and saved voices | Preview | Gateway/server storage providers | Existing routes still use local/provider paths. A route-independent `ArtifactStore` foundation now maps tenant-scoped opaque IDs to durable `media_assets` rows and private provider keys. | HTTP upload/download | Keep routes local-only. The facade has bounded, integrity-checked reads and local/remote-like provider conformance tests, but no media or voice route has been migrated to it. |
| Model administration | Preview/operator | Direct local runtime and filesystem mutation | Downloads, model files, lifecycle locks | Progress plus load/unload/delete | Never expose through the inference worker surface; require separate operator authorization and resource-safe lifecycle control. |

## Requirement map

| Requirement | Existing code to reuse | Initial change |
|---|---|---|
| Backend-neutral private contract | Serializable worker concepts in `batch_runtime/types.rs`; realtime version/sequence conventions | Add a small accelerator-free protocol crate with explicit protocol/schema versions, identities, assignment, deployment/generation, status/capacity, request/events, cancellation, and stable errors. |
| Real HTTP separation | Axum/Tokio server stack and workspace Reqwest client | Implemented for chat over a bounded loopback-only private client. Remote network endpoints remain disabled until mutually authenticated TLS and operator-pinned approval land. |
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

Multimodal chat remains on the current explicit local execution path until its
artifact transport semantics have dedicated tests. This is a migration-ledger
restriction, not removal of the working local behavior.

## Phase 6 artifact foundation

The first artifact slice adds a route-independent `ArtifactStore` facade over
the existing `MediaStorageProvider` and SQLite-compatible `media_assets` table.
It issues opaque UUID references, keeps storage keys internal, records tenant
ownership in versioned server-authored metadata, validates size, content type,
SHA-256 and provider tenant metadata, and materializes reads through a fixed-size
streaming buffer with a configured hard byte limit. The local filesystem
provider and a deterministic remote-like provider run through the same
conformance contract. No schema migration or external service is required.

Deletion first tombstones the durable row and then removes the provider object.
If physical deletion fails, access stays denied and the error explicitly calls
for a later garbage-collection retry. Retention is recorded as ephemeral,
job-owned, or durable; automatic expiry and garbage collection are not yet
implemented. Existing media, saved-voice, speech-history, job-output, and
multimodal routes do not use this facade yet and remain restricted exactly as
listed above. Attempt-specific output publication, transactional winner
selection, remote artifact-service authentication, and fleet database ownership
remain Phase 6 work; this foundation alone does not enable a fleet route.
