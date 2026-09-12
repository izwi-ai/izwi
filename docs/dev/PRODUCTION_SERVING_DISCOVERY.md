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
| `POST /v1/chat/completions` | Stable | In-process `RuntimeService` through `app/chat.rs` | Stateless for this route; model/tokenizer remain runtime-owned | JSON and SSE; bounded 64-event application channel with explicit interruption handling | Migrate text-only non-streaming first. Keep gateway-side OpenAI parsing and response mapping. Preserve local and SSE paths until separately migrated. |
| `POST /v1/audio/speech` | Stable | In-process `RuntimeService` | Saved voices, reference audio, codecs, long-form spool files, speech history | Binary or SSE, model-specific timeout and bounded audio/event queues | Keep local-only until worker artifact/voice resolution and binary-stream contracts are explicit. |
| `POST /v1/audio/transcriptions` | Stable | In-process `RuntimeService` | Multipart/JSON audio, optional alignment model, timestamp/subtitle formatting | JSON or SSE with bounded terminal delivery | Keep local-only initially; migrate after bounded artifact/stream transfer is available. |
| `GET /v1/models` | Stable | Reads live local runtime/catalog | Catalog and loaded-model state are currently coupled | Non-streaming | Preserve locally; gateway needs a catalog separate from ready deployment status before migration. |
| Chat threads | Preview | In-process chat runtime | SQLite history and per-thread process locks | JSON/SSE | Keep local-only until durable ownership and remote execution are separated. |
| Realtime transcription | Preview | In-process realtime app/runtime | Process-local session, rolling audio, bounded frame/command/output queues | WebSocket; owner-bound state | Keep local-only until session affinity and private realtime protocol exist. |
| Realtime voice | Preview | In-process workflow coordinator | ASR/chat/TTS stage state, barge-in, voice persistence, session admission | WebSocket with multi-stage cancellation | Keep local-only; later bind each stateful stage to an explicit worker owner. |
| Jobs and speech history | Preview | In-process DB-polling batch worker plus runtime | Transactional SQLite jobs/stages/artifacts, leases, attempt tokens | Poll/SSE/cancel with durable state | Reuse stores and fencing later; do not route as synchronous HTTP work in Phase 1. |
| Media and saved voices | Preview | Gateway/server storage providers | Local/provider objects; some reads materialize complete values | HTTP upload/download | Keep local-only until tenant scoping and bounded remote artifact access are proven. |
| Model administration | Preview/operator | Direct local runtime and filesystem mutation | Downloads, model files, lifecycle locks | Progress plus load/unload/delete | Never expose through the inference worker surface; require separate operator authorization and resource-safe lifecycle control. |

## Requirement map

| Requirement | Existing code to reuse | Initial change |
|---|---|---|
| Backend-neutral private contract | Serializable worker concepts in `batch_runtime/types.rs`; realtime version/sequence conventions | Add a small accelerator-free protocol crate with explicit protocol/schema versions, identities, assignment, deployment/generation, status/capacity, request/events, cancellation, and stable errors. |
| Real HTTP separation | Axum/Tokio server stack and workspace Reqwest client | Add a real loopback TCP mock worker and a bounded client; do not treat an in-process trait call as transport evidence. |
| Worker-authoritative capacity | `InferenceCoordinator` and RAII execution/resource leases | Mock uses atomic fail-fast admission; real worker later adapts `RuntimeService` rather than adding a competing scheduler. |
| Bounded parsing and retention | Existing bounded route channels/audio parsers | Limit request bytes, NDJSON line bytes, total stream bytes, event count, client pool concurrency, and mock attempt retention. EOF without terminal is interrupted/unknown. |
| Cancellation safety | Runtime pending guards, exact abort, terminal quarantine, detached blocking ownership | Gateway requests cancellation but never releases worker capacity. Mock must retain its permit through simulated teardown after disconnect/timeout. |
| Incarnation/model generation | Batch worker `instance_id`; lifecycle model instance generation | Carry and validate both on every invocation, status, acknowledgement, lookup, and cancellation. |
| Service authentication | Enterprise auth/policy abstraction and request context | Add a separate scoped bearer credential with fixed-time verification. Never forward caller authorization headers. |
| Gateway-only operation | Runtime-independent liveness is reusable; current readiness/health are not | Phase 2 must introduce a process/state path that does not call `RuntimeService::new`, preload, warm, or start the local batch worker. |
| Real CPU worker | Existing `RuntimeService` with explicit CPU backend and current chat methods | Only after Phase 1 transport tests pass, host it in a separate worker process with a small already-supported fixture. |

## First vertical slice

The first migrated behavior is text-only, non-streaming
`POST /v1/chat/completions`. Public authentication, request validation,
OpenAI-compatible schema handling, tool parsing, and response encoding stay in
the gateway. The private request carries only normalized, typed chat input and a
gateway-attested caller context. The worker validates its own incarnation,
deployment generation, readiness, and capacity before acknowledging admission.

Multimodal chat and SSE remain on the current explicit local execution path
until their transport semantics have dedicated tests. This is a migration
ledger restriction, not removal of the working local behavior.
