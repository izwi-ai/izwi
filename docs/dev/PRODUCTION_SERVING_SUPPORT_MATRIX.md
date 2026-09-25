# Production-serving support matrix

Status: exact-profile support statement, not a production-readiness certificate

This matrix applies to Izwi's separated gateway/supervisor/worker architecture.
It does not change the support contract of the existing `--role local` desktop
and CLI workflows. A source path, parser test, mock worker, or compilation result
is never promoted to real hardware or performance evidence.

## Current profile cells

| Topology | Gateway | Worker/backend | Model/artifact evidence | Current status |
|---|---|---|---|---|
| One machine, one device | One hardware-independent gateway | One supervised CPU worker | Generated tiny GGUF for the already-supported `LFM2.5-1.2B-Instruct-GGUF` route completed JSON and SSE through a subprocess worker | Development vertical slice proven; not a production model, capacity, soak, or performance certificate |
| One machine, multiple devices | One gateway | Multiple independent mock workers | Deterministic chat replicas prove compatible routing, independent request identity, worker-authoritative capacity, retry fencing, and failure isolation | Mock transport evidence only; no physical multi-device claim |
| One machine, Apple silicon | One gateway | Metal worker (metal-feature build, device identity `metal:<registryID>` verified at startup) | **Real Metal inference executed**: 120 gateway-routed generations through a metal worker (debug build, tiny synthetic fixture, `benchmarks/manifests/ds07-metal-baseline.json`); `backend_parity.rs` proves greedy Metal outputs equal CPU outputs on the fixture, including the DS1.5 committed-prefix attach leg (`benchmarks/manifests/ds15-metal-summary.json`); supervisor launch + exact device identity process-evidenced (`multi_lane_launch.rs`) | Development Metal lane proven on the fixture; not a production model, capacity, soak, or performance certificate |
| One machine, NVIDIA | One gateway | CUDA-capable worker binary via `--cuda-worker-binary` | The supervisor accepts, validates, and launches CUDA assignments with exact declared UUID and device-visibility environment (process test `multi_lane_launch.rs`); no CUDA toolchain or NVIDIA device was available in this work session | Supervision code-complete; CUDA execution not run — no CUDA or multi-GPU support claim |
| Multiple machines | One gateway | Versioned node/worker-pinned HTTPS approvals | Fleet topology fails closed without mTLS client identity, HTTPS, and exact operator-pinned node/worker IDs; the bundled worker remains loopback-only and no proxy enforcement, separate-machine handshake, or artifact path was exercised | Not supported as an operational profile |
| Multiple gateways | Two or more gateways | Any worker fleet | Worker admission stays authoritative, but registry, circuit state, tenant rate state, durable providers, and active-work quota ownership are not shared authorities | Not supported; Phase 8 gates remain open |

## Committed prefix reuse by model family and backend (DS1.6)

The catalog (`crates/izwi-core/src/catalog/prefix_reuse.rs`) records one cell
per model family per backend lane; serving surfaces default to catalog-auto,
which engages reuse only for cells with lane evidence. An explicit
`IZWI_ENABLE_PREFIX_CACHING=1` bypasses the table (salt required); an explicit
`0` is the kill switch. Auto degrades to Disabled with a recorded reason when
the runtime's page budget cannot fit a safe prefix reserve.

| Family | CPU | Metal | CUDA |
|---|---|---|---|
| Qwen3.8 chat (hybrid) | Supported — process parity (`backend_parity` prefix leg) | Supported — process parity (metal-feature leg) | Not enabled — no lane evidence |
| Qwen3 chat (dense) | Supported — DS1.2 fixture suite | Not enabled — contract declared, no lane parity | Not enabled — no lane evidence |
| Gemma3 chat (dense) | Supported — DS1.2 fixture suite | Not enabled — contract declared, no lane parity | Not enabled — no lane evidence |
| Voxtral (LM) | Not enabled — contract declared, no lane parity | Not enabled | Not enabled |
| Nemotron streaming ASR (`izwi-realtime-v1`) | Supported — real process evidence: the real worker binary with the real `Nemotron-3.5-ASR-Streaming-0.6B` artifact streamed `data/fox.wav` (36 frames, 24 kHz) through the WebSocket subprotocol → transcript "The quick brown fox jumps.", one Completed terminal, clean close (`real_cpu_realtime.rs`, ignored test, `IZWI_RT_EVIDENCE_BACKEND=cpu`) | Supported — same test with the metal-feature worker and device identity `metal:<registryID>` verified at startup (`IZWI_RT_EVIDENCE_BACKEND=metal`) | Not enabled — no lane evidence |
| Qwen3.5 chat (hybrid) | Excluded — hybrid reuse unproven (DS1.1 scope) | Excluded | Excluded |
| LFM2 chat (hybrid) | Excluded — hybrid reuse unproven | Excluded | Excluded |
| ASR / TTS / diarization / aligner families | Excluded — managed reuse is chat-task-gated | Excluded | Excluded |

This is an evidence statement, not a performance certificate: reuse on the
fixture lanes is counter-proven (counters + output equivalence), not
wall-clock-proven.

The only remotely advertised inference route in gateway mode is text-only
`POST /v1/chat/completions`, including its existing JSON and SSE response forms.
Every other route family remains explicitly local-only or absent as recorded in
the [route migration ledger](PRODUCTION_SERVING_DISCOVERY.md#route-migration-ledger).

## Single-gateway failure semantics

The current topology has one public gateway and makes no availability claim for
gateway loss.

- New requests cannot be admitted while the gateway is unavailable. A
  replacement gateway rebuilds its bounded registry from explicit approvals and
  requires fresh authenticated worker status before routing.
- A client connection lost with the gateway cannot prove whether an accepted
  worker invocation stopped. The worker retains its own execution capacity until
  completion or confirmed teardown; callers must treat the result as unknown and
  must not blindly retry it on another worker.
- Partially emitted SSE output is terminally interrupted, never reconstructed or
  transparently replayed. A replacement gateway cannot resume that stream.
- Gateway-local admission, circuit, metrics, tenant rate state, and accepted-work
  ownership reset with the process. Within one live gateway, accepted-work
  ownership survives public timeout/disconnect until exact worker teardown is
  proven; it is not a crash-persistent or shared fleet authority. This is one
  reason the multiple-gateway and strict fleet-quota profiles remain unsupported.
- Gateway mode intentionally owns no SQLite database, model runtime, accelerator,
  process-local session, or durable artifact provider. Durable/local workflows do
  not fail over through a replacement gateway because they are not advertised by
  this profile.
- Already-running workers remain independently owned by the supervisor. Worker
  health does not make the public endpoint available, and a missed gateway poll
  is not evidence that worker execution ended.

Recovery is replacement, not replay: start the same approved gateway build and
configuration, require fresh status for the exact deployment generation, restore
ingress only after readiness and bounded smoke checks, and report ambiguous
client attempts as interrupted.

## Evidence required to promote a cell

For each proposed cell, retain the exact revision, build features, model and
artifact revision, generation, backend/device identity, resource allocation,
test commands, raw results, and approver. At minimum:

- CPU needs a supported production-sized model, bounded concurrency,
  cancellation, restart, overload, soak, and measured resource evidence.
- Metal and CUDA each need serving-artifact compilation plus real execution on
  the named hardware; compilation alone is insufficient.
- Multiple physical devices need simultaneous independent invocations tied to
  distinct explicit assignments and one-worker failure without peer restart.
- Multiple machines need an authenticated TLS/mTLS handshake, remote artifact
  transport, partition/reconnect behavior, and stream-interruption evidence.
- Multiple gateways need shared or conservatively partitioned registry, quota,
  active-work, durable-state, artifact, and session ownership with outage tests.

Use the [release checklist](PRODUCTION_SERVING_RELEASE_CHECKLIST.md) to approve
an exact cell. Unavailable lanes must remain `not run` rather than `passed`.
