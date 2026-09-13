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
| One machine, Apple silicon | One gateway | Metal-capable worker source | No separated Metal worker execution or serving-specific compilation retained for this architecture; the supervisor executable rejects Metal assignments | Not supported as a supervised production-serving profile |
| One machine, NVIDIA | One gateway | CUDA-capable worker source | No CUDA toolchain or NVIDIA device was available in this work session | Not run; no CUDA or multi-GPU support claim |
| Multiple machines | One gateway | Authorized HTTPS worker endpoints | Client policy supports verified HTTPS, bounded private roots, and one shared mTLS client identity; the bundled worker remains loopback-only and no separate-machine handshake/artifact path was exercised | Not supported as an operational profile |
| Multiple gateways | Two or more gateways | Any worker fleet | Worker admission stays authoritative, but registry, circuit state, tenant rate state, durable providers, and active-work quota ownership are not shared authorities | Not supported; Phase 8 gates remain open |

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
