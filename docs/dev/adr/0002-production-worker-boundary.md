# ADR 0002: Initial production worker boundary

Status: accepted for the Phase 1 vertical slice

Date: 2026-09-12

## Context

Izwi's current server and inference runtime share one process and one concrete
`AppState`. The production-serving plan requires a hardware-independent gateway
and independently managed workers without replacing the existing inference
engine, scheduler, memory manager, public API encoders, or local workflows.

## Decision

1. Put private worker identities and messages in a new accelerator-free
   workspace crate. It may depend on serialization and HTTP primitives, but not
   on `izwi-core`, Candle, or a backend feature. Workers translate protocol
   backend/task values at their runtime boundary.
2. Use versioned HTTP with typed JSON control responses and incrementally parsed
   NDJSON invocation events. Parsing is bounded by request bytes, line bytes,
   total stream bytes, event count, and accumulated output bytes.
3. Keep `RuntimeService`, its `InferenceCoordinator`, loaded models, device
   context, and all execution/resource leases inside each real worker. The
   gateway's load view is advisory; final admission is an atomic worker action.
4. Treat worker incarnation and loaded-model generation as separate mandatory
   fences. A restart or reload cannot be hidden by a stable endpoint.
5. On timeout or client disconnect, request cancellation from the known worker.
   Do not infer teardown from a dropped HTTP future, do not release worker
   capacity at the gateway, and do not retry another worker after acceptance or
   any emitted output. EOF without a terminal event is an unknown/interrupted
   result.
6. Use a separate fail-closed service credential for the private interface.
   The gateway creates the caller context from its authenticated principal and
   never forwards public authorization or identity headers as authority.
7. Preserve the combined local server as an explicit compatibility profile
   during migration. A future gateway-only process/state path must omit
   `RuntimeService` construction entirely; a dummy CPU runtime is not an
   acceptable substitute.
8. Keep durable jobs, artifacts, history, saved voices, and live session
   ownership on their current local paths until their ledger entries are
   deliberately migrated. SQLite remains valid for one local deployment
   authority and is not represented as multi-gateway state.

## Consequences

- Phase 1 can be exercised on CPU-only CI using deterministic real-socket mock
  workers and without downloading model weights.
- The initial public slice can reuse the stable chat request/response mapping
  while changing only its execution seam.
- The protocol types cannot directly embed core engine objects; explicit
  conversion code is required in the real worker.
- Streaming chat, media routes, supervisor/device assignment, registry
  freshness, and fleet storage remain later reviewable changes rather than
  being implied by the first mock-worker success.
