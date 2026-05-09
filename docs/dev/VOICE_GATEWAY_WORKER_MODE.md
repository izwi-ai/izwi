# Voice Gateway And Worker Mode

Status: Proposed

Date: 2026-05-09

## Purpose

Izwi should stay a production-grade single-node Voice AI Runtime by default.
Gateway or worker mode is optional infrastructure for deployments that need
multi-process or multi-node routing. It must build on the local runtime seams:
typed requests, voice sessions, adapter metadata, model lifecycle snapshots,
stream policies, and voice metrics.

## When To Use It

Use gateway/worker mode when at least one of these is true:

- Multiple model families need independent process lifecycles.
- A deployment needs health-aware routing across worker processes.
- A deployment needs session affinity for realtime voice sessions.
- Model residency policy is different per worker class.
- One machine cannot satisfy the required concurrent voice sessions.

Do not use it just to organize local code. The in-process runtime remains the
primary developer and desktop path.

## Proposed Shape

```text
Transport APIs
  -> Voice Gateway
      -> session registry
      -> admission control
      -> health-aware routing
      -> model/capability routing
      -> per-session affinity
  -> Runtime Worker
      -> VoiceSession
      -> Inference Broker
      -> adapters, residency, executor
```

## Gateway Responsibilities

- Accept protocol requests from HTTP, WebSocket, WebRTC, telephony, desktop, or
  future RPC frontends.
- Resolve public model aliases and defaults into capability requirements.
- Choose a worker by capability, model residency, health, load, and session
  affinity.
- Route all events for an active realtime session to the same worker unless the
  session is explicitly migrated.
- Enforce global admission control, tenant policy, and circuit breakers.
- Aggregate worker health, model lifecycle, stream pressure, and voice metrics.

## Worker Responsibilities

- Own local `VoiceSession` state.
- Own model adapters and loaded handles.
- Own executor/scheduler/device resources.
- Emit runtime metrics and lifecycle events.
- Reject work that violates local capability or residency constraints.

## Minimal Control Messages

- `RegisterWorker`
- `WorkerHeartbeat`
- `WorkerDraining`
- `ListCapabilities`
- `ListModelLifecycle`
- `OpenVoiceSession`
- `VoiceSessionEvent`
- `CloseVoiceSession`
- `SubmitInference`
- `CancelRequest`

## Readiness Contract

A worker is routable only when:

- It responds to heartbeat checks.
- Its runtime reports no unrecovered worker panic.
- It can satisfy the requested capability through the adapter registry.
- Required model artifacts are available or allowed to be loaded on demand.
- Residency budget policy permits the requested model.

## Non-Goals

- No distributed KV cache in the first gateway mode.
- No prefill/decode disaggregation until local voice latency metrics justify it.
- No gateway dependency for desktop or single-node server use.
- No backend expansion outside the support matrix.

## Migration Requirements

- Keep all existing local APIs working without gateway mode enabled.
- Add worker-mode conformance tests before adding remote transport code.
- Preserve voice metric names from `voice_metric_catalog()`.
- Preserve model lifecycle terms from `ModelLifecycleSnapshot`.
- Keep session events compatible with `VoiceSession` state transitions.
