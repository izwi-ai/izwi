# ADR 0007: Autoscaling is supervisor-managed over pre-declared standby replicas

Status: Accepted
Date: 2026-09-27

## Context

DS7 of the distributed-serving plan asks the supervisor to become a capacity
manager within explicit bounds (DINV-08): scale up on sustained queue depth,
scale down after a stabilized idle window with drain, every decision
respecting bounds, the node resource ledger, and hysteresis — and off by
default. The building blocks existed: the worker status snapshot already
carries the capacity signals (`queued_invocations`,
`active_invocations`, `reserved_sessions`), the supervisor already owns the
launch/readiness/drain machinery (`drain_and_stop`), and DS6 established the
shared approvals file with runtime view adoption as the only
supervisor↔gateway coupling.

Two prior patterns constrain the design. The node config is fail-closed and
strictly validated (`deny_unknown_fields`, aggregate budget checks, device
exclusivity), and DS6 required rollout replacements to be pre-declared in a
full target node config rather than fabricated at runtime. DINV-08's ledger
requirement has to interact with that: an autoscaler that fabricates workers
(templated ids, ephemeral ports, generated assignments) would move resource
authorization out of config validation and reintroduce exactly the
launch-into-OOM hazard DS7.3 forbids.

## Decision

1. **Pre-declared standby replicas, not runtime worker fabrication.** All
   `max_workers` workers of an autoscaled deployment are fully declared in
   the node TOML (own worker id, bind, assignment, budgets). At startup the
   supervisor launches exactly the first `min_workers` (config order, the
   "core set"); the rest are standbys that only a scale-up decision starts.
   Because `NodeConfig::validate` already checks the sum over *all declared*
   workers against the node budget and device exclusivity, every reachable
   scale-out state fits the ledger by construction — the config check is the
   primary budget guarantee, and the runtime ledger is a structural guard.
2. **A single node-level `[autoscaling]` block carries per-deployment
   policies.** `min_workers`/`max_workers` (1–64; `max_workers` must equal
   the declared count for that deployment, so no declared worker can exist
   that can never launch), `scale_up_queue_depth` (1–100000),
   `scale_up_sustained_polls` (1–3600), and
   `scale_down_stabilization_window_ms` (1000–86400000). Absent block = the
   exact static behavior of before. The schema version stays 2.
3. **Signals and zero-work observation come from the supervisor's own status
   polls** — the first periodic status polling of running workers. Scale-up:
   a running worker at or above the queue-depth threshold for N consecutive
   evaluations. Scale-down: a candidate with zero queued, zero active, and
   zero reserved sessions (realtime included) for the whole stabilization
   window. A deployment is never evaluated while any of its running workers
   is unobservable (launch in flight, restart pending, failed poll) — no
   decisions on partial observability.
4. **Scale events ride the existing lifecycle.** Scale-up launches the
   standby through the standard launch/readiness path and publishes its v1
   pinned approvals line only after readiness (an `Admitting` phase makes
   the publish retry explicit and observable). Scale-down removes the line
   first (the gateway stops routing after its view refresh), then closes the
   control pipe (the worker stops admission immediately, covering the
   gateway TTL gap), waits for zero work, and only then calls
   `drain_and_stop`. The supervisor never TERMs work that has not reached
   zero; a hung worker is bounded by its own worker-side shutdown policy or
   process exit, either of which completes the scale-down.
5. **Hysteresis is per-deployment and shared by both directions:** no scale
   event may start within `scale_down_stabilization_window_ms` of the
   previous scale event (recorded at initiation — the capacity posture
   changes when the line is removed). Scale-down candidates exclude the core
   set, so the deployment can always return to its static posture; among
   eligible candidates the choice is deterministic (config order).
6. **The ledger reserves supervised-slot budgets and rejects overcommit with
   actionable diagnostics** (`ResourceLedger`): running and restart-pending
   slots alike hold their reservations, so a crash-looping worker's budget
   is never handed to a scale-up; host memory, CPU threads, Metal/CUDA
   device exclusivity are checked per candidate. Rejections are logged and
   never launched.
7. **Autoscaling and coordinated rollout are mutually exclusive in one
   supervisor run** (both own the shared approvals view; the check precedes
   rollout state staging). Scale state is intentionally in-memory: a
   supervisor restart relaunches the declared min set and reconciles the
   view to it, so a crashed supervisor never leaves phantom approved
   capacity behind.
8. **View ownership is precise.** The supervisor owns exactly the v1 pinned
   lines of its own node's autoscaled deployments (added/removed by pinned
   `(node_id, worker_id)` identity, writes atomic under a sidecar lock);
   every unrelated line passes through verbatim; a standalone-form line
   colliding with an autoscaled endpoint fails the write closed.

## Consequences

- The operator declares capacity once (all replicas) and bounds it once
  (min/max); the supervisor cannot create workers the config did not
  authorize, keeping DINV-08's "impossible scale-ups are rejected" property
  structural rather than best-effort.
- Scale-up latency includes model load per replica, as with any cold start;
  the sustained-polls threshold and evaluation interval are the operator's
  tuning surface for that trade-off. Metal/CUDA deployments scale within
  device exclusivity (one worker per exclusive device), so multi-replica
  scale-out on those lanes requires distinct declared devices.
- The gateway needs no changes: it already adopts changed views at runtime,
  including same-generation replica adds and removes (proven by the DS6
  T37 rig and the registry capacity fixes that came out of it).
- Test evidence (T38): policy unit tests (signals → decisions, hysteresis,
  bounds, ledger overcommit, view editing) plus a real-process rig —
  sustained queue depth scales 1→2 with the approvals line published after
  readiness, the idle window scales 2→1 with drain observed and the core
  set untouched, the disabled configuration launches everything and never
  touches the view, and a rollout plan is refused while autoscaling is on.
