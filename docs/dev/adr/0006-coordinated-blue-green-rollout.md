# ADR 0006: Generation cutovers are supervisor-coordinated blue-green rollouts with a structural DINV-07 invariant

Status: Accepted
Date: 2026-09-27

## Context

DS6 of the distributed-serving plan turns a model-generation cutover into one
operator action with automatic abort. Before this phase, the runbook procedure
was a manual, restart-coupled sequence: drain, restart the supervisor on a new
node config, hand-edit the gateway's approvals file, restart the gateway, and
hope readiness held — with the operator as the only rollback mechanism and
nothing preventing a window in which two generations of one deployment were
admission-eligible at once (the DINV-07 hazard). The pre-existing
`--canary-worker-id` flag only ordered launches within one boot; it never
moved a gateway's routing between generations.

The building blocks already existed: the supervisor's launch/readiness/drain
machinery (`drain_and_stop`), the gateway's shared approvals file
(`IZWI_GATEWAY_SHARED_APPROVALS_PATH`) with TTL-cached versioned views, and
the plan's DINV-07 rule that two generations of one deployment must never both
be admission-eligible.

## Decision

1. **A rollout is a declarative, validated plan.** A TOML plan (schema
   version 1) names the target node config, the shared approvals path, the
   canary worker id, a soak window (`window_secs`, default 30, bounded at
   3600), and an abort grace (`abort_grace_secs`, default 30, bounded at
   300). The supervisor CLI is the only accepted interface
   (`--rollout-plan`/`--rollout-abort`/`--rollout-status`, mutually
   exclusive, and never combined with `--validate-only` or
   `--canary-worker-id`); the plan is fully validated — including
   `NodeConfig::validate` on the target — before any mutation, and its
   digest is bound into the persisted rollout state so a resume cannot
   apply a different plan.
2. **The shared approvals file stays the only supervisor↔gateway coupling.**
   The supervisor computes marker-free approval views and writes them
   atomically (temp file + rename); the gateway adopts changed views at
   runtime from the file it already polls. No new RPC, no coordinator role
   for the gateway, no in-memory side channel.
3. **Eligibility is a pure function of the view (DINV-07 structural).** One
   approved generation ⇒ eligible. Two generations ⇒ the lower generation is
   current — eligible until the first higher-generation worker is observed
   Ready, at which point the same table mutation flips the lower generation
   to draining-previous and the higher to eligible. Two generations can
   never both be eligible, and there is no zero-eligible instant, because
   the swap is triggered *by* a Ready observation of the successor. The
   gateway enforces the same rule in its deployment table (selection,
   readiness probe, and realtime stage pools all consult the
   eligible-generation gate), and a rejected view retains the previous
   state fail-closed.
4. **The state machine has an honest point of no return.** Phases:
   `launching_replacement` (canary first) → `window_open` (both generations
   approved; the gateway cut over when the successor observed Ready; the old
   generation never stopped serving) → `draining_old` (commit view written;
   the old generation stops admitting) → `committed`. Abort is legal in the
   reversible states and restores the pre-rollout approvals bytes
   byte-identically, waits the abort grace so the gateway's refresh stops
   routing to the replacements first, then drains them; the previous
   generation never stopped serving. Once the commit view is written,
   `draining_old` is irreversible — the workers' control-pipe EOF makes an
   in-place restore impossible — so the abort paths are deliberately absent
   there. Automatic aborts (canary/replacement readiness failure, a
   replacement exiting during the window, SIGUSR2, shutdown) keep the
   supervisor alive and supervising the old config: exiting would EOF-drain
   the old generation and manufacture exactly the outage the abort exists
   to prevent.
5. **Rollout state is persisted for resume, and a fresh start fails
   closed.** `runtime_directory/rollout-state.json` (atomic, bounded)
   carries the plan digest, replacement worker records, window deadline, and
   the pre-rollout approvals bytes used for restore. A fresh supervisor
   start that finds a non-terminal state file refuses to launch anything —
   the operator must resume with the same plan (`--rollout-plan`) or abort
   (`--rollout-abort`); a `committed` state additionally requires the
   committed target config's digest. Terminal states auto-clear.
6. **Evidence is process-level on both sides.** Supervisor tests prove
   canary-failure abort with byte-identical approvals restore, promotion
   through window → drain → commit with on-disk view assertions,
   replacement-exit abort, SIGKILL resume, and status/abort command
   semantics (`tests/rollout.rs`). The gateway rig proves continuous
   traffic across window → abort-restore and window → commit view
   transitions with zero failed requests and the draining predecessor
   receiving nothing after cutover (T37, the fleet-rig pattern with real
   gateway processes and mock workers on real TCP).

## Consequences

- The runbook's manual canary section is replaced by the rollout command;
  the manual approvals-file edit step is gone. The pre-existing
  `--canary-worker-id` keeps its narrower role: launch ordering within a
  single plain boot, never combined with a rollout plan.
- Rollback after a *committed* rollout is a new rollout plan that targets
  the previous configuration with a fresh generation — never an ABA
  identity reuse — because the old workers' drain cannot be undone.
- The coordinator is single-node by construction: one supervisor, one
  gateway, one shared approvals file. A fleet-wide cutover is a separate
  decision and is not claimed here.
- Degraded aborts are honest: if the approvals restore itself fails, the
  supervisor leaves the state resumable, keeps workers running, and reports
  the failure instead of pretending the previous view came back.
- The gateway treats the approvals file as live configuration: refresh
  adoption means an operator editing the file by hand now has runtime
  effect — the manual surgery the rollout exists to remove is still
  possible and remains the operator's responsibility to avoid.
