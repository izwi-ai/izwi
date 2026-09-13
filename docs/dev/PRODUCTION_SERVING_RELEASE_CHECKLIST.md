# Production-serving migration, release checklist, and risk register

Status: release gate template, not a production-readiness certificate

Use this document with the
[single-node operations runbook](PRODUCTION_SERVING_SINGLE_NODE_RUNBOOK.md),
[route migration ledger](PRODUCTION_SERVING_DISCOVERY.md),
[packaging evidence](PRODUCTION_SERVING_PACKAGING.md), and
[architecture decision](adr/0002-production-worker-boundary.md). A release is
approved by an operator for one exact profile, backend, model, artifact revision,
and hardware class. Evidence from another cell does not carry over.

## Supported migration boundary

The current remote public surface is `POST /v1/chat/completions`, including its
JSON and SSE forms. The gateway must return 404 for local-only model, speech,
transcription, Responses, administrative, and voice-session routes until each
route has a typed worker transport and its state/artifact dependencies satisfy
the route ledger.

The safe initial migration shape is one trusted CPU node:

1. Keep the existing local/desktop service available for every unmigrated route.
2. Install one identical Izwi revision for gateway, supervisor, and worker.
3. Validate a schema-v2 CPU node file without launching children.
4. Start the supervisor and wait for the exact deployment generation to become
   ready before starting the gateway.
5. Send only chat traffic selected for this deployment to the gateway.
6. Keep the old local route available as an explicit rollback destination; do
   not silently fall back per request after uncertain worker acceptance.

Do not migrate filesystem-backed URLs, process-local sessions, local chat
history, voice rows, workflows, or administrative mutation routes merely by
making their handlers reachable. Their ownership remains as recorded in the
route ledger.

## Rollback procedure

Rollback changes traffic ownership; it does not replay uncertain requests.

1. Stop new gateway admission and wait for the documented drain deadline.
2. Inspect worker active-invocation state. A client disconnect or expired public
   timeout is not proof of worker teardown.
3. Route new requests back to the previous explicitly supported endpoint.
4. Leave the supervisor running until accepted work completes or exact attempts
   confirm cancellation/teardown, then drain workers.
5. Preserve logs and the exact node/deployment configuration for incident
   review. Do not reuse an old model generation under a new artifact revision.
6. Revert binaries and configuration together. Re-run validate-only before
   starting the prior generation.

Durable jobs use attempt-token and lease fencing. Never edit their SQLite rows
by hand as a rollback mechanism. Synchronous or partially streamed chat is not
durably replayable and must be reported as failed/interrupted when its outcome
is uncertain.

## Release checklist

Copy this section into the release record and attach exact command output.

### Identity and source

- [ ] Record commit, branch/tag, dirty status, Rust toolchain, build profile, OS,
  architecture, and package hashes.
- [ ] Record gateway, supervisor, and worker versions and prove they came from
  the same approved build.
- [ ] Record model alias, immutable artifact revision, model generation,
  precision/representation, tokenizer revision, and task capability.
- [ ] Record the exact supported profile: local desktop, one-node CPU,
  one-node Metal, one-node CUDA, multi-device node, or multi-machine.

### Configuration and security

- [ ] Parse the schema-v2 node configuration with `--validate-only`; retain the
  bounded redacted output.
- [ ] Verify resource budgets against the service allocation and physical host.
- [ ] Verify every worker has an explicit device assignment and unique endpoint;
  never repair an unsupported Metal/CUDA assignment by changing it to CPU.
- [ ] Verify public API and private worker credentials are secret references,
  absent from argv, config artifacts, diagnostics, and logs.
- [ ] Keep worker HTTP on numeric loopback. For an approved remote profile,
  require certificate-verified HTTPS through a trusted TLS/mTLS termination
  layer and record its trust configuration.
- [ ] Terminate public TLS before any non-loopback gateway bind and explicitly
  acknowledge the trusted ingress setting.
- [ ] Keep CORS disabled or use a reviewed explicit origin allowlist.
- [ ] Prove administrative and unmigrated routes are absent from the gateway.

### Contracts and correctness

- [ ] Run serving protocol, client, worker, supervisor, gateway, registry, and
  packaging contract suites serially on constrained builders.
- [ ] Prove version/auth failures, malformed and oversized input, model/backend/
  generation incompatibility, authoritative capacity rejection, timeout, and
  cancellation behavior.
- [ ] Prove one alternate dispatch occurs only after non-acceptance is known,
  keeps the request ID, changes the attempt ID, consumes the original deadline,
  and never retries after uncertain acceptance or partial output.
- [ ] Prove worker capacity survives disconnect/timeout until real completion or
  confirmed teardown.
- [ ] Prove drain rejects new work while accepted work retains ownership.
- [ ] Reopen the durable store and prove exact-attempt recovery/fencing for every
  enabled durable job route.
- [ ] Run public compatibility fixtures for JSON and streaming output.

### Operations and evidence

- [ ] Capture readiness transitions for load, warm-up, serving, drain, failure,
  restart, quarantine, and generation replacement.
- [ ] Verify restart-window limits, bounded backoff/jitter, parent-loss handling,
  and stale-incarnation fencing.
- [ ] Exercise overload at gateway and worker boundaries; verify queues, buffers,
  uploads, attempt records, tenant state, and maintenance batches remain bounded.
- [ ] Verify privacy-safe logs contain correlation and bounded classification
  fields but no prompts, audio, bearer tokens, arbitrary tenant strings, or
  secret certificate/key bytes.
- [ ] Retain mock-worker, real CPU, accelerator compilation, real accelerator,
  multi-device, soak, and performance evidence as distinct records.
- [ ] Run `cargo clean` after the serialized build/test batch on constrained
  hosts and record the space reclaimed.

### Hardware and performance gates

- [ ] CPU: execute a supported model through separate gateway and worker
  processes on the target CPU/memory allocation.
- [ ] Metal: compile the exact serving artifacts and execute on the target Apple
  hardware. A device visible to an unrelated test is insufficient.
- [ ] CUDA: compile with the release CUDA toolchain and execute on the target GPU,
  driver, and memory allocation.
- [ ] Multi-device: prove distinct workers execute independent concurrent work on
  their assigned devices without fallback or shared-device collision.
- [ ] Performance: predeclare workload and latency/throughput/resource budgets;
  retain raw warm/cold results and variability. Do not extrapolate user counts.
- [ ] Mark unavailable hardware lanes `not run`; never convert them to passed.

### Approval

- [ ] Review all open high-severity risks below for the selected profile.
- [ ] Name the approver, approval time, allowed profile, rollback owner, and
  monitoring window.
- [ ] State unsupported routes/backends explicitly in release notes.
- [ ] Do not use “production-ready” unless every required gate for the exact
  claimed profile has evidence and no blocking risk remains.

## Evidence record template

```text
revision/tag:
dirty status:
build command and features:
test command:
result (passed/failed/not run):
profile and topology:
model/artifact/generation:
backend/device/precision/representation:
OS/driver/toolchain:
CPU/RAM/device-memory allocation:
request distribution and sample count:
warm/cold state:
latency/throughput/resource results:
raw log/artifact location and digest:
known deviations:
reviewer/date:
```

## Outstanding risk register

| ID | Severity | Applies to | Current control | Release effect / exit gate |
|---|---|---|---|---|
| PSR-001 | High | Remote/multi-machine | Worker clients require verified HTTPS; bundled workers bind loopback | No native remote profile until TLS/mTLS termination and a separate-machine contract test are approved |
| PSR-002 | High | Metal/CUDA supervisor | Schema and workers carry explicit assignments; supervisor CLI rejects unsupported lanes without CPU fallback | Do not release supervisor-managed Metal/CUDA until trusted inventory/launch integration and target hardware tests pass |
| PSR-003 | High | Multiple gateways | Registry, circuit state, and tenant rate limits are process-local | No strict fleet quota/HA claim until Phase 8 shared or conservatively partitioned authorities pass restart/partition tests |
| PSR-004 | High | Unmigrated APIs | Gateway exposes only chat and returns 404 for representative local routes | Do not advertise models, audio, Responses, workflows, history, admin, or live-session APIs remotely before their ledger gates pass |
| PSR-005 | High | Durable idempotency | Worker attempt IDs suppress bounded local duplicates; job attempt tokens fence writers | Do not promise public durable replay until tenant-scoped reservation/commit/replay, digest conflicts, expiry, and route integration pass |
| PSR-006 | Medium | Durable cancellation UI/projections | Store retains requested/stopping ownership and fences results until executor teardown | Reconcile route projections after asynchronous terminal cancellation before claiming complete cross-route cancellation state |
| PSR-007 | Medium | Tenant concurrency | Gateway has bounded global admission and process-local request rates; worker admission is authoritative | Do not claim per-tenant active-work limits until permits reconcile against terminal worker state or confirmed teardown |
| PSR-008 | Medium | Observability | Structured request logs and existing local runtime metrics exist; runbook describes checks | Complete bounded gateway/router/worker metrics and private export before unattended production operation |
| PSR-009 | Medium | Artifact lifecycle | Opaque tenant-scoped store validates size/type/digest and tombstones deletion | Add retention/orphan/losing-attempt cleanup and remote artifact transport before advertising artifact-dependent remote routes |
| PSR-010 | Medium | Credential rotation | Public gateway has one API key/tenant and gateways use one private worker credential pair | Document coordinated rotation; add multi-key/per-worker identity where the deployment requires independent revocation |
| PSR-011 | High | Hardware/performance claims | Source, parser, mock, and tiny CPU evidence are separated | No Metal, CUDA, multi-GPU, soak, capacity, latency, or user-count claim without exact retained target evidence |
| PSR-012 | Medium | Gateway availability | One gateway can route multiple independent workers | Document single-gateway interruption; Phase 8 crash/replacement testing gates HA claims |

Review this register whenever a route, topology, backend, model, storage provider,
credential strategy, or externally visible support claim changes. Closing a risk
requires linked evidence; deleting the row is not evidence.
