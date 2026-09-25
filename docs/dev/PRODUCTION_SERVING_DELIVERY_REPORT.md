# Production serving: delivery report (serving plan Section 19)

Consolidated record of the `production-serving` implementation programme
against `IZWI_PRODUCTION_SERVING_IMPLEMENTATION_PLAN.md`, including the
distributed-serving plan (`PRODUCTION_DISTRIBUTED_SERVING_PLAN.md`) work.
Supersedes the scattered records: `tasks/production-serving-status-analysis-2026-09-16.md`,
the `tasks/todo.md` session entries (2026-09-13 … present), and the blank
template in `PRODUCTION_SERVING_RELEASE_CHECKLIST.md`.

## Baseline and resulting revision

- Baseline: `7804eee5` (merge-base of `production-serving` and `main`),
  "Enable concurrent Fish Audio serving with reliable streaming and adaptive
  batching (#215)".
- Serving-programme result: 106 commits through `9fe4506d` (gateway, worker,
  supervisor, protocol, registry, fleet store, docs, tests).
- Distributed-serving programme (DS0 open): plan `6ab2851d`; DS0.3 `b682a99b`;
  DS0.4 `09759937`; DS0.6 `b9df5e07`; DS0.1 `1d2fbf7e`.

## Completed phases and task IDs

| Phase | Status | Notes |
|---|---|---|
| P0 Discovery & ledger | Done | Baseline, inventory, route migration ledger, ADRs 0001/0002 |
| P1 Contracts & mock harness | Done | Versioned protocol, bounded parsing, mock worker, 23 real-HTTP contract tests |
| P2 Real CPU vertical slice | Done | Real-process worker test; public chat route end to end |
| P3 Supervisor & assignments | Done for CPU; DS0.6 lifted the Metal/CUDA supervision gate (process-evidenced); real accelerator execution still hardware-gated |
| P4 Registry & multi-worker routing | Done (code) | T02/T07/T09 real-HTTP tests; multi-process fleet rig open (P4.3-full) |
| P5 Safeguards, streams, retry | Done | Perimeter, tenant limits, bounded SSE, one-alternate retry, drain; scoped per-principal keys open (DS0.5) |
| P6 Durable jobs, artifacts | Done for the single-node text profile | Voice route migration remains gated per ledger |
| P7 Secure one-gateway fleet | Done (code) | TLS/mTLS, topology policy, remote artifact transport; separate-machine handshake evidence open |
| P8 Multi-gateway correctness | Done (code, SQLite) | Partitioned quotas, claims, outage degradation; PG/MySQL execution + shared-atomic default open (DS5) |
| P9 Operational release | Partial | Validate-only, drain, canary, packaging, benchmark harness done; soak/benchmark runs open |
| DS0 Foundations | 5 of 9 done | DS0.1/0.3/0.4/0.6 closed; DS0.2 (this report), DS0.5, DS0.7, DS0.8 open |

## Supported deployment profiles

- **Development mock/CPU slice: supported.** Real transport and real CPU
  process evidence; explicitly not a production claim.
- **Production single-node:** software-complete for the text-chat profile;
  not claimed production-ready until the soak/overload/benchmark gates run on
  target hardware.
- **Secure multi-machine (one gateway):** code-complete, not an operational
  claim until separate-machine TLS handshake and node-loss evidence exist.
- **Multi-gateway fleet:** not a supported claim (Phase 8/DS5 gates open).

## Actual backend/model/hardware cells tested

- CPU: real subprocess worker executing a generated tiny LFM2 GGUF through the
  public `/v1/chat/completions` route (JSON + SSE). Apple Silicon host,
  macOS, debug builds.
- Metal / CUDA / multi-GPU / multi-machine: **not run.** Supervision of Metal
  and CUDA lanes is process-evidenced with fake lane workers
  (`multi_lane_launch.rs`); compilation and mock evidence are never promoted
  to execution evidence.

## Tests executed and results (verified 2026-09-22/23)

`izwi-server` lib 675 (gateway 74, worker_registry 21 incl.); serving
protocol 16; client 5 + 23 contract (mock-worker feature); supervisor 31 lib
+ 8 bin + integration (packaged, topology ×4, validate-only, worker-recovery,
multi-lane); worker 12 lib + 5 bin + real-CPU-process; boundary gate passes;
fmt/clippy clean on touched code (pre-existing denies documented).

DS1.x additions (verified 2026-09-25): worker suites incl. the
`prefix_attach_repro` regression and the qwen38 hybrid prefix parity leg
(CPU ×2 processes + Metal, attached prefill bit-identical to fresh);
izwi-core qwen38 attach + engine cache suites green; gateway benchmark
harness 18 unit tests; DS1.5 runner smoke test.

## Tests not executed and why

- CUDA/Metal *inference*, multi-GPU, multi-machine, soak, overload matrices:
  no accelerator/fabric hardware in the work environment (Metal build/run is
  the next local lane).
- PG/MySQL fleet SQL execution: no PostgreSQL/MySQL instance in CI yet (DS5).
- Live benchmark runs on real production-sized models and CUDA lanes:
  fixture-scale DS0.7/DS1.5 runs executed (see below); speed claims still
  require real-model runs.

## API / configuration compatibility changes

Public OpenAI-compatible chat surface unchanged. Additive: `max_parallel_model_loads`
(node config v2, default 1), supervisor device-declaration flags, gateway
status-poll jitter (validation now reserves 10% TTL headroom). The gateway
exposes only `POST /v1/chat/completions` + probes/admin; all other route
families remain local-only per the ledger.

## Known limitations and failure behavior

Documented in `PRODUCTION_SERVING_SUPPORT_MATRIX.md` §failure semantics:
expired status blocks new routing only; lost connections retain worker
capacity until confirmed teardown; partial streams terminate explicitly;
single-gateway availability only; tenant rate state is process-local unless
fleet mode is on.

## Measured performance and test conditions

Fixture-scale only; no speed claim. DS0.7 baseline manifests plus DS1.5
prefix-caching evidence (`benchmarks/manifests/ds15-{cpu,metal}-{shared,cold}.json`
+ summaries, 2026-09-25): gateway-routed shared-prefix workloads attach
committed prefixes (36 attaches, 4736 avoided-prefill tokens per lane) while
cold workloads reuse nothing (0 attaches); TTFT delta ≈ 0 at fixture scale —
prefill of ~130 tiny tokens is microseconds, so reuse is counter-proven, not
wall-clock-proven. Real-model TTFT/throughput claims remain future work.

## Security / deployment assumptions

API-key perimeter with separate inference/admin/metrics keys (per-principal
scoped keys = DS0.5); secrets only via bounded `env:`/`file:` references;
workers loopback-plaintext or TLS/mTLS; supervisor device inventory is
operator-declared and verified by worker-side assigned-device selection.

## Rollback procedure

Roll back by redeploying the previous binaries and manifest; worker
configurations and fleet tables are additive (schema v2, additive protocol
fields tolerated absent). Never run two supervisor generations against one
device: the generation fence blocks startup until the old generation exits.
No destructive migrations shipped.

## Next highest-priority task

DS1.6 completion — per-backend prefix-reuse enablement in the capability
catalog (CPU+Metal parity evidence is already green) and the DS1.2b
default-on decision; then DS2 cache-aware routing. DS0.5 scoped per-principal
API keys remains a standalone security slice.
