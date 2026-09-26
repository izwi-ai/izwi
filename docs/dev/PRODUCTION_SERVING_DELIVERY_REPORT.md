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
| P4 Registry & multi-worker routing | Done | T02/T07/T09 real-HTTP tests; the multi-process fleet rig landed with DS5.2 |
| P5 Safeguards, streams, retry | Done | Perimeter, tenant limits, bounded SSE, one-alternate retry, drain; DS0.5 scoped per-principal keys closed |
| P6 Durable jobs, artifacts | Done for the single-node text profile | Voice route migration remains gated per ledger |
| P7 Secure one-gateway fleet | Done (code) | TLS/mTLS, topology policy, remote artifact transport; separate-machine handshake evidence open |
| P8 Multi-gateway correctness | Done | Partitioned quotas, claims, outage degradation; DS5 validated the shared stores and posture with real processes |
| P9 Operational release | Partial | Validate-only, drain, canary, packaging, benchmark harness done; soak/benchmark runs open |
| DS0 Foundations | Done (9 of 9) | DS0.1 process gateway proof, DS0.2 this report, DS0.3 poll jitter, DS0.4 model-load slots, DS0.5 scoped per-principal keys, DS0.6 multi-lane supervision, DS0.7 CPU/Metal baselines, DS0.8 backend parity harness |
| DS1 Committed prefix reuse | Done | DS1.1–DS1.6 closed: paged reuse, tensor-snapshot forks, admission probe + cursor-lost re-plan, catalog-auto default-on with per-family evidence cells |
| DS2 Cache-aware routing | Done | Protocol minor-1 routing signals, worker exposure, cache affinity + conversation pinning (default OFF), CPU+Metal 2-worker evidence |
| DS3 Realtime voice over the worker boundary | Done | `izwi-realtime-v1` end to end (worker route, client transport, gateway relay behind flag), TTS-stream stage, DS3.4 public envelope translation |
| DS4 Hierarchical KV offload | Done | DS4.1–DS4.5 closed: design note + ADR 0004, host pool substrate, demotion, promotion, counters, engine-level acceptance, CPU+Metal benchmark evidence (CUDA `not run`) |
| DS5 Fleet authority | Done | DS5.1 PostgreSQL validated (URL connect path, dialect-aware migrator, execution suites in CI; the PG claim race needed a worker-keyed advisory lock), DS5.2 two-gateway/two-worker fleet rig (SQLite + PG lanes: T07/T20/P8.1/steering/crash+TTL/DINV-06 outage), DS5.3 explicit fleet posture + claim-TTL env + maintenance sweep, DS5.4 T25 generation-fence process evidence, DS5.5 ADR 0005 + docs; MySQL recorded unvalidated (error 1170 TEXT-key burden) |

## Supported deployment profiles

- **Development mock/CPU slice: supported.** Real transport and real CPU
  process evidence; explicitly not a production claim.
- **Production single-node:** software-complete for the text-chat profile;
  not claimed production-ready until the soak/overload/benchmark gates run on
  target hardware.
- **Secure multi-machine (one gateway):** code-complete, not an operational
  claim until separate-machine TLS handshake and node-loss evidence exist.
- **Multi-gateway fleet:** supported on the validated store lanes (shared
  SQLite on one host, shared PostgreSQL for a server-backed fleet) per
  DS5.2's process rig and ADR 0005; admission stays worker-authoritative,
  MySQL coordination stays unvalidated, and separate-machine TLS handshakes
  remain a separate gate.

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

DS1.6 additions (verified 2026-09-25): catalog prefix-reuse inventory tests;
prefix engagement/leniency unit tests (config, engine config, serve_runtime
tri-state); worker suites green with the new catalog-auto admission leg (no
prefix env, counters assert through normal admission) and the independent
kill-switch leg; backend_parity 4 CPU legs + both Metal legs green on the
metal-feature build (`--include-ignored`); izwi-core lib 2647, cli 72,
server 678 (one unrelated pre-existing cancellation-race flake reproduced
once and passed on retry and on the clean tree); clippy/fmt clean on touched
code.

DS2/DS3 additions (verified 2026-09-25/26): routing-signal protocol tests,
2-worker CPU+Metal rig manifests, realtime worker-route/client/gateway
suites, gateway TTS-pool tests, envelope-translation tests; the DS3.4
translation commit is `4e121999`.

DS4 additions (verified 2026-09-26, commits `39c62ca9`…DS4.5): izwi-core lib
2664 green including the new offload unit suite (pool budget, residency
transitions, chain purge, promotion round-trip with seeded bytes) and the
managed-layer demote→promote→re-demote cycle test; worker suites green
including the new engine-level acceptance test `ds4_host_offload.rs`
(concurrent shared-prefix sessions on an undersized arena, greedy replay
byte-identical to the cold run) and the DS1.5 attach regression with host
continuation extended over snapshot-sharing arenas; mock-worker contract
suite 23 green; benchmark rig `run-ds4-offload-benchmark.sh` passes both
lanes with hard gates (CPU on-leg demotions=10/promotions=8/host_pages=2;
Metal on-leg demotions=7/promotions=3/host_pages=4; 8 MiB budget respected).
Two DS4.2 gaps surfaced and fixed by the DS4.4 work: the host pool budget is
now part of the model's load-time resource authorization, and the worker
Prometheus endpoint renders the DS4 counters.

## Tests not executed and why

- CUDA/Metal *inference*, multi-GPU, multi-machine, soak, overload matrices:
  no accelerator/fabric hardware in the work environment (Metal build/run is
  the next local lane).
- MySQL fleet SQL execution: recorded unvalidated — a live MySQL 8 probe
  fails on the first table (`TEXT PRIMARY KEY` needs a key length, MySQL
  error 1170); validating MySQL means a dedicated schema variant (ADR 0005).
  PostgreSQL fleet and durable-store execution runs in CI
  (`backend-truth.yml` fleet-stores job) and locally against Homebrew
  PostgreSQL 18.
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
wall-clock-proven. DS2 routing evidence (`ds2-{cpu,metal}-summary.json`) and
DS4 offload evidence (`ds4-{cpu,metal}-summary.json`, 2026-09-26) are the
same class of counter-proven fixture evidence: DS4's on-legs show demotion,
promotion, and budget containment with prefix reuse preserved, and the
engine-level test pins byte-identical greedy output against the cold run.
Real-model TTFT/throughput claims remain future work.

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

DS5 is complete: the multi-gateway fleet profile is validated on SQLite and
PostgreSQL coordination stores with real two-gateway process evidence
(ADR 0005), the supervisor generation fence is process-proven, and MySQL
stays honestly unvalidated. Next per the plan: the remaining DS0 remnant is
none (DS0.1–DS0.8 closed), so the next phase is DS6 (declarative rollout
with canary promotion), with CUDA lanes staying `not run` until hardware
evidence exists.
