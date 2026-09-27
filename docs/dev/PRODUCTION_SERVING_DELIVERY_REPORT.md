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
| P9 Operational release | Partial | Validate-only, drain, packaging, benchmark harness done; DS6 replaced the manual canary with the coordinated blue-green rollout; soak/benchmark runs open |
| DS0 Foundations | Done (9 of 9) | DS0.1 process gateway proof, DS0.2 this report, DS0.3 poll jitter, DS0.4 model-load slots, DS0.5 scoped per-principal keys, DS0.6 multi-lane supervision, DS0.7 CPU/Metal baselines, DS0.8 backend parity harness |
| DS1 Committed prefix reuse | Done | DS1.1–DS1.6 closed: paged reuse, tensor-snapshot forks, admission probe + cursor-lost re-plan, catalog-auto default-on with per-family evidence cells |
| DS2 Cache-aware routing | Done | Protocol minor-1 routing signals, worker exposure, cache affinity + conversation pinning (default OFF), CPU+Metal 2-worker evidence |
| DS3 Realtime voice over the worker boundary | Done | `izwi-realtime-v1` end to end (worker route, client transport, gateway relay behind flag), TTS-stream stage, DS3.4 public envelope translation |
| DS4 Hierarchical KV offload | Done | DS4.1–DS4.5 closed: design note + ADR 0004, host pool substrate, demotion, promotion, counters, engine-level acceptance, CPU+Metal benchmark evidence (CUDA `not run`) |
| DS5 Fleet authority | Done | DS5.1 PostgreSQL validated (URL connect path, dialect-aware migrator, execution suites in CI; the PG claim race needed a worker-keyed advisory lock), DS5.2 two-gateway/two-worker fleet rig (SQLite + PG lanes: T07/T20/P8.1/steering/crash+TTL/DINV-06 outage), DS5.3 explicit fleet posture + claim-TTL env + maintenance sweep, DS5.4 T25 generation-fence process evidence, DS5.5 ADR 0005 + docs; MySQL recorded unvalidated (error 1170 TEXT-key burden) |
| DS6 Coordinated blue-green rollout | Done | Declarative rollout plan + supervisor state machine (`43d2e7bb`/`3ec044e3`), shared approval format as a protocol contract (`bf30273a`), gateway per-pool DINV-07 eligibility rule (`c539e914`) with runtime approvals-view adoption (`0ec7b5c5`), supervisor process tests + T37 gateway zero-dropped-requests rig (`7fd649eb`/`d9e5abe0`), ADR 0006 + runbook/support-matrix/delivery docs |
| DS7 Signal-driven worker autoscaling | Done | Validated per-deployment policy config + pre-declared standby replicas (`111fcfb4`), supervisor scale up/down wiring with ResourceLedger + approvals view editor (same commit), supervisor-only process rig T38: 1→2 on sustained queue depth, 2→1 on the idle window with drain, disabled-config static, rollout conflict refused (`4702d08a`), ADR 0007 + runbook/support-matrix/delivery docs |

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

DS6 additions (verified 2026-09-26/27, commits `bf30273a`…`d9e5abe0`):
supervisor rollout module 10 unit tests + 5 process tests (`tests/rollout.rs`
with python fake workers serving the real readiness contract: canary-failure
abort with byte-identical approvals restore, promote through window → drain →
commit, replacement-exit abort, SIGKILL resume + fail-closed fresh start,
status/abort command semantics); gateway cutover rig (real gateway process,
two mock workers on real TCP, continuous traffic across window → abort-restore
and window → commit view transitions with zero failed requests, served-count
proofs that the draining predecessor receives nothing after cutover);
protocol round-trip tests for the shared approval format; izwi-server lib +
supervisor suites green, clippy/fmt clean on touched code.

DS7 additions (verified 2026-09-27, commits `111fcfb4`…`4702d08a`): a new
supervisor `autoscale` module holds the process-free primitives — per-
deployment scale state machine (signals → decisions under bounds and shared
hysteresis), `ResourceLedger` (reserved declared budgets of supervised
slots; host-memory, CPU-thread, and device-exclusivity rejections with
actionable diagnostics), and the shared approvals view editor (v1 pinned
identity add/remove, atomic writes under a sidecar lock, verbatim
preservation of unrelated lines, fail-closed standalone-line collision) —
with unit tests for signals→decisions, hysteresis, bounds, ledger
overcommit, and view idempotence/reconciliation. Process evidence (T38,
`tests/autoscale.rs`): the real supervisor binary supervises fake CPU
workers whose status documents the test rewrites per request — sustained
queue depth scales 1→2 (standby launched through the readiness path, v1
line published only after readiness), the idle window scales 2→1
(unapprove → admission stop → drain → clean exit, core set untouched, no
scale-down inside the stabilization window), the disabled configuration
launches every declared worker and never touches the view, `--validate-only`
reports the parsed policy, and `--rollout-plan` is refused while autoscaling
is on. Full supervisor suite green (86 tests incl. all DS6 rigs), clippy
clean on new code, fmt clean.

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
fields tolerated absent). Model-generation rollback on one node uses the
coordinated rollout (a new rollout plan targeting the previous configuration
with a fresh generation — never a generation reuse, ADR 0006). Never run two
supervisor generations against one
device: the generation fence blocks startup until the old generation exits.
No destructive migrations shipped.

## Next highest-priority task

DS7 is complete: the supervisor is a capacity manager within explicit bounds
— pre-declared standby replicas, sustained-queue-depth scale-up, stabilized
idle-window scale-down with drain, ledger-respecting decisions, and shared
hysteresis — off by default and proven with real processes (ADR 0007, T38).
Next per the plan: DS9 (API completeness: `usage.cached_tokens`, grammar-
constrained `response_format`, logprobs, MTP generalization), with DS8
(vLLM engine lane) still behind its decision-gate ADR. CUDA lanes stay
`not run` until hardware evidence exists.
